//! Wav2Vec2 (stable-layer-norm variant) model, CPU implementation.
//!
//! Port of the HF transformers 5.17 computation graph for this checkpoint:
//!
//! ```text
//! z-normed waveform (N,)
//!   7× conv1d (valid, stride s, bias) → LayerNorm(512) → gelu(erf)   [50 fps]
//!   LayerNorm(512) → Linear(512→1024)                                [feature_projection]
//!   depthwise pos conv (k=128, pad 64, groups=16) → drop last → gelu → add
//!   24× [h += Attn(LN(h)); h += FFN(final_LN(h))]                     [pre-LN]
//!   LayerNorm(1024) → lm_head(1024→10288) → log_softmax
//! ```
//!
//! Every matmul (linears, attention QK^T / PV, conv taps, pos-conv taps)
//! goes through the `gemm` crate (packed kernels, FMA, rayon).  The only
//! free choices are accumulation orders; every op matches the reference
//! math (erf gelu, biased LN variance with eps inside the sqrt, fp32
//! attention with 1/8 scaling).  `fast_exp` / `fast_erf` (in [`crate::simd`])
//! are ~1e-7 accurate, the same class as libm, and the golden test pins the
//! final token timestamps to the Python reference.
//!
//! Layout: this file holds the model types and checkpoint loading,
//! `kernels` the elementwise / GEMM wrappers, `forward` the forward
//! pass and its scratch buffers.

mod forward;
mod kernels;
pub(crate) mod prof;

pub(crate) use forward::Scratch;

use anyhow::Result;

use crate::config::Wav2Vec2Config;
use crate::weights::{get_f32, RawTensor};

pub(crate) struct Linear {
    /// `[out, in]` exactly as stored in the checkpoint.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub out: usize,
    pub in_: usize,
}

pub(crate) struct LayerNorm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub eps: f64,
}

pub(crate) struct ConvLayer {
    /// `[out, in, k]`
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub out: usize,
    pub in_: usize,
    pub k: usize,
    pub stride: usize,
    pub ln: LayerNorm,
}

pub(crate) struct EncoderLayer {
    /// Fused q/k/v projection: rows `0..hidden` are Q, then K, then V.
    pub qkv: Linear,
    pub out_proj: Linear,
    pub ln1: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub ln2: LayerNorm,
}

pub(crate) struct Model {
    pub cfg: Wav2Vec2Config,
    pub conv: Vec<ConvLayer>,
    pub feat_proj_ln: LayerNorm,
    pub feat_proj: Linear,
    pub pos_conv_weight: Vec<f32>, // [1024, in_per_group=64, 128], weight_norm resolved
    pub pos_conv_bias: Vec<f32>,
    pub layers: Vec<EncoderLayer>,
    pub final_ln: LayerNorm,
    pub lm_head: Linear,
}

/// Which intermediate stages `forward` should collect (golden diffing).
#[derive(Debug, Clone, Default)]
pub(crate) struct StageSet {
    pub conv: bool,
    pub proj: bool,
    pub pos_conv: bool,
    pub enc_in: bool,
    pub layers: Option<Vec<usize>>, // indices; attn_out too
    pub enc_final: bool,
}

#[derive(Default)]
pub(crate) struct Stages {
    pub conv: Vec<Vec<f32>>, // per conv layer, (T, C) row-major
    pub proj: Vec<f32>,
    pub pos_conv: Vec<f32>,
    pub enc_in: Vec<f32>,
    pub attn_out: std::collections::HashMap<usize, Vec<f32>>,
    pub layer_out: std::collections::HashMap<usize, Vec<f32>>,
    pub enc_final: Vec<f32>,
}

impl Model {
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let cfg = Wav2Vec2Config::load(model_dir)?;
        // the K^T transpose in forward_with bands 64 channels at a time
        anyhow::ensure!(
            cfg.hidden_size % 64 == 0,
            "hidden_size {} not divisible by 64 (K^T banding)",
            cfg.hidden_size
        );
        // the CPU phase profiler has one slot per conv layer (CONV0..CONV6)
        anyhow::ensure!(
            cfg.conv_kernel.len() <= 7,
            "conv stack has {} layers, profiler tracks 7",
            cfg.conv_kernel.len()
        );
        let tensors = crate::weights::load_tensors(model_dir)?;

        let mut conv = Vec::new();
        // conv layer inputs: 1, then conv_dim[i-1]
        let mut prev_dim = 1usize;
        for i in 0..cfg.conv_kernel.len() {
            let out = cfg.conv_dim[i];
            let k = cfg.conv_kernel[i];
            let stride = cfg.conv_stride[i];
            let w = get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.weight"))?;
            let b = get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.bias"))?;
            let lw = get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.weight"))?;
            let lb = get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.bias"))?;
            anyhow::ensure!(w.1 == vec![out, prev_dim, k], "conv{i} weight {:?}", w.1);
            anyhow::ensure!(b.1 == vec![out], "conv{i} bias {:?}", b.1);
            conv.push(ConvLayer {
                weight: w.0,
                bias: b.0,
                out,
                in_: prev_dim,
                k,
                stride,
                ln: LayerNorm { w: lw.0, b: lb.0, eps: 1e-5 },
            });
            prev_dim = out;
        }

        let ln_eps = cfg.layer_norm_eps;
        let get_ln = |tensors: &std::collections::HashMap<String, RawTensor>, base: String, dim: usize| -> Result<LayerNorm> {
            let w = get_f32(tensors, &format!("{base}.weight"))?;
            let b = get_f32(tensors, &format!("{base}.bias"))?;
            anyhow::ensure!(w.1 == vec![dim] && b.1 == vec![dim]);
            Ok(LayerNorm { w: w.0, b: b.0, eps: ln_eps })
        };

        let fpw = get_f32(&tensors, "wav2vec2.feature_projection.projection.weight")?;
        let fpb = get_f32(&tensors, "wav2vec2.feature_projection.projection.bias")?;
        let feat_proj = Linear { w: fpw.0, b: fpb.0, out: cfg.hidden_size, in_: cfg.conv_dim[7 - 1] };

        // pos conv: weight_norm with dim=2 -> per-tap norms over (out, in)
        let g = get_f32(&tensors, "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0")?;
        let v = get_f32(&tensors, "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1")?;
        let pb = get_f32(&tensors, "wav2vec2.encoder.pos_conv_embed.conv.bias")?;
        anyhow::ensure!(g.1 == vec![1, 1, cfg.num_conv_pos_embeddings], "pos g {:?}", g.1);
        let (out_c, in_pg, k) = (v.1[0], v.1[1], v.1[2]);
        let mut pos_w = v.0.clone();
        for kk in 0..k {
            let gv = g.0[kk];
            let mut norm = 0.0f32;
            for o in 0..out_c {
                for ii in 0..in_pg {
                    let x = v.0[(o * in_pg + ii) * k + kk];
                    norm += x * x;
                }
            }
            let inv = gv / norm.sqrt();
            for o in 0..out_c {
                for ii in 0..in_pg {
                    pos_w[(o * in_pg + ii) * k + kk] *= inv;
                }
            }
        }

        let hidden = cfg.hidden_size;
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let base = |sub: &str| format!("wav2vec2.encoder.layers.{i}.{sub}");
            let lin = |name: String, out: usize, inp: usize| -> Result<Linear> {
                let w = get_f32(&tensors, &format!("{name}.weight"))?;
                let b = get_f32(&tensors, &format!("{name}.bias"))?;
                anyhow::ensure!(w.1 == vec![out, inp], "{name} {:?}", w.1);
                Ok(Linear { w: w.0, b: b.0, out, in_: inp })
            };
            // fuse q/k/v into one (3*hidden, hidden) projection
            let qw = get_f32(&tensors, &base("attention.q_proj.weight"))?;
            let qb = get_f32(&tensors, &base("attention.q_proj.bias"))?;
            let kw = get_f32(&tensors, &base("attention.k_proj.weight"))?;
            let kb = get_f32(&tensors, &base("attention.k_proj.bias"))?;
            let vw = get_f32(&tensors, &base("attention.v_proj.weight"))?;
            let vb = get_f32(&tensors, &base("attention.v_proj.bias"))?;
            let mut qkv_w = qw.0;
            qkv_w.extend_from_slice(&kw.0);
            qkv_w.extend_from_slice(&vw.0);
            let mut qkv_b = qb.0;
            qkv_b.extend_from_slice(&kb.0);
            qkv_b.extend_from_slice(&vb.0);
            layers.push(EncoderLayer {
                qkv: Linear { w: qkv_w, b: qkv_b, out: 3 * hidden, in_: hidden },
                out_proj: lin(base("attention.out_proj"), hidden, hidden)?,
                ln1: get_ln(&tensors, base("layer_norm"), hidden)?,
                ff1: lin(base("feed_forward.intermediate_dense"), cfg.intermediate_size, hidden)?,
                ff2: lin(base("feed_forward.output_dense"), hidden, cfg.intermediate_size)?,
                ln2: get_ln(&tensors, base("final_layer_norm"), hidden)?,
            });
        }

        let lhw = get_f32(&tensors, "lm_head.weight")?;
        let lhb = get_f32(&tensors, "lm_head.bias")?;
        let lm_head = Linear { w: lhw.0, b: lhb.0, out: cfg.vocab_size, in_: hidden };

        Ok(Self {
            cfg,
            conv,
            feat_proj_ln: get_ln(&tensors, "wav2vec2.feature_projection.layer_norm".into(), 512)?,
            feat_proj,
            pos_conv_weight: pos_w,
            pos_conv_bias: pb.0,
            layers,
            final_ln: get_ln(&tensors, "wav2vec2.encoder.layer_norm".into(), hidden)?,
            lm_head,
        })
    }
}

/// Host-side lm head only — the GPU tower's stand-in for [`Model`]'s head.
///
/// The GPU tower can park the encoder stream (T×hidden, 4 KB/frame) instead
/// of the (T×vocab) logits (41 KB/frame) and re-run the head per window on
/// demand; this holds the ~42 MB of head weights that takes, so a long
/// transcript no longer needs a full CPU [`Model`] on the side.  Same
/// [`Linear`], same GEMM call as the CPU tower's hidden-form path, so the
/// recomputed logits are bit-identical given the same hidden rows.
pub(crate) struct LmHeadCpu {
    pub linear: Linear,
}

impl LmHeadCpu {
    /// Load `lm_head.weight` / `lm_head.bias` (and the config for the two
    /// dims) from the checkpoint directory.
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let cfg = Wav2Vec2Config::load(model_dir)?;
        let tensors = crate::weights::load_tensors(model_dir)?;
        let lhw = get_f32(&tensors, "lm_head.weight")?;
        let lhb = get_f32(&tensors, "lm_head.bias")?;
        Ok(Self {
            linear: Linear { w: lhw.0, b: lhb.0, out: cfg.vocab_size, in_: cfg.hidden_size },
        })
    }

    /// Re-run the head over a `(rows, hidden)` block, writing bias-free
    /// `(rows, vocab)` logits — the same GEMM [`Model::lm_head_into_logits`]
    /// runs, so the values match the CPU tower's.
    pub fn into_logits(&self, hidden: &[f32], rows: usize, logits: &mut [f32]) {
        debug_assert_eq!(hidden.len(), rows * self.linear.in_);
        debug_assert!(logits.len() >= rows * self.linear.out);
        self.linear.apply_into_nobias(hidden, rows, logits);
    }

    /// The trellis columns of one bias-free logit row (log-softmax + gather);
    /// returns the row's normaliser.
    pub fn gather_lp_row(&self, logits: &[f32], cols: &[i32], out: &mut [f32]) -> f32 {
        kernels::log_softmax_gather_row_c(logits, &self.linear.b, cols, out)
    }

    pub fn bias(&self) -> &[f32] {
        &self.linear.b
    }

    pub fn vocab(&self) -> usize {
        self.linear.out
    }
}
