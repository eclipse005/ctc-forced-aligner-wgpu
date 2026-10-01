//! Wav2Vec2 (stable-layer-norm variant) forward, CPU implementation.
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
//! attention with 1/8 scaling).  `fast_exp` / `fast_erf` are ~1e-7
//! accurate, the same class as libm, and the golden test pins the final
//! token timestamps to the Python reference.

use anyhow::{Context, Result};
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use crate::config::Wav2Vec2Config;
use crate::weights::{get_f32, RawTensor};

pub struct Linear {
    /// `[out, in]` exactly as stored in the checkpoint.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub out: usize,
    pub in_: usize,
}

pub struct LayerNorm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub eps: f64,
}

pub struct ConvLayer {
    /// `[out, in, k]`
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub out: usize,
    pub in_: usize,
    pub k: usize,
    pub stride: usize,
    pub ln: LayerNorm,
}

pub struct EncoderLayer {
    /// Fused q/k/v projection: rows `0..hidden` are Q, then K, then V.
    pub qkv: Linear,
    pub out_proj: Linear,
    pub ln1: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub ln2: LayerNorm,
}

pub struct Model {
    pub cfg: Wav2Vec2Config,
    pub conv: Vec<ConvLayer>,
    pub feat_proj_ln: LayerNorm,
    pub feat_proj: Linear,
    pub pos_conv_weight: Vec<f32>, // [1024, in_per_group=64, 128], weight_norm resolved
    pub pos_conv_bias: Vec<f32>,
    pub layers: Vec<EncoderLayer>,
    pub final_ln: LayerNorm,
    pub lm_head: Linear,
    pub vocab: Vec<(String, usize)>,
    pub char_to_id: std::collections::HashMap<char, usize>,
    pub unk_id: usize,
}

/// Which intermediate stages `forward` should collect (golden diffing).
#[derive(Debug, Clone, Default)]
pub struct StageSet {
    pub conv: bool,
    pub proj: bool,
    pub pos_conv: bool,
    pub enc_in: bool,
    pub layers: Option<Vec<usize>>, // indices; attn_out too
    pub enc_final: bool,
}

#[derive(Default)]
pub struct Stages {
    pub conv: Vec<Vec<f32>>, // per conv layer, (T, C) row-major
    pub proj: Vec<f32>,
    pub pos_conv: Vec<f32>,
    pub enc_in: Vec<f32>,
    pub attn_out: std::collections::HashMap<usize, Vec<f32>>,
    pub layer_out: std::collections::HashMap<usize, Vec<f32>>,
    pub enc_final: Vec<f32>,
}

// ---------------------------------------------------------------------------
// fast exp / erf live in `crate::simd` next to their AVX2 twins; the scalar
// wrappers below only route between the SIMD path and the scalar fallback.
// ---------------------------------------------------------------------------

fn gelu(x: &mut [f32]) {
    // AVX2 path: erf via polynomial, 8 lanes per iteration.  The feature
    // probe is a cached atomic read; hoisting it per chunk is plenty.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::simd::avx2::have_avx2_fma();
    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        return x.par_chunks_mut(1 << 14).for_each(|chunk| {
            unsafe { crate::simd::avx2::gelu_inplace(chunk) };
        });
    }
    x.par_chunks_mut(1 << 14).for_each(|chunk| {
        for v in chunk.iter_mut() {
            // exact gelu: 0.5*x*(1+erf(x/sqrt(2)))
            *v = 0.5 * *v * (1.0 + crate::simd::fast_erf(*v / std::f32::consts::SQRT_2));
        }
    });
}

/// dst += src, parallel over blocks.
fn add_par(dst: &mut [f32], src: &[f32]) {
    dst.par_chunks_mut(1 << 15)
        .zip(src.par_chunks(1 << 15))
        .for_each(|(d, s)| {
            for (a, b) in d.iter_mut().zip(s) {
                *a += b;
            }
        });
}

/// Softmax one row in place (`fast_exp` under the hood).
#[inline]
fn softmax_row(row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2::have_avx2_fma() {
        unsafe { crate::simd::avx2::softmax_inplace(row) };
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for sc in row.iter_mut() {
        *sc = crate::simd::fast_exp(*sc - max);
        sum += *sc;
    }
    let inv = 1.0 / sum;
    for sc in row.iter_mut() {
        *sc *= inv;
    }
}

/// log_softmax one row in place: x - max - ln(sum(exp(x - max))).
#[inline]
fn log_softmax_row(row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2::have_avx2_fma() {
        unsafe { crate::simd::avx2::log_softmax_inplace(row) };
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in row.iter() {
        sum += crate::simd::fast_exp(*x - max);
    }
    let lsum = sum.ln();
    for x in row.iter_mut() {
        *x = *x - max - lsum;
    }
}

// ---------------------------------------------------------------------------
// GEMM wrappers
//
// gemm computes dst = alpha·dst + beta·lhs·rhs; when `read_dst` is false
// the crate zeroes alpha, so dst is never read.  Strides are (cs, rs) =
// (column step, row step).
// ---------------------------------------------------------------------------

/// y (rows, out) = x (rows, in) @ w^T + b, with w stored `[out, in]`.
fn linear_gemm(y: &mut [f32], x: &[f32], rows: usize, w: &[f32], b: &[f32], out: usize, in_: usize) {
    for row in y[..rows * out].chunks_exact_mut(out) {
        row.copy_from_slice(b);
    }
    unsafe {
        gemm::<f32>(
            rows,
            out,
            in_,
            y.as_mut_ptr(),
            1,
            out as isize,
            true, // read dst (the prefilled bias)
            x.as_ptr(),
            1,
            in_ as isize,
            w.as_ptr(),
            in_ as isize,
            1,
            1.0,
            1.0,
            false,
            false,
            false,
            Parallelism::Rayon(0),
        );
    }
}

impl Linear {
    pub fn apply_into(&self, x: &[f32], rows: usize, y: &mut [f32]) {
        linear_gemm(y, x, rows, &self.w, &self.b, self.out, self.in_);
    }

    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * self.out];
        self.apply_into(x, rows, &mut y);
        y
    }
}

impl LayerNorm {
    pub fn apply(&self, h: &mut [f32], cols: usize) {
        // h is (rows, cols); normalise each row over cols.  The reductions
        // use 8 independent lanes so the compiler emits vaddps instead of a
        // serial FADD chain.
        let w = &self.w;
        let b = &self.b;
        let eps = self.eps;
        h.par_chunks_exact_mut(cols).for_each(|row| {
            let n = cols as f32;
            let mut acc = [0.0f32; 8];
            let mut chunks = row.chunks_exact(8);
            for c in chunks.by_ref() {
                for i in 0..8 {
                    acc[i] += c[i];
                }
            }
            let mut sum = 0.0f32;
            for v in chunks.remainder() {
                sum += *v;
            }
            for a in acc {
                sum += a;
            }
            let mean = sum / n;
            let mut acc2 = [0.0f32; 8];
            let mut chunks = row.chunks_exact(8);
            for c in chunks.by_ref() {
                for i in 0..8 {
                    let d = c[i] - mean;
                    acc2[i] += d * d;
                }
            }
            let mut var = 0.0f32;
            for v in chunks.remainder() {
                let d = *v - mean;
                var += d * d;
            }
            for a in acc2 {
                var += a;
            }
            var /= n;
            let inv = ((var as f64 + eps).sqrt()) as f32;
            for (x, (w, b)) in row.iter_mut().zip(w.iter().zip(b.iter())) {
                *x = (*x - mean) / inv * w + b;
            }
        });
    }
}

/// Raw pointer wrapper so the pos-conv group loop can share the output
/// buffer across rayon tasks; each group writes a disjoint column strip.
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn add(&self, n: usize) -> *mut f32 {
        unsafe { self.0.add(n) }
    }
}

/// Transient forward buffers.  `forward_with` reuses them across calls so the
/// chunked encoder does not re-allocate (and re-fault) ~90 MB per chunk;
/// every buffer is fully overwritten before it is read, so stale contents
/// from a previous (longer) forward are harmless.
#[derive(Default)]
pub struct Scratch {
    ln: Vec<f32>,
    qkv: Vec<f32>,
    kt: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    h: Vec<f32>,
    h2: Vec<f32>,
    ff: Vec<f32>,
    ff2: Vec<f32>,
    scores: Vec<f32>,
}

fn fit(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
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

        // vocab: char -> id, from vocab.json; blank = 0 (<s>), unk = <unk>
        let vocab_raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("vocab.json")).context("read vocab.json")?,
        )?;
        let mut vocab: Vec<(String, usize)> = Vec::with_capacity(cfg.vocab_size);
        let mut char_to_id = std::collections::HashMap::new();
        for (tok, id) in vocab_raw.as_object().context("vocab.json object")? {
            let id = id.as_u64().context("vocab id")? as usize;
            vocab.push((tok.clone(), id));
            let mut chars = tok.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                char_to_id.insert(c, id);
            }
        }
        let unk_id = vocab
            .iter()
            .find(|(t, _)| t == "<unk>")
            .map(|(_, i)| *i)
            .unwrap_or(3);

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
            vocab,
            char_to_id,
            unk_id,
        })
    }

    /// Char-level tokenisation mirroring the Python `_tokenise`: unknown chars
    /// are skipped so every timestamp maps to one visible character.
    pub fn tokenise(&self, text: &str) -> (Vec<usize>, Vec<String>) {
        let mut ids = Vec::new();
        let mut pieces = Vec::new();
        for c in text.chars() {
            if let Some(&id) = self.char_to_id.get(&c) {
                if id == self.unk_id {
                    continue;
                }
                ids.push(id);
                pieces.push(c.to_string());
            }
        }
        (ids, pieces)
    }

    /// Full forward on one z-normalised waveform. Returns log_probs (T, V).
    pub fn forward(&self, input: &[f32], stages: &StageSet) -> Result<(Vec<f32>, Stages)> {
        let mut scratch = Scratch::default();
        self.forward_with(input, stages, &mut scratch)
    }

    /// Same forward, reusing scratch buffers across calls: `forward` allocates
    /// ~90 MB of transient buffers per invocation, which the chunked encoder
    /// pays once per chunk.  Every buffer is fully overwritten before use, so
    /// stale contents are harmless.
    pub fn forward_with(
        &self,
        input: &[f32],
        stages: &StageSet,
        scratch: &mut Scratch,
    ) -> Result<(Vec<f32>, Stages)> {
        let mut out_stages = Stages::default();

        // ---- conv stack: activations stay (T, C) row-major after each conv
        let mut cur = self.conv_step(&self.conv[0], input, input.len());
        if stages.conv {
            out_stages.conv.push(cur.clone());
        }
        for cl in self.conv.iter().skip(1) {
            let t_in = cur.len() / cl.in_;
            cur = self.conv_step(cl, &cur, t_in);
            if stages.conv {
                out_stages.conv.push(cur.clone());
            }
        }

        let t = cur.len() / 512;

        // ---- encoder-layer scratch: fit to this forward's shapes and split
        // into disjoint slices (buffers are fully overwritten before read)
        let hidden = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let head_dim = hidden / heads;
        let intermediate = self.cfg.intermediate_size;
        let collect_layers = stages.layers.clone().unwrap_or_default();
        let q_block = t.min(2048);
        for buf in [
            &mut scratch.ln,
            &mut scratch.attn,
            &mut scratch.proj,
            &mut scratch.h,
            &mut scratch.h2,
            &mut scratch.ff2,
        ] {
            fit(buf, t * hidden);
        }
        fit(&mut scratch.qkv, t * 3 * hidden);
        fit(&mut scratch.kt, hidden * t);
        fit(&mut scratch.ff, t * intermediate);
        fit(&mut scratch.scores, q_block * t);
        let Scratch {
            ln,
            qkv,
            kt,
            attn,
            proj,
            h: h_buf,
            h2: h2_buf,
            ff,
            ff2,
            scores: sc,
        } = &mut *scratch;
        let ln_buf = &mut ln[..t * hidden];
        let qkv_buf = &mut qkv[..t * 3 * hidden];
        let kt_buf = &mut kt[..hidden * t];
        let attn_buf = &mut attn[..t * hidden];
        let proj_buf = &mut proj[..t * hidden];
        let ff_buf = &mut ff[..t * intermediate];
        let ff2_buf = &mut ff2[..t * hidden];
        let scores = &mut sc[..q_block * t];

        // ---- feature projection: LN -> linear (residual stream in scratch)
        self.feat_proj_ln.apply(&mut cur, 512);
        let mut h: &mut [f32] = &mut h_buf[..t * hidden];
        let mut h_next: &mut [f32] = &mut h2_buf[..t * hidden];
        self.feat_proj.apply_into(&cur, t, h);
        if stages.proj {
            out_stages.proj = h.to_vec();
        }

        // ---- positional conv: depthwise, groups=16, k=128, pad 64, drop last
        let pos = self.pos_conv(h, t);
        if stages.pos_conv {
            out_stages.pos_conv = pos.clone();
        }
        add_par(h, &pos);
        if stages.enc_in {
            out_stages.enc_in = h.to_vec();
        }

        // scratch buffers reused across layers: attention scores are blocked
        // over queries (max 2048 rows) so memory stays O(t) for long inputs

        for (li, layer) in self.layers.iter().enumerate() {
            // attention block: h_next = h + out_proj(attn(LN(h)))
            ln_buf.copy_from_slice(h);
            layer.ln1.apply(ln_buf, hidden);
            layer.qkv.apply_into(ln_buf, t, qkv_buf);
            // qkv_buf is (T, 3·hidden) row-major: Q in columns 0..hidden, K in
            // hidden..2·hidden, V in 2·hidden..3·hidden (strided views below).
            // K is transposed once per layer into a contiguous (hidden, T)
            // buffer so the scores gemm packs each head's K^T rows sequentially
            // instead of gathering them from a 3·hidden-strided view.
            kt_buf
                .par_chunks_mut(64 * t)
                .enumerate()
                .for_each(|(band, chunk)| {
                    let i0 = band * 64;
                    for j in 0..t {
                        let src = &qkv_buf[j * 3 * hidden + hidden + i0..j * 3 * hidden + hidden + i0 + 64];
                        for (ii, v) in src.iter().enumerate() {
                            chunk[ii * t + j] = *v;
                        }
                    }
                });

            for head in 0..heads {
                let off = head * head_dim;
                // block over queries so the score matrix stays small
                for q0 in (0..t).step_by(q_block) {
                    let qb = q_block.min(t - q0);
                    let sptr = scores.as_mut_ptr();
                    // scores (qb, t) = q_h (qb, 64) @ k_h^T (64, t), scaled 1/8
                    // lhs: Q rows q0.. (element (j, d) = buf[(q0+j)·3h + off + d])
                    // rhs: contiguous K^T rows [off .. off+64)
                    unsafe {
                        gemm::<f32>(
                            qb,
                            t,
                            head_dim,
                            sptr,
                            1,
                            t as isize,
                            false,
                            qkv_buf.as_ptr().add(q0 * 3 * hidden + off),
                            1,
                            (3 * hidden) as isize,
                            kt_buf.as_ptr().add(off * t),
                            1,
                            t as isize,
                            1.0,
                            0.125,
                            false,
                            false,
                            false,
                            Parallelism::Rayon(0),
                        );
                    }
                    scores[..qb * t].par_chunks_exact_mut(t).for_each(softmax_row);
                    // attn rows (qb, 64) = scores (qb, t) @ v_h (t, 64)
                    // rhs: V (element (i, d) = buf[i·3h + 2h + off + d], rs=3h, cs=1)
                    unsafe {
                        gemm::<f32>(
                            qb,
                            head_dim,
                            t,
                            attn_buf.as_mut_ptr().add(q0 * hidden + off),
                            1,
                            hidden as isize,
                            false,
                            scores.as_ptr(),
                            1,
                            t as isize,
                            qkv_buf.as_ptr().add(2 * hidden + off),
                            1,
                            (3 * hidden) as isize,
                            1.0,
                            1.0,
                            false,
                            false,
                            false,
                            Parallelism::Rayon(0),
                        );
                    }
                }
            }
            layer.out_proj.apply_into(attn_buf, t, proj_buf);
            h_next.copy_from_slice(h);
            add_par(h_next, proj_buf);
            if collect_layers.contains(&li) {
                out_stages.attn_out.insert(li, h_next.to_vec());
            }

            // FFN block: h = h_next + ff2(gelu(ff1(LN(h_next))))
            ln_buf.copy_from_slice(h_next);
            layer.ln2.apply(ln_buf, hidden);
            layer.ff1.apply_into(ln_buf, t, ff_buf);
            gelu(ff_buf);
            layer.ff2.apply_into(ff_buf, t, ff2_buf);
            add_par(h_next, ff2_buf);
            std::mem::swap(&mut h, &mut h_next);
            if collect_layers.contains(&li) {
                out_stages.layer_out.insert(li, h.to_vec());
            }
        }

        self.final_ln.apply(h, hidden);
        if stages.enc_final {
            out_stages.enc_final = h.to_vec();
        }

        // ---- lm head + log softmax
        // log_probs = logits - max - ln(sum(exp(logits - max))): the exp is
        // needed only for the row sum, so there is no per-element ln.
        let logits = self.lm_head.apply(h, t);
        let mut log_probs = logits;
        let vocab = self.cfg.vocab_size;
        log_probs.par_chunks_exact_mut(vocab).for_each(log_softmax_row);

        Ok((log_probs, out_stages))
    }

    /// One conv layer: y = bias + Σ_tap x_shifted @ w_tap, then LN + gelu.
    ///
    /// The k-tap sum is k GEMMs against shifted strided views of x, so the
    /// `[out, in, k]` weight is consumed directly through strides:
    /// tap matrix B_tap (in, out) has element (ci, c) = w[(c·in + ci)·k + tap].
    pub fn conv_step(&self, cl: &ConvLayer, x: &[f32], t_in: usize) -> Vec<f32> {
        let (cout, cin, k, s) = (cl.out, cl.in_, cl.k, cl.stride);
        assert!(
            t_in >= k,
            "conv input too short: {} frames < kernel {k}",
            t_in
        );
        let t_out = (t_in - k) / s + 1;
        let mut y = vec![0.0f32; t_out * cout];
        for row in y.chunks_exact_mut(cout) {
            row.copy_from_slice(&cl.bias);
        }
        for kt in 0..k {
            // A_tap (t_out, cin): element (j, ci) = x[(j·s + tap)·cin + ci]
            unsafe {
                gemm::<f32>(
                    t_out,
                    cout,
                    cin,
                    y.as_mut_ptr(),
                    1,
                    cout as isize,
                    true,
                    x.as_ptr().add(kt * cin),
                    1,
                    (s * cin) as isize,
                    cl.weight.as_ptr().add(kt),
                    (cin * k) as isize,
                    k as isize,
                    1.0,
                    1.0,
                    false,
                    false,
                    false,
                    Parallelism::Rayon(0),
                );
            }
        }
        cl.ln.apply(&mut y, cout);
        gelu(&mut y);
        y
    }

    pub fn pos_conv(&self, h: &[f32], t: usize) -> Vec<f32> {
        // h: (T, 1024) row-major; depthwise conv over T with groups=16.
        // Pad the input with k/2 zero rows so every tap is a clean shifted
        // view, then each (group, tap) pair is one small GEMM accumulated
        // into this group's column strip of the output.
        let hidden = self.cfg.hidden_size;
        let k = self.cfg.num_conv_pos_embeddings;
        let groups = self.cfg.num_conv_pos_embedding_groups;
        let in_pg = hidden / groups;
        let pad = k / 2;

        let mut hp = vec![0.0f32; (t + 2 * pad) * hidden];
        hp[pad * hidden..pad * hidden + t * hidden].copy_from_slice(h);

        let mut out = vec![0.0f32; t * hidden]; // drop the last frame
        for row in out.chunks_exact_mut(hidden) {
            row.copy_from_slice(&self.pos_conv_bias);
        }

        let out_ptr = SendPtr(out.as_mut_ptr());
        (0..groups).into_par_iter().for_each(move |g| {
            // SAFETY: each group writes a disjoint 64-column strip of `out`.
            let dst = out_ptr.add(g * in_pg);
            for tk in 0..k {
                // A (t, in_pg): hp rows [tap .. tap+t), columns of this group
                // B (in_pg, in_pg): element (ci, oc) =
                //     w[((g·in_pg + oc)·in_pg + ci)·k + tap]
                unsafe {
                    gemm::<f32>(
                        t,
                        in_pg,
                        in_pg,
                        dst,
                        1,
                        hidden as isize,
                        true,
                        hp.as_ptr().add(tk * hidden + g * in_pg),
                        1,
                        hidden as isize,
                        self.pos_conv_weight.as_ptr().add(g * in_pg * in_pg * k + tk),
                        (in_pg * k) as isize,
                        k as isize,
                        1.0,
                        1.0,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
            }
        });
        // gelu applied by caller order: conv -> pad-remove -> activation
        gelu(&mut out);
        out
    }
}
