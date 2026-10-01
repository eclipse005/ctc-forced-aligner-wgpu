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
//! The only free choices are GEMM accumulation orders; every op matches the
//! reference math (erf gelu, biased LN variance with eps inside the sqrt,
//! fp32 attention with 1/8 scaling).

use anyhow::{Context, Result};
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
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
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

fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        // exact gelu: 0.5*x*(1+erf(x/sqrt(2)))
        *v = 0.5 * *v * (1.0 + libm::erff(*v / std::f32::consts::SQRT_2));
    }
}

impl LayerNorm {
    fn apply(&self, h: &mut [f32], cols: usize) {
        // h is (rows, cols); normalise each row over cols
        for row in h.chunks_exact_mut(cols) {
            let n = cols as f32;
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
            let inv = ((var as f64 + self.eps).sqrt()) as f32;
            for (x, (w, b)) in row.iter_mut().zip(self.w.iter().zip(&self.b)) {
                *x = (*x - mean) / inv * w + b;
            }
        }
    }
}

impl Linear {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        // x: (rows, in) row-major; w: (out, in); y: (rows, out)
        let n = self.out;
        let k = self.in_;
        let mut y = vec![0.0f32; rows * n];
        y.par_chunks_mut(n).enumerate().for_each(|(r, yrow)| {
            let xrow = &x[r * k..(r + 1) * k];
            // broadcast rows of w^T: accumulate per input element
            for (xi, xv) in xrow.iter().enumerate() {
                if *xv == 0.0 {
                    continue;
                }
                let wcol = &self.w[xi..]; // w[o][xi] sits at self.w[o * in_ + xi]
                for (o, yv) in yrow.iter_mut().enumerate() {
                    *yv += xv * wcol[o * k];
                }
            }
            for (yv, bv) in yrow.iter_mut().zip(&self.b) {
                *yv += bv;
            }
        });
        y
    }
}

impl Model {
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let cfg = Wav2Vec2Config::load(model_dir)?;
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

        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let base = |sub: &str| format!("wav2vec2.encoder.layers.{i}.{sub}");
            let lin = |name: String, out: usize, inp: usize| -> Result<Linear> {
                let w = get_f32(&tensors, &format!("{name}.weight"))?;
                let b = get_f32(&tensors, &format!("{name}.bias"))?;
                anyhow::ensure!(w.1 == vec![out, inp], "{name} {:?}", w.1);
                Ok(Linear { w: w.0, b: b.0, out, in_: inp })
            };
            layers.push(EncoderLayer {
                q: lin(base("attention.q_proj"), cfg.hidden_size, cfg.hidden_size)?,
                k: lin(base("attention.k_proj"), cfg.hidden_size, cfg.hidden_size)?,
                v: lin(base("attention.v_proj"), cfg.hidden_size, cfg.hidden_size)?,
                out_proj: lin(base("attention.out_proj"), cfg.hidden_size, cfg.hidden_size)?,
                ln1: get_ln(&tensors, base("layer_norm"), cfg.hidden_size)?,
                ff1: lin(base("feed_forward.intermediate_dense"), cfg.intermediate_size, cfg.hidden_size)?,
                ff2: lin(base("feed_forward.output_dense"), cfg.hidden_size, cfg.intermediate_size)?,
                ln2: get_ln(&tensors, base("final_layer_norm"), cfg.hidden_size)?,
            });
        }

        let hidden_size = cfg.hidden_size;
        let lhw = get_f32(&tensors, "lm_head.weight")?;
        let lhb = get_f32(&tensors, "lm_head.bias")?;
        let lm_head = Linear { w: lhw.0, b: lhb.0, out: cfg.vocab_size, in_: cfg.hidden_size };

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
            final_ln: get_ln(&tensors, "wav2vec2.encoder.layer_norm".into(), hidden_size)?,
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
        let mut out_stages = Stages::default();

        // ---- conv stack: activations stay (T, C) row-major after each conv
        let mut cur: Vec<f32>;
        {
            // layer 0: in = 1 channel (the raw waveform)
            let c0 = &self.conv[0];
            let t_out = (input.len() - c0.k) / c0.stride + 1;
            cur = vec![0.0f32; t_out * c0.out];
            cur.par_chunks_mut(c0.out).enumerate().for_each(|(j, row)| {
                let base = j * c0.stride;
                for (c, o) in row.iter_mut().enumerate() {
                    let w = &c0.weight[c * c0.k..(c + 1) * c0.k];
                    let mut acc = c0.bias[c];
                    for (t, wt) in w.iter().enumerate() {
                        acc += input[base + t] * wt;
                    }
                    *o = acc;
                }
            });
            c0.ln.apply(&mut cur, c0.out);
            gelu(&mut cur);
            if stages.conv {
                out_stages.conv.push(cur.clone());
            }
        }
        for (_li, cl) in self.conv.iter().enumerate().skip(1) {
            let t_in = cur.len() / cl.in_;
            let t_out = (t_in - cl.k) / cl.stride + 1;
            let mut next = vec![0.0f32; t_out * cl.out];
            // weight is [out, in, k]: element (c, ci, t) at (c*in + ci)*k + t
            next.par_chunks_mut(cl.out).enumerate().for_each(|(j, row)| {
                let base = j * cl.stride;
                for (c, o) in row.iter_mut().enumerate() {
                    let w = &cl.weight[c * cl.k * cl.in_..(c + 1) * cl.k * cl.in_];
                    let mut acc = cl.bias[c];
                    for ci in 0..cl.in_ {
                        let w_ci = &w[ci * cl.k..(ci + 1) * cl.k];
                        for (t, wv) in w_ci.iter().enumerate() {
                            acc += cur[(base + t) * cl.in_ + ci] * wv;
                        }
                    }
                    *o = acc;
                }
            });
            cl.ln.apply(&mut next, cl.out);
            gelu(&mut next);
            cur = next;
            if stages.conv {
                out_stages.conv.push(cur.clone());
            }
        }

        let t = cur.len() / 512;
        let mut h = cur; // (T, 512)

        // ---- feature projection: LN -> linear
        self.feat_proj_ln.apply(&mut h, 512);
        h = self.feat_proj.apply(&h, t);
        if stages.proj {
            out_stages.proj = h.clone();
        }

        // ---- positional conv: depthwise, groups=16, k=128, pad 64, drop last
        let pos = self.pos_conv(&h, t);
        if stages.pos_conv {
            out_stages.pos_conv = pos.clone();
        }
        for (a, b) in h.iter_mut().zip(&pos) {
            *a += b;
        }
        if stages.enc_in {
            out_stages.enc_in = h.clone();
        }

        // ---- encoder layers (pre-LN)
        let hidden = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let head_dim = hidden / heads;
        let collect_layers = stages.layers.clone().unwrap_or_default();
        for (li, layer) in self.layers.iter().enumerate() {
            let residual = h.clone();
            let mut ln1 = h.clone();
            layer.ln1.apply(&mut ln1, hidden);

            let q = layer.q.apply(&ln1, t);
            let k = layer.k.apply(&ln1, t);
            let v = layer.v.apply(&ln1, t);

            let mut attn = vec![0.0f32; t * hidden];
            attn.par_chunks_mut(hidden).enumerate().for_each(|(t1, orow)| {
                let mut acc = vec![0.0f32; hidden];
                for head in 0..heads {
                    let off = head * head_dim;
                    // scores over all t2
                    let mut scores = vec![0.0f32; t];
                    let qrow = &q[t1 * hidden + off..t1 * hidden + off + head_dim];
                    for (t2, sc) in scores.iter_mut().enumerate() {
                        let krow = &k[t2 * hidden + off..t2 * hidden + off + head_dim];
                        let mut d = 0.0f32;
                        for (a, b) in qrow.iter().zip(krow) {
                            d += a * b;
                        }
                        *sc = d * 0.125;
                    }
                    // softmax
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - max).exp();
                        sum += *sc;
                    }
                    let inv = 1.0 / sum;
                    for (t2, p) in scores.iter().enumerate() {
                        let vrow = &v[t2 * hidden + off..t2 * hidden + off + head_dim];
                        for (d, vv) in vrow.iter().enumerate() {
                            acc[off + d] += p * inv * vv;
                        }
                    }
                }
                orow.copy_from_slice(&acc);
            });
            let mut attn_out = layer.out_proj.apply(&attn, t);
            for (r, a) in residual.iter().zip(attn_out.iter_mut()) {
                *a += r;
            }
            if collect_layers.contains(&li) {
                out_stages.attn_out.insert(li, attn_out.clone());
            }
            // residual is the pre-LN attention output; the FFN input is its LN
            let mut ln2 = attn_out.clone();
            layer.ln2.apply(&mut ln2, hidden);
            let mut ff = layer.ff1.apply(&ln2, t);
            gelu(&mut ff);
            let ff = layer.ff2.apply(&ff, t);
            for (a, b) in attn_out.iter_mut().zip(&ff) {
                *a += b;
            }
            h = attn_out;
            if collect_layers.contains(&li) {
                out_stages.layer_out.insert(li, h.clone());
            }
        }

        self.final_ln.apply(&mut h, hidden);
        if stages.enc_final {
            out_stages.enc_final = h.clone();
        }

        // ---- lm head + log softmax
        let logits = self.lm_head.apply(&h, t);
        let mut log_probs = logits;
        for row in log_probs.chunks_mut(self.cfg.vocab_size) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            let lsum = sum.ln();
            for x in row.iter_mut() {
                *x = x.ln() - lsum;
            }
        }

        Ok((log_probs, out_stages))
    }

    fn pos_conv(&self, h: &[f32], t: usize) -> Vec<f32> {
        // h: (T, 1024) row-major; depthwise conv over T with groups=16
        let hidden = self.cfg.hidden_size;
        let k = self.cfg.num_conv_pos_embeddings;
        let groups = self.cfg.num_conv_pos_embedding_groups;
        let in_pg = hidden / groups;
        let pad = k / 2;
        let t_out = t + pad * 2 - k + 1; // t + 1
        let mut out = vec![0.0f32; (t_out - 1) * hidden]; // drop the last frame

        // weight [1024, in_pg, k] is depthwise per group: out channel c reads
        // in channels (c / in_pg) * in_pg .. +in_pg
        out.par_chunks_mut(hidden).enumerate().for_each(|(j, orow)| {
            for (c, o) in orow.iter_mut().enumerate() {
                let g = c / in_pg;
                let ci0 = g * in_pg;
                let w = &self.pos_conv_weight[c * in_pg * k..(c + 1) * in_pg * k];
                let mut acc = self.pos_conv_bias[c];
                for tk in 0..k {
                    let src = j as i64 + tk as i64 - pad as i64;
                    if src < 0 || src >= t as i64 {
                        continue; // zero padding
                    }
                    let inrow = &h[src as usize * hidden + ci0..src as usize * hidden + ci0 + in_pg];
                    // weight [c][ci][tk] at (c*in_pg + ci)*k + tk: stride-k gather
                    for (ci, xv) in inrow.iter().enumerate() {
                        acc += xv * w[ci * k + tk];
                    }
                }
                *o = acc;
            }
        });
        // gelu applied by caller order: conv -> pad-remove -> activation
        gelu(&mut out);
        out
    }
}

