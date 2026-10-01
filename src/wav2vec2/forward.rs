//! The CPU forward pass: conv stack → feature projection → pos conv →
//! encoder layers → lm_head, plus the reusable [`Scratch`] buffers.

use anyhow::Result;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use super::kernels::{add_par, gelu, log_softmax_row, SendPtr, softmax_row};
use super::{ConvLayer, Model, StageSet, Stages};

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
