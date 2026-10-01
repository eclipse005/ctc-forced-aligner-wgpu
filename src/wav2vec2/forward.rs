//! The CPU forward pass: conv stack → feature projection → pos conv →
//! encoder layers → lm_head, plus the reusable [`Scratch`] buffers.

use anyhow::Result;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use super::kernels::{
    add3_bias_par, add_bias_par, add_par, conv_ln_gelu, gelu, gelu_bias, log_softmax_gather_row,
    log_softmax_row, scale_attn_rows_par, softmax_row_sum, SendPtr,
};
use super::prof::{prof, *};
use super::{ConvLayer, Model, StageSet, Stages};

/// Transient forward buffers.  `forward_with` reuses them across calls so the
/// chunked encoder does not re-allocate (and re-fault) ~90 MB per chunk;
/// every buffer is fully overwritten before it is read, so stale contents
/// from a previous (longer) forward are harmless.
#[derive(Default)]
pub struct Scratch {
    ln: Vec<f32>,
    /// LayerNorm of the conv stack's last output, on its way into feat_proj.
    conv_ln: Vec<f32>,
    /// Depthwise positional conv: zero-padded input, output, and the
    /// per-group accumulation strips.
    pos_hp: Vec<f32>,
    pos_out: Vec<f32>,
    pos_acc: Vec<f32>,
    qkv: Vec<f32>,
    kt: Vec<f32>,
    attn: Vec<f32>,
    /// Per-head softmax row sums: `sums[h·t + j]`, for the normalisation
    /// folded into the attention output.
    sums: Vec<f32>,
    proj: Vec<f32>,
    h: Vec<f32>,
    h2: Vec<f32>,
    ff: Vec<f32>,
    ff2: Vec<f32>,
    scores: Vec<f32>,
    /// Conv-stack output ping-pong: even layers write `conv_p`, odd layers
    /// write `conv_q`.  conv0's output dominates (223 MB for a 34 s chunk),
    /// so these are sized per parity rather than to a single maximum.
    conv_p: Vec<f32>,
    conv_q: Vec<f32>,
    /// Gathered taps for conv layers whose banded input is smaller than their
    /// output — in practice only conv0 (in=1, k=10 -> a 4 MB `a` against a
    /// 223 MB `y`), which lets the layer be one GEMM instead of ten.
    conv_a: Vec<f32>,
    /// lm-head logits, reused across chunks so the (t, vocab) matrix — 70 MB
    /// per 34 s chunk — is never re-allocated and re-faulted.
    logits: Vec<f32>,
}

fn fit(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// Queries per attention tile.
///
/// Swept on the 265K (t = 1700): 1700 -> 18.4x, 1024 -> 17.0x, 512 -> 15.9x,
/// 256 -> 15.4x, 128 -> 13.8x.  Shrinking the tile so the score matrix fits
/// L2 instead of L3 *loses*, because `gemm`'s throughput is steeply dependent
/// on m and that costs more than the cache traffic it saves.  So the tile is
/// left at the whole chunk, and `CTC_Q_BLOCK` only exists to re-check that on
/// a different machine.
fn q_block_default(t: usize) -> usize {
    const DEFAULT: usize = 2048;
    let want = std::env::var("CTC_Q_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT);
    want.min(t).max(1)
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
        self.forward_inner(input, stages, scratch, None)
    }

    /// Same forward, but the lm-head epilogue emits only the gathered trellis
    /// labels' log-probs — a `(t, expanded.len())` matrix instead of the full
    /// `(t, vocab)` one.  The values equal `log_probs[..][st]` of
    /// [`Model::forward_with`] to the GEMM's destination-add rounding; the DP
    /// consumes exactly these columns, so nothing else is materialised.
    pub fn forward_gathered_with(
        &self,
        input: &[f32],
        stages: &StageSet,
        scratch: &mut Scratch,
        expanded: &[usize],
    ) -> Result<(Vec<f32>, Stages)> {
        self.forward_inner(input, stages, scratch, Some(expanded))
    }

    fn forward_inner(
        &self,
        input: &[f32],
        stages: &StageSet,
        scratch: &mut Scratch,
        gather: Option<&[usize]>,
    ) -> Result<(Vec<f32>, Stages)> {
        let mut out_stages = Stages::default();

        // ---- shapes first, so every scratch buffer is sized once and reused
        // by every chunk instead of re-allocating the ~334 MB the conv
        // ping-pong buffers hold (plus the ~90 MB the encoder needs).
        let mut t_out = Vec::with_capacity(self.conv.len());
        let mut t_cur = input.len();
        for cl in self.conv.iter() {
            anyhow::ensure!(
                t_cur >= cl.k,
                "conv input too short: {} frames < kernel {}",
                t_cur,
                cl.k
            );
            let n = (t_cur - cl.k) / cl.stride + 1;
            t_out.push(n);
            t_cur = n;
        }
        let t = *t_out.last().expect("conv stack is empty");

        let hidden = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let head_dim = hidden / heads;
        let intermediate = self.cfg.intermediate_size;
        let collect_layers = stages.layers.clone().unwrap_or_default();
        // Attention is tiled over queries (see [`q_block_default`] for why the
        // tile is left at the whole chunk).  Each tile is exactly the same
        // arithmetic whatever its size — the scores GEMM accumulates over k in
        // the same order, and softmax / PV are row-wise.
        let q_block = q_block_default(t);

        let (mut even_max, mut odd_max, mut band_max) = (0usize, 0usize, 0usize);
        for (i, (&n, cl)) in t_out.iter().zip(self.conv.iter()).enumerate() {
            let sz = n * cl.out;
            if i % 2 == 0 {
                even_max = even_max.max(sz);
            } else {
                odd_max = odd_max.max(sz);
            }
            if cl.in_ * cl.k <= cl.out {
                band_max = band_max.max(n * cl.in_ * cl.k);
            }
        }
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
        fit(&mut scratch.conv_ln, t * 512);
        fit(
            &mut scratch.pos_hp,
            (t + self.cfg.num_conv_pos_embeddings) * hidden,
        );
        fit(&mut scratch.pos_out, t * hidden);
        fit(&mut scratch.pos_acc, t * hidden);
        fit(&mut scratch.qkv, t * 3 * hidden);
        fit(&mut scratch.kt, hidden * t);
        fit(&mut scratch.sums, heads * t);
        fit(&mut scratch.ff, t * intermediate);
        fit(&mut scratch.scores, q_block * t);
        fit(&mut scratch.conv_p, even_max);
        fit(&mut scratch.conv_q, odd_max);
        fit(&mut scratch.conv_a, band_max);
        fit(&mut scratch.logits, t * self.cfg.vocab_size);

        let Scratch {
            ln,
            conv_ln,
            pos_hp,
            pos_out,
            pos_acc,
            qkv,
            kt,
            attn,
            sums,
            proj,
            h: h_buf,
            h2: h2_buf,
            ff,
            ff2,
            scores: sc,
            conv_p,
            conv_q,
            conv_a,
            logits,
        } = &mut *scratch;
        let ln_buf = &mut ln[..t * hidden];
        let conv_ln_buf = &mut conv_ln[..t * 512];
        let qkv_buf = &mut qkv[..t * 3 * hidden];
        let kt_buf = &mut kt[..hidden * t];
        let attn_buf = &mut attn[..t * hidden];
        let proj_buf = &mut proj[..t * hidden];
        let ff_buf = &mut ff[..t * intermediate];
        let ff2_buf = &mut ff2[..t * hidden];
        let scores = &mut sc[..q_block * t];

        // ---- conv stack: activations stay (T, C) row-major after each conv.
        // Even layers write `conv_p`, odd layers write `conv_q`, and each
        // layer's output is the next one's input.  That ping-pong aliases one
        // buffer as destination and the other as source across iterations,
        // which the borrow checker cannot see past a loop body, so the buffers
        // are reached through raw pointers — the same pattern pos_conv uses.
        let p = SendPtr(conv_p.as_mut_ptr());
        let q = SendPtr(conv_q.as_mut_ptr());
        let a_ptr = SendPtr(conv_a.as_mut_ptr());

        let mut src: &[f32] = input;
        let mut to_p = true;
        for (i, cl) in self.conv.iter().enumerate() {
            let n_out = t_out[i] * cl.out;
            // `conv_a` is only sized for the layers that take the banded path,
            // so the slice must not be built for the others — a `from_raw_parts`
            // past the allocation is UB even if nothing ever reads it.
            let banded = cl.in_ * cl.k <= cl.out;
            let n_a = if banded { t_out[i] * cl.in_ * cl.k } else { 0 };
            let dst_ptr = if to_p { p.add(0) } else { q.add(0) };
            prof!(CONV0 + i, {
                // SAFETY: `dst_ptr` is the ping-pong buffer that is *not* the
                // source of this layer — `src` points into the other one (or
                // into `input` for layer 0), so the two never overlap, and the
                // fits above gave that parity at least `n_out` floats.  `a` is
                // either the exact banded size `conv_a` was fitted to, or empty.
                unsafe {
                    let dst = std::slice::from_raw_parts_mut(dst_ptr, n_out);
                    let a: &mut [f32] = if banded {
                        std::slice::from_raw_parts_mut(a_ptr.add(0), n_a)
                    } else {
                        &mut []
                    };
                    self.conv_step_into(cl, src, t_out[i], dst, a);
                }
            });
            src = unsafe { std::slice::from_raw_parts(dst_ptr, n_out) };
            to_p = !to_p;
            if stages.conv {
                out_stages.conv.push(src.to_vec());
            }
        }
        // the stack's last output becomes an owned buffer for the epilogue
        // (3.5 MB — conv6's share, not conv0's)
        let cur: Vec<f32> = src.to_vec();

        // ---- feature projection: LN -> linear (residual stream in scratch)
        let mut h: &mut [f32] = &mut h_buf[..t * hidden];
        let mut h_next: &mut [f32] = &mut h2_buf[..t * hidden];
        prof!(FEAT_PROJ, {
            self.feat_proj_ln.apply_from(&cur, conv_ln_buf, 512);
            self.feat_proj.apply_into(conv_ln_buf, t, h);
        });
        if stages.proj {
            out_stages.proj = h.to_vec();
        }

        // ---- positional conv: depthwise, groups=16, k=128, pad 64, drop last
        prof!(POS_CONV, {
            self.pos_conv_into(h, t, pos_hp, pos_out, pos_acc);
        });
        if stages.pos_conv {
            out_stages.pos_conv = pos_out.to_vec();
        }
        prof!(RESIDUAL_ADD, {
            add_par(h, pos_out);
        });
        if stages.enc_in {
            out_stages.enc_in = h.to_vec();
        }

        // scratch buffers reused across layers: attention scores are blocked
        // over queries (max 2048 rows) so memory stays O(t) for long inputs

        for (li, layer) in self.layers.iter().enumerate() {
            // attention block: h_next = h + out_proj(attn(LN(h)))
            prof!(LAYER_NORM, {
                layer.ln1.apply_from(h, ln_buf, hidden);
            });
            prof!(QKV_GEMM, {
                layer.qkv.apply_into(ln_buf, t, qkv_buf);
            });
            // qkv_buf is (T, 3·hidden) row-major: Q in columns 0..hidden, K in
            // hidden..2·hidden, V in 2·hidden..3·hidden (strided views below).
            // K is transposed once per layer into a contiguous (hidden, T)
            // buffer so the scores gemm packs each head's K^T rows sequentially
            // instead of gathering them from a 3·hidden-strided view.
            prof!(KT_TRANSPOSE, {
                kt_buf
                    .par_chunks_mut(64 * t)
                    .enumerate()
                    .for_each(|(band, chunk)| {
                        let i0 = band * 64;
                        for j in 0..t {
                            let src = &qkv_buf
                                [j * 3 * hidden + hidden + i0..j * 3 * hidden + hidden + i0 + 64];
                            for (ii, v) in src.iter().enumerate() {
                                chunk[ii * t + j] = *v;
                            }
                        }
                    });
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
                    prof!(SCORES_GEMM, {
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
                    });
                    // pass 1 of the softmax: store exp(x - max), keep the row
                    // sum; the normalisation happens once on the attention
                    // output below instead of here on every score tile
                    prof!(ATTN_SOFTMAX, {
                        let sums_h = &mut sums[head * t + q0..head * t + q0 + qb];
                        scores[..qb * t]
                            .par_chunks_exact_mut(t)
                            .zip(sums_h.par_iter_mut())
                            .for_each(|(row, s)| *s = softmax_row_sum(row));
                    });
                    // attn rows (qb, 64) = scores (qb, t) @ v_h (t, 64)
                    // rhs: V (element (i, d) = buf[i·3h + 2h + off + d], rs=3h, cs=1)
                    prof!(PV_GEMM, {
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
                    });
                }
            }
            prof!(ATTN_SOFTMAX, {
                scale_attn_rows_par(attn_buf, sums, t);
            });
            prof!(OUT_PROJ, {
                layer.out_proj.apply_into_nobias(attn_buf, t, proj_buf);
            });
            prof!(RESIDUAL_ADD, {
                add3_bias_par(h_next, h, proj_buf, &layer.out_proj.b);
            });
            if collect_layers.contains(&li) {
                out_stages.attn_out.insert(li, h_next.to_vec());
            }

            // FFN block: h = h_next + ff2(gelu(ff1(LN(h_next))))
            prof!(LAYER_NORM, {
                layer.ln2.apply_from(h_next, ln_buf, hidden);
            });
            prof!(FF1_GEMM, {
                layer.ff1.apply_into_nobias(ln_buf, t, ff_buf);
            });
            prof!(GELU, {
                gelu_bias(ff_buf, &layer.ff1.b);
            });
            prof!(FF2_GEMM, {
                layer.ff2.apply_into_nobias(ff_buf, t, ff2_buf);
            });
            prof!(RESIDUAL_ADD, {
                add_bias_par(h_next, ff2_buf, &layer.ff2.b);
            });
            std::mem::swap(&mut h, &mut h_next);
            if collect_layers.contains(&li) {
                out_stages.layer_out.insert(li, h.to_vec());
            }
        }

        prof!(LAYER_NORM, {
            self.final_ln.apply_from(h, ln_buf, hidden);
        });
        if stages.enc_final {
            // `ln_buf` holds the normalised stream: `final_ln` is out of place,
            // so `h` is still the pre-norm residual at this point.
            out_stages.enc_final = ln_buf.to_vec();
        }

        // ---- lm head + log softmax
        // log_probs = logits - max - ln(sum(exp(logits - max))): the exp is
        // needed only for the row sum, so there is no per-element ln.
        //
        // The gathered path never materialises the (t, vocab) matrix: the
        // GEMM writes biased-free logits into reused scratch (no bias prefill,
        // no destination read), and one fused pass computes each row's
        // max / sum-exp and emits only the trellis columns.
        let vocab = self.cfg.vocab_size;
        let log_probs: Vec<f32>;
        match gather {
            None => {
                let mut lp;
                prof!(LM_HEAD, {
                    lp = self.lm_head.apply(ln_buf, t);
                });
                prof!(LOG_SOFTMAX, {
                    lp.par_chunks_exact_mut(vocab).for_each(log_softmax_row);
                });
                log_probs = lp;
            }
            Some(expanded) => {
                let logits_buf = &mut logits[..t * vocab];
                prof!(LM_HEAD, {
                    self.lm_head.apply_into_nobias(ln_buf, t, logits_buf);
                });
                let s = expanded.len();
                let mut out = vec![0.0f32; t * s];
                prof!(LOG_SOFTMAX, {
                    let cols: Vec<i32> = expanded.iter().map(|&x| x as i32).collect();
                    let bias = &self.lm_head.b;
                    logits_buf
                        .par_chunks_exact(vocab)
                        .zip(out.par_chunks_mut(s))
                        .for_each(|(x, o)| log_softmax_gather_row(x, bias, &cols, o));
                });
                log_probs = out;
            }
        }
        if enabled() {
            forward_done();
        }

        Ok((log_probs, out_stages))
    }

    /// One conv layer writing into `y` (which must hold `t_out * cout`
    /// floats and is fully overwritten):
    ///
    /// ```text
    /// y = gelu(LN(conv_bias + Σ_tap x_shifted @ w_tap))
    /// ```
    ///
    /// Two shapes are handled:
    ///
    /// * `in * k <= out` — the taps are gathered into one banded `a`
    ///   (`t_out × in·k`) so the layer becomes a *single* GEMM and the output
    ///   is written once instead of k times.  conv0 (`in=1, k=10, out=512`) is
    ///   the case that matters: its output is 223 MB per 34 s chunk, and the
    ///   old k accumulating `k=1` GEMMs meant k read-modify-write sweeps over
    ///   all of it — the single most expensive phase of the CPU forward
    ///   despite being <1% of its arithmetic.
    /// * otherwise the k tap GEMMs accumulate straight into `y`, all with
    ///   `read_dst = false` so `y` is only ever written, never read.
    ///
    /// Either way the bias is folded into the LayerNorm pass (see
    /// [`conv_ln_gelu`]) instead of being prefilled into `y` first.
    pub fn conv_step_into(
        &self,
        cl: &ConvLayer,
        x: &[f32],
        t_out: usize,
        y: &mut [f32],
        a: &mut [f32],
    ) {
        let (cout, cin, k, s) = (cl.out, cl.in_, cl.k, cl.stride);
        assert!(
            x.len() >= (t_out - 1) * s * cin + cin * k,
            "conv input too short for {} output frames",
            t_out
        );
        let y = &mut y[..t_out * cout];

        if cin * k <= cout {
            // banded A: element (j, d) = x[j·s·in + (d%k)·in + d/k], i.e. the
            // (in, k) tap window flattened tap-minor into d = ci·k + tap
            let kd = cin * k;
            assert!(
                a.len() >= t_out * kd,
                "banded tap buffer too small: {} < {}",
                a.len(),
                t_out * kd
            );
            let a = &mut a[..t_out * kd];
            if cin == 1 {
                for (j, row) in a.chunks_exact_mut(k).enumerate() {
                    row.copy_from_slice(&x[j * s..j * s + k]);
                }
            } else {
                for (j, row) in a.chunks_exact_mut(kd).enumerate() {
                    let base = j * s * cin;
                    for tap in 0..k {
                        for ci in 0..cin {
                            row[ci * k + tap] = x[base + tap * cin + ci];
                        }
                    }
                }
            }
            // A (t_out, kd) row-major; B (kd, out) with a `kd` column step,
            // which is exactly the checkpoint's [out, in, k] layout.
            unsafe {
                gemm::<f32>(
                    t_out,
                    cout,
                    kd,
                    y.as_mut_ptr(),
                    1,
                    cout as isize,
                    false,
                    a.as_ptr(),
                    1,
                    kd as isize,
                    cl.weight.as_ptr(),
                    kd as isize,
                    1,
                    1.0,
                    1.0,
                    false,
                    false,
                    false,
                    Parallelism::Rayon(0),
                );
            }
        } else {
            for kt in 0..k {
                // A_tap (t_out, cin): element (j, ci) = x[(j·s + tap)·cin + ci].
                // Only the *first* tap can write instead of read: `gemm`
                // zeroes alpha when `read_dst` is false, so dst is stored
                // rather than accumulated into.  Later taps must accumulate,
                // which costs one extra read pass over `y` — still fewer than
                // prefilling the bias up front.
                unsafe {
                    gemm::<f32>(
                        t_out,
                        cout,
                        cin,
                        y.as_mut_ptr(),
                        1,
                        cout as isize,
                        kt != 0,
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
        }
        conv_ln_gelu(y, cout, &cl.bias, &cl.ln);
    }

    /// Allocating wrapper around [`Model::conv_step_into`]; the forward pass
    /// uses the scratch-backed variant instead.
    pub fn conv_step(&self, cl: &ConvLayer, x: &[f32], t_in: usize) -> Vec<f32> {
        let t_out = (t_in - cl.k) / cl.stride + 1;
        let mut y = vec![0.0f32; t_out * cl.out];
        let mut a = vec![0.0f32; t_out * cl.in_ * cl.k];
        self.conv_step_into(cl, x, t_out, &mut y, &mut a);
        y
    }

    /// Allocating wrapper around [`Model::pos_conv_into`]; the forward pass
    /// uses the scratch-backed variant.
    pub fn pos_conv(&self, h: &[f32], t: usize) -> Vec<f32> {
        let hidden = self.cfg.hidden_size;
        let k = self.cfg.num_conv_pos_embeddings;
        let pad = k / 2;
        let mut hp = vec![0.0f32; (t + 2 * pad) * hidden];
        let mut out = vec![0.0f32; t * hidden];
        let mut acc = vec![0.0f32; t * hidden];
        self.pos_conv_into(h, t, &mut hp, &mut out, &mut acc);
        out
    }

    /// Depthwise positional conv over time, writing into caller-owned buffers.
    ///
    /// `h` is (T, hidden) row-major, groups = 16, k = 128 taps, input padded by
    /// k/2 on both sides, last frame dropped, then gelu.  Each (group, tap)
    /// is one (T, 64) x (64, 64) GEMM.  The (group, row-block) grid runs over
    /// rayon: 16 group tasks would leave 4 of 20 threads idle for the whole
    /// op and serialise 128 small GEMMs per task; splitting rows as well
    /// fills the machine (the tap order per output row is unchanged, so the
    /// values are bit-identical).
    ///
    /// The taps accumulate into a **contiguous** (T, 64) scratch strip and only
    /// then land in this group's column strip of `out`.  Accumulating straight
    /// into `out` meant every tap read-modify-wrote a strided slice of the 7 MB
    /// output: 128 taps x 16 groups x ~435 KB, about 1.8 GB of traffic per 34 s
    /// chunk.  The strip is 435 KB and stays in L2, so only one strided write
    /// per group survives to memory.
    ///
    /// `hp` must hold `(t + k) * hidden`, `out` and `acc` `t * hidden` each
    /// (`acc` is the per-group strip: groups x (t x hidden/groups)).
    pub fn pos_conv_into(
        &self,
        h: &[f32],
        t: usize,
        hp: &mut [f32],
        out: &mut [f32],
        acc: &mut [f32],
    ) {
        const ROW_SPLITS: usize = 4;
        let hidden = self.cfg.hidden_size;
        let k = self.cfg.num_conv_pos_embeddings;
        let groups = self.cfg.num_conv_pos_embedding_groups;
        let in_pg = hidden / groups;
        let pad = k / 2;

        // pad the input with k/2 zero rows so every tap is a clean shifted
        // view; only the pads need clearing, the body is overwritten
        let hp = &mut hp[..(t + 2 * pad) * hidden];
        hp[..pad * hidden].fill(0.0);
        hp[pad * hidden..pad * hidden + t * hidden].copy_from_slice(h);
        hp[pad * hidden + t * hidden..].fill(0.0);

        let out = &mut out[..t * hidden]; // drop the last frame

        let out_ptr = SendPtr(out.as_mut_ptr());
        let acc_ptr = SendPtr(acc.as_mut_ptr());
        let w = &self.pos_conv_weight;
        let bias = &self.pos_conv_bias;
        let strip = t * in_pg;
        let rows_per = t.div_ceil(ROW_SPLITS);
        (0..groups * ROW_SPLITS).into_par_iter().for_each(move |task| {
            let g = task / ROW_SPLITS;
            let blk = task % ROW_SPLITS;
            let row0 = blk * rows_per;
            let rows = rows_per.min(t.saturating_sub(row0));
            if rows == 0 {
                return;
            }
            let strip_ptr = acc_ptr.add(g * strip + row0 * in_pg);
            let group_bias = &bias[g * in_pg..(g + 1) * in_pg];
            // The strip is the accumulator, so it carries the bias — the same
            // order the taps used to accumulate in, which keeps the rounding
            // identical to the pre-optimisation path.  `out` needs no
            // initialisation: every column strip is overwritten exactly once.
            //
            // SAFETY: task (g, blk) owns acc[g·strip + row0·in_pg ..][..rows·in_pg]
            // outright; no other task touches those rows.
            let acc_strip = unsafe { std::slice::from_raw_parts_mut(strip_ptr, rows * in_pg) };
            for row in acc_strip.chunks_exact_mut(in_pg) {
                row.copy_from_slice(group_bias);
            }
            for tk in 0..k {
                // A (rows, in_pg): hp rows [row0 + tap .. row0 + tap + rows), columns of this group
                // B (in_pg, in_pg): element (ci, oc) =
                //     w[((g·in_pg + oc)·in_pg + ci)·k + tap]
                //
                // SAFETY: reads only hp and this group's weight slice; writes
                // only its own accumulator strip rows.
                unsafe {
                    gemm::<f32>(
                        rows,
                        in_pg,
                        in_pg,
                        strip_ptr,
                        1,
                        in_pg as isize,
                        true,
                        hp.as_ptr().add((tk + row0) * hidden + g * in_pg),
                        1,
                        hidden as isize,
                        w.as_ptr().add(g * in_pg * in_pg * k + tk),
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
            // SAFETY: each task writes a disjoint (rows, 64) patch of its
            // group's column strip of `out`, reading only its own strip rows.
            let dst = out_ptr.add(g * in_pg + row0 * hidden);
            for j in 0..rows {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        strip_ptr.add(j * in_pg),
                        dst.add(j * hidden),
                        in_pg,
                    );
                }
            }
        });
        // gelu applied by caller order: conv -> pad-remove -> activation
        gelu(out);
    }
}
