//! CPU kernels: elementwise ops and GEMM wrappers used by the forward pass.
//!
//! gelu / softmax / log_softmax route to the AVX2 kernels in [`crate::simd`]
//! when the CPU has them, and fall back to the scalar fast math otherwise.

use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use crate::simd::{fast_erf, fast_exp};

use super::{LayerNorm, Linear};

pub(super) fn gelu(x: &mut [f32]) {
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
            *v = 0.5 * *v * (1.0 + fast_erf(*v / std::f32::consts::SQRT_2));
        }
    });
}

/// dst = b (broadcast per row) then exact gelu, over `(rows, cols)`.
///
/// ff1's bias is folded here instead of being prefilled into the GEMM
/// destination — see [`add3_bias_par`].  The row is L1-resident, so the extra
/// bias sweep inside the row is free.
pub(super) fn gelu_bias(x: &mut [f32], bias: &[f32]) {
    x.par_chunks_exact_mut(bias.len()).for_each(|row| {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += *b;
        }
        #[cfg(target_arch = "x86_64")]
        if crate::simd::avx2::have_avx2_fma() {
            unsafe { crate::simd::avx2::gelu_inplace(row) };
            return;
        }
        for v in row.iter_mut() {
            *v = 0.5 * *v * (1.0 + fast_erf(*v / std::f32::consts::SQRT_2));
        }
    });
}

/// dst += src, parallel over blocks.
pub(super) fn add_par(dst: &mut [f32], src: &[f32]) {
    dst.par_chunks_mut(1 << 15)
        .zip(src.par_chunks(1 << 15))
        .for_each(|(d, s)| {
            for (a, b) in d.iter_mut().zip(s) {
                *a += b;
            }
        });
}

/// dst = a + x + bias, over `(rows, cols)` row-major buffers, parallel over
/// rows.
///
/// The residual add is where a linear layer's bias lands now that its GEMM
/// runs with `read_dst = false`: writing the bias into `y` before the GEMM and
/// having the GEMM read it back cost two extra sweeps of the output per call —
/// 28 MB each way for ff1, ~2 GB per 34 s chunk across the encoder.
pub(super) fn add3_bias_par<'a>(dst: &mut [f32], a: &[f32], x: &[f32], bias: &'a [f32]) {
    dst.par_chunks_exact_mut(bias.len())
        .zip(a.par_chunks(bias.len()))
        .zip(x.par_chunks(bias.len()))
        .for_each(|((d, p), q)| {
            for (((o, r), v), b) in d.iter_mut().zip(p).zip(q).zip(bias) {
                *o = *r + *v + *b;
            }
        });
}

/// dst += x + bias, same layout as [`add3_bias_par`].
pub(super) fn add_bias_par(dst: &mut [f32], x: &[f32], bias: &[f32]) {
    dst.par_chunks_exact_mut(bias.len())
        .zip(x.par_chunks(bias.len()))
        .for_each(|(d, q)| {
            for ((o, r), b) in d.iter_mut().zip(q).zip(bias) {
                *o += *r + *b;
            }
        });
}

/// Softmax pass 1 over one row: store `exp(x - max)` and return the row sum.
///
/// The normalisation lives in [`scale_attn_rows_par`] instead, applied to the
/// (t, 64) attention tiles once per layer rather than to every (t, t) score
/// tile — see the simd.rs note.
#[inline]
pub(super) fn softmax_row_sum(row: &mut [f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2::have_avx2_fma() {
        unsafe { return crate::simd::avx2::softmax_exp_sum_inplace(row) };
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for sc in row.iter_mut() {
        *sc = fast_exp(*sc - max);
        sum += *sc;
    }
    sum
}

/// Normalisation moved out of the softmax: `attn[j] *= 1 / sums[j]` over
/// `(t, hidden)` rows, where `sums[h·t + j]` is head `h`'s exp sum for query
/// `j`.  Column block `h` (64 wide) belongs to head `h`.
pub(super) fn scale_attn_rows_par(attn: &mut [f32], sums: &[f32], t: usize) {
    let hidden = attn.len() / t;
    let heads = sums.len() / t;
    debug_assert_eq!(sums.len(), heads * t);
    let head_dim = hidden / heads;
    attn.par_chunks_exact_mut(hidden)
        .enumerate()
        .for_each(|(j, row)| {
            for h in 0..heads {
                let inv = 1.0 / sums[h * t + j];
                for v in row[h * head_dim..(h + 1) * head_dim].iter_mut() {
                    *v *= inv;
                }
            }
        });
}

/// log_softmax one row in place: x - max - ln(sum(exp(x - max))).
#[inline]
pub(super) fn log_softmax_row(row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2::have_avx2_fma() {
        unsafe { crate::simd::avx2::log_softmax_inplace(row) };
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in row.iter() {
        sum += fast_exp(*x - max);
    }
    let lsum = sum.ln();
    for x in row.iter_mut() {
        *x = *x - max - lsum;
    }
}

/// Fused lm-head epilogue for one row: `out[j] = log_softmax(x + bias)[cols[j]]`.
///
/// The full `(t, vocab)` log_probs matrix is never materialised: max and
/// sum-exp sweep the logits once (the row is L2-resident, freshly written by
/// the GEMM), and only the trellis labels' columns are written out.  The
/// reduction order mirrors [`log_softmax_row`] / the AVX2
/// `log_softmax_inplace` lane for lane, so the gathered values agree with the
/// full-matrix path to the GEMM's destination-add rounding.
#[inline]
pub(super) fn log_softmax_gather_row(x: &[f32], bias: &[f32], cols: &[i32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), bias.len());
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2::have_avx2_fma() {
        unsafe { crate::simd::avx2::log_softmax_gather_inplace(x, bias, cols, out) };
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for (v, b) in x.iter().zip(bias) {
        max = max.max(*v + *b);
    }
    let mut sum = 0.0f32;
    for (v, b) in x.iter().zip(bias) {
        sum += fast_exp(*v + *b - max);
    }
    let c = max + sum.ln();
    for (o, &st) in out.iter_mut().zip(cols) {
        let st = st as usize;
        *o = x[st] + bias[st] - c;
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

    /// `y = x @ w^T` with the bias left off — the caller folds it into a pass
    /// that already touches `y` (residual add, gelu, ...).  `y` is written
    /// without being read, which skips both the bias prefill and the GEMM's
    /// read of the prefilled destination.
    pub fn apply_into_nobias(&self, x: &[f32], rows: usize, y: &mut [f32]) {
        unsafe {
            gemm::<f32>(
                rows,
                self.out,
                self.in_,
                y.as_mut_ptr(),
                1,
                self.out as isize,
                false,
                x.as_ptr(),
                1,
                self.in_ as isize,
                self.w.as_ptr(),
                self.in_ as isize,
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

    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * self.out];
        self.apply_into(x, rows, &mut y);
        y
    }
}

impl LayerNorm {
    /// Normalise `(rows, cols)` **out of place**: `dst = LayerNorm(src)`.
    ///
    /// The encoder keeps its normalised activations in a different buffer from
    /// the residual stream, and copying first would cost an extra read+write
    /// of the whole tensor per call — 48 such calls per 34 s chunk.
    pub fn apply_from(&self, src: &[f32], dst: &mut [f32], cols: usize) {
        let w = &self.w;
        let b = &self.b;
        let eps = self.eps;
        dst.par_chunks_exact_mut(cols)
            .zip(src.par_chunks(cols))
            .for_each(|(d, s)| {
                norm_row(s, d, w, b, eps);
            });
    }
}

/// One row of LayerNorm: `dst = (src - mean) / sqrt(var + eps) * w + b`.
///
/// The reductions use 8 independent lanes so the compiler emits vaddps instead
/// of a serial FADD chain; the lane order and the epsilon-inside-the-sqrt are
/// what keep this matching the Python reference.
fn norm_row(src: &[f32], dst: &mut [f32], w: &[f32], b: &[f32], eps: f64) {
    let n = src.len() as f32;
    let mut acc = [0.0f32; 8];
    let mut chunks = src.chunks_exact(8);
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
    let mut chunks = src.chunks_exact(8);
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
    for ((x, (w, b)), s) in dst.iter_mut().zip(w.iter().zip(b.iter())).zip(src.iter()) {
        *x = (*s - mean) / inv * w + b;
    }
}

/// Fused conv epilogue: `row = gelu(LN(row + conv_bias))`.
///
/// The conv GEMMs run with `read_dst = false`, so their output carries no
/// bias; folding it into the LayerNorm pass here makes the bias add free (that
/// pass already reads and rewrites the row) and removes a full
/// read-modify-write sweep over the layer's output — which for conv0 is
/// 223 MB per chunk.  gelu is applied in the same pass instead of a second
/// one, saving another read + write.
///
/// The reduction order deliberately mirrors [`LayerNorm::apply_from`] lane for
/// lane so the two paths agree bit for bit.
pub(super) fn conv_ln_gelu(y: &mut [f32], cols: usize, conv_bias: &[f32], ln: &LayerNorm) {
    let w = &ln.w;
    let b = &ln.b;
    let eps = ln.eps;
    #[cfg(target_arch = "x86_64")]
    let avx2 = crate::simd::avx2::have_avx2_fma();

    y.par_chunks_exact_mut(cols).for_each(|row| {
        for (v, cb) in row.iter_mut().zip(conv_bias.iter()) {
            *v += *cb;
        }
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

        #[cfg(target_arch = "x86_64")]
        if avx2 {
            unsafe { crate::simd::avx2::gelu_inplace(row) };
            return;
        }
        for v in row.iter_mut() {
            *v = 0.5 * *v * (1.0 + fast_erf(*v / std::f32::consts::SQRT_2));
        }
    });
}

/// Raw pointer wrapper so the pos-conv group loop can share the output
/// buffer across rayon tasks; each group writes a disjoint column strip.
pub(super) struct SendPtr(pub(super) *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    pub(super) fn add(&self, n: usize) -> *mut f32 {
        unsafe { self.0.add(n) }
    }
}
