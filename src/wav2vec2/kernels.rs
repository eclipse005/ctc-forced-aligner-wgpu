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

/// Softmax one row in place (`fast_exp` under the hood).
#[inline]
pub(super) fn softmax_row(row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2::have_avx2_fma() {
        unsafe { crate::simd::avx2::softmax_inplace(row) };
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for sc in row.iter_mut() {
        *sc = fast_exp(*sc - max);
        sum += *sc;
    }
    let inv = 1.0 / sum;
    for sc in row.iter_mut() {
        *sc *= inv;
    }
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
pub(super) struct SendPtr(pub(super) *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    pub(super) fn add(&self, n: usize) -> *mut f32 {
        unsafe { self.0.add(n) }
    }
}
