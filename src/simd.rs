//! AVX2 kernels for the CPU elementwise hot spots: gelu (erf), softmax,
//! and log_softmax.
//!
//! Accuracy matches the scalar `fast_exp` / `fast_erf` (same polynomials
//! and constants); only the evaluation is 8 lanes wide.  All exp arguments
//! are <= 0 (softmax, gelu), so the exp helper only needs the underflow
//! side; a degree-7 Taylor of 2^f on [-0.5, 0.5] keeps the relative error
//! near 5e-9, below f32 rounding.

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const LOG2E: f32 = 1.4426950408889634;
    const LN2: f64 = 0.6931471805599453;
    const E1: f32 = LN2 as f32;
    const E2: f32 = (LN2 * LN2 / 2.0) as f32;
    const E3: f32 = (LN2 * LN2 * LN2 / 6.0) as f32;
    const E4: f32 = (LN2 * LN2 * LN2 * LN2 / 24.0) as f32;
    const E5: f32 = (LN2 * LN2 * LN2 * LN2 * LN2 / 120.0) as f32;
    const E6: f32 = (LN2 * LN2 * LN2 * LN2 * LN2 * LN2 / 720.0) as f32;
    const E7: f32 = (LN2 * LN2 * LN2 * LN2 * LN2 * LN2 * LN2 / 5040.0) as f32;

    /// exp(x) per lane. Accurate for x <= 0 (underflows to 0 below ~-88);
    /// large positive x would overflow, but no call site produces those.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp8(x: __m256) -> __m256 {
        let y = _mm256_mul_ps(x, _mm256_set1_ps(LOG2E));
        let k = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);
        let f = _mm256_sub_ps(y, k);
        let mut p = _mm256_set1_ps(E7);
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(E6));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(E5));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(E4));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(E3));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(E2));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(E1));
        p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));
        // 2^k via exponent bits; k < -126 underflows to zero
        let ki = _mm256_add_epi32(_mm256_cvtps_epi32(k), _mm256_set1_epi32(127));
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32(ki, 23));
        let r = _mm256_mul_ps(scale, p);
        let under = _mm256_cmp_ps::<_CMP_LT_OQ>(k, _mm256_set1_ps(-126.0));
        _mm256_blendv_ps(r, _mm256_setzero_ps(), under)
    }

    /// erf(x) per lane (Abramowitz & Stegun 7.1.26, max abs err 1.5e-7).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn erf8(x: __m256) -> __m256 {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let ax = _mm256_and_ps(x, abs_mask);
        let t = _mm256_div_ps(
            _mm256_set1_ps(1.0),
            _mm256_fmadd_ps(_mm256_set1_ps(0.327_591_1), ax, _mm256_set1_ps(1.0)),
        );
        let mut poly = _mm256_set1_ps(1.061_405_429);
        poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-1.453_152_027));
        poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.421_413_741));
        poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-0.284_496_736));
        poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(0.254_829_592));
        poly = _mm256_mul_ps(poly, t);
        let neg_ax2 = _mm256_xor_ps(_mm256_mul_ps(ax, ax), _mm256_set1_ps(-0.0)); // -ax^2
        let e = exp8(neg_ax2);
        let r = _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(poly, e));
        // copysign: erf is odd — magnitude from r, sign bit from x
        let mag = _mm256_and_ps(r, abs_mask);
        let sign = _mm256_andnot_ps(abs_mask, x);
        _mm256_or_ps(mag, sign)
    }

    #[inline]
    unsafe fn reduce_sum(acc: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s4 = _mm_add_ps(lo, hi);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
        _mm_cvtss_f32(s1)
    }

    /// gelu in place: 0.5*x*(1+erf(x/sqrt(2))), 8 lanes at a time.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gelu_inplace(x: &mut [f32]) {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let frac_1_sqrt2 = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
        let mut chunks = x.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            let e = erf8(_mm256_mul_ps(v, frac_1_sqrt2));
            let g = _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, e));
            _mm256_storeu_ps(chunk.as_mut_ptr(), g);
        }
        for v in chunks.into_remainder() {
            *v = 0.5 * *v
                * (1.0 + crate::wav2vec2::fast_erf(*v / std::f32::consts::SQRT_2));
        }
    }

    /// Softmax one row in place.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_inplace(row: &mut [f32]) -> f32 {
        let mut max = f32::NEG_INFINITY;
        for v in row.iter() {
            max = max.max(*v);
        }
        // exp(x - max) with lane-partitioned sum
        let maxv = _mm256_set1_ps(max);
        let mut acc = _mm256_setzero_ps();
        let mut chunks = row.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(chunk.as_ptr()), maxv));
            acc = _mm256_add_ps(acc, e);
            _mm256_storeu_ps(chunk.as_mut_ptr(), e);
        }
        let mut tail_sum = 0.0f32;
        for v in chunks.into_remainder() {
            let e = crate::wav2vec2::fast_exp(*v - max);
            *v = e;
            tail_sum += e;
        }
        let sum = reduce_sum(acc) + tail_sum;
        let inv = _mm256_set1_ps(1.0 / sum);
        let mut chunks = row.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            _mm256_storeu_ps(
                chunk.as_mut_ptr(),
                _mm256_mul_ps(_mm256_loadu_ps(chunk.as_ptr()), inv),
            );
        }
        for v in chunks.into_remainder() {
            *v *= 1.0 / sum;
        }
        sum
    }

    /// log_softmax one row in place: x - max - ln(sum(exp(x - max))).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn log_softmax_inplace(row: &mut [f32]) {
        let mut max = f32::NEG_INFINITY;
        for v in row.iter() {
            max = max.max(*v);
        }
        let maxv = _mm256_set1_ps(max);
        let mut acc = _mm256_setzero_ps();
        let mut chunks = row.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(chunk.as_ptr()), maxv));
            acc = _mm256_add_ps(acc, e);
        }
        let mut tail_sum = 0.0f32;
        for v in chunks.into_remainder() {
            tail_sum += crate::wav2vec2::fast_exp(*v - max);
        }
        let lsum = (reduce_sum(acc) + tail_sum).ln();
        let c = _mm256_set1_ps(max + lsum);
        let mut chunks = row.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            _mm256_storeu_ps(
                chunk.as_mut_ptr(),
                _mm256_sub_ps(_mm256_loadu_ps(chunk.as_ptr()), c),
            );
        }
        for v in chunks.into_remainder() {
            *v -= max + lsum;
        }
    }

    pub fn have_avx2_fma() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
}
