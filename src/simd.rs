//! Fast math for the CPU elementwise hot spots: scalar `fast_exp` / `fast_erf`
//! and their AVX2 twins for gelu (erf), softmax, and log_softmax.
//!
//! Accuracy matches across the scalar and AVX2 paths (same polynomials and
//! constants); the AVX2 versions only evaluate 8 lanes wide.  All exp
//! arguments are <= 0 (softmax, gelu), so the exp helpers only need the
//! underflow side; a degree-7 Taylor of 2^f on [-0.5, 0.5] keeps the relative
//! error near 5e-9, below f32 rounding.  erf is Abramowitz & Stegun 7.1.26
//! (max abs err 1.5e-7, same class as libm erff).

// exp(x) = 2^(x·log2e): split into k = round(x·log2e) and f ∈ [-0.5, 0.5),
// 2^k via exponent bits and 2^f via a degree-7 Taylor series (rel err
// ~5e-9, below the f32 rounding).  All call sites pass x <= 0, so only
// the underflow side needs care.

const LOG2_E: f32 = 1.4426950408889634;
const LN2: f64 = 0.6931471805599453;
const E1: f32 = LN2 as f32;
const E2: f32 = (LN2 * LN2 / 2.0) as f32;
const E3: f32 = (LN2 * LN2 * LN2 / 6.0) as f32;
const E4: f32 = (LN2 * LN2 * LN2 * LN2 / 24.0) as f32;
const E5: f32 = (LN2 * LN2 * LN2 * LN2 * LN2 / 120.0) as f32;
const E6: f32 = (LN2 * LN2 * LN2 * LN2 * LN2 * LN2 / 720.0) as f32;
const E7: f32 = (LN2 * LN2 * LN2 * LN2 * LN2 * LN2 * LN2 / 5040.0) as f32;

#[inline]
pub(crate) fn fast_exp(x: f32) -> f32 {
    let xf = x * LOG2_E;
    // round to nearest integer via the 1.5·2^23 magic number
    let k = (xf + 12582912.0) - 12582912.0;
    if k < -126.0 {
        return 0.0; // true result is subnormal; the sum is unaffected
    }
    let f = xf - k;
    let p = E7 * f + E6;
    let p = p * f + E5;
    let p = p * f + E4;
    let p = p * f + E3;
    let p = p * f + E2;
    let p = p * f + E1;
    let p = p * f + 1.0;
    let scale = f32::from_bits((((k + 127.0) as i32) as u32) << 23);
    scale * p
}

/// Abramowitz & Stegun 7.1.26 (max abs err 1.5e-7, same class as libm erff).
#[inline]
pub(crate) fn fast_erf(x: f32) -> f32 {
    let ax = x.abs();
    if ax > 5.0 {
        // exp(-25) = 1.4e-11: erf is ±1 to f32 precision
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t)
            * fast_exp(-ax * ax);
    if x < 0.0 {
        -y
    } else {
        y
    }
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use super::{E1, E2, E3, E4, E5, E6, E7, LOG2_E};
    use std::arch::x86_64::*;

    /// exp(x) per lane. Accurate for x <= 0 (underflows to 0 below ~-88);
    /// large positive x would overflow, but no call site produces those.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp8(x: __m256) -> __m256 {
        let y = _mm256_mul_ps(x, _mm256_set1_ps(LOG2_E));
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
            *v = 0.5 * *v * (1.0 + super::fast_erf(*v / std::f32::consts::SQRT_2));
        }
    }

    /// Softmax one row in place.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_inplace(row: &mut [f32]) -> f32 {
        let sum = softmax_exp_sum_inplace(row);
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

    /// Softmax pass 1 only: scale by `exp(x - max)` in place and return the
    /// row sum.  The normalisation is folded into the attention output
    /// instead — dividing the (t, 64) attention tile rows costs 336 MB per
    /// chunk, versus ~8.9 GB of L3 traffic for a third sweep over the
    /// (t, t) score tiles.
    ///
    /// Same lane structure as [`softmax_inplace`], so the exp values and the
    /// sum are identical.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_exp_sum_inplace(row: &mut [f32]) -> f32 {
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
            let e = super::fast_exp(*v - max);
            *v = e;
            tail_sum += e;
        }
        reduce_sum(acc) + tail_sum
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
            tail_sum += super::fast_exp(*v - max);
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

    /// Fused lm-head epilogue for one row:
    /// `out[j] = log_softmax(x + bias)[cols[j]]`.
    ///
    /// Same lane structure as [`log_softmax_inplace`]: serial max over the
    /// biased row, 8-lane exp sums, `out = (x+b) - (max + ln(sum))` on the
    /// gathered columns only.
    ///
    /// # Safety
    /// `x` and `bias` must have the same length; every index in `cols` must be
    /// in bounds for both; `out` must hold `cols.len()` floats.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn log_softmax_gather_inplace(
        x: &[f32],
        bias: &[f32],
        cols: &[i32],
        out: &mut [f32],
    ) {
        let mut max = f32::NEG_INFINITY;
        for (v, b) in x.iter().zip(bias) {
            max = max.max(*v + *b);
        }
        let maxv = _mm256_set1_ps(max);
        let mut acc = _mm256_setzero_ps();
        let mut chunks = x.chunks_exact(8);
        let mut bchunks = bias.chunks_exact(8);
        for (c, bc) in chunks.by_ref().zip(bchunks.by_ref()) {
            let e = exp8(_mm256_sub_ps(
                _mm256_add_ps(_mm256_loadu_ps(c.as_ptr()), _mm256_loadu_ps(bc.as_ptr())),
                maxv,
            ));
            acc = _mm256_add_ps(acc, e);
        }
        let mut tail_sum = 0.0f32;
        for (v, b) in chunks.remainder().iter().zip(bchunks.remainder().iter()) {
            tail_sum += super::fast_exp(*v + *b - max);
        }
        let c = max + (reduce_sum(acc) + tail_sum).ln();
        let cv = _mm256_set1_ps(c);
        let mut ci = 0;
        while ci + 8 <= cols.len() {
            let idx = _mm256_loadu_si256(cols.as_ptr().add(ci) as *const __m256i);
            let g = _mm256_i32gather_ps(x.as_ptr(), idx, 4);
            let gb = _mm256_i32gather_ps(bias.as_ptr(), idx, 4);
            _mm256_storeu_ps(
                out.as_mut_ptr().add(ci),
                _mm256_sub_ps(_mm256_add_ps(g, gb), cv),
            );
            ci += 8;
        }
        // scalar remainder after the 8-lane gathers
        for j in ci..cols.len() {
            let st = cols[j] as usize;
            out[j] = x[st] + bias[st] - c;
        }
    }

    pub fn have_avx2_fma() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
}
