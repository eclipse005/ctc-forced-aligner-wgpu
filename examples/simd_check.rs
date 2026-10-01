//! Sanity-check the AVX2 kernels against the scalar reference paths.
use ctc_forced_aligner_wgpu::simd::avx2;

fn fast_erf(x: f32) -> f32 {
    let ax = x.abs();
    if ax > 5.0 { return if x < 0.0 { -1.0 } else { 1.0 }; }
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t)
            * ((-ax * ax) * std::f32::consts::LOG2_E) .exp2();
    if x < 0.0 { -y } else { y }
}

fn main() {
    // gelu: avx2 vs scalar
    let vals: Vec<f32> = (-500..500).map(|i| i as f32 * 0.04).collect();
    let mut a = vals.clone();
    unsafe { avx2::gelu_inplace(&mut a) };
    let mut max = 0.0f32;
    for (i, v) in vals.iter().enumerate() {
        let want = 0.5 * v * (1.0 + fast_erf(v / std::f32::consts::SQRT_2));
        max = max.max((a[i] - want).abs());
    }
    println!("gelu  max abs diff vs scalar: {max:.3e}");

    // erf8 lane-by-lane vs scalar A&S
    {
        let e = std::arch::is_x86_feature_detected!("avx2");
        let _ = e;
    }
    {
        // scalar A&S reference
        fn erf_ref(x: f32) -> f32 {
            let ax = x.abs();
            if ax > 5.0 { return if x < 0.0 { -1.0 } else { 1.0 }; }
            let t = 1.0 / (1.0 + 0.327_591_1 * ax);
            let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t
                - 0.284_496_736) * t + 0.254_829_592) * t) * (-ax * ax).exp();
            if x < 0.0 { -y } else { y }
        }
        let xs: Vec<f32> = (-60..60).map(|i| i as f32 * 0.1).collect();
        let mut got = xs.clone();
        unsafe {
            // erf8 via gelu trick is not direct; call through a tiny wrapper:
            use std::arch::x86_64::*;
            if is_x86_feature_detected!("avx2") {
                let mut chunks = got.chunks_exact_mut(8);
                for c in chunks.by_ref() {
                    let v = _mm256_loadu_ps(c.as_ptr());
                    let e = avx2::erf8(v);
                    _mm256_storeu_ps(c.as_mut_ptr(), e);
                }
            }
        }
        let mut shown = 0;
        for (i, x) in xs.iter().enumerate() {
            let want = erf_ref(*x);
            let d = (got[i] - want).abs();
            if d > 1e-5 && shown < 8 {
                println!("erf({x}) got {} want {want} diff {d}", got[i]);
                shown += 1;
            }
        }
        if shown == 0 { println!("erf8: all within 1e-5"); }
    }

    // softmax row: avx2 vs scalar
    let row: Vec<f32> = (0..1024).map(|i| ((i * 7919) % 100) as f32 * 0.03 - 1.5).collect();
    let mut r1 = row.clone();
    let mut r2 = row.clone();
    unsafe { avx2::softmax_inplace(&mut r1) };
    {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in &row { sum += (x - m).exp(); }
        for (x, y) in r2.iter_mut().zip(&row) { *x = (y - m).exp() / sum; }
    }
    let mut d = 0.0f32;
    for (a, b) in r1.iter().zip(&r2) { d = d.max((a - b).abs()); }
    println!("softmax max abs diff vs std exp: {d:.3e}");

    // log_softmax row: avx2 vs scalar
    let mut l1 = row.clone();
    let mut l2 = row.clone();
    unsafe { avx2::log_softmax_inplace(&mut l1) };
    {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in &row { sum += (x - m).exp(); }
        let lsum = sum.ln();
        for (x, y) in l2.iter_mut().zip(&row) { *x = y - m - lsum; }
    }
    let mut d = 0.0f32;
    for (a, b) in l1.iter().zip(&l2) { d = d.max((a - b).abs()); }
    println!("log_softmax max abs diff vs scalar: {d:.3e}");
}
