//! GEMM throughput on the exact shapes the CPU forward uses.
//!
//! `gemm_bench <mode> [m_scale] [iters] [sustained]`
//!
//! The `sustained` flag reports the mean of the last quarter of the run
//! instead of the whole thing.  A 5-iteration burst measures the boost clock
//! and a warm cache; the real forward is ~11 s of back-to-back AVX2 and can
//! settle well below it, which is the gap between the microbench ceiling and
//! what the forward pass actually reaches.

use gemm::{gemm, Parallelism};

fn bench(name: &str, m: usize, n: usize, k: usize, par: Parallelism, iters: usize, sustained: bool) {
    let x: Vec<f32> = vec![0.1; m * k];
    let w: Vec<f32> = vec![0.2; n * k];
    let mut y = vec![0.0f32; m * n];
    // warmup
    unsafe {
        gemm::<f32>(m, n, k, y.as_mut_ptr(), 1, n as isize, false,
            x.as_ptr(), 1, k as isize, w.as_ptr(), k as isize, 1,
            1.0, 1.0, false, false, false, par);
    }
    let mut each = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        unsafe {
            gemm::<f32>(m, n, k, y.as_mut_ptr(), 1, n as isize, false,
                x.as_ptr(), 1, k as isize, w.as_ptr(), k as isize, 1,
                1.0, 1.0, false, false, false, par);
        }
        each.push(t0.elapsed().as_secs_f64());
    }
    let from = if sustained { iters * 3 / 4 } else { 0 };
    let window = &each[from..];
    let dt = window.iter().sum::<f64>() / window.len() as f64;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    let ms = dt * 1000.0;
    let gps = gflop / dt;
    let first = each[0] * 1000.0;
    println!(
        "{name:<10} ({m}x{n}x{k}): {ms:8.2} ms  {gps:7.1} GFLOP/s   (first iter {first:.2} ms{})",
        if sustained { ", steady" } else { "" }
    );
}

/// Isolate the per-call weight-packing cost.  `gemm` 0.18 allocates and fills
/// its packing buffers inside every call and exposes no way to reuse them, so
/// the same (n, k) weight is re-packed on every forward — 1.27 GB per 34 s
/// chunk across the encoder.  Packing depends on (n, k), not on m, so timing
/// the same shape at a tiny m (compute negligible, packing unchanged) against
/// the real m splits the two apart.
fn pack_probe(name: &str, m: usize, n: usize, k: usize, iters: usize) -> f64 {
    let x: Vec<f32> = vec![0.1; m * k];
    let w: Vec<f32> = vec![0.2; n * k];
    let mut y = vec![0.0f32; m * n];
    let mut run = || unsafe {
        for _ in 0..iters {
            gemm::<f32>(m, n, k, y.as_mut_ptr(), 1, n as isize, false,
                x.as_ptr(), 1, k as isize, w.as_ptr(), k as isize, 1,
                1.0, 1.0, false, false, false, Parallelism::Rayon(0));
        }
    };
    run();
    let t0 = std::time::Instant::now();
    run();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    println!(
        "  {name:<12} m={m:<5} {ms:7.2} ms  {gflop:7.2} GFLOP total",
        gflop = gflop * iters as f64
    );
    ms
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("rayon");
    let m_scale: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let sustained = args
        .get(4)
        .map(|s| s == "sustained" || s == "1")
        .unwrap_or(false);
    let m = 1700 * m_scale;
    let shapes = [
        ("ff1", m, 4096usize, 1024usize),
        ("ff2", m, 1024, 4096),
        ("qkv", m, 3072, 1024),
        ("scores", m, m, 64),
        ("pv", m, 64, m),
    ];
    let par = match mode {
        "none" => Parallelism::None,
        "r8" => Parallelism::Rayon(8),
        "r12" => Parallelism::Rayon(12),
        "r16" => Parallelism::Rayon(16),
        "r20" => Parallelism::Rayon(20),
        _ => Parallelism::Rayon(0),
    };
    println!(
        "mode = {mode}, threads = {}, m = {m}, iters = {iters}{}",
        rayon::current_num_threads(),
        if sustained { ", steady-state window" } else { "" }
    );
    for (name, m, n, k) in shapes {
        bench(name, m, n, k, par, iters, sustained);
    }

    if mode == "pack" {
        println!("\nweight-packing probe (weight is re-packed on every call):");
        for (name, n, k) in [("ff1", 4096usize, 1024usize), ("qkv", 3072, 1024), ("proj", 1024, 1024)] {
            let big = pack_probe(name, 1700, n, k, 20);
            let tiny = pack_probe(name, 4, n, k, 20);
            println!(
                "  -> {name}: packing ~= {tiny:.2} ms of {big:.2} ms  ({:.0}%)",
                100.0 * tiny / big
            );
        }
    }
}
