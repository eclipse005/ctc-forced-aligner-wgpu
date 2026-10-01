//! GEMM throughput on the exact shapes the CPU forward uses.
use gemm::{gemm, Parallelism};

fn bench(name: &str, m: usize, n: usize, k: usize, par: Parallelism) {
    let x: Vec<f32> = vec![0.1; m * k];
    let w: Vec<f32> = vec![0.2; n * k];
    let mut y = vec![0.0f32; m * n];
    // warmup
    unsafe {
        gemm::<f32>(m, n, k, y.as_mut_ptr(), 1, n as isize, false,
            x.as_ptr(), 1, k as isize, w.as_ptr(), k as isize, 1,
            1.0, 1.0, false, false, false, par);
    }
    let iters = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            gemm::<f32>(m, n, k, y.as_mut_ptr(), 1, n as isize, false,
                x.as_ptr(), 1, k as isize, w.as_ptr(), k as isize, 1,
                1.0, 1.0, false, false, false, par);
        }
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    let ms = dt * 1000.0;
    let gps = gflop / dt;
    println!("{name:<10} ({m}x{n}x{k}): {ms:8.2} ms  {gps:7.1} GFLOP/s");
}

fn main() {
    let m_scale: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let m = 1700 * m_scale;
    let shapes = [
        ("ff1", m, 4096usize, 1024usize),
        ("ff2", m, 1024, 4096),
        ("qkv", m, 3072, 1024),
        ("scores", m, m, 64),
        ("pv", m, 64, m),
    ];
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("rayon");
    let par = match mode {
        "none" => Parallelism::None,
        "r8" => Parallelism::Rayon(8),
        "r12" => Parallelism::Rayon(12),
        "r16" => Parallelism::Rayon(16),
        "r20" => Parallelism::Rayon(20),
        _ => Parallelism::Rayon(0),
    };
    println!("mode = {mode}, threads = {}", rayon::current_num_threads());
    for (name, m, n, k) in shapes {
        bench(name, m, n, k, par);
    }
}
