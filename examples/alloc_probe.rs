//! Allocation probe: loop the encoder's GEMM shapes and print the live-set
//! counter after each iteration, to pin down which layer of the stack
//! retains memory across calls (the 15m live set doubles at chunk
//! 1/2/4/8/16 — a power-of-two growing cache lives somewhere).
//!
//! `alloc_probe [iters]`

#[global_allocator]
static GLOBAL: ctc_forced_aligner_wgpu::alloc_stats::Stats =
    ctc_forced_aligner_wgpu::alloc_stats::Stats;

use gemm::{gemm, Parallelism};

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    // the five linear shapes, one full encoder pass worth of calls per cycle
    let shapes: [(&str, usize, usize, usize); 5] = [
        ("ff1", 1700, 4096, 1024),
        ("ff2", 1700, 1024, 4096),
        ("qkv", 1700, 3072, 1024),
        ("out", 1700, 1024, 1024),
        ("lm", 1700, 10288, 1024),
    ];
    // keep every scratch alive for the whole run so only callee allocations move
    let mut keeps: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = shapes
        .iter()
        .map(|(_, m, n, k)| (vec![0.02f32; m * k], vec![0.03f32; n * k], vec![0f32; m * n]))
        .collect();
    for i in 0..iters {
        let idx = i % shapes.len();
        let (name, m, n, k) = shapes[idx];
        let (x, w, y) = &mut keeps[idx];
        unsafe {
            gemm::<f32>(m, n, k, y.as_mut_ptr(), 1, n as isize, false,
                x.as_ptr(), 1, k as isize, w.as_ptr(), k as isize, 1,
                1.0, 1.0, false, false, false, Parallelism::Rayon(0));
        }
        let (live, peak) = ctc_forced_aligner_wgpu::alloc_stats::stats();
        println!("iter {i:>3} ({name}): live {live:>12}  peak {peak:>12}");
    }
}
