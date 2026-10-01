//! Spike: can a hand-written packed AVX2 SGEMM beat the `gemm` crate's P-core
//! efficiency (65% of peak) on the shapes the encoder actually uses?
//!
//! **Verdict after this spike (2026-10-01): no, not without a real tuning
//! campaign.**  A first-pass 6x16 packed kernel — k-major panels, (m, nc)
//! task split — benchmarks honestly at ~25% *below* the crate at r20
//! (ff1 1094 vs 1420, qkv 1014 vs 1439, ff2 1238 vs 1374 GFLOP/s) and 2.4x
//! below on lm_head's wide-n shape (771 vs 1823).  The crate's kernel is
//! genuinely strong; beating it needs 2-3x k-unrolls, load pairing,
//! prefetch, tuned nc/mc blocking, and pre-packed weight reuse.  That is the
//! "independent project" the handover notes predicted — the entry barrier is
//! confirmed, so this example stays as the starting point, not integrated.
//!
//! scar tissue: the first version of this spike had a `t1 = per.min(mt - t0)`
//! range bug that silently skipped every block after the first — the bench
//! reported 1991 GFLOP/s doing 3% of the work.  The correctness check passed
//! because it ran serial.  Correctness gates must always cover the parallel
//! task decomposition.
//!
//! `sgemm_spike [r8|r20|none|all]` — benches ours vs `gemm` and cross-checks
//! values (tolerance level: accumulation order differs from the crate).

use std::arch::x86_64::*;

/// Raw-pointer wrapper so the packed panels can be shared across rayon tasks
/// (same pattern as the forward pass's `SendPtr`).
struct Raw<T>(*const T);
unsafe impl<T> Send for Raw<T> {}
unsafe impl<T> Sync for Raw<T> {}
impl<T> Raw<T> {
    fn get(&self) -> *const T {
        self.0
    }
}
struct RawMut<T>(*mut T);
unsafe impl<T> Send for RawMut<T> {}
unsafe impl<T> Sync for RawMut<T> {}
impl<T> RawMut<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// C (m, n) row-major = A (m, k) row-major * B^T where B is (n, k) row-major.
/// C is overwritten (the forward's `read_dst = false` semantics).
unsafe fn sgemm_packed(
    m: usize,
    n: usize,
    k: usize,
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    serial: bool,
) {
    const KC: usize = 512;
    let nt = n / 16; // bench shapes keep n a multiple of 16
    let mut bpack = vec![0.0f32; KC * 16 * nt];

    let mut k0 = 0;
    while k0 < k {
        let kc = KC.min(k - k0);
        // pack this kc slice of every n-tile: [kc][16] k-major
        for ti in 0..nt {
            let dst = bpack.as_mut_ptr().add(ti * kc * 16);
            for kk in 0..kc {
                for j in 0..16 {
                    *dst.add(kk * 16 + j) = *b.add((ti * 16 + j) * k + k0 + kk);
                }
            }
        }
        // m-tiles of 6 rows; each task owns a contiguous block of tiles and
        // streams every packed B panel from L3
        let mt = m.div_ceil(6);
        let blocks = if serial { 1 } else { (mt / 3).max(1).min(rayon::current_num_threads() * 2) };
        let per = mt.div_ceil(blocks);
        let a_r = Raw(a);
        let bpack_r = Raw(bpack.as_ptr());
        let c_r = RawMut(c);
        let block_run = |blk: usize| {
            // each block packs its own A panels — a shared scratch would race
            let mut apack = vec![0.0f32; kc * 6];
            let ap = apack.as_mut_ptr();
            let a = a_r.get();
            let bpack_ptr = bpack_r.get();
            let c = c_r.get();
            let t0 = blk * per;
            let t1 = (t0 + per).min(mt);
            for t in t0..t1 {
                let m0 = t * 6;
                let rows = 6.min(m - m0);
                // pack the A panel [kc][rows] once per m-tile, reads consecutive
                for i in 0..rows {
                    let src = a.add((m0 + i) * k + k0);
                    for kk in 0..kc {
                        *ap.add(kk * rows + i) = *src.add(kk);
                    }
                }
                for ti in 0..nt {
                    let bpanel = bpack_ptr.add(ti * kc * 16);
                    let cp = c.add(m0 * n + ti * 16);
                    match rows {
                        6 => kernel_rows::<6>(kc, ap, bpanel, cp, n),
                        5 => kernel_rows::<5>(kc, ap, bpanel, cp, n),
                        4 => kernel_rows::<4>(kc, ap, bpanel, cp, n),
                        3 => kernel_rows::<3>(kc, ap, bpanel, cp, n),
                        2 => kernel_rows::<2>(kc, ap, bpanel, cp, n),
                        _ => kernel_rows::<1>(kc, ap, bpanel, cp, n),
                    }
                }
            }
        };
        if serial {
            for blk in 0..blocks {
                block_run(blk);
            }
        } else {
            use rayon::prelude::*;
            (0..blocks).into_par_iter().for_each(block_run);
        }
        k0 += kc;
    }
}

/// `C tile (R x 16) += A panel (kc x R) * B panel (kc x 16)`, then store
/// (overwrite — accumulators start at zero and each (kc, tile) pair runs once).
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_rows<const R: usize>(kc: usize, a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    let z = || _mm256_setzero_ps();
    let (mut c00, mut c01, mut c10, mut c11, mut c20, mut c21) = (z(), z(), z(), z(), z(), z());
    let (mut c30, mut c31, mut c40, mut c41, mut c50, mut c51) = (z(), z(), z(), z(), z(), z());
    for kk in 0..kc {
        let b0 = _mm256_loadu_ps(b.add(kk * 16));
        let b1 = _mm256_loadu_ps(b.add(kk * 16 + 8));
        let ar = a.add(kk * R);
        let a0 = _mm256_broadcast_ss(&*ar);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        if R > 1 {
            let a1 = _mm256_broadcast_ss(&*ar.add(1));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
        }
        if R > 2 {
            let a2 = _mm256_broadcast_ss(&*ar.add(2));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
        }
        if R > 3 {
            let a3 = _mm256_broadcast_ss(&*ar.add(3));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
        }
        if R > 4 {
            let a4 = _mm256_broadcast_ss(&*ar.add(4));
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);
        }
        if R > 5 {
            let a5 = _mm256_broadcast_ss(&*ar.add(5));
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
        }
    }
    if 0 < R { store_row::<0, R>(c, n, c00, c01); }
    if 1 < R { store_row::<1, R>(c, n, c10, c11); }
    if 2 < R { store_row::<2, R>(c, n, c20, c21); }
    if 3 < R { store_row::<3, R>(c, n, c30, c31); }
    if 4 < R { store_row::<4, R>(c, n, c40, c41); }
    if 5 < R { store_row::<5, R>(c, n, c50, c51); }
}

#[inline]
unsafe fn store_row<const ROW: usize, const R: usize>(c: *mut f32, n: usize, lo: __m256, hi: __m256) {
    if ROW < R {
        let p = c.add(ROW * n);
        _mm256_storeu_ps(p, lo);
        _mm256_storeu_ps(p.add(8), hi);
    }
}

fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2/FMA required");
    }
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("all");
    let m = 1700usize;
    let shapes: [(&str, usize, usize); 4] = [
        ("ff1", 4096, 1024),
        ("qkv", 3072, 1024),
        ("ff2", 1024, 4096),
        ("lm_head", 10288, 1024),
    ];

    // correctness vs the gemm crate on a small odd shape — both serial and
    // parallel task paths (the t-range split was silently skipping blocks once)
    for serial in [true, false] {
        let (m, n, k) = (173usize, 2064usize, 257usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.11 - 0.37).collect();
        let mut c1 = vec![0f32; m * n];
        let mut c2 = vec![0f32; m * n];
        unsafe {
            gemm::gemm::<f32>(m, n, k, c1.as_mut_ptr(), 1, n as isize, false,
                a.as_ptr(), 1, k as isize, b.as_ptr(), k as isize, 1,
                1.0, 1.0, false, false, false, gemm::Parallelism::None);
            sgemm_packed(m, n, k, c2.as_mut_ptr(), a.as_ptr(), b.as_ptr(), serial);
        }
        let mx = c1.iter().zip(&c2).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        println!("correctness vs gemm (173x2064x257, serial={serial}): max abs diff {mx:.6}");
    }

    let par = match mode {
        "none" => Some(gemm::Parallelism::None),
        "r8" => Some(gemm::Parallelism::Rayon(8)),
        "r20" => Some(gemm::Parallelism::Rayon(20)),
        _ => None,
    };
    let serial = mode == "none";
    println!("threads = {}, mode = {mode}", rayon::current_num_threads());
    let iters = 6usize;
    for (name, n, k) in shapes {
        let a: Vec<f32> = vec![0.02; m * k];
        let b: Vec<f32> = vec![0.03; n * k];
        let mut c = vec![0f32; m * n];
        unsafe {
            sgemm_packed(m, n, k, c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), serial);
            if let Some(p) = par {
                gemm::gemm::<f32>(m, n, k, c.as_mut_ptr(), 1, n as isize, false,
                    a.as_ptr(), 1, k as isize, b.as_ptr(), k as isize, 1,
                    1.0, 1.0, false, false, false, p);
            }
        }
        let mut ours = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            unsafe { sgemm_packed(m, n, k, c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), serial) };
            ours.push(t.elapsed().as_secs_f64());
        }
        let dt = ours[3..].iter().sum::<f64>() / (iters - 3) as f64;
        let gf = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
        println!("{name:<8} ours  {:8.2} ms  {:7.1} GFLOP/s", dt * 1000.0, gf / dt);
        if let Some(p) = par {
            let mut theirs = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = std::time::Instant::now();
                unsafe {
                    gemm::gemm::<f32>(m, n, k, c.as_mut_ptr(), 1, n as isize, false,
                        a.as_ptr(), 1, k as isize, b.as_ptr(), k as isize, 1,
                        1.0, 1.0, false, false, false, p);
                }
                theirs.push(t.elapsed().as_secs_f64());
            }
            let dt2 = theirs[3..].iter().sum::<f64>() / (iters - 3) as f64;
            println!("{name:<8} gemm  {:8.2} ms  {:7.1} GFLOP/s", dt2 * 1000.0, gf / dt2);
        }
    }
}
