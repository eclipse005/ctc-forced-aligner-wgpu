//! Pin down the gemm crate's calling conventions with tiny numeric cases.
use gemm::{gemm, Parallelism};

/// naive: dst(m,n) = alpha*dst + beta*A(m,k)@B(k,n), all row-major
fn naive(dst: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
    let old = dst.to_vec();
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            dst[i * n + j] = alpha * old[i * n + j] + beta * acc;
        }
    }
}

fn check(name: &str, got: &[f32], want: &[f32]) {
    let max = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("{name}: max abs diff {max:.3e} {}", if max < 1e-5 { "OK" } else { "MISMATCH" });
}

fn main() {
    let (m, n, k) = (5usize, 4usize, 3usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37 - 2.0).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.53 + 1.0).cos()).collect();

    // 1. plain overwrite, row-major dst
    let mut got = vec![0.0f32; m * n];
    unsafe {
        gemm::<f32>(m, n, k, got.as_mut_ptr(), 1, n as isize, false,
            a.as_ptr(), 1, k as isize, b.as_ptr(), 1, n as isize,
            1.0, 1.0, false, false, false, Parallelism::None);
    }
    let mut want = vec![0.0f32; m * n];
    naive(&mut want, &a, &b, m, n, k, 0.0, 1.0);
    check("overwrite alpha=1 beta=1", &got, &want);

    // 2. accumulate into prefilled dst
    let bias: Vec<f32> = (0..n).map(|i| 0.5 + i as f32).collect();
    let mut got = vec![0.0f32; m * n];
    for r in got.chunks_mut(n) { r.copy_from_slice(&bias); }
    unsafe {
        gemm::<f32>(m, n, k, got.as_mut_ptr(), 1, n as isize, true,
            a.as_ptr(), 1, k as isize, b.as_ptr(), 1, n as isize,
            1.0, 1.0, false, false, false, Parallelism::None);
    }
    let mut want = vec![0.0f32; m * n];
    for r in want.chunks_mut(n) { r.copy_from_slice(&bias); }
    naive(&mut want, &a, &b, m, n, k, 1.0, 1.0);
    check("accumulate alpha=1 beta=1", &got, &want);

    // 3. strided dst: 4-column strip of a 9-wide row-major matrix
    let (wide, off) = (9usize, 3usize);
    let mut got = vec![0.0f32; m * wide];
    for r in got.chunks_mut(wide) { r[off..off + n].copy_from_slice(&bias); }
    unsafe {
        gemm::<f32>(m, n, k, got.as_mut_ptr().add(off), 1, wide as isize, true,
            a.as_ptr(), 1, k as isize, b.as_ptr(), 1, n as isize,
            1.0, 1.0, false, false, false, Parallelism::None);
    }
    let mut want = vec![0.0f32; m * wide];
    for (r, wr) in got.iter().zip(want.chunks_mut(wide)) {
        let _ = r;
        wr[off..off + n].copy_from_slice(&bias);
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k { acc += a[i * k + p] * b[p * n + j]; }
            want[i * wide + off + j] = bias[j] + acc;
        }
    }
    check("strided dst strip", &got, &want);

    // 4. pos-conv B pattern: B(ci,oc) = W[(oc*pg + ci)*kk + tap], kk taps
    let (pg, kk, tap) = (3usize, 4usize, 2usize);
    let w: Vec<f32> = (0..pg * pg * kk).map(|i| (i as f32 * 0.11 - 1.0).sin()).collect();
    let b_tap: Vec<f32> = (0..pg * pg).map(|ci_oc| {
        let ci = ci_oc / pg;
        let oc = ci_oc % pg;
        w[(oc * pg + ci) * kk + tap]
    }).collect();
    let a2: Vec<f32> = (0..m * pg).map(|i| (i as f32 * 0.29).cos()).collect();
    let mut got = vec![0.0f32; m * pg];
    unsafe {
        gemm::<f32>(m, pg, pg, got.as_mut_ptr(), 1, pg as isize, false,
            a2.as_ptr(), 1, pg as isize, w.as_ptr().add(tap), (pg * kk) as isize, kk as isize,
            1.0, 1.0, false, false, false, Parallelism::None);
    }
    let mut want = vec![0.0f32; m * pg];
    naive(&mut want, &a2, &b_tap, m, pg, pg, 0.0, 1.0);
    check("pos-conv B stride pattern", &got, &want);

    // 5. does read_dst=false really skip reading dst?
    let mut got = vec![5.0f32; m * n];
    unsafe {
        gemm::<f32>(m, n, k, got.as_mut_ptr(), 1, n as isize, false,
            a.as_ptr(), 1, k as isize, b.as_ptr(), 1, n as isize,
            1.0, 1.0, false, false, false, Parallelism::None);
    }
    let mut want = vec![0.0f32; m * n];
    naive(&mut want, &a, &b, m, n, k, 0.0, 1.0);
    check("overwrite dst=5 read_dst=false (skip read => OK)", &got, &want);

    // 6. alpha=0 forces a clean overwrite even if dst IS read
    let mut got = vec![5.0f32; m * n];
    unsafe {
        gemm::<f32>(m, n, k, got.as_mut_ptr(), 1, n as isize, false,
            a.as_ptr(), 1, k as isize, b.as_ptr(), 1, n as isize,
            0.0, 1.0, false, false, false, Parallelism::None);
    }
    check("overwrite dst=5 alpha=0 (=> OK)", &got, &want);

    // 7. fused vs split columns at realistic sizes (the qkv fusion shape)
    let (rows, kk2, out_full, out_one) = (786usize, 1024usize, 3072usize, 1024usize);
    let x7: Vec<f32> = (0..rows * kk2).map(|i| (i as f32 * 0.113).sin()).collect();
    let w7: Vec<f32> = (0..out_full * kk2).map(|i| (i as f32 * 0.017 - 3.0).sin()).collect();
    let b7: Vec<f32> = (0..out_full).map(|i| (i as f32 * 0.0031).cos()).collect();

    let mut y_full = vec![0.0f32; rows * out_full];
    for r in y_full.chunks_mut(out_full) { r.copy_from_slice(&b7); }
    unsafe {
        gemm::<f32>(rows, out_full, kk2, y_full.as_mut_ptr(), 1, out_full as isize, true,
            x7.as_ptr(), 1, kk2 as isize, w7.as_ptr(), kk2 as isize, 1,
            1.0, 1.0, false, false, false, Parallelism::Rayon(0));
    }
    let mut y_one = vec![0.0f32; rows * out_one];
    for r in y_one.chunks_mut(out_one) { r.copy_from_slice(&b7[..out_one]); }
    unsafe {
        gemm::<f32>(rows, out_one, kk2, y_one.as_mut_ptr(), 1, out_one as isize, true,
            x7.as_ptr(), 1, kk2 as isize, w7.as_ptr(), kk2 as isize, 1,
            1.0, 1.0, false, false, false, Parallelism::Rayon(0));
    }
    let mut mx = 0.0f32;
    for r in 0..rows {
        for c in 0..out_one {
            mx = mx.max((y_full[r * out_full + c] - y_one[r * out_one + c]).abs());
        }
    }
    println!("fused out=3072 vs split out=1024 (first block): max {mx:.3e} {}",
        if mx < 1e-4 { "OK" } else { "MISMATCH" });
}
