//! Apple Accelerate `cblas_sgemm` binding, used on aarch64 macOS in place of
//! the `gemm` crate for the CPU forward pass.
//!
//! Why: on Apple Silicon the CPU encoder spends ~66% of its wall time in the
//! five GEMM phases (see `CTC_CPU_PROFILE=1`), while everything the scalar and
//! NEON kernels touch — gelu, softmax, layer norm — is under 8% combined.
//! Optimising those elementwise passes cannot move the total, so the GEMMs are
//! routed to Accelerate instead, which drives the AMX matrix coprocessors.
//!
//! Only the row-major, non-conjugated, `f32` cases the encoder actually uses
//! are exposed here. `b_trans` selects between `CblasNoTrans` and `CblasTrans`
//! for the right-hand operand, which covers the `[out, in]` checkpoint layout
//! without a separate transpose pass.
//!
//! The call is left threaded: Accelerate parallelises internally via GCD, which
//! is faster than nesting it inside a rayon pool. Every call site routed here
//! runs on a single thread for that reason; `pos_conv` keeps the `gemm` crate
//! because it executes inside `par_iter`.

use std::os::raw::c_int;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

/// `c = alpha·a·b + beta·c`, all row-major.
///
/// `a` is `(m, k)` with row stride `lda`, `c` is `(m, n)` with row stride
/// `ldc`. When `b_trans` is set, `b` holds a `(n, k)` row-major operand and is
/// read as its transpose (`ldb = k`); otherwise it is `(k, n)` with `ldb = n`.
///
/// # Safety
/// All pointers must address buffers large enough for the shape given by
/// `m`/`n`/`k` and the strides, with no overlap between `c` and `a`/`b`.
#[inline]
pub unsafe fn sgemm_rm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b_trans: bool,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    // Every call site in this crate derives these from a few thousand at most;
    // the cast cannot truncate. Kept explicit rather than `as _` so a future
    // larger shape fails loudly instead of silently wrapping.
    debug_assert!(m <= i32::MAX as usize);
    debug_assert!(n <= i32::MAX as usize);
    debug_assert!(k <= i32::MAX as usize);
    if m == 0 || n == 0 {
        return;
    }
    cblas_sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        if b_trans { CBLAS_TRANS } else { CBLAS_NO_TRANS },
        m as c_int,
        n as c_int,
        k as c_int,
        alpha,
        a,
        lda as c_int,
        b,
        ldb as c_int,
        beta,
        c,
        ldc as c_int,
    );
}
