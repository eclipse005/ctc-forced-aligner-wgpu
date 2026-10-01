//! Opt-in phase timing for the CPU forward pass.
//!
//! `CTC_CPU_PROFILE=1` accumulates wall time per phase across every forward
//! and prints the table at the end of an alignment run (stderr).  With the
//! variable unset the [`prof!`] macro costs one relaxed atomic load per
//! phase and changes no arithmetic.
//!
//! The GPU tower has its own profiler (`CTC_PROFILE=1/2`); this one is the
//! CPU counterpart, kept deliberately coarser — the question it answers is
//! "which phase owns the chunk's wall time", not per-kernel microtiming.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Phase slots; keep in sync with the indices used in `forward.rs`.
///
/// The conv stack gets one slot per layer, addressed as `CONV0 + i` so a
/// checkpoint with a different depth does not need new constants (the model
/// loader asserts the stack is at most 7 layers deep).
pub(crate) const CONV0: usize = 0;
pub(crate) const FEAT_PROJ: usize = 7;
pub(crate) const POS_CONV: usize = 8;
pub(crate) const LAYER_NORM: usize = 9;
pub(crate) const QKV_GEMM: usize = 10;
pub(crate) const KT_TRANSPOSE: usize = 11;
pub(crate) const SCORES_GEMM: usize = 12;
pub(crate) const ATTN_SOFTMAX: usize = 13;
pub(crate) const PV_GEMM: usize = 14;
pub(crate) const OUT_PROJ: usize = 15;
pub(crate) const FF1_GEMM: usize = 16;
pub(crate) const GELU: usize = 17;
pub(crate) const FF2_GEMM: usize = 18;
pub(crate) const RESIDUAL_ADD: usize = 19;
pub(crate) const LM_HEAD: usize = 20;
pub(crate) const LOG_SOFTMAX: usize = 21;

pub(crate) const PHASES: [&str; 22] = [
    "conv0",
    "conv1",
    "conv2",
    "conv3",
    "conv4",
    "conv5",
    "conv6",
    "feat_proj",
    "pos_conv",
    "layer_norm",
    "qkv_gemm",
    "kt_transpose",
    "scores_gemm",
    "attn_softmax",
    "pv_gemm",
    "out_proj",
    "ff1_gemm",
    "gelu",
    "ff2_gemm",
    "residual_add",
    "lm_head",
    "log_softmax",
];

const N: usize = 22;

static NANOS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static FORWARDS: AtomicU64 = AtomicU64::new(0);

/// True when `CTC_CPU_PROFILE` selects phase profiling.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CTC_CPU_PROFILE")
            .map(|v| v == "1" || v == "2")
            .unwrap_or(false)
    })
}

pub(crate) fn add(idx: usize, d: Duration) {
    NANOS[idx].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    CALLS[idx].fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn forward_done() {
    FORWARDS.fetch_add(1, Ordering::Relaxed);
}

/// Print the accumulated table, sorted by time, to stderr.
pub(crate) fn dump(label: &str) {
    if !enabled() {
        return;
    }
    let fw = FORWARDS.swap(0, Ordering::Relaxed).max(1);
    let mut rows: Vec<(&str, f64, u64)> = (0..N)
        .map(|i| {
            (
                PHASES[i],
                NANOS[i].load(Ordering::Relaxed) as f64 / 1e9,
                CALLS[i].load(Ordering::Relaxed),
            )
        })
        .collect();
    for i in 0..N {
        NANOS[i].store(0, Ordering::Relaxed);
        CALLS[i].store(0, Ordering::Relaxed);
    }
    let total: f64 = rows.iter().map(|r| r.1).sum();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\n=== CPU phase profile: {label} — {fw} forward(s) ===");
    eprintln!("{:<16} {:>10} {:>7} {:>10}", "phase", "total_s", "pct", "ms/fwd");
    let mut accounted = 0.0;
    for (name, s, _calls) in rows {
        if s > 0.0 {
            eprintln!(
                "{name:<16} {s:>10.3} {:>6.1}% {:>10.2}",
                100.0 * s / total,
                1000.0 * s / fw as f64
            );
        }
        accounted += s;
    }
    eprintln!("{:<16} {accounted:>10.3}", "SUM");
    eprintln!();
}

/// Time `$b` and bank the result under phase `$i`.
macro_rules! prof {
    ($i:expr, $b:block) => {{
        if $crate::wav2vec2::prof::enabled() {
            let __start = std::time::Instant::now();
            $b;
            $crate::wav2vec2::prof::add($i, __start.elapsed());
        } else {
            $b;
        }
    }};
}
pub(crate) use prof;
