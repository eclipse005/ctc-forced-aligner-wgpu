//! CTC Viterbi on the GPU, f64 alphas, full rows.
//!
//! Alphas stay f64 because an hour-long path sits near −80000, where an f32
//! ulp is large enough to move a boundary. Devices without `SHADER_F64`
//! (Apple Silicon Metal, Intel Arc/iGPU) run the same DP on `df64` — a
//! `(hi, lo)` pair of f32s carrying 48 significand bits, laid out exactly like
//! an f64 so every buffer, checkpoint and backpointer stride is unchanged.
//! The choice table is kept only when it fits the Viterbi budget. Longer
//! files store one alpha row per segment and rebuild that segment while
//! tracing back.
//!
//! Mapped copies wait on their own submission index. `poll(wait_indefinitely)`
//! would also wait for the encoder already queued behind that copy. A slot is
//! reused `PIPE_DEPTH` submissions later.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::gpu::Gpu;
use crate::viterbi::{build_expanded_labels, dp_plan, CTC_STAR_SCORE};

/// Kept frames in one window. A 30 s window is 1500 frames at 50 Hz.
const BACK_CAP: usize = 2048;
/// Windows or replay chunks queued ahead of the CPU. Two was short enough
/// that recording the next command buffer drained the device.
pub(crate) const PIPE_DEPTH: usize = 4;
/// Visible uniform range. The slot stride is at least this, and a multiple of
/// the device's dynamic-offset alignment.
const UNI_SIZE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepCfg {
    s: u32,
    stride: u32,
    row: u32,
    mode: u32,
    back_row: u32,
    rb_u32: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherCfg {
    row0: u32,
    rows: u32,
    vocab: u32,
    s: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TraceCfg {
    lo: u32,
    hi: u32,
    t_len: u32,
    rb_u32: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

const _: () = assert!(std::mem::size_of::<StepCfg>() == 32);
const _: () = assert!(std::mem::size_of::<GatherCfg>() == 16);
const _: () = assert!(std::mem::size_of::<TraceCfg>() == 32);

/// Which alpha representation the DP runs in.
///
/// `F64` is `SHADER_F64` straight: 53 significand bits, and the only path that
/// existed before `df64`. `Df64` is a pair of f32s (`hi`, `lo`) summing to the
/// value — 48 significand bits, evaluable on any adapter, and the same 8 bytes
/// per state so no buffer or stride has to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DpPrecision {
    F64,
    Df64,
}

impl DpPrecision {
    /// The `(f64)` / `(df64)` tag the run log carries.
    fn tag(self) -> &'static str {
        match self {
            DpPrecision::F64 => "f64",
            DpPrecision::Df64 => "df64",
        }
    }
}

/// Resolve `CTC_GPU_DP_MODE` against the adapter's `SHADER_F64`.
///
/// `auto`/unset/empty follow the device; `f64` is a request that fails rather
/// than silently downgrading precision, because a caller that asked for it is
/// asking about the last ulp; `df64` works everywhere.
pub(crate) fn select_dp_precision(has_f64: bool, env_val: Option<&str>) -> Result<DpPrecision> {
    let mode = env_val.map(str::trim).unwrap_or_default();
    match mode.to_ascii_lowercase().as_str() {
        "" | "auto" => {
            if has_f64 {
                Ok(DpPrecision::F64)
            } else {
                Ok(DpPrecision::Df64)
            }
        }
        "f64" => {
            if has_f64 {
                Ok(DpPrecision::F64)
            } else {
                bail!("CTC_GPU_DP_MODE=f64 requested, but adapter lacks SHADER_F64")
            }
        }
        "df64" => Ok(DpPrecision::Df64),
        other => bail!("CTC_GPU_DP_MODE={other} is not one of auto, f64, df64"),
    }
}

/// The `f32` half of the finite unreachable sentinel. `-inf` must never reach
/// the two-sum below: `(-inf) - (-inf)` is a NaN, which would silently poison
/// every transition that flows through an unreachable state.
pub(crate) const DF_NINF_HI: f32 = -3.4e38;
/// Anything at or below this is unreachable. Well above `DF_NINF_HI`, well
/// below any real path score (an hour of audio is near −80,000).
const DF_NINF_CUT: f32 = -1.0e37;
const DF_NINF_CUT_F64: f64 = -1.0e37;

/// One f64 as a `df64` pair.
///
/// `hi` carries 24 bits and `lo` the next 24, so the pair is exact to about
/// 48 bits: at the −80,000 of an hour-long alpha that is an ulp below 1e-9.
/// Negative infinity (and anything past the sentinel cutoff) becomes the
/// finite sentinel the shader compares against.
pub(crate) fn f64_to_df64(x: f64) -> [f32; 2] {
    if (!x.is_finite() && x.is_sign_negative()) || x <= DF_NINF_CUT_F64 {
        return [DF_NINF_HI, 0.0];
    }
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    [hi, lo]
}

/// One `df64` pair back to f64, the inverse of [`f64_to_df64`] on the values
/// the DP produces.
pub(crate) fn df64_to_f64(hi: f32, lo: f32) -> f64 {
    if hi <= DF_NINF_CUT {
        return f64::NEG_INFINITY;
    }
    (hi as f64) + (lo as f64)
}

/// An alpha row as the 8 bytes per state the device buffer holds.
///
/// Both precisions are 8 bytes per state, which is what keeps the alpha
/// ping-pong, the checkpoint rows and `load_alpha` identical between them.
pub(crate) fn encode_alpha_bytes(precision: DpPrecision, alpha: &[f64]) -> Vec<u8> {
    let mut bytes = vec![0u8; alpha.len() * 8];
    for (i, v) in alpha.iter().enumerate() {
        let at = i * 8;
        match precision {
            DpPrecision::F64 => bytes[at..at + 8].copy_from_slice(&v.to_le_bytes()),
            DpPrecision::Df64 => {
                let [hi, lo] = f64_to_df64(*v);
                bytes[at..at + 4].copy_from_slice(&hi.to_le_bytes());
                bytes[at + 4..at + 8].copy_from_slice(&lo.to_le_bytes());
            }
        }
    }
    bytes
}

/// Decode what [`encode_alpha_bytes`] wrote. `raw` must be 8-byte values;
/// trailing bytes that do not fill one are ignored.
pub(crate) fn decode_alpha_bytes(precision: DpPrecision, raw: &[u8]) -> Vec<f64> {
    raw.as_chunks::<8>()
        .0
        .iter()
        .map(|c| match precision {
            DpPrecision::F64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                f64::from_le_bytes(b)
            }
            DpPrecision::Df64 => {
                let hi = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let lo = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                df64_to_f64(hi, lo)
            }
        })
        .collect()
}

const STEP_WGSL: &str = r#"
struct StepCfg {
    s: u32,
    stride: u32,
    row: u32,
    mode: u32,
    back_row: u32,
    rb_u32: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> prev_row: array<f64>;
@group(0) @binding(1) var<storage, read_write> next_row: array<f64>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> cols: array<u32>;
@group(0) @binding(4) var<storage, read> skip_dead: array<u32>;
@group(0) @binding(5) var<storage, read> star: array<u32>;
@group(0) @binding(6) var<storage, read_write> back: array<u32>;
@group(0) @binding(7) var<storage, read> ninf_buf: array<f64>;
@group(0) @binding(8) var<uniform> cfg: StepCfg;

// One workgroup covers 4096 states (256 threads x 16 states). A thread's 16
// states are STRIDED by the workgroup width, not contiguous: at loop step k
// the 32 threads of a warp touch 32 consecutive states, so every f64 row
// load is a coalesced 256 B. The row-major mapping this replaced made each
// thread own states [g*16, g*16+16), which read 16 cache lines per warp load
// — measured 120 us per frame on the DP, 16x the row's actual traffic.
var<workgroup> choices: array<u32, 4096>;

fn emit_at(st: u32) -> f32 {
    if (star[st] != 0u) {
        return -1.0;
    }
    return src[cfg.row * cfg.stride + cols[st]];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let base = wid.x * 4096u;
    let me = lid.x;
    let ninf = ninf_buf[0];
    for (var k = 0u; k < 16u; k = k + 1u) {
        let st = base + k * 256u + me;
        var choice = 0u;
        if (st < cfg.s) {
            let stay = prev_row[st];
            var adv = ninf;
            var skip_v = ninf;
            if (st >= 1u) {
                adv = prev_row[st - 1u];
            }
            if (st >= 2u && skip_dead[st] == 0u) {
                skip_v = prev_row[st - 2u];
            }
            var best = stay;
            if (stay >= adv && stay >= skip_v) {
                // stay: choice 0
            } else if (adv >= skip_v) {
                choice = 1u;
                best = adv;
            } else {
                choice = 2u;
                best = skip_v;
            }
            next_row[st] = best + f64(emit_at(st));
        }
        choices[k * 256u + me] = choice;
    }
    workgroupBarrier();
    // Pack the row's 2-bit choices: u32 u holds states [u*16, u*16+16), the
    // layout the CPU traceback (and the device trace kernel) read.
    let u = wid.x * 256u + me;
    if (u < cfg.rb_u32) {
        var packed = 0u;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let local = u * 16u + j - base;
            let c = choices[(local >> 8u) * 256u + (local & 255u)];
            packed = packed | ((c & 3u) << (j * 2u));
        }
        back[cfg.back_row * cfg.rb_u32 + u] = packed;
    }
}
"#;

const INIT_WGSL: &str = r#"
struct StepCfg {
    s: u32,
    stride: u32,
    row: u32,
    mode: u32,
    back_row: u32,
    rb_u32: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> next_row: array<f64>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> cols: array<u32>;
@group(0) @binding(3) var<storage, read> star: array<u32>;
@group(0) @binding(4) var<storage, read> ninf_buf: array<f64>;
@group(0) @binding(5) var<uniform> cfg: StepCfg;

fn emit_at(st: u32) -> f32 {
    if (star[st] != 0u) {
        return -1.0;
    }
    return src[cfg.row * cfg.stride + cols[st]];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let st = gid.x;
    if (st >= cfg.s) {
        return;
    }
    var v = ninf_buf[0];
    if (st <= 1u) {
        v = f64(emit_at(st));
    }
    next_row[st] = v;
}
"#;

/// The `df64` step kernel: the f64 one, with each alpha carried as a
/// `(hi, lo)` f32 pair and a FINITE unreachable sentinel.
///
/// `-inf` never enters the two-sum: `(-inf) - (-inf)` is a NaN, and one NaN
/// would silently corrupt every transition that flows through an unreachable
/// state — the damage would surface as a wrong boundary seconds away from the
/// state that produced it. `-3.4e38` is finite, far below any real path score,
/// and is tested by threshold rather than equality.
///
/// Tie-breaking is the f64 rule, `stay >= adv && stay >= skip`: two
/// unreachable states compare equal (`df_ge` is true for identical hi/lo), so
/// `stay` wins exactly as `-inf >= -inf` does. For normalized finite pairs the
/// lexicographic order on `(hi, lo)` is the exact order of the values.
const STEP_DF64_WGSL: &str = r#"
struct StepCfg {
    s: u32,
    stride: u32,
    row: u32,
    mode: u32,
    back_row: u32,
    rb_u32: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> prev_row: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> next_row: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> cols: array<u32>;
@group(0) @binding(4) var<storage, read> skip_dead: array<u32>;
@group(0) @binding(5) var<storage, read> star: array<u32>;
@group(0) @binding(6) var<storage, read_write> back: array<u32>;
@group(0) @binding(7) var<storage, read> ninf_buf: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> cfg: StepCfg;

// Same strided 4096-state workgroup as the f64 kernel: 16 KiB, and a warp's
// 32 threads still touch 32 consecutive states per step, so the 8-byte rows
// load as coalesced 256 B.
var<workgroup> choices: array<u32, 4096>;

fn emit_at(st: u32) -> f32 {
    if (star[st] != 0u) {
        return -1.0;
    }
    return src[cfg.row * cfg.stride + cols[st]];
}

fn df_is_ninf(a: vec2<f32>) -> bool {
    return a.x <= -1.0e37;
}

fn df_ge(a: vec2<f32>, b: vec2<f32>) -> bool {
    return (a.x > b.x) || (a.x == b.x && a.y >= b.y);
}

// A clamp whose bounds no value of this DP can reach: |alpha| stays near
// 80,000 and every residual is smaller still.
//
// It is not decoration. Vulkan implementations are allowed to simplify float
// expressions algebraically, and this machine's NVIDIA driver does: without
// the clamps it rewrites `(a.x + b) - a.x` to `b`, which makes every error
// term below zero and silently turns the pair back into plain f32 (measured:
// the alpha came out 3.7e-4 off the f64 run over 700 frames instead of 1e-8).
// `min`/`max` are exact here and opaque to that rewriting, so each partial
// result reaches the next step as the value IEEE arithmetic produced.
fn df_keep(x: f32) -> f32 {
    return min(max(x, -3.0e38), 3.0e38);
}

// Knuth TwoSum of (a.x, b), then a FastTwoSum renormalization of the pair
// with a's own residual. Any unreachable operand short-circuits to the
// sentinel before an infinity can reach the subtraction.
fn df_add_f32(a: vec2<f32>, b: f32) -> vec2<f32> {
    if (df_is_ninf(a) || b <= -1.0e37) {
        return vec2<f32>(-3.4e38, 0.0);
    }
    let s = df_keep(a.x + b);
    let v = df_keep(s - a.x);
    let p = df_keep(s - v);
    let e = df_keep((a.x - p) + (b - v));
    let lo = df_keep(a.y + e);
    let hi = df_keep(s + lo);
    let err = df_keep(lo - (hi - s));
    return vec2<f32>(hi, err);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let base = wid.x * 4096u;
    let me = lid.x;
    let ninf = ninf_buf[0];
    for (var k = 0u; k < 16u; k = k + 1u) {
        let st = base + k * 256u + me;
        var choice = 0u;
        if (st < cfg.s) {
            let stay = prev_row[st];
            var adv = ninf;
            var skip_v = ninf;
            if (st >= 1u) {
                adv = prev_row[st - 1u];
            }
            if (st >= 2u && skip_dead[st] == 0u) {
                skip_v = prev_row[st - 2u];
            }
            var best = stay;
            if (df_ge(stay, adv) && df_ge(stay, skip_v)) {
                // stay: choice 0
            } else if (df_ge(adv, skip_v)) {
                choice = 1u;
                best = adv;
            } else {
                choice = 2u;
                best = skip_v;
            }
            next_row[st] = df_add_f32(best, emit_at(st));
        }
        choices[k * 256u + me] = choice;
    }
    workgroupBarrier();
    let u = wid.x * 256u + me;
    if (u < cfg.rb_u32) {
        var packed = 0u;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let local = u * 16u + j - base;
            let c = choices[(local >> 8u) * 256u + (local & 255u)];
            packed = packed | ((c & 3u) << (j * 2u));
        }
        back[cfg.back_row * cfg.rb_u32 + u] = packed;
    }
}
"#;

/// The `df64` first-frame kernel: [`INIT_WGSL`] over `vec2<f32>` rows.
const INIT_DF64_WGSL: &str = r#"
struct StepCfg {
    s: u32,
    stride: u32,
    row: u32,
    mode: u32,
    back_row: u32,
    rb_u32: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> next_row: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> cols: array<u32>;
@group(0) @binding(3) var<storage, read> star: array<u32>;
@group(0) @binding(4) var<storage, read> ninf_buf: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> cfg: StepCfg;

fn emit_at(st: u32) -> f32 {
    if (star[st] != 0u) {
        return -1.0;
    }
    return src[cfg.row * cfg.stride + cols[st]];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let st = gid.x;
    if (st >= cfg.s) {
        return;
    }
    var v = ninf_buf[0];
    if (st <= 1u) {
        let e = emit_at(st);
        if (e > -1.0e37) {
            v = vec2<f32>(e, 0.0);
        }
    }
    next_row[st] = v;
}
"#;

const GATHER_WGSL: &str = r#"
struct Cfg { row0: u32, rows: u32, vocab: u32, s: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> states: array<u32>;
@group(0) @binding(2) var<storage, read> cols: array<u32>;
@group(0) @binding(3) var<storage, read> star: array<u32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= cfg.rows) {
        return;
    }
    let g = cfg.row0 + local;
    let st = states[g];
    var v = -1.0;
    if (st < cfg.s && star[st] == 0u) {
        v = logits[local * cfg.vocab + cols[st]];
    }
    outp[g] = v;
}
"#;

const TRACE_WGSL: &str = r#"
struct TraceCfg {
    lo: u32,
    hi: u32,
    t_len: u32,
    rb_u32: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}
@group(0) @binding(0) var<storage, read> back: array<u32>;
@group(0) @binding(1) var<storage, read_write> states: array<u32>;
@group(0) @binding(2) var<uniform> cfg: TraceCfg;

// One thread walks the segment's packed choices backwards, exactly the
// arithmetic of the CPU rewind: cur -= choice(row = t - lo, cur), states[t-1]
// = cur. Serial by nature — every step reads the step before it — so a single
// invocation is the whole point: the choices never leave the device. The walk
// starts from states[hi], which the segment after this one wrote (the file's
// last state is seeded before the first segment runs), so segments chain on
// the device and the CPU never waits per segment.
@compute @workgroup_size(1)
fn main() {
    var cur = states[cfg.hi];
    var t = cfg.hi;
    loop {
        if (t < cfg.lo) {
            break;
        }
        let row = t - cfg.lo;
        let v = (back[row * cfg.rb_u32 + (cur >> 4u)] >> ((cur & 15u) * 2u)) & 3u;
        cur = cur - min(cur, v);
        states[t - 1u] = cur;
        t = t - 1u;
    }
}
"#;

/// Device state for one file.
///
/// The choice table is Θ(frames × states). An hour is about 5.5 GB, and
/// writing that to disk only moves the same bytes. The forward pass keeps
/// the table only when it fits the fixed Viterbi budget. Past that it keeps
/// one alpha row per segment — a few tens of megabytes — and the traceback
/// rebuilds a single segment of choices. Encoder rows stay on the host
/// because a second encode would cost more than the rows do.
pub(crate) struct GpuDp {
    alpha: [wgpu::Buffer; 2],
    back: wgpu::Buffer,
    cfg: wgpu::Buffer,
    ninf: wgpu::Buffer,
    cols: wgpu::Buffer,
    skip: wgpu::Buffer,
    star: wgpu::Buffer,
    /// One window of encoder rows. Replay uploads a slice here; the full
    /// audio never gets a device copy.
    stage: wgpu::Buffer,
    states_buf: wgpu::Buffer,
    score_out: wgpu::Buffer,
    init_pipe: wgpu::ComputePipeline,
    step_pipe: wgpu::ComputePipeline,
    gather_pipe: wgpu::ComputePipeline,
    trace_pipe: wgpu::ComputePipeline,
    init_layout: wgpu::BindGroupLayout,
    step_layout: wgpu::BindGroupLayout,
    gather_layout: wgpu::BindGroupLayout,
    trace_layout: wgpu::BindGroupLayout,
    trace_uni: wgpu::Buffer,
    front: usize,
    frames: usize,
    n_frames: usize,
    rb: usize,
    rb_u32: u32,
    s: u32,
    hidden: usize,
    slot: u64,
    /// Windows whose DP has been submitted. The slot for the next one is
    /// `windows % PIPE_DEPTH`.
    windows: usize,
    collected: usize,
    slots: [MapSlot; PIPE_DEPTH],
    hidden_host: Vec<f32>,
    backs: BackStore,
    /// True when the choice table does not fit the budget. The forward then
    /// stores alpha checkpoints and the traceback rebuilds one segment.
    linear: bool,
    seg: usize,
    ckpt: wgpu::Buffer,
    ckpt_host: Vec<f64>,
    /// `f64` on an adapter with `SHADER_F64`, the `vec2<f32>` pair otherwise.
    /// Both are 8 bytes per state, so only the kernels and the host-side
    /// encode/decode change.
    precision: DpPrecision,
}

impl GpuDp {
    pub(crate) fn new(
        gpu: &Gpu,
        token_ids: &[usize],
        star_id: usize,
        blank_id: usize,
        n_frames: usize,
        hidden: usize,
    ) -> Result<Self> {
        let precision = select_dp_precision(
            gpu.has_f64(),
            std::env::var("CTC_GPU_DP_MODE").ok().as_deref(),
        )?;
        Self::new_with_precision(
            gpu, token_ids, star_id, blank_id, n_frames, hidden, precision,
        )
    }

    /// [`GpuDp::new`] with the precision chosen by the caller instead of by
    /// `CTC_GPU_DP_MODE`. The tests run both precisions on one device.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_precision(
        gpu: &Gpu,
        token_ids: &[usize],
        star_id: usize,
        blank_id: usize,
        n_frames: usize,
        hidden: usize,
        precision: DpPrecision,
    ) -> Result<Self> {
        anyhow::ensure!(
            CTC_STAR_SCORE == -1.0,
            "gpu viterbi shader hardcodes the star score -1.0"
        );
        anyhow::ensure!(n_frames >= 1, "gpu viterbi: no frames");
        anyhow::ensure!(hidden >= 1, "gpu viterbi: empty hidden");
        let labels = build_expanded_labels(token_ids, blank_id);
        let s = labels.len();
        anyhow::ensure!(s >= 1, "gpu viterbi: no states");
        let s_u = u32::try_from(s).context("state count")?;
        let rb = s.div_ceil(4);
        let rb_u32 = u32::try_from(s.div_ceil(16)).context("backpointer groups")?;

        let mut cols = vec![0u32; s];
        let mut skip = vec![0u32; s];
        let mut star = vec![0u32; s];
        let blank_col = u32::try_from(blank_id).context("blank id")?;
        for (st, &lab) in labels.iter().enumerate() {
            cols[st] = if lab == star_id {
                blank_col
            } else {
                u32::try_from(lab).context("label id")?
            };
        }
        for st in 2..s {
            if labels[st] == blank_id || labels[st] == labels[st - 2] {
                skip[st] = 1;
            }
        }
        if star_id != usize::MAX {
            for (i, &tok) in token_ids.iter().enumerate() {
                if tok == star_id {
                    star[2 * i + 1] = 1;
                }
            }
        }

        let alpha = [
            gpu.storage("dp-alpha-0", (s * 8) as u64),
            gpu.storage("dp-alpha-1", (s * 8) as u64),
        ];
        fill_ninf(gpu, &alpha[0], s, precision);
        fill_ninf(gpu, &alpha[1], s, precision);
        let ninf = gpu.storage("dp-ninf", 8);
        // The kernels read their unreachable value from here: `-inf` for f64,
        // the finite sentinel for df64.
        gpu.upload(&ninf, &encode_alpha_bytes(precision, &[f64::NEG_INFINITY]));
        let cols_buf = gpu.storage("dp-cols", (s * 4) as u64);
        let skip_buf = gpu.storage("dp-skip", (s * 4) as u64);
        let star_buf = gpu.storage("dp-star", (s * 4) as u64);
        gpu.upload(&cols_buf, bytemuck::cast_slice(&cols));
        gpu.upload(&skip_buf, bytemuck::cast_slice(&skip));
        gpu.upload(&star_buf, bytemuck::cast_slice(&star));

        let slot = (gpu.device.limits().min_uniform_buffer_offset_alignment as u64).max(UNI_SIZE);
        let cfg = gpu.uniform("dp-cfg", slot * BACK_CAP as u64);
        let (seg, linear) = dp_plan(n_frames, s, rb);
        // Non-linear: the whole table is read back per window, so the device
        // buffer holds one window. Linear: the choices are traced back ON the
        // device one segment at a time, so it holds one segment instead —
        // bounded by the same budget that picked `seg`.
        let back_rows = if linear { seg } else { BACK_CAP };
        let back_bytes = (back_rows as u64)
            .checked_mul(rb_u32 as u64)
            .and_then(|n| n.checked_mul(4))
            .context("back buffer size")?;
        let back = gpu.storage("dp-back", back_bytes);
        let stage_bytes = (BACK_CAP as u64)
            .checked_mul(hidden as u64)
            .and_then(|n| n.checked_mul(4))
            .context("hidden stage size")?;
        let stage = gpu.storage("dp-stage", stage_bytes);
        // The per-window readback maps hold one window of hidden rows and one
        // window of choice rows. The linear plan traces choices back on the
        // device instead, so its slots only ever carry the hidden rows.
        let slot_back_bytes = if linear { 16 } else { back_bytes };
        let slots = [
            MapSlot::new(gpu, 0, stage_bytes, slot_back_bytes)?,
            MapSlot::new(gpu, 1, stage_bytes, slot_back_bytes)?,
            MapSlot::new(gpu, 2, stage_bytes, slot_back_bytes)?,
            MapSlot::new(gpu, 3, stage_bytes, slot_back_bytes)?,
        ];
        let states_buf = gpu.storage("dp-states", (n_frames * 4) as u64);
        let score_out = gpu.storage("dp-scores", (n_frames * 4) as u64);

        let init_layout = bind_layout(
            gpu,
            "viterbi-init",
            &[
                storage_layout(0, false),
                storage_layout(1, true),
                storage_layout(2, true),
                storage_layout(3, true),
                storage_layout(4, true),
                uniform_layout(5),
            ],
        );
        let step_layout = bind_layout(
            gpu,
            "viterbi-step",
            &[
                storage_layout(0, true),
                storage_layout(1, false),
                storage_layout(2, true),
                storage_layout(3, true),
                storage_layout(4, true),
                storage_layout(5, true),
                storage_layout(6, false),
                storage_layout(7, true),
                uniform_layout(8),
            ],
        );
        let gather_layout = bind_layout(
            gpu,
            "viterbi-gather",
            &[
                storage_layout(0, true),
                storage_layout(1, true),
                storage_layout(2, true),
                storage_layout(3, true),
                storage_layout(4, false),
                uniform_static(5, 16),
            ],
        );
        let trace_layout = bind_layout(
            gpu,
            "viterbi-trace",
            &[
                storage_layout(0, true),
                storage_layout(1, false),
                uniform_static(2, 32),
            ],
        );
        let (init_src, step_src) = match precision {
            DpPrecision::F64 => (INIT_WGSL, STEP_WGSL),
            DpPrecision::Df64 => (INIT_DF64_WGSL, STEP_DF64_WGSL),
        };
        let init_pipe = pipe(gpu, "viterbi-init", init_src, &init_layout)?;
        let step_pipe = pipe(gpu, "viterbi-step", step_src, &step_layout)?;
        let gather_pipe = pipe(gpu, "viterbi-gather", GATHER_WGSL, &gather_layout)?;
        let trace_pipe = pipe(gpu, "viterbi-trace", TRACE_WGSL, &trace_layout)?;
        // Fold the DP's four pipelines into the same cache file the encoder
        // wrote. A warm start then compiles them too; an unwritable cache
        // directory must not fail the DP, so the result is dropped.
        let _ = gpu.save_pipeline_cache();
        let trace_uni = gpu.uniform("dp-trace-uni", 32);

        let ncheck = if linear { (n_frames - 1) / seg + 1 } else { 0 };
        let ckpt_bytes = (ncheck as u64)
            .saturating_mul(s as u64)
            .saturating_mul(8)
            .max(16);
        let ckpt = gpu.storage("dp-ckpt", ckpt_bytes);
        // The table fits: keep it in RAM. It does not: keep nothing of it.
        // A zero ceiling makes every push spill, and the linear path never pushes.
        let backs = BackStore::new(rb, if linear { 0 } else { usize::MAX });
        let back_mb = (n_frames.saturating_sub(1) as f64) * (rb as f64) / (1024.0 * 1024.0);
        let hid_mb = (n_frames as f64) * (hidden as f64) * 4.0 / (1024.0 * 1024.0);
        if linear {
            let ck_mb = (ncheck * s * 8) as f64 / (1024.0 * 1024.0);
            let seg_mb = (seg as f64) * (rb as f64) / (1024.0 * 1024.0);
            let prec = precision.tag();
            eprintln!(
                "ctc-aligner: viterbi on gpu ({prec}), checkpoints {ck_mb:.0} MB every {seg} frames, one segment {seg_mb:.0} MB rebuilt on traceback, hidden grows to {hid_mb:.0} MB (full table would be {back_mb:.0} MB)"
            );
        } else {
            let prec = precision.tag();
            eprintln!(
                "ctc-aligner: viterbi on gpu ({prec}), one pass, backpointers {back_mb:.0} MB, hidden grows to {hid_mb:.0} MB"
            );
        }

        Ok(Self {
            alpha,
            back,
            cfg,
            ninf,
            cols: cols_buf,
            skip: skip_buf,
            star: star_buf,
            stage,
            states_buf,
            score_out,
            init_pipe,
            step_pipe,
            gather_pipe,
            trace_pipe,
            init_layout,
            step_layout,
            gather_layout,
            trace_layout,
            trace_uni,
            front: 0,
            frames: 0,
            n_frames,
            rb,
            rb_u32,
            s: s_u,
            hidden,
            slot,
            windows: 0,
            collected: 0,
            slots,
            hidden_host: Vec::new(),
            backs,
            linear,
            seg,
            ckpt,
            ckpt_host: Vec::new(),
            precision,
        })
    }

    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    pub(crate) fn stage(&self) -> &wgpu::Buffer {
        &self.stage
    }

    pub(crate) fn chunk_limit(&self) -> usize {
        BACK_CAP
    }

    pub(crate) fn hidden_rows(&self, frame: usize, n: usize) -> Result<&[f32]> {
        let start = frame.checked_mul(self.hidden).context("hidden index")?;
        let end = start
            .checked_add(n.checked_mul(self.hidden).context("hidden len")?)
            .context("hidden end")?;
        self.hidden_host.get(start..end).with_context(|| {
            format!(
                "hidden archive has {} rows, need {frame}+{n}",
                self.hidden_host.len() / self.hidden.max(1)
            )
        })
    }

    /// DP submits whose rows are back in RAM. One encoder window may be more
    /// than one submit when it exceeds the per-submit frame cap.
    pub(crate) fn collected(&self) -> usize {
        self.collected
    }

    /// DP submits issued so far.
    pub(crate) fn submitted(&self) -> usize {
        self.windows
    }

    /// How many traceback steps this file's plan has: one per segment it
    /// rebuilds on the linear-space plan, one for the single pass. The run's
    /// progress denominator needs this before the traceback starts, so it
    /// reads `n_frames` — the plan's own answer — and not `frames`, which is
    /// the running count of frames *submitted* and is still zero here.
    pub(crate) fn traceback_segments(&self) -> usize {
        if self.linear {
            (self.n_frames.saturating_sub(1)) / self.seg.max(1) + 1
        } else {
            1
        }
    }

    pub(crate) fn check_store(&self) -> Result<()> {
        anyhow::ensure!(
            self.hidden_host.len() == self.frames * self.hidden,
            "hidden rows {} != {} frames",
            self.hidden_host.len() / self.hidden.max(1),
            self.frames
        );
        if self.linear {
            let ncheck = (self.frames - 1) / self.seg + 1;
            anyhow::ensure!(
                self.ckpt_host.len() == ncheck * self.s as usize,
                "checkpoints {} != {ncheck}",
                self.ckpt_host.len() / (self.s as usize).max(1)
            );
        } else {
            anyhow::ensure!(
                self.backs.rows() == self.frames.saturating_sub(1),
                "backpointer rows {} != {}",
                self.backs.rows(),
                self.frames.saturating_sub(1)
            );
        }
        anyhow::ensure!(
            self.slots.iter().all(|s| s.flight.is_none()),
            "a viterbi window is still mapped"
        );
        Ok(())
    }

    pub(crate) fn states_buf(&self) -> &wgpu::Buffer {
        &self.states_buf
    }

    pub(crate) fn score_buf(&self) -> &wgpu::Buffer {
        &self.score_out
    }

    pub(crate) fn gather_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.gather_pipe
    }

    /// Bytes of one gather-uniform slot (`row0, rows, vocab, s`).
    pub(crate) fn gather_cfg(&self, row0: u32, rows: u32, vocab: u32) -> [u8; 16] {
        let cfg = GatherCfg {
            row0,
            rows,
            vocab,
            s: self.s,
        };
        let mut out = [0u8; 16];
        out.copy_from_slice(bytemuck::bytes_of(&cfg));
        out
    }

    pub(crate) fn gather_bind_group(
        &self,
        gpu: &Gpu,
        logits: &wgpu::Buffer,
        uni: &wgpu::Buffer,
        uni_off: u64,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-gather"),
            layout: &self.gather_layout,
            entries: &[
                buf_entry(0, logits),
                buf_entry(1, &self.states_buf),
                buf_entry(2, &self.cols),
                buf_entry(3, &self.star),
                buf_entry(4, &self.score_out),
                uni_entry(5, uni, uni_off, 16),
            ],
        })
    }

    /// Free the mapped slot this window will reuse. With four slots the wait
    /// is for window N-4, so the device still has the later windows queued.
    pub(crate) fn prepare_slot(&mut self, gpu: &Gpu) -> Result<()> {
        let slot = self.windows % PIPE_DEPTH;
        if self.slots[slot].flight.is_some() {
            self.collect_slot(gpu, slot)?;
        }
        Ok(())
    }

    /// Wait out the windows still mapped. The newest submit is the last DP,
    /// so this is the end of the encode, not a drain between windows.
    pub(crate) fn finish(&mut self, gpu: &Gpu) -> Result<()> {
        let n = self.windows;
        let start = n.saturating_sub(PIPE_DEPTH);
        for i in start..n {
            self.collect_slot(gpu, i % PIPE_DEPTH)?;
        }
        if self.linear {
            let ncheck = (self.frames - 1) / self.seg + 1;
            let bytes = (ncheck * self.s as usize * 8) as u64;
            let raw = gpu.readback(&self.ckpt, bytes)?;
            anyhow::ensure!(raw.len() == bytes as usize, "checkpoint readback short");
            self.ckpt_host = decode_alpha_bytes(self.precision, &raw);
        }
        let (ram, disk) = self.backs.megabytes();
        let hid = (self.hidden_host.len() * 4) as f64 / (1024.0 * 1024.0);
        let ck = (self.ckpt_host.len() * 8) as f64 / (1024.0 * 1024.0);
        if self.linear {
            eprintln!(
                "ctc-aligner: viterbi checkpoints {ck:.0} MB, hidden {hid:.0} MB, choices not stored"
            );
        } else if disk > 0.0 {
            eprintln!(
                "ctc-aligner: viterbi kept {ram:.0} MB backpointers in RAM, {disk:.0} MB spilled, hidden {hid:.0} MB"
            );
        } else {
            eprintln!(
                "ctc-aligner: viterbi kept {ram:.0} MB backpointers in RAM, hidden {hid:.0} MB"
            );
        }
        Ok(())
    }

    /// Record this window's DP and its readback copies, and submit them behind
    /// the encoder that just wrote `logits` and `out_x`. Does not wait.
    pub(crate) fn submit_window(
        &mut self,
        gpu: &Gpu,
        logits: &wgpu::Buffer,
        out_x: &wgpu::Buffer,
        vocab: usize,
        row0: usize,
        n_kept: usize,
    ) -> Result<()> {
        anyhow::ensure!(n_kept >= 1, "empty viterbi window");
        anyhow::ensure!(
            n_kept <= BACK_CAP,
            "viterbi submit has {n_kept} frames; the per-submit cap is {BACK_CAP}"
        );
        let slot_i = self.windows % PIPE_DEPTH;
        anyhow::ensure!(
            self.slots[slot_i].flight.is_none(),
            "viterbi slot {slot_i} is still mapped"
        );
        let prof = dp_prof::enabled();
        let t_rec = prof.then(std::time::Instant::now);
        let stride = u32::try_from(vocab).context("vocab")?;
        let mut kinds = Vec::with_capacity(n_kept);
        let mut cfgs = Vec::with_capacity(n_kept);
        let mut back_row = 0u32;
        for local in 0..n_kept {
            let row = u32::try_from(row0 + local).context("logit row")?;
            let init = self.frames == 0 && local == 0;
            kinds.push(!init);
            cfgs.push(StepCfg {
                s: self.s,
                stride,
                row,
                mode: 0,
                back_row: if init { 0 } else { back_row },
                rb_u32: self.rb_u32,
                pad0: 0,
                pad1: 0,
            });
            if !init {
                back_row += 1;
            }
        }
        let produced = back_row as usize;
        let uni = self.slot as usize;
        let mut raw = vec![0u8; n_kept * uni];
        for (i, cfg) in cfgs.iter().enumerate() {
            let off = i * uni;
            let b = bytemuck::bytes_of(cfg);
            raw[off..off + b.len()].copy_from_slice(b);
        }
        gpu.queue.write_buffer(&self.cfg, 0, &raw);

        let bg_init = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-init"),
            layout: &self.init_layout,
            entries: &[
                buf_entry(0, &self.alpha[0]),
                buf_entry(1, logits),
                buf_entry(2, &self.cols),
                buf_entry(3, &self.star),
                buf_entry(4, &self.ninf),
                dyn_uni(5, &self.cfg),
            ],
        });
        let bg01 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-step-0"),
            layout: &self.step_layout,
            entries: &[
                buf_entry(0, &self.alpha[0]),
                buf_entry(1, &self.alpha[1]),
                buf_entry(2, logits),
                buf_entry(3, &self.cols),
                buf_entry(4, &self.skip),
                buf_entry(5, &self.star),
                buf_entry(6, &self.back),
                buf_entry(7, &self.ninf),
                dyn_uni(8, &self.cfg),
            ],
        });
        let bg10 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-step-1"),
            layout: &self.step_layout,
            entries: &[
                buf_entry(0, &self.alpha[1]),
                buf_entry(1, &self.alpha[0]),
                buf_entry(2, logits),
                buf_entry(3, &self.cols),
                buf_entry(4, &self.skip),
                buf_entry(5, &self.star),
                buf_entry(6, &self.back),
                buf_entry(7, &self.ninf),
                dyn_uni(8, &self.cfg),
            ],
        });

        let hidden_bytes = n_kept * self.hidden * 4;
        let hidden_src = (row0 * self.hidden * 4) as u64;
        let stride_b = self.rb_u32 as usize * 4;
        // The linear plan already ran the DP for its alpha. The choices are
        // rebuilt one segment at a time, so they are not copied off the device.
        let back_bytes = if self.linear { 0 } else { produced * stride_b };

        let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viterbi"),
            });
        enc.copy_buffer_to_buffer(
            out_x,
            hidden_src,
            &self.slots[slot_i].hidden,
            0,
            hidden_bytes as u64,
        );
        let mut front = self.front;
        let step_groups = self.s.div_ceil(16).div_ceil(256).max(1);
        let init_groups = self.s.div_ceil(256).max(1);
        // One compute pass for the window's frames; dispatches inside a pass
        // are ordered with full memory visibility, so the alpha ping-pong is
        // exact without a pass boundary per frame. Checkpoint copies are
        // encoder-level commands and split the pass where they fall — at most
        // one per window for any real segment length.
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("viterbi-frames"),
            timestamp_writes: None,
        });
        for (i, &is_step) in kinds.iter().enumerate() {
            let off = (i * uni) as u32;
            if is_step {
                pass.set_pipeline(&self.step_pipe);
                let bg = if front == 0 { &bg01 } else { &bg10 };
                pass.set_bind_group(0, bg, &[off]);
                pass.dispatch_workgroups(step_groups, 1, 1);
                front ^= 1;
            } else {
                pass.set_pipeline(&self.init_pipe);
                pass.set_bind_group(0, &bg_init, &[off]);
                pass.dispatch_workgroups(init_groups, 1, 1);
                front = 0;
            }
            let global = self.frames + i;
            if self.linear && global.is_multiple_of(self.seg) && i + 1 < n_kept {
                drop(pass);
                let row = self.s as u64 * 8;
                let k = (global / self.seg) as u64;
                enc.copy_buffer_to_buffer(&self.alpha[front], 0, &self.ckpt, k * row, row);
                pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viterbi-frames"),
                    timestamp_writes: None,
                });
            }
        }
        drop(pass);
        if self.linear {
            let last_global = self.frames + n_kept - 1;
            if last_global.is_multiple_of(self.seg) {
                let row = self.s as u64 * 8;
                let k = (last_global / self.seg) as u64;
                enc.copy_buffer_to_buffer(&self.alpha[front], 0, &self.ckpt, k * row, row);
            }
        }
        if back_bytes > 0 {
            enc.copy_buffer_to_buffer(
                &self.back,
                0,
                &self.slots[slot_i].back,
                0,
                back_bytes as u64,
            );
        }
        let index = gpu.queue.submit([enc.finish()]);
        drop((bg_init, bg01, bg10));
        if let Some(t) = t_rec {
            dp_prof::add(&dp_prof::RECORD_US, t.elapsed());
        }
        let t_sub = prof.then(std::time::Instant::now);
        // Validation is reported on a non-blocking poll. Waiting here would
        // drain the encoder this DP was queued behind.
        gpu.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| anyhow::anyhow!("poll for viterbi submit: {e}"))?;
        if let Some(e) = pollster::block_on(guard.pop()) {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            bail!("gpu viterbi: {e}");
        }

        let (tx_h, rx_h) = mpsc::channel();
        self.slots[slot_i]
            .hidden
            .slice(..map_end(hidden_bytes))
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_h.send(r);
            });
        let rx_b = if back_bytes > 0 {
            let (tx_b, rx_b) = mpsc::channel();
            self.slots[slot_i]
                .back
                .slice(..map_end(back_bytes))
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx_b.send(r);
                });
            Some(rx_b)
        } else {
            None
        };
        self.slots[slot_i].flight = Some(Flight {
            index,
            hidden_bytes,
            back_rows: if self.linear { 0 } else { produced },
            rx_h,
            rx_b,
        });
        self.front = front;
        self.frames += n_kept;
        self.windows += 1;
        if let Some(t) = t_sub {
            dp_prof::add(&dp_prof::SUBMIT_US, t.elapsed());
        }
        Ok(())
    }

    pub(crate) fn is_linear(&self) -> bool {
        self.linear
    }

    pub(crate) fn seg(&self) -> usize {
        self.seg
    }

    pub(crate) fn state_count(&self) -> usize {
        self.s as usize
    }

    pub(crate) fn checkpoint(&self, k: usize) -> Result<&[f64]> {
        let s = self.s as usize;
        let start = k.checked_mul(s).context("checkpoint index")?;
        self.ckpt_host.get(start..start + s).with_context(|| {
            format!(
                "missing checkpoint {k} of {}",
                self.ckpt_host.len() / s.max(1)
            )
        })
    }

    /// Put a segment's starting alpha in buffer 0. The next step reads it.
    pub(crate) fn load_alpha(&mut self, gpu: &Gpu, alpha: &[f64]) -> Result<()> {
        anyhow::ensure!(alpha.len() == self.s as usize, "alpha width");
        let bytes = encode_alpha_bytes(self.precision, alpha);
        gpu.upload(&self.alpha[0], &bytes);
        self.front = 0;
        Ok(())
    }

    /// Record `n` step frames whose logits are packed at rows `0..n`, writing
    /// choice rows `back_row0..back_row0+n` of the device back buffer. The
    /// caller submits the encoder. Bind groups must stay alive until then.
    ///
    /// All frames share one compute pass: dispatches within a pass are ordered
    /// with full memory visibility (WebGPU), and the per-frame pass boundaries
    /// this replaced cost CPU recording time on every window of the file.
    pub(crate) fn record_steps(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        logits: &wgpu::Buffer,
        vocab: usize,
        n: usize,
        back_row0: usize,
    ) -> Result<StepBinds> {
        anyhow::ensure!((1..=BACK_CAP).contains(&n), "replay chunk {n}");
        let stride = u32::try_from(vocab).context("vocab")?;
        let uni = self.slot as usize;
        let mut raw = vec![0u8; n * uni];
        for local in 0..n {
            let cfg = StepCfg {
                s: self.s,
                stride,
                row: u32::try_from(local).context("replay row")?,
                mode: 0,
                back_row: u32::try_from(back_row0 + local).context("replay back row")?,
                rb_u32: self.rb_u32,
                pad0: 0,
                pad1: 0,
            };
            let off = local * uni;
            let b = bytemuck::bytes_of(&cfg);
            raw[off..off + b.len()].copy_from_slice(b);
        }
        gpu.queue.write_buffer(&self.cfg, 0, &raw);
        let g0 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-replay-0"),
            layout: &self.step_layout,
            entries: &[
                buf_entry(0, &self.alpha[0]),
                buf_entry(1, &self.alpha[1]),
                buf_entry(2, logits),
                buf_entry(3, &self.cols),
                buf_entry(4, &self.skip),
                buf_entry(5, &self.star),
                buf_entry(6, &self.back),
                buf_entry(7, &self.ninf),
                dyn_uni(8, &self.cfg),
            ],
        });
        let g1 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-replay-1"),
            layout: &self.step_layout,
            entries: &[
                buf_entry(0, &self.alpha[1]),
                buf_entry(1, &self.alpha[0]),
                buf_entry(2, logits),
                buf_entry(3, &self.cols),
                buf_entry(4, &self.skip),
                buf_entry(5, &self.star),
                buf_entry(6, &self.back),
                buf_entry(7, &self.ninf),
                dyn_uni(8, &self.cfg),
            ],
        });
        let mut front = self.front;
        let groups = self.s.div_ceil(16).div_ceil(256).max(1);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viterbi-replay-frames"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.step_pipe);
            for local in 0..n {
                let off = (local * uni) as u32;
                let bg = if front == 0 { &g0 } else { &g1 };
                pass.set_bind_group(0, bg, &[off]);
                pass.dispatch_workgroups(groups, 1, 1);
                front ^= 1;
            }
        }
        self.front = front;
        Ok(StepBinds { _g0: g0, _g1: g1 })
    }

    /// Record the segment's device-side traceback: one thread walks the
    /// choice rows this segment's replay just wrote, backwards from
    /// `states[hi]`, leaving `states[lo-1..hi-1]` on the device. Submitted,
    /// not waited on — segments chain through `states_buf` itself and the
    /// whole state path is read back once at the end of the traceback.
    pub(crate) fn trace_segment(&mut self, gpu: &Gpu, lo: usize, hi: usize) -> Result<()> {
        let cfg = TraceCfg {
            lo: u32::try_from(lo).context("trace lo")?,
            hi: u32::try_from(hi).context("trace hi")?,
            t_len: u32::try_from(self.n_frames).context("trace t_len")?,
            rb_u32: self.rb_u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };
        gpu.queue
            .write_buffer(&self.trace_uni, 0, bytemuck::bytes_of(&cfg));
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-trace"),
            layout: &self.trace_layout,
            entries: &[
                buf_entry(0, &self.back),
                buf_entry(1, &self.states_buf),
                buf_entry(2, &self.trace_uni),
            ],
        });
        let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viterbi-trace"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("viterbi-trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit([enc.finish()]);
        drop(bg);
        gpu.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| anyhow::anyhow!("poll for viterbi trace: {e}"))?;
        if let Some(e) = pollster::block_on(guard.pop()) {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            bail!("gpu viterbi trace: {e}");
        }
        Ok(())
    }

    /// Walk the packed choices. Same end rule as the CPU DP: `alpha[s-2] > alpha[s-1]`
    /// picks s-2, and each step subtracts the 2-bit choice.
    pub(crate) fn trace_states(&mut self, alpha: &[f64]) -> Result<(Vec<i32>, f64)> {
        anyhow::ensure!(!self.linear, "linear viterbi rebuilds choices on traceback");
        let t_len = self.frames;
        let s = alpha.len();
        anyhow::ensure!(s == self.s as usize && s >= 1, "alpha width");
        anyhow::ensure!(t_len >= 1, "no frames");
        anyhow::ensure!(
            self.backs.rows() == t_len - 1,
            "backpointer rows {} != {}",
            self.backs.rows(),
            t_len - 1
        );
        let mut s_end = s - 1;
        if s >= 2 && alpha[s - 2] > alpha[s - 1] {
            s_end = s - 2;
        }
        let total = alpha[s_end];
        let mut states = vec![0i32; t_len];
        states[t_len - 1] = s_end as i32;
        let mut cur = s_end;
        for t in (1..t_len).rev() {
            cur = cur.saturating_sub(self.backs.choice(t - 1, cur)?);
            states[t - 1] = cur as i32;
        }
        Ok((states, total))
    }

    fn collect_slot(&mut self, gpu: &Gpu, slot: usize) -> Result<()> {
        let flight = match self.slots[slot].flight.take() {
            Some(f) => f,
            None => return Ok(()),
        };
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(flight.index),
                timeout: None,
            })
            .map_err(|e| anyhow::anyhow!("poll for viterbi window: {e}"))?;
        recv_map(&flight.rx_h, gpu)?;
        if let Some(rx) = &flight.rx_b {
            recv_map(rx, gpu)?;
        }

        let mut raw = vec![0u8; flight.hidden_bytes];
        {
            let slice = self.slots[slot]
                .hidden
                .slice(..map_end(flight.hidden_bytes));
            let mapped = slice.get_mapped_range()?;
            raw.copy_from_slice(&mapped[..flight.hidden_bytes]);
        }
        self.slots[slot].hidden.unmap();
        anyhow::ensure!(raw.len() % 4 == 0, "hidden readback short");
        let rows: &[f32] = bytemuck::cast_slice(&raw);
        let need = self.hidden_host.len() + rows.len();
        anyhow::ensure!(
            need <= self.n_frames * self.hidden,
            "hidden archive exceeds {} frames",
            self.n_frames
        );
        self.push_hidden(rows)?;

        if flight.back_rows > 0 {
            let stride_b = self.rb_u32 as usize * 4;
            let take = flight.back_rows * stride_b;
            let mut packed = vec![0u8; flight.back_rows * self.rb];
            {
                let slice = self.slots[slot].back.slice(..map_end(take));
                let mapped = slice.get_mapped_range()?;
                anyhow::ensure!(mapped.len() >= take, "backpointer readback short");
                for row in 0..flight.back_rows {
                    let src = row * stride_b;
                    let dst = row * self.rb;
                    packed[dst..dst + self.rb].copy_from_slice(&mapped[src..src + self.rb]);
                }
            }
            self.slots[slot].back.unmap();
            self.backs.push(&packed)?;
        }
        self.collected += 1;
        Ok(())
    }

    /// Grow the encoder-row vec by about 32 MB at a time, not by doubling and
    /// not by reserving the whole file up front.
    fn push_hidden(&mut self, rows: &[f32]) -> Result<()> {
        const STEP: usize = 8 * 1024 * 1024;
        if self.hidden_host.capacity() < self.hidden_host.len() + rows.len() {
            let add = rows.len().max(STEP);
            self.hidden_host
                .try_reserve_exact(add)
                .context("encoder-row growth")?;
        }
        self.hidden_host.extend_from_slice(rows);
        Ok(())
    }

    pub(crate) fn read_alpha(&self, gpu: &Gpu) -> Result<Vec<f64>> {
        let bytes = gpu.readback(&self.alpha[self.front], self.s as u64 * 8)?;
        anyhow::ensure!(bytes.len() == self.s as usize * 8, "alpha readback short");
        Ok(decode_alpha_bytes(self.precision, &bytes))
    }
}

pub(crate) struct StepBinds {
    // Held so the bind groups outlive `queue.submit`.
    _g0: wgpu::BindGroup,
    _g1: wgpu::BindGroup,
}

struct Flight {
    index: wgpu::SubmissionIndex,
    hidden_bytes: usize,
    back_rows: usize,
    rx_h: Receiver<Result<(), wgpu::BufferAsyncError>>,
    rx_b: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

struct MapSlot {
    hidden: wgpu::Buffer,
    back: wgpu::Buffer,
    flight: Option<Flight>,
}

impl MapSlot {
    fn new(gpu: &Gpu, index: usize, hidden_bytes: u64, back_bytes: u64) -> Result<Self> {
        Ok(Self {
            hidden: map_buf(gpu, &format!("dp-hidden-map-{index}"), hidden_bytes)?,
            back: map_buf(gpu, &format!("dp-back-map-{index}"), back_bytes)?,
            flight: None,
        })
    }
}

fn map_buf(gpu: &Gpu, label: &str, bytes: u64) -> Result<wgpu::Buffer> {
    // 16 bytes past the copy: map_end rounds a 4-byte size up to 8.
    let size = (bytes + 31) & !15;
    Ok(gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// Map ranges are 8-byte aligned. Copy sizes are 4-byte aligned, so the mapped
/// range is at most 4 bytes past the bytes we read.
pub(crate) fn map_end(bytes: usize) -> u64 {
    let n = bytes as u64;
    (n + 7) & !7
}

fn recv_map(rx: &Receiver<Result<(), wgpu::BufferAsyncError>>, gpu: &Gpu) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        match rx.try_recv() {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(e)) => return Err(anyhow::Error::from(e).context("map buffer")),
            Err(TryRecvError::Disconnected) => bail!("map callback dropped"),
            Err(TryRecvError::Empty) => {
                if Instant::now() > deadline {
                    bail!("map callback did not arrive");
                }
                gpu.device
                    .poll(wgpu::PollType::Poll)
                    .map_err(|e| anyhow::anyhow!("poll for viterbi map: {e}"))?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// Packed backpointer rows. RAM is a list of fixed slabs so growing does not
/// copy the prefix. Once `len` crosses `ceiling`, later rows append to a file.
struct BackStore {
    rb: usize,
    ceiling: usize,
    slab_rows: usize,
    slab_bytes: usize,
    slabs: Vec<Vec<u8>>,
    len: usize,
    file_rows: usize,
    file: Option<std::fs::File>,
    path: Option<std::path::PathBuf>,
    cache: Vec<u8>,
    cache_lo: usize,
    cache_hi: usize,
}

impl BackStore {
    #[cfg(test)]
    fn with_slab_rows(rb: usize, ceiling: usize, slab_rows: usize) -> Self {
        let mut s = Self::new(rb, ceiling);
        s.slab_rows = slab_rows.max(1);
        s.slab_bytes = s.slab_rows * rb;
        s
    }

    fn new(rb: usize, ceiling: usize) -> Self {
        let slab_rows = ((64 * 1024 * 1024) / rb.max(1)).max(1);
        Self {
            rb,
            ceiling,
            slab_rows,
            slab_bytes: slab_rows * rb,
            slabs: Vec::new(),
            len: 0,
            file_rows: 0,
            file: None,
            path: None,
            cache: Vec::new(),
            cache_lo: 0,
            cache_hi: 0,
        }
    }

    fn rows(&self) -> usize {
        self.len / self.rb + self.file_rows
    }

    #[cfg(test)]
    fn ram_bytes(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn spilled_rows(&self) -> usize {
        self.file_rows
    }

    fn megabytes(&self) -> (f64, f64) {
        let ram = self.len as f64 / (1024.0 * 1024.0);
        let disk = (self.file_rows * self.rb) as f64 / (1024.0 * 1024.0);
        (ram, disk)
    }

    fn push(&mut self, packed: &[u8]) -> Result<()> {
        anyhow::ensure!(
            packed.len().is_multiple_of(self.rb),
            "partial backpointer row"
        );
        if packed.is_empty() {
            return Ok(());
        }
        let grew = self.len.saturating_add(packed.len());
        if self.ceiling == 0 || self.file.is_some() || grew > self.ceiling {
            self.write_spill(packed)?;
            return Ok(());
        }
        let mut rest = packed;
        while !rest.is_empty() {
            if self
                .slabs
                .last()
                .map(|s| s.len() == self.slab_bytes)
                .unwrap_or(true)
            {
                let mut slab = Vec::new();
                slab.try_reserve_exact(self.slab_bytes)
                    .context("backpointer slab")?;
                self.slabs.push(slab);
            }
            let slab = self.slabs.last_mut().context("backpointer slab")?;
            let room = self.slab_bytes - slab.len();
            let take = room.min(rest.len());
            slab.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            self.len += take;
        }
        Ok(())
    }

    fn write_spill(&mut self, packed: &[u8]) -> Result<()> {
        if self.file.is_none() {
            let n = SPILL_SEQ.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("ctc-dp-back-{}-{n}.bin", std::process::id()));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .with_context(|| format!("backpointer spill {}", path.display()))?;
            let cap_mb = self.ceiling as f64 / (1024.0 * 1024.0);
            eprintln!(
                "ctc-aligner: backpointer RAM cap {cap_mb:.0} MB reached, spilling to {}",
                path.display()
            );
            self.path = Some(path);
            self.file = Some(file);
        }
        let file = self.file.as_mut().context("spill file")?;
        file.write_all(packed).context("backpointer spill write")?;
        self.file_rows += packed.len() / self.rb;
        Ok(())
    }

    /// 2-bit choice at `(row, st)`, matching the CPU packing.
    fn choice(&mut self, row: usize, st: usize) -> Result<usize> {
        let ram_rows = self.len / self.rb;
        let byte_i = st >> 2;
        anyhow::ensure!(byte_i < self.rb, "backpointer state {st} outside the row");
        let byte = if row < ram_rows {
            let slab = &self.slabs[row / self.slab_rows];
            let off = (row % self.slab_rows) * self.rb + byte_i;
            slab[off]
        } else {
            self.spill_byte(row - ram_rows, byte_i)?
        };
        Ok(((byte >> ((st & 3) * 2)) & 0b11) as usize)
    }

    fn spill_byte(&mut self, file_row: usize, byte_i: usize) -> Result<u8> {
        if file_row < self.cache_lo || file_row >= self.cache_hi {
            let hi = file_row + 1;
            let lo = hi.saturating_sub(256);
            let n = hi - lo;
            self.cache.resize(n * self.rb, 0);
            let file = self.file.as_mut().context("spill file")?;
            file.flush().context("backpointer spill flush")?;
            file.seek(SeekFrom::Start((lo as u64) * (self.rb as u64)))
                .context("backpointer spill seek")?;
            file.read_exact(&mut self.cache)
                .context("backpointer spill read")?;
            self.cache_lo = lo;
            self.cache_hi = hi;
        }
        let off = (file_row - self.cache_lo) * self.rb + byte_i;
        Ok(self.cache[off])
    }
}

impl Drop for BackStore {
    fn drop(&mut self) {
        self.file.take();
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);

/// Env-gated (`CTC_DP_PROF=1`) phase counters for the GPU DP. The counters
/// answer one question — where does the DP's wall time go — and cost one
/// relaxed atomic load per call when profiling is off.
pub(crate) mod dp_prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static ON: AtomicU64 = AtomicU64::new(2); // 2 = unchecked, 1 = on, 0 = off

    pub(crate) static RECORD_US: AtomicU64 = AtomicU64::new(0);
    pub(crate) static SUBMIT_US: AtomicU64 = AtomicU64::new(0);
    pub(crate) static TRACE_US: AtomicU64 = AtomicU64::new(0);
    pub(crate) static SCORES_US: AtomicU64 = AtomicU64::new(0);

    fn on() -> bool {
        match ON.load(Ordering::Relaxed) {
            1 => true,
            0 => false,
            _ => {
                let flag = std::env::var("CTC_DP_PROF").ok().as_deref() == Some("1");
                ON.store(if flag { 1 } else { 0 }, Ordering::Relaxed);
                flag
            }
        }
    }

    /// Profiling gate, read once per call. Pair with [`add`]: `let t =
    /// enabled().then(Instant::now); … add(&CELL, t.unwrap().elapsed())` —
    /// no closures, so the timed region keeps its natural variable scopes.
    pub(crate) fn enabled() -> bool {
        on()
    }

    pub(crate) fn add(cell: &AtomicU64, d: Duration) {
        if on() {
            cell.fetch_add(d.as_micros() as u64, Ordering::Relaxed);
        }
    }

    /// Time `f` into `cell` when profiling is on; run `f` bare otherwise.
    pub(crate) fn time<R>(cell: &AtomicU64, f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let t = std::time::Instant::now();
        let r = f();
        cell.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        r
    }

    pub(crate) fn dump(label: &str) {
        if !on() {
            return;
        }
        let us = |c: &AtomicU64| c.swap(0, Ordering::Relaxed) as f64 / 1e6;
        eprintln!(
            "[dp-prof] {label}: record+submit {:.3}s, poll+map {:.3}s, traceback {:.3}s, scores {:.3}s",
            us(&RECORD_US),
            us(&SUBMIT_US),
            us(&TRACE_US),
            us(&SCORES_US),
        );
    }
}

fn fill_ninf(gpu: &Gpu, buf: &wgpu::Buffer, n: usize, precision: DpPrecision) {
    let unit = encode_alpha_bytes(precision, &[f64::NEG_INFINITY]);
    let mut bytes = vec![0u8; n * 8];
    for chunk in bytes.as_chunks_mut::<8>().0 {
        chunk.copy_from_slice(&unit);
    }
    gpu.upload(buf, &bytes);
}

fn storage_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(32),
        },
        count: None,
    }
}

fn uniform_static(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size),
        },
        count: None,
    }
}

fn bind_layout(
    gpu: &Gpu,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    gpu.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
}

fn pipe(
    gpu: &Gpu,
    label: &str,
    wgsl: &str,
    layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::ComputePipeline> {
    let pl = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
    gpu.pipeline(label, wgsl, "main", Some(&pl))
}

fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn dyn_uni(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    uni_entry(binding, buffer, 0, UNI_SIZE)
}

fn uni_entry(
    binding: u32,
    buffer: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: Some(std::num::NonZeroU64::new(size).unwrap()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // The df64 rule set, mirrored from `STEP_DF64_WGSL` line for line.
    //
    // The pure tests below drive exactly these, so a change to the shader
    // that is not made here too shows up as a path difference rather than as
    // a silently different answer.
    // ------------------------------------------------------------------

    fn df_is_ninf(a: [f32; 2]) -> bool {
        a[0] <= -1.0e37
    }

    fn df_ge(a: [f32; 2], b: [f32; 2]) -> bool {
        a[0] > b[0] || (a[0] == b[0] && a[1] >= b[1])
    }

    /// The shader's `df_keep`. Rust evaluates float expressions in the order
    /// written, so this is the identity here; it is mirrored because the
    /// shader needs it to stop a driver from folding the two-sum away.
    fn df_keep(x: f32) -> f32 {
        x.min(3.0e38).max(-3.0e38)
    }

    fn df_add_f32(a: [f32; 2], b: f32) -> [f32; 2] {
        if df_is_ninf(a) || b <= -1.0e37 {
            return [DF_NINF_HI, 0.0];
        }
        let s = df_keep(a[0] + b);
        let v = df_keep(s - a[0]);
        let p = df_keep(s - v);
        let e = df_keep((a[0] - p) + (b - v));
        let lo = df_keep(a[1] + e);
        let hi = df_keep(s + lo);
        let err = df_keep(lo - (hi - s));
        [hi, err]
    }

    /// `(choices, final alpha)`. The rule is the shader's: `stay` wins a tie,
    /// then `advance`, then `skip`.
    fn reference_dp_f64(
        logits: &[f32],
        vocab: usize,
        cols: &[usize],
        skip_dead: &[bool],
        t_len: usize,
        s: usize,
    ) -> (Vec<u8>, Vec<f64>) {
        let emit = |t: usize, st: usize| logits[t * vocab + cols[st]] as f64;
        let mut prev = vec![f64::NEG_INFINITY; s];
        for st in 0..s {
            if st <= 1 {
                prev[st] = emit(0, st);
            }
        }
        let mut choices = vec![0u8; t_len * s];
        for t in 1..t_len {
            let mut next = vec![f64::NEG_INFINITY; s];
            for st in 0..s {
                let stay = prev[st];
                let adv = if st >= 1 {
                    prev[st - 1]
                } else {
                    f64::NEG_INFINITY
                };
                let skip = if st >= 2 && !skip_dead[st] {
                    prev[st - 2]
                } else {
                    f64::NEG_INFINITY
                };
                let (choice, best) = if stay >= adv && stay >= skip {
                    (0u8, stay)
                } else if adv >= skip {
                    (1u8, adv)
                } else {
                    (2u8, skip)
                };
                next[st] = best + emit(t, st);
                // The device's back-row layout: the choice that formed frame
                // `t` lives in row `t-1`, which is what `trace_back` reads.
                choices[(t - 1) * s + st] = choice;
            }
            prev = next;
        }
        (choices, prev)
    }

    /// The same DP on `df64` pairs. Returns the choice table, the final alpha
    /// and whether any stored half ever became a NaN.
    fn reference_dp_df64(
        logits: &[f32],
        vocab: usize,
        cols: &[usize],
        skip_dead: &[bool],
        t_len: usize,
        s: usize,
    ) -> (Vec<u8>, Vec<[f32; 2]>, bool) {
        let emit = |t: usize, st: usize| logits[t * vocab + cols[st]];
        let ninf = [DF_NINF_HI, 0.0];
        let mut prev = vec![ninf; s];
        for st in 0..s {
            if st <= 1 {
                let e = emit(0, st);
                prev[st] = if e > -1.0e37 { [e, 0.0] } else { ninf };
            }
        }
        let mut choices = vec![0u8; t_len * s];
        let mut nan = false;
        for t in 1..t_len {
            // A separate row: `adv`/`skip` read the *previous* frame while
            // this loop overwrites the current one.
            let mut next = vec![ninf; s];
            for st in 0..s {
                let stay = prev[st];
                let adv = if st >= 1 { prev[st - 1] } else { ninf };
                let skip = if st >= 2 && !skip_dead[st] {
                    prev[st - 2]
                } else {
                    ninf
                };
                let (choice, best) = if df_ge(stay, adv) && df_ge(stay, skip) {
                    (0u8, stay)
                } else if df_ge(adv, skip) {
                    (1u8, adv)
                } else {
                    (2u8, skip)
                };
                let v = df_add_f32(best, emit(t, st));
                nan |= v[0].is_nan() || v[1].is_nan();
                next[st] = v;
                choices[(t - 1) * s + st] = choice;
            }
            prev = next;
        }
        (choices, prev, nan)
    }

    /// Walk a choice table back from the same end state the kernels pick:
    /// `s-2` when its score beats `s-1`, `s-1` otherwise.
    fn trace_back(choices: &[u8], alpha: &[f64], s: usize, t_len: usize) -> Vec<usize> {
        let mut cur = s - 1;
        if s >= 2 && alpha[s - 2] > alpha[s - 1] {
            cur = s - 2;
        }
        let mut states = vec![0usize; t_len];
        states[t_len - 1] = cur;
        for t in (1..t_len).rev() {
            let c = choices[(t - 1) * s + cur] as usize;
            cur -= cur.min(c);
            states[t - 1] = cur;
        }
        states
    }

    /// Per-frame emissions whose mean is `-0.44`, so an hour of frames lands
    /// the alpha near −80,000, with a ~1e-5 spread between states: orders of
    /// magnitude above the df64 rounding error and far below f32's.
    ///
    /// `dead` is one vocabulary column that stays at `-inf` over `window`,
    /// which makes the states carrying that label unreachable there — the
    /// only way a real `-inf` reaches the DP, and the case the finite
    /// sentinel exists for.
    fn synthetic_logits(
        vocab: usize,
        t_len: usize,
        dead: usize,
        window: std::ops::Range<usize>,
    ) -> Vec<f32> {
        let mut out = vec![0f32; t_len * vocab];
        for t in 0..t_len {
            for v in 0..vocab {
                let wiggle = ((t * 37 + v * 11) % 7) as f32 - 3.0;
                out[t * vocab + v] = -0.44 + 1e-5 * wiggle;
            }
        }
        for t in window {
            if t < t_len {
                out[t * vocab + dead] = f32::NEG_INFINITY;
            }
        }
        out
    }

    fn expanded(tokens: &[usize], blank: usize) -> Vec<usize> {
        let mut out = vec![blank; 2 * tokens.len() + 1];
        for (i, &t) in tokens.iter().enumerate() {
            out[2 * i + 1] = t;
        }
        out
    }

    fn skip_dead(labels: &[usize], blank: usize) -> Vec<bool> {
        (0..labels.len())
            .map(|st| st >= 2 && (labels[st] == blank || labels[st] == labels[st - 2]))
            .collect()
    }

    #[test]
    fn dp_precision_follows_the_adapter_and_the_override() {
        // No override: whatever the device can do, it does.
        assert_eq!(select_dp_precision(true, None).unwrap(), DpPrecision::F64);
        assert_eq!(
            select_dp_precision(true, Some("")).unwrap(),
            DpPrecision::F64
        );
        assert_eq!(
            select_dp_precision(true, Some("auto")).unwrap(),
            DpPrecision::F64
        );
        assert_eq!(
            select_dp_precision(true, Some("AUTO")).unwrap(),
            DpPrecision::F64
        );
        assert_eq!(select_dp_precision(false, None).unwrap(), DpPrecision::Df64);
        assert_eq!(
            select_dp_precision(false, Some("auto")).unwrap(),
            DpPrecision::Df64
        );
        // `f64` is a request, not a wish: it fails on an adapter that cannot
        // run it instead of quietly dropping 5 bits.
        assert_eq!(
            select_dp_precision(true, Some("f64")).unwrap(),
            DpPrecision::F64
        );
        let err = select_dp_precision(false, Some("f64")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("SHADER_F64"), "{msg}");
        assert!(msg.contains("CTC_GPU_DP_MODE=f64"), "{msg}");
        // `df64` works everywhere, including where f64 would.
        assert_eq!(
            select_dp_precision(true, Some("df64")).unwrap(),
            DpPrecision::Df64
        );
        assert_eq!(
            select_dp_precision(false, Some("DF64")).unwrap(),
            DpPrecision::Df64
        );
        // Anything else is refused with the accepted set in the message.
        for bad in ["f32", "double", "1", "on"] {
            let err = select_dp_precision(true, Some(bad)).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains(bad), "{msg}");
            assert!(msg.contains("df64"), "{msg}");
        }
    }

    #[test]
    fn df64_round_trips_hold_half_the_f64_bits() {
        // The sentinel, both ways.
        assert_eq!(f64_to_df64(f64::NEG_INFINITY), [DF_NINF_HI, 0.0]);
        assert_eq!(df64_to_f64(DF_NINF_HI, 0.0), f64::NEG_INFINITY);
        // Past the cutoff on the finite side is unreachable too.
        assert_eq!(f64_to_df64(-1.0e37), [DF_NINF_HI, 0.0]);
        assert_eq!(f64_to_df64(-1.0e40), [DF_NINF_HI, 0.0]);
        assert!(df64_to_f64(-1.0e37, 0.0).is_infinite());

        // Exactly representable values stay exact.
        for v in [0.0f64, -0.5, -1.0, -2.0, -4096.0] {
            let [hi, lo] = f64_to_df64(v);
            assert_eq!(df64_to_f64(hi, lo), v, "{v}");
            assert_eq!(lo, 0.0, "{v} has no residual");
        }

        // The interesting one: an hour-long alpha, where an f32 ulp is 1/128
        // and the f32 path would move a boundary. df64's is below 1e-9.
        for v in [
            -80_000.123_456_789f64,
            -79_999.999_987_654,
            -12_345.678_901,
            1e-7,
        ] {
            let [hi, lo] = f64_to_df64(v);
            let back = df64_to_f64(hi, lo);
            assert!((back - v).abs() < 1e-9, "{v} came back as {back}");
            // and the pair is a genuine refinement of the f32 half
            assert!((hi as f64 - v).abs() >= (back - v).abs(), "{v}");
        }
        // The ulp at -80,000, spelled out: 48 bits of significand.
        let at = f64_to_df64(-80_000.0);
        let step = f64_to_df64(-80_000.0 + 1e-9);
        assert_eq!(at[0], step[0], "1e-9 is below one f32 ulp here");
        assert_ne!(at[1], step[1], "but df64 resolves it");

        // The byte codec is the buffer layout: 8 bytes per state, both ways.
        let alpha = vec![-80_000.123_456_789f64, 0.0, -1.5, f64::NEG_INFINITY];
        for p in [DpPrecision::F64, DpPrecision::Df64] {
            let bytes = encode_alpha_bytes(p, &alpha);
            assert_eq!(bytes.len(), alpha.len() * 8, "{p:?}");
            let back = decode_alpha_bytes(p, &bytes);
            assert_eq!(back.len(), alpha.len(), "{p:?}");
            for (i, (got, want)) in back.iter().zip(&alpha).enumerate() {
                if want.is_infinite() {
                    assert_eq!(got, want, "{p:?} [{i}]");
                } else {
                    assert!((got - want).abs() < 1e-9, "{p:?} [{i}]: {want} vs {got}");
                }
            }
        }
    }

    #[test]
    fn df64_rules_match_f64_over_an_hour_of_frames() {
        // A 3-state tie: two unreachable states compare equal, so `stay`
        // wins, exactly as `-inf >= -inf` does in f64.
        assert!(df_ge([DF_NINF_HI, 0.0], [DF_NINF_HI, 0.0]));
        assert!(df_is_ninf([DF_NINF_HI, 0.0]));
        assert!(!df_is_ninf([-80_000.0, 0.0]));
        // and a finite pair orders lexicographically
        let a = f64_to_df64(-100.25);
        let b = f64_to_df64(-100.5);
        assert!(df_ge(a, b) && !df_ge(b, a));

        // An hour at 50 Hz.
        let t_len = 180_000usize;
        let tokens: Vec<usize> = (1..=8).collect();
        let blank = 0usize;
        let vocab = 9usize;
        let labels = expanded(&tokens, blank);
        let s = labels.len();
        let dead = skip_dead(&labels, blank);
        // Label 3 is unreachable for 60 frames: the states carrying it (and
        // the blanks behind them) carry -inf through 60 transitions.
        let logits = synthetic_logits(vocab, t_len, 3, 100..160);

        let (c64, alpha64) = reference_dp_f64(&logits, vocab, &labels, &dead, t_len, s);
        let (cd, alphad, nan) = reference_dp_df64(&logits, vocab, &labels, &dead, t_len, s);

        assert!(!nan, "a df64 half became NaN");
        for a in &alphad {
            assert!(a[0].is_finite() || df_is_ninf(*a), "{a:?}");
            assert!(a[1].is_finite(), "{a:?}");
        }

        // The magnitude the test is about: an hour of -0.44 per frame.
        let total = alpha64[s - 1].max(alpha64[s - 2]);
        assert!(
            (-85_000.0..-75_000.0).contains(&total),
            "alpha {total} is not an hour of frames"
        );
        // and the unreachable window really was unreachable: the state that
        // carries label 3 is -inf for those frames in both runs.
        let recovered = alphad
            .iter()
            .map(|a| df64_to_f64(a[0], a[1]))
            .collect::<Vec<_>>();
        for (i, (f, d)) in alpha64.iter().zip(&recovered).enumerate() {
            if f.is_infinite() {
                assert!(d.is_infinite(), "state {i}: f64 -inf, df64 {d}");
            } else {
                assert!((f - d).abs() < 1e-6, "state {i}: {f} vs {d}");
            }
        }

        // The point of the exercise: the path cannot move.
        let path64 = trace_back(&c64, &alpha64, s, t_len);
        let path_df = trace_back(&cd, &recovered, s, t_len);
        assert_eq!(path64.len(), t_len);
        for t in 0..t_len {
            assert_eq!(path64[t], path_df[t], "path diverged at frame {t}");
        }
        // none of the choice tables drifted either
        assert_eq!(c64, cd, "a choice differs from the f64 rule");
    }

    #[test]
    fn df64_shaders_parse_and_validate() {
        // No device needed: this is the check a machine without Vulkan can
        // still run, and the one that proves the `vec2<f32>` rows, the
        // sentinel arithmetic and the workgroup declarations are well typed.
        crate::shaders::naga_validate(INIT_DF64_WGSL, "viterbi-init-df64");
        crate::shaders::naga_validate(STEP_DF64_WGSL, "viterbi-step-df64");
        // The f64 kernels are untouched, so they must still validate too.
        crate::shaders::naga_validate(INIT_WGSL, "viterbi-init-f64");
        crate::shaders::naga_validate(STEP_WGSL, "viterbi-step-f64");
        // 8 bytes per state on both sides of the boundary: no stride changes.
        assert_eq!(std::mem::size_of::<[f32; 2]>(), std::mem::size_of::<f64>());
        for src in [INIT_DF64_WGSL, STEP_DF64_WGSL] {
            assert!(src.contains("array<vec2<f32>>"), "a df64 row is a pair");
            // no double-width type anywhere: the DP runs on f32 pairs alone
            assert!(
                !src.contains("array<f64>"),
                "df64 must not declare f64 rows"
            );
            assert!(!src.contains("f64("), "df64 must not cast to f64");
            assert!(!src.contains(": f64"), "df64 must not name f64");
        }
        assert!(STEP_DF64_WGSL.contains("var<workgroup> choices: array<u32, 4096>;"));
    }

    // ------------------------------------------------------------------
    // The one GPU test in this module. It takes the same lock `shaders::tests`
    // does: `cargo test` runs both in parallel threads and this machine's
    // driver does not survive two Vulkan devices in one process.
    // ------------------------------------------------------------------

    fn vulkan() -> Option<Gpu> {
        match pollster::block_on(Gpu::new_with(crate::gpu::DeviceSelector::Runtime {
            api: wgpu::Backend::Vulkan,
            index: 0,
        })) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skipping the GPU dp checks: no Vulkan adapter ({e:#})");
                None
            }
        }
    }

    #[test]
    fn vulkan_df64_dp_matches_the_f64_kernel_and_the_cpu_dp() {
        let _gpu_lock = crate::shaders::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(gpu) = vulkan() else { return };
        eprintln!("[dp] {}", gpu.describe());

        let tokens: Vec<usize> = (1..=6).collect();
        let blank = 0usize;
        let vocab = 8usize;
        let t_len = 700usize;
        let labels = expanded(&tokens, blank);
        let s = labels.len();
        let dead = skip_dead(&labels, blank);
        // Label 3 is unreachable for 60 frames, so the kernels carry the
        // sentinel through `df_add_f32` and the -inf path through the f64 one.
        let logits = synthetic_logits(vocab, t_len, 3, 100..160);

        let (c64, alpha64_cpu) = reference_dp_f64(&logits, vocab, &labels, &dead, t_len, s);
        let path_cpu = trace_back(&c64, &alpha64_cpu, s, t_len);

        let logits_buf = gpu.storage("t-dp-logits", (t_len * vocab * 4) as u64);
        gpu.upload(&logits_buf, bytemuck::cast_slice(&logits));
        let out_x = gpu.storage("t-dp-hidden", (t_len * 4) as u64);

        let run = |precision: DpPrecision| -> (Vec<i32>, Vec<f64>) {
            let mut dp =
                GpuDp::new_with_precision(&gpu, &tokens, usize::MAX, blank, t_len, 1, precision)
                    .unwrap_or_else(|e| panic!("{precision:?} pipeline: {e:#}"));
            dp.submit_window(&gpu, &logits_buf, &out_x, vocab, 0, t_len)
                .unwrap_or_else(|e| panic!("{precision:?} submit: {e:#}"));
            dp.finish(&gpu)
                .unwrap_or_else(|e| panic!("{precision:?} finish: {e:#}"));
            let alpha = dp
                .read_alpha(&gpu)
                .unwrap_or_else(|e| panic!("{precision:?} readback: {e:#}"));
            assert!(
                alpha.iter().all(|v| !v.is_nan()),
                "{precision:?} produced a NaN"
            );
            let (states, _total) = dp
                .trace_states(&alpha)
                .unwrap_or_else(|e| panic!("{precision:?} traceback: {e:#}"));
            (states, alpha)
        };

        // The rule set on the host, for comparison with the device's copy of
        // it: if the two ever disagree, the shader and this module have
        // drifted apart.
        let (cd, ad, _) = reference_dp_df64(&logits, vocab, &labels, &dead, t_len, s);
        let host_total: Vec<f64> = ad.iter().map(|a| df64_to_f64(a[0], a[1])).collect();
        for (i, (a, b)) in host_total.iter().zip(&alpha64_cpu).enumerate() {
            if b.is_infinite() {
                assert!(a.is_infinite(), "host df64 state {i}: {a} vs {b}");
            } else {
                assert!((a - b).abs() < 1e-6, "host df64 state {i}: {a} vs {b}");
            }
        }
        assert_eq!(
            trace_back(&cd, &host_total, s, t_len)
                .iter()
                .map(|&s| s as i32)
                .collect::<Vec<_>>(),
            path_cpu.iter().map(|&s| s as i32).collect::<Vec<_>>(),
            "the host mirror of the df64 rules disagrees with the f64 DP"
        );

        // df64 works on every adapter, including this one.
        let (states_df, alpha_df) = run(DpPrecision::Df64);
        let want: Vec<i32> = path_cpu.iter().map(|&s| s as i32).collect();
        let worst = alpha_df
            .iter()
            .zip(&alpha64_cpu)
            .filter(|(_, b)| b.is_finite())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert_eq!(
            states_df, want,
            "df64 path differs from the CPU f64 DP (worst alpha gap {worst:e})"
        );
        for (i, (a, b)) in alpha_df.iter().zip(&alpha64_cpu).enumerate() {
            if b.is_infinite() {
                assert!(a.is_infinite(), "df64 state {i}: {a} vs {b}");
            } else {
                assert!((a - b).abs() < 1e-6, "df64 state {i}: {a} vs {b}");
            }
        }

        // And f64, where the adapter has it, agrees with both.
        if gpu.has_f64() {
            let (states_64, alpha_64) = run(DpPrecision::F64);
            assert_eq!(states_64, want, "f64 path differs from the CPU f64 DP");
            for (i, (a, b)) in alpha_64.iter().zip(&alpha_df).enumerate() {
                assert!((a - b).abs() < 1e-6, "state {i}: f64 {a} vs df64 {b}");
            }
        } else {
            eprintln!("[dp] no SHADER_F64 on this adapter; df64 matched the CPU DP alone");
        }
    }

    #[test]
    fn slabs_round_trip_across_boundaries() {
        let mut s = BackStore::with_slab_rows(4, 1024 * 1024, 3);
        for i in 0..10u8 {
            s.push(&[i, i.wrapping_add(1), i.wrapping_add(2), i.wrapping_add(3)])
                .unwrap();
        }
        assert_eq!(s.rows(), 10);
        assert_eq!(s.spilled_rows(), 0);
        for i in 0..10usize {
            let b = i as u8;
            assert_eq!(s.choice(i, 0).unwrap(), (b & 0b11) as usize);
            assert_eq!(
                s.choice(i, 4).unwrap(),
                ((b.wrapping_add(1)) & 0b11) as usize
            );
        }
    }

    #[test]
    fn spill_keeps_the_prefix_and_reads_the_suffix() {
        let mut s = BackStore::with_slab_rows(4, 8, 2);
        for i in 0..6u8 {
            s.push(&[i, 0, 0, 0]).unwrap();
        }
        assert_eq!(s.ram_bytes() / 4, 2);
        assert_eq!(s.spilled_rows(), 4);
        for i in 0..6usize {
            assert_eq!(s.choice(i, 0).unwrap(), (i as u8 & 0b11) as usize);
        }
    }

    #[test]
    fn zero_cap_spills_every_row() {
        let mut s = BackStore::with_slab_rows(4, 0, 2);
        s.push(&[1, 2, 3, 4]).unwrap();
        s.push(&[5, 6, 7, 8]).unwrap();
        assert_eq!(s.ram_bytes(), 0);
        assert_eq!(s.spilled_rows(), 2);
        assert_eq!(s.choice(0, 0).unwrap(), 1 & 0b11);
        assert_eq!(s.choice(1, 1).unwrap(), (5 >> 2) & 0b11);
    }
}
