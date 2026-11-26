//! CTC Viterbi on the GPU: signed 64-bit `Q31.32` fixed-point alphas.
//!
//! An hour of audio sums to an alpha near −80,000, where one f32 ulp is 1/128
//! — large enough to move a boundary, and the reason this DP used to need
//! `SHADER_F64` on some devices and a `df64` `(hi, lo)` pair on the rest. The
//! frame-max normalization that replaced them shifts each row by one scalar
//! per frame, which keeps the *leading* states near zero but cannot keep a
//! state that lags the row max by thousands of nats near zero at the same
//! time: `90s_ja.wav` starts with an ASCII header the Japanese audio does not
//! pronounce, so the leading `<blank>` stays at 0 while the forced states sit
//! −2,400 nats behind. There the lagging path rounded at 2.4e-4, and a
//! one-frame boundary flipped.
//!
//! So the alphas are integers: `Q31.32`, 32 fractional bits of a signed
//! 64-bit value, held in WGSL as `vec2<u32>` — `x` the low word, `y` the high
//! word's bit pattern — which in little-endian memory is bit for bit a Rust
//! `i64`. Integer addition and comparison are exact and associative by spec,
//! so every state carries 2^-32 (2.3e-10) of absolute precision whether it
//! sits at 0, at −2,600, or at −80,000, and the kernel is plain `u32`/`i32`
//! arithmetic on every adapter: no `SHADER_F64`, no `df64`, no `df_keep`.
//!
//! The range is ±2.1e9, which at a tenth of a nat a frame is over ten
//! thousand hours of audio. Nothing is added back on the host either: the row
//! *is* the absolute log-probability, so a checkpoint is the row itself and
//! replaying a segment from it reproduces the forward pass bit for bit.
//!
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
    /// Bit 0: the ping the host dispatches this frame with. The shader never
    /// reads it — which row is `prev` is the bind group's job — but the host
    /// asserts the uniform and the bind group agree, which is how the replay's
    /// ping bookkeeping stayed honest.
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

// ---------------------------------------------------------------------------
// Q31.32 on the host.
//
// The value is `hi + lo / 2^32` in two's complement, so its little-endian
// byte image is `[lo: u32, hi: i32]` — exactly the `vec2<u32>` the kernels
// load. Decoding splits the halves rather than dividing the `i64`, because
// `hi as f64 + lo as f64 * 2^-32` keeps all 53 bits of mantissa even when
// `|v| > 2^53`, which an `i64 as f64` cast would not.
// ---------------------------------------------------------------------------

/// The unreachable sentinel: `i64::MIN`, bytes `[0u32, 0x8000_0000u32]`.
/// `-inf` must never reach the recurrence — `(-inf) - (-inf)` is a NaN, and
/// one NaN would silently poison every transition that flows through an
/// unreachable state, surfacing as a wrong boundary seconds away from the
/// state that produced it. The high word being `i32::MIN` puts it far below
/// [`Q_NINF_CUT`], and the kernels test that by threshold, not equality.
const Q_NINF: i64 = i64::MIN;
/// Anything at or below this is unreachable: −2e9 in value terms, which no
/// live score over any real file comes close to.
const Q_NINF_CUT: i64 = (-2_000_000_000i64) << 32;
/// `2^-32`, the Q31.32 fractional step.
const Q_INV: f64 = 1.0 / 4_294_967_296.0;

/// A Q31.32 value back in `f64`, with the sentinel mapped to `-inf`.
fn q64_to_f64(v: i64) -> f64 {
    if v <= Q_NINF_CUT {
        f64::NEG_INFINITY
    } else {
        let hi = (v >> 32) as i32;
        let lo = v as u32;
        (hi as f64) + (lo as f64) * Q_INV
    }
}

/// One Q31.32 row out of device bytes: `[lo: u32, hi: i32]` per value.
fn decode_q64_bytes(raw: &[u8]) -> Vec<i64> {
    raw.as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            let lo = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let hi = i32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            ((hi as i64) << 32) | (lo as i64)
        })
        .collect()
}

/// The inverse of [`decode_q64_bytes`], written word by word so it is
/// endian-independent.
fn encode_q64_bytes(vals: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 8);
    for &v in vals {
        out.extend_from_slice(&(v as u32).to_le_bytes());
        out.extend_from_slice(&((v >> 32) as i32).to_le_bytes());
    }
    out
}

/// States one workgroup owns, the `base = wid.x * 4096u` stride of the step
/// kernel and the 16 KiB its `choices` array holds.
const WG_STATES: usize = 4096;

/// `state_meta` bits 0..29: the vocabulary column the state gathers.
const META_COL: u32 = (1 << 30) - 1;
/// `state_meta` bit 30: the skip arc is forbidden (`skip_dead`).
const META_SKIP_DEAD: u32 = 1 << 30;
/// `state_meta` bit 31: the state is a `<star>` target.
const META_STAR: u32 = 1 << 31;

/// Workgroups the kernels need for `s` states: one per 4096, at least one.
fn num_workgroups(s: usize) -> u32 {
    (s.div_ceil(WG_STATES)).max(1) as u32
}

/// The `Q31.32` arithmetic, shared by the step and init kernels.
const Q_WGSL: &str = r#"
const NINF_HI: i32 = -2147483647 - 1;
const NINF_CUT_HI: i32 = -2000000000;
const Q_SCALE: f32 = 4294967296.0;

fn q_ninf() -> vec2<u32> {
    return vec2<u32>(0u, bitcast<u32>(NINF_HI));
}

fn q_ge(a: vec2<u32>, b: vec2<u32>) -> bool {
    let ah = bitcast<i32>(a.y);
    let bh = bitcast<i32>(b.y);
    return (ah > bh) || (ah == bh && a.x >= b.x);
}

fn f32_to_q(x: f32) -> vec2<u32> {
    // Catches -inf, NaN, and sentinel values cleanly without NaN propagation.
    if (!(x > -1000000.0)) {
        return q_ninf();
    }
    let cx = min(x, 1000000.0);
    let t = trunc(cx);
    let r = cx - t;
    // Since |r| < 1.0 in f32, |r| <= 1.0 - 2^-24, so |r| * 2^32 <= 2^32 - 256
    // < u32::MAX.
    let mag = u32(round(abs(r) * Q_SCALE));
    var hi = i32(t);
    var lo = mag;
    if (r < 0.0 && mag != 0u) {
        hi = hi - 1;
        lo = 0u - mag;
    }
    return vec2<u32>(lo, bitcast<u32>(hi));
}

fn q_add_f32(a: vec2<u32>, e: f32) -> vec2<u32> {
    let ah = bitcast<i32>(a.y);
    if (ah <= NINF_CUT_HI || !(e > -1000000.0)) {
        return q_ninf();
    }
    let b = f32_to_q(e);
    let lo = a.x + b.x;
    let carry = select(0, 1, lo < a.x);
    let hi = ah + bitcast<i32>(b.y) + carry;
    if (hi <= NINF_CUT_HI) {
        return q_ninf();
    }
    return vec2<u32>(lo, bitcast<u32>(hi));
}
"#;

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

@group(0) @binding(0) var<storage, read> prev_row: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> next_row: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> state_meta: array<u32>;
@group(0) @binding(4) var<storage, read_write> back: array<u32>;
@group(0) @binding(5) var<uniform> cfg: StepCfg;

const META_COL: u32 = 0x3fffffffu;
const META_SKIP_DEAD: u32 = 0x40000000u;
const META_STAR: u32 = 0x80000000u;
"#;

const STEP_BODY_WGSL: &str = r#"
// One workgroup covers 4096 states (256 threads x 16 states). A thread's 16
// states are STRIDED by the workgroup width, not contiguous: at loop step k
// the 32 threads of a warp touch 32 consecutive states, so every row load is
// a coalesced 256 B of vec2<u32>. The row-major mapping this replaced made
// each thread own states [g*16, g*16+16), which read 16 cache lines per warp
// load — measured 120 us per frame on the DP, 16x the row's actual traffic.
// 4096 u32 is exactly the 16 KiB workgroup-storage floor.
var<workgroup> choices: array<u32, 4096>;

fn emit_at(st: u32) -> f32 {
    let m = state_meta[st];
    if ((m & META_STAR) != 0u) {
        return -1.0;
    }
    return src[cfg.row * cfg.stride + (m & META_COL)];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let me = lid.x;
    let base = wid.x * 4096u;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let st = base + k * 256u + me;
        var choice = 0u;
        if (st < cfg.s) {
            let stay = prev_row[st];
            var adv = q_ninf();
            var skip_v = q_ninf();
            if (st >= 1u) {
                adv = prev_row[st - 1u];
            }
            if (st >= 2u && (state_meta[st] & META_SKIP_DEAD) == 0u) {
                skip_v = prev_row[st - 2u];
            }
            var best = stay;
            if (q_ge(stay, adv) && q_ge(stay, skip_v)) {
                // stay: choice 0
            } else if (q_ge(adv, skip_v)) {
                choice = 1u;
                best = adv;
            } else {
                choice = 2u;
                best = skip_v;
            }
            next_row[st] = q_add_f32(best, emit_at(st));
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

@group(0) @binding(0) var<storage, read_write> next_row: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> state_meta: array<u32>;
@group(0) @binding(3) var<uniform> cfg: StepCfg;

const META_COL: u32 = 0x3fffffffu;
const META_STAR: u32 = 0x80000000u;
"#;

const INIT_BODY_WGSL: &str = r#"
fn emit_at(st: u32) -> f32 {
    let m = state_meta[st];
    if ((m & META_STAR) != 0u) {
        return -1.0;
    }
    return src[cfg.row * cfg.stride + (m & META_COL)];
}

// Frame 0: only the two states a path can start in are reachable. Both are
// absolute, so there is no row shift to record — the number in the buffer is
// the log-probability itself.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let st = gid.x;
    if (st < cfg.s) {
        var v = q_ninf();
        if (st <= 1u) {
            v = f32_to_q(emit_at(st));
        }
        next_row[st] = v;
    }
}
"#;

const GATHER_WGSL: &str = r#"
struct Cfg { row0: u32, rows: u32, vocab: u32, s: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> states: array<u32>;
@group(0) @binding(2) var<storage, read> state_meta: array<u32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Cfg;

const META_COL: u32 = 0x3fffffffu;
const META_STAR: u32 = 0x80000000u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= cfg.rows) {
        return;
    }
    let g = cfg.row0 + local;
    let st = states[g];
    var v = -1.0;
    if (st < cfg.s) {
        let m = state_meta[st];
        if ((m & META_STAR) == 0u) {
            v = logits[local * cfg.vocab + (m & META_COL)];
        }
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

/// Refuse a DP whose device buffers cannot be bound, before allocating them.
///
/// `max_binding` is [`Gpu::effective_max_binding`]: the smaller of the
/// adapter's `max_storage_buffer_binding_size` and `max_buffer_size`, because
/// a buffer is created against the one and bound against the other. The three
/// sizes the caller knows are passed in; `alpha`, `meta`, `states` and
/// `scores` are derived here from the state and frame counts, so a caller
/// cannot forget one.
fn check_dp_buffer_limits(
    max_binding: u64,
    s: usize,
    n_frames: usize,
    back_bytes: u64,
    stage_bytes: u64,
    ckpt_bytes: u64,
) -> Result<()> {
    let frames = n_frames as u64;
    let states = s as u64;
    for (what, need) in [
        // One Q31.32 row: 8 bytes a state.
        ("dp-alpha", states * 8),
        ("dp-meta", states * 4),
        ("dp-back", back_bytes),
        ("dp-stage", stage_bytes),
        ("dp-ckpt", ckpt_bytes),
        ("dp-states", frames * 4),
        ("dp-scores", frames * 4),
    ] {
        if need > max_binding {
            anyhow::bail!(
                "{what} needs {need} B for {n_frames} frames x {s} states, over this adapter's \
                 storage-buffer binding limit of {max_binding} B; set CTC_GPU_DP=0 to run the DP \
                 on the host, or use a smaller --window"
            );
        }
    }
    Ok(())
}

/// Device state for one file.
///
/// The choice table is Θ(frames × states). An hour is about 5.5 GB, and
/// writing that to disk only moves the same bytes. The forward pass keeps
/// the table only when it fits the fixed Viterbi budget. Past that it keeps
/// one alpha row per segment — a few tens of megabytes — and the traceback
/// rebuilds a single segment of choices. Encoder rows stay on the host
/// because a second encode would cost more than the rows do.
pub(crate) struct GpuDp {
    /// Two ping-pong rows of Q31.32 alpha, 8 bytes a state.
    alpha: [wgpu::Buffer; 2],
    back: wgpu::Buffer,
    cfg: wgpu::Buffer,
    /// `cols`, `skip_dead` and `star` in one u32 per state: three buffers and
    /// three scattered reads per state became one of each.
    meta: wgpu::Buffer,
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
    ckpt_host: Vec<i64>,
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
        Self::with_plan(gpu, token_ids, star_id, blank_id, n_frames, hidden, None)
    }

    /// [`GpuDp::new`] with the space plan forced instead of resolved by
    /// [`dp_plan`]. The checkpoint-replay test needs the linear plan on a file
    /// short enough that the budget would keep the whole table.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_linear_for_test(
        gpu: &Gpu,
        token_ids: &[usize],
        star_id: usize,
        blank_id: usize,
        n_frames: usize,
        hidden: usize,
        seg: usize,
    ) -> Result<Self> {
        Self::with_plan(
            gpu,
            token_ids,
            star_id,
            blank_id,
            n_frames,
            hidden,
            Some((seg, true)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_plan(
        gpu: &Gpu,
        token_ids: &[usize],
        star_id: usize,
        blank_id: usize,
        n_frames: usize,
        hidden: usize,
        plan: Option<(usize, bool)>,
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
        let mut meta = vec![0u32; s];
        let blank_col = u32::try_from(blank_id).context("blank id")?;
        for (st, &lab) in labels.iter().enumerate() {
            cols[st] = if lab == star_id {
                blank_col
            } else {
                u32::try_from(lab).context("label id")?
            };
        }
        anyhow::ensure!(
            cols.iter().all(|&c| c <= META_COL),
            "gpu viterbi: vocabulary column over the 30-bit state_meta field"
        );
        for st in 0..s {
            meta[st] = cols[st];
            if st >= 2 && (labels[st] == blank_id || labels[st] == labels[st - 2]) {
                meta[st] |= META_SKIP_DEAD;
            }
        }
        if star_id != usize::MAX {
            for (i, &tok) in token_ids.iter().enumerate() {
                if tok == star_id {
                    meta[2 * i + 1] |= META_STAR;
                }
            }
        }

        // Every buffer the DP binds is sized and validated before the first
        // allocation: one past the binding limit otherwise surfaces as a
        // validation error inside the first window, with nothing pointing at
        // the transcript or the window length.
        let (seg, linear) = plan.unwrap_or_else(|| dp_plan(n_frames, s, rb));
        // Non-linear: the whole table is read back per window, so the device
        // buffer holds one window. Linear: the choices are traced back ON the
        // device one segment at a time, so it holds one segment instead —
        // bounded by the same budget that picked `seg`.
        let back_rows = if linear { seg } else { BACK_CAP };
        let back_bytes = (back_rows as u64)
            .checked_mul(rb_u32 as u64)
            .and_then(|n| n.checked_mul(4))
            .context("back buffer size")?;
        let stage_bytes = (BACK_CAP as u64)
            .checked_mul(hidden as u64)
            .and_then(|n| n.checked_mul(4))
            .context("hidden stage size")?;
        let ncheck = if linear { (n_frames - 1) / seg + 1 } else { 0 };
        let ckpt_bytes = (ncheck as u64)
            .saturating_mul(s as u64)
            .saturating_mul(8)
            .max(16);
        check_dp_buffer_limits(
            gpu.effective_max_binding(),
            s,
            n_frames,
            back_bytes,
            stage_bytes,
            ckpt_bytes,
        )?;

        let alpha = [
            gpu.storage("dp-alpha-0", (s * 8) as u64),
            gpu.storage("dp-alpha-1", (s * 8) as u64),
        ];
        fill_ninf(gpu, &alpha[0], s);
        fill_ninf(gpu, &alpha[1], s);
        let meta_buf = gpu.storage("dp-meta", (s * 4) as u64);
        gpu.upload(&meta_buf, bytemuck::cast_slice(&meta));

        let slot = (gpu.device.limits().min_uniform_buffer_offset_alignment as u64).max(UNI_SIZE);
        let cfg = gpu.uniform("dp-cfg", slot * BACK_CAP as u64);
        let back = gpu.storage("dp-back", back_bytes);
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
                uniform_layout(3),
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
                storage_layout(4, false),
                uniform_layout(5),
            ],
        );
        let gather_layout = bind_layout(
            gpu,
            "viterbi-gather",
            &[
                storage_layout(0, true),
                storage_layout(1, true),
                storage_layout(2, true),
                storage_layout(3, false),
                uniform_static(4, 16),
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
        let init_src = format!("{INIT_WGSL}{Q_WGSL}{INIT_BODY_WGSL}");
        let step_src = format!("{STEP_WGSL}{Q_WGSL}{STEP_BODY_WGSL}");
        let init_pipe = pipe(gpu, "viterbi-init", &init_src, &init_layout)?;
        let step_pipe = pipe(gpu, "viterbi-step", &step_src, &step_layout)?;
        let gather_pipe = pipe(gpu, "viterbi-gather", GATHER_WGSL, &gather_layout)?;
        let trace_pipe = pipe(gpu, "viterbi-trace", TRACE_WGSL, &trace_layout)?;
        // Fold the DP's four pipelines into the same cache file the encoder
        // wrote. A warm start then compiles them too; an unwritable cache
        // directory must not fail the DP, so the result is dropped.
        let _ = gpu.save_pipeline_cache();
        let trace_uni = gpu.uniform("dp-trace-uni", 32);

        let ckpt = gpu.storage("dp-ckpt", ckpt_bytes);
        // The table fits: keep it in RAM. It does not: keep nothing of it.
        // A zero ceiling makes every push spill, and the linear path never pushes.
        let backs = BackStore::new(rb, if linear { 0 } else { usize::MAX });
        let back_mb = (n_frames.saturating_sub(1) as f64) * (rb as f64) / (1024.0 * 1024.0);
        let hid_mb = (n_frames as f64) * (hidden as f64) * 4.0 / (1024.0 * 1024.0);
        if linear {
            let ck_mb = (ncheck * s * 8) as f64 / (1024.0 * 1024.0);
            let seg_mb = (seg as f64) * (rb as f64) / (1024.0 * 1024.0);
            eprintln!(
                "ctc-aligner: viterbi on gpu (q31.32), checkpoints {ck_mb:.0} MB every {seg} frames, one segment {seg_mb:.0} MB rebuilt on traceback, hidden grows to {hid_mb:.0} MB (full table would be {back_mb:.0} MB)"
            );
        } else {
            eprintln!(
                "ctc-aligner: viterbi on gpu (q31.32), one pass, backpointers {back_mb:.0} MB, hidden grows to {hid_mb:.0} MB"
            );
        }

        Ok(Self {
            alpha,
            back,
            cfg,
            meta: meta_buf,
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
                buf_entry(2, &self.meta),
                buf_entry(3, &self.score_out),
                uni_entry(4, uni, uni_off, 16),
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
            self.ckpt_host = decode_q64_bytes(&raw);
            anyhow::ensure!(
                self.ckpt_host.len() == ncheck * self.s as usize,
                "checkpoint row count"
            );
        }
        anyhow::ensure!(self.frames >= 1, "gpu viterbi: no frames read back");
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
        let mut front = self.front;
        for local in 0..n_kept {
            let row = u32::try_from(row0 + local).context("logit row")?;
            let init = self.frames == 0 && local == 0;
            if init {
                front = 0;
            }
            kinds.push(!init);
            cfgs.push(StepCfg {
                s: self.s,
                stride,
                row,
                mode: if init { 0 } else { front as u32 },
                back_row: if init { 0 } else { back_row },
                rb_u32: self.rb_u32,
                pad0: 0,
                pad1: 0,
            });
            if !init {
                back_row += 1;
                front ^= 1;
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
                buf_entry(2, &self.meta),
                dyn_uni(3, &self.cfg),
            ],
        });
        let bg01 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-step-0"),
            layout: &self.step_layout,
            entries: &[
                buf_entry(0, &self.alpha[0]),
                buf_entry(1, &self.alpha[1]),
                buf_entry(2, logits),
                buf_entry(3, &self.meta),
                buf_entry(4, &self.back),
                dyn_uni(5, &self.cfg),
            ],
        });
        let bg10 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-step-1"),
            layout: &self.step_layout,
            entries: &[
                buf_entry(0, &self.alpha[1]),
                buf_entry(1, &self.alpha[0]),
                buf_entry(2, logits),
                buf_entry(3, &self.meta),
                buf_entry(4, &self.back),
                dyn_uni(5, &self.cfg),
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
        let step_groups = num_workgroups(self.s as usize);
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
                debug_assert_eq!(front as u32, cfgs[i].mode, "ping drifted");
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

    /// One checkpoint row: the Q31.32 alpha this segment replays from. It is
    /// the absolute row, not a normalized one — there is no shift to restore,
    /// so [`load_alpha`](Self::load_alpha) uploads it as it stands.
    pub(crate) fn checkpoint(&self, k: usize) -> Result<&[i64]> {
        let s = self.s as usize;
        let start = k.checked_mul(s).context("checkpoint index")?;
        self.ckpt_host.get(start..start + s).with_context(|| {
            format!(
                "missing checkpoint {k} of {}",
                self.ckpt_host.len() / s.max(1)
            )
        })
    }

    /// Put a segment's starting alpha in buffer 0. The rows are absolute, so
    /// replaying from a checkpoint reproduces the forward pass's numbers bit
    /// for bit with nothing else to restore.
    pub(crate) fn load_alpha(&mut self, gpu: &Gpu, alpha: &[i64]) -> Result<()> {
        anyhow::ensure!(alpha.len() == self.s as usize, "alpha width");
        gpu.upload(&self.alpha[0], &encode_q64_bytes(alpha));
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
        // Two counters on purpose: `mode` decides the ping each frame's
        // uniform carries, `front` decides which bind group that frame is
        // dispatched with. They must agree, and one variable for both is how
        // the replay ended up reading the previous segment's row.
        let mut flags = self.front;
        for local in 0..n {
            let cfg = StepCfg {
                s: self.s,
                stride,
                row: u32::try_from(local).context("replay row")?,
                mode: flags as u32,
                back_row: u32::try_from(back_row0 + local).context("replay back row")?,
                rb_u32: self.rb_u32,
                pad0: 0,
                pad1: 0,
            };
            flags ^= 1;
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
                buf_entry(3, &self.meta),
                buf_entry(4, &self.back),
                dyn_uni(5, &self.cfg),
            ],
        });
        let g1 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viterbi-replay-1"),
            layout: &self.step_layout,
            entries: &[
                buf_entry(0, &self.alpha[1]),
                buf_entry(1, &self.alpha[0]),
                buf_entry(2, logits),
                buf_entry(3, &self.meta),
                buf_entry(4, &self.back),
                dyn_uni(5, &self.cfg),
            ],
        });
        let groups = num_workgroups(self.s as usize);
        let mut front = self.front;
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

    /// The last alpha row as `f64` absolute log-probabilities; an unreachable
    /// state is `-inf` again instead of the sentinel.
    pub(crate) fn read_alpha(&self, gpu: &Gpu) -> Result<Vec<f64>> {
        let bytes = gpu.readback(&self.alpha[self.front], self.s as u64 * 8)?;
        anyhow::ensure!(bytes.len() == self.s as usize * 8, "alpha readback short");
        Ok(decode_q64_bytes(&bytes)
            .into_iter()
            .map(q64_to_f64)
            .collect())
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

fn fill_ninf(gpu: &Gpu, buf: &wgpu::Buffer, n: usize) {
    let row = vec![Q_NINF; n];
    gpu.upload(buf, &encode_q64_bytes(&row));
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
    // The Q31.32 rule set, mirrored from `Q_WGSL` / `STEP_BODY_WGSL` line for
    // line.
    //
    // The pure tests below drive exactly these, so a change to the shader
    // that is not made here too shows up as a path difference rather than as
    // a silently different answer.
    // ------------------------------------------------------------------

    /// `Q_SCALE` in the shader's own precision.
    const Q_SCALE_F32: f32 = 4294967296.0;
    /// `NINF_CUT_HI`: a high word at or below this is unreachable.
    const NINF_CUT_HI: i32 = -2_000_000_000;

    /// Mirror of `f32_to_q`.
    fn f32_to_q(x: f32) -> i64 {
        if !(x > -1_000_000.0) {
            return Q_NINF;
        }
        let cx = x.min(1_000_000.0);
        let t = cx.trunc();
        let r = cx - t;
        let mag = (r.abs() * Q_SCALE_F32).round() as u32;
        let mut hi = t as i32;
        let mut lo = mag;
        if r < 0.0 && mag != 0 {
            hi -= 1;
            lo = 0u32.wrapping_sub(mag);
        }
        ((hi as i64) << 32) | (lo as i64)
    }

    /// Mirror of `q_ge`.
    fn q_ge(a: i64, b: i64) -> bool {
        let ah = (a >> 32) as i32;
        let bh = (b >> 32) as i32;
        ah > bh || (ah == bh && (a as u32) >= (b as u32))
    }

    /// Mirror of `q_add_f32`.
    fn q_add_f32(a: i64, e: f32) -> i64 {
        let ah = (a >> 32) as i32;
        if ah <= NINF_CUT_HI || !(e > -1_000_000.0) {
            return Q_NINF;
        }
        let b = f32_to_q(e);
        let lo = (a as u32).wrapping_add(b as u32);
        let carry = i32::from(lo < a as u32);
        let hi = ah.wrapping_add((b >> 32) as i32).wrapping_add(carry);
        if hi <= NINF_CUT_HI {
            return Q_NINF;
        }
        ((hi as i64) << 32) | (lo as i64)
    }

    /// `(choices, final alpha)` from a full-precision `f64` DP. The rule is
    /// the shader's: `stay` wins a tie, then `advance`, then `skip`.
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

    /// The host mirror of the kernels: the same Q31.32 recurrence, the same
    /// sentinel. Returns the choice table and the final absolute row.
    fn reference_dp_q32(
        logits: &[f32],
        vocab: usize,
        cols: &[usize],
        skip_dead: &[bool],
        t_len: usize,
        s: usize,
    ) -> (Vec<u8>, Vec<i64>) {
        let (choices, rows) =
            reference_dp_q32_rows(logits, vocab, cols, skip_dead, t_len, s);
        (choices, rows.last().cloned().expect("one row per frame"))
    }

    /// [`reference_dp_q32`] keeping every row, for the checkpoint tests.
    fn reference_dp_q32_rows(
        logits: &[f32],
        vocab: usize,
        cols: &[usize],
        skip_dead: &[bool],
        t_len: usize,
        s: usize,
    ) -> (Vec<u8>, Vec<Vec<i64>>) {
        let emit = |t: usize, st: usize| logits[t * vocab + cols[st]];
        let mut prev = vec![Q_NINF; s];
        for st in 0..s {
            if st <= 1 {
                prev[st] = f32_to_q(emit(0, st));
            }
        }
        let mut choices = vec![0u8; t_len * s];
        let mut rows = vec![prev.clone()];
        for t in 1..t_len {
            let mut next = vec![Q_NINF; s];
            for st in 0..s {
                let stay = prev[st];
                let adv = if st >= 1 { prev[st - 1] } else { Q_NINF };
                let skip = if st >= 2 && !skip_dead[st] {
                    prev[st - 2]
                } else {
                    Q_NINF
                };
                let (choice, best) = if q_ge(stay, adv) && q_ge(stay, skip) {
                    (0u8, stay)
                } else if q_ge(adv, skip) {
                    (1u8, adv)
                } else {
                    (2u8, skip)
                };
                next[st] = q_add_f32(best, emit(t, st));
                choices[(t - 1) * s + st] = choice;
            }
            prev = next;
            rows.push(prev.clone());
        }
        (choices, rows)
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
    /// the absolute alpha near −80,000.
    ///
    /// `spread` sets how far apart competing paths sit. It has to stay well
    /// above the Q31.32 error the recurrence accumulates over `t_len` frames
    /// (< 1e-4 even at 180,000 frames), or the test would be measuring which
    /// of two numerically indistinguishable paths wins rather than whether the
    /// two recurrences agree.
    ///
    /// `dead` is one vocabulary column that stays at `-inf` over `window`,
    /// which makes the states carrying that label unreachable there — the
    /// only way a real `-inf` reaches the DP, and the case the sentinel
    /// exists for.
    fn synthetic_logits(
        vocab: usize,
        t_len: usize,
        dead: usize,
        window: std::ops::Range<usize>,
        spread: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; t_len * vocab];
        for t in 0..t_len {
            for v in 0..vocab {
                let wiggle = ((t * 37 + v * 11) % 7) as f32 - 3.0;
                out[t * vocab + v] = -0.44 + spread * wiggle;
            }
        }
        for t in window {
            if t < t_len {
                out[t * vocab + dead] = f32::NEG_INFINITY;
            }
        }
        out
    }

    /// Emissions with a *pinned* path: at every frame one column is `PIN`
    /// better than the rest, so the winning path's margin per decision is
    /// billions of times the Q31.32 step, whether that is over 100 frames or
    /// over 180,000. That is what makes "the path is exactly the f64 path" a
    /// statement about the recurrence rather than about which of two paths
    /// separated by 1e-10 wins a coin toss — the flat
    /// [`synthetic_logits`] field below is where near-ties are exercised.
    ///
    /// 25 frames on a blank, then 25 on the next label, monotonically: the
    /// path has to advance, hold, and take the skip arc over the blank between
    /// one label and the next. `block` is the frames per stage.
    const PIN: f32 = 4.0;
    fn synthetic_logits_walk(
        vocab: usize,
        t_len: usize,
        dead: usize,
        window: std::ops::Range<usize>,
        block: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; t_len * vocab];
        for t in 0..t_len {
            // The path never has to move backwards, which no CTC path can.
            let block = t / block;
            let max_label = vocab - 1;
            let good = if block > 2 * max_label {
                max_label
            } else if block % 2 == 0 {
                0
            } else {
                1 + (block / 2).min(max_label - 1)
            };
            for v in 0..vocab {
                // Every column gets its own value: an *exact* tie is broken by
                // accumulated rounding rather than by the rule, so a row of
                // identical emissions would make "the path is identical" a
                // statement about that rounding. Distinct values 1e-3 apart sit
                // seven orders of magnitude above the drift measured below, and
                // the tie-break rules themselves are exercised by the shorter,
                // flat-spread GPU tests.
                out[t * vocab + v] = -0.44 + 1e-3 * v as f32 - if v == good { 0.0 } else { PIN };
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

    /// The largest gap between the device's de-coded `f64` row and the CPU's,
    /// over the states that are reachable in both.
    fn worst_gap(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .filter(|(_, want)| want.is_finite())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn the_shaders_are_native_integer_and_fit_the_webgpu_floors() {
        let init_src = format!("{INIT_WGSL}{Q_WGSL}{INIT_BODY_WGSL}");
        let step_src = format!("{STEP_WGSL}{Q_WGSL}{STEP_BODY_WGSL}");
        for (src, what) in [
            (init_src.as_str(), "viterbi-init"),
            (step_src.as_str(), "viterbi-step"),
            (GATHER_WGSL, "viterbi-gather"),
            (TRACE_WGSL, "viterbi-trace"),
        ] {
            crate::shaders::naga_validate(src, what);
            // No double precision anywhere: not the type, not a cast, not the
            // `enable` directive that would unlock it.
            assert!(!src.contains("f64"), "{what} still names f64");
            assert!(!src.contains("df_keep"), "{what} still has the df64 clamp");
            assert!(!src.contains("enable "), "{what} enables a feature");
            assert!(!src.contains("subgroups"), "{what} needs subgroups");
        }
        // The rows the DP binds are Q31.32 (8 bytes a state) and u32 metadata.
        assert!(step_src.contains("prev_row: array<vec2<u32>>"));
        assert!(step_src.contains("next_row: array<vec2<u32>>"));
        assert!(init_src.contains("next_row: array<vec2<u32>>"));
        assert!(step_src.contains("state_meta: array<u32>"));
        assert!(!step_src.contains("cols: array<u32>"));
        assert!(!step_src.contains("star: array<u32>"));
        assert!(!step_src.contains("ninf_buf"));
        assert!(!step_src.contains("norm: array"));
        // 4096 u32 of workgroup storage is exactly WebGPU's 16 KiB floor, and
        // it is the only array the step kernel declares.
        assert_eq!(step_src.matches("var<workgroup>").count(), 1);
        assert!(step_src.contains("var<workgroup> choices: array<u32, 4096>;"));
        assert_eq!(WG_STATES * std::mem::size_of::<u32>(), 16 * 1024);
        // One barrier: no frame-max reduction is left to synchronise.
        assert_eq!(step_src.matches("workgroupBarrier()").count(), 1);
        // StepCfg is the 32-byte uniform the per-frame write packs; the
        // bindings are 5 storage + 1 dynamic uniform, 3 + 1, 4 + 1, 2 + 1.
        assert_eq!(std::mem::size_of::<StepCfg>(), 32);
        assert_eq!(
            [5u32, 3, 4, 2],
            [
                step_src.as_str(),
                init_src.as_str(),
                GATHER_WGSL,
                TRACE_WGSL
            ]
            .map(|s| { s.matches("var<storage").count() as u32 })
        );
        for (src, what) in [
            (init_src.as_str(), "viterbi-init"),
            (step_src.as_str(), "viterbi-step"),
            (GATHER_WGSL, "viterbi-gather"),
            (TRACE_WGSL, "viterbi-trace"),
        ] {
            assert_eq!(src.matches("var<uniform>").count(), 1, "{what}");
        }
    }

    #[test]
    fn q32_conversions_hold_their_corners() {
        let step = 1.0 / 4_294_967_296.0;
        // Exact values land exactly.
        assert_eq!(f32_to_q(0.0), 0);
        assert_eq!(q64_to_f64(f32_to_q(0.0)), 0.0);
        assert_eq!(q64_to_f64(f32_to_q(-1.0)), -1.0);
        assert_eq!(f32_to_q(-1.0), -4_294_967_296);
        assert_eq!(q64_to_f64(f32_to_q(-0.25)), -0.25);
        assert_eq!(q64_to_f64(f32_to_q(0.5)), 0.5);

        // Within half a step for everything else.
        for x in [-1e-8f32, 1e-8, 3.25, -3.25, 123.456, -0.7, 0.7] {
            let got = q64_to_f64(f32_to_q(x));
            assert!(
                (got - f64::from(x)).abs() <= step * 0.5 + 1e-15,
                "{x} -> {got}"
            );
        }

        // Underflow to zero: the fractional part rounds away, and no sign
        // bit survives to make it -0.
        assert_eq!(f32_to_q(-1e-12), 0);
        assert_eq!(q64_to_f64(f32_to_q(-1e-12)), 0.0);

        // The worst-case magnitude: |r| <= 1 - 2^-24, so `r * 2^32` is
        // 2^32 - 256 and never wraps the u32.
        let below_one = -(1.0 - 2f32.powi(-24));
        let q = f32_to_q(below_one);
        assert_eq!(q >> 32, -1);
        assert_eq!(q as u32, 256);
        assert_eq!(q64_to_f64(q), f64::from(below_one));
        let above = 1.0 - 2f32.powi(-24);
        let q = f32_to_q(above);
        assert_eq!(q >> 32, 0);
        assert_eq!(q as u32, u32::MAX - 255);
        assert_eq!(q64_to_f64(q), f64::from(above));

        // Out-of-band inputs: -inf, NaN and the sentinel-ish magnitude all
        // decode to the unreachable sentinel instead of poisoning the DP.
        for x in [f32::NEG_INFINITY, f32::NAN, -3.4e38, -1.0e7] {
            assert_eq!(f32_to_q(x), Q_NINF, "{x}");
            assert_eq!(q64_to_f64(f32_to_q(x)), f64::NEG_INFINITY, "{x}");
        }
        assert_eq!(q64_to_f64(Q_NINF), f64::NEG_INFINITY);
        assert_eq!(q64_to_f64(Q_NINF_CUT), f64::NEG_INFINITY);
        assert!(q64_to_f64(Q_NINF_CUT + (1 << 32)).is_finite());

        // Carry across the low word, both directions.
        let half = f32_to_q(0.5);
        assert_eq!(q64_to_f64(q_add_f32(half, 0.5)), 1.0);
        assert_eq!(q_add_f32(half, 0.5), 1i64 << 32);
        assert_eq!(q64_to_f64(q_add_f32(f32_to_q(0.75), 0.25)), 1.0);
        assert_eq!(q64_to_f64(q_add_f32(f32_to_q(-0.5), -0.5)), -1.0);
        // A fractional add just below -1 borrows into `hi` and lands on the
        // 2^-32 grid.
        let near_one = q64_to_f64(f32_to_q(-0.99999994));
        let got = q64_to_f64(q_add_f32(f32_to_q(-0.99999994), -1e-9));
        assert!((got - (near_one - 1e-9)).abs() <= step, "{got}");
        assert!(got > -1.0 && got < -0.9999999, "{got}");

        // A sentinel predecessor stays sentinel: no subtraction can resurrect
        // an unreachable state, and a dead emission kills a live one.
        assert_eq!(q_add_f32(Q_NINF, -0.44), Q_NINF);
        assert_eq!(q_add_f32(Q_NINF, f32::NEG_INFINITY), Q_NINF);
        assert_eq!(q_add_f32(f32_to_q(-1.0), f32::NEG_INFINITY), Q_NINF);

        // Ordering matches the integer ordering of the pair.
        assert!(q_ge(f32_to_q(1.0), f32_to_q(0.9999999)));
        assert!(!q_ge(f32_to_q(0.9999999), f32_to_q(1.0)));
        assert!(q_ge(Q_NINF, Q_NINF));
        assert!(q_ge(f32_to_q(0.0), Q_NINF));
        assert!(!q_ge(Q_NINF, f32_to_q(0.0)));
        // A value one step below zero still compares below zero.
        assert!(!q_ge(-1, 0));
        assert!(q_ge(0, -1));
    }

    #[test]
    fn q32_keeps_every_state_and_the_whole_choice_table_over_an_hour() {
        // An hour at 50 Hz. The absolute alpha reaches −80,000, where one f32
        // ulp is 1/128; Q31.32's step is 2.3e-10 there too, and the row *is*
        // the absolute number, so no state's precision depends on how far
        // behind the row max it has fallen.
        let t_len = 180_000usize;
        let tokens: Vec<usize> = (1..=8).collect();
        let blank = 0usize;
        let vocab = 9usize;
        let labels = expanded(&tokens, blank);
        let s = labels.len();
        let dead = skip_dead(&labels, blank);
        // Label 3 is unreachable for 60 frames: the states carrying it (and
        // the blanks behind them) carry the sentinel through 60 transitions,
        // and the pinned path has to route around the frames where that
        // column would otherwise be the best one.
        let logits = synthetic_logits_walk(vocab, t_len, 3, 100..160, 25);

        let (c64, alpha64) = reference_dp_f64(&logits, vocab, &labels, &dead, t_len, s);
        let (cq, row) = reference_dp_q32(&logits, vocab, &labels, &dead, t_len, s);
        let recovered: Vec<f64> = row.iter().map(|&v| q64_to_f64(v)).collect();

        // The magnitude this test is about: an hour of −0.44 a frame.
        let total64 = alpha64[s - 1].max(alpha64[s - 2]);
        assert!(
            (-95_000.0..-70_000.0).contains(&total64),
            "alpha {total64} is not an hour of frames"
        );

        // Every state, including the ones that lagged the row max by
        // thousands, is within 1e-4 of the f64 run. Every emission in this
        // field is an f32 below 2^7, so it sits on the 2^-32 grid exactly and
        // the accumulated sum stays on it too: `worst` comes out at zero, the
        // bound the recurrence promises without spending any of it.
        let mut worst = 0.0f64;
        for (i, (got, want)) in recovered.iter().zip(&alpha64).enumerate() {
            if want.is_infinite() {
                assert!(got.is_infinite(), "state {i}: f64 -inf, q32 {got}");
                continue;
            }
            let gap = (got - want).abs();
            worst = worst.max(gap);
            assert!(gap < 1e-4, "state {i}: {want} vs {got} (gap {gap:e})");
        }
        assert!(worst < 1e-9, "a state drifted by {worst:e} over {t_len} frames");

        // ... and the recurrence resolves every arc exactly as `f64` does,
        // so the choice table — not just the path the traceback walks — is
        // identical frame for frame.
        assert_eq!(cq, c64, "the Q31.32 choice table differs from the f64 one");

        let path64 = trace_back(&c64, &alpha64, s, t_len);
        let path_q = trace_back(&cq, &recovered, s, t_len);
        assert_eq!(path64.len(), t_len);
        for t in 0..t_len {
            assert_eq!(path64[t], path_q[t], "path diverged at frame {t}");
        }

        // A row that ends *inside* the dead window: a state there is either
        // the sentinel (its predecessors were already dead) or the logits'
        // own `-inf`. Neither survives, and neither is resurrected.
        let t_dead = 130usize;
        let logits_dead = synthetic_logits_walk(vocab, t_dead, 3, 100..t_dead, 25);
        let (_, row_dead) = reference_dp_q32(&logits_dead, vocab, &labels, &dead, t_dead, s);
        let dead_states: Vec<usize> = (0..s).filter(|&st| labels[st] == 3).collect();
        assert!(!dead_states.is_empty(), "the dead label owns no state");
        for st in dead_states {
            assert_eq!(
                q64_to_f64(row_dead[st]),
                f64::NEG_INFINITY,
                "state {st} was resurrected at the end of a dead window"
            );
        }
    }

    // ------------------------------------------------------------------
    // The GPU tests in this module. Each takes the same lock `shaders::tests`
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

    /// Run one forward pass on the device and return its state path, its
    /// absolute alpha and the raw Q31.32 bytes of the last row.
    fn gpu_forward(
        gpu: &Gpu,
        tokens: &[usize],
        blank: usize,
        vocab: usize,
        logits: &[f32],
        t_len: usize,
    ) -> (Vec<i32>, Vec<f64>, Vec<i64>) {
        let logits_buf = gpu.storage("t-dp-logits", (logits.len() * 4) as u64);
        gpu.upload(&logits_buf, bytemuck::cast_slice(logits));
        let out_x = gpu.storage("t-dp-hidden", (t_len * 4) as u64);
        let mut dp = GpuDp::new(gpu, tokens, usize::MAX, blank, t_len, 1)
            .unwrap_or_else(|e| panic!("dp pipeline: {e:#}"));
        dp.submit_window(gpu, &logits_buf, &out_x, vocab, 0, t_len)
            .unwrap_or_else(|e| panic!("dp submit: {e:#}"));
        dp.finish(gpu)
            .unwrap_or_else(|e| panic!("dp finish: {e:#}"));
        let alpha = dp
            .read_alpha(gpu)
            .unwrap_or_else(|e| panic!("dp readback: {e:#}"));
        assert!(alpha.iter().all(|v| !v.is_nan()), "the DP produced a NaN");
        let raw = gpu
            .readback(&dp.alpha[dp.front], (dp.s as usize * 8) as u64)
            .unwrap_or_else(|e| panic!("raw alpha readback: {e:#}"));
        let raw = decode_q64_bytes(&raw);
        let (states, total) = dp
            .trace_states(&alpha)
            .unwrap_or_else(|e| panic!("dp traceback: {e:#}"));
        // The reported total is the last row's own entry, bit for bit.
        assert_eq!(total, alpha[states[t_len - 1] as usize]);
        (states, alpha, raw)
    }

    #[test]
    fn vulkan_q32_dp_matches_the_cpu_dp() {
        let _gpu_lock = crate::shaders::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(gpu) = vulkan() else { return };
        eprintln!("[dp] {}", gpu.describe());

        let tokens: Vec<usize> = (1..=21).collect();
        let blank = 0usize;
        let vocab = 23usize;
        let t_len = 100usize;
        let labels = expanded(&tokens, blank);
        let s = labels.len();
        assert_eq!(s, 43, "the standard trellis width");
        let dead = skip_dead(&labels, blank);
        // Label 3 is unreachable for 20 frames, so the kernels carry the
        // sentinel through the recurrence rather than only at the edges.
        let logits = synthetic_logits_walk(vocab, t_len, 3, 40..60, 5);

        let (c64, alpha64) = reference_dp_f64(&logits, vocab, &labels, &dead, t_len, s);
        let path_cpu = trace_back(&c64, &alpha64, s, t_len);
        let want: Vec<i32> = path_cpu.iter().map(|&s| s as i32).collect();

        // The device against the f64 reference: the whole point of the
        // exercise is that these two agree on the path, not just on the score.
        let (states, alpha, raw) = gpu_forward(&gpu, &tokens, blank, vocab, &logits, t_len);
        let gap = worst_gap(&alpha, &alpha64);
        assert_eq!(
            states, want,
            "the q31.32 path differs from the f64 DP (gap {gap:e})"
        );
        for (i, (a, b)) in alpha.iter().zip(&alpha64).enumerate() {
            if b.is_infinite() {
                assert!(a.is_infinite(), "state {i}: {a} vs {b}");
            } else {
                assert!((a - b).abs() < 1e-6, "state {i}: {a} vs {b}");
            }
        }
        // ... and the device's own row is bit-for-bit the host mirror's.
        let (_, mirror) = reference_dp_q32(&logits, vocab, &labels, &dead, t_len, s);
        assert_eq!(raw, mirror, "the device row is not the host mirror's");

        // The flat field: competing paths sit within 1e-3 of each other, three
        // orders of magnitude above Q31.32's 2.3e-10 step, so the tie-break has
        // real margins and the device must land on the f64 path too.
        let flat = synthetic_logits(vocab, t_len, 3, 40..60, 0.05);
        let (c64f, alpha64f) = reference_dp_f64(&flat, vocab, &labels, &dead, t_len, s);
        let flat_want: Vec<i32> = trace_back(&c64f, &alpha64f, s, t_len)
            .iter()
            .map(|&s| s as i32)
            .collect();
        let (flat_states, flat_alpha, flat_raw) =
            gpu_forward(&gpu, &tokens, blank, vocab, &flat, t_len);
        let flat_gap = worst_gap(&flat_alpha, &alpha64f);
        assert_eq!(
            flat_states, flat_want,
            "the flat-field path differs from the f64 DP (gap {flat_gap:e})"
        );
        for (i, (a, b)) in flat_alpha.iter().zip(&alpha64f).enumerate() {
            if b.is_infinite() {
                assert!(a.is_infinite(), "state {i}: {a} vs {b}");
            } else {
                assert!((a - b).abs() < 1e-6, "state {i}: {a} vs {b}");
            }
        }
        let (_, flat_mirror) = reference_dp_q32(&flat, vocab, &labels, &dead, t_len, s);
        assert_eq!(
            flat_raw, flat_mirror,
            "the device's flat-field row is not the host mirror's"
        );
    }

    #[test]
    fn vulkan_multi_workgroup_q32_matches_the_cpu_dp() {
        let _gpu_lock = crate::shaders::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(gpu) = vulkan() else { return };
        eprintln!("[dp] {}", gpu.describe());

        // 4151 states is one past the 4096 a workgroup owns, so the choice
        // packing and the row state span two workgroups.
        let tokens: Vec<usize> = (1..=2075).collect();
        let blank = 0usize;
        let vocab = 2076usize;
        let t_len = 40usize;
        let labels = expanded(&tokens, blank);
        let s = labels.len();
        assert_eq!(s, 4151, "one state past a single workgroup");
        assert_eq!(num_workgroups(s), 2, "the row must span two groups");
        let dead = skip_dead(&labels, blank);
        let logits = synthetic_logits_walk(vocab, t_len, 7, 10..20, 4);

        let (c64, alpha64) = reference_dp_f64(&logits, vocab, &labels, &dead, t_len, s);
        let path_cpu = trace_back(&c64, &alpha64, s, t_len);
        let (states, alpha, raw) = gpu_forward(&gpu, &tokens, blank, vocab, &logits, t_len);
        let want: Vec<i32> = path_cpu.iter().map(|&s| s as i32).collect();
        let gap = worst_gap(&alpha, &alpha64);
        assert_eq!(
            states, want,
            "the two-workgroup path differs from the f64 DP (gap {gap:e})"
        );
        for (i, (a, b)) in alpha.iter().zip(&alpha64).enumerate() {
            if b.is_infinite() {
                assert!(a.is_infinite(), "state {i}: {a} vs {b}");
            } else {
                assert!((a - b).abs() < 1e-6, "state {i}: {a} vs {b}");
            }
        }
        let (_, mirror) = reference_dp_q32(&logits, vocab, &labels, &dead, t_len, s);
        assert_eq!(raw, mirror, "the device row is not the host mirror's");
    }

    #[test]
    fn vulkan_linear_checkpoint_replay_reproduces_the_forward_pass() {
        let _gpu_lock = crate::shaders::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(gpu) = vulkan() else { return };
        eprintln!("[dp] {}", gpu.describe());

        let tokens: Vec<usize> = (1..=21).collect();
        let blank = 0usize;
        let vocab = 23usize;
        let t_len = 60usize;
        let seg = 17usize;
        let labels = expanded(&tokens, blank);
        let s = labels.len();
        let dead = skip_dead(&labels, blank);
        let logits = synthetic_logits_walk(vocab, t_len, 3, 20..30, 4);

        let (c64, alpha64) = reference_dp_f64(&logits, vocab, &labels, &dead, t_len, s);
        let path_cpu = trace_back(&c64, &alpha64, s, t_len);
        let want: Vec<i32> = path_cpu.iter().map(|&s| s as i32).collect();
        let (_, mirror_rows) = reference_dp_q32_rows(&logits, vocab, &labels, &dead, t_len, s);

        let logits_buf = gpu.storage("t-dp-logits", (logits.len() * 4) as u64);
        gpu.upload(&logits_buf, bytemuck::cast_slice(&logits));
        let out_x = gpu.storage("t-dp-hidden", (t_len * 4) as u64);
        // Forced linear on a file the budget would keep whole: this is the
        // plan a long file takes, and the one whose replay has to be exact.
        let mut dp = GpuDp::new_linear_for_test(&gpu, &tokens, usize::MAX, blank, t_len, 1, seg)
            .unwrap_or_else(|e| panic!("linear dp pipeline: {e:#}"));
        assert!(dp.is_linear(), "the test forced the linear plan");
        assert_eq!(dp.seg(), seg);
        dp.submit_window(&gpu, &logits_buf, &out_x, vocab, 0, t_len)
            .unwrap_or_else(|e| panic!("linear dp submit: {e:#}"));
        dp.finish(&gpu)
            .unwrap_or_else(|e| panic!("linear dp finish: {e:#}"));
        let alpha = dp
            .read_alpha(&gpu)
            .unwrap_or_else(|e| panic!("linear dp readback: {e:#}"));
        let ncheck = (t_len - 1) / seg + 1;

        // The forward pass itself is the same DP: the plan changes what is
        // stored, never what is computed.
        for (i, (a, b)) in alpha.iter().zip(&alpha64).enumerate() {
            if b.is_infinite() {
                assert!(a.is_infinite(), "state {i}: {a} vs {b}");
            } else {
                assert!((a - b).abs() < 1e-6, "state {i}: {a} vs {b}");
            }
        }

        // Every checkpoint row is the forward's own row at that frame, bit
        // for bit: there is no shift to fold in, so the row is the value.
        assert_eq!(dp.ckpt_host.len(), ncheck * s, "one row per segment start");
        for k in 0..ncheck {
            let got = &dp.ckpt_host[k * s..(k + 1) * s];
            assert_eq!(
                got,
                &mirror_rows[k * seg][..],
                "checkpoint {k} is not the frame {} row",
                k * seg
            );
        }

        // Replay every segment the way `gpu_dp_traceback` does: seed the end
        // state, then for each segment load its checkpoint, re-run the steps
        // from it, and walk the fresh choices back.
        let s_end = if s >= 2 && alpha[s - 2] > alpha[s - 1] {
            s - 2
        } else {
            s - 1
        };
        let seed = s_end as u32;
        gpu.queue.write_buffer(
            dp.states_buf(),
            ((t_len - 1) * 4) as u64,
            bytemuck::bytes_of(&seed),
        );
        let mut b = ((t_len - 1) / seg) * seg;
        loop {
            let k = b / seg;
            let lo = b + 1;
            let hi = (b + seg).min(t_len - 1);
            if lo <= hi {
                let start = dp
                    .checkpoint(k)
                    .unwrap_or_else(|e| panic!("checkpoint {k}: {e:#}"))
                    .to_vec();
                dp.load_alpha(&gpu, &start)
                    .unwrap_or_else(|e| panic!("load alpha {k}: {e:#}"));
                let n = hi - lo + 1;
                // The replay chunk's logits live at rows 0..n, which is where
                // the lm-head replay writes them.
                gpu.upload(
                    &logits_buf,
                    bytemuck::cast_slice(&logits[lo * vocab..(hi + 1) * vocab]),
                );
                let mut enc = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("t-dp-replay"),
                    });
                let binds = dp
                    .record_steps(&gpu, &mut enc, &logits_buf, vocab, n, 0)
                    .unwrap_or_else(|e| panic!("replay {lo}..={hi}: {e:#}"));
                gpu.queue.submit([enc.finish()]);
                drop(binds);
                // The invariant the whole plan rests on: replaying the segment
                // from checkpoint k reproduces checkpoint k+1 exactly, because
                // the row is absolute and the recurrence is integer.
                if (k + 1) * seg <= t_len - 1 {
                    let raw = gpu
                        .readback(&dp.alpha[dp.front], (s * 8) as u64)
                        .unwrap_or_else(|e| panic!("replay readback {k}: {e:#}"));
                    let replay = decode_q64_bytes(&raw);
                    let next = dp
                        .checkpoint(k + 1)
                        .unwrap_or_else(|e| panic!("checkpoint {}: {e:#}", k + 1))
                        .to_vec();
                    assert_eq!(
                        replay, next,
                        "segment {k}: the replay is not the forward row"
                    );
                }
                dp.trace_segment(&gpu, lo, hi)
                    .unwrap_or_else(|e| panic!("trace {lo}..={hi}: {e:#}"));
            }
            if b == 0 {
                break;
            }
            b -= seg;
        }
        let bytes = gpu
            .readback(dp.states_buf(), (t_len * 4) as u64)
            .unwrap_or_else(|e| panic!("state readback: {e:#}"));
        let states: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        assert_eq!(
            states, want,
            "the replayed path differs from the CPU f64 DP"
        );
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

    #[test]
    fn dp_buffer_limits_accept_a_normal_run_and_reject_oversized_buffers() {
        // 2047 MiB, what this machine's Vulkan adapter reports; a minute of
        // audio is 3000 frames and a 43-state trellis needs tens of MiB.
        let cap = 2047u64 << 20;
        let (s, n) = (43usize, 3000usize);
        assert!(check_dp_buffer_limits(cap, s, n, 16 << 20, 64 << 20, 8 << 20).is_ok());
        // The two the caller does not size itself are covered too: 8 bytes a
        // state for the ping-pong rows, 4 for the metadata, 4 a frame.
        assert!(check_dp_buffer_limits(cap, (cap / 8 + 1) as usize, n, 16, 16, 16).is_err());
        assert!(check_dp_buffer_limits(cap, (cap / 4 + 1) as usize, n, 16, 16, 16).is_err());
        assert!(check_dp_buffer_limits(cap, s, (cap / 4 + 1) as usize, 16, 16, 16).is_err());

        // Each named buffer is reported with what it needs, the adapter limit
        // and both ways out.
        for (what, back, stage, ckpt) in [
            ("dp-back", cap + 1, 16, 16),
            ("dp-stage", 16, cap + 1, 16),
            ("dp-ckpt", 16, 16, cap + 1),
        ] {
            let err = check_dp_buffer_limits(cap, s, n, back, stage, ckpt).unwrap_err();
            let msg = format!("{err:#}");
            assert!(msg.contains(what), "{msg}");
            assert!(msg.contains(&(cap + 1).to_string()), "{msg}");
            assert!(msg.contains(&cap.to_string()), "{msg}");
            assert!(msg.contains("CTC_GPU_DP=0"), "{msg}");
            assert!(msg.contains("--window"), "{msg}");
        }

        // The derived sizes are exactly 8 bytes a state (the Q31.32 row) and
        // 4 a state for the metadata and 4 a frame for state and score.
        let alpha_b = s as u64 * 8;
        assert_eq!(alpha_b, s as u64 * std::mem::size_of::<i64>() as u64);
        assert!(check_dp_buffer_limits(alpha_b, s, 0, 0, 0, 0).is_ok());
        assert!(check_dp_buffer_limits(alpha_b - 1, s, 0, 0, 0, 0).is_err());
        let states_b = n as u64 * 4;
        assert!(check_dp_buffer_limits(states_b, 1, n, 0, 0, 0).is_ok());
        assert!(check_dp_buffer_limits(states_b - 1, 1, n, 0, 0, 0).is_err());
    }

    #[test]
    fn q64_bytes_round_trip_independently_of_the_host_layout() {
        let vals = [
            Q_NINF,
            Q_NINF_CUT,
            Q_NINF_CUT + 1,
            -1,
            0,
            1,
            1i64 << 32,
            -4_294_967_296,
            ((i32::MIN + 1) as i64) << 32,
        ];
        let bytes = encode_q64_bytes(&vals);
        assert_eq!(bytes.len(), vals.len() * 8);
        assert_eq!(decode_q64_bytes(&bytes), vals);
        // The sentinel's byte image is the zero low word and a negative high
        // word, which is what the kernel's `vec2<u32>(0u, bitcast(i32::MIN))`
        // writes on a little-endian device.
        assert_eq!(&bytes[..8], &[0, 0, 0, 0, 0, 0, 0, 0x80]);
        // -1.0 is `[0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF]`.
        assert_eq!(&bytes[56..64], &[0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn state_meta_packs_the_column_the_skip_flag_and_the_star() {
        // The three buffers this replaced, in one word: a 30-bit column, the
        // skip-dead bit, the star bit.
        let tokens = [4usize, 9, 4];
        let labels = expanded(&tokens, 0);
        let mut meta = vec![0u32; labels.len()];
        for (st, &lab) in labels.iter().enumerate() {
            meta[st] = u32::try_from(lab).unwrap();
            if st >= 2 && (lab == 0 || lab == labels[st - 2]) {
                meta[st] |= META_SKIP_DEAD;
            }
        }
        meta[3] |= META_STAR;
        // A star target keeps the blank's column, so the mask still resolves
        // a real vocabulary index for the score replay.
        meta[5] |= META_STAR;
        for st in 0..meta.len() {
            let m = meta[st];
            assert_eq!(
                m & META_COL,
                u32::try_from(labels[st]).unwrap(),
                "state {st}"
            );
            assert_eq!(
                m & META_SKIP_DEAD != 0,
                st >= 2 && (labels[st] == 0 || labels[st] == labels[st - 2]),
                "state {st}"
            );
        }
        assert_ne!(meta[3] & META_STAR, 0);
        assert_eq!(meta[4] & META_STAR, 0);
        // The bits do not collide: a full 30-bit column leaves both flags.
        assert_eq!(META_COL, 0x3fff_ffff);
        assert_eq!(META_SKIP_DEAD, 0x4000_0000);
        assert_eq!(META_STAR, 0x8000_0000);
        assert_eq!(META_COL & META_SKIP_DEAD, 0);
        assert_eq!(META_COL & META_STAR, 0);
        assert_eq!(META_SKIP_DEAD & META_STAR, 0);
    }
}
