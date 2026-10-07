//! CTC Viterbi on the GPU, f64 alphas, full rows.
//!
//! Alphas stay f64 because an hour-long path sits near −80000, where an f32
//! ulp is large enough to move a boundary. The choice table is kept only when
//! it fits the Viterbi budget. Longer files store one alpha row per segment
//! and rebuild that segment while tracing back.
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
        fill_ninf(gpu, &alpha[0], s);
        fill_ninf(gpu, &alpha[1], s);
        let ninf = gpu.storage("dp-ninf", 8);
        gpu.upload(&ninf, &f64::NEG_INFINITY.to_le_bytes());
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
        let init_pipe = pipe(gpu, "viterbi-init", INIT_WGSL, &init_layout)?;
        let step_pipe = pipe(gpu, "viterbi-step", STEP_WGSL, &step_layout)?;
        let gather_pipe = pipe(gpu, "viterbi-gather", GATHER_WGSL, &gather_layout)?;
        let trace_pipe = pipe(gpu, "viterbi-trace", TRACE_WGSL, &trace_layout)?;
        let trace_uni = gpu.uniform("dp-trace-uni", 32);

        let ncheck = if linear {
            (n_frames - 1) / seg + 1
        } else {
            0
        };
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
            eprintln!(
                "ctc-aligner: viterbi on gpu (f64), checkpoints {ck_mb:.0} MB every {seg} frames, one segment {seg_mb:.0} MB rebuilt on traceback, hidden grows to {hid_mb:.0} MB (full table would be {back_mb:.0} MB)"
            );
        } else {
            eprintln!(
                "ctc-aligner: viterbi on gpu (f64), one pass, backpointers {back_mb:.0} MB, hidden grows to {hid_mb:.0} MB"
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
        self.hidden_host
            .get(start..end)
            .with_context(|| format!("hidden archive has {} rows, need {frame}+{n}", self.hidden_host.len() / self.hidden.max(1)))
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
            self.ckpt_host = raw
                .chunks_exact(8)
                .map(|c| {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(c);
                    f64::from_le_bytes(b)
                })
                .collect();
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
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
            if self.linear && global % self.seg == 0 && i + 1 < n_kept {
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
            if last_global % self.seg == 0 {
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
        let mut bytes = vec![0u8; alpha.len() * 8];
        for (i, v) in alpha.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
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
        anyhow::ensure!(n >= 1 && n <= BACK_CAP, "replay chunk {n}");
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
        gpu.queue.write_buffer(&self.trace_uni, 0, bytemuck::bytes_of(&cfg));
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
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
            let slice = self.slots[slot].hidden.slice(..map_end(flight.hidden_bytes));
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
        if self.collected == 1 || self.collected % 16 == 0 {
            let hid = (self.hidden_host.len() * 4) as f64 / (1024.0 * 1024.0);
            if self.linear {
                eprintln!(
                    "[align] dp windows {}, hidden {hid:.0} MB, choices not stored",
                    self.collected
                );
            } else {
                let (ram, disk) = self.backs.megabytes();
                eprintln!(
                    "[align] dp windows {}, backpointers {ram:.0} MB ram + {disk:.0} MB spilled, hidden {hid:.0} MB",
                    self.collected
                );
            }
        }
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
        let mut out = Vec::with_capacity(self.s as usize);
        for chunk in bytes.chunks_exact(8) {
            let mut b = [0u8; 8];
            b.copy_from_slice(chunk);
            out.push(f64::from_le_bytes(b));
        }
        Ok(out)
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
        anyhow::ensure!(packed.len() % self.rb == 0, "partial backpointer row");
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
            if self.slabs.last().map(|s| s.len() == self.slab_bytes).unwrap_or(true) {
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
            let path = std::env::temp_dir().join(format!(
                "ctc-dp-back-{}-{n}.bin",
                std::process::id()
            ));
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
    let bits = f64::NEG_INFINITY.to_le_bytes();
    let mut bytes = vec![0u8; n * 8];
    for chunk in bytes.chunks_exact_mut(8) {
        chunk.copy_from_slice(&bits);
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

fn bind_layout(gpu: &Gpu, label: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn pipe(gpu: &Gpu, label: &str, wgsl: &str, layout: &wgpu::BindGroupLayout) -> Result<wgpu::ComputePipeline> {
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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

fn uni_entry(binding: u32, buffer: &wgpu::Buffer, offset: u64, size: u64) -> wgpu::BindGroupEntry<'_> {
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
    use super::BackStore;

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
            assert_eq!(s.choice(i, 4).unwrap(), ((b.wrapping_add(1)) & 0b11) as usize);
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
