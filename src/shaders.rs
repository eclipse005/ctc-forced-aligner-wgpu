//! WGSL kernel builders (fp32).
//!
//! Conventions:
//! * activations are (rows, cols) row-major f32 storage buffers;
//! * weight matrices for GEMM are pre-transposed to (K, N) at load time, so
//!   the B tile loads along N are contiguous;
//! * `gemm` carries row strides for A and B so sliced operands (attention
//!   heads) work without copies; `conv_gemm` computes the im2col A-tile
//!   on the fly for the strided conv stack.

#[path = "gemm_kernel.rs"]
mod gemm_kernel;

pub use gemm_kernel::{
    conv_gemm, conv_gemm_variant, gemm_bias, gemm_bias_tiled, gemm_shared_bytes, gemm_tile,
    gemm_variant, pick_gemm_variant, pick_gemm_variant_for, pos_conv, GemmVariant, KS, MT, NT,
    POS_KS, POS_MT, POS_NT,
};

/// What a device's subgroup width means for a reduction kernel.
///
/// `Warp32`/`Warp64` are the two shapes a warp-wide `subgroupAdd` can take;
/// `Generic` is any other advertised range (Intel 8..32, wraparound 32..64),
/// where the kernel has to read the width from `@builtin(subgroup_size)`.
/// `None` keeps the shared-memory tree.
///
/// Classified from the adapter's reported limits, never from the vendor, so a
/// driver that clamps its range is followed rather than guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupMode {
    None,
    Warp32,
    Warp64,
    Generic,
}

impl SubgroupMode {
    /// `has_subgroups` is the negotiated [`wgpu::Features::SUBGROUP`] flag;
    /// `min`/`max` are the adapter's subgroup size range.
    pub fn from_limits(
        has_subgroups: bool,
        min_subgroup_size: u32,
        max_subgroup_size: u32,
    ) -> Self {
        if !has_subgroups || min_subgroup_size == 0 || max_subgroup_size == 0 {
            return Self::None;
        }
        if min_subgroup_size == 32 && max_subgroup_size == 32 {
            return Self::Warp32;
        }
        if min_subgroup_size == 64 && max_subgroup_size == 64 {
            return Self::Warp64;
        }
        Self::Generic
    }
}

/// Row LayerNorm over `cols` (affine, biased variance, eps inside sqrt) then
/// gelu when `do_gelu` is 1.  256 threads per row; the row lives in registers
/// as one `vec4` per thread (cols is 512 or 1024), and the mean / variance
/// reductions use the device's subgroup shape ([`SubgroupMode`]) — the
/// previous serial 256-step loop by thread 0 was why this kernel ran at
/// 14 GB/s.
///
/// Two shaders: [`layernorm`] normalises in place (conv stack, feature
/// projection, final LN); [`layernorm_sd`] reads `src` and writes `dst`,
/// which lets the encoder layers LN the residual stream without a staging
/// copy.
pub(crate) fn layernorm(sg: SubgroupMode) -> String {
    emit_ln(
        sg,
        "
@group(0) @binding(0) var<storage, read_write> h: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b4: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> cfg: Cfg;",
        "h[idx]",
        "h[idx]",
    )
}

/// Source/destination variant; uniform sits at binding 4.
pub(crate) fn layernorm_sd(sg: SubgroupMode) -> String {
    emit_ln(
        sg,
        "
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> b4: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> cfg: Cfg;",
        "src[idx]",
        "dst[idx]",
    )
}

// ---- workgroup reductions shared by the four row kernels -------------------
//
// Each row kernel folds 256 per-thread partials into one workgroup total. The
// shared-memory tree that [`SubgroupMode::None`] keeps needs eight barrier
// round trips per reduction; the subgroup paths need one store, one barrier
// and one unguarded subgroup op.
//
// The "unguarded" part is what shapes the code. Naga's static uniformity
// analysis treats `local_invocation_id` as non-uniform, so a `subgroupAdd`
// inside `if (lid.x < 32u)` — the obvious way to skip lanes past the last
// subgroup — is a shader-creation error. Every subgroup op below therefore
// sits in workgroup-uniform control flow (guarded only by `workgroup_id`, the
// uniform buffer, and `.0`-lane id comparisons that never wrap an op), and the
// total reaches every subgroup at once instead of being broadcast from
// subgroup 0.
//
// `select` evaluates both arms, so the array index is clamped with `min`: a
// lane past the last partial must not read out of bounds even though its
// value is discarded.
//
// Emitted as functions rather than inlined at each site: a workgroup barrier
// inside a function called from uniform control flow is legal, and it keeps
// the kernel bodies readable.
//
// `Generic` assumes the advertised subgroup holds at least 4 invocations, so
// 256 threads span at most 64 subgroups — the smallest width a WebGPU adapter
// has ever reported, hence the 64-slot arrays below.

/// Extra `@builtin` parameters `main` needs for this mode.
fn red_builtins(sg: SubgroupMode) -> &'static str {
    match sg {
        SubgroupMode::None => "",
        SubgroupMode::Warp32 | SubgroupMode::Warp64 => {
            ",\n        @builtin(subgroup_invocation_id) sg_lane: u32"
        }
        SubgroupMode::Generic => {
            ",\n        @builtin(subgroup_invocation_id) sg_lane: u32,\n        @builtin(subgroup_size) sg_size: u32"
        }
    }
}

/// The arguments a reduction helper takes after the value(s) it folds.
fn red_args(sg: SubgroupMode) -> &'static str {
    match sg {
        SubgroupMode::None => ", lid.x",
        SubgroupMode::Warp32 | SubgroupMode::Warp64 => ", lid.x, sg_lane",
        SubgroupMode::Generic => ", lid.x, sg_lane, sg_size",
    }
}

/// One reduction shared between two independent sums: `wg_sum2(vec2)` returns
/// both totals. Mean and variance are such a pair, so the subgroup paths spend
/// a single barrier on the two of them.
const RED_PAIR_NONE: &str = r#"
var<workgroup> red: array<f32, 256u>;

fn wg_sum2(v: vec2<f32>, lid: u32) -> vec2<f32> {
    red[lid] = v.x;
    workgroupBarrier();
    for (var st = 128u; st >= 1u; st = st >> 1u) {
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
    }
    let a = red[0];
    // barrier: red[0] is read above and overwritten below
    workgroupBarrier();
    red[lid] = v.y;
    workgroupBarrier();
    for (var st = 128u; st >= 1u; st = st >> 1u) {
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
    }
    return vec2<f32>(a, red[0]);
}
"#;

/// The same pair folded by a fixed-width warp: `@N@` subgroups of
/// `256 / @N@` lanes, whose lane-0 partials one more `subgroupAdd` reduces.
const RED_PAIR_WARP: &str = r#"
var<workgroup> red0: array<f32, @N@u>;
var<workgroup> red1: array<f32, @N@u>;

fn wg_sum2(v: vec2<f32>, lid: u32, lane: u32) -> vec2<f32> {
    let a = subgroupAdd(v.x);
    let b = subgroupAdd(v.y);
    if (lane == 0u) {
        red0[lid @SHIFT@] = a;
        red1[lid @SHIFT@] = b;
    }
    workgroupBarrier();
    return vec2<f32>(
        subgroupAdd(select(0.0, red0[min(lane, @TOP@u)], lane < @N@u)),
        subgroupAdd(select(0.0, red1[min(lane, @TOP@u)], lane < @N@u)));
}
"#;

/// The same pair with the width read from `@builtin(subgroup_size)`.
const RED_PAIR_GENERIC: &str = r#"
var<workgroup> red0: array<f32, 64u>;
var<workgroup> red1: array<f32, 64u>;

fn wg_sum2(v: vec2<f32>, lid: u32, lane: u32, sz: u32) -> vec2<f32> {
    let a = subgroupAdd(v.x);
    let b = subgroupAdd(v.y);
    let sg_id = lid / sz;
    let n_sg = (256u + sz - 1u) / sz;
    if (lane == 0u) {
        red0[sg_id] = a;
        red1[sg_id] = b;
    }
    workgroupBarrier();
    var sa = 0.0;
    var sb = 0.0;
    // each lane takes the partials `lane`, `lane + sz`, ... ; the subgroup
    // op below is past the loop, so it runs in uniform control flow
    for (var j = lane; j < n_sg; j = j + sz) {
        sa = sa + red0[j];
        sb = sb + red1[j];
    }
    return vec2<f32>(subgroupAdd(sa), subgroupAdd(sb));
}
"#;

fn red_pair(sg: SubgroupMode) -> String {
    match sg {
        SubgroupMode::None => RED_PAIR_NONE.to_string(),
        SubgroupMode::Warp32 => RED_PAIR_WARP
            .replace("@N@", "8")
            .replace("@TOP@", "7")
            .replace("@SHIFT@", ">> 5u"),
        SubgroupMode::Warp64 => RED_PAIR_WARP
            .replace("@N@", "4")
            .replace("@TOP@", "3")
            .replace("@SHIFT@", ">> 6u"),
        SubgroupMode::Generic => RED_PAIR_GENERIC.to_string(),
    }
}

/// A max followed by a sum that depends on it: two reductions that cannot
/// share a barrier. `wg_max` and `wg_sum` use separate slots so the second
/// write needs no barrier of its own.
const RED_MAXSUM_NONE: &str = r#"
var<workgroup> red: array<f32, 256u>;

fn wg_max(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    for (var st = 128u; st >= 1u; st = st >> 1u) {
        if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    // barrier: red[0] is read above and overwritten by the sum step
    workgroupBarrier();
    return m;
}

fn wg_sum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    for (var st = 128u; st >= 1u; st = st >> 1u) {
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
    }
    return red[0];
}
"#;

const RED_MAXSUM_WARP: &str = r#"
var<workgroup> red0: array<f32, @N@u>;
var<workgroup> red1: array<f32, @N@u>;

fn wg_max(v: f32, lid: u32, lane: u32) -> f32 {
    let p = subgroupMax(v);
    if (lane == 0u) { red0[lid @SHIFT@] = p; }
    workgroupBarrier();
    return subgroupMax(select(-3.4e38, red0[min(lane, @TOP@u)], lane < @N@u));
}

fn wg_sum(v: f32, lid: u32, lane: u32) -> f32 {
    let p = subgroupAdd(v);
    if (lane == 0u) { red1[lid @SHIFT@] = p; }
    workgroupBarrier();
    return subgroupAdd(select(0.0, red1[min(lane, @TOP@u)], lane < @N@u));
}
"#;

const RED_MAXSUM_GENERIC: &str = r#"
var<workgroup> red0: array<f32, 64u>;
var<workgroup> red1: array<f32, 64u>;

fn wg_max(v: f32, lid: u32, lane: u32, sz: u32) -> f32 {
    let p = subgroupMax(v);
    let sg_id = lid / sz;
    let n_sg = (256u + sz - 1u) / sz;
    if (lane == 0u) { red0[sg_id] = p; }
    workgroupBarrier();
    var acc = -3.4e38;
    for (var j = lane; j < n_sg; j = j + sz) { acc = max(acc, red0[j]); }
    return subgroupMax(acc);
}

fn wg_sum(v: f32, lid: u32, lane: u32, sz: u32) -> f32 {
    let p = subgroupAdd(v);
    let sg_id = lid / sz;
    let n_sg = (256u + sz - 1u) / sz;
    if (lane == 0u) { red1[sg_id] = p; }
    workgroupBarrier();
    var acc = 0.0;
    for (var j = lane; j < n_sg; j = j + sz) { acc = acc + red1[j]; }
    return subgroupAdd(acc);
}
"#;

fn red_max_sum(sg: SubgroupMode) -> String {
    match sg {
        SubgroupMode::None => RED_MAXSUM_NONE.to_string(),
        SubgroupMode::Warp32 => RED_MAXSUM_WARP
            .replace("@N@", "8")
            .replace("@TOP@", "7")
            .replace("@SHIFT@", ">> 5u"),
        SubgroupMode::Warp64 => RED_MAXSUM_WARP
            .replace("@N@", "4")
            .replace("@TOP@", "3")
            .replace("@SHIFT@", ">> 6u"),
        SubgroupMode::Generic => RED_MAXSUM_GENERIC.to_string(),
    }
}

fn emit_ln(sg: SubgroupMode, bindings: &str, load: &str, store: &str) -> String {
    let mut s = String::from("struct Cfg { rows: u32, cols: u32, eps_x_1e6: u32, do_gelu: u32 }\n");
    s.push_str(bindings);
    s.push('\n');
    s.push_str(&red_pair(sg));
    s.push_str(
        r#"
fn erf(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * exp(-a * a);
    return s * y;
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf(x / 1.4142135623730951));
}

fn gelu4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>@BUILTINS@) {
    // y stacks past the 65535 workgroup limit (conv0 of a 34 s chunk is ~1e5 rows).
    let row = wid.x + wid.y * 65535u;
    if (row >= cfg.rows) { return; }
    let cv = cfg.cols / 4u;
    let idx = row * cv + lid.x;
    var v = vec4<f32>(0.0);
    if (lid.x < cv) { v = @LOAD@; }
    // ---- mean and biased variance E[x^2] - mean^2 in one pass
    let sr = wg_sum2(vec2<f32>(v.x + v.y + v.z + v.w, dot(v, v))@ARGS@);
    let mean = sr.x / f32(cfg.cols);
    // a silent or constant row has variance 0 in exact arithmetic; f32
    // cancellation can leave a negative residue, and sqrt() of it is a NaN
    let varian = max(sr.y / f32(cfg.cols) - mean * mean, 0.0);
    let eps = f32(cfg.eps_x_1e6) * 1e-6;
    let inv = 1.0 / sqrt(varian + eps);
    if (lid.x < cv) {
        var o = (v - vec4<f32>(mean)) * inv * w4[lid.x] + b4[lid.x];
        if (cfg.do_gelu == 1u) { o = gelu4(o); }
        @STORE@ = o;
    }
}
"#,
    );
    s.replace("@BUILTINS@", red_builtins(sg))
        .replace("@ARGS@", red_args(sg))
        .replace("@LOAD@", load)
        .replace("@STORE@", store)
}

/// `out[i] = a[i] + b[i]`
pub(crate) fn add() -> String {
    r#"
struct Cfg { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // the host folds workgroups across x and y, because n = t*hidden and one
    // dimension caps at 65535 workgroups (an unchunked run over >341 s of
    // audio reaches 16k+ frames); see `flat_grid` in wav2vec2_gpu.rs
    let i = (wg.x + wg.y * 65535u) * 256u + lid.x;
    if (i < cfg.n) { o[i] = a[i] + b[i]; }
}
"#
    .to_string()
}

/// Row softmax over `cols`, in place. One workgroup per row.
///
/// Columns up to 2048 stay in named registers: one coalesced load, the same
/// strided sum order as the scalar loop, then one store. Longer rows fall
/// back to the three-pass loop.
///
/// Rows start every `cfg.stride` elements, `cfg.cols` apart: the caller pads
/// the row pitch up to a multiple of 4 so the attention GEMMs can read it as
/// `vec4`s. The padding columns are never read or written here.
pub(crate) fn softmax(sg: SubgroupMode) -> String {
    let mut s = String::from(
        r#"struct Cfg { rows: u32, cols: u32, stride: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;
"#,
    );
    s.push_str(&red_max_sum(sg));
    s.push_str(
        r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>@BUILTINS@) {
    let row = wid.x + wid.y * 65535u;
    if (row >= cfg.rows) { return; }
    let base = row * cfg.stride;
    let cols = cfg.cols;
    if (cols <= 2048u) {
        let i0 = lid.x;
        let i1 = lid.x + 256u;
        let i2 = lid.x + 512u;
        let i3 = lid.x + 768u;
        let i4 = lid.x + 1024u;
        let i5 = lid.x + 1280u;
        let i6 = lid.x + 1536u;
        let i7 = lid.x + 1792u;
        var x0 = -3.4e38;
        var x1 = -3.4e38;
        var x2 = -3.4e38;
        var x3 = -3.4e38;
        var x4 = -3.4e38;
        var x5 = -3.4e38;
        var x6 = -3.4e38;
        var x7 = -3.4e38;
        if (i0 < cols) { x0 = h[base + i0]; }
        if (i1 < cols) { x1 = h[base + i1]; }
        if (i2 < cols) { x2 = h[base + i2]; }
        if (i3 < cols) { x3 = h[base + i3]; }
        if (i4 < cols) { x4 = h[base + i4]; }
        if (i5 < cols) { x5 = h[base + i5]; }
        if (i6 < cols) { x6 = h[base + i6]; }
        if (i7 < cols) { x7 = h[base + i7]; }
        var mx = x0;
        mx = max(mx, x1);
        mx = max(mx, x2);
        mx = max(mx, x3);
        mx = max(mx, x4);
        mx = max(mx, x5);
        mx = max(mx, x6);
        mx = max(mx, x7);
        let m = wg_max(mx@ARGS@);
        var e0 = 0.0;
        var e1 = 0.0;
        var e2 = 0.0;
        var e3 = 0.0;
        var e4 = 0.0;
        var e5 = 0.0;
        var e6 = 0.0;
        var e7 = 0.0;
        if (i0 < cols) { e0 = exp(x0 - m); }
        if (i1 < cols) { e1 = exp(x1 - m); }
        if (i2 < cols) { e2 = exp(x2 - m); }
        if (i3 < cols) { e3 = exp(x3 - m); }
        if (i4 < cols) { e4 = exp(x4 - m); }
        if (i5 < cols) { e5 = exp(x5 - m); }
        if (i6 < cols) { e6 = exp(x6 - m); }
        if (i7 < cols) { e7 = exp(x7 - m); }
        var sum = e0;
        sum = sum + e1;
        sum = sum + e2;
        sum = sum + e3;
        sum = sum + e4;
        sum = sum + e5;
        sum = sum + e6;
        sum = sum + e7;
        let inv = 1.0 / wg_sum(sum@ARGS@);
        if (i0 < cols) { h[base + i0] = e0 * inv; }
        if (i1 < cols) { h[base + i1] = e1 * inv; }
        if (i2 < cols) { h[base + i2] = e2 * inv; }
        if (i3 < cols) { h[base + i3] = e3 * inv; }
        if (i4 < cols) { h[base + i4] = e4 * inv; }
        if (i5 < cols) { h[base + i5] = e5 * inv; }
        if (i6 < cols) { h[base + i6] = e6 * inv; }
        if (i7 < cols) { h[base + i7] = e7 * inv; }
    } else {
        var mx = -3.4e38;
        for (var i = lid.x; i < cols; i = i + 256u) {
            mx = max(mx, h[base + i]);
        }
        let m = wg_max(mx@ARGS@);
        var sum = 0.0;
        for (var i = lid.x; i < cols; i = i + 256u) {
            let e = exp(h[base + i] - m);
            h[base + i] = e;
            sum = sum + e;
        }
        let inv = 1.0 / wg_sum(sum@ARGS@);
        for (var i = lid.x; i < cols; i = i + 256u) {
            h[base + i] = h[base + i] * inv;
        }
    }
}
"#,
    );
    s.replace("@BUILTINS@", red_builtins(sg))
        .replace("@ARGS@", red_args(sg))
}

/// Row log_softmax over `cols` (two-pass, in-place).
pub(crate) fn log_softmax(sg: SubgroupMode) -> String {
    let mut s = String::from(
        r#"struct Cfg { rows: u32, cols: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;
"#,
    );
    s.push_str(&red_max_sum(sg));
    s.push_str(
        r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>@BUILTINS@) {
    let row = wid.x + wid.y * 65535u;
    if (row >= cfg.rows) { return; }
    let base = row * cfg.cols;
    let cols = cfg.cols;
    var mx = -3.4e38;
    for (var i = lid.x; i < cols; i = i + 256u) {
        mx = max(mx, h[base + i]);
    }
    let m = wg_max(mx@ARGS@);
    var sum = 0.0;
    for (var i = lid.x; i < cols; i = i + 256u) {
        sum = sum + exp(h[base + i] - m);
    }
    let lsum = log(wg_sum(sum@ARGS@));
    for (var i = lid.x; i < cols; i = i + 256u) {
        h[base + i] = h[base + i] - m - lsum;
    }
}
"#,
    );
    s.replace("@BUILTINS@", red_builtins(sg))
        .replace("@ARGS@", red_args(sg))
}

/// Gather the expanded trellis labels' log-probs: out[t * S + s] =
/// lp[t * V + labels[s]].  The forced-alignment Viterbi only ever reads
/// those T*S values, so the GPU downloads ~30 KB instead of the whole
/// T*10288 matrix (70 MB for a 34 s chunk).
///
/// `star[s] == 1` marks the `<star>` sentinel's states: their score is the
/// reference's appended zero column, a constant (-1.0 here — see
/// `CTC_STAR_SCORE` in `align_inference`), not a real log-probability.
/// Stamping them here replaces a host loop that walked every (frame, star)
/// cell per chunk — column-major scattered writes the GPU does for free
/// while gathering.
pub(crate) fn gather() -> String {
    r#"
struct Cfg { total: u32, s: u32, v: u32, _p0: u32 }
@group(0) @binding(0) var<storage, read> lp: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Cfg;
@group(0) @binding(4) var<storage, read> star: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // the host folds workgroups across x and y (one dimension caps at 65535);
    // see `flat_grid` in wav2vec2_gpu.rs
    let i = (wg.x + wg.y * 65535u) * 256u + lid.x;
    if (i < cfg.total) {
        let t = i / cfg.s;
        let s = i % cfg.s;
        let lp = lp[t * cfg.v + labels[s]];
        out[i] = select(lp, -1.0, star[s] == 1u);
    }
}
"#
    .to_string()
}

/// (rows, cols) -> (cols, rows), into rows of `o_stride` elements.
///
/// The output pitch is the caller's: the attention GEMM reads `K^T` with
/// `vec4` loads, which needs every output row 16-byte aligned, so the host
/// pads the pitch to a multiple of 4 and the columns `rows..o_stride` keep
/// nothing the model reads.
pub(crate) fn transpose() -> String {
    r#"
struct Cfg {
    rows: u32, cols: u32, a_stride: u32, a_off: u32,
    o_stride: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if (r < cfg.rows && c < cfg.cols) {
        o[c * cfg.o_stride + r] = a[cfg.a_off + r * cfg.a_stride + c];
    }
}
"#
    .to_string()
}

/// conv layer 0: raw waveform (1 channel) -> (T_out, 512).
pub(crate) fn conv0() -> String {
    r#"
struct Cfg { t_out: u32, c: u32, k: u32, stride: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;   // (n,)
@group(0) @binding(1) var<storage, read> w: array<f32>;   // (c, k)
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>; // (t_out, c)
@group(0) @binding(4) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let c = gid.y;
    if (j >= cfg.t_out || c >= cfg.c) { return; }
    var acc = bias[c];
    let base = j * cfg.stride;
    for (var t = 0u; t < cfg.k; t = t + 1u) {
        acc = acc + x[base + t] * w[c * cfg.k + t];
    }
    o[j * cfg.c + c] = acc;
}
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{DeviceSelector, Gpu};

    /// Every shape the row kernels can be built for.
    const MODES: [SubgroupMode; 4] = [
        SubgroupMode::None,
        SubgroupMode::Warp32,
        SubgroupMode::Warp64,
        SubgroupMode::Generic,
    ];

    /// WebGPU's general workgroup-storage ceiling, for the pure structural
    /// checks. The GPU test passes the adapter's own limit instead.
    const TEST_STORAGE: u32 = 48 * 1024;

    #[test]
    fn subgroup_mode_follows_the_adapter_range() {
        // No feature, or a range the driver left at zero: no subgroup path.
        assert_eq!(SubgroupMode::from_limits(false, 32, 32), SubgroupMode::None);
        assert_eq!(SubgroupMode::from_limits(false, 0, 0), SubgroupMode::None);
        assert_eq!(SubgroupMode::from_limits(true, 0, 32), SubgroupMode::None);
        assert_eq!(SubgroupMode::from_limits(true, 32, 0), SubgroupMode::None);
        // NVIDIA / RDNA: one fixed width.
        assert_eq!(
            SubgroupMode::from_limits(true, 32, 32),
            SubgroupMode::Warp32
        );
        // AMD GCN/Vega: 64-wide.
        assert_eq!(
            SubgroupMode::from_limits(true, 64, 64),
            SubgroupMode::Warp64
        );
        // Intel and anything that reports a range: read the width at runtime.
        assert_eq!(
            SubgroupMode::from_limits(true, 8, 32),
            SubgroupMode::Generic
        );
        assert_eq!(
            SubgroupMode::from_limits(true, 32, 64),
            SubgroupMode::Generic
        );
    }

    /// The row kernels, built for `sg`.
    fn row_kernels(sg: SubgroupMode) -> [(&'static str, String); 4] {
        [
            ("ln", layernorm(sg)),
            ("ln_sd", layernorm_sd(sg)),
            ("softmax", softmax(sg)),
            ("log_softmax", log_softmax(sg)),
        ]
    }

    #[test]
    fn no_row_kernel_enables_subgroups() {
        // Naga rejects `enable subgroups;` outright — the subgroup builtins
        // arrive with the device feature, and the directive breaks the module.
        for sg in MODES {
            for (what, src) in row_kernels(sg) {
                assert!(!src.contains("enable"), "{what}({sg:?}): {src}");
                if sg == SubgroupMode::None {
                    // the fallback stays the shared-memory tree, with no
                    // subgroup builtin anywhere
                    assert!(src.contains("for (var st = 128u"), "{what}({sg:?})");
                    assert!(!src.contains("subgroup"), "{what}({sg:?})");
                    continue;
                }
                assert!(
                    src.contains("@builtin(subgroup_invocation_id) sg_lane: u32"),
                    "{what}({sg:?})"
                );
                // Uniform control flow: no subgroup op may share a line with
                // a lane-dependent branch. The only lane test in these
                // kernels guards a workgroup store, never an op.
                for line in src.lines() {
                    if line.contains("subgroupAdd") || line.contains("subgroupMax") {
                        assert!(
                            !line.trim_start().starts_with("if "),
                            "{what}({sg:?}) folds under a lane guard: {line}"
                        );
                    }
                }
                // `select` evaluates both arms, so the workgroup read is
                // clamped: no lane reads past the last partial
                match sg {
                    SubgroupMode::Warp32 => {
                        assert!(src.contains("red0[min(lane, 7u)]"), "{what}: {src}");
                        assert!(src.contains("array<f32, 8u>"), "{what}: {src}");
                    }
                    SubgroupMode::Warp64 => {
                        assert!(src.contains("red0[min(lane, 3u)]"), "{what}: {src}");
                        assert!(src.contains("array<f32, 4u>"), "{what}: {src}");
                    }
                    SubgroupMode::Generic => {
                        assert!(src.contains("array<f32, 64u>"), "{what}: {src}");
                        assert!(
                            src.contains("@builtin(subgroup_size) sg_size: u32"),
                            "{what}"
                        );
                        // the strided sum runs under a lane-varying trip count,
                        // so the op that folds it comes after the loop
                        assert!(
                            src.contains("for (var j = lane; j < n_sg;"),
                            "{what}: {src}"
                        );
                    }
                    SubgroupMode::None => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn layernorm_clamps_the_variance_before_the_sqrt() {
        // A silent or constant row has zero variance; f32 cancellation can
        // leave it slightly negative, and sqrt() of that is a NaN row.
        for sg in MODES {
            for (what, src) in [("ln", layernorm(sg)), ("ln_sd", layernorm_sd(sg))] {
                assert!(
                    src.contains("let varian = max(sr.y / f32(cfg.cols) - mean * mean, 0.0);"),
                    "{what}({sg:?}): {src}"
                );
            }
        }
    }

    #[test]
    fn conv_gemm_hoists_the_im2col_tap_out_of_the_row_loop() {
        for variant in GemmVariant::ALL {
            for src in [
                conv_gemm_variant(variant, TEST_STORAGE),
                conv_gemm(TEST_STORAGE),
            ] {
                assert!(src.contains("let tap0 = k0 / d.c_in;"), "{variant:?}");
                assert!(
                    src.contains("let single = (k0 + ") && src.contains("/ d.c_in == tap0;"),
                    "{variant:?}"
                );
                assert!(src.contains("let last_src = (m0 + "), "{variant:?}");
                assert!(src.contains("src * c_in4 + ci0_4 + kvi"), "{variant:?}");
                assert!(
                    src.contains("d.c_in % 4u == 0u") && src.contains("ci0 % 4u == 0u"),
                    "{variant:?}"
                );
                // the guarded gather still covers the tiles the fast path
                // refuses (partial edges, unaligned taps)
                assert!(src.contains("let tl = kl / d.c_in;"), "{variant:?}");
                assert!(src.contains("workgroupBarrier"), "{variant:?}");
                // single- and double-buffered bodies must stay balanced
                let mut depth = 0i32;
                for c in src.chars() {
                    match c {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
                    assert!(depth >= 0, "{variant:?} closes early");
                }
                assert_eq!(depth, 0, "{variant:?}");
            }
        }
        // the plain GEMM never carries the im2col path
        assert!(!gemm_variant(GemmVariant::Base, TEST_STORAGE).contains("tap0"));
        assert!(!gemm_bias(TEST_STORAGE).contains("tap0"));
    }

    // -----------------------------------------------------------------------
    // The one test that opens a GPU.
    //
    // `cargo test` runs tests in parallel threads and this machine's driver
    // does not survive two Vulkan devices in one process, so every device
    // check lives in the single `#[test]` below: it opens exactly one device,
    // on Vulkan (DX12 hangs on this machine).
    // -----------------------------------------------------------------------

    fn vulkan() -> Option<Gpu> {
        match pollster::block_on(Gpu::new_with(DeviceSelector::Runtime {
            api: wgpu::Backend::Vulkan,
            index: 0,
        })) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                eprintln!("skipping the GPU shader checks: no Vulkan adapter ({e:#})");
                None
            }
        }
    }

    /// Deterministic pseudo-noise in [-1, 1): one LCG step per index, so any
    /// element can be rebuilt without carrying an RNG through the test.
    fn noise(i: usize) -> f32 {
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((x >> 40) & 0xffff) as f32 / 32768.0) - 1.0
    }

    fn storage_f32(gpu: &Gpu, label: &str, data: &[f32]) -> wgpu::Buffer {
        let b = gpu.storage(label, (data.len() * 4) as u64);
        gpu.upload(&b, bytemuck::cast_slice(data));
        b
    }

    fn uniform_u32(gpu: &Gpu, label: &str, words: &[u32]) -> wgpu::Buffer {
        let b = gpu.uniform(label, (words.len() * 4) as u64);
        gpu.upload(&b, bytemuck::cast_slice(words));
        b
    }

    fn bind<'a>(i: u32, b: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
        wgpu::BindGroupEntry {
            binding: i,
            resource: b.as_entire_binding(),
        }
    }

    fn run(
        gpu: &Gpu,
        pipe: &wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry],
        gx: u32,
        gy: u32,
    ) {
        let layout = pipe.get_bind_group_layout(0);
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries,
        });
        let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        gpu.queue.submit([enc.finish()]);
        gpu.flush().expect("dispatch flush");
        if let Some(e) = pollster::block_on(guard.pop()) {
            panic!("dispatch rejected: {e}");
        }
    }

    fn read(gpu: &Gpu, b: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let bytes = gpu.readback(b, (n * 4) as u64).expect("readback");
        assert_eq!(bytes.len(), n * 4);
        let mut out = Vec::with_capacity(n);
        for c in bytes.chunks(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        out
    }

    fn pipeline(gpu: &Gpu, label: &str, src: &str) -> wgpu::ComputePipeline {
        gpu.pipeline(label, src, "main", None)
            .unwrap_or_else(|e| panic!("{label} did not compile or validate: {e:#}"))
    }

    fn gemm_pipeline(gpu: &Gpu, label: &str, src: &str) -> wgpu::ComputePipeline {
        gpu.pipeline_no_zero(label, src, "main", None)
            .unwrap_or_else(|e| panic!("{label} did not compile or validate: {e:#}"))
    }

    /// Three rows: random, all-zero, and a constant that is not exactly
    /// representable — the two constant rows are what a negative variance
    /// would turn into NaN.
    fn ln_rows(rows: usize, cols: usize) -> Vec<f32> {
        let mut d: Vec<f32> = (0..rows * cols).map(|i| noise(i) + 0.4).collect();
        for c in 0..cols {
            d[cols + c] = 0.0;
            d[2 * cols + c] = 0.1;
        }
        d
    }

    fn ln_affine(cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w = (0..cols).map(|c| 1.0 + 0.25 * noise(900_001 + c)).collect();
        let b = (0..cols).map(|c| 0.125 * noise(424_242 + c)).collect();
        (w, b)
    }

    fn ln_in_place(gpu: &Gpu, mode: SubgroupMode, rows: usize, cols: usize) -> Vec<f32> {
        let pipe = pipeline(gpu, &format!("t-ln-{mode:?}"), &layernorm(mode));
        let (w, b) = ln_affine(cols);
        let h = storage_f32(gpu, "t-ln-h", &ln_rows(rows, cols));
        let wb = storage_f32(gpu, "t-ln-w", &w);
        let bb = storage_f32(gpu, "t-ln-b", &b);
        let uni = uniform_u32(gpu, "t-ln-u", &[rows as u32, cols as u32, 10, 0]);
        run(
            gpu,
            &pipe,
            &[bind(0, &h), bind(1, &wb), bind(2, &bb), bind(3, &uni)],
            rows as u32,
            1,
        );
        read(gpu, &h, rows * cols)
    }

    /// `(src after the run, dst)`: the source/destination variant must leave
    /// the residual stream it reads untouched.
    fn ln_sd(gpu: &Gpu, mode: SubgroupMode, rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let pipe = pipeline(gpu, &format!("t-lns-{mode:?}"), &layernorm_sd(mode));
        let (w, b) = ln_affine(cols);
        let src = storage_f32(gpu, "t-lns-s", &ln_rows(rows, cols));
        let dst = storage_f32(gpu, "t-lns-d", &vec![9.0f32; rows * cols]);
        let wb = storage_f32(gpu, "t-lns-w", &w);
        let bb = storage_f32(gpu, "t-lns-b", &b);
        let uni = uniform_u32(gpu, "t-lns-u", &[rows as u32, cols as u32, 10, 0]);
        run(
            gpu,
            &pipe,
            &[
                bind(0, &src),
                bind(1, &dst),
                bind(2, &wb),
                bind(3, &bb),
                bind(4, &uni),
            ],
            rows as u32,
            1,
        );
        (read(gpu, &src, rows * cols), read(gpu, &dst, rows * cols))
    }

    fn softmax_run(
        gpu: &Gpu,
        mode: SubgroupMode,
        rows: usize,
        cols: usize,
        stride: usize,
        data: &[f32],
    ) -> Vec<f32> {
        let pipe = pipeline(gpu, &format!("t-sm-{mode:?}"), &softmax(mode));
        let h = storage_f32(gpu, "t-sm-h", data);
        let uni = uniform_u32(gpu, "t-sm-u", &[rows as u32, cols as u32, stride as u32, 0]);
        run(gpu, &pipe, &[bind(0, &h), bind(1, &uni)], rows as u32, 1);
        read(gpu, &h, rows * stride)
    }

    fn log_softmax_run(
        gpu: &Gpu,
        mode: SubgroupMode,
        rows: usize,
        cols: usize,
        data: &[f32],
    ) -> Vec<f32> {
        let pipe = pipeline(gpu, &format!("t-lsm-{mode:?}"), &log_softmax(mode));
        let h = storage_f32(gpu, "t-lsm-h", data);
        let uni = uniform_u32(gpu, "t-lsm-u", &[rows as u32, cols as u32, 0, 0]);
        run(gpu, &pipe, &[bind(0, &h), bind(1, &uni)], rows as u32, 1);
        read(gpu, &h, rows * cols)
    }

    #[test]
    fn vulkan_shaders_compile_and_the_subgroup_folds_match_the_tree() {
        let Some(gpu) = vulkan() else { return };
        eprintln!("[shaders] {}", gpu.describe());
        let max_storage = gpu.limits().max_compute_workgroup_storage_size;
        // The device's own shape plus every shape it can build: a kernel that
        // only ever ran as `None` would still be compiled by nobody.
        let mut modes: Vec<SubgroupMode> = Vec::new();
        for m in MODES {
            if m == SubgroupMode::None || gpu.has_subgroups() {
                modes.push(m);
            }
        }
        assert_eq!(modes[0], SubgroupMode::None);
        let device_mode = gpu.subgroup_mode();
        if device_mode != SubgroupMode::None {
            assert!(modes.contains(&device_mode), "{device_mode:?} not built");
        }
        // Numerically comparable: `None` against the shape this device runs
        // and against `Generic`, whose width comes from the device at
        // runtime. `Warp64` folds 4 subgroups of 64 lanes and only means
        // anything on a 64-wide adapter, so it is compiled here and checked
        // against the tree wherever such a device runs this test.
        let mut numeric: Vec<SubgroupMode> = Vec::new();
        if gpu.has_subgroups() {
            numeric.push(SubgroupMode::Generic);
            if device_mode != SubgroupMode::None && device_mode != SubgroupMode::Generic {
                numeric.push(device_mode);
            }
        }

        // ---- every shape compiles and passes uniformity validation
        for &m in &modes {
            for (what, src) in row_kernels(m) {
                let _ = pipeline(&gpu, &format!("t-{what}-{m:?}"), &src);
            }
        }

        // ---- transpose writes into the padded output pitch
        {
            let tpipe = pipeline(&gpu, "t-transpose", &transpose());
            let (rows, cols, a_stride, a_off, o_stride) = (5usize, 3usize, 7usize, 2usize, 8usize);
            let a: Vec<f32> = (0..a_off + rows * a_stride)
                .map(|i| i as f32 + 0.5)
                .collect();
            let a_buf = storage_f32(&gpu, "t-tr-a", &a);
            let o_buf = storage_f32(&gpu, "t-tr-o", &vec![-1.0f32; cols * o_stride]);
            let uni = uniform_u32(
                &gpu,
                "t-tr-u",
                &[
                    rows as u32,
                    cols as u32,
                    a_stride as u32,
                    a_off as u32,
                    o_stride as u32,
                    0,
                    0,
                    0,
                ],
            );
            run(
                &gpu,
                &tpipe,
                &[bind(0, &a_buf), bind(1, &o_buf), bind(2, &uni)],
                (rows as u32).div_ceil(16),
                (cols as u32).div_ceil(16),
            );
            let o = read(&gpu, &o_buf, cols * o_stride);
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(
                        o[c * o_stride + r],
                        a[a_off + r * a_stride + c],
                        "transpose [{r},{c}]"
                    );
                }
            }
            for c in 0..cols {
                for r in rows..o_stride {
                    assert_eq!(
                        o[c * o_stride + r],
                        -1.0,
                        "transpose wrote padding [{c},{r}]"
                    );
                }
            }
        }

        // ---- layernorm, in place and source/destination
        let (rows, cols) = (3usize, 512usize);
        let base = ln_in_place(&gpu, SubgroupMode::None, rows, cols);
        for (i, v) in base.iter().enumerate() {
            assert!(v.is_finite(), "tree layernorm [{i}] = {v}");
        }
        for &m in &numeric {
            let got = ln_in_place(&gpu, m, rows, cols);
            for (i, (want, have)) in base.iter().zip(&got).enumerate() {
                assert!(have.is_finite(), "layernorm {m:?} [{i}] = {have}");
                assert!(
                    (want - have).abs() <= 1e-5,
                    "layernorm {m:?} [{i}]: tree {want} vs {have}"
                );
            }
        }
        let (base_src, base_dst) = ln_sd(&gpu, SubgroupMode::None, rows, cols);
        for &m in &numeric {
            let (src, dst) = ln_sd(&gpu, m, rows, cols);
            for (i, ((want, have), keep)) in base_dst.iter().zip(&dst).zip(&src).enumerate() {
                assert!(have.is_finite(), "layernorm_sd {m:?} [{i}] = {have}");
                assert!(
                    (want - have).abs() <= 1e-5,
                    "layernorm_sd {m:?} [{i}]: tree {want} vs {have}"
                );
                assert_eq!(*keep, base_src[i], "layernorm_sd {m:?} wrote src [{i}]");
            }
        }

        // ---- softmax, with the row pitch wider than the row
        for (cols, stride) in [(1699usize, 1700usize), (2051, 2052)] {
            let rows = 3usize;
            let mut data: Vec<f32> = (0..rows * stride).map(|i| 6.0 * noise(i) - 3.0).collect();
            // sentinel in the padding: a row shorter than its pitch must
            // leave the tail exactly as it found it
            for r in 0..rows {
                for c in cols..stride {
                    data[r * stride + c] = 7.0;
                }
            }
            let compact: Vec<f32> = (0..rows)
                .flat_map(|r| data[r * stride..r * stride + cols].to_vec())
                .collect();
            let base = softmax_run(&gpu, SubgroupMode::None, rows, cols, cols, &compact);
            for &m in &numeric {
                let got = softmax_run(&gpu, m, rows, cols, stride, &data);
                for r in 0..rows {
                    for c in 0..cols {
                        let want = base[r * cols + c];
                        let have = got[r * stride + c];
                        assert!(
                            (want - have).abs() <= 1e-5,
                            "softmax {m:?} cols={cols} stride={stride} [{r},{c}]: \
                             {want} vs {have}"
                        );
                    }
                    for c in cols..stride {
                        assert_eq!(
                            got[r * stride + c],
                            7.0,
                            "softmax {m:?} cols={cols} wrote padding [{r},{c}]"
                        );
                    }
                    let sum: f32 = (0..cols).map(|c| got[r * stride + c]).sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "softmax {m:?} cols={cols} row {r} sums to {sum}"
                    );
                }
            }
        }

        // ---- log_softmax over rows longer than one thread step
        {
            let (rows, cols) = (2usize, 1027usize);
            let data: Vec<f32> = (0..rows * cols).map(|i| 8.0 * noise(i)).collect();
            let base = log_softmax_run(&gpu, SubgroupMode::None, rows, cols, &data);
            for &m in &numeric {
                let got = log_softmax_run(&gpu, m, rows, cols, &data);
                for i in 0..data.len() {
                    assert!(
                        (base[i] - got[i]).abs() <= 1e-5,
                        "log_softmax {m:?} [{i}]: tree {} vs {}",
                        base[i],
                        got[i]
                    );
                }
                for r in 0..rows {
                    let sum: f32 = (0..cols).map(|c| got[r * cols + c].exp()).sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "log_softmax {m:?} row {r} exp-sums to {sum}"
                    );
                }
            }
        }

        // ---- conv_gemm: the tile-level im2col path against a host
        //      reference, at the tile this adapter actually picked
        {
            let variant = pick_gemm_variant(max_storage);
            let (mt, nt) = gemm_tile(variant);
            let (m, n, k, c_in, stride) = (mt as usize, nt as usize, 512usize, 512usize, 1usize);
            let t_in = m * stride;
            let x: Vec<f32> = (0..t_in * c_in).map(|i| noise(i) + 0.1).collect();
            let wt: Vec<f32> = (0..k * n).map(|i| 0.5 * noise(500_000 + i)).collect();
            let bias: Vec<f32> = (0..n).map(|i| 0.25 * noise(700_000 + i)).collect();
            let src = conv_gemm_variant(variant, max_storage);
            let pipe = gemm_pipeline(&gpu, &format!("t-conv-{variant:?}"), &src);
            let xb = storage_f32(&gpu, "t-conv-x", &x);
            let wb = storage_f32(&gpu, "t-conv-w", &wt);
            let bb = storage_f32(&gpu, "t-conv-b", &bias);
            let ob = storage_f32(&gpu, "t-conv-o", &vec![0f32; m * n]);
            let uni = uniform_u32(
                &gpu,
                "t-conv-u",
                &[
                    m as u32,
                    n as u32,
                    k as u32,
                    c_in as u32,
                    stride as u32,
                    t_in as u32,
                    1,
                    1,
                ],
            );
            run(
                &gpu,
                &pipe,
                &[
                    bind(0, &xb),
                    bind(1, &wb),
                    bind(2, &bb),
                    bind(3, &ob),
                    bind(4, &uni),
                    bind(5, &xb),
                    bind(6, &wb),
                ],
                (m as u32).div_ceil(mt),
                (n as u32).div_ceil(nt),
            );
            let got = read(&gpu, &ob, m * n);
            for mm in 0..m {
                for nn in 0..n {
                    let mut want = bias[nn];
                    for kk in 0..k {
                        let row = mm * stride + kk / c_in;
                        want += x[row * c_in + (kk % c_in)] * wt[kk * n + nn];
                    }
                    let have = got[mm * n + nn];
                    assert!(
                        (want - have).abs() <= 1e-4 * want.abs().max(1.0),
                        "conv_gemm [{mm},{nn}]: host {want} vs {have}"
                    );
                }
            }
        }

        // ---- the tiles this adapter cannot run still have to validate
        for variant in GemmVariant::ALL {
            if variant.shared_bytes() > max_storage {
                continue;
            }
            let _ = gemm_pipeline(
                &gpu,
                &format!("t-conv-{variant:?}"),
                &conv_gemm_variant(variant, max_storage),
            );
        }
    }
}
