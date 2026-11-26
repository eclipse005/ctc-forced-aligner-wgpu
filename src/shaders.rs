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
    conv_gemm, conv_gemm_variant, conv_gemm_variant_with, gemm_bias, gemm_bias_tiled,
    gemm_shared_bytes, gemm_tile, gemm_variant, gemm_variant_with, pick_gemm_variant,
    pick_gemm_variant_for, pick_gemm_variant_for_with, pick_gemm_variant_with, pos_conv,
    GemmVariant, KS, MT, NT, POS_KS, POS_MT, POS_NT,
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

/// (rows, cols) -> (cols, rows), into rows of `o_stride` elements starting
/// at column `o_off`.
///
/// The output pitch is the caller's: the attention GEMM reads `K^T` with
/// `vec4` loads, which needs every output row 16-byte aligned, so the host
/// pads the pitch to a multiple of 4 and the columns `rows..o_stride` keep
/// nothing the model reads. The offsets are what let the loader stream a
/// checkpoint through one staging buffer: `a_off` picks a row of the source
/// and `o_off` the columns the three fused `qkv` projections land in.
pub(crate) fn transpose() -> String {
    r#"
struct Cfg {
    rows: u32, cols: u32, a_stride: u32, a_off: u32,
    o_stride: u32, o_off: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if (r < cfg.rows && c < cfg.cols) {
        o[cfg.o_off + c * cfg.o_stride + r] = a[cfg.a_off + r * cfg.a_stride + c];
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

/// Query rows one FlashAttention workgroup owns. The 32 KiB build (64 queries,
/// 32 keys) is the fastest and runs wherever the adapter has the storage; the
/// 16 KiB build (32 queries, 16 keys) is what keeps the fused path alive down
/// to WebGPU's floor.
pub(crate) const FLASH_BQ_SMALL: u32 = 32;
pub(crate) const FLASH_BQ_LARGE: u32 = 64;
pub(crate) const FLASH_HEAD_DIM: u32 = 64;

/// Workgroup storage [`flash_attn`] asks for at `max_storage`: 32,768 B for
/// the 64x32 tile (`sq4` 16,384 + `skv4` 8,192 + `sp4` 8,192) and 14,336 B for
/// the 32x16 one (`sq4` 8,192 + `skv4` 4,096 + `sp2` 2,048). Both are exact:
/// every tile is padded only by the XOR swizzle, never by dead columns.
pub(crate) fn flash_attn_shared_bytes(max_storage: u32) -> u32 {
    if max_storage >= 32 * 1024 {
        64 * 16 * 16 + 32 * 16 * 16 + 64 * 8 * 16
    } else {
        32 * 16 * 16 + 16 * 16 * 16 + 32 * 8 * 8
    }
}

/// The cross-lane fold Phase B needs over the 8 `lx` lanes that share a query
/// row. With 8-wide subgroups it is three butterfly `subgroupShuffleXor`
/// steps; without them the P tile's own row is the scratch pad (zero extra
/// workgroup storage), which costs two barriers per fold.
///
/// `vars` are the per-row registers (`bm0..`, `bl0..`) and `sp` the P array:
/// the store, the read after it and the read's own barrier are what keep the
/// array from being both at once.
fn flash_fold(vars: &[&str], sp: &str, subgroup: bool, max: bool) -> String {
    let row = |m: usize| {
        if m == 0 {
            "ly".to_string()
        } else {
            format!("ly + {}u", 16 * m)
        }
    };
    let mut s = String::new();
    if subgroup {
        for v in vars {
            for mask in [4u32, 2, 1] {
                if max {
                    s.push_str(&format!(
                        "        {v} = max({v}, subgroupShuffleXor({v}, {mask}u));\n"
                    ));
                } else {
                    s.push_str(&format!(
                        "        {v} = {v} + subgroupShuffleXor({v}, {mask}u);\n"
                    ));
                }
            }
        }
        return s;
    }
    for (m, v) in vars.iter().enumerate() {
        s.push_str(&format!("        {sp}[({}) * 8u + lx].x = {v};\n", row(m)));
    }
    s.push_str("        workgroupBarrier();\n");
    for (m, v) in vars.iter().enumerate() {
        let (init, step) = if max {
            (
                "-3.4e38",
                format!("max({v}, {sp}[({}) * 8u + j].x)", row(m)),
            )
        } else {
            ("0.0", format!("{v} + {sp}[({}) * 8u + j].x", row(m)))
        };
        s.push_str(&format!("        {v} = {init};\n"));
        s.push_str("        for (var j = 0u; j < 8u; j = j + 1u) {\n");
        s.push_str(&format!("            {v} = {step};\n"));
        s.push_str("        }\n");
    }
    s.push_str("        workgroupBarrier();\n");
    s
}

/// Fused bidirectional attention: `softmax(q k^T * scale) v` per head, straight
/// from the packed `qkv` buffer and straight into the output.
///
/// The materialised path this replaces wrote a (heads, t, t_pad) scores tensor,
/// transposed K into a second buffer, read the scores back through a softmax
/// kernel and then through the PV GEMM. Here the scores never leave the
/// workgroup: a block of queries stays in shared memory across every key tile,
/// so `q` is read once from global and every `vec4` load feeds eight to eleven
/// FMAs.
///
/// Sizing is the adapter's call. `bq` is 64 queries x 32 keys (32,768 B of
/// workgroup storage) wherever the adapter has that much, and 32 x 16
/// (14,336 B) down to WebGPU's 16 KiB floor. The rest of this comment describes
/// the 64x32 build; the 32x16 one is the same kernel with half the rows, half
/// the keys per thread and `vec2` key groups.
///
/// ## Thread mapping
///
/// `@workgroup_size(128)` is deliberately 1-D: `lid.x` then maps linearly into
/// subgroups on every backend, so the 8 lanes `lx = lid.x & 7` that share a
/// query row (`ly = lid.x >> 3`) are one aligned 8-lane group whenever
/// `subgroup_min_size >= 8`, and the row reductions are three `shuffleXor`
/// steps with no barrier. Each thread owns 4 query rows (`ly + 16m`), 4
/// contiguous keys (`lx * 4 + n`) and 2 output `vec4` columns (`lx`, `lx + 8`),
/// i.e. 16 score accumulators and 8 output accumulators in registers.
///
/// ## Shared memory (32,768 B, no padding columns)
///
/// * `sq4[q * 16 + (d4 ^ (q & 7))]`: the 64 queries, transposed so a `vec4` is
///   4 head dimensions. Phase A reads it as an 8-lane broadcast (the 8 lanes
///   of a row group all want the same `vec4`), which is conflict-free with or
///   without the XOR: the swizzle is kept so all three tiles share one
///   indexing rule, and costs one XOR hoisted out of the `d4` loop.
/// * `skv4[k * 16 + (d4 ^ (k >> 2))]`: the 32-key tile, K first and then the
///   same slots refilled with V. `k >> 2` is 0..7 over the tile, so in phase A
///   (`k = lx * 4 + n`) the swizzle is `d4 ^ lx` for all four of a thread's
///   keys, and in phase C (`k = k4 * 4 + c`) it is `k4`, both hoisted out of
///   the inner loop. The 8 lanes of a row group then read 8 distinct 16-byte
///   banks and the other three quarters of the warp read the same 8 for their
///   own rows: conflict-free either way.
/// * `sp4[q * 8 + (k4 ^ (q & 3))]`: `exp(s - m)` as `vec4`s, one per 4 keys.
///   `(q & 3)` is `ly & 3` for all four rows, so the read side hoists it too.
///
/// ## Barriers
///
/// Four per 32 keys: K staged, phase A done (so V may overwrite it), V staged
/// with phase B folded in, phase C done (so the next K may overwrite it). V
/// loading overlaps the softmax, and the P tile is written in the same window
/// as the V tile, so both are covered by the one barrier between them. The
/// subgroup-less fold adds a barrier on each side of its two reads.
pub(crate) fn flash_attn(max_storage: u32, use_subgroup_8: bool) -> (String, u32) {
    let shared = flash_attn_shared_bytes(max_storage);
    assert!(
        shared <= max_storage,
        "flash attention needs {shared} B of workgroup storage, adapter has {max_storage}"
    );
    if max_storage >= 32 * 1024 {
        (flash_attn_large(use_subgroup_8), FLASH_BQ_LARGE)
    } else {
        (flash_attn_small(use_subgroup_8), FLASH_BQ_SMALL)
    }
}

/// The 64-query, 32-key build: 32,768 B of workgroup storage.
fn flash_attn_large(subgroup: bool) -> String {
    const BODY: &str = r#"
struct FlashAttnCfg {
    t: u32,
    heads: u32,
    hidden4: u32,
    _p0: u32,
    scale: f32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

// qkv as (t, 3 * hidden) viewed as vec4s: one row is 3 * hidden4 vec4s, and a
// head's 64 values are 16 consecutive vec4s.
@group(0) @binding(0) var<storage, read> qkv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out4: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> cfg: FlashAttnCfg;

var<workgroup> sq4: array<vec4<f32>, 1024u>;    // 64 x 16, 16,384 B
var<workgroup> skv4: array<vec4<f32>, 512u>;    // 32 x 16,  8,192 B
var<workgroup> sp4: array<vec4<f32>, 512u>;     // 64 x 8,   8,192 B

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x & 7u;
    let ly = lid.x >> 3u;
    // tile loader: 8 rows x 16 vec4 columns per pass
    let r0 = lid.x >> 4u;
    let c4 = lid.x & 15u;
    let head = wid.y;
    let q_base = wid.x * 64u;
    let row = 3u * cfg.hidden4;
    let q_off = head * 16u;
    let k_off = cfg.hidden4 + head * 16u;
    let v_off = 2u * cfg.hidden4 + head * 16u;

    // ---- prologue: this block's 64 queries, zero-padded past t
    for (var u = 0u; u < 8u; u = u + 1u) {
        let qr = r0 + u * 8u;
        let gq = q_base + qr;
        var qv = vec4<f32>(0.0);
        if (gq < cfg.t) {
            qv = qkv4[gq * row + q_off + c4];
        }
        sq4[qr * 16u + (c4 ^ (qr & 7u))] = qv;
    }

    var o00 = vec4<f32>(0.0);
    var o01 = vec4<f32>(0.0);
    var o10 = vec4<f32>(0.0);
    var o11 = vec4<f32>(0.0);
    var o20 = vec4<f32>(0.0);
    var o21 = vec4<f32>(0.0);
    var o30 = vec4<f32>(0.0);
    var o31 = vec4<f32>(0.0);
    var m0 = -3.4e38;
    var l0 = 0.0;
    var m1 = -3.4e38;
    var l1 = 0.0;
    var m2 = -3.4e38;
    var l2 = 0.0;
    var m3 = -3.4e38;
    var l3 = 0.0;

    // no barrier here: the K stage below is the first thing phase A reads
    // after, and it carries its own.
    let ktiles = (cfg.t + 31u) / 32u;
    for (var kt = 0u; kt < ktiles; kt = kt + 1u) {
        let k0 = kt * 32u;

        // ---- K tile, key-major and swizzled by key group
        for (var u = 0u; u < 4u; u = u + 1u) {
            let kr = r0 + u * 8u;
            let gk = k0 + kr;
            var kv = vec4<f32>(0.0);
            if (gk < cfg.t) {
                kv = qkv4[gk * row + k_off + c4];
            }
            skv4[kr * 16u + (c4 ^ (kr >> 2u))] = kv;
        }
        workgroupBarrier();

        // ---- A: 4 queries x 4 keys = 16 scores. `qs` is shared by the four
        //      rows ((q & 7) == (ly & 7)) and `ks` by the four keys
        //      ((k >> 2) == lx), so each d4 costs two XORs and 8 loads.
        //      The dots are spelled out one component at a time rather than
        //      written `dot(q, k)`: both reach the driver as `OpDot`, but the
        //      mul-add chain is what contracts into four FFMAs without the
        //      driver having to take `OpDot` apart (1.7% on a P104-100).
        var s00 = 0.0;
        var s01 = 0.0;
        var s02 = 0.0;
        var s03 = 0.0;
        var s10 = 0.0;
        var s11 = 0.0;
        var s12 = 0.0;
        var s13 = 0.0;
        var s20 = 0.0;
        var s21 = 0.0;
        var s22 = 0.0;
        var s23 = 0.0;
        var s30 = 0.0;
        var s31 = 0.0;
        var s32 = 0.0;
        var s33 = 0.0;
        for (var d4 = 0u; d4 < 16u; d4 = d4 + 1u) {
            let qs = d4 ^ (ly & 7u);
            let ks = d4 ^ lx;
            let kv0 = skv4[lx * 64u + ks];
            let kv1 = skv4[lx * 64u + 16u + ks];
            let kv2 = skv4[lx * 64u + 32u + ks];
            let kv3 = skv4[lx * 64u + 48u + ks];
            let qv0 = sq4[ly * 16u + qs];
            s00 = s00 + qv0.x * kv0.x;
            s00 = s00 + qv0.y * kv0.y;
            s00 = s00 + qv0.z * kv0.z;
            s00 = s00 + qv0.w * kv0.w;
            s01 = s01 + qv0.x * kv1.x;
            s01 = s01 + qv0.y * kv1.y;
            s01 = s01 + qv0.z * kv1.z;
            s01 = s01 + qv0.w * kv1.w;
            s02 = s02 + qv0.x * kv2.x;
            s02 = s02 + qv0.y * kv2.y;
            s02 = s02 + qv0.z * kv2.z;
            s02 = s02 + qv0.w * kv2.w;
            s03 = s03 + qv0.x * kv3.x;
            s03 = s03 + qv0.y * kv3.y;
            s03 = s03 + qv0.z * kv3.z;
            s03 = s03 + qv0.w * kv3.w;
            let qv1 = sq4[(ly + 16u) * 16u + qs];
            s10 = s10 + qv1.x * kv0.x;
            s10 = s10 + qv1.y * kv0.y;
            s10 = s10 + qv1.z * kv0.z;
            s10 = s10 + qv1.w * kv0.w;
            s11 = s11 + qv1.x * kv1.x;
            s11 = s11 + qv1.y * kv1.y;
            s11 = s11 + qv1.z * kv1.z;
            s11 = s11 + qv1.w * kv1.w;
            s12 = s12 + qv1.x * kv2.x;
            s12 = s12 + qv1.y * kv2.y;
            s12 = s12 + qv1.z * kv2.z;
            s12 = s12 + qv1.w * kv2.w;
            s13 = s13 + qv1.x * kv3.x;
            s13 = s13 + qv1.y * kv3.y;
            s13 = s13 + qv1.z * kv3.z;
            s13 = s13 + qv1.w * kv3.w;
            let qv2 = sq4[(ly + 32u) * 16u + qs];
            s20 = s20 + qv2.x * kv0.x;
            s20 = s20 + qv2.y * kv0.y;
            s20 = s20 + qv2.z * kv0.z;
            s20 = s20 + qv2.w * kv0.w;
            s21 = s21 + qv2.x * kv1.x;
            s21 = s21 + qv2.y * kv1.y;
            s21 = s21 + qv2.z * kv1.z;
            s21 = s21 + qv2.w * kv1.w;
            s22 = s22 + qv2.x * kv2.x;
            s22 = s22 + qv2.y * kv2.y;
            s22 = s22 + qv2.z * kv2.z;
            s22 = s22 + qv2.w * kv2.w;
            s23 = s23 + qv2.x * kv3.x;
            s23 = s23 + qv2.y * kv3.y;
            s23 = s23 + qv2.z * kv3.z;
            s23 = s23 + qv2.w * kv3.w;
            let qv3 = sq4[(ly + 48u) * 16u + qs];
            s30 = s30 + qv3.x * kv0.x;
            s30 = s30 + qv3.y * kv0.y;
            s30 = s30 + qv3.z * kv0.z;
            s30 = s30 + qv3.w * kv0.w;
            s31 = s31 + qv3.x * kv1.x;
            s31 = s31 + qv3.y * kv1.y;
            s31 = s31 + qv3.z * kv1.z;
            s31 = s31 + qv3.w * kv1.w;
            s32 = s32 + qv3.x * kv2.x;
            s32 = s32 + qv3.y * kv2.y;
            s32 = s32 + qv3.z * kv2.z;
            s32 = s32 + qv3.w * kv2.w;
            s33 = s33 + qv3.x * kv3.x;
            s33 = s33 + qv3.y * kv3.y;
            s33 = s33 + qv3.z * kv3.z;
            s33 = s33 + qv3.w * kv3.w;
        }
        workgroupBarrier();

        // ---- B: rescale, mask, row max over the 8 lanes, exp, row sum. The
        //      V tile streams in while this runs: it is the only barrier
        //      between the K reads of phase A and the V overwrite.
        for (var u = 0u; u < 4u; u = u + 1u) {
            let kr = r0 + u * 8u;
            let gk = k0 + kr;
            var vv = vec4<f32>(0.0);
            if (gk < cfg.t) {
                vv = qkv4[gk * row + v_off + c4];
            }
            skv4[kr * 16u + (c4 ^ (kr >> 2u))] = vv;
        }
        let kb = k0 + lx * 4u;
        let k0ok = kb < cfg.t;
        let k1ok = kb + 1u < cfg.t;
        let k2ok = kb + 2u < cfg.t;
        let k3ok = kb + 3u < cfg.t;
        s00 = select(-3.4e38, s00 * cfg.scale, k0ok);
        s01 = select(-3.4e38, s01 * cfg.scale, k1ok);
        s02 = select(-3.4e38, s02 * cfg.scale, k2ok);
        s03 = select(-3.4e38, s03 * cfg.scale, k3ok);
        s10 = select(-3.4e38, s10 * cfg.scale, k0ok);
        s11 = select(-3.4e38, s11 * cfg.scale, k1ok);
        s12 = select(-3.4e38, s12 * cfg.scale, k2ok);
        s13 = select(-3.4e38, s13 * cfg.scale, k3ok);
        s20 = select(-3.4e38, s20 * cfg.scale, k0ok);
        s21 = select(-3.4e38, s21 * cfg.scale, k1ok);
        s22 = select(-3.4e38, s22 * cfg.scale, k2ok);
        s23 = select(-3.4e38, s23 * cfg.scale, k3ok);
        s30 = select(-3.4e38, s30 * cfg.scale, k0ok);
        s31 = select(-3.4e38, s31 * cfg.scale, k1ok);
        s32 = select(-3.4e38, s32 * cfg.scale, k2ok);
        s33 = select(-3.4e38, s33 * cfg.scale, k3ok);
        var bm0 = max(max(s00, s01), max(s02, s03));
        var bm1 = max(max(s10, s11), max(s12, s13));
        var bm2 = max(max(s20, s21), max(s22, s23));
        var bm3 = max(max(s30, s31), max(s32, s33));
/*FOLD_MAX*/
        let mn0 = max(m0, bm0);
        let mn1 = max(m1, bm1);
        let mn2 = max(m2, bm2);
        let mn3 = max(m3, bm3);
        // a row with no prior key has no `exp` to carry: `select` gives it 0
        let a0 = select(0.0, exp(m0 - mn0), m0 > -1.0e37);
        let a1 = select(0.0, exp(m1 - mn1), m1 > -1.0e37);
        let a2 = select(0.0, exp(m2 - mn2), m2 > -1.0e37);
        let a3 = select(0.0, exp(m3 - mn3), m3 > -1.0e37);
        let p00 = select(0.0, exp(s00 - mn0), k0ok);
        let p01 = select(0.0, exp(s01 - mn0), k1ok);
        let p02 = select(0.0, exp(s02 - mn0), k2ok);
        let p03 = select(0.0, exp(s03 - mn0), k3ok);
        let p10 = select(0.0, exp(s10 - mn1), k0ok);
        let p11 = select(0.0, exp(s11 - mn1), k1ok);
        let p12 = select(0.0, exp(s12 - mn1), k2ok);
        let p13 = select(0.0, exp(s13 - mn1), k3ok);
        let p20 = select(0.0, exp(s20 - mn2), k0ok);
        let p21 = select(0.0, exp(s21 - mn2), k1ok);
        let p22 = select(0.0, exp(s22 - mn2), k2ok);
        let p23 = select(0.0, exp(s23 - mn2), k3ok);
        let p30 = select(0.0, exp(s30 - mn3), k0ok);
        let p31 = select(0.0, exp(s31 - mn3), k1ok);
        let p32 = select(0.0, exp(s32 - mn3), k2ok);
        let p33 = select(0.0, exp(s33 - mn3), k3ok);
        var bl0 = p00 + p01 + p02 + p03;
        var bl1 = p10 + p11 + p12 + p13;
        var bl2 = p20 + p21 + p22 + p23;
        var bl3 = p30 + p31 + p32 + p33;
/*FOLD_SUM*/
        l0 = l0 * a0 + bl0;
        l1 = l1 * a1 + bl1;
        l2 = l2 * a2 + bl2;
        l3 = l3 * a3 + bl3;
        m0 = mn0;
        m1 = mn1;
        m2 = mn2;
        m3 = mn3;
        o00 = o00 * a0;
        o01 = o01 * a0;
        o10 = o10 * a1;
        o11 = o11 * a1;
        o20 = o20 * a2;
        o21 = o21 * a2;
        o30 = o30 * a3;
        o31 = o31 * a3;
        let ps = lx ^ (ly & 3u);
        sp4[ly * 8u + ps] = vec4<f32>(p00, p01, p02, p03);
        sp4[(ly + 16u) * 8u + ps] = vec4<f32>(p10, p11, p12, p13);
        sp4[(ly + 32u) * 8u + ps] = vec4<f32>(p20, p21, p22, p23);
        sp4[(ly + 48u) * 8u + ps] = vec4<f32>(p30, p31, p32, p33);
        workgroupBarrier();

        // ---- C: O += P V over the 8 key groups. One `vec4` of weights per
        //      row and 8 `vec4` of V per group: 128 FMAs per 12 loads.
        for (var kg = 0u; kg < 8u; kg = kg + 1u) {
            let pg = kg ^ (ly & 3u);
            let p0 = sp4[ly * 8u + pg];
            let p1 = sp4[(ly + 16u) * 8u + pg];
            let p2 = sp4[(ly + 32u) * 8u + pg];
            let p3 = sp4[(ly + 48u) * 8u + pg];
            let vs0 = lx ^ kg;
            let vs1 = (lx + 8u) ^ kg;
            let base = kg * 64u;
            let v00 = skv4[base + vs0];
            let v01 = skv4[base + vs1];
            o00 = o00 + p0.x * v00;
            o01 = o01 + p0.x * v01;
            o10 = o10 + p1.x * v00;
            o11 = o11 + p1.x * v01;
            o20 = o20 + p2.x * v00;
            o21 = o21 + p2.x * v01;
            o30 = o30 + p3.x * v00;
            o31 = o31 + p3.x * v01;
            let v10 = skv4[base + 16u + vs0];
            let v11 = skv4[base + 16u + vs1];
            o00 = o00 + p0.y * v10;
            o01 = o01 + p0.y * v11;
            o10 = o10 + p1.y * v10;
            o11 = o11 + p1.y * v11;
            o20 = o20 + p2.y * v10;
            o21 = o21 + p2.y * v11;
            o30 = o30 + p3.y * v10;
            o31 = o31 + p3.y * v11;
            let v20 = skv4[base + 32u + vs0];
            let v21 = skv4[base + 32u + vs1];
            o00 = o00 + p0.z * v20;
            o01 = o01 + p0.z * v21;
            o10 = o10 + p1.z * v20;
            o11 = o11 + p1.z * v21;
            o20 = o20 + p2.z * v20;
            o21 = o21 + p2.z * v21;
            o30 = o30 + p3.z * v20;
            o31 = o31 + p3.z * v21;
            let v30 = skv4[base + 48u + vs0];
            let v31 = skv4[base + 48u + vs1];
            o00 = o00 + p0.w * v30;
            o01 = o01 + p0.w * v31;
            o10 = o10 + p1.w * v30;
            o11 = o11 + p1.w * v31;
            o20 = o20 + p2.w * v30;
            o21 = o21 + p2.w * v31;
            o30 = o30 + p3.w * v30;
            o31 = o31 + p3.w * v31;
        }
        workgroupBarrier();
    }

    // ---- epilogue: the normalised rows of the block that exist
    let o_off = head * 16u + lx;
    if (q_base + ly < cfg.t) {
        out4[(q_base + ly) * cfg.hidden4 + o_off] = o00 / l0;
        out4[(q_base + ly) * cfg.hidden4 + o_off + 8u] = o01 / l0;
    }
    if (q_base + ly + 16u < cfg.t) {
        out4[(q_base + ly + 16u) * cfg.hidden4 + o_off] = o10 / l1;
        out4[(q_base + ly + 16u) * cfg.hidden4 + o_off + 8u] = o11 / l1;
    }
    if (q_base + ly + 32u < cfg.t) {
        out4[(q_base + ly + 32u) * cfg.hidden4 + o_off] = o20 / l2;
        out4[(q_base + ly + 32u) * cfg.hidden4 + o_off + 8u] = o21 / l2;
    }
    if (q_base + ly + 48u < cfg.t) {
        out4[(q_base + ly + 48u) * cfg.hidden4 + o_off] = o30 / l3;
        out4[(q_base + ly + 48u) * cfg.hidden4 + o_off + 8u] = o31 / l3;
    }
}
"#;
    BODY.replace(
        "/*FOLD_MAX*/",
        &flash_fold(&["bm0", "bm1", "bm2", "bm3"], "sp4", subgroup, true),
    )
    .replace(
        "/*FOLD_SUM*/",
        &flash_fold(&["bl0", "bl1", "bl2", "bl3"], "sp4", subgroup, false),
    )
}

/// The 32-query, 16-key build: 14,336 B of workgroup storage, under WebGPU's
/// 16 KiB floor. Same schedule as [`flash_attn_large`] with half the rows, two
/// keys per thread and `vec2` key groups.
fn flash_attn_small(subgroup: bool) -> String {
    const BODY: &str = r#"
struct FlashAttnCfg {
    t: u32,
    heads: u32,
    hidden4: u32,
    _p0: u32,
    scale: f32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

@group(0) @binding(0) var<storage, read> qkv4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out4: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> cfg: FlashAttnCfg;

var<workgroup> sq4: array<vec4<f32>, 512u>;     // 32 x 16, 8,192 B
var<workgroup> skv4: array<vec4<f32>, 256u>;    // 16 x 16, 4,096 B
var<workgroup> sp2: array<vec2<f32>, 256u>;     // 32 x 8,  2,048 B

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x & 7u;
    let ly = lid.x >> 3u;
    let r0 = lid.x >> 4u;
    let c4 = lid.x & 15u;
    let head = wid.y;
    let q_base = wid.x * 32u;
    let row = 3u * cfg.hidden4;
    let q_off = head * 16u;
    let k_off = cfg.hidden4 + head * 16u;
    let v_off = 2u * cfg.hidden4 + head * 16u;

    // ---- prologue: this block's 32 queries, zero-padded past t
    for (var u = 0u; u < 4u; u = u + 1u) {
        let qr = r0 + u * 8u;
        let gq = q_base + qr;
        var qv = vec4<f32>(0.0);
        if (gq < cfg.t) {
            qv = qkv4[gq * row + q_off + c4];
        }
        sq4[qr * 16u + (c4 ^ (qr & 7u))] = qv;
    }

    var o00 = vec4<f32>(0.0);
    var o01 = vec4<f32>(0.0);
    var o10 = vec4<f32>(0.0);
    var o11 = vec4<f32>(0.0);
    var m0 = -3.4e38;
    var l0 = 0.0;
    var m1 = -3.4e38;
    var l1 = 0.0;

    let ktiles = (cfg.t + 15u) / 16u;
    for (var kt = 0u; kt < ktiles; kt = kt + 1u) {
        let k0 = kt * 16u;

        // ---- K tile
        for (var u = 0u; u < 2u; u = u + 1u) {
            let kr = r0 + u * 8u;
            let gk = k0 + kr;
            var kv = vec4<f32>(0.0);
            if (gk < cfg.t) {
                kv = qkv4[gk * row + k_off + c4];
            }
            skv4[kr * 16u + (c4 ^ (kr >> 1u))] = kv;
        }
        workgroupBarrier();

        // ---- A: 2 queries x 2 keys = 4 scores, spelled out like the 64x32
        //      build's so both compile to the same four FFMAs per dot
        var s00 = 0.0;
        var s01 = 0.0;
        var s10 = 0.0;
        var s11 = 0.0;
        for (var d4 = 0u; d4 < 16u; d4 = d4 + 1u) {
            let qs = d4 ^ (ly & 7u);
            let ks = d4 ^ lx;
            let kv0 = skv4[lx * 32u + ks];
            let kv1 = skv4[lx * 32u + 16u + ks];
            let qv0 = sq4[ly * 16u + qs];
            s00 = s00 + qv0.x * kv0.x;
            s00 = s00 + qv0.y * kv0.y;
            s00 = s00 + qv0.z * kv0.z;
            s00 = s00 + qv0.w * kv0.w;
            s01 = s01 + qv0.x * kv1.x;
            s01 = s01 + qv0.y * kv1.y;
            s01 = s01 + qv0.z * kv1.z;
            s01 = s01 + qv0.w * kv1.w;
            let qv1 = sq4[(ly + 16u) * 16u + qs];
            s10 = s10 + qv1.x * kv0.x;
            s10 = s10 + qv1.y * kv0.y;
            s10 = s10 + qv1.z * kv0.z;
            s10 = s10 + qv1.w * kv0.w;
            s11 = s11 + qv1.x * kv1.x;
            s11 = s11 + qv1.y * kv1.y;
            s11 = s11 + qv1.z * kv1.z;
            s11 = s11 + qv1.w * kv1.w;
        }
        workgroupBarrier();

        // ---- B: V tile, rescale, mask, row max, exp, row sum
        for (var u = 0u; u < 2u; u = u + 1u) {
            let kr = r0 + u * 8u;
            let gk = k0 + kr;
            var vv = vec4<f32>(0.0);
            if (gk < cfg.t) {
                vv = qkv4[gk * row + v_off + c4];
            }
            skv4[kr * 16u + (c4 ^ (kr >> 1u))] = vv;
        }
        let kb = k0 + lx * 2u;
        let k0ok = kb < cfg.t;
        let k1ok = kb + 1u < cfg.t;
        s00 = select(-3.4e38, s00 * cfg.scale, k0ok);
        s01 = select(-3.4e38, s01 * cfg.scale, k1ok);
        s10 = select(-3.4e38, s10 * cfg.scale, k0ok);
        s11 = select(-3.4e38, s11 * cfg.scale, k1ok);
        var bm0 = max(s00, s01);
        var bm1 = max(s10, s11);
/*FOLD_MAX*/
        let mn0 = max(m0, bm0);
        let mn1 = max(m1, bm1);
        let a0 = select(0.0, exp(m0 - mn0), m0 > -1.0e37);
        let a1 = select(0.0, exp(m1 - mn1), m1 > -1.0e37);
        let p00 = select(0.0, exp(s00 - mn0), k0ok);
        let p01 = select(0.0, exp(s01 - mn0), k1ok);
        let p10 = select(0.0, exp(s10 - mn1), k0ok);
        let p11 = select(0.0, exp(s11 - mn1), k1ok);
        var bl0 = p00 + p01;
        var bl1 = p10 + p11;
/*FOLD_SUM*/
        l0 = l0 * a0 + bl0;
        l1 = l1 * a1 + bl1;
        m0 = mn0;
        m1 = mn1;
        o00 = o00 * a0;
        o01 = o01 * a0;
        o10 = o10 * a1;
        o11 = o11 * a1;
        let ps = lx ^ (ly & 3u);
        sp2[ly * 8u + ps] = vec2<f32>(p00, p01);
        sp2[(ly + 16u) * 8u + ps] = vec2<f32>(p10, p11);
        workgroupBarrier();

        // ---- C: O += P V over the 8 key groups of 2 keys
        for (var kg = 0u; kg < 8u; kg = kg + 1u) {
            let pg = kg ^ (ly & 3u);
            let p0 = sp2[ly * 8u + pg];
            let p1 = sp2[(ly + 16u) * 8u + pg];
            let vs0 = lx ^ kg;
            let vs1 = (lx + 8u) ^ kg;
            let base = kg * 32u;
            let v00 = skv4[base + vs0];
            let v01 = skv4[base + vs1];
            o00 = o00 + p0.x * v00;
            o01 = o01 + p0.x * v01;
            o10 = o10 + p1.x * v00;
            o11 = o11 + p1.x * v01;
            let v10 = skv4[base + 16u + vs0];
            let v11 = skv4[base + 16u + vs1];
            o00 = o00 + p0.y * v10;
            o01 = o01 + p0.y * v11;
            o10 = o10 + p1.y * v10;
            o11 = o11 + p1.y * v11;
        }
        workgroupBarrier();
    }

    // ---- epilogue
    let o_off = head * 16u + lx;
    if (q_base + ly < cfg.t) {
        out4[(q_base + ly) * cfg.hidden4 + o_off] = o00 / l0;
        out4[(q_base + ly) * cfg.hidden4 + o_off + 8u] = o01 / l0;
    }
    if (q_base + ly + 16u < cfg.t) {
        out4[(q_base + ly + 16u) * cfg.hidden4 + o_off] = o10 / l1;
        out4[(q_base + ly + 16u) * cfg.hidden4 + o_off + 8u] = o11 / l1;
    }
}
"#;
    BODY.replace(
        "/*FOLD_MAX*/",
        &flash_fold(&["bm0", "bm1"], "sp2", subgroup, true),
    )
    .replace(
        "/*FOLD_SUM*/",
        &flash_fold(&["bl0", "bl1"], "sp2", subgroup, false),
    )
}

/// Parse and validate a module the way a device would, without opening one.
///
/// The `f16` GEMM tiles and the `df64` kernels cannot be compiled on a GPU
/// that lacks the feature they need (this machine's P104-100 has no
/// `SHADER_F16`), so their type check runs in naga instead: it rejects an
/// `f32`/`f16` mix in an expression, a `vec4<f32>` stored into a `vec4<f16>`
/// tile, an undefined binding or an unbalanced block, exactly as a driver's
/// front end would before the pipeline is created.
#[cfg(test)]
pub(crate) fn naga_validate(src: &str, what: &str) {
    let module = wgpu::naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{what} does not parse: {}", e.emit_to_string(src)));
    let mut v = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("{what} does not validate: {e:?}"));
}

/// Serializes every test that opens a Vulkan device.
///
/// `cargo test` runs the test functions in parallel threads and this machine's
/// driver does not survive two devices in one process, so the shader checks
/// and the `df64` DP check each take this lock first. It lives here because
/// both modules already depend on [`Gpu`].
#[cfg(test)]
pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

    /// The two FlashAttention tiles, keyed by the adapter storage they need.
    const FLASH_TILES: [(u32, u32, u32); 2] = [
        // (max_storage, bq, workgroup bytes)
        (16 * 1024, 32, 14_336),
        (32 * 1024, 64, 32_768),
    ];

    #[test]
    fn flash_attn_tiles_fit_the_storage_they_are_built_for() {
        // The declarations are the layout: 16 B per `vec4`, 8 B per `vec2`,
        // with no padding columns anywhere (the swizzle needs none).
        let layout = |bq: u32| {
            if bq == 64 {
                [
                    ("var<workgroup> sq4: array<vec4<f32>, 1024u>;", 1024 * 16),
                    ("var<workgroup> skv4: array<vec4<f32>, 512u>;", 512 * 16),
                    ("var<workgroup> sp4: array<vec4<f32>, 512u>;", 512 * 16),
                ]
            } else {
                [
                    ("var<workgroup> sq4: array<vec4<f32>, 512u>;", 512 * 16),
                    ("var<workgroup> skv4: array<vec4<f32>, 256u>;", 256 * 16),
                    ("var<workgroup> sp2: array<vec2<f32>, 256u>;", 256 * 8),
                ]
            }
        };
        for (storage, bq, bytes) in FLASH_TILES {
            assert_eq!(flash_attn_shared_bytes(storage), bytes, "{storage} B");
            // the tile must not use more than the floor it is built for
            assert!(bytes <= storage, "{storage} B");
            let (decls, sum) = (layout(bq), layout(bq).iter().map(|d| d.1).sum::<u32>());
            assert_eq!(sum, bytes, "{storage} B");
            for subgroup in [false, true] {
                let (src, want_bq) = flash_attn(storage, subgroup);
                assert_eq!(want_bq, bq, "{storage} B");
                for (decl, _) in decls {
                    assert!(src.contains(decl), "{storage} B {subgroup}: {decl}");
                }
                // 1-D workgroup: `lid.x` maps linearly into subgroups, which
                // is what makes the 8 `lx` lanes one aligned group.
                assert!(src.contains("@workgroup_size(128)"), "{storage} B");
                assert!(!src.contains("enable"), "{storage} B");
                assert!(!src.contains("@workgroup_size(16, 16)"), "{storage} B");
                // Four barriers per key tile in the subgroup build (K staged,
                // phase A done, V staged with phase B, phase C done); the
                // shared-memory fold brackets both of its reads.
                let barriers = src.matches("workgroupBarrier()").count();
                assert_eq!(barriers, if subgroup { 4 } else { 8 }, "{storage} B");
                let folds = if subgroup {
                    // 2 folds x rows x 3 masks
                    2 * (if bq == 64 { 4 } else { 2 }) * 3
                } else {
                    0
                };
                assert_eq!(
                    src.matches("subgroupShuffleXor").count(),
                    folds,
                    "{storage} B"
                );
            }
        }
    }

    #[test]
    fn flash_attn_validates_for_every_tile_and_fold() {
        for storage in [16 * 1024, 32 * 1024, 48 * 1024] {
            for subgroup in [false, true] {
                let (src, bq) = flash_attn(storage, subgroup);
                naga_validate(&src, &format!("flash_attn({storage}, {subgroup})"));
                // bq | 32 query rows per workgroup, 16 head dimensions each
                let slots = bq * 16;
                assert!(src.contains(&format!("var<workgroup> sq4: array<vec4<f32>, {slots}u>")));
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
        // `cargo test` runs the test functions in parallel threads and this
        // machine's driver does not survive two devices in one process.
        let _gpu_lock = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

        // ---- transpose writes into the padded output pitch at `o_off`
        {
            let tpipe = pipeline(&gpu, "t-transpose", &transpose());
            let (rows, cols, a_stride, a_off, o_stride) = (5usize, 3usize, 7usize, 2usize, 8usize);
            let a: Vec<f32> = (0..a_off + rows * a_stride)
                .map(|i| i as f32 + 0.5)
                .collect();
            let a_buf = storage_f32(&gpu, "t-tr-a", &a);
            // 0 is the `K^T` case; a non-zero one is the second of the three
            // fused `qkv` projections, which the loader writes into one
            // buffer at column offsets. A dropped offset overwrites `q`
            // with `k` and still passes an all-zero test.
            for o_off in [0usize, rows] {
                let n = o_off + cols * o_stride;
                let o_buf = storage_f32(&gpu, "t-tr-o", &vec![-1.0f32; n]);
                let uni = uniform_u32(
                    &gpu,
                    "t-tr-u",
                    &[
                        rows as u32,
                        cols as u32,
                        a_stride as u32,
                        a_off as u32,
                        o_stride as u32,
                        o_off as u32,
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
                let o = read(&gpu, &o_buf, n);
                for r in 0..rows {
                    for c in 0..cols {
                        assert_eq!(
                            o[o_off + c * o_stride + r],
                            a[a_off + r * a_stride + c],
                            "transpose [{r},{c}] at o_off {o_off}"
                        );
                    }
                }
                for c in 0..cols {
                    for r in rows..o_stride {
                        assert_eq!(
                            o[o_off + c * o_stride + r],
                            -1.0,
                            "transpose wrote padding [{c},{r}] at o_off {o_off}"
                        );
                    }
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

        // ---- flash attention: fused softmax(Q K^T * scale) V, straight from
        //      the packed qkv buffer, against an f64 reference. Every tile this
        //      adapter can run is checked in both fold shapes, so the
        //      subgroup-less build the model falls back to on other hardware
        //      is exercised here as code, not only as a naga parse.
        {
            let (heads, hidden) = (4usize, 256usize);
            let head_dim = hidden / heads;
            assert_eq!(head_dim as u32, FLASH_HEAD_DIM, "the kernel's head width");
            let hidden4 = (hidden / 4) as u32;
            let scale = 0.125f64;
            // not tile-aligned in any direction: 1 is one row of one tile, 17
            // and 67 straddle a 16- and a 64-row block, 131 leaves 3 rows of
            // a 32-key tile live.
            for storage in [16 * 1024, 32 * 1024] {
                if storage > max_storage {
                    continue;
                }
                for subgroup in [false, true] {
                    if subgroup && !gpu.has_subgroup_8() {
                        continue;
                    }
                    let (src, bq) = flash_attn(storage, subgroup);
                    let label = format!("t-flash-{storage}-{subgroup}");
                    let pipe = pipeline(&gpu, &label, &src);
                    for t_len in [1usize, 17, 67, 131] {
                        let qkv: Vec<f32> = (0..t_len * 3 * hidden).map(noise).collect();
                        let qb = storage_f32(&gpu, &format!("{label}-qkv"), &qkv);
                        let ob = storage_f32(
                            &gpu,
                            &format!("{label}-out"),
                            &vec![-9.0f32; t_len * hidden],
                        );
                        let uni = uniform_u32(
                            &gpu,
                            &format!("{label}-u"),
                            &[
                                t_len as u32,
                                heads as u32,
                                hidden4,
                                0,
                                (scale as f32).to_bits(),
                                0,
                                0,
                                0,
                            ],
                        );
                        run(
                            &gpu,
                            &pipe,
                            &[bind(0, &qb), bind(1, &ob), bind(2, &uni)],
                            (t_len as u32).div_ceil(bq),
                            heads as u32,
                        );
                        let got = read(&gpu, &ob, t_len * hidden);
                        for q in 0..t_len {
                            for h in 0..heads {
                                // f64 reference: full softmax, no online update
                                let mut logits = vec![0f64; t_len];
                                for (k, slot) in logits.iter_mut().enumerate() {
                                    let mut acc = 0f64;
                                    for d in 0..head_dim {
                                        acc += qkv[q * 3 * hidden + h * head_dim + d] as f64
                                            * qkv[k * 3 * hidden + hidden + h * head_dim + d]
                                                as f64;
                                    }
                                    *slot = acc * scale;
                                }
                                let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                                let p: Vec<f64> = logits.iter().map(|s| (s - m).exp()).collect();
                                let den: f64 = p.iter().sum();
                                for d in 0..head_dim {
                                    let mut acc = 0f64;
                                    for (k, &pk) in p.iter().enumerate() {
                                        acc += pk
                                            * qkv[k * 3 * hidden + 2 * hidden + h * head_dim + d]
                                                as f64;
                                    }
                                    let want = acc / den;
                                    let have = got[q * hidden + h * head_dim + d];
                                    assert!(
                                        have.is_finite(),
                                        "flash {label} T={t_len} head {h} q {q} d {d} = {have}"
                                    );
                                    assert!(
                                        (want - have as f64).abs() < 1e-5,
                                        "flash {label} T={t_len} head {h} q {q} d {d}: \
                                         {want} vs {have}"
                                    );
                                }
                            }
                        }
                    }
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
