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
    conv_gemm, gemm_bias, gemm_bias_tiled, gemm_shared_bytes, pos_conv, KS, MT, NT, POS_KS, POS_MT,
    POS_NT,
};

/// Row LayerNorm over `cols` (affine, biased variance, eps inside sqrt) then
/// gelu when `do_gelu` is 1.  One workgroup per row, 256 threads.
pub fn layernorm() -> String {
    r#"
struct Cfg { rows: u32, cols: u32, eps_x_1e6: u32, do_gelu: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Cfg;

// red[0..256) = per-thread partials; red[256] = row sum; red[257] = row
// sum of squares.  Distinct slots: no write-after-read races on the scalars.
var<workgroup> red: array<f32, 258u>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    // y stacks past the 65535 workgroup limit (conv0 of a 34 s chunk is ~1e5 rows).
    let row = wid.x + wid.y * 65535u;
    if (row >= cfg.rows) { return; }
    let cols = cfg.cols;
    let base = row * cols;
    var sum = 0.0;
    var sq = 0.0;
    for (var i = lid.x; i < cols; i = i + 256u) {
        let v = h[base + i];
        sum = sum + v;
        sq = sq + v * v;
    }
    red[lid.x] = sum;
    workgroupBarrier();
    if (lid.x == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) { s = s + red[i]; }
        red[256] = s;
    }
    workgroupBarrier();
    let mean = red[256] / f32(cols);
    red[lid.x] = sq;
    workgroupBarrier();
    if (lid.x == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) { s = s + red[i]; }
        red[257] = s;
    }
    workgroupBarrier();
    // biased variance: E[x^2] - mean^2
    let varian = red[257] / f32(cols) - mean * mean;
    let eps = f32(cfg.eps_x_1e6) * 1e-6;
    let inv = 1.0 / sqrt(varian + eps);
    for (var i = lid.x; i < cols; i = i + 256u) {
        var v = (h[base + i] - mean) * inv * w[i] + b[i];
        if (cfg.do_gelu == 1u) {
            v = 0.5 * v * (1.0 + erf(v / 1.4142135623730951));
        }
        h[base + i] = v;
    }
}

// wgsl has no erf; a rational approximation (Abramowitz & Stegun 7.1.26, |e|<1.5e-7)
// keeps the gelu inside fp32 noise of the reference for these magnitudes.
fn erf(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * exp(-a * a);
    return s * y;
}
"#
    .to_string()
}

/// Elementwise gelu over n elements.
pub fn gelu() -> String {
    r#"
struct Cfg { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < cfg.n) {
        let v = h[i];
        h[i] = 0.5 * v * (1.0 + erf(v / 1.4142135623730951));
    }
}

fn erf(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * exp(-a * a);
    return s * y;
}
"#
    .to_string()
}

/// `out[i] = a[i]` — plain copy (avoids reading a zero buffer out of bounds)
pub fn copy() -> String {
    r#"
struct Cfg { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < cfg.n) { o[i] = a[i]; }
}
"#
    .to_string()
}

/// `out[i] = a[i] + b[i]`
pub fn add() -> String {
    r#"
struct Cfg { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
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
pub fn softmax() -> String {
    r#"
struct Cfg { rows: u32, cols: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;

var<workgroup> red: array<f32, 256u>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * 65535u;
    if (row >= cfg.rows) { return; }
    let base = row * cfg.cols;
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
        red[lid.x] = mx;
        workgroupBarrier();
        if (lid.x == 0u) {
            var m = -3.4e38;
            for (var i = 0u; i < 256u; i = i + 1u) { m = max(m, red[i]); }
            red[0] = m;
        }
        workgroupBarrier();
        let m = red[0];
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
        red[lid.x] = sum;
        workgroupBarrier();
        if (lid.x == 0u) {
            var s = 0.0;
            for (var i = 0u; i < 256u; i = i + 1u) { s = s + red[i]; }
            red[1] = s;
        }
        workgroupBarrier();
        let inv = 1.0 / red[1];
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
        red[lid.x] = mx;
        workgroupBarrier();
        if (lid.x == 0u) {
            var m = -3.4e38;
            for (var i = 0u; i < 256u; i = i + 1u) { m = max(m, red[i]); }
            red[0] = m;
        }
        workgroupBarrier();
        let m = red[0];
        var sum = 0.0;
        for (var i = lid.x; i < cols; i = i + 256u) {
            let e = exp(h[base + i] - m);
            h[base + i] = e;
            sum = sum + e;
        }
        red[lid.x] = sum;
        workgroupBarrier();
        if (lid.x == 0u) {
            var s = 0.0;
            for (var i = 0u; i < 256u; i = i + 1u) { s = s + red[i]; }
            red[1] = s;
        }
        workgroupBarrier();
        let inv = 1.0 / red[1];
        for (var i = lid.x; i < cols; i = i + 256u) {
            h[base + i] = h[base + i] * inv;
        }
    }
}
"#
    .to_string()
}

/// Row log_softmax over `cols` (two-pass, in-place).
pub fn log_softmax() -> String {
    r#"
struct Cfg { rows: u32, cols: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;

var<workgroup> red: array<f32, 256u>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * 65535u;
    if (row >= cfg.rows) { return; }
    let base = row * cfg.cols;
    let cols = cfg.cols;
    var mx = -3.4e38;
    for (var i = lid.x; i < cols; i = i + 256u) {
        mx = max(mx, h[base + i]);
    }
    red[lid.x] = mx;
    workgroupBarrier();
    if (lid.x == 0u) {
        var m = -3.4e38;
        for (var i = 0u; i < 256u; i = i + 1u) { m = max(m, red[i]); }
        red[0] = m;
    }
    workgroupBarrier();
    let m = red[0];
    var sum = 0.0;
    for (var i = lid.x; i < cols; i = i + 256u) {
        sum = sum + exp(h[base + i] - m);
    }
    red[lid.x] = sum;
    workgroupBarrier();
    if (lid.x == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) { s = s + red[i]; }
        red[1] = s;
    }
    workgroupBarrier();
    let lsum = log(red[1]);
    for (var i = lid.x; i < cols; i = i + 256u) {
        h[base + i] = h[base + i] - m - lsum;
    }
}
"#
    .to_string()
}

/// (rows, cols) -> (cols, rows)
pub fn transpose() -> String {
    r#"
struct Cfg { rows: u32, cols: u32, a_stride: u32, a_off: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if (r < cfg.rows && c < cfg.cols) {
        o[c * cfg.rows + r] = a[cfg.a_off + r * cfg.a_stride + c];
    }
}
"#
    .to_string()
}

/// conv layer 0: raw waveform (1 channel) -> (T_out, 512).
pub fn conv0() -> String {
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
