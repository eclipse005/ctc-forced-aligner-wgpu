//! Tiled fp32 GEMM for Vulkan on Pascal.
//!
//! 16×16 threads. `lid.x` owns a column strip and `lid.y` an interleaved row
//! strip, so a warp stores consecutive outputs and broadcasts A. B lives in
//! shared memory as `vec4`s along N. Each K step is an outer product of one
//! `vec4` of A (four K lanes) with those B rows. Shared A is padded so
//! consecutive rows start four banks apart (`sa_stride % 8 == 1`).
//!
//! Accumulators stay named `vec4` locals. An `array<f32, N>` accumulator
//! spills out of registers on this backend.

use std::fmt::Write;

pub const MT: u32 = 128;
pub const NT: u32 = 64;
pub const KS: u32 = 32;
const UNROLL_K: bool = true;

/// Positional conv tile. N is one group (64 channels). K is `tap * in_pg + ci`,
/// so each step of 4 is four channels of one tap and the FMA order matches the
/// scalar kernel. 128×64×32 is the same register schedule as the encoder GEMM.
pub const POS_MT: u32 = 128;
pub const POS_NT: u32 = 64;
pub const POS_KS: u32 = 32;

pub fn gemm_bias() -> String {
    gemm_bias_tiled(MT, NT, KS, UNROLL_K)
}

pub fn gemm_bias_tiled(mt: u32, nt: u32, ks: u32, unroll: bool) -> String {
    emit(mt, nt, ks, unroll, Kind::Gemm)
}

pub fn conv_gemm() -> String {
    emit(MT, NT, KS, UNROLL_K, Kind::Conv)
}

pub fn pos_conv() -> String {
    emit_pos(POS_MT, POS_NT, POS_KS)
}

pub fn gemm_shared_bytes(mt: u32, nt: u32, ks: u32) -> u32 {
    tile(mt, nt, ks).bytes
}

struct Tile {
    mt: u32,
    nt: u32,
    ks: u32,
    tm: u32,
    tn_vec: u32,
    sa_stride: u32,
    sb_stride: u32,
    bytes: u32,
}

fn tile(mt: u32, nt: u32, ks: u32) -> Tile {
    assert!(mt >= 16 && mt % 16 == 0, "MT must be a multiple of 16");
    assert!(nt >= 64 && nt % 64 == 0, "NT must be a multiple of 64");
    assert!(ks >= 4 && ks % 4 == 0, "KS must be a multiple of 4");
    let tm = mt / 16;
    let tn = nt / 16;
    assert!(tn % 4 == 0, "NT/16 must be a multiple of 4");
    let tn_vec = tn / 4;
    let payload = ks / 4;
    // Float stride = sa_stride * 4 ≡ 4 (mod 32), so consecutive rows do not
    // share banks on a vec4 load. That is sa_stride ≡ 1 (mod 8).
    let sa_stride = payload + (1 + 8 - (payload % 8)) % 8;
    let sb_stride = nt / 4;
    let bytes = (mt * sa_stride + ks * sb_stride) * 16;
    assert!(bytes <= 48 * 1024, "shared memory {bytes} exceeds 48 KiB");
    Tile { mt, nt, ks, tm, tn_vec, sa_stride, sb_stride, bytes }
}

#[derive(Clone, Copy)]
enum Kind {
    Gemm,
    Conv,
}

fn emit(mt: u32, nt: u32, ks: u32, unroll: bool, kind: Kind) -> String {
    let t = tile(mt, nt, ks);
    let mut s = String::with_capacity(32 * 1024);
    match kind {
        Kind::Gemm => s.push_str(
            r#"struct Dims {
    m: u32, n: u32, k: u32,
    a_stride: u32, b_stride: u32, c_stride: u32,
    a_off: u32, b_off: u32, c_off: u32,
    scale: f32,
}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
"#,
        ),
        Kind::Conv => s.push_str(
            r#"struct Dims {
    m: u32, n: u32, k: u32,
    c_in: u32, stride: u32, t_in: u32, _p0: u32,
}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
"#,
        ),
    }
    let _ = writeln!(s, "var<workgroup> sa: array<vec4<f32>, {}u>;", t.mt * t.sa_stride);
    let _ = writeln!(s, "var<workgroup> sb: array<vec4<f32>, {}u>;", t.ks * t.sb_stride);
    s.push_str(
        "
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
",
    );
    let _ = writeln!(s, "    let m0 = wid.x * {}u;", t.mt);
    let _ = writeln!(s, "    let n0 = wid.y * {}u;", t.nt);
    s.push_str("    let tid = lid.x + lid.y * 16u;\n");
    for row in 0..t.tm {
        let _ = writeln!(s, "    let row{row} = lid.y + {row}u * 16u;");
    }
    for row in 0..t.tm {
        for col in 0..t.tn_vec {
            let _ = writeln!(s, "    var acc_{row}_{col} = vec4<f32>(0.0);");
        }
    }
    let _ = writeln!(s, "    let m_full = m0 + {}u <= d.m;", t.mt);
    let _ = writeln!(s, "    let n_full = n0 + {}u <= d.n;", t.nt);
    let _ = writeln!(s, "    let ktiles = (d.k + {}u - 1u) / {}u;", t.ks, t.ks);
    s.push_str("    for (var kt = 0u; kt < ktiles; kt = kt + 1u) {\n");
    let _ = writeln!(s, "        let k0 = kt * {}u;", t.ks);
    let _ = writeln!(s, "        let k_full = k0 + {}u <= d.k;", t.ks);
    emit_loads(&mut s, &t, kind);
    s.push_str("        workgroupBarrier();\n");
    emit_fma(&mut s, &t, unroll);
    s.push_str("        workgroupBarrier();\n");
    s.push_str("    }\n");
    emit_store(&mut s, &t, kind);
    s.push_str("}\n");
    s
}

fn emit_loads(s: &mut String, t: &Tile, kind: Kind) {
    let a_vecs = t.mt * (t.ks / 4);
    let b_vecs = t.ks * t.sb_stride;
    let kv = t.ks / 4;
    match kind {
        Kind::Gemm => {
            s.push_str("        if (m_full && k_full) {\n");
            emit_a_fast(s, t, a_vecs, kv);
            s.push_str("        } else {\n");
            emit_a_slow(s, t, a_vecs, kv);
            s.push_str("        }\n");
            s.push_str("        if (n_full && k_full) {\n");
            emit_b_fast(s, t, b_vecs, "d.b_off + gk * d.b_stride + gn", "b");
            s.push_str("        } else {\n");
            emit_b_slow(s, t, b_vecs, "d.b_off + gk * d.b_stride + gn", "b");
            s.push_str("        }\n");
        }
        Kind::Conv => {
            // im2col crosses a tap when c_in is not a multiple of 4, so the
            // guarded gather is used for every K tile. Conv is a few layers
            // once per chunk; the encoder GEMMs dominate.
            emit_im2col(s, t, a_vecs, kv);
            s.push_str("        if (n_full && k_full) {\n");
            emit_b_fast(s, t, b_vecs, "gk * d.n + gn", "wt");
            s.push_str("        } else {\n");
            emit_b_slow(s, t, b_vecs, "gk * d.n + gn", "wt");
            s.push_str("        }\n");
        }
    }
}

fn emit_a_fast(s: &mut String, t: &Tile, a_vecs: u32, kv: u32) {
    let _ = writeln!(s, "            for (var i = tid; i < {a_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {kv}u;");
    let _ = writeln!(s, "                let kvi = i % {kv}u;");
    s.push_str("                let base = d.a_off + (m0 + rr) * d.a_stride + (k0 + kvi * 4u);\n");
    let _ = writeln!(
        s,
        "                sa[(rr * {}u) + kvi] = vec4<f32>(a[base], a[base + 1u], a[base + 2u], a[base + 3u]);",
        t.sa_stride
    );
    s.push_str("            }\n");
}

fn emit_a_slow(s: &mut String, t: &Tile, a_vecs: u32, kv: u32) {
    let _ = writeln!(s, "            for (var i = tid; i < {a_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {kv}u;");
    let _ = writeln!(s, "                let kvi = i % {kv}u;");
    s.push_str("                let gm = m0 + rr;\n");
    s.push_str("                let gk = k0 + kvi * 4u;\n");
    s.push_str("                var v = vec4<f32>(0.0);\n");
    s.push_str("                if (gm < d.m && gk + 3u < d.k) {\n");
    s.push_str("                    let base = d.a_off + gm * d.a_stride + gk;\n");
    s.push_str("                    v = vec4<f32>(a[base], a[base + 1u], a[base + 2u], a[base + 3u]);\n");
    s.push_str("                } else if (gm < d.m && gk < d.k) {\n");
    s.push_str("                    let base = d.a_off + gm * d.a_stride + gk;\n");
    s.push_str("                    v.x = a[base];\n");
    s.push_str("                    if (gk + 1u < d.k) { v.y = a[base + 1u]; }\n");
    s.push_str("                    if (gk + 2u < d.k) { v.z = a[base + 2u]; }\n");
    s.push_str("                    if (gk + 3u < d.k) { v.w = a[base + 3u]; }\n");
    s.push_str("                }\n");
    let _ = writeln!(s, "                sa[(rr * {}u) + kvi] = v;", t.sa_stride);
    s.push_str("            }\n");
}

fn emit_b_fast(s: &mut String, t: &Tile, b_vecs: u32, base: &str, buf: &str) {
    let _ = writeln!(s, "            for (var i = tid; i < {b_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {}u;", t.sb_stride);
    let _ = writeln!(s, "                let cv = i % {}u;", t.sb_stride);
    s.push_str("                let gk = k0 + rr;\n");
    s.push_str("                let gn = n0 + cv * 4u;\n");
    let _ = writeln!(s, "                let base = {base};");
    let _ = writeln!(
        s,
        "                sb[(rr * {}u) + cv] = vec4<f32>({buf}[base], {buf}[base + 1u], {buf}[base + 2u], {buf}[base + 3u]);",
        t.sb_stride
    );
    s.push_str("            }\n");
}

fn emit_b_slow(s: &mut String, t: &Tile, b_vecs: u32, base: &str, buf: &str) {
    let _ = writeln!(s, "            for (var i = tid; i < {b_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {}u;", t.sb_stride);
    let _ = writeln!(s, "                let cv = i % {}u;", t.sb_stride);
    s.push_str("                let gk = k0 + rr;\n");
    s.push_str("                let gn = n0 + cv * 4u;\n");
    s.push_str("                var v = vec4<f32>(0.0);\n");
    s.push_str("                if (gk < d.k && gn + 3u < d.n) {\n");
    let _ = writeln!(s, "                    let base = {base};");
    let _ = writeln!(
        s,
        "                    v = vec4<f32>({buf}[base], {buf}[base + 1u], {buf}[base + 2u], {buf}[base + 3u]);"
    );
    s.push_str("                } else if (gk < d.k && gn < d.n) {\n");
    let _ = writeln!(s, "                    let base = {base};");
    let _ = writeln!(s, "                    v.x = {buf}[base];");
    let _ = writeln!(s, "                    if (gn + 1u < d.n) {{ v.y = {buf}[base + 1u]; }}");
    let _ = writeln!(s, "                    if (gn + 2u < d.n) {{ v.z = {buf}[base + 2u]; }}");
    let _ = writeln!(s, "                    if (gn + 3u < d.n) {{ v.w = {buf}[base + 3u]; }}");
    s.push_str("                }\n");
    let _ = writeln!(s, "                sb[(rr * {}u) + cv] = v;", t.sb_stride);
    s.push_str("            }\n");
}

fn emit_im2col(s: &mut String, t: &Tile, a_vecs: u32, kv: u32) {
    let _ = writeln!(s, "            for (var i = tid; i < {a_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {kv}u;");
    let _ = writeln!(s, "                let kvi = i % {kv}u;");
    s.push_str("                let gm = m0 + rr;\n");
    s.push_str("                let kk = k0 + kvi * 4u;\n");
    s.push_str("                var v = vec4<f32>(0.0);\n");
    s.push_str("                if (gm < d.m && kk < d.k) {\n");
    s.push_str("                    let t0 = kk / d.c_in;\n");
    s.push_str("                    let ci = kk % d.c_in;\n");
    s.push_str("                    let src = gm * d.stride + t0;\n");
    s.push_str("                    if (kk + 3u < d.k && src < d.t_in && ci + 3u < d.c_in) {\n");
    s.push_str("                        let base = src * d.c_in + ci;\n");
    s.push_str("                        v = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);\n");
    s.push_str("                    } else {\n");
    for lane in 0..4u32 {
        let comp = ['x', 'y', 'z', 'w'][lane as usize];
        if lane == 0 {
            s.push_str("                        {\n");
        } else {
            let _ = writeln!(s, "                        if (kk + {lane}u < d.k) {{");
        }
        let _ = writeln!(s, "                            let kl = kk + {lane}u;");
        s.push_str("                            let tl = kl / d.c_in;\n");
        s.push_str("                            let cil = kl % d.c_in;\n");
        s.push_str("                            let srcl = gm * d.stride + tl;\n");
        let _ = writeln!(
            s,
            "                            if (srcl < d.t_in) {{ v.{comp} = x[srcl * d.c_in + cil]; }}"
        );
        s.push_str("                        }\n");
    }
    s.push_str("                    }\n");
    s.push_str("                }\n");
    let _ = writeln!(s, "                sa[(rr * {}u) + kvi] = v;", t.sa_stride);
    s.push_str("            }\n");
}

fn emit_fma(s: &mut String, t: &Tile, unroll: bool) {
    if unroll {
        for kv in 0..t.ks / 4 {
            emit_fma_step(s, t, &format!("{kv}u"));
        }
    } else {
        let _ = writeln!(s, "        for (var kvi = 0u; kvi < {}u; kvi = kvi + 1u) {{", t.ks / 4);
        emit_fma_step(s, t, "kvi");
        s.push_str("        }\n");
    }
}

fn emit_fma_step(s: &mut String, t: &Tile, kv: &str) {
    s.push_str("        {\n");
    for lane in 0..4u32 {
        for col in 0..t.tn_vec {
            let _ = writeln!(
                s,
                "            let b_{lane}_{col} = sb[((({kv}) * 4u + {lane}u) * {}u) + lid.x + {}u];",
                t.sb_stride,
                col * 16
            );
        }
    }
    for row in 0..t.tm {
        let _ = writeln!(
            s,
            "            let a_{row} = sa[(row{row} * {}u) + ({kv})];",
            t.sa_stride
        );
        for col in 0..t.tn_vec {
            let acc = format!("acc_{row}_{col}");
            let _ = writeln!(s, "            {acc} = {acc} + a_{row}.x * b_0_{col};");
            let _ = writeln!(s, "            {acc} = {acc} + a_{row}.y * b_1_{col};");
            let _ = writeln!(s, "            {acc} = {acc} + a_{row}.z * b_2_{col};");
            let _ = writeln!(s, "            {acc} = {acc} + a_{row}.w * b_3_{col};");
        }
    }
    s.push_str("        }\n");
}

fn emit_store(s: &mut String, t: &Tile, kind: Kind) {
    let (dst, scale, base) = match kind {
        Kind::Gemm => ("c", "d.scale", "d.c_off + gm * d.c_stride + gn"),
        Kind::Conv => ("out", "1.0", "gm * d.n + gn"),
    };
    s.push_str("    if (m_full && n_full) {\n");
    emit_store_rows(s, t, dst, scale, base, false);
    s.push_str("    } else {\n");
    emit_store_rows(s, t, dst, scale, base, true);
    s.push_str("    }\n");
}

fn emit_store_rows(s: &mut String, t: &Tile, dst: &str, scale: &str, base: &str, masked: bool) {
    for col in 0..t.tn_vec {
        let _ = writeln!(s, "        {{");
        let _ = writeln!(s, "            let gn = n0 + (lid.x + {}u) * 4u;", col * 16);
        if masked {
            s.push_str("            var biasv = vec4<f32>(0.0);\n");
            s.push_str("            if (gn + 3u < d.n) {\n");
            s.push_str("                biasv = vec4<f32>(bias[gn], bias[gn + 1u], bias[gn + 2u], bias[gn + 3u]);\n");
            s.push_str("            } else {\n");
            s.push_str("                if (gn < d.n) { biasv.x = bias[gn]; }\n");
            s.push_str("                if (gn + 1u < d.n) { biasv.y = bias[gn + 1u]; }\n");
            s.push_str("                if (gn + 2u < d.n) { biasv.z = bias[gn + 2u]; }\n");
            s.push_str("                if (gn + 3u < d.n) { biasv.w = bias[gn + 3u]; }\n");
            s.push_str("            }\n");
        } else {
            s.push_str("            let biasv = vec4<f32>(bias[gn], bias[gn + 1u], bias[gn + 2u], bias[gn + 3u]);\n");
        }
        for row in 0..t.tm {
            let _ = writeln!(s, "            {{");
            let _ = writeln!(s, "                let gm = m0 + row{row};");
            let _ = writeln!(s, "                let o = acc_{row}_{col} * {scale} + biasv;");
            let _ = writeln!(s, "                let base = {base};");
            if masked {
                s.push_str("                if (gm < d.m && gn + 3u < d.n) {\n");
                store4(s, dst);
                s.push_str("                } else if (gm < d.m) {\n");
                store_masked(s, dst);
                s.push_str("                }\n");
            } else {
                store4(s, dst);
            }
            s.push_str("            }\n");
        }
        s.push_str("        }\n");
    }
}

/// Grouped positional conv. Workgroup `(time_tile, group)` writes gelu(bias + conv)
/// for `POS_MT` frames and all `POS_NT` outputs of that group.
///
/// Weights are `(group, K, N)` with `K = tap * in_pg + ci` and `N` contiguous,
/// so B loads along the output channel. `in_pg` and `K` are multiples of the
/// tile, which keeps every K step inside one tap.
fn emit_pos(mt: u32, nt: u32, ks: u32) -> String {
    let t = tile(mt, nt, ks);
    let kv = ks / 4;
    let a_vecs = mt * kv;
    let b_vecs = ks * t.sb_stride;
    let mut s = String::with_capacity(48 * 1024);
    s.push_str(
        r#"struct Dims {
    m: u32, n: u32, k: u32, c: u32,
    in_pg: u32, pad: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;

fn erf(v: f32) -> f32 {
    let s = sign(v);
    let a = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * exp(-a * a);
    return s * y;
}

fn gelu(v: f32) -> f32 {
    return 0.5 * v * (1.0 + erf(v / 1.4142135623730951));
}
"#,
    );
    let _ = writeln!(s, "var<workgroup> sa: array<vec4<f32>, {}u>;", t.mt * t.sa_stride);
    let _ = writeln!(s, "var<workgroup> sb: array<vec4<f32>, {}u>;", t.ks * t.sb_stride);
    s.push_str(
        "
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
",
    );
    let _ = writeln!(s, "    let m0 = wid.x * {}u;", t.mt);
    s.push_str("    let n0 = 0u;\n");
    s.push_str("    let gch = wid.y * d.n;\n");
    s.push_str("    let tid = lid.x + lid.y * 16u;\n");
    for row in 0..t.tm {
        let _ = writeln!(s, "    let row{row} = lid.y + {row}u * 16u;");
    }
    for col in 0..t.tn_vec {
        let _ = writeln!(s, "    let gn{col} = (lid.x + {}u) * 4u;", col * 16);
        let _ = writeln!(
            s,
            "    let bias{col} = vec4<f32>(bias[gch + gn{col}], bias[gch + gn{col} + 1u], bias[gch + gn{col} + 2u], bias[gch + gn{col} + 3u]);"
        );
        for row in 0..t.tm {
            let _ = writeln!(s, "    var acc_{row}_{col} = bias{col};");
        }
    }
    let _ = writeln!(s, "    let m_full = m0 + {}u <= d.m;", t.mt);
    let _ = writeln!(s, "    let ktiles = (d.k + {}u - 1u) / {}u;", t.ks, t.ks);
    s.push_str("    for (var kt = 0u; kt < ktiles; kt = kt + 1u) {\n");
    let _ = writeln!(s, "        let k0 = kt * {}u;", t.ks);
    s.push_str("        let tk0 = k0 / d.in_pg;\n");
    let _ = writeln!(s, "        let single = (k0 + {}u) / d.in_pg == tk0;", ks - 1);
    s.push_str("        let ci0 = k0 % d.in_pg;\n");
    s.push_str("        let src0 = i32(m0) + i32(tk0) - i32(d.pad);\n");
    let _ = writeln!(s, "        let src1 = src0 + {}i;", mt as i32 - 1);
    let _ = writeln!(s, "        let k_full = k0 + {}u <= d.k;", t.ks);
    s.push_str("        if (single && k_full && m_full && src0 >= 0i && u32(src1) < d.m) {\n");
    let _ = writeln!(s, "            let xch = gch + ci0;");
    let _ = writeln!(s, "            for (var i = tid; i < {a_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {kv}u;");
    let _ = writeln!(s, "                let kvi = i % {kv}u;");
    s.push_str("                let base = u32(src0 + i32(rr)) * d.c + xch + kvi * 4u;\n");
    let _ = writeln!(
        s,
        "                sa[(rr * {}u) + kvi] = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);",
        t.sa_stride
    );
    s.push_str("            }\n");
    s.push_str("        } else {\n");
    let _ = writeln!(s, "            for (var i = tid; i < {a_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {kv}u;");
    let _ = writeln!(s, "                let kvi = i % {kv}u;");
    s.push_str("                let gm = m0 + rr;\n");
    s.push_str("                let kk = k0 + kvi * 4u;\n");
    s.push_str("                let tk = kk / d.in_pg;\n");
    s.push_str("                let ci = kk % d.in_pg;\n");
    s.push_str("                let src = i32(gm) + i32(tk) - i32(d.pad);\n");
    s.push_str("                var v = vec4<f32>(0.0);\n");
    s.push_str("                if (gm < d.m && kk + 3u < d.k && src >= 0i && u32(src) < d.m && ci + 3u < d.in_pg) {\n");
    s.push_str("                    let base = u32(src) * d.c + gch + ci;\n");
    s.push_str("                    v = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);\n");
    s.push_str("                } else if (gm < d.m) {\n");
    for lane in 0..4u32 {
        let comp = ['x', 'y', 'z', 'w'][lane as usize];
        let _ = writeln!(s, "                    if (kk + {lane}u < d.k) {{");
        let _ = writeln!(s, "                        let kl = kk + {lane}u;");
        s.push_str("                        let tl = kl / d.in_pg;\n");
        s.push_str("                        let cil = kl % d.in_pg;\n");
        s.push_str("                        let srcl = i32(gm) + i32(tl) - i32(d.pad);\n");
        let _ = writeln!(
            s,
            "                        if (srcl >= 0i && u32(srcl) < d.m) {{ v.{comp} = x[u32(srcl) * d.c + gch + cil]; }}"
        );
        s.push_str("                    }\n");
    }
    s.push_str("                }\n");
    let _ = writeln!(s, "                sa[(rr * {}u) + kvi] = v;", t.sa_stride);
    s.push_str("            }\n");
    s.push_str("        }\n");
    emit_b_fast(&mut s, &t, b_vecs, "(wid.y * d.k + gk) * d.n + gn", "w");
    s.push_str("        workgroupBarrier();\n");
    emit_fma(&mut s, &t, true);
    s.push_str("        workgroupBarrier();\n");
    s.push_str("    }\n");
    s.push_str("    if (m_full) {\n");
    emit_pos_store(&mut s, &t, false);
    s.push_str("    } else {\n");
    emit_pos_store(&mut s, &t, true);
    s.push_str("    }\n");
    s.push_str("}\n");
    s
}

fn emit_pos_store(s: &mut String, t: &Tile, masked: bool) {
    for col in 0..t.tn_vec {
        let _ = writeln!(s, "        {{");
        let _ = writeln!(s, "            let gn = gn{col};");
        for row in 0..t.tm {
            let _ = writeln!(s, "            {{");
            let _ = writeln!(s, "                let gm = m0 + row{row};");
            let _ = writeln!(
                s,
                "                let y = vec4<f32>(gelu(acc_{row}_{col}.x), gelu(acc_{row}_{col}.y), gelu(acc_{row}_{col}.z), gelu(acc_{row}_{col}.w));"
            );
            s.push_str("                let base = gm * d.c + gch + gn;\n");
            if masked {
                s.push_str("                if (gm < d.m) {\n");
                let _ = writeln!(s, "                    o[base] = y.x;");
                let _ = writeln!(s, "                    o[base + 1u] = y.y;");
                let _ = writeln!(s, "                    o[base + 2u] = y.z;");
                let _ = writeln!(s, "                    o[base + 3u] = y.w;");
                s.push_str("                }\n");
            } else {
                let _ = writeln!(s, "                o[base] = y.x;");
                let _ = writeln!(s, "                o[base + 1u] = y.y;");
                let _ = writeln!(s, "                o[base + 2u] = y.z;");
                let _ = writeln!(s, "                o[base + 3u] = y.w;");
            }
            s.push_str("            }\n");
        }
        s.push_str("        }\n");
    }
}

fn store4(s: &mut String, dst: &str) {
    let _ = writeln!(s, "                    {dst}[base] = o.x;");
    let _ = writeln!(s, "                    {dst}[base + 1u] = o.y;");
    let _ = writeln!(s, "                    {dst}[base + 2u] = o.z;");
    let _ = writeln!(s, "                    {dst}[base + 3u] = o.w;");
}

fn store_masked(s: &mut String, dst: &str) {
    let _ = writeln!(s, "                    if (gn < d.n) {{ {dst}[base] = o.x; }}");
    let _ = writeln!(s, "                    if (gn + 1u < d.n) {{ {dst}[base + 1u] = o.y; }}");
    let _ = writeln!(s, "                    if (gn + 2u < d.n) {{ {dst}[base + 2u] = o.z; }}");
    let _ = writeln!(s, "                    if (gn + 3u < d.n) {{ {dst}[base + 3u] = o.w; }}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_build() {
        for (m, n, k) in [(64, 64, 32), (128, 64, 32), (128, 128, 32), (256, 64, 32), (64, 128, 32), (64, 64, 64)] {
            let src = gemm_bias_tiled(m, n, k, true);
            assert!(src.contains("workgroupBarrier"), "{m}x{n}x{k}");
            assert!(gemm_shared_bytes(m, n, k) <= 48 * 1024);
        }
        assert!(conv_gemm().contains("workgroupBarrier"));
        assert!(gemm_bias().contains("d.scale"));
        let pos = pos_conv();
        assert!(pos.contains("gelu(") && pos.contains("wid.y"));
        assert!(gemm_shared_bytes(POS_MT, POS_NT, POS_KS) <= 48 * 1024);
        for src in [gemm_bias(), conv_gemm(), pos, gemm_bias_tiled(128, 128, 32, false)] {
            let mut depth = 0i32;
            for c in src.chars() {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                assert!(depth >= 0, "{src}");
            }
            assert_eq!(depth, 0);
        }
    }
}
