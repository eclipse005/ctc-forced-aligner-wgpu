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
//!
//! Workgroup storage is not a constant here: every generator takes the
//! adapter's `max_compute_workgroup_storage_size` and validates its tile
//! against it while emitting the shader. DX12/GL report 32768 B where Vulkan
//! reports 49152 B, so a tile that fits one backend must not be assumed to
//! fit another.

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

pub fn gemm_bias(max_storage: u32) -> String {
    gemm_bias_tiled(MT, NT, KS, UNROLL_K, max_storage)
}

pub fn gemm_bias_tiled(mt: u32, nt: u32, ks: u32, unroll: bool, max_storage: u32) -> String {
    emit(mt, nt, ks, unroll, Kind::Gemm, false, max_storage)
}

pub fn conv_gemm(max_storage: u32) -> String {
    emit(MT, NT, KS, UNROLL_K, Kind::Conv, false, max_storage)
}

/// The encoder GEMM shader, per the buffering policy picked for this device.
///
/// Swept on the P104-100 (Pascal, 3 m fixture, all bit-identical): base
/// 128×64×32 single-buffered **48.2×**; db16 (128×64×16, double-buffered
/// banks) 39.6×; m64 (64×64×32, double-buffered) 33.2×; m64s (64×64×32,
/// single) 44.4×; n128s (64×128×32, single) 44.1×.
///
/// Both directions lose. Double-buffering prefetches the next K-tile behind
/// the current FMAs, but the doubled shared footprint drops the workgroups
/// per SM from 3 to 2, and occupancy hides more latency on Pascal than the
/// prefetch does. Shrinking the tile buys occupancy back but pays it away in
/// load instructions per FMA. So the tile stays 128×64×32, single-buffered —
/// `CTC_GEMM` exists to re-check that on a different machine, the same way
/// [`super::super::wav2vec2::forward::q_block_default`] does on the CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GemmVariant {
    /// 128×64×32, single-buffered — the default.
    Base,
    Db16,
    M64,
    /// 64×64×32, single-buffered: 17 KiB of shared memory, 5 workgroups/SM.
    M64s,
    /// 64×128×32, single-buffered: 25 KiB, half the B-tile traffic of base.
    N128s,
}

impl GemmVariant {
    pub const ALL: [GemmVariant; 5] = [
        GemmVariant::Base,
        GemmVariant::Db16,
        GemmVariant::M64,
        GemmVariant::M64s,
        GemmVariant::N128s,
    ];

    /// (mt, nt, ks, banks): one source of truth for a variant's geometry.
    /// The dispatch grid, both shader generators and the storage budget all
    /// read from here, so they cannot drift apart.
    fn shape(self) -> (u32, u32, u32, u32) {
        match self {
            GemmVariant::Base => (MT, NT, KS, 1),
            GemmVariant::Db16 => (MT, NT, 16, 2),
            GemmVariant::M64 => (64, NT, KS, 2),
            GemmVariant::M64s => (64, NT, KS, 1),
            GemmVariant::N128s => (64, 128, KS, 1),
        }
    }

    /// Workgroup storage the variant needs, both banks included.
    pub fn shared_bytes(self) -> u32 {
        let (mt, nt, ks, banks) = self.shape();
        banks * tile(mt, nt, ks).bytes
    }
}

/// Pick the tile for this adapter. `max_storage` is the device's
/// `max_compute_workgroup_storage_size`; a variant that does not fit is
/// refused here, at selection time, instead of producing an invalid compute
/// pipeline that only surfaces as a bind-group error during the forward.
pub fn pick_gemm_variant(max_storage: u32) -> GemmVariant {
    let variant = match std::env::var("CTC_GEMM") {
        Ok(v) => match v.as_str() {
            "base" => GemmVariant::Base,
            "db16" => GemmVariant::Db16,
            "m64" => GemmVariant::M64,
            "m64s" => GemmVariant::M64s,
            "n128s" => GemmVariant::N128s,
            other => panic!("unknown CTC_GEMM variant {other}"),
        },
        Err(_) => GemmVariant::Base,
    };
    let need = variant.shared_bytes();
    assert!(
        need <= max_storage,
        "CTC_GEMM={variant:?} needs {need} B of workgroup storage, adapter allows \
         {max_storage} B; variants that fit here: {:?}",
        GemmVariant::ALL
            .iter()
            .copied()
            .filter(|v| v.shared_bytes() <= max_storage)
            .collect::<Vec<_>>()
    );
    variant
}

pub fn gemm_variant(variant: GemmVariant, max_storage: u32) -> String {
    let (mt, nt, ks, banks) = variant.shape();
    emit(mt, nt, ks, UNROLL_K, Kind::Gemm, banks == 2, max_storage)
}

pub fn conv_gemm_variant(variant: GemmVariant, max_storage: u32) -> String {
    let (mt, nt, ks, banks) = variant.shape();
    emit(mt, nt, ks, UNROLL_K, Kind::Conv, banks == 2, max_storage)
}

/// The (mt, nt) tile a variant was built with; dispatch grids tile with the
/// same numbers or part of the output is never covered.
pub fn gemm_tile(variant: GemmVariant) -> (u32, u32) {
    let (mt, nt, _, _) = variant.shape();
    (mt, nt)
}

pub fn pos_conv(max_storage: u32) -> String {
    emit_pos(POS_MT, POS_NT, POS_KS, max_storage)
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
    // Pure geometry. Whether `bytes` is legal depends on the adapter, so the
    // check lives in `emit`/`emit_pos` where the device budget is known.
    Tile { mt, nt, ks, tm, tn_vec, sa_stride, sb_stride, bytes }
}

#[derive(Clone, Copy)]
enum Kind {
    Gemm,
    Conv,
}

fn emit(mt: u32, nt: u32, ks: u32, unroll: bool, kind: Kind, dbuf: bool, max_storage: u32) -> String {
    let t = tile(mt, nt, ks);
    let shared = if dbuf { 2 * t.bytes } else { t.bytes };
    assert!(
        shared <= max_storage,
        "{mt}x{nt}x{ks} GEMM needs {shared} B of workgroup storage, adapter allows {max_storage} B"
    );
    let mut s = String::with_capacity(48 * 1024);
    match kind {
        Kind::Gemm => s.push_str(
            r#"struct Dims {
    m: u32, n: u32, k: u32,
    a_stride: u32, b_stride: u32, c_stride: u32,
    a_off: u32, b_off: u32, c_off: u32,
    a_z: u32, b_z: u32, c_z: u32,
    epilogue: u32,
    scale: f32,
    a_vec: u32, b_vec: u32, _p0: u32,
}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
@group(0) @binding(5) var<storage, read> res: array<f32>;
// the same buffers as bindings 0/1, viewed as vec4s: when a tile's base
// address is 16-byte aligned (host-checked, per dispatch), one 128-bit load
// replaces four 32-bit loads
@group(0) @binding(6) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> b4: array<vec4<f32>>;

// epilogue == 1 fuses the FFN gelu; == 2 adds the residual stream
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
"#,
        ),
        Kind::Conv => s.push_str(
            r#"struct Dims {
    m: u32, n: u32, k: u32,
    c_in: u32, stride: u32, t_in: u32,
    a_vec: u32, b_vec: u32,
}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> wt4: array<vec4<f32>>;
"#,
        ),
    }
    let banks: &[(u32, u32)] = if dbuf { &[(0, 0), (1, 1)] } else { &[(0, 0)] };
    for (ai, _) in banks {
        let _ = writeln!(s, "var<workgroup> sa{ai}: array<vec4<f32>, {}u>;", t.mt * t.sa_stride);
    }
    for (_, bi) in banks {
        let _ = writeln!(s, "var<workgroup> sb{bi}: array<vec4<f32>, {}u>;", t.ks * t.sb_stride);
    }
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
    if matches!(kind, Kind::Gemm) {
        // per-head batched dispatches: workgroup z picks the head, the
        // *_z strides step a_off / b_off / c_off between heads
        s.push_str(
            "    let a_off = d.a_off + wid.z * d.a_z;\n    let b_off = d.b_off + wid.z * d.b_z;\n    let c_off = d.c_off + wid.z * d.c_z;\n",
        );
    }
    let _ = writeln!(s, "    let ktiles = (d.k + {}u - 1u) / {}u;", t.ks, t.ks);
    if !dbuf {
        s.push_str("    for (var kt = 0u; kt < ktiles; kt = kt + 1u) {\n");
        let _ = writeln!(s, "        let k0 = kt * {}u;", t.ks);
        let _ = writeln!(s, "        let k_full = k0 + {}u <= d.k;", t.ks);
        emit_loads(&mut s, &t, kind, "sa0", "sb0");
        s.push_str("        workgroupBarrier();\n");
        emit_fma(&mut s, &t, unroll, "sa0", "sb0");
        s.push_str("        workgroupBarrier();\n");
        s.push_str("    }\n");
    } else {
        // Double-buffered: tile 0 lands in bank 0, then each iteration loads
        // the next tile into the other bank while the FMAs drain the current
        // one. One barrier per tile: it retires the bank the FMAs just read
        // and publishes the prefetch the FMAs will read next iteration.
        s.push_str("    {\n        let k0 = 0u;\n");
        let _ = writeln!(s, "        let k_full = k0 + {}u <= d.k;", t.ks);
        emit_loads(&mut s, &t, kind, "sa0", "sb0");
        s.push_str("    }\n");
        s.push_str("    workgroupBarrier();\n");
        s.push_str("    for (var kt = 0u; kt < ktiles; kt = kt + 2u) {\n");
        s.push_str("        if (kt + 1u < ktiles) {\n");
        let _ = writeln!(s, "            let k0 = (kt + 1u) * {}u;", t.ks);
        let _ = writeln!(s, "            let k_full = k0 + {}u <= d.k;", t.ks);
        emit_loads(&mut s, &t, kind, "sa1", "sb1");
        s.push_str("        }\n");
        emit_fma(&mut s, &t, unroll, "sa0", "sb0");
        s.push_str("        workgroupBarrier();\n");
        s.push_str("        if (kt + 1u < ktiles) {\n");
        s.push_str("            if (kt + 2u < ktiles) {\n");
        let _ = writeln!(s, "                let k0 = (kt + 2u) * {}u;", t.ks);
        let _ = writeln!(s, "                let k_full = k0 + {}u <= d.k;", t.ks);
        emit_loads(&mut s, &t, kind, "sa0", "sb0");
        s.push_str("            }\n");
        emit_fma(&mut s, &t, unroll, "sa1", "sb1");
        s.push_str("            workgroupBarrier();\n");
        s.push_str("        }\n");
        s.push_str("    }\n");
    }
    emit_store(&mut s, &t, kind);
    s.push_str("}\n");
    s
}

fn emit_loads(s: &mut String, t: &Tile, kind: Kind, sa: &str, sb: &str) {
    let a_vecs = t.mt * (t.ks / 4);
    let b_vecs = t.ks * t.sb_stride;
    let kv = t.ks / 4;
    match kind {
        Kind::Gemm => {
            s.push_str("        if (m_full && k_full) {\n");
            emit_a_fast(s, t, a_vecs, kv, sa);
            s.push_str("        } else {\n");
            emit_a_slow(s, t, a_vecs, kv, sa);
            s.push_str("        }\n");
            s.push_str("        if (n_full && k_full) {\n");
            emit_b_fast(s, t, b_vecs, "b_off + gk * d.b_stride + gn", "b", sb, Some("b4"));
            s.push_str("        } else {\n");
            emit_b_slow(s, t, b_vecs, "b_off + gk * d.b_stride + gn", "b", sb);
            s.push_str("        }\n");
        }
        Kind::Conv => {
            // im2col crosses a tap when c_in is not a multiple of 4, so the
            // guarded gather is used for every K tile. Conv is a few layers
            // once per chunk; the encoder GEMMs dominate.
            emit_im2col(s, t, a_vecs, kv, sa);
            s.push_str("        if (n_full && k_full) {\n");
            emit_b_fast(s, t, b_vecs, "gk * d.n + gn", "wt", sb, Some("wt4"));
            s.push_str("        } else {\n");
            emit_b_slow(s, t, b_vecs, "gk * d.n + gn", "wt", sb);
            s.push_str("        }\n");
        }
    }
}

fn emit_a_fast(s: &mut String, t: &Tile, a_vecs: u32, kv: u32, sa: &str) {
    let _ = writeln!(s, "            for (var i = tid; i < {a_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {kv}u;");
    let _ = writeln!(s, "                let kvi = i % {kv}u;");
    s.push_str("                let base = a_off + (m0 + rr) * d.a_stride + (k0 + kvi * 4u);\n");
    s.push_str("                if (d.a_vec == 1u) {\n");
    let _ = writeln!(
        s,
        "                    {sa}[(rr * {}u) + kvi] = a4[base / 4u];",
        t.sa_stride
    );
    s.push_str("                } else {\n");
    let _ = writeln!(
        s,
        "                    {sa}[(rr * {}u) + kvi] = vec4<f32>(a[base], a[base + 1u], a[base + 2u], a[base + 3u]);",
        t.sa_stride
    );
    s.push_str("                }\n");
    s.push_str("            }\n");
}

fn emit_a_slow(s: &mut String, t: &Tile, a_vecs: u32, kv: u32, sa: &str) {
    let _ = writeln!(s, "            for (var i = tid; i < {a_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {kv}u;");
    let _ = writeln!(s, "                let kvi = i % {kv}u;");
    s.push_str("                let gm = m0 + rr;\n");
    s.push_str("                let gk = k0 + kvi * 4u;\n");
    s.push_str("                var v = vec4<f32>(0.0);\n");
    s.push_str("                if (gm < d.m && gk + 3u < d.k) {\n");
    s.push_str("                    let base = a_off + gm * d.a_stride + gk;\n");
    s.push_str("                    v = vec4<f32>(a[base], a[base + 1u], a[base + 2u], a[base + 3u]);\n");
    s.push_str("                } else if (gm < d.m && gk < d.k) {\n");
    s.push_str("                    let base = a_off + gm * d.a_stride + gk;\n");
    s.push_str("                    v.x = a[base];\n");
    s.push_str("                    if (gk + 1u < d.k) { v.y = a[base + 1u]; }\n");
    s.push_str("                    if (gk + 2u < d.k) { v.z = a[base + 2u]; }\n");
    s.push_str("                    if (gk + 3u < d.k) { v.w = a[base + 3u]; }\n");
    s.push_str("                }\n");
    let _ = writeln!(s, "                {sa}[(rr * {}) + kvi] = v;", t.sa_stride);
    s.push_str("            }\n");
}

/// `vec4_view` is `Some("b4")` for the GEMM/conv kernels (whose bindings
/// carry a vec4 alias of `buf`) and `None` for the pos-conv kernel, which has
/// no such binding: the branch is not emitted at all, since WGSL validates
/// every referenced binding even under a dead condition.
fn emit_b_fast(
    s: &mut String,
    t: &Tile,
    b_vecs: u32,
    base: &str,
    buf: &str,
    sb: &str,
    vec4_view: Option<&str>,
) {
    let _ = writeln!(s, "            for (var i = tid; i < {b_vecs}u; i = i + 256u) {{");
    let _ = writeln!(s, "                let rr = i / {}u;", t.sb_stride);
    let _ = writeln!(s, "                let cv = i % {}u;", t.sb_stride);
    s.push_str("                let gk = k0 + rr;\n");
    s.push_str("                let gn = n0 + cv * 4u;\n");
    let _ = writeln!(s, "                let base = {base};");
    if let Some(view) = vec4_view {
        s.push_str("                if (d.b_vec == 1u) {\n");
        let _ = writeln!(
            s,
            "                    {sb}[(rr * {}u) + cv] = {view}[base / 4u];",
            t.sb_stride
        );
        s.push_str("                } else {\n");
    }
    let _ = writeln!(
        s,
        "                    {sb}[(rr * {}u) + cv] = vec4<f32>({buf}[base], {buf}[base + 1u], {buf}[base + 2u], {buf}[base + 3u]);",
        t.sb_stride
    );
    if vec4_view.is_some() {
        s.push_str("                }\n");
    }
    s.push_str("            }\n");
}

fn emit_b_slow(s: &mut String, t: &Tile, b_vecs: u32, base: &str, buf: &str, sb: &str) {
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
    let _ = writeln!(s, "                {sb}[(rr * {}u) + cv] = v;", t.sb_stride);
    s.push_str("            }\n");
}

fn emit_im2col(s: &mut String, t: &Tile, a_vecs: u32, kv: u32, sa: &str) {
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
    s.push_str("                        if (d.a_vec == 1u) {\n");
    s.push_str("                            v = x4[base / 4u];\n");
    s.push_str("                        } else {\n");
    s.push_str("                            v = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);\n");
    s.push_str("                        }\n");
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
    let _ = writeln!(s, "                {sa}[(rr * {}u) + kvi] = v;", t.sa_stride);
    s.push_str("            }\n");
}

fn emit_fma(s: &mut String, t: &Tile, unroll: bool, sa: &str, sb: &str) {
    if unroll {
        for kv in 0..t.ks / 4 {
            emit_fma_step(s, t, &format!("{kv}u"), sa, sb);
        }
    } else {
        let _ = writeln!(s, "        for (var kvi = 0u; kvi < {}u; kvi = kvi + 1u) {{", t.ks / 4);
        emit_fma_step(s, t, "kvi", sa, sb);
        s.push_str("        }\n");
    }
}

fn emit_fma_step(s: &mut String, t: &Tile, kv: &str, sa: &str, sb: &str) {
    s.push_str("        {\n");
    for lane in 0..4u32 {
        for col in 0..t.tn_vec {
            let _ = writeln!(
                s,
                "            let b_{lane}_{col} = {sb}[((({kv}) * 4u + {lane}u) * {}u) + lid.x + {}u];",
                t.sb_stride,
                col * 16
            );
        }
    }
    for row in 0..t.tm {
        let _ = writeln!(
            s,
            "            let a_{row} = {sa}[(row{row} * {}u) + ({kv})];",
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
    let (dst, scale, base, gemm) = match kind {
        Kind::Gemm => ("c", "d.scale", "c_off + gm * d.c_stride + gn", true),
        Kind::Conv => ("out", "1.0", "gm * d.n + gn", false),
    };
    s.push_str("    if (m_full && n_full) {\n");
    emit_store_rows(s, t, dst, scale, base, false, gemm);
    s.push_str("    } else {\n");
    emit_store_rows(s, t, dst, scale, base, true, gemm);
    s.push_str("    }\n");
}

fn emit_store_rows(
    s: &mut String,
    t: &Tile,
    dst: &str,
    scale: &str,
    base: &str,
    masked: bool,
    gemm: bool,
) {
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
            let _ = writeln!(s, "                let base = {base};");
            let _ = writeln!(s, "                var o = acc_{row}_{col} * {scale} + biasv;");
            if gemm {
                s.push_str("                if (d.epilogue == 1u) {
");
                s.push_str("                    o = gelu4(o);
");
                s.push_str("                } else if (d.epilogue == 2u) {
");
                s.push_str("                    var resv = vec4<f32>(0.0);
");
                s.push_str("                    if (gm < d.m && gn + 3u < d.n) {
");
                s.push_str("                        resv = vec4<f32>(res[base], res[base + 1u], res[base + 2u], res[base + 3u]);
");
                s.push_str("                    } else if (gm < d.m) {
");
                s.push_str("                        if (gn < d.n) { resv.x = res[base]; }
");
                s.push_str("                        if (gn + 1u < d.n) { resv.y = res[base + 1u]; }
");
                s.push_str("                        if (gn + 2u < d.n) { resv.z = res[base + 2u]; }
");
                s.push_str("                        if (gn + 3u < d.n) { resv.w = res[base + 3u]; }
");
                s.push_str("                    }
");
                s.push_str("                    o = o + resv;
");
                s.push_str("                }
");
            }
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
fn emit_pos(mt: u32, nt: u32, ks: u32, max_storage: u32) -> String {
    let t = tile(mt, nt, ks);
    assert!(
        t.bytes <= max_storage,
        "{mt}x{nt}x{ks} pos conv needs {} B of workgroup storage, adapter allows {max_storage} B",
        t.bytes
    );
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
    emit_b_fast(&mut s, &t, b_vecs, "(wid.y * d.k + gk) * d.n + gn", "w", "sb", None);
    s.push_str("        workgroupBarrier();\n");
    emit_fma(&mut s, &t, true, "sa", "sb");
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

    /// WebGPU's general ceiling, used only to prove the geometry helpers are
    /// sane. Real runs pass the adapter's own limit instead.
    const TEST_STORAGE: u32 = 48 * 1024;

    #[test]
    fn tiles_build() {
        for (m, n, k) in [(64, 64, 32), (128, 64, 32), (128, 128, 32), (256, 64, 32), (64, 128, 32), (64, 64, 64)] {
            let src = gemm_bias_tiled(m, n, k, true, TEST_STORAGE);
            assert!(src.contains("workgroupBarrier"), "{m}x{n}x{k}");
            assert!(gemm_shared_bytes(m, n, k) <= 48 * 1024);
        }
        assert!(conv_gemm(TEST_STORAGE).contains("workgroupBarrier"));
        assert!(gemm_bias(TEST_STORAGE).contains("d.scale"));
        // the double-buffered variants share the arithmetic but carry two
        // banks: their source must stay balanced and inside the budget
        for variant in [GemmVariant::Db16, GemmVariant::M64] {
            for src in [
                gemm_variant(variant, TEST_STORAGE),
                conv_gemm_variant(variant, TEST_STORAGE),
            ] {
                assert!(src.contains("sa0") && src.contains("sa1"), "{variant:?}");
                assert!(src.contains("workgroupBarrier"), "{variant:?}");
                let mut depth = 0i32;
                for c in src.chars() {
                    match c {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
                    assert!(depth >= 0, "{variant:?}");
                }
                assert_eq!(depth, 0, "{variant:?}");
            }
        }
        assert_eq!(2 * gemm_shared_bytes(MT, NT, 16), 45_056, "db16 banks");
        assert_eq!(2 * gemm_shared_bytes(64, NT, KS), 34_816, "m64 banks");
        let pos = pos_conv(TEST_STORAGE);
        assert!(pos.contains("gelu(") && pos.contains("wid.y"));
        assert!(gemm_shared_bytes(POS_MT, POS_NT, POS_KS) <= 48 * 1024);
        for src in [
            gemm_bias(TEST_STORAGE),
            conv_gemm(TEST_STORAGE),
            pos,
            gemm_bias_tiled(128, 128, 32, false, TEST_STORAGE),
        ] {
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

    #[test]
    fn adapter_budget_is_enforced_where_the_kernel_is_made() {
        // DX12/GL report 32768 B on this machine; Vulkan reports 49152 B.
        const DX12: u32 = 32 * 1024;
        assert!(GemmVariant::Base.shared_bytes() <= DX12);
        assert_eq!(GemmVariant::Db16.shared_bytes(), 45_056);
        assert_eq!(GemmVariant::M64.shared_bytes(), 34_816);
        let fits: Vec<_> = GemmVariant::ALL
            .iter()
            .copied()
            .filter(|v| v.shared_bytes() <= DX12)
            .collect();
        assert_eq!(
            fits,
            vec![GemmVariant::Base, GemmVariant::M64s, GemmVariant::N128s]
        );

        // An oversized tile must be refused while generating the shader, with
        // the real numbers in the message — not emitted and rejected later by
        // the driver as an unrelated bind-group error.
        let err = std::panic::catch_unwind(|| gemm_variant(GemmVariant::Db16, DX12))
            .expect_err("db16 must not compile on a 32 KiB adapter");
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(msg.contains("32768") && msg.contains("45056"), "{msg}");
        assert!(gemm_variant(GemmVariant::Base, DX12).contains("workgroupBarrier"));

        // The positional conv takes the same budget.
        assert!(std::panic::catch_unwind(|| pos_conv(1024)).is_err());
    }
}
