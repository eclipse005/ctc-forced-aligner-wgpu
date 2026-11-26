//! Wav2Vec2 forward on wgpu (fp32).  Same computation graph as the CPU twin
//! (`wav2vec2.rs`); weights are pre-transposed to (K, N) at load so every
//! GEMM's B tile loads contiguously.
//!
//! Per 30 s chunk (T ≈ 1500 frames): conv stack (implicit-GEMM conv), feature
//! projection, depthwise positional conv, 24 pre-LN encoder layers with the
//! attention scores materialised as 16 per-head GEMMs, LM head, log_softmax.

use std::path::Path;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::config::Wav2Vec2Config;
use crate::gpu::{BulkUpload, Gpu};
use crate::shaders;
use crate::weights::get_f32;
use rayon::prelude::*;

const WG: u32 = 256;
const WG_LIMIT: u32 = 65535;

/// Workgroup counts in one dimension cannot exceed 65535. Row-wise kernels
/// read `workgroup_id.x + workgroup_id.y * 65535`.
fn row_grid(rows: u32) -> (u32, u32) {
    if rows <= WG_LIMIT {
        (rows.max(1), 1)
    } else {
        (WG_LIMIT, rows.div_ceil(WG_LIMIT))
    }
}

/// Same fold for flat one-thread-per-element kernels: `x` takes the first
/// 65535 workgroups, `y` the rest, and the shader rebuilds the element index
/// from `(workgroup_id.x + workgroup_id.y * 65535) * WG +
/// local_invocation_id.x`.  Needed wherever the element count scales with the
/// transcript (the trellis gather is T*S) rather than with `t` alone.
fn flat_grid(total: u32) -> (u32, u32) {
    let groups = total.div_ceil(WG).max(1);
    if groups <= WG_LIMIT {
        (groups, 1)
    } else {
        (WG_LIMIT, groups.div_ceil(WG_LIMIT))
    }
}

/// 1 when a vec4 view of one operand is 16-byte aligned for this dispatch:
/// the base offset, the row stride and the per-head z step are all multiples
/// of 4, so `view[base / 4]` addresses every tile exactly. Otherwise the
/// shader falls back to four scalar loads (same values, same order).
fn vec4_ok(off: u32, stride: u32, z: u32) -> u32 {
    u32::from(off.is_multiple_of(4) && stride.is_multiple_of(4) && z.is_multiple_of(4))
}

/// The little-endian bytes of `data`: borrowed in place on the little-endian
/// hosts this runs on, converted only where the byte order differs, so a
/// multi-GiB weight upload costs no temporary heap.
fn le_bytes(data: &[f32]) -> std::borrow::Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    {
        std::borrow::Cow::Borrowed(bytemuck::cast_slice(data))
    }
    #[cfg(not(target_endian = "little"))]
    {
        std::borrow::Cow::Owned(data.iter().flat_map(|f| f.to_le_bytes()).collect())
    }
}

/// `[out, in_pg, taps]` → `[group, tap * in_pg + ci, oc]`, oc contiguous.
fn transpose_pos(w: &[f32], out_c: usize, in_pg: usize, taps: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_c * in_pg * taps);
    let k_total = in_pg * taps;
    let mut out = vec![0f32; w.len()];
    out.par_chunks_mut(in_pg)
        .enumerate()
        .for_each(|(idx, dst)| {
            let g = idx / k_total;
            let kval = idx % k_total;
            let tk = kval / in_pg;
            let ci = kval % in_pg;
            for oc in 0..in_pg {
                let o = g * in_pg + oc;
                dst[oc] = w[(o * in_pg + ci) * taps + tk];
            }
        });
    out
}

/// Encoder frame rate for the `--window` advice: the conv stack strides 320
/// samples of 16 kHz audio per output frame.
const FRAMES_PER_SEC: u64 = 50;

/// The largest whole-second window whose `scores` and `logits` both fit the
/// adapter's binding limit.
///
/// `scores` is `heads · t²` f32 and so falls with the square of the window,
/// `logits` is `t · vocab` f32 and falls linearly; each gives its own ceiling
/// and the smaller wins. Kept at 1 so the advice is always actionable.
fn recommended_window_secs(
    t: usize,
    scores_bytes: u64,
    logits_bytes: u64,
    max_binding: u64,
) -> u64 {
    let t = t.max(1) as f64;
    let max = max_binding as f64;
    let linear = if logits_bytes == 0 {
        t
    } else {
        t * (max / logits_bytes as f64).min(1.0)
    };
    let quadratic = if scores_bytes == 0 {
        t
    } else {
        t * (max / scores_bytes as f64).min(1.0).sqrt()
    };
    let frames = linear.min(quadratic);
    ((frames / FRAMES_PER_SEC as f64) as u64).max(1)
}

/// Refuse a window whose attention scores or logits cannot be bound before a
/// single buffer is allocated.
///
/// A buffer bigger than `max_storage_buffer_binding_size` cannot be bound at
/// all, so the failure would otherwise surface as a validation error from
/// inside the first dispatch, with nothing pointing at the window length.
fn check_binding_limits(
    t: usize,
    scores_bytes: u64,
    logits_bytes: u64,
    max_binding: u64,
) -> Result<()> {
    let window = recommended_window_secs(t, scores_bytes, logits_bytes, max_binding);
    for (what, need) in [("attention scores", scores_bytes), ("logits", logits_bytes)] {
        if need > max_binding {
            anyhow::bail!(
                "a {t}-frame window needs {need} B for the {what}, over this adapter's \
                 max_storage_buffer_binding_size of {max_binding} B; use --window {window} \
                 (seconds) or less"
            );
        }
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GemmDims {
    m: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    b_stride: u32,
    c_stride: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    a_z: u32,
    b_z: u32,
    c_z: u32,
    epilogue: u32,
    scale: f32,
    /// 1 when every vec4 view of `a` (`a4[base/4]`) is 16-byte aligned for
    /// this dispatch — off, stride and the per-head z step are all multiples
    /// of 4. The scores GEMM's B (stride t) and the PV GEMM's A fail this.
    a_vec: u32,
    b_vec: u32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ConvDims {
    m: u32,
    n: u32,
    k: u32,
    c_in: u32,
    stride: u32,
    t_in: u32,
    /// c_in (512) and n (512) are multiples of 4, so both vec4 views are
    /// always aligned for the conv stack.
    a_vec: u32,
    b_vec: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct PosDims {
    m: u32,
    n: u32,
    k: u32,
    c: u32,
    in_pg: u32,
    pad: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Cfg4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// [`shaders::transpose`]'s uniform. Separate from [`Cfg4`] because the output
/// pitch is independent of its width: `kt` is `(hidden, t_pad)`, so the rows
/// the attention GEMM reads with `vec4` loads start every `t_pad` elements
/// while only `t` of them hold data.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct TransposeCfg {
    rows: u32,
    cols: u32,
    a_stride: u32,
    a_off: u32,
    o_stride: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// [`shaders::flash_attn`]'s uniform: the frame count, the head count, the
/// `vec4` width of one hidden row and the attention scale.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
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

const _: () = assert!(std::mem::size_of::<FlashAttnCfg>() == 32);

/// One recorded dispatch. Uniform bytes live in a side buffer; the bind group
/// already points at its slot, so replay is just set/dispatch.
/// Cloned across chunks of the same length: the buffers and the grid don't change.
#[derive(Clone)]
struct Job {
    pipe: usize,
    bg: wgpu::BindGroup,
    gx: u32,
    gy: u32,
    gz: u32,
}

/// One encoder layer's prepared weights, in upload order.
///
/// The biases travel with their weights: unpacking a `(Vec<f32>, Vec<f32>)`
/// tuple to keep only the bias is what made the loader read `k_proj`,
/// `v_proj` and `final_layer_norm` twice per layer.
struct LayerWeights {
    qkv_wt: Vec<f32>,
    qkv_b: Vec<f32>,
    out_wt: Vec<f32>,
    out_b: Vec<f32>,
    ff1_wt: Vec<f32>,
    ff1_b: Vec<f32>,
    ff2_wt: Vec<f32>,
    ff2_b: Vec<f32>,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
}

/// What `CTC_TILE64` says about the short-window `64×64×32` tile.
///
/// It only means anything when the base tile is wider than 64 rows (a 64-row
/// main tile leaves nothing to gain) and the `M64s` pipeline fits the
/// adapter's workgroup storage. `on`/`1`/`true` asks for it explicitly, which
/// is how it gets benchmarked on a device without `SHADER_F16`; `auto` — the
/// default — enables it only where `f16` tiles make the extra pipeline cheap,
/// so a Pascal card compiles exactly what it had before.
fn resolve_tile64(
    has_f16: bool,
    main_mt: u32,
    max_storage: u32,
    use_f16: bool,
    env_val: Option<&str>,
) -> bool {
    let possible =
        main_mt > 64 && shaders::GemmVariant::M64s.shared_bytes_with(use_f16) <= max_storage;
    if !possible {
        return false;
    }
    match env_val
        .map(str::trim)
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "0" | "false" | "off" => false,
        "1" | "true" | "on" => true,
        _ => has_f16,
    }
}

/// Pick the `f32`-shared-tile GEMM variant used by materialised attention
/// (`scores` / `pv`) when the primary linear GEMM narrowed its tiles to `f16`.
///
/// An `f16` adapter with 16 KiB of workgroup storage can pick `Base` (13,312 B
/// in f16) and a 32 KiB adapter can pick `Db16` (22,528 B in f16), neither of
/// which fits as `f32`. Falling back to the fastest `f32` tile that fits keeps
/// model load from panicking on those adapters.
fn resolve_f32_gemm_variant(
    primary: shaders::GemmVariant,
    max_storage: u32,
) -> shaders::GemmVariant {
    if primary.shared_bytes_with(false) <= max_storage {
        primary
    } else {
        shaders::pick_gemm_variant_for_with(max_storage, false, None)
    }
}

/// What `CTC_FLASH_ATTN` says about the fused attention path.
///
/// `head_dim` is a hard gate: the kernel is written for the 64-wide head every
/// Wav2Vec2 checkpoint uses. `auto` keeps the tuned materialised path where it
/// fits and performs (this P104 at 30 s) and engages wherever it cannot: a
/// `scores` tensor past the binding limit, a window past the size cap, or a
/// modern adapter with `f16` shared tiles.
fn resolve_flash_attn(
    env_val: Option<&str>,
    head_dim: usize,
    scores_bytes: u64,
    max_binding: u64,
    scores_exceeds_cap: bool,
    has_f16: bool,
) -> bool {
    if head_dim != shaders::FLASH_HEAD_DIM as usize {
        return false;
    }
    match env_val
        .map(str::trim)
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "0" | "false" | "off" => false,
        "1" | "true" | "on" => true,
        _ => scores_bytes > max_binding || scores_exceeds_cap || has_f16,
    }
}

struct Pipe {
    gemm: wgpu::ComputePipeline,
    /// The same schedule with `f32` shared tiles. Equal to `gemm` when the
    /// device has no `f16`: attention scores are read back into an `f32`
    /// epilogue and an `f16` rounded score would move softmax weights.
    gemm_f32: wgpu::ComputePipeline,
    conv_gemm: wgpu::ComputePipeline,
    conv0: wgpu::ComputePipeline,
    pos_conv: wgpu::ComputePipeline,
    ln: wgpu::ComputePipeline,
    ln_sd: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    log_softmax: wgpu::ComputePipeline,
    transpose: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    /// `None` unless `resolve_tile64` asked for the short-window tile.
    gemm64: Option<wgpu::ComputePipeline>,
    gemm64_f32: Option<wgpu::ComputePipeline>,
    /// `None` on an adapter whose workgroup storage cannot hold the kernel's
    /// 16 KiB tile; the materialised attention path stays in use there.
    flash_attn: Option<wgpu::ComputePipeline>,
}

const P_GEMM: usize = 0;
const P_CONV_GEMM: usize = 1;
const P_CONV0: usize = 2;
const P_POS: usize = 3;
const P_LN: usize = 4;
const P_LN_SD: usize = 5;
const P_ADD: usize = 6;
const P_SOFTMAX: usize = 7;
const P_LOGSOFTMAX: usize = 8;
const P_TRANSPOSE: usize = 9;
const P_GATHER: usize = 10;
const P_GEMM64: usize = 11;
const P_GEMM_F32: usize = 12;
const P_GEMM64_F32: usize = 13;
const P_FLASH_ATTN: usize = 14;

/// `CTC_PROFILE=2` buckets that are not one pipeline. The main GEMM schedule
/// serves the projections, `scores` and `pv`, and the profile splits those by
/// dispatch shape.
const BUCKET_SCORES: usize = 15;
const BUCKET_PV: usize = 16;
const BUCKET_MAIN: usize = 17;
/// One profile slot per pipeline id plus the synthetic buckets; the
/// `CTC_PROFILE=2` name table must have exactly this many entries.
const PROFILE_SLOTS: usize = BUCKET_MAIN + 1;

/// The uniform binding index of each pipeline, and whether its uniform buffer
/// is shared (one big buffer + per-dispatch offsets).
fn uni_binding(p: usize) -> u32 {
    match p {
        P_GEMM | P_CONV_GEMM | P_CONV0 | P_POS | P_LN_SD | P_GEMM64 | P_GEMM_F32 | P_GEMM64_F32 => {
            4
        }
        P_LN | P_ADD => 3,
        P_SOFTMAX | P_LOGSOFTMAX => 1,
        P_TRANSPOSE | P_FLASH_ATTN => 2,
        P_GATHER => 3,
        _ => unreachable!(),
    }
}

pub(crate) struct GpuModel {
    gpu: Gpu,
    pub cfg: Wav2Vec2Config,
    pipes: Pipe,
    /// The picked GEMM tile (matches the compiled pipelines): every dispatch
    /// grid is computed from these, so a variant with a different M tile
    /// covers the same output with a different grid.
    gemm_mt: u32,
    gemm_nt: u32,
    /// Tile of the `f32`-shared attention GEMM (`scores` / `pv`). Matches
    /// `(gemm_mt, gemm_nt)` whenever the primary variant also fits as `f32`;
    /// narrower when an `f16` tile (e.g. `Base` at 16 KiB) would overflow `f32`.
    gemm_f32_mt: u32,
    gemm_f32_nt: u32,
    /// The positional conv's M tile, likewise read off the compiled kernel:
    /// an adapter under 26 KiB of workgroup storage runs the 64-frame tile.
    pos_mt: u32,
    /// Staging recycled across windows. Two sets: the one being copied and
    /// the one being collected. A window above 16 MB uses the transfer queue
    /// when the device has one. The hidden readback (~7 MB) stays on wgpu's
    /// cached MAP_READ buffers.
    readback: std::sync::Mutex<Vec<StagingSet>>,
    /// Transfer-queue pool. After `readback` so the staging sets drop first:
    /// they wait out their fence, then this destroys the command pool.
    xfer: std::sync::OnceLock<Option<crate::xfer::Engine>>,
    conv0_w: wgpu::Buffer,
    conv_wt: Vec<wgpu::Buffer>,
    conv_b: Vec<wgpu::Buffer>,
    conv_ln_w: Vec<wgpu::Buffer>,
    conv_ln_b: Vec<wgpu::Buffer>,
    fp_ln_w: wgpu::Buffer,
    fp_ln_b: wgpu::Buffer,
    fp_wt: wgpu::Buffer,
    fp_b: wgpu::Buffer,
    pos_w: wgpu::Buffer,
    pos_b: wgpu::Buffer,
    qkv_wt: Vec<wgpu::Buffer>,
    qkv_b: Vec<wgpu::Buffer>,
    out_wt: Vec<wgpu::Buffer>,
    out_b: Vec<wgpu::Buffer>,
    ff1_wt: Vec<wgpu::Buffer>,
    ff1_b: Vec<wgpu::Buffer>,
    ff2_wt: Vec<wgpu::Buffer>,
    ff2_b: Vec<wgpu::Buffer>,
    ln1_w: Vec<wgpu::Buffer>,
    ln1_b: Vec<wgpu::Buffer>,
    ln2_w: Vec<wgpu::Buffer>,
    ln2_b: Vec<wgpu::Buffer>,
    final_ln_w: wgpu::Buffer,
    final_ln_b: wgpu::Buffer,
    lm_wt: wgpu::Buffer,
    lm_b: wgpu::Buffer,
    zeros: wgpu::Buffer,
    /// Expanded trellis labels for the gathered alignment readback.
    labels: wgpu::Buffer,
    /// 1 at the `<star>` states' indices, 0 elsewhere; the gather kernel
    /// writes `CTC_STAR_SCORE` for those columns (see `shaders::gather`).
    star_flags: wgpu::Buffer,
    /// Activation workspace reused across chunks of the same length.
    scratch: std::sync::Mutex<Option<Scratch>>,
}

impl GpuModel {
    pub fn load(model_dir: &Path, selector: crate::gpu::DeviceSelector) -> Result<Self> {
        let gpu = match selector {
            crate::gpu::DeviceSelector::Cpu => anyhow::bail!("Cpu selector reaches the GPU tower"),
            sel => pollster::block_on(Gpu::new_with(sel))?,
        };
        let cfg = Wav2Vec2Config::load(model_dir)?;
        let tensors = crate::weights::load_tensors(model_dir)?;

        // ---- pos-conv weight norm, resolved exactly like the CPU twin
        let g = get_f32(
            &tensors,
            "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        )?;
        let v = get_f32(
            &tensors,
            "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        )?;
        let (out_c, in_pg, k) = (v.1[0], v.1[1], v.1[2]);
        let mut pos_w = v.0.clone();
        for kk in 0..k {
            let gv = g.0[kk];
            let mut norm = 0.0f32;
            for o in 0..out_c {
                for ii in 0..in_pg {
                    let x = v.0[(o * in_pg + ii) * k + kk];
                    norm += x * x;
                }
            }
            let inv = gv / norm.sqrt();
            for o in 0..out_c {
                for ii in 0..in_pg {
                    pos_w[(o * in_pg + ii) * k + kk] *= inv;
                }
            }
        }
        let groups = cfg.num_conv_pos_embedding_groups;
        anyhow::ensure!(
            out_c == cfg.hidden_size
                && groups > 0
                && in_pg == cfg.hidden_size / groups
                && groups * in_pg == cfg.hidden_size
                && k == cfg.num_conv_pos_embeddings
                && in_pg == shaders::POS_NT as usize
                && (in_pg * k).is_multiple_of(shaders::POS_KS as usize),
            "pos conv {out_c}x{in_pg}x{k} does not match the tiled kernel"
        );
        let pos_wt = transpose_pos(&pos_w, out_c, in_pg, k);

        // ---- weight prep: transposes and fused qkv, in parallel
        let transpose = |w: &[f32], out: usize, inp: usize| -> Vec<f32> {
            (0..inp * out)
                .into_par_iter()
                .map(|i| {
                    let kk = i / out;
                    let n = i % out;
                    w[n * inp + kk]
                })
                .collect()
        };
        let conv_wt: Vec<Vec<f32>> = (1..7)
            .map(|i| {
                let (out, cin, kk) = (512usize, 512usize, cfg.conv_kernel[i]);
                let w = get_f32(
                    &tensors,
                    &format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.weight"),
                )
                .unwrap()
                .0;
                let mut wt = vec![0.0f32; kk * cin * out];
                wt.par_chunks_mut(out).enumerate().for_each(|(idx, row)| {
                    let t = idx / cin;
                    let ci = idx % cin;
                    for (o, slot) in row.iter_mut().enumerate() {
                        *slot = w[(o * cin + ci) * kk + t];
                    }
                });
                wt
            })
            .collect();
        let lm_wt = transpose(
            &get_f32(&tensors, "lm_head.weight")?.0,
            cfg.vocab_size,
            cfg.hidden_size,
        );
        let fp_wt = transpose(
            &get_f32(&tensors, "wav2vec2.feature_projection.projection.weight")?.0,
            cfg.hidden_size,
            512,
        );

        let lin = |name: String| -> Result<(Vec<f32>, Vec<f32>)> {
            let w = get_f32(&tensors, &format!("{name}.weight"))?.0;
            let b = get_f32(&tensors, &format!("{name}.bias"))?.0;
            Ok((w, b))
        };

        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .into_par_iter()
            .map(|i| {
                let base = |sub: &str| format!("wav2vec2.encoder.layers.{i}.{sub}");
                let tr = |w: &[f32], out: usize, inp: usize| transpose(w, out, inp);
                let hs = cfg.hidden_size;
                // one read per tensor: the weight is 4 MiB a layer, so asking
                // for it twice cost ~1.1 GB of pure copies over 24 layers
                let (qw, qb) = lin(base("attention.q_proj")).unwrap();
                let (kw, kb) = lin(base("attention.k_proj")).unwrap();
                let (vw, vb) = lin(base("attention.v_proj")).unwrap();
                let (ow, ob) = lin(base("attention.out_proj")).unwrap();
                let (f1w, f1b) = lin(base("feed_forward.intermediate_dense")).unwrap();
                let (f2w, f2b) = lin(base("feed_forward.output_dense")).unwrap();
                let (l1w, l1b) = lin(base("layer_norm")).unwrap();
                let (l2w, l2b) = lin(base("final_layer_norm")).unwrap();
                // fused (K, 3K): row k = [Wq[:,k] | Wk[:,k] | Wv[:,k]]
                let qt = tr(&qw, hs, hs);
                let kt = tr(&kw, hs, hs);
                let vt = tr(&vw, hs, hs);
                let mut qkv = vec![0.0f32; hs * 3 * hs];
                for r in 0..hs {
                    qkv[r * 3 * hs..r * 3 * hs + hs].copy_from_slice(&qt[r * hs..(r + 1) * hs]);
                    qkv[r * 3 * hs + hs..r * 3 * hs + 2 * hs]
                        .copy_from_slice(&kt[r * hs..(r + 1) * hs]);
                    qkv[r * 3 * hs + 2 * hs..r * 3 * hs + 3 * hs]
                        .copy_from_slice(&vt[r * hs..(r + 1) * hs]);
                }
                let mut qkvb = qb;
                qkvb.extend_from_slice(&kb);
                qkvb.extend_from_slice(&vb);
                LayerWeights {
                    qkv_wt: qkv,
                    qkv_b: qkvb,
                    out_wt: tr(&ow, hs, hs),
                    out_b: ob,
                    ff1_wt: tr(&f1w, cfg.intermediate_size, hs),
                    ff1_b: f1b,
                    ff2_wt: tr(&f2w, hs, cfg.intermediate_size),
                    ff2_b: f2b,
                    ln1_w: l1w,
                    ln1_b: l1b,
                    ln2_w: l2w,
                    ln2_b: l2b,
                }
            })
            .collect();

        // ---- upload
        let mut up = gpu.uploader();
        let put = |up: &mut BulkUpload, data: &[f32], label: &str| -> wgpu::Buffer {
            let bytes = le_bytes(data);
            let buf = up.storage(label, bytes.len() as u64);
            up.upload(&buf, &bytes).unwrap();
            buf
        };

        let conv0_w = put(
            &mut up,
            &get_f32(
                &tensors,
                "wav2vec2.feature_extractor.conv_layers.0.conv.weight",
            )?
            .0,
            "conv0.w",
        );
        let conv_wt_b: Vec<_> = conv_wt
            .iter()
            .enumerate()
            .map(|(i, wt)| put(&mut up, wt, &format!("conv{}.wt", i + 1)))
            .collect();
        let conv_b: Vec<_> = (0..7)
            .map(|i| {
                put(
                    &mut up,
                    &get_f32(
                        &tensors,
                        &format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.bias"),
                    )
                    .unwrap()
                    .0,
                    &format!("conv{i}.b"),
                )
            })
            .collect();
        let (mut conv_ln_w, mut conv_ln_b) = (Vec::new(), Vec::new());
        for i in 0..7 {
            conv_ln_w.push(put(
                &mut up,
                &get_f32(
                    &tensors,
                    &format!("wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.weight"),
                )?
                .0,
                &format!("conv{i}.lnw"),
            ));
            conv_ln_b.push(put(
                &mut up,
                &get_f32(
                    &tensors,
                    &format!("wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.bias"),
                )?
                .0,
                &format!("conv{i}.lnb"),
            ));
        }
        let fp_ln_w = put(
            &mut up,
            &get_f32(&tensors, "wav2vec2.feature_projection.layer_norm.weight")?.0,
            "fp.lnw",
        );
        let fp_ln_b = put(
            &mut up,
            &get_f32(&tensors, "wav2vec2.feature_projection.layer_norm.bias")?.0,
            "fp.lnb",
        );
        let fp_wt_b = put(&mut up, &fp_wt, "fp.wt");
        let fp_b = put(
            &mut up,
            &get_f32(&tensors, "wav2vec2.feature_projection.projection.bias")?.0,
            "fp.b",
        );
        let pos_w_b = put(&mut up, &pos_wt, "pos.w");
        let pos_b_b = put(
            &mut up,
            &get_f32(&tensors, "wav2vec2.encoder.pos_conv_embed.conv.bias")?.0,
            "pos.b",
        );

        let mut qkv_wt_b = Vec::new();
        let mut qkv_b_b = Vec::new();
        let mut out_wt_b = Vec::new();
        let mut out_b_b = Vec::new();
        let mut ff1_wt_b = Vec::new();
        let mut ff1_b_b = Vec::new();
        let mut ff2_wt_b = Vec::new();
        let mut ff2_b_b = Vec::new();
        let mut ln1_w_b = Vec::new();
        let mut ln1_b_b = Vec::new();
        let mut ln2_w_b = Vec::new();
        for (i, l) in layers.iter().enumerate() {
            qkv_wt_b.push(put(&mut up, &l.qkv_wt, &format!("l{i}.qkvt")));
            qkv_b_b.push(put(&mut up, &l.qkv_b, &format!("l{i}.qkvb")));
            out_wt_b.push(put(&mut up, &l.out_wt, &format!("l{i}.outwt")));
            out_b_b.push(put(&mut up, &l.out_b, &format!("l{i}.outb")));
            ff1_wt_b.push(put(&mut up, &l.ff1_wt, &format!("l{i}.ff1wt")));
            ff1_b_b.push(put(&mut up, &l.ff1_b, &format!("l{i}.ff1b")));
            ff2_wt_b.push(put(&mut up, &l.ff2_wt, &format!("l{i}.ff2wt")));
            ff2_b_b.push(put(&mut up, &l.ff2_b, &format!("l{i}.ff2b")));
            ln1_w_b.push(put(&mut up, &l.ln1_w, &format!("l{i}.ln1w")));
            ln1_b_b.push(put(&mut up, &l.ln1_b, &format!("l{i}.ln1b")));
            ln2_w_b.push(put(&mut up, &l.ln2_w, &format!("l{i}.ln2w")));
        }
        let ln2_b_buf: Vec<_> = layers
            .iter()
            .enumerate()
            .map(|(i, l)| put(&mut up, &l.ln2_b, &format!("l{i}.ln2b")))
            .collect();
        let final_ln_w = put(
            &mut up,
            &get_f32(&tensors, "wav2vec2.encoder.layer_norm.weight")?.0,
            "final.lnw",
        );
        let final_ln_b = put(
            &mut up,
            &get_f32(&tensors, "wav2vec2.encoder.layer_norm.bias")?.0,
            "final.lnb",
        );
        let lm_wt_b = put(&mut up, &lm_wt, "lm.wt");
        let lm_b_b = put(&mut up, &get_f32(&tensors, "lm_head.bias")?.0, "lm.b");
        let zeros = put(&mut up, &vec![0.0f32; 4096], "zeros");
        up.finish()?;

        let labels_buf = gpu.storage("trellis-labels", 65536 * 4);
        let star_flags = gpu.storage("star-flags", 65536 * 4);
        let mk = |src: &str, entry: &str| -> Result<wgpu::ComputePipeline> {
            gpu.pipeline(entry, src, "main", None)
        };
        // The GEMM fills every shared element it reads. Skipping the WebGPU
        // workgroup zero removes a serial prologue on Pascal.
        let mk_gemm = |src: &str, entry: &str| -> Result<wgpu::ComputePipeline> {
            gpu.pipeline_no_zero(entry, src, "main", None)
        };
        // One GEMM schedule for the whole model, picked once at load; the
        // swept result and why the alternatives lost live on `GemmVariant`.
        // The adapter's workgroup-storage limit is an input to both the
        // choice and the shader generation, so a tile that cannot compile on
        // this backend is refused before any pipeline is created.
        let max_storage = gpu.limits().max_compute_workgroup_storage_size;
        // `f16` shared tiles halve the footprint of every staged tile; the
        // weights and activations stay f32 either way. It needs SHADER_F16,
        // which `use_f16_kernel` gates on the negotiated device.
        let use_f16 = gpu.use_f16_kernel();
        let variant = shaders::pick_gemm_variant_with(max_storage, use_f16);
        let (gemm_mt, gemm_nt) = shaders::gemm_tile(variant);
        let (pos_src, pos_mt) = shaders::pos_conv(max_storage);
        // The row kernels fold a row with the device's subgroup shape when it
        // has one: two barriers per reduction instead of eighteen tree steps.
        let sg = gpu.subgroup_mode();
        let gemm = mk_gemm(
            &shaders::gemm_variant_with(variant, max_storage, use_f16),
            "gemm",
        )?;
        // Attention scores and `pv` run on f32 tiles whether or not the linear
        // GEMMs narrowed: the softmax weights are computed straight from the
        // score tile, and a rounded score moves a weight. Without `f16` this
        // is the same pipeline, so nothing extra is compiled on Pascal.
        let variant_f32 = if use_f16 {
            resolve_f32_gemm_variant(variant, max_storage)
        } else {
            variant
        };
        let (gemm_f32_mt, gemm_f32_nt) = shaders::gemm_tile(variant_f32);
        let gemm_f32 = if use_f16 {
            mk_gemm(
                &shaders::gemm_variant_with(variant_f32, max_storage, false),
                "gemm_f32",
            )?
        } else {
            gemm.clone()
        };
        // The short-window 64x64 tile: a separate pipeline, because the tile
        // geometry is baked into the shader.
        let tile64 = resolve_tile64(
            gpu.has_f16(),
            gemm_mt,
            max_storage,
            use_f16,
            std::env::var("CTC_TILE64").ok().as_deref(),
        );
        let gemm64 = if tile64 {
            Some(mk_gemm(
                &shaders::gemm_variant_with(shaders::GemmVariant::M64s, max_storage, use_f16),
                "gemm64",
            )?)
        } else {
            None
        };
        let gemm64_f32 = if tile64 && use_f16 {
            if shaders::GemmVariant::M64s.shared_bytes_with(false) <= max_storage {
                Some(mk_gemm(
                    &shaders::gemm_variant_with(shaders::GemmVariant::M64s, max_storage, false),
                    "gemm64_f32",
                )?)
            } else {
                None
            }
        } else {
            gemm64.clone()
        };
        // 16 KiB of shared memory is exactly WebGPU's floor, so this only
        // drops out on an adapter below the spec — and then the materialised
        // path is what runs, not a failed model load.
        let flash_attn = if max_storage >= 16 * 1024 {
            Some(mk(&shaders::flash_attn(), "flash_attn")?)
        } else {
            None
        };
        let pipes = Pipe {
            gemm,
            gemm_f32,
            conv_gemm: mk_gemm(
                &shaders::conv_gemm_variant_with(variant, max_storage, use_f16),
                "conv_gemm",
            )?,
            conv0: mk(&shaders::conv0(), "conv0")?,
            pos_conv: mk_gemm(&pos_src, "pos_conv")?,
            ln: mk(&shaders::layernorm(sg), "ln")?,
            ln_sd: mk(&shaders::layernorm_sd(sg), "ln_sd")?,
            add: mk(&shaders::add(), "add")?,
            softmax: mk(&shaders::softmax(sg), "softmax")?,
            log_softmax: mk(&shaders::log_softmax(sg), "log_softmax")?,
            transpose: mk(&shaders::transpose(), "transpose")?,
            gather: mk(&shaders::gather(), "gather")?,
            gemm64,
            gemm64_f32,
            flash_attn,
        };
        // The cache is a warm-start optimisation: an unwritable cache
        // directory must not fail a model load.
        let _ = gpu.save_pipeline_cache();

        Ok(Self {
            gpu,
            cfg,
            pipes,
            gemm_mt,
            gemm_nt,
            gemm_f32_mt,
            gemm_f32_nt,
            pos_mt,
            readback: std::sync::Mutex::new(Vec::new()),
            xfer: std::sync::OnceLock::new(),
            conv0_w,
            conv_wt: conv_wt_b,
            conv_b,
            conv_ln_w,
            conv_ln_b,
            fp_ln_w,
            fp_ln_b,
            fp_wt: fp_wt_b,
            fp_b,
            pos_w: pos_w_b,
            pos_b: pos_b_b,
            qkv_wt: qkv_wt_b,
            qkv_b: qkv_b_b,
            out_wt: out_wt_b,
            out_b: out_b_b,
            ff1_wt: ff1_wt_b,
            ff1_b: ff1_b_b,
            ff2_wt: ff2_wt_b,
            ff2_b: ff2_b_b,
            ln1_w: ln1_w_b,
            ln1_b: ln1_b_b,
            ln2_w: ln2_w_b,
            ln2_b: ln2_b_buf,
            final_ln_w,
            final_ln_b,
            lm_wt: lm_wt_b,
            lm_b: lm_b_b,
            zeros,
            labels: labels_buf,
            star_flags,
            scratch: std::sync::Mutex::new(None),
        })
    }

    pub fn describe(&self) -> String {
        self.gpu.describe()
    }

    /// Drop the activation scratch (recorded dispatch graph + buffers); the
    /// next encode rebuilds it.
    ///
    /// `align()` calls this once per file. The recorded-graph replay is only
    /// sound for the file whose dispatches recorded it: a scratch reused
    /// across files has regions no dispatch of the new file writes (the
    /// zero-padded context holes in front of window 1, buffers sized by the
    /// previous file), and those regions still hold the previous file's
    /// samples. The first window then encodes against the wrong context and
    /// the DP mis-places the transcript from its first word on — measured on
    /// a 60 s slice re-aligned after a 51 s one: the first token landed 4.4 s
    /// late with the scratch shared, on the frame-accurate mark with a fresh
    /// one. Within one file the windows keep sharing the scratch; that is
    /// the reuse the cache exists for.
    pub fn reset_scratch(&self) {
        *self.scratch.lock().unwrap() = None;
    }

    /// Forward one z-normalised chunk; returns log_probs (T, V) on the host.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let p = self.forward_begin(input, None, false, None, &[], None, true)?;
        self.collect(p)
    }

    /// Same forward, but instead of the whole (T, V) log-prob matrix only
    /// the expanded trellis labels' values come back: (T, S), S =
    /// expanded.len().  The alignment Viterbi reads nothing else, and the
    /// 70 MB download of a 34 s chunk shrinks to ~30 KB.
    ///
    /// `keep` restricts the readback to a row range of the gathered block —
    /// the windowed aligner keeps only the middle `win` rows of each chunk
    /// and drops the context rows, so they are never downloaded.
    pub fn forward_gathered(
        &self,
        input: &[f32],
        expanded: &[u32],
        keep: Option<(usize, usize)>,
        stars: &[u32],
    ) -> Result<Vec<f32>> {
        let p = self.forward_begin(input, Some(expanded), false, keep, stars, None, true)?;
        self.collect(p)
    }

    /// Pipelined [`forward_gathered`]: submit the compute and return
    /// immediately; the caller collects the previous chunk's result while
    /// this one computes. `prev` is that previous pending — its staging
    /// copies are enqueued here, ahead of this chunk's compute, because this
    /// chunk's dispatches are what overwrite the result buffers.
    pub fn forward_gathered_begin(
        &self,
        input: &[f32],
        expanded: &[u32],
        keep: Option<(usize, usize)>,
        stars: &[u32],
        prev: Option<&mut GpuPending>,
    ) -> Result<GpuPending> {
        self.forward_begin(input, Some(expanded), false, keep, stars, prev, true)
    }

    /// Pipelined [`forward_hidden`] — see [`GpuModel::forward_gathered_begin`].
    pub fn forward_hidden_begin(
        &self,
        input: &[f32],
        prev: Option<&mut GpuPending>,
    ) -> Result<GpuPending> {
        self.forward_begin(input, None, true, None, &[], prev, true)
    }

    /// Pipelined full-logits [`forward`] — see
    /// [`GpuModel::forward_gathered_begin`].
    pub fn forward_logits_begin(
        &self,
        input: &[f32],
        prev: Option<&mut GpuPending>,
    ) -> Result<GpuPending> {
        self.forward_begin(input, None, false, None, &[], prev, true)
    }

    /// Compile the f64 Viterbi pipelines and allocate the per-file buffers.
    /// Fails if the shader does not validate; the caller keeps the CPU DP only
    /// when the device never advertised [`wgpu::Features::SHADER_F64`].
    pub(crate) fn new_dp(
        &self,
        token_ids: &[usize],
        star_id: usize,
        blank_id: usize,
        n_frames: usize,
    ) -> Result<crate::viterbi_gpu::GpuDp> {
        crate::viterbi_gpu::GpuDp::new(
            &self.gpu,
            token_ids,
            star_id,
            blank_id,
            n_frames,
            self.cfg.hidden_size,
        )
    }

    /// Encode one window and queue its DP behind that encode. The DP submit
    /// returns without waiting, so the next window's encoder can be recorded
    /// while this one is still on the device. Returns the chunk's frame count,
    /// context rows included.
    ///
    /// A window longer than the DP's per-submit cap (a 60 s window is 3000
    /// frames against a 2048-frame cap) splits into consecutive submits. The
    /// DP is a chain of per-frame dispatches, so the split continues it
    /// exactly: frames accumulate, each submit reads its own row range of the
    /// same encoded window, and each carries its own slice of the hidden and
    /// choice readbacks.
    pub(crate) fn gpu_dp_window(
        &self,
        input: &[f32],
        dp: &mut crate::viterbi_gpu::GpuDp,
        row0: usize,
        n_kept: usize,
    ) -> Result<usize> {
        if n_kept > 0 {
            dp.prepare_slot(&self.gpu)?;
        }
        let (logits, out_x, t) = {
            let _pending = self.forward_begin(input, None, false, None, &[], None, false)?;
            let guard = self.scratch.lock().unwrap();
            let s = guard
                .as_ref()
                .context("scratch dropped before gpu viterbi")?;
            (s.logits.clone(), s.out_x.clone(), s.t)
        };
        if n_kept > 0 {
            anyhow::ensure!(
                row0 + n_kept <= t,
                "window kept rows {row0}+{n_kept} outside chunk of {t}"
            );
            let mut off = 0usize;
            loop {
                let m = (n_kept - off).min(dp.chunk_limit());
                if off > 0 {
                    // The split consumes readback slots faster than one per
                    // window; wait out the slot this submit reuses.
                    dp.prepare_slot(&self.gpu)?;
                }
                dp.submit_window(
                    &self.gpu,
                    &logits,
                    &out_x,
                    self.cfg.vocab_size,
                    row0 + off,
                    m,
                )?;
                off += m;
                if off == n_kept {
                    break;
                }
            }
        }
        Ok(t)
    }

    /// Map the windows still in flight. Called once, after the last encoder.
    pub(crate) fn gpu_dp_finish(&self, dp: &mut crate::viterbi_gpu::GpuDp) -> Result<()> {
        dp.finish(&self.gpu)
    }

    pub(crate) fn gpu_dp_alpha(&self, dp: &crate::viterbi_gpu::GpuDp) -> Result<Vec<f64>> {
        dp.read_alpha(&self.gpu)
    }

    /// Rebuild choices from the alpha checkpoints, walking the file backwards.
    ///
    /// The next segment is submitted before the previous one is mapped, so the
    /// device does not go idle at each boundary. The state readback after the
    /// loop is when this phase has finished.
    pub(crate) fn gpu_dp_traceback(
        &self,
        dp: &mut crate::viterbi_gpu::GpuDp,
        alpha_last: &[f64],
    ) -> Result<(Vec<i32>, f64)> {
        let t_len = dp.frames();
        anyhow::ensure!(alpha_last.len() == dp.state_count(), "alpha width");
        anyhow::ensure!(t_len >= 1, "empty traceback");
        let mut s_end = alpha_last.len() - 1;
        if alpha_last.len() >= 2 && alpha_last[s_end - 1] > alpha_last[s_end] {
            s_end -= 1;
        }
        let total = alpha_last[s_end];
        let (logits, rows_cap) = {
            let guard = self.scratch.lock().unwrap();
            let s = guard
                .as_ref()
                .context("scratch dropped before viterbi traceback")?;
            (s.logits.clone(), s.t)
        };
        anyhow::ensure!(rows_cap >= 1, "empty traceback chunk");
        let seg = dp.seg();
        let chunk = rows_cap.min(dp.chunk_limit());
        // Seed the walk's end state: every segment's trace starts from
        // states[hi], and the first segment processed covers hi = t_len-1.
        let seed = s_end as u32;
        self.gpu.queue.write_buffer(
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
                let start_alpha = dp.checkpoint(k)?.to_vec();
                dp.load_alpha(&self.gpu, &start_alpha)?;
                let mut frame = lo;
                while frame <= hi {
                    let m = (hi - frame + 1).min(chunk);
                    self.submit_replay_chunk(dp, &logits, frame, m, frame - lo)?;
                    frame += m;
                }
                // Walk this segment's choices on the device. Ordered after
                // the replay chunks, and the next segment's trace reads the
                // state this one leaves in states[lo-1] — no host round trip.
                dp.trace_segment(&self.gpu, lo, hi)?;
            }
            if b == 0 {
                break;
            }
            b -= seg;
        }
        // The state path is complete on the device; one readback replaces the
        // per-chunk choice maps this used to wait on.
        let bytes = self.gpu.readback(dp.states_buf(), (t_len * 4) as u64)?;
        anyhow::ensure!(bytes.len() == t_len * 4, "state readback short");
        let states: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((states, total))
    }

    /// Replay one chunk's DP from the segment checkpoint: upload the chunk's
    /// archived hidden rows, re-run the lm head and log-softmax on device,
    /// re-run the DP steps, leaving the chunk's packed choices in the device
    /// back buffer at rows `back_row0..back_row0+m`. No readback.
    fn submit_replay_chunk(
        &self,
        dp: &mut crate::viterbi_gpu::GpuDp,
        logits: &wgpu::Buffer,
        frame: usize,
        m: usize,
        back_row0: usize,
    ) -> Result<()> {
        let hidden = self.cfg.hidden_size;
        let vocab = u32::try_from(self.cfg.vocab_size).context("vocab")?;
        let hidden_u = u32::try_from(hidden).context("hidden")?;
        let prof = crate::viterbi_gpu::dp_prof::enabled();
        let t_rec = prof.then(std::time::Instant::now);
        let guard = self
            .gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        self.gpu
            .upload(dp.stage(), bytemuck::cast_slice(dp.hidden_rows(frame, m)?));
        let dims = GemmDims {
            m: m as u32,
            n: vocab,
            k: hidden_u,
            a_stride: hidden_u,
            b_stride: vocab,
            c_stride: vocab,
            a_off: 0,
            b_off: 0,
            c_off: 0,
            a_z: 0,
            b_z: 0,
            c_z: 0,
            epilogue: 0,
            scale: 1.0,
            a_vec: vec4_ok(0, hidden_u, 0),
            b_vec: vec4_ok(0, vocab, 0),
            _p0: 0,
        };
        let ls = Cfg4 {
            a: m as u32,
            b: vocab,
            c: 0,
            d: 0,
        };
        let gemm_bytes = bytemuck::bytes_of(&dims);
        let ls_bytes = bytemuck::bytes_of(&ls);
        let mut raw = vec![0u8; 512];
        raw[..gemm_bytes.len()].copy_from_slice(gemm_bytes);
        raw[256..256 + ls_bytes.len()].copy_from_slice(ls_bytes);
        let uni = self.gpu.uniform("dp-replay-uni", 512);
        self.gpu.queue.write_buffer(&uni, 0, &raw);
        let gemm_layout = self.pipes.gemm.get_bind_group_layout(0);
        let ls_layout = self.pipes.log_softmax.get_bind_group_layout(0);
        let stage = dp.stage().clone();
        let gemm_bg = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dp-replay-gemm"),
                layout: &gemm_layout,
                entries: &[
                    score_buf(0, &stage),
                    score_buf(1, &self.lm_wt),
                    score_buf(2, &self.lm_b),
                    score_buf(3, logits),
                    score_uni(4, &uni, 0, gemm_bytes.len() as u64),
                    score_buf(5, &self.zeros),
                    score_buf(6, &stage),
                    score_buf(7, &self.lm_wt),
                ],
            });
        let ls_bg = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dp-replay-logsoftmax"),
                layout: &ls_layout,
                entries: &[
                    score_buf(0, logits),
                    score_uni(1, &uni, 256, ls_bytes.len() as u64),
                ],
            });
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dp-replay"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dp-replay-gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipes.gemm);
            pass.set_bind_group(0, &gemm_bg, &[]);
            pass.dispatch_workgroups(
                (m as u32).div_ceil(self.gemm_mt),
                vocab.div_ceil(self.gemm_nt),
                1,
            );
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dp-replay-logsoftmax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipes.log_softmax);
            pass.set_bind_group(0, &ls_bg, &[]);
            let (gx, gy) = row_grid(m as u32);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let steps = dp.record_steps(
            &self.gpu,
            &mut enc,
            logits,
            self.cfg.vocab_size,
            m,
            back_row0,
        )?;
        self.gpu.queue.submit([enc.finish()]);
        drop((gemm_bg, ls_bg, uni, steps));
        if let Some(t) = t_rec {
            crate::viterbi_gpu::dp_prof::add(&crate::viterbi_gpu::dp_prof::RECORD_US, t.elapsed());
        }
        let t_sub = prof.then(std::time::Instant::now);
        self.gpu
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| anyhow::anyhow!("poll for viterbi traceback: {e}"))?;
        if let Some(e) = pollster::block_on(guard.pop()) {
            let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
            anyhow::bail!("gpu viterbi traceback: {e}");
        }
        if let Some(t) = t_sub {
            crate::viterbi_gpu::dp_prof::add(&crate::viterbi_gpu::dp_prof::SUBMIT_US, t.elapsed());
        }
        Ok(())
    }

    /// Replay the lm head over the archived encoder rows and gather the f32
    /// the DP added on the traced state. One chunk of the scratch logits at a time.
    pub(crate) fn gpu_dp_scores(
        &self,
        dp: &crate::viterbi_gpu::GpuDp,
        states: &[i32],
    ) -> Result<Vec<f64>> {
        let t_len = states.len();
        if t_len == 0 {
            return Ok(Vec::new());
        }
        anyhow::ensure!(t_len == dp.frames(), "score frame count");
        let (logits, rows_cap) = {
            let guard = self.scratch.lock().unwrap();
            let s = guard
                .as_ref()
                .context("scratch dropped before score replay")?;
            (s.logits.clone(), s.t)
        };
        anyhow::ensure!(rows_cap >= 1, "empty score chunk");
        let mut packed = Vec::with_capacity(t_len);
        for &st in states {
            anyhow::ensure!(st >= 0, "negative trellis state");
            packed.push(st as u32);
        }
        self.gpu
            .upload(dp.states_buf(), bytemuck::cast_slice(&packed));

        let hidden = self.cfg.hidden_size;
        let vocab = u32::try_from(self.cfg.vocab_size).context("vocab")?;
        let hidden_u = u32::try_from(hidden).context("hidden")?;
        let gemm_layout = self.pipes.gemm.get_bind_group_layout(0);
        let ls_layout = self.pipes.log_softmax.get_bind_group_layout(0);
        let uni = self.gpu.uniform("dp-score-uni", 768);
        let guard = self
            .gpu
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);

        let mut row = 0usize;
        while row < t_len {
            let m = (t_len - row).min(rows_cap).min(dp.chunk_limit());
            self.gpu
                .upload(dp.stage(), bytemuck::cast_slice(dp.hidden_rows(row, m)?));
            let dims = GemmDims {
                m: m as u32,
                n: vocab,
                k: hidden_u,
                a_stride: hidden_u,
                b_stride: vocab,
                c_stride: vocab,
                a_off: 0,
                b_off: 0,
                c_off: 0,
                a_z: 0,
                b_z: 0,
                c_z: 0,
                epilogue: 0,
                scale: 1.0,
                a_vec: vec4_ok(0, hidden_u, 0),
                b_vec: vec4_ok(0, vocab, 0),
                _p0: 0,
            };
            let ls = Cfg4 {
                a: m as u32,
                b: vocab,
                c: 0,
                d: 0,
            };
            let gather = dp.gather_cfg(row as u32, m as u32, vocab);
            let gemm_bytes = bytemuck::bytes_of(&dims);
            let ls_bytes = bytemuck::bytes_of(&ls);
            let mut raw = vec![0u8; 768];
            raw[..gemm_bytes.len()].copy_from_slice(gemm_bytes);
            raw[256..256 + ls_bytes.len()].copy_from_slice(ls_bytes);
            raw[512..512 + gather.len()].copy_from_slice(&gather);
            self.gpu.queue.write_buffer(&uni, 0, &raw);

            let gemm_bg = self
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("dp-score-gemm"),
                    layout: &gemm_layout,
                    entries: &[
                        score_buf(0, dp.stage()),
                        score_buf(1, &self.lm_wt),
                        score_buf(2, &self.lm_b),
                        score_buf(3, &logits),
                        score_uni(4, &uni, 0, gemm_bytes.len() as u64),
                        score_buf(5, &self.zeros),
                        score_buf(6, dp.stage()),
                        score_buf(7, &self.lm_wt),
                    ],
                });
            let ls_bg = self
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("dp-score-logsoftmax"),
                    layout: &ls_layout,
                    entries: &[
                        score_buf(0, &logits),
                        score_uni(1, &uni, 256, ls_bytes.len() as u64),
                    ],
                });
            let gather_bg = dp.gather_bind_group(&self.gpu, &logits, &uni, 512);
            let mut enc = self
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dp-score"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dp-score-gemm"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipes.gemm);
                pass.set_bind_group(0, &gemm_bg, &[]);
                pass.dispatch_workgroups(
                    (m as u32).div_ceil(self.gemm_mt),
                    vocab.div_ceil(self.gemm_nt),
                    1,
                );
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dp-score-logsoftmax"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipes.log_softmax);
                pass.set_bind_group(0, &ls_bg, &[]);
                let (gx, gy) = row_grid(m as u32);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dp-score-gather"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(dp.gather_pipeline());
                pass.set_bind_group(0, &gather_bg, &[]);
                pass.dispatch_workgroups((m as u32).div_ceil(256).max(1), 1, 1);
            }
            self.gpu.queue.submit([enc.finish()]);
            drop((gemm_bg, ls_bg, gather_bg));
            row += m;
        }
        self.gpu
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| anyhow::anyhow!("poll for score replay: {e}"))?;
        if let Some(e) = pollster::block_on(guard.pop()) {
            let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
            anyhow::bail!("gpu score replay: {e}");
        }

        let bytes = self.gpu.readback(dp.score_buf(), (t_len * 4) as u64)?;
        anyhow::ensure!(bytes.len() == t_len * 4, "score readback short");
        let mut out = Vec::with_capacity(t_len);
        for c in bytes.as_chunks::<4>().0 {
            let mut b = [0u8; 4];
            b.copy_from_slice(c);
            out.push(f64::from(f32::from_le_bytes(b)));
        }
        Ok(out)
    }

    /// Same forward, but the lm head and log-softmax never run: the host gets
    /// the post-final-LN encoder stream (T, hidden) — 7 MB a chunk against
    /// 70 MB of logits.  The head is re-run on the host per window
    /// ([`crate::wav2vec2::LmHeadCpu`]) when the DP needs trellis columns,
    /// which is what keeps a long transcript's emissions out of RAM.
    pub fn forward_hidden(&self, input: &[f32]) -> Result<Vec<f32>> {
        let p = self.forward_begin(input, None, true, None, &[], None, true)?;
        self.collect(p)
    }

    /// Columns per row of [`GpuModel::forward`]'s log-prob block.  The aligner
    /// compares it against the trellis width to decide which of the two
    /// equivalent trellis representations is the smaller thing to hold.
    pub fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    /// Width of one [`GpuModel::forward_hidden`] row.
    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_begin(
        &self,
        input: &[f32],
        gather: Option<&[u32]>,
        read_hidden: bool,
        keep: Option<(usize, usize)>,
        stars: &[u32],
        prev: Option<&mut GpuPending>,
        download: bool,
    ) -> Result<GpuPending> {
        let gpu = &self.gpu;
        // the GEMM pipelines were compiled with this tile; every grid below
        // must tile with the same numbers or part of the output is never
        // dispatched
        let gemm_mt = self.gemm_mt;
        let gemm_nt = self.gemm_nt;
        let hidden = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let heads = self.cfg.num_attention_heads;
        let head_dim = hidden / heads;

        let mut t = input.len();
        for i in 0..7 {
            t = (t - self.cfg.conv_kernel[i]) / self.cfg.conv_stride[i] + 1;
        }
        let t0 = (input.len() - self.cfg.conv_kernel[0]) / self.cfg.conv_stride[0] + 1;
        // Both operands of the attention GEMMs are read as `vec4`s only when
        // every row starts on a 16-byte boundary, and the scores / `K^T`
        // rows are `t` elements wide. Padding the row pitch to a multiple of
        // 4 buys the 128-bit loads for every window instead of the windows
        // whose frame count happens to be a multiple of 4.
        let t_pad = t.next_multiple_of(4);
        let t_pad_u = t_pad as u32;
        let f32s = |n: usize| (n * 4) as u64;

        let scores_n = heads
            .checked_mul(t)
            .and_then(|n| n.checked_mul(t_pad))
            .context("frame count overflow")?;
        if t > 12000 {
            anyhow::bail!(
                "sequence of {t} frames is too long for one GPU forward; use --window 30"
            );
        }
        // A 30 s window is 144 MB of scores and 62 MB of logits; a 90 s one is
        // 1.3 GB of scores, past what a 2 GiB adapter can bind at all. Both
        // buffers are checked before anything is allocated, so the failure
        // names the window instead of surfacing as a bind-group error.
        let scores_bytes = (scores_n as u64)
            .checked_mul(4)
            .context("attention scores size overflow")?;
        let logits_bytes = ((if read_hidden { t * hidden } else { t * vocab }) as u64)
            .checked_mul(4)
            .context("logits size overflow")?;
        let max_binding = gpu.limits().max_storage_buffer_binding_size;
        // Fused attention is decided before either check: it is what removes
        // the `scores` tensor, and with it the window limit those checks are
        // about. `auto` therefore engages exactly where the materialised path
        // would have refused or underperformed.
        let use_flash = self.pipes.flash_attn.is_some()
            && resolve_flash_attn(
                std::env::var("CTC_FLASH_ATTN").ok().as_deref(),
                head_dim,
                scores_bytes,
                max_binding,
                scores_n > 200_000_000,
                gpu.has_f16(),
            );
        if !use_flash && scores_n > 200_000_000 {
            anyhow::bail!(
                "sequence of {t} frames is too long for one GPU forward; use --window 30"
            );
        }
        check_binding_limits(
            t,
            if use_flash { 0 } else { scores_bytes },
            logits_bytes,
            max_binding,
        )?;
        // Short windows go through the 64x64 tile when the load-time selector
        // built it: it is a pure function of the window length, so a replayed
        // graph and the run that recorded it always agree.
        let use_tile64 = self.pipes.gemm64.is_some() && t <= 512;
        let use_attn_tile64 = self.pipes.gemm64_f32.is_some() && t <= 512;
        // The linear GEMMs may narrow their shared tiles; attention's `scores`
        // and `pv` may not, so they own a second copy of the same schedule (or
        // an f32-fitting fallback when an f16 primary tile overflows f32).
        let gemm_pipe = if use_tile64 { P_GEMM64 } else { P_GEMM };
        let attn_pipe = if use_attn_tile64 {
            P_GEMM64_F32
        } else {
            P_GEMM_F32
        };
        let (main_mt, main_nt) = if use_tile64 {
            (64u32, 64u32)
        } else {
            (gemm_mt, gemm_nt)
        };
        let (attn_mt, attn_nt) = if use_attn_tile64 {
            (64u32, 64u32)
        } else {
            (self.gemm_f32_mt, self.gemm_f32_nt)
        };
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        // `CTC_PROFILE=2` splits the GEMM schedule by dispatch shape; these
        // are the `scores` grids, per tile.
        let scores_gy_main = (t as u32).div_ceil(self.gemm_f32_nt);
        let scores_gy_64 = (t as u32).div_ceil(64);
        let (x_in, convs, cached, mut act, ubuf, gather_buf) = {
            let mut guard = self.scratch.lock().unwrap();
            let gathered = gather.is_some();
            let reuse = guard
                .as_ref()
                .map(|s| {
                    s.n_in == input.len()
                        && s.t == t
                        && s.gathered == gathered
                        && s.hidden == read_hidden
                        && s.flash == use_flash
                        && s.tile64 == use_tile64
                })
                .unwrap_or(false);
            if !reuse {
                let st = |label: &str, n: usize| gpu.storage(label, f32s(n));
                let mut convs = Vec::with_capacity(7);
                let mut rows = t0;
                convs.push(st("c0", rows * 512));
                for i in 1..7 {
                    rows = (rows - self.cfg.conv_kernel[i]) / self.cfg.conv_stride[i] + 1;
                    convs.push(st(&format!("c{i}"), rows * 512));
                }
                let xbuf = st("x", t * hidden);
                *guard = Some(Scratch {
                    n_in: input.len(),
                    t,
                    gathered,
                    hidden: read_hidden,
                    flash: use_flash,
                    tile64: use_tile64,
                    x_in: st("x_in", input.len()),
                    convs,
                    x: xbuf.clone(),
                    out_x: xbuf,
                    t1: st("t1", t * hidden),
                    t2: st("t2", t * hidden.max(self.cfg.intermediate_size)),
                    t3: st("t3", t * hidden),
                    qkv: st("qkv", t * 3 * hidden),
                    // Fused attention reads `qkv` and writes `attn_o`: there is
                    // no K^T and no scores tensor, so those two handles are
                    // 4-byte placeholders instead of 176-625 MiB of VRAM.
                    kt: st("kt", if use_flash { 4 } else { hidden * t_pad }),
                    scores: st("scores", if use_flash { 4 } else { scores_n }),
                    attn_o: st("attn_o", t * hidden),
                    // the hidden readback never runs the head, so its logits
                    // buffer doubles as a (t, hidden) scratch
                    logits: st("logits", if read_hidden { t * hidden } else { t * vocab }),
                    // the gather's (t, S) output lives here too: creating a
                    // fresh 36 MB storage per chunk showed up as host time on
                    // every window, and the dispatch can then join the
                    // recorded graph instead of being re-recorded per call
                    gather_out: gather.map(|e| st("gather-out", t * e.len())),
                    ubuf: gpu.uniform("dims", 4096 * UNIFORM_ALIGN),
                    jobs: Vec::new(),
                    uni: Vec::new(),
                });
            }
            let s = guard.as_ref().unwrap();
            let cached = if !s.jobs.is_empty() {
                Some((s.jobs.clone(), s.uni.clone()))
            } else {
                None
            };
            (
                s.x_in.clone(),
                s.convs.clone(),
                cached,
                ActSet {
                    x: s.x.clone(),
                    t1: s.t1.clone(),
                    t2: s.t2.clone(),
                    t3: s.t3.clone(),
                    qkv: s.qkv.clone(),
                    kt: s.kt.clone(),
                    scores: s.scores.clone(),
                    attn_o: s.attn_o.clone(),
                    logits: s.logits.clone(),
                },
                s.ubuf.clone(),
                s.gather_out.clone(),
            )
        };

        // Record every dispatch, then upload uniforms once and replay.
        // write_buffer while a compute pass is open races the uniform buffer
        // on this wgpu version, and a per-dispatch pass + poll blows the
        // Windows TDR budget in the other direction. One pass per 8 encoder
        // layers stays under ~2 s on Pascal and still retires in order.
        let pipes: [&wgpu::ComputePipeline; 15] = [
            &self.pipes.gemm,
            &self.pipes.conv_gemm,
            &self.pipes.conv0,
            &self.pipes.pos_conv,
            &self.pipes.ln,
            &self.pipes.ln_sd,
            &self.pipes.add,
            &self.pipes.softmax,
            &self.pipes.log_softmax,
            &self.pipes.transpose,
            &self.pipes.gather,
            // The three optional pipelines are indexed by their recorded
            // `Job::pipe`, which only ever names one of them when it exists;
            // the fallback keeps the array total for the profile's sake.
            self.pipes.gemm64.as_ref().unwrap_or(&self.pipes.gemm),
            &self.pipes.gemm_f32,
            self.pipes.gemm64_f32.as_ref().unwrap_or(&self.pipes.gemm),
            self.pipes.flash_attn.as_ref().unwrap_or(&self.pipes.gemm),
        ];
        let layouts: Vec<wgpu::BindGroupLayout> =
            pipes.iter().map(|p| p.get_bind_group_layout(0)).collect();
        let uni_cap = (4096 * UNIFORM_ALIGN) as usize;
        let mut uni: Vec<u8> = Vec::with_capacity(uni_cap.min(1200 * UNIFORM_ALIGN as usize));
        let mut batches: Vec<Vec<Job>> = vec![Vec::with_capacity(400)];

        macro_rules! dispatch {
            ($p:expr, $dims:expr, $gx:expr, $gy:expr, $($bind:expr),* $(,)?) => {{
                let dims_owned = $dims;
                let dims_bytes = bytemuck::bytes_of(&dims_owned);
                let uni_off = uni.len() as u64;
                let padded = dims_bytes.len().next_multiple_of(UNIFORM_ALIGN as usize);
                if uni.len() + padded > uni_cap {
                    anyhow::bail!("uniform scratch exhausted at {t} frames");
                }
                uni.extend_from_slice(dims_bytes);
                uni.resize(uni.len() + padded - dims_bytes.len(), 0);
                let mut entries: Vec<wgpu::BindGroupEntry> = vec![$($bind),*];
                entries.push(wgpu::BindGroupEntry {
                    binding: uni_binding($p),
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &ubuf,
                        offset: uni_off,
                        size: Some(std::num::NonZeroU64::new(dims_bytes.len() as u64).unwrap()),
                    }),
                });
                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[$p],
                    entries: &entries,
                });
                batches.last_mut().unwrap().push(Job {
                    pipe: $p,
                    bg,
                    gx: $gx,
                    gy: $gy,
                    gz: 1,
                });
            }};
        }

        // Same, with a non-trivial workgroup z: one dispatch covers all
        // per-head GEMMs (head index = wid.z, offsets step by the *_z dims).
        macro_rules! dispatch_z {
            ($p:expr, $dims:expr, $gx:expr, $gy:expr, $gz:expr, $($bind:expr),* $(,)?) => {{
                let dims_owned = $dims;
                let dims_bytes = bytemuck::bytes_of(&dims_owned);
                let uni_off = uni.len() as u64;
                let padded = dims_bytes.len().next_multiple_of(UNIFORM_ALIGN as usize);
                if uni.len() + padded > uni_cap {
                    anyhow::bail!("uniform scratch exhausted at {t} frames");
                }
                uni.extend_from_slice(dims_bytes);
                uni.resize(uni.len() + padded - dims_bytes.len(), 0);
                let mut entries: Vec<wgpu::BindGroupEntry> = vec![$($bind),*];
                entries.push(wgpu::BindGroupEntry {
                    binding: uni_binding($p),
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &ubuf,
                        offset: uni_off,
                        size: Some(std::num::NonZeroU64::new(dims_bytes.len() as u64).unwrap()),
                    }),
                });
                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[$p],
                    entries: &entries,
                });
                batches.last_mut().unwrap().push(Job {
                    pipe: $p,
                    bg,
                    gx: $gx,
                    gy: $gy,
                    gz: $gz,
                });
            }};
        }

        macro_rules! bind {
            ($idx:expr, $b:expr) => {
                wgpu::BindGroupEntry {
                    binding: $idx,
                    resource: $b.as_entire_binding(),
                }
            };
        }

        macro_rules! kick {
            () => {{
                batches.push(Vec::with_capacity(400));
            }};
        }
        // Same length => same grid, same buffers, same uniforms. Only the
        // waveform upload changes, so the recorded graph is reused.
        let use_cache = cached.is_some();
        if let Some((jobs, bytes)) = cached {
            batches = jobs;
            uni = bytes;
        }
        if !use_cache {
            // ---- conv0 (raw waveform -> (T0, 512))
            dispatch!(
                P_CONV0,
                Cfg4 {
                    a: t0 as u32,
                    b: 512,
                    c: self.cfg.conv_kernel[0] as u32,
                    d: self.cfg.conv_stride[0] as u32
                },
                (t0 as u32).div_ceil(16),
                (512u32).div_ceil(16),
                bind!(0, &x_in),
                bind!(1, &self.conv0_w),
                bind!(2, &self.conv_b[0]),
                bind!(3, &convs[0]),
            );
            dispatch!(
                P_LN,
                Cfg4 {
                    a: t0 as u32,
                    b: 512,
                    c: 10,
                    d: 1
                }, // eps 1e-5 -> 10 * 1e-6; gelu on
                row_grid(t0 as u32).0,
                row_grid(t0 as u32).1,
                bind!(0, &convs[0]),
                bind!(1, &self.conv_ln_w[0]),
                bind!(2, &self.conv_ln_b[0]),
            );

            // ---- conv layers 1..6 (each reads the previous layer's buffer)
            let mut rows = t0;
            for i in 1..7 {
                let tout = (rows - self.cfg.conv_kernel[i]) / self.cfg.conv_stride[i] + 1;
                dispatch!(
                    P_CONV_GEMM,
                    ConvDims {
                        m: tout as u32,
                        n: 512,
                        k: (self.cfg.conv_kernel[i] * 512) as u32,
                        c_in: 512,
                        stride: self.cfg.conv_stride[i] as u32,
                        t_in: rows as u32,
                        a_vec: 1,
                        b_vec: 1,
                    },
                    (tout as u32).div_ceil(gemm_mt),
                    (512u32).div_ceil(gemm_nt),
                    bind!(0, &convs[i - 1]),
                    bind!(1, &self.conv_wt[i - 1]),
                    bind!(2, &self.conv_b[i]),
                    bind!(3, &convs[i]),
                    bind!(5, &convs[i - 1]),
                    bind!(6, &self.conv_wt[i - 1]),
                );
                dispatch!(
                    P_LN,
                    Cfg4 {
                        a: tout as u32,
                        b: 512,
                        c: 10,
                        d: 1
                    },
                    row_grid(tout as u32).0,
                    row_grid(tout as u32).1,
                    bind!(0, &convs[i]),
                    bind!(1, &self.conv_ln_w[i]),
                    bind!(2, &self.conv_ln_b[i]),
                );
                rows = tout;
            }
            let cur = &convs[6];

            // ---- feature projection: LN(512) on conv6 output, then Linear(512->1024)
            dispatch!(
                P_LN,
                Cfg4 {
                    a: rows as u32,
                    b: 512,
                    c: 10,
                    d: 0
                },
                row_grid(rows as u32).0,
                row_grid(rows as u32).1,
                bind!(0, &cur),
                bind!(1, &self.fp_ln_w),
                bind!(2, &self.fp_ln_b),
            );
            dispatch!(
                gemm_pipe,
                GemmDims {
                    m: rows as u32,
                    n: hidden as u32,
                    k: 512,
                    a_stride: 512,
                    b_stride: hidden as u32,
                    c_stride: hidden as u32,
                    a_off: 0,
                    b_off: 0,
                    c_off: 0,
                    a_z: 0,
                    b_z: 0,
                    c_z: 0,
                    epilogue: 0,
                    scale: 1.0,
                    a_vec: vec4_ok(0, 512, 0),
                    b_vec: vec4_ok(0, hidden as u32, 0),
                    _p0: 0,
                },
                (rows as u32).div_ceil(main_mt),
                (hidden as u32).div_ceil(main_nt),
                bind!(0, &cur),
                bind!(1, &self.fp_wt),
                bind!(2, &self.fp_b),
                bind!(3, &act.x),
                bind!(5, &self.zeros),
                bind!(6, &cur),
                bind!(7, &self.fp_wt),
            );

            // ---- positional conv + add. One group per workgroup-y, gelu fused.
            let pos_groups = self.cfg.num_conv_pos_embedding_groups as u32;
            let pos_n = (hidden as u32) / pos_groups;
            let pos_taps = self.cfg.num_conv_pos_embeddings as u32;
            dispatch!(
                P_POS,
                PosDims {
                    m: t as u32,
                    n: pos_n,
                    k: pos_n * pos_taps,
                    c: hidden as u32,
                    in_pg: pos_n,
                    pad: pos_taps / 2,
                    _p0: 0,
                    _p1: 0,
                },
                (t as u32).div_ceil(self.pos_mt),
                pos_groups,
                bind!(0, &act.x),
                bind!(1, &self.pos_w),
                bind!(2, &self.pos_b),
                bind!(3, &act.t1),
            );
            // x = x + pos: write to scratch (a buffer cannot be read- and
            // write-bound in the same dispatch), then swap
            //
            // n is t*hidden, so a window past 16384 frames overflows one
            // dimension's 65535-workgroup limit: at 20 ms per frame that is a
            // 341 s chunk, which the unchunked path (`--window 0`) can reach.
            // The add shader folds workgroups across x and y; see `flat_grid`.
            let (add_x, add_y) = flat_grid((t * hidden) as u32);
            dispatch!(
                P_ADD,
                Cfg4 {
                    a: (t * hidden) as u32,
                    b: 0,
                    c: 0,
                    d: 0
                },
                add_x,
                add_y,
                bind!(0, &act.x),
                bind!(1, &act.t1),
                bind!(2, &act.t3),
            );
            std::mem::swap(&mut act.x, &mut act.t3);

            // ---- encoder layers. Kick splits the recording into command buffers
            // of 8 layers so a 30 s chunk cannot sit in one submit past TDR.
            for li in 0..self.cfg.num_hidden_layers {
                if li > 0 && li % 8 == 0 {
                    kick!();
                }
                let (t_rows, cols) = (t as u32, hidden as u32);
                // t1 = LN(x): src/dst LN, x stays untouched as the residual
                dispatch!(
                    P_LN_SD,
                    Cfg4 {
                        a: t_rows,
                        b: cols,
                        c: 10,
                        d: 0
                    },
                    row_grid(t_rows).0,
                    row_grid(t_rows).1,
                    bind!(0, &act.x),
                    bind!(1, &act.t1),
                    bind!(2, &self.ln1_w[li]),
                    bind!(3, &self.ln1_b[li]),
                );
                // qkv
                dispatch!(
                    gemm_pipe,
                    GemmDims {
                        m: t_rows,
                        n: 3 * cols,
                        k: cols,
                        a_stride: cols,
                        b_stride: 3 * cols,
                        c_stride: 3 * cols,
                        a_off: 0,
                        b_off: 0,
                        c_off: 0,
                        a_z: 0,
                        b_z: 0,
                        c_z: 0,
                        epilogue: 0,
                        scale: 1.0,
                        a_vec: vec4_ok(0, cols, 0),
                        b_vec: vec4_ok(0, 3 * cols, 0),
                        _p0: 0,
                    },
                    t_rows.div_ceil(main_mt),
                    (3 * cols).div_ceil(main_nt),
                    bind!(0, &act.t1),
                    bind!(1, &self.qkv_wt[li]),
                    bind!(2, &self.qkv_b[li]),
                    bind!(3, &act.qkv),
                    bind!(5, &self.zeros),
                    bind!(6, &act.t1),
                    bind!(7, &self.qkv_wt[li]),
                );
                if use_flash {
                    // Fused attention: one dispatch reads Q, K and V straight
                    // out of `qkv` and writes `attn_o`. No K^T transpose, no
                    // (heads, t, t_pad) scores tensor, no separate softmax.
                    dispatch!(
                        P_FLASH_ATTN,
                        FlashAttnCfg {
                            t: t_rows,
                            heads: heads as u32,
                            hidden4: (hidden / 4) as u32,
                            scale: attn_scale,
                            ..Default::default()
                        },
                        t_rows.div_ceil(shaders::FLASH_BQ),
                        heads as u32,
                        bind!(0, &act.qkv),
                        bind!(1, &act.attn_o),
                    );
                } else {
                    // K^T: rows hidden, cols t, from qkv rows offset T*1024, stride 3072
                    dispatch!(
                        P_TRANSPOSE,
                        // K block starts `hidden` elements into each (T, 3*hidden) row;
                        // the output rows are `t_pad` wide, so the scores GEMM below
                        // can read them with vec4 loads
                        TransposeCfg {
                            rows: t as u32,
                            cols: hidden as u32,
                            a_stride: (3 * hidden) as u32,
                            a_off: hidden as u32,
                            o_stride: t_pad_u,
                            ..Default::default()
                        },
                        (t as u32).div_ceil(16),
                        (hidden as u32).div_ceil(16),
                        bind!(0, &act.qkv),
                        bind!(1, &act.kt),
                    );
                    // per-head scores, z-batched: scores_h = q_h @ kt_h over wid.z.
                    // Both operands are vec4-able now: kt rows start every t_pad
                    // elements, so b_vec is 1 for every window
                    dispatch_z!(
                        attn_pipe,
                        GemmDims {
                            m: t_rows,
                            n: t_rows,
                            k: head_dim as u32,
                            a_stride: 3 * cols,
                            b_stride: t_pad_u,
                            c_stride: t_pad_u,
                            a_off: 0,
                            a_z: head_dim as u32,
                            b_off: 0,
                            b_z: head_dim as u32 * t_pad_u,
                            c_off: 0,
                            c_z: t_rows * t_pad_u,
                            epilogue: 0,
                            scale: attn_scale,
                            a_vec: vec4_ok(0, 3 * cols, head_dim as u32),
                            b_vec: vec4_ok(0, t_pad_u, head_dim as u32 * t_pad_u),
                            _p0: 0,
                        },
                        t_rows.div_ceil(attn_mt),
                        t_rows.div_ceil(attn_nt),
                        heads as u32,
                        bind!(0, &act.qkv),
                        bind!(1, &act.kt),
                        bind!(2, &self.zeros),
                        bind!(3, &act.scores),
                        bind!(5, &self.zeros),
                        bind!(6, &act.qkv),
                        bind!(7, &act.kt),
                    );
                    // softmax over each (head, query) row; the row pitch is t_pad
                    dispatch!(
                        P_SOFTMAX,
                        Cfg4 {
                            a: (heads as u32 * t_rows),
                            b: t_rows,
                            c: t_pad_u,
                            d: 0
                        },
                        row_grid(heads as u32 * t_rows).0,
                        row_grid(heads as u32 * t_rows).1,
                        bind!(0, &act.scores),
                    );
                    // per-head weighted V, z-batched: attn_h = scores_h @ v_h.
                    // B is a vec4-able slice of qkv and A (scores) now has an
                    // aligned row pitch, so a_vec is 1 for every window too
                    dispatch_z!(
                        attn_pipe,
                        GemmDims {
                            m: t_rows,
                            n: head_dim as u32,
                            k: t_rows,
                            a_stride: t_pad_u,
                            b_stride: 3 * cols,
                            c_stride: cols,
                            a_off: 0,
                            a_z: t_rows * t_pad_u,
                            b_off: (2 * hidden) as u32,
                            b_z: head_dim as u32,
                            c_off: 0,
                            c_z: head_dim as u32,
                            epilogue: 0,
                            scale: 1.0,
                            a_vec: vec4_ok(0, t_pad_u, t_rows * t_pad_u),
                            b_vec: vec4_ok((2 * hidden) as u32, 3 * cols, head_dim as u32),
                            _p0: 0,
                        },
                        t_rows.div_ceil(attn_mt),
                        (head_dim as u32).div_ceil(attn_nt),
                        heads as u32,
                        bind!(0, &act.scores),
                        bind!(1, &act.qkv),
                        bind!(2, &self.zeros),
                        bind!(3, &act.attn_o),
                        bind!(5, &self.zeros),
                        bind!(6, &act.scores),
                        bind!(7, &act.qkv),
                    );
                }
                // out proj + residual
                dispatch!(
                    gemm_pipe,
                    GemmDims {
                        m: t_rows,
                        n: cols,
                        k: cols,
                        a_stride: cols,
                        b_stride: cols,
                        c_stride: cols,
                        a_off: 0,
                        b_off: 0,
                        c_off: 0,
                        a_z: 0,
                        b_z: 0,
                        c_z: 0,
                        epilogue: 2,
                        scale: 1.0,
                        a_vec: vec4_ok(0, cols, 0),
                        b_vec: vec4_ok(0, cols, 0),
                        _p0: 0,
                    },
                    t_rows.div_ceil(main_mt),
                    cols.div_ceil(main_nt),
                    bind!(0, &act.attn_o),
                    bind!(1, &self.out_wt[li]),
                    bind!(2, &self.out_b[li]),
                    bind!(3, &act.t3),
                    bind!(5, &act.x),
                    bind!(6, &act.attn_o),
                    bind!(7, &self.out_wt[li]),
                );
                // FFN: t3 is the new residual; swap x <-> t3
                std::mem::swap(&mut act.x, &mut act.t3);
                // t1 = LN(x) — again src/dst, no staging copy
                dispatch!(
                    P_LN_SD,
                    Cfg4 {
                        a: t_rows,
                        b: cols,
                        c: 10,
                        d: 0
                    },
                    row_grid(t_rows).0,
                    row_grid(t_rows).1,
                    bind!(0, &act.x),
                    bind!(1, &act.t1),
                    bind!(2, &self.ln2_w[li]),
                    bind!(3, &self.ln2_b[li]),
                );
                dispatch!(
                    gemm_pipe,
                    GemmDims {
                        m: t_rows,
                        n: self.cfg.intermediate_size as u32,
                        k: cols,
                        a_stride: cols,
                        b_stride: self.cfg.intermediate_size as u32,
                        c_stride: self.cfg.intermediate_size as u32,
                        a_off: 0,
                        b_off: 0,
                        c_off: 0,
                        a_z: 0,
                        b_z: 0,
                        c_z: 0,
                        epilogue: 1,
                        scale: 1.0,
                        a_vec: vec4_ok(0, cols, 0),
                        b_vec: vec4_ok(0, self.cfg.intermediate_size as u32, 0),
                        _p0: 0,
                    },
                    t_rows.div_ceil(main_mt),
                    (self.cfg.intermediate_size as u32).div_ceil(main_nt),
                    bind!(0, &act.t1),
                    bind!(1, &self.ff1_wt[li]),
                    bind!(2, &self.ff1_b[li]),
                    bind!(3, &act.t2),
                    bind!(5, &self.zeros),
                    bind!(6, &act.t1),
                    bind!(7, &self.ff1_wt[li]),
                );
                dispatch!(
                    gemm_pipe,
                    GemmDims {
                        m: t_rows,
                        n: cols,
                        k: self.cfg.intermediate_size as u32,
                        a_stride: self.cfg.intermediate_size as u32,
                        b_stride: cols,
                        c_stride: cols,
                        a_off: 0,
                        b_off: 0,
                        c_off: 0,
                        a_z: 0,
                        b_z: 0,
                        c_z: 0,
                        epilogue: 2,
                        scale: 1.0,
                        a_vec: vec4_ok(0, self.cfg.intermediate_size as u32, 0),
                        b_vec: vec4_ok(0, cols, 0),
                        _p0: 0,
                    },
                    t_rows.div_ceil(main_mt),
                    cols.div_ceil(main_nt),
                    bind!(0, &act.t2),
                    bind!(1, &self.ff2_wt[li]),
                    bind!(2, &self.ff2_b[li]),
                    bind!(3, &act.t1),
                    bind!(5, &act.x),
                    bind!(6, &act.t2),
                    bind!(7, &self.ff2_wt[li]),
                );
                std::mem::swap(&mut act.x, &mut act.t1);
            }

            // ---- final LN, LM head, log softmax.  The hidden readback stops at
            // the LN: act.x then holds the stream the host-side head re-run
            // consumes, and the two head dispatches stay out of the graph.
            dispatch!(
                P_LN,
                Cfg4 {
                    a: t as u32,
                    b: hidden as u32,
                    c: 10,
                    d: 0
                },
                row_grid(t as u32).0,
                row_grid(t as u32).1,
                bind!(0, &act.x),
                bind!(1, &self.final_ln_w),
                bind!(2, &self.final_ln_b),
            );
            if !read_hidden {
                dispatch!(
                    gemm_pipe,
                    GemmDims {
                        m: t as u32,
                        n: vocab as u32,
                        k: hidden as u32,
                        a_stride: hidden as u32,
                        b_stride: vocab as u32,
                        c_stride: vocab as u32,
                        a_off: 0,
                        b_off: 0,
                        c_off: 0,
                        a_z: 0,
                        b_z: 0,
                        c_z: 0,
                        epilogue: 0,
                        scale: 1.0,
                        a_vec: vec4_ok(0, hidden as u32, 0),
                        b_vec: vec4_ok(0, vocab as u32, 0),
                        _p0: 0,
                    },
                    (t as u32).div_ceil(main_mt),
                    (vocab as u32).div_ceil(main_nt),
                    bind!(0, &act.x),
                    bind!(1, &self.lm_wt),
                    bind!(2, &self.lm_b),
                    bind!(3, &act.logits),
                    bind!(5, &self.zeros),
                    bind!(6, &act.x),
                    bind!(7, &self.lm_wt),
                );
                dispatch!(
                    P_LOGSOFTMAX,
                    Cfg4 {
                        a: t as u32,
                        b: vocab as u32,
                        c: 0,
                        d: 0
                    },
                    row_grid(t as u32).0,
                    row_grid(t as u32).1,
                    bind!(0, &act.logits),
                );
            }
            // gathered alignment: one dispatch collecting the trellis labels'
            // log-probs, so the readback below is (T, S) instead of (T, V). Part
            // of the recorded graph: only the labels buffer's *contents* change
            // per chunk, never its handle, so a replay reruns the dispatch as
            // recorded against the scratch's gather-out buffer.
            if let Some(expanded) = gather {
                // S grows with the transcript, so T*S overflows a single
                // dimension's 65535-workgroup limit on anything but a short
                // transcript (15 m audio: T=1700, S=31130 -> 206595 groups).
                // The gather kernel folds workgroups across x and y; see
                // `flat_grid`.
                anyhow::ensure!(
                    expanded.len() <= 65536,
                    "transcript too long for the labels buffer"
                );
                let total = u32::try_from(t * expanded.len()).context("gather size")?;
                let (gx, gy) = flat_grid(total);
                dispatch!(
                    P_GATHER,
                    Cfg4 {
                        a: total,
                        b: expanded.len() as u32,
                        c: vocab as u32,
                        d: 0
                    },
                    gx,
                    gy,
                    bind!(0, &act.logits),
                    bind!(1, &self.labels),
                    bind!(2, gather_buf.as_ref().context("missing gather buffer")?),
                    bind!(4, &self.star_flags),
                );
            }
            {
                let mut guard = self.scratch.lock().unwrap();
                if let Some(s) = guard.as_mut() {
                    s.jobs = batches.clone();
                    s.uni = uni.clone();
                    s.out_x = act.x.clone();
                }
            }
        }

        // gathered alignment: read back the trellis columns the caller keeps
        // (the middle `win` rows), not the context rows it drops — one 4-byte
        // aligned range of the gather-out buffer, straight into f32s
        let prof_mode = std::env::var("CTC_PROFILE").ok();
        let prof = prof_mode.as_deref() == Some("1");
        // CTC_PROFILE=2 timestamps every dispatch of the first forward so the
        // per-kernel split is visible. Later chunks stay on the fast path.
        static SPLIT_ONCE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        let do_split = prof_mode.as_deref() == Some("2")
            && gpu.features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && !SPLIT_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed);
        let t_replay = std::time::Instant::now();
        // Input and every uniform slot go out before the first command buffer.
        // The three layer-groups are queued back to back; each buffer stays
        // under the Windows TDR window, and one wait keeps the GPU busy.
        // The previous chunk's result still lives in scratch, and this chunk's
        // dispatches overwrite it. Arm the download first. The cached path
        // copies on this queue, so the copy finishes before the overwrite.
        // The logits path only bounces into a side buffer here; its DMA is on
        // the transfer queue. On this P104 the copy still finishes after the
        // compute submitted below.
        if let Some(p) = prev {
            p.arm(&self.gpu)?;
        }
        gpu.upload(&x_in, bytemuck::cast_slice(input));
        if let Some(expanded) = gather {
            gpu.upload(&self.labels, bytemuck::cast_slice(expanded));
            // the gather kernel stamps the star states' columns itself; the
            // host loop this replaces walked every (frame, star) cell per
            // chunk in column-major scattered writes
            let mut flags = vec![0u32; expanded.len()];
            for &s in stars {
                if (s as usize) < expanded.len() {
                    flags[s as usize] = 1;
                }
            }
            gpu.upload(&self.star_flags, bytemuck::cast_slice(&flags));
        }
        if !uni.is_empty() {
            gpu.queue.write_buffer(&ubuf, 0, &uni);
        }
        let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut split: Option<(wgpu::QuerySet, wgpu::Buffer, u32, Vec<(usize, u32)>)> = None;
        if do_split {
            let n_jobs: u32 = batches.iter().map(|b| b.len() as u32).sum();
            let qcount = n_jobs * 2;
            let qs = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("fwd-ts"),
                ty: wgpu::QueryType::Timestamp,
                count: qcount,
            });
            let qbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fwd-ts"),
                size: (qcount as u64 * 8).next_multiple_of(256),
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            let mut job_meta = Vec::with_capacity(n_jobs as usize);
            let mut qi = 0u32;
            for batch in &batches {
                for job in batch {
                    job_meta.push((job.pipe, job.gy));
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &qs,
                            beginning_of_pass_write_index: Some(qi),
                            end_of_pass_write_index: Some(qi + 1),
                        }),
                    });
                    pass.set_pipeline(pipes[job.pipe]);
                    pass.set_bind_group(0, &job.bg, &[]);
                    pass.dispatch_workgroups(job.gx, job.gy, job.gz);
                    drop(pass);
                    qi += 2;
                }
            }
            enc.resolve_query_set(&qs, 0..qcount, &qbuf, 0);
            gpu.queue.submit([enc.finish()]);
            split = Some((qs, qbuf, qcount, job_meta));
        } else {
            for batch in &batches {
                if batch.is_empty() {
                    continue;
                }
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    for job in batch {
                        pass.set_pipeline(pipes[job.pipe]);
                        pass.set_bind_group(0, &job.bg, &[]);
                        pass.dispatch_workgroups(job.gx, job.gy, job.gz);
                    }
                }
                gpu.queue.submit([enc.finish()]);
            }
        }
        // One non-blocking pump: submit-time validation is processed here, so
        // the error scope's future resolves without waiting for the GPU. (The
        // split path above already waited — its query readback blocks.)
        gpu.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| anyhow::anyhow!("poll: {e}"))?;
        if let Some(e) = pollster::block_on(guard.pop()) {
            anyhow::bail!("gpu validation: {e}");
        }
        if let Some((_qs, qbuf, qcount, job_meta)) = split {
            let bytes = gpu.readback(&qbuf, qcount as u64 * 8)?;
            let ticks: &[u64] = bytemuck::cast_slice(&bytes);
            let period = gpu.queue.get_timestamp_period() as f64;
            // Indexed by pipeline id (P_*), then by the three synthetic
            // buckets. gelu has no pipeline (fused into the LN and the GEMM
            // epilogue), so the names after P_LN_SD were off by one slot until
            // this list stopped carrying a gelu entry.
            let names = [
                "gemm",
                "conv_gemm",
                "conv0",
                "pos",
                "ln",
                "ln_sd",
                "add",
                "softmax",
                "logsoftmax",
                "transpose",
                "gather",
                "gemm64",
                "gemm_f32",
                "gemm64_f32",
                "flash_attn",
                "  scores",
                "  pv",
                "  gemm-main",
            ];
            // The score dispatch is `t/nt` workgroups wide on whichever tile
            // served it: the short-window pipeline has its own 64-wide N.
            let scores_gy = (scores_gy_main, scores_gy_64);
            assert_eq!(names.len(), PROFILE_SLOTS);
            let mut acc = [0f64; PROFILE_SLOTS];
            let mut cnt = [0u32; PROFILE_SLOTS];
            for (i, &(p, gy)) in job_meta.iter().enumerate() {
                let dt = ticks[i * 2 + 1].saturating_sub(ticks[i * 2]) as f64 * period / 1e6;
                if p < acc.len() {
                    // Split the GEMM schedule by dispatch shape: pv is n=64
                    // (gy=1), scores is n=t, everything else is a main GEMM.
                    let b = match p {
                        P_GEMM | P_GEMM_F32 => {
                            if gy == 1 {
                                BUCKET_PV
                            } else if gy == scores_gy.0 {
                                BUCKET_SCORES
                            } else {
                                BUCKET_MAIN
                            }
                        }
                        P_GEMM64 | P_GEMM64_F32 => {
                            if gy == 1 {
                                BUCKET_PV
                            } else if gy == scores_gy.1 {
                                BUCKET_SCORES
                            } else {
                                BUCKET_MAIN
                            }
                        }
                        other => other,
                    };
                    acc[b] += dt;
                    cnt[b] += 1;
                }
            }
            let total: f64 = acc.iter().sum();
            eprintln!("[fwd] T={t} gpu split {total:.1} ms");
            for (i, name) in names.iter().enumerate() {
                if cnt[i] > 0 {
                    eprintln!("  {name:12} {:>7.1} ms  x{}", acc[i], cnt[i]);
                }
            }
        }
        if prof {
            eprintln!(
                "[fwd] T={t} submit {:.1} ms ({} batches, {} uniforms) — collecting in background",
                t_replay.elapsed().as_secs_f64() * 1000.0,
                batches.len(),
                uni.len() / UNIFORM_ALIGN as usize
            );
        }
        // The result buffer, the rows the caller keeps, and the staging that
        // the NEXT begin fills: from here the GPU computes while the host
        // collects the previous chunk.
        let (buffer, offset, floats) = if let Some(expanded) = gather {
            let states = expanded.len();
            // clamp to the rows this forward actually produced (the caller's
            // range is written for a full window; the last one may be short)
            let (lo, hi) = keep
                .map(|(lo, hi)| {
                    let lo = lo.min(t);
                    (lo, hi.min(t).max(lo))
                })
                .unwrap_or((0, t));
            (
                gather_buf
                    .as_ref()
                    .context("missing gather buffer")?
                    .clone(),
                (lo * states * 4) as u64,
                (hi - lo) * states,
            )
        } else if read_hidden {
            // the hidden readback needs the buffer the recorded final LN
            // wrote — `out_x` is saved when the graph is recorded (the local
            // `act.x` it bound stops being valid the moment the call's swap
            // sequence is skipped on a replay), so fetch the latest saved
            // handle here rather than the call-start snapshot
            let out_x = self
                .scratch
                .lock()
                .unwrap()
                .as_ref()
                .context("scratch dropped before hidden readback")?
                .out_x
                .clone();
            (out_x, 0, t * hidden)
        } else {
            (act.logits.clone(), 0, t * vocab)
        };
        if !download {
            // The DP reads logits on device. Allocating the staging buffer
            // here would pin a full logits copy on the host.
            return Ok(GpuPending {
                buffer,
                pieces: Vec::new(),
                staging: StagingSet::Plain(Vec::new()),
                floats: 0,
                armed: None,
            });
        }
        Ok(GpuPending::new(
            &self.gpu,
            &self.readback,
            self.xfer_engine((floats * 4) as u64),
            buffer,
            offset,
            floats,
        ))
    }

    /// The transfer queue, for a logits-sized readback. Hidden stays on
    /// cached `MAP_READ`. `CTC_READBACK=cached` forces that path too.
    fn xfer_engine(&self, bytes: u64) -> Option<&crate::xfer::Engine> {
        if bytes < crate::xfer::MIN_BYTES || self.gpu.xfer_family.is_none() {
            return None;
        }
        if std::env::var("CTC_READBACK").ok().as_deref() == Some("cached") {
            return None;
        }
        self.xfer
            .get_or_init(|| crate::xfer::Engine::open(&self.gpu))
            .as_ref()
    }

    /// Map a pending chunk's result back to the host.
    ///
    /// A pending armed by the next chunk's begin collects while that chunk's
    /// compute is in flight. The cached path waits on map callbacks. The
    /// transfer path waits on its own fence: `poll(Wait)` would also wait out
    /// the compute the DMA is supposed to overlap. The last chunk arms itself
    /// here.
    pub fn collect(&self, mut p: GpuPending) -> Result<Vec<f32>> {
        if p.staging.is_xfer() {
            return self.collect_xfer(p);
        }
        let gpu = &self.gpu;
        if p.armed.is_none() {
            p.arm(gpu)?;
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| anyhow::anyhow!("poll for readback: {e}"))?;
        }
        let Some(Armed::Maps(rxs)) = p.armed.take() else {
            anyhow::bail!("pending maps already consumed");
        };
        // pump until every piece's map has landed; each poll also retires
        // whatever compute is in flight
        let t_wait = std::time::Instant::now();
        let mut got: Vec<Option<()>> = vec![None; rxs.len()];
        loop {
            let mut done = true;
            for (i, rx) in rxs.iter().enumerate() {
                if got[i].is_some() {
                    continue;
                }
                match rx.try_recv() {
                    Ok(Ok(())) => got[i] = Some(()),
                    Ok(Err(e)) => {
                        eprintln!("[map debug] inner error = {e:?}");
                        return Err(anyhow::Error::from(e).context("map buffer"));
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => done = false,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        anyhow::bail!("map callback dropped")
                    }
                }
            }
            if done {
                break;
            }
            gpu.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| anyhow::anyhow!("poll for maps: {e}"))?;
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let wait_ms = t_wait.elapsed().as_secs_f64() * 1000.0;
        let t_host = std::time::Instant::now();
        let staging = std::mem::take(&mut p.staging);
        let StagingSet::Plain(bufs) = &staging else {
            anyhow::bail!("cached readback missing its buffers");
        };
        let mut out: Vec<f32> = Vec::with_capacity(p.floats);
        for (buf, &(_, take)) in bufs.iter().zip(&p.pieces) {
            let slice = buf.slice(..);
            let mapped = slice.get_mapped_range()?;
            out.extend_from_slice(bytemuck::cast_slice(&mapped[..take as usize]));
            drop(mapped);
            buf.unmap();
        }
        let host_ms = t_host.elapsed().as_secs_f64() * 1000.0;
        log_readback(wait_ms, host_ms, out.len(), "cached");
        self.recycle_readback(staging);
        Ok(out)
    }

    fn collect_xfer(&self, mut p: GpuPending) -> Result<Vec<f32>> {
        if !matches!(p.armed, Some(Armed::Xfer)) {
            p.arm(&self.gpu)?;
        }
        let StagingSet::Xfer(set) = &p.staging else {
            anyhow::bail!("transfer readback missing its buffers");
        };
        let wait_ms = set.wait()?;
        let t_host = std::time::Instant::now();
        let mut out = Vec::with_capacity(p.floats);
        set.copy_out(&p.pieces, &mut out);
        let host_ms = t_host.elapsed().as_secs_f64() * 1000.0;
        log_readback(wait_ms, host_ms, out.len(), "xfer");
        // Reclaim the bounce submit. `Poll` does not wait for the encoder
        // that is still overlapping this DMA.
        self.gpu
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| anyhow::anyhow!("poll after transfer: {e}"))?;
        let staging = std::mem::take(&mut p.staging);
        self.recycle_readback(staging);
        Ok(out)
    }

    fn recycle_readback(&self, staging: StagingSet) {
        let keep = match &staging {
            StagingSet::Plain(pieces) => !pieces.is_empty(),
            StagingSet::Xfer(_) => true,
        };
        if !keep {
            return;
        }
        let mut pool = self.readback.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < 2 {
            pool.push(staging);
        }
    }
}

const UNIFORM_ALIGN: u64 = 256;

fn score_buf(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn score_uni(
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

/// Plain `MAP_READ` stays in 16 MiB pieces. One large map fails after a long
/// dispatch sequence on some drivers. Transfer-queue bounces are not this map,
/// so those copies can be one piece per window.
const PLAIN_READBACK_CHUNK: u64 = 16 << 20;
const XFER_READBACK_CHUNK: u64 = 128 << 20;

fn log_readback(wait_ms: f64, host_ms: f64, floats: usize, kind: &str) {
    if matches!(
        std::env::var("CTC_PROFILE").ok().as_deref(),
        Some("1") | Some("2")
    ) {
        let mb = (floats * 4) as f64 / (1024.0 * 1024.0);
        eprintln!("[readback] wait {wait_ms:.1} ms  host {host_ms:.1} ms  {mb:.1} MB {kind}");
    }
}

enum StagingSet {
    Plain(Vec<wgpu::Buffer>),
    Xfer(crate::xfer::XferSet),
}

impl Default for StagingSet {
    fn default() -> Self {
        Self::Plain(Vec::new())
    }
}

impl StagingSet {
    fn is_xfer(&self) -> bool {
        matches!(self, Self::Xfer(_))
    }

    fn fits(&self, pieces: &[(u64, u64)], xfer: bool) -> bool {
        match self {
            Self::Plain(bufs) if !xfer => {
                bufs.len() == pieces.len()
                    && bufs
                        .iter()
                        .zip(pieces)
                        .all(|(buf, &(_, take))| buf.size() >= take)
            }
            Self::Xfer(set) if xfer => set.fits(pieces),
            _ => false,
        }
    }
}

/// A recycled staging set whose pieces are each at least as big as `pieces`.
/// The first two windows allocate; every later window of the same shape
/// reuses one of those two.
fn split_pieces(offset: u64, bytes: u64, chunk: u64) -> Vec<(u64, u64)> {
    let mut pieces = Vec::new();
    let mut off = offset;
    let end = offset + bytes;
    while off < end {
        let take = chunk.min(end - off);
        pieces.push((off, take));
        off += take;
    }
    pieces
}

/// A transfer-queue set for these pieces, from the pool or a fresh alloc.
/// `None` means the caller should use 16 MiB `MAP_READ` pieces instead.
fn take_xfer(
    gpu: &Gpu,
    engine: &crate::xfer::Engine,
    pool: &std::sync::Mutex<Vec<StagingSet>>,
    pieces: &[(u64, u64)],
) -> Option<StagingSet> {
    {
        let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = pool.iter().position(|set| set.fits(pieces, true)) {
            return Some(pool.swap_remove(i));
        }
    }
    let takes: Vec<u64> = pieces.iter().map(|&(_, take)| take).collect();
    engine.alloc_set(&gpu.device, &takes).map(StagingSet::Xfer)
}

fn take_readback(
    gpu: &Gpu,
    engine: Option<&crate::xfer::Engine>,
    pool: &std::sync::Mutex<Vec<StagingSet>>,
    pieces: &[(u64, u64)],
) -> StagingSet {
    let xfer = engine.is_some();
    {
        let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = pool.iter().position(|set| set.fits(pieces, xfer)) {
            return pool.swap_remove(i);
        }
    }
    if let Some(engine) = engine {
        let takes: Vec<u64> = pieces.iter().map(|&(_, take)| take).collect();
        if let Some(set) = engine.alloc_set(&gpu.device, &takes) {
            return StagingSet::Xfer(set);
        }
    }
    StagingSet::Plain(
        pieces
            .iter()
            .map(|&(_, take)| plain_staging(gpu, take))
            .collect(),
    )
}

fn plain_staging(gpu: &Gpu, take: u64) -> wgpu::Buffer {
    let size = (take + 3) & !3;
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

enum Armed {
    Maps(Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>),
    Xfer,
}

/// One submitted chunk whose result is still on the device.
///
/// Staging is allocated at submit time. The next chunk's begin arms the
/// download before its own compute overwrites the result. On the cached path
/// that is a queue copy plus a map. On the logits path it is a device-local
/// bounce; the DMA runs on the transfer queue. On this P104 it does not
/// overlap that compute. The last chunk arms itself in `collect`.
pub(crate) struct GpuPending {
    /// the device buffer holding this chunk's result: the gather-out buffer
    /// (gathered form), the recorded final LN's output (hidden form), or the
    /// logits buffer (full form)
    buffer: wgpu::Buffer,
    /// source byte offset and byte length of each staging piece
    pieces: Vec<(u64, u64)>,
    staging: StagingSet,
    floats: usize,
    armed: Option<Armed>,
}

impl GpuPending {
    fn new(
        gpu: &Gpu,
        pool: &std::sync::Mutex<Vec<StagingSet>>,
        engine: Option<&crate::xfer::Engine>,
        buffer: wgpu::Buffer,
        offset: u64,
        floats: usize,
    ) -> Self {
        let bytes = (floats * 4) as u64;
        if bytes >= crate::xfer::MIN_BYTES {
            if let Some(engine) = engine {
                let pieces = split_pieces(offset, bytes, XFER_READBACK_CHUNK);
                if let Some(staging) = take_xfer(gpu, engine, pool, &pieces) {
                    return Self {
                        buffer,
                        pieces,
                        staging,
                        floats,
                        armed: None,
                    };
                }
            }
        }
        let pieces = split_pieces(offset, bytes, PLAIN_READBACK_CHUNK);
        let staging = take_readback(gpu, None, pool, &pieces);
        Self {
            buffer,
            pieces,
            staging,
            floats,
            armed: None,
        }
    }

    /// Enqueue this pending's download. Called from the next chunk's begin,
    /// before that chunk's compute submits.
    fn arm(&mut self, gpu: &Gpu) -> Result<()> {
        match &mut self.staging {
            StagingSet::Xfer(set) => {
                set.arm(gpu, &self.buffer, &self.pieces)?;
                self.armed = Some(Armed::Xfer);
            }
            StagingSet::Plain(bufs) => {
                {
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    for (buf, &(src_off, take)) in bufs.iter().zip(&self.pieces) {
                        enc.copy_buffer_to_buffer(&self.buffer, src_off, buf, 0, take);
                    }
                    gpu.queue.submit([enc.finish()]);
                }
                let mut rxs = Vec::new();
                for buf in bufs {
                    let slice = buf.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |r| {
                        let _ = tx.send(r);
                    });
                    rxs.push(rx);
                }
                self.armed = Some(Armed::Maps(rxs));
            }
        }
        Ok(())
    }
}

struct Scratch {
    n_in: usize,
    t: usize,
    /// true when this scratch was built for the gathered readback (which
    /// never touches the MAP_READ staging, so it is a stub there)
    gathered: bool,
    /// true when this scratch was built for the hidden readback: the lm head
    /// and log-softmax dispatches are not in the recorded graph, so the
    /// cached replay is only valid for the same mode
    hidden: bool,
    /// true when the graph was recorded with the fused attention dispatch:
    /// the `kt`/`scores` handles are placeholders there and the recorded
    /// dispatches differ, so a replay must never cross the two modes
    flash: bool,
    /// true when the graph was recorded for a window the 64x64 tile covers
    tile64: bool,
    /// the buffer the recorded graph's final LN writes: `act.x` is a *local*
    /// handle the recording's odd number of swaps leaves pointing at `t3`,
    /// so a replay (whose swaps never run) must not read back through it
    out_x: wgpu::Buffer,
    x_in: wgpu::Buffer,
    convs: Vec<wgpu::Buffer>,
    x: wgpu::Buffer,
    t1: wgpu::Buffer,
    t2: wgpu::Buffer,
    t3: wgpu::Buffer,
    qkv: wgpu::Buffer,
    kt: wgpu::Buffer,
    scores: wgpu::Buffer,
    attn_o: wgpu::Buffer,
    logits: wgpu::Buffer,
    gather_out: Option<wgpu::Buffer>,
    ubuf: wgpu::Buffer,
    jobs: Vec<Vec<Job>>,
    uni: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::{
        check_binding_limits, resolve_flash_attn, resolve_tile64, transpose_pos, uni_binding,
        P_FLASH_ATTN, P_GEMM, P_GEMM64, P_GEMM64_F32, P_GEMM_F32,
    };
    use crate::shaders::{GemmVariant, FLASH_HEAD_DIM};

    /// The model's shape: 16 heads, 10288 vocabulary, f32.
    const HEADS: u64 = 16;
    const VOCAB: u64 = 10288;
    /// 2047 MiB, what this machine's Vulkan adapter reports.
    const BINDING: u64 = 2047 << 20;

    fn bytes(t: u64, read_hidden: bool, hidden: u64) -> (u64, u64) {
        let scores = HEADS * t * t * 4;
        let logits = (if read_hidden { t * hidden } else { t * VOCAB }) * 4;
        (scores, logits)
    }

    #[test]
    fn binding_limits_accept_a_normal_window() {
        // 30 s at 50 fps: 1500 frames -> 144 MB of scores, 62 MB of logits.
        let (scores, logits) = bytes(1500, false, 1024);
        assert_eq!(scores, 144_000_000);
        assert!(logits < BINDING);
        assert!(check_binding_limits(1500, scores, logits, BINDING).is_ok());
        // even a 60 s window still fits this adapter
        let (scores, logits) = bytes(3000, false, 1024);
        assert!(check_binding_limits(3000, scores, logits, BINDING).is_ok());
    }

    #[test]
    fn binding_limits_reject_an_oversized_scores_buffer() {
        // 90 s: 4500 frames -> 1.3 GB of scores, over a 1 GiB binding cap.
        let cap = 1024 << 20;
        let (scores, logits) = bytes(4500, false, 1024);
        assert!(scores > cap && logits < cap);
        let err = check_binding_limits(4500, scores, logits, cap).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("attention scores"), "{msg}");
        assert!(msg.contains(&scores.to_string()), "{msg}");
        assert!(msg.contains(&cap.to_string()), "{msg}");
        // the advice is a window whose scores fit, not the one that failed
        assert!(msg.contains("--window"), "{msg}");
        let secs: u64 = msg
            .split("--window ")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("a concrete --window value");
        assert!(secs >= 1 && secs < 90, "{secs}s does not fit: {msg}");
        let (fits, _) = bytes(secs * 50, false, 1024);
        assert!(fits <= cap, "{secs}s still needs {fits} B of {cap}");
    }

    #[test]
    fn binding_limits_reject_an_oversized_logits_buffer() {
        // A 1 MiB cap: the scores of a 100-frame window fit (640 KB, they
        // grow with t²), but its 4 MB of logits do not.
        let cap = 1 << 20;
        let (scores, logits) = bytes(100, false, 1024);
        assert!(scores < cap && logits > cap, "{scores} {logits}");
        let err = check_binding_limits(100, scores, logits, cap).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("logits"), "{msg}");
        assert!(msg.contains(&logits.to_string()), "{msg}");
        assert!(msg.contains(&cap.to_string()), "{msg}");
        assert!(msg.contains("--window"), "{msg}");
    }

    #[test]
    fn binding_limits_measure_the_hidden_path_by_hidden_size() {
        // The hidden readback never allocates the (t, vocab) logits buffer, so
        // its ceiling is the (t, hidden) one.
        let cap = 8 << 20;
        let (scores, full) = bytes(300, false, 1024);
        let (_, hidden) = bytes(300, true, 1024);
        assert!(hidden < full);
        assert!(check_binding_limits(300, scores, hidden, cap).is_ok());
        assert!(check_binding_limits(300, scores, full, cap).is_err());
    }

    /// 48 KiB, what this machine's Vulkan adapter reports.
    const STORAGE: u32 = 48 * 1024;

    #[test]
    fn the_short_window_tile_follows_its_selector() {
        // A 128-row main tile has something to gain; `M64s` needs 17,408 B
        // as f32 and 8,704 B as f16.
        assert!(resolve_tile64(false, 128, STORAGE, false, None) == false);
        assert!(resolve_tile64(true, 128, STORAGE, false, None));
        assert!(resolve_tile64(true, 128, STORAGE, true, None));
        // `on` is how it gets measured on a device without f16.
        for on in ["1", "true", "TRUE", "on", " On "] {
            assert!(resolve_tile64(false, 128, STORAGE, false, Some(on)), "{on}");
        }
        for off in ["0", "false", "off", "OFF"] {
            assert!(
                !resolve_tile64(true, 128, STORAGE, true, Some(off)),
                "{off}"
            );
        }
        // Off for `auto` where f16 does not make the extra pipeline cheap,
        // which is the zero-regression case on this P104.
        assert!(!resolve_tile64(false, 128, STORAGE, false, Some("auto")));
        assert!(!resolve_tile64(false, 128, STORAGE, false, Some("")));
        // Nothing to gain when the main tile is already 64 rows.
        assert!(!resolve_tile64(true, 64, STORAGE, true, Some("on")));
        // Nor where the tile does not fit: f32 M64s is 17,408 B, f16 8,704 B.
        assert!(!resolve_tile64(false, 128, 8_704, false, Some("on")));
        assert!(resolve_tile64(false, 128, 8_704, true, Some("on")));
        assert!(!resolve_tile64(false, 128, 8_703, true, Some("on")));
        // The chosen variant is the one the dispatch grid assumes.
        assert_eq!(
            GemmVariant::M64s.shared_bytes_with(false),
            17_408,
            "64x64x32 tile"
        );
    }

    #[test]
    fn fused_attention_engages_where_the_scores_tensor_cannot() {
        let head_dim = FLASH_HEAD_DIM as usize;
        let mib = 1u64 << 20;
        // 30 s: 144 MB of scores, well inside this adapter's 2047 MB binding.
        let normal = 144_000_000u64;
        assert!(!resolve_flash_attn(
            None,
            head_dim,
            normal,
            2047 * mib,
            false,
            false
        ));
        // The same window on a device whose binding limit is below it.
        assert!(resolve_flash_attn(
            None,
            head_dim,
            normal,
            64 * mib,
            false,
            false
        ));
        // A window past the hard size cap.
        assert!(resolve_flash_attn(
            None,
            head_dim,
            normal,
            2047 * mib,
            true,
            false
        ));
        // A modern adapter: f16 tiles make the fused path the better one.
        assert!(resolve_flash_attn(
            None,
            head_dim,
            normal,
            2047 * mib,
            false,
            true
        ));
        // Explicit switches win in both directions.
        for on in ["1", "true", "on"] {
            assert!(resolve_flash_attn(
                Some(on),
                head_dim,
                normal,
                2047 * mib,
                false,
                false
            ));
        }
        for off in ["0", "false", "off"] {
            assert!(!resolve_flash_attn(
                Some(off),
                head_dim,
                1 << 40,
                1,
                true,
                true
            ));
        }
        // The kernel is written for the 64-wide head of every Wav2Vec2
        // checkpoint; anything else keeps the materialised path.
        assert!(!resolve_flash_attn(
            Some("on"),
            80,
            normal,
            2047 * mib,
            false,
            false
        ));
        assert!(!resolve_flash_attn(
            Some("on"),
            32,
            normal,
            2047 * mib,
            false,
            false
        ));
    }

    #[test]
    fn the_new_pipelines_keep_the_uniform_binding_their_shaders_declare() {
        // `gemm`-family shaders take `Dims` at binding 4, flash attention its
        // `FlashAttnCfg` at binding 2. A mismatch is a bind-group validation
        // error at the first dispatch, not at load.
        assert_eq!(uni_binding(P_GEMM), 4);
        assert_eq!(uni_binding(P_GEMM_F32), 4);
        assert_eq!(uni_binding(P_GEMM64), 4);
        assert_eq!(uni_binding(P_GEMM64_F32), 4);
        assert_eq!(uni_binding(P_FLASH_ATTN), 2);
        assert_eq!(
            std::mem::size_of::<super::FlashAttnCfg>(),
            32,
            "FlashAttnCfg is 32 bytes"
        );
    }

    #[test]
    fn f32_attention_tile_falls_back_when_f16_main_tile_exceeds_f32_budget() {
        use crate::shaders::GemmVariant;
        // 16 KiB f16 adapter picks Base (13,312 B in f16), which needs 26,624 B
        // in f32 -> must fall back to M64K16 (13,312 B in f32) instead of panicking.
        assert_eq!(
            super::resolve_f32_gemm_variant(GemmVariant::Base, 16 * 1024),
            GemmVariant::M64K16
        );
        // 32 KiB f16 adapter with explicit Db16 (22,528 B in f16) needs 45,056 B
        // in f32 -> falls back to Base (26,624 B in f32).
        assert_eq!(
            super::resolve_f32_gemm_variant(GemmVariant::Db16, 32 * 1024),
            GemmVariant::Base
        );
        // 48 KiB adapter keeps the same variant.
        assert_eq!(
            super::resolve_f32_gemm_variant(GemmVariant::Db16, 48 * 1024),
            GemmVariant::Db16
        );
    }

    #[test]
    fn pos_weight_index_matches_scalar_dot() {
        let (out_c, in_pg, taps, frames) = (128usize, 64, 8, 5);
        let mut w = vec![0f32; out_c * in_pg * taps];
        for (i, x) in w.iter_mut().enumerate() {
            *x = (i % 17) as f32;
        }
        let tw = transpose_pos(&w, out_c, in_pg, taps);
        let mut x = vec![0f32; frames * out_c];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 3) % 19) as f32 - 8.0;
        }
        let pad = taps / 2;
        let k_total = in_pg * taps;
        for j in 0..frames {
            for o in 0..out_c {
                let g = o / in_pg;
                let oc = o % in_pg;
                let mut acc = 0.0f32;
                for tk in 0..taps {
                    let src = j as isize + tk as isize - pad as isize;
                    if src < 0 || src >= frames as isize {
                        continue;
                    }
                    for ci in 0..in_pg {
                        acc += x[src as usize * out_c + g * in_pg + ci]
                            * w[(o * in_pg + ci) * taps + tk];
                    }
                }
                let mut acc2 = 0.0f32;
                for kval in 0..k_total {
                    let tk = kval / in_pg;
                    let ci = kval % in_pg;
                    let src = j as isize + tk as isize - pad as isize;
                    if src < 0 || src >= frames as isize {
                        continue;
                    }
                    acc2 += x[src as usize * out_c + g * in_pg + ci]
                        * tw[(g * k_total + kval) * in_pg + oc];
                }
                assert!((acc - acc2).abs() < 1e-3, "j={j} o={o} {acc} vs {acc2}");
            }
        }
    }
}

struct ActSet {
    x: wgpu::Buffer,
    t1: wgpu::Buffer,
    t2: wgpu::Buffer,
    t3: wgpu::Buffer,
    qkv: wgpu::Buffer,
    kt: wgpu::Buffer,
    scores: wgpu::Buffer,
    attn_o: wgpu::Buffer,
    logits: wgpu::Buffer,
}
