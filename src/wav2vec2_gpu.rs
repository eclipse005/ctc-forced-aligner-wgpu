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
    u32::from(off % 4 == 0 && stride % 4 == 0 && z % 4 == 0)
}

/// `[out, in_pg, taps]` → `[group, tap * in_pg + ci, oc]`, oc contiguous.
fn transpose_pos(w: &[f32], out_c: usize, in_pg: usize, taps: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_c * in_pg * taps);
    let k_total = in_pg * taps;
    let mut out = vec![0f32; w.len()];
    out.par_chunks_mut(in_pg).enumerate().for_each(|(idx, dst)| {
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

struct Pipe {
    gemm: wgpu::ComputePipeline,
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

/// The uniform binding index of each pipeline, and whether its uniform buffer
/// is shared (one big buffer + per-dispatch offsets).
fn uni_binding(p: usize) -> u32 {
    match p {
        P_GEMM | P_CONV_GEMM | P_CONV0 | P_POS | P_LN_SD => 4,
        P_LN | P_ADD => 3,
        P_SOFTMAX | P_LOGSOFTMAX => 1,
        P_TRANSPOSE => 2,
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
        let g = get_f32(&tensors, "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0")?;
        let v = get_f32(&tensors, "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1")?;
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
                let w = get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.weight"))
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
        let lm_wt = transpose(&get_f32(&tensors, "lm_head.weight")?.0, cfg.vocab_size, cfg.hidden_size);
        let fp_wt = transpose(&get_f32(&tensors, "wav2vec2.feature_projection.projection.weight")?.0, cfg.hidden_size, 512);

        let lin = |name: String| -> Result<(Vec<f32>, Vec<f32>)> {
            let w = get_f32(&tensors, &format!("{name}.weight"))?.0;
            let b = get_f32(&tensors, &format!("{name}.bias"))?.0;
            Ok((w, b))
        };

        let layers: Vec<(Vec<f32>, [Vec<f32>; 10])> = (0..cfg.num_hidden_layers)
            .into_par_iter()
            .map(|i| {
                let base = |sub: &str| format!("wav2vec2.encoder.layers.{i}.{sub}");
                let tr = |w: &[f32], out: usize, inp: usize| transpose(w, out, inp);
                let (qw, qb) = lin(base("attention.q_proj")).unwrap();
                let (kw, _kb) = lin(base("attention.k_proj")).unwrap();
                let (vw, _vb) = lin(base("attention.v_proj")).unwrap();
                let (kb, vb) = (
                    lin(base("attention.k_proj")).unwrap().1,
                    lin(base("attention.v_proj")).unwrap().1,
                );
                let (ow, ob) = lin(base("attention.out_proj")).unwrap();
                let (f1w, f1b) = lin(base("feed_forward.intermediate_dense")).unwrap();
                let (f2w, f2b) = lin(base("feed_forward.output_dense")).unwrap();
                let (l1w, l1b) = lin(base("layer_norm")).unwrap();
                let (l2w, _l2b) = lin(base("final_layer_norm")).unwrap();
                // fused (K, 3K): row k = [Wq[:,k] | Wk[:,k] | Wv[:,k]]
                let qt = tr(&qw, cfg.hidden_size, cfg.hidden_size);
                let kt = tr(&kw, cfg.hidden_size, cfg.hidden_size);
                let vt = tr(&vw, cfg.hidden_size, cfg.hidden_size);
                let hs = cfg.hidden_size;
                let mut qkv = vec![0.0f32; hs * 3 * hs];
                for r in 0..hs {
                    qkv[r * 3 * hs..r * 3 * hs + hs].copy_from_slice(&qt[r * hs..(r + 1) * hs]);
                    qkv[r * 3 * hs + hs..r * 3 * hs + 2 * hs].copy_from_slice(&kt[r * hs..(r + 1) * hs]);
                    qkv[r * 3 * hs + 2 * hs..r * 3 * hs + 3 * hs].copy_from_slice(&vt[r * hs..(r + 1) * hs]);
                }
                let mut qkvb = qb.clone();
                qkvb.extend_from_slice(&kb);
                qkvb.extend_from_slice(&vb);
                (
                    qkvb,
                    [
                        qkv,
                        tr(&ow, cfg.hidden_size, cfg.hidden_size),
                        ob,
                        tr(&f1w, cfg.intermediate_size, cfg.hidden_size),
                        f1b,
                        tr(&f2w, cfg.hidden_size, cfg.intermediate_size),
                        f2b,
                        l1w,
                        l1b,
                        l2w,
                    ],
                )
            })
            .collect();
        // ln2 biases were folded into the array order: rebuild them
        let ln2bs: Vec<Vec<f32>> = (0..cfg.num_hidden_layers)
            .map(|i| {
                lin(format!("wav2vec2.encoder.layers.{i}.final_layer_norm")).unwrap().1
            })
            .collect();

        // ---- upload
        let mut up = gpu.uploader();
        let put = |up: &mut BulkUpload, data: &[f32], label: &str| -> wgpu::Buffer {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for f in data {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            let buf = up.storage(label, bytes.len() as u64);
            up.upload(&buf, &bytes).unwrap();
            buf
        };

        let conv0_w = put(&mut up, &get_f32(&tensors, "wav2vec2.feature_extractor.conv_layers.0.conv.weight")?.0, "conv0.w");
        let conv_wt_b: Vec<_> = conv_wt
            .iter()
            .enumerate()
            .map(|(i, wt)| put(&mut up, wt, &format!("conv{}.wt", i + 1)))
            .collect();
        let conv_b: Vec<_> = (0..7)
            .map(|i| {
                put(
                    &mut up,
                    &get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.bias")).unwrap().0,
                    &format!("conv{i}.b"),
                )
            })
            .collect();
        let (mut conv_ln_w, mut conv_ln_b) = (Vec::new(), Vec::new());
        for i in 0..7 {
            conv_ln_w.push(put(&mut up, &get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.weight"))?.0, &format!("conv{i}.lnw")));
            conv_ln_b.push(put(&mut up, &get_f32(&tensors, &format!("wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.bias"))?.0, &format!("conv{i}.lnb")));
        }
        let fp_ln_w = put(&mut up, &get_f32(&tensors, "wav2vec2.feature_projection.layer_norm.weight")?.0, "fp.lnw");
        let fp_ln_b = put(&mut up, &get_f32(&tensors, "wav2vec2.feature_projection.layer_norm.bias")?.0, "fp.lnb");
        let fp_wt_b = put(&mut up, &fp_wt, "fp.wt");
        let fp_b = put(&mut up, &get_f32(&tensors, "wav2vec2.feature_projection.projection.bias")?.0, "fp.b");
        let pos_w_b = put(&mut up, &pos_wt, "pos.w");
        let pos_b_b = put(&mut up, &get_f32(&tensors, "wav2vec2.encoder.pos_conv_embed.conv.bias")?.0, "pos.b");

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
        for (i, (qkvb, arr)) in layers.iter().enumerate() {
            let base = |sub: &str| format!("wav2vec2.encoder.layers.{i}.{sub}");
            qkv_wt_b.push(put(&mut up, &arr[0], &format!("l{i}.qkvt")));
            qkv_b_b.push(put(&mut up, qkvb, &format!("l{i}.qkvb")));
            out_wt_b.push(put(&mut up, &arr[1], &format!("l{i}.outwt")));
            out_b_b.push(put(&mut up, &arr[2], &format!("l{i}.outb")));
            ff1_wt_b.push(put(&mut up, &arr[3], &format!("l{i}.ff1wt")));
            ff1_b_b.push(put(&mut up, &arr[4], &format!("l{i}.ff1b")));
            ff2_wt_b.push(put(&mut up, &arr[5], &format!("l{i}.ff2wt")));
            ff2_b_b.push(put(&mut up, &arr[6], &format!("l{i}.ff2b")));
            ln1_w_b.push(put(&mut up, &arr[7], &format!("l{i}.ln1w")));
            ln1_b_b.push(put(&mut up, &arr[8], &format!("l{i}.ln1b")));
            let _ = base;
            ln2_w_b.push(put(&mut up, &arr[9], &format!("l{i}.ln2w")));
        }
        let ln2_b_buf: Vec<_> = ln2bs
            .iter()
            .enumerate()
            .map(|(i, b)| put(&mut up, b, &format!("l{i}.ln2b")))
            .collect();
        let final_ln_w = put(&mut up, &get_f32(&tensors, "wav2vec2.encoder.layer_norm.weight")?.0, "final.lnw");
        let final_ln_b = put(&mut up, &get_f32(&tensors, "wav2vec2.encoder.layer_norm.bias")?.0, "final.lnb");
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
        let variant = shaders::pick_gemm_variant();
        let (gemm_mt, gemm_nt) = shaders::gemm_tile(variant);
        let pipes = Pipe {
            gemm: mk_gemm(&shaders::gemm_variant(variant), "gemm")?,
            conv_gemm: mk_gemm(&shaders::conv_gemm_variant(variant), "conv_gemm")?,
            conv0: mk(&shaders::conv0(), "conv0")?,
            pos_conv: mk_gemm(&shaders::pos_conv(), "pos_conv")?,
            ln: mk(&shaders::layernorm(), "ln")?,
            ln_sd: mk(&shaders::layernorm_sd(), "ln_sd")?,
            add: mk(&shaders::add(), "add")?,
            softmax: mk(&shaders::softmax(), "softmax")?,
            log_softmax: mk(&shaders::log_softmax(), "log_softmax")?,
            transpose: mk(&shaders::transpose(), "transpose")?,
            gather: mk(&shaders::gather(), "gather")?,
        };

        Ok(Self {
            gpu,
            cfg,
            pipes,
            gemm_mt,
            gemm_nt,
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
        let p = self.forward_begin(input, None, false, None, &[], None)?;
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
        let p = self.forward_begin(input, Some(expanded), false, keep, stars, None)?;
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
        self.forward_begin(input, Some(expanded), false, keep, stars, prev)
    }

    /// Pipelined [`forward_hidden`] — see [`GpuModel::forward_gathered_begin`].
    pub fn forward_hidden_begin(
        &self,
        input: &[f32],
        prev: Option<&mut GpuPending>,
    ) -> Result<GpuPending> {
        self.forward_begin(input, None, true, None, &[], prev)
    }

    /// Pipelined full-logits [`forward`] — see
    /// [`GpuModel::forward_gathered_begin`].
    pub fn forward_logits_begin(
        &self,
        input: &[f32],
        prev: Option<&mut GpuPending>,
    ) -> Result<GpuPending> {
        self.forward_begin(input, None, false, None, &[], prev)
    }

    /// Same forward, but the lm head and log-softmax never run: the host gets
    /// the post-final-LN encoder stream (T, hidden) — 7 MB a chunk against
    /// 70 MB of logits.  The head is re-run on the host per window
    /// ([`crate::wav2vec2::LmHeadCpu`]) when the DP needs trellis columns,
    /// which is what keeps a long transcript's emissions out of RAM.
    pub fn forward_hidden(&self, input: &[f32]) -> Result<Vec<f32>> {
        let p = self.forward_begin(input, None, true, None, &[], None)?;
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
        mut prev: Option<&mut GpuPending>,
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
        let f32s = |n: usize| (n * 4) as u64;

        let scores_n = heads
            .checked_mul(t)
            .and_then(|n| n.checked_mul(t))
            .context("frame count overflow")?;
        if t > 12000 || scores_n > 200_000_000 {
            anyhow::bail!(
                "sequence of {t} frames is too long for one GPU forward; use --window 30"
            );
        }
        let (x_in, convs, cached, mut act, ubuf, gather_buf) = {
            let mut guard = self.scratch.lock().unwrap();
            let gathered = gather.is_some();
            let reuse = guard
                .as_ref()
                .map(|s| {
                    s.n_in == input.len() && s.t == t && s.gathered == gathered && s.hidden == read_hidden
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
                    x_in: st("x_in", input.len()),
                    convs,
                    x: xbuf.clone(),
                    out_x: xbuf,
                    t1: st("t1", t * hidden),
                    t2: st("t2", t * hidden.max(self.cfg.intermediate_size)),
                    t3: st("t3", t * hidden),
                    qkv: st("qkv", t * 3 * hidden),
                    kt: st("kt", hidden * t),
                    scores: st("scores", scores_n),
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
        let pipes: [&wgpu::ComputePipeline; 11] = [
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
                wgpu::BindGroupEntry { binding: $idx, resource: $b.as_entire_binding() }
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
            Cfg4 { a: t0 as u32, b: 512, c: self.cfg.conv_kernel[0] as u32, d: self.cfg.conv_stride[0] as u32 },
            (t0 as u32).div_ceil(16), (512u32).div_ceil(16),
            bind!(0, &x_in), bind!(1, &self.conv0_w), bind!(2, &self.conv_b[0]), bind!(3, &convs[0]),
        );
        dispatch!(
            P_LN,
            Cfg4 { a: t0 as u32, b: 512, c: 10, d: 1 }, // eps 1e-5 -> 10 * 1e-6; gelu on
            row_grid(t0 as u32).0, row_grid(t0 as u32).1,
            bind!(0, &convs[0]), bind!(1, &self.conv_ln_w[0]), bind!(2, &self.conv_ln_b[0]),
        );

        // ---- conv layers 1..6 (each reads the previous layer's buffer)
        let mut rows = t0;
        for i in 1..7 {
            let tout = (rows - self.cfg.conv_kernel[i]) / self.cfg.conv_stride[i] + 1;
            dispatch!(
                P_CONV_GEMM,
                ConvDims {
                    m: tout as u32, n: 512,
                    k: (self.cfg.conv_kernel[i] * 512) as u32,
                    c_in: 512, stride: self.cfg.conv_stride[i] as u32, t_in: rows as u32,
                    a_vec: 1, b_vec: 1,
                },
                (tout as u32).div_ceil(gemm_mt), (512u32).div_ceil(gemm_nt),
                bind!(0, &convs[i - 1]), bind!(1, &self.conv_wt[i - 1]), bind!(2, &self.conv_b[i]), bind!(3, &convs[i]),
                bind!(5, &convs[i - 1]), bind!(6, &self.conv_wt[i - 1]),
            );
            dispatch!(
                P_LN,
                Cfg4 { a: tout as u32, b: 512, c: 10, d: 1 },
                row_grid(tout as u32).0, row_grid(tout as u32).1,
                bind!(0, &convs[i]), bind!(1, &self.conv_ln_w[i]), bind!(2, &self.conv_ln_b[i]),
            );
            rows = tout;
        }
        let cur = &convs[6];

        // ---- feature projection: LN(512) on conv6 output, then Linear(512->1024)
        dispatch!(
            P_LN,
            Cfg4 { a: rows as u32, b: 512, c: 10, d: 0 },
            row_grid(rows as u32).0, row_grid(rows as u32).1,
            bind!(0, &cur), bind!(1, &self.fp_ln_w), bind!(2, &self.fp_ln_b),
        );
        dispatch!(
            P_GEMM,
            GemmDims {
                m: rows as u32, n: hidden as u32, k: 512,
                a_stride: 512, b_stride: hidden as u32, c_stride: hidden as u32,
                a_off: 0, b_off: 0, c_off: 0, a_z: 0, b_z: 0, c_z: 0, epilogue: 0, scale: 1.0,
                a_vec: vec4_ok(0, 512, 0), b_vec: vec4_ok(0, hidden as u32, 0), _p0: 0,
            },
            (rows as u32).div_ceil(gemm_mt), (hidden as u32).div_ceil(gemm_nt),
            bind!(0, &cur), bind!(1, &self.fp_wt), bind!(2, &self.fp_b), bind!(3, &act.x), bind!(5, &self.zeros),
            bind!(6, &cur), bind!(7, &self.fp_wt),
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
            (t as u32).div_ceil(shaders::POS_MT),
            pos_groups,
            bind!(0, &act.x), bind!(1, &self.pos_w), bind!(2, &self.pos_b), bind!(3, &act.t1),
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
            Cfg4 { a: (t * hidden) as u32, b: 0, c: 0, d: 0 },
            add_x, add_y,
            bind!(0, &act.x), bind!(1, &act.t1), bind!(2, &act.t3),
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
                Cfg4 { a: t_rows, b: cols, c: 10, d: 0 },
                row_grid(t_rows).0, row_grid(t_rows).1,
                bind!(0, &act.x), bind!(1, &act.t1), bind!(2, &self.ln1_w[li]), bind!(3, &self.ln1_b[li]),
            );
            // qkv
            dispatch!(
                P_GEMM,
                GemmDims {
                    m: t_rows, n: 3 * cols, k: cols,
                    a_stride: cols, b_stride: 3 * cols, c_stride: 3 * cols,
                    a_off: 0, b_off: 0, c_off: 0, a_z: 0, b_z: 0, c_z: 0, epilogue: 0, scale: 1.0,
                    a_vec: vec4_ok(0, cols, 0), b_vec: vec4_ok(0, 3 * cols, 0), _p0: 0,
                },
                t_rows.div_ceil(gemm_mt), (3 * cols).div_ceil(gemm_nt),
                bind!(0, &act.t1), bind!(1, &self.qkv_wt[li]), bind!(2, &self.qkv_b[li]), bind!(3, &act.qkv), bind!(5, &self.zeros),
                bind!(6, &act.t1), bind!(7, &self.qkv_wt[li]),
            );
            // K^T: rows hidden, cols t, from qkv rows offset T*1024, stride 3072
            dispatch!(
                P_TRANSPOSE,
                // K block starts `hidden` elements into each (T, 3*hidden) row
                Cfg4 { a: t as u32, b: hidden as u32, c: (3 * hidden) as u32, d: hidden as u32 },
                (t as u32).div_ceil(16), (hidden as u32).div_ceil(16),
                bind!(0, &act.qkv), bind!(1, &act.kt),
            );
            // per-head scores, z-batched: scores_h = q_h @ kt_h over wid.z.
            // A is a vec4-able slice of qkv; B (kt) strides by t, which is odd
            // for most windows, so b_vec computes to 0 and the scalar path runs
            dispatch_z!(
                P_GEMM,
                GemmDims {
                    m: t_rows, n: t_rows, k: head_dim as u32,
                    a_stride: 3 * cols, b_stride: t as u32, c_stride: t_rows,
                    a_off: 0, a_z: head_dim as u32,
                    b_off: 0, b_z: head_dim as u32 * t as u32,
                    c_off: 0, c_z: t_rows * t_rows,
                    epilogue: 0,
                    scale: 0.125, // head_dim^-0.5: the attention scaling
                    a_vec: vec4_ok(0, 3 * cols, head_dim as u32),
                    b_vec: vec4_ok(0, t as u32, head_dim as u32 * t as u32),
                    _p0: 0,
                },
                t_rows.div_ceil(gemm_mt), t_rows.div_ceil(gemm_nt), heads as u32,
                bind!(0, &act.qkv), bind!(1, &act.kt), bind!(2, &self.zeros), bind!(3, &act.scores), bind!(5, &self.zeros),
                bind!(6, &act.qkv), bind!(7, &act.kt),
            );
            // softmax over each (head, query) row
            dispatch!(
                P_SOFTMAX,
                Cfg4 { a: (heads as u32 * t_rows), b: t_rows, c: 0, d: 0 },
                row_grid(heads as u32 * t_rows).0, row_grid(heads as u32 * t_rows).1,
                bind!(0, &act.scores),
            );
            // per-head weighted V, z-batched: attn_h = scores_h @ v_h.
            // B is a vec4-able slice of qkv; A (scores) strides by t, so
            // a_vec computes to 0 and the scalar path runs
            dispatch_z!(
                P_GEMM,
                GemmDims {
                    m: t_rows, n: head_dim as u32, k: t_rows,
                    a_stride: t_rows, b_stride: 3 * cols, c_stride: cols,
                    a_off: 0, a_z: t_rows * t_rows,
                    b_off: (2 * hidden) as u32, b_z: head_dim as u32,
                    c_off: 0, c_z: head_dim as u32,
                    epilogue: 0,
                    scale: 1.0,
                    a_vec: vec4_ok(0, t_rows, t_rows * t_rows),
                    b_vec: vec4_ok((2 * hidden) as u32, 3 * cols, head_dim as u32),
                    _p0: 0,
                },
                t_rows.div_ceil(gemm_mt), (head_dim as u32).div_ceil(gemm_nt), heads as u32,
                bind!(0, &act.scores), bind!(1, &act.qkv), bind!(2, &self.zeros), bind!(3, &act.attn_o), bind!(5, &self.zeros),
                bind!(6, &act.scores), bind!(7, &act.qkv),
            );
            // out proj + residual
            dispatch!(
                P_GEMM,
                GemmDims {
                    m: t_rows, n: cols, k: cols,
                    a_stride: cols, b_stride: cols, c_stride: cols,
                    a_off: 0, b_off: 0, c_off: 0, a_z: 0, b_z: 0, c_z: 0, epilogue: 2, scale: 1.0,
                    a_vec: vec4_ok(0, cols, 0), b_vec: vec4_ok(0, cols, 0), _p0: 0,
                },
                t_rows.div_ceil(gemm_mt), cols.div_ceil(gemm_nt),
                bind!(0, &act.attn_o), bind!(1, &self.out_wt[li]), bind!(2, &self.out_b[li]), bind!(3, &act.t3), bind!(5, &act.x),
                bind!(6, &act.attn_o), bind!(7, &self.out_wt[li]),
            );
            // FFN: t3 is the new residual; swap x <-> t3
            std::mem::swap(&mut act.x, &mut act.t3);
            // t1 = LN(x) — again src/dst, no staging copy
            dispatch!(
                P_LN_SD,
                Cfg4 { a: t_rows, b: cols, c: 10, d: 0 },
                row_grid(t_rows).0, row_grid(t_rows).1,
                bind!(0, &act.x), bind!(1, &act.t1), bind!(2, &self.ln2_w[li]), bind!(3, &self.ln2_b[li]),
            );
            dispatch!(
                P_GEMM,
                GemmDims {
                    m: t_rows, n: self.cfg.intermediate_size as u32, k: cols,
                    a_stride: cols, b_stride: self.cfg.intermediate_size as u32, c_stride: self.cfg.intermediate_size as u32,
                    a_off: 0, b_off: 0, c_off: 0, a_z: 0, b_z: 0, c_z: 0, epilogue: 1, scale: 1.0,
                    a_vec: vec4_ok(0, cols, 0), b_vec: vec4_ok(0, self.cfg.intermediate_size as u32, 0), _p0: 0,
                },
                t_rows.div_ceil(gemm_mt), (self.cfg.intermediate_size as u32).div_ceil(gemm_nt),
                bind!(0, &act.t1), bind!(1, &self.ff1_wt[li]), bind!(2, &self.ff1_b[li]), bind!(3, &act.t2), bind!(5, &self.zeros),
                bind!(6, &act.t1), bind!(7, &self.ff1_wt[li]),
            );
            dispatch!(
                P_GEMM,
                GemmDims {
                    m: t_rows, n: cols, k: self.cfg.intermediate_size as u32,
                    a_stride: self.cfg.intermediate_size as u32, b_stride: cols, c_stride: cols,
                    a_off: 0, b_off: 0, c_off: 0, a_z: 0, b_z: 0, c_z: 0, epilogue: 2, scale: 1.0,
                    a_vec: vec4_ok(0, self.cfg.intermediate_size as u32, 0), b_vec: vec4_ok(0, cols, 0), _p0: 0,
                },
                t_rows.div_ceil(gemm_mt), cols.div_ceil(gemm_nt),
                bind!(0, &act.t2), bind!(1, &self.ff2_wt[li]), bind!(2, &self.ff2_b[li]), bind!(3, &act.t1), bind!(5, &act.x),
                bind!(6, &act.t2), bind!(7, &self.ff2_wt[li]),
            );
            std::mem::swap(&mut act.x, &mut act.t1);
        }

        // ---- final LN, LM head, log softmax.  The hidden readback stops at
        // the LN: act.x then holds the stream the host-side head re-run
        // consumes, and the two head dispatches stay out of the graph.
        dispatch!(
            P_LN,
            Cfg4 { a: t as u32, b: hidden as u32, c: 10, d: 0 },
            row_grid(t as u32).0, row_grid(t as u32).1,
            bind!(0, &act.x), bind!(1, &self.final_ln_w), bind!(2, &self.final_ln_b),
        );
        if !read_hidden {
        dispatch!(
            P_GEMM,
            GemmDims {
                m: t as u32, n: vocab as u32, k: hidden as u32,
                a_stride: hidden as u32, b_stride: vocab as u32, c_stride: vocab as u32,
                a_off: 0, b_off: 0, c_off: 0, a_z: 0, b_z: 0, c_z: 0, epilogue: 0, scale: 1.0,
                a_vec: vec4_ok(0, hidden as u32, 0), b_vec: vec4_ok(0, vocab as u32, 0), _p0: 0,
            },
            (t as u32).div_ceil(gemm_mt), (vocab as u32).div_ceil(gemm_nt),
            bind!(0, &act.x), bind!(1, &self.lm_wt), bind!(2, &self.lm_b), bind!(3, &act.logits), bind!(5, &self.zeros),
            bind!(6, &act.x), bind!(7, &self.lm_wt),
        );
        dispatch!(
            P_LOGSOFTMAX,
            Cfg4 { a: t as u32, b: vocab as u32, c: 0, d: 0 },
            row_grid(t as u32).0, row_grid(t as u32).1,
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
                Cfg4 { a: total, b: expanded.len() as u32, c: vocab as u32, d: 0 },
                gx, gy,
                bind!(0, &act.logits), bind!(1, &self.labels), bind!(2, gather_buf.as_ref().context("missing gather buffer")?),
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
        static SPLIT_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_split = prof_mode.as_deref() == Some("2")
            && gpu.features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && !SPLIT_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed);
        let t_replay = std::time::Instant::now();
        // Input and every uniform slot go out before the first command buffer.
        // The three layer-groups are queued back to back; each buffer stays
        // under the Windows TDR window, and one wait keeps the GPU busy.
        // The window pipeline's hinge: the previous chunk's result buffers are
        // scratch-resident and THIS chunk's dispatches are what overwrite them,
        // so its device->host copies must be enqueued here, ahead of our
        // compute. Arming the maps now means the map callbacks fire when those
        // copies execute -- not when this chunk's compute does -- so the
        // caller's collect() below overlaps our 500+ ms of compute.
        if let Some(p) = prev.as_deref_mut() {
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
            // indexed by pipeline id (P_*): gelu has no pipeline (fused into
            // the LN and the GEMM epilogue), so the names after P_LN_SD were
            // off by one slot until this list stopped carrying a gelu entry
            let names = [
                "gemm", "conv_gemm", "conv0", "pos", "ln", "ln_sd", "add", "softmax",
                "logsoftmax", "transpose", "gather", "  scores", "  pv", "  gemm-main",
            ];
            let mut acc = [0f64; 16];
            let mut cnt = [0u32; 16];
            for (i, &(p, gy)) in job_meta.iter().enumerate() {
                let dt = ticks[i * 2 + 1].saturating_sub(ticks[i * 2]) as f64 * period / 1e6;
                if p < acc.len() {
                    // split the gemm pipeline by dispatch shape: pv is n=64
                    // (gy=1), scores is n=t, everything else is a main GEMM.
                    // The three buckets are the last three names above.
                    let mut b = p as usize;
                    if p == P_GEMM {
                        b = if gy == 1 {
                            12
                        } else if gy == (t as u32).div_ceil(gemm_nt) {
                            11
                        } else {
                            13
                        };
                    }
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
                gather_buf.as_ref().context("missing gather buffer")?.clone(),
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
        Ok(GpuPending::new(&self.gpu, buffer, offset, floats))
    }

    /// Map a pending chunk's result back to the host.
    ///
    /// A pending whose staging copies were armed by the *next* chunk's begin
    /// collects while the GPU still computes: the pump below waits only for
    /// the copies' maps, which execute ahead of that compute. The last chunk
    /// of a run arms itself here and waits for its own copy.
    pub fn collect(&self, mut p: GpuPending) -> Result<Vec<f32>> {
        let gpu = &self.gpu;
        if p.armed.is_none() {
            p.arm(gpu)?;
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| anyhow::anyhow!("poll for readback: {e}"))?;
        }
        let rxs = p.armed.take().context("pending maps already consumed")?;
        // pump until every piece's map has landed; each poll also retires
        // whatever compute is in flight
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
        let mut out: Vec<f32> = Vec::with_capacity(p.floats);
        for (piece, &(_, take)) in p.staging.iter().zip(&p.pieces) {
            let slice = piece.slice(..);
            let mapped = slice.get_mapped_range()?;
            out.extend_from_slice(bytemuck::cast_slice(&mapped[..take as usize]));
            drop(mapped);
            piece.unmap();
        }
        Ok(out)
    }
}

const UNIFORM_ALIGN: u64 = 256;

const READBACK_CHUNK: u64 = 16 << 20;

/// One submitted chunk whose result is still on the device.
///
/// The staging pieces are allocated at submit time. The NEXT chunk's begin
/// enqueues the device->host copies into them — ahead of its own compute,
/// which is what overwrites the result buffers — and arms the maps, so
/// `collect` pumps until the copies land while the GPU keeps computing. A
/// pending that no begin follows (the last chunk of a run) arms itself in
/// `collect` and waits for its own copy.
pub(crate) struct GpuPending {
    /// the device buffer holding this chunk's result: the gather-out buffer
    /// (gathered form), the recorded final LN's output (hidden form), or the
    /// logits buffer (full form)
    buffer: wgpu::Buffer,
    /// source byte offset and byte length of each staging piece
    pieces: Vec<(u64, u64)>,
    staging: Vec<wgpu::Buffer>,
    floats: usize,
    /// map receivers, once armed
    armed: Option<Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
}

impl GpuPending {
    fn new(gpu: &Gpu, buffer: wgpu::Buffer, offset: u64, floats: usize) -> Self {
        let bytes = (floats * 4) as u64;
        let mut pieces = Vec::new();
        let mut staging = Vec::new();
        let mut off = offset;
        while off < offset + bytes {
            let take = READBACK_CHUNK.min(offset + bytes - off);
            let size = (take + 3) & !3;
            staging.push(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            pieces.push((off, take));
            off += take;
        }
        Self {
            buffer,
            pieces,
            staging,
            floats,
            armed: None,
        }
    }

    /// Enqueue this pending's staging copies and arm the maps. Called from the
    /// next chunk's begin, before its compute submits: the queue executes the
    /// copies first, so they read the result before the compute overwrites it,
    /// and the map callbacks fire when the copies do.
    fn arm(&mut self, gpu: &Gpu) -> Result<()> {
        {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            for (piece, &(src_off, take)) in self.staging.iter().zip(&self.pieces) {
                enc.copy_buffer_to_buffer(&self.buffer, src_off, piece, 0, take);
            }
            gpu.queue.submit([enc.finish()]);
        }
        let mut rxs = Vec::with_capacity(self.staging.len());
        for piece in &self.staging {
            let slice = piece.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rxs.push(rx);
        }
        self.armed = Some(rxs);
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
    use super::transpose_pos;

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

