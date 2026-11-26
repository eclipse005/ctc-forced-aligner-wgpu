//! Device / queue wrapper and buffer plumbing (adapted from the
//! qwen-aligner-wgpu reference implementation).
//!
//! Storage buffers are always allocated 16-byte padded; activations here are
//! f32 (the model checkpoint is f32; a 16-bit storage mode can be added later
//! the same way the reference does).

use anyhow::{bail, Context, Result};

use crate::shaders::SubgroupMode;

/// A wgpu device plus the queue, adapter info and negotiated limits.
pub(crate) struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub(crate) info: wgpu::AdapterInfo,
    /// The features the device was actually opened with. The adapter's set is
    /// a superset: `request_device` may drop one it cannot honour, so every
    /// capability check reads this.
    pub features: wgpu::Features,
    /// The limits the device was actually opened with, likewise negotiated
    /// rather than merely advertised.
    pub limits: wgpu::Limits,
    pub pipeline_cache: Option<wgpu::PipelineCache>,
    /// Where [`Gpu::save_pipeline_cache`] writes, when caching is supported.
    pipeline_cache_path: Option<std::path::PathBuf>,
    /// Bytes last loaded from / written to the cache file. Comparing against
    /// them keeps a warm start from rewriting a multi-megabyte file it would
    /// only reproduce byte for byte.
    pipeline_cache_bytes: std::sync::Mutex<Vec<u8>>,
    /// Dedicated Vulkan transfer family, alongside the compute queue on family 0.
    /// `None` when the device has no copy-only queue or the open fell back.
    pub xfer_family: Option<u32>,
}

/// Outlives the device-create callback. The callback's queue priority slice
/// has to stay valid until `vkCreateDevice` returns.
#[cfg(any(target_os = "linux", target_os = "windows"))]
static XFER_Q_PRIO: f32 = 1.0;

/// One device: `auto`, `cpu`, `vulkan[:i]`, `dx12[:i]`, `#n`, or a name substring.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    #[default]
    Auto,
    Runtime {
        api: wgpu::Backend,
        index: usize,
    },
    Cpu,
    Index(usize),
    Name(String),
}

pub(crate) const RUNTIMES: &[(&str, wgpu::Backend)] = &[
    ("vulkan", wgpu::Backend::Vulkan),
    ("metal", wgpu::Backend::Metal),
    ("dx12", wgpu::Backend::Dx12),
    ("d3d12", wgpu::Backend::Dx12),
    ("gl", wgpu::Backend::Gl),
];

impl DeviceSelector {
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if s.eq_ignore_ascii_case("dual") {
            bail!("one device only");
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if s.eq_ignore_ascii_case("gpu") {
            bail!("`gpu` is a backend, not an adapter name — use Backend::Gpu or `--device gpu`");
        }
        if let Some(rest) = s.strip_prefix('#') {
            return Ok(Self::Index(rest.trim().parse().context("device index")?));
        }
        if let Ok(i) = s.parse::<usize>() {
            return Ok(Self::Index(i));
        }
        let (name, index) = match s.split_once(':') {
            Some((n, i)) => (n, i.trim().parse().context("runtime device index")?),
            None => (s, 0usize),
        };
        if let Some((_, api)) = RUNTIMES.iter().find(|(n, _)| name.eq_ignore_ascii_case(n)) {
            return Ok(Self::Runtime { api: *api, index });
        }
        Ok(Self::Name(s.to_lowercase()))
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        match self {
            Self::Auto | Self::Index(_) | Self::Cpu => true,
            Self::Name(n) => info.name.to_lowercase().contains(n),
            Self::Runtime { api, .. } => info.backend == *api,
        }
    }
}

/// Which tower to run, separate from which adapter [`DeviceSelector`] names.
///
/// * [`Self::Auto`] — a real GPU, or the CPU tower when none can be opened.
///   An unpinned selector tries discrete, then integrated, then a virtual GPU.
/// * [`Self::Cpu`] — the CPU tower. No adapter.
/// * [`Self::Gpu`] — that adapter, or an error. `DeviceSelector::Auto` walks
///   discrete then integrated and does not fall back to the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Cpu,
    Gpu(DeviceSelector),
}

impl Backend {
    /// `auto` / `cpu` / `gpu`, or anything [`DeviceSelector::parse`] accepts
    /// wrapped as [`Self::Gpu`] (`vulkan:0`, `#1`, an adapter substring).
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if s.eq_ignore_ascii_case("gpu") {
            return Ok(Self::Gpu(DeviceSelector::Auto));
        }
        Ok(Self::Gpu(DeviceSelector::parse(s)?))
    }

    /// `Cpu` and `Auto` keep their policies. Any other selector is a required GPU.
    pub fn from_selector(selector: DeviceSelector) -> Self {
        match selector {
            DeviceSelector::Cpu => Self::Cpu,
            DeviceSelector::Auto => Self::Auto,
            other => Self::Gpu(other),
        }
    }
}

/// Most preferred API first. macOS ranks Metal above Vulkan (MoltenVK is a
/// translation layer on top of the same GPU); every other platform ranks
/// Vulkan first. [`api_rank`] is derived from this array, so the two cannot
/// disagree.
#[cfg(target_os = "macos")]
const API_RANK_ORDER: [wgpu::Backend; 4] = [
    wgpu::Backend::Metal,
    wgpu::Backend::Vulkan,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
];

#[cfg(not(target_os = "macos"))]
const API_RANK_ORDER: [wgpu::Backend; 4] = [
    wgpu::Backend::Vulkan,
    wgpu::Backend::Metal,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
];

/// The one API ordering: index in [`API_RANK_ORDER`], or 4 for anything not
/// listed there. Lower is preferred.
fn api_rank(backend: wgpu::Backend) -> u8 {
    API_RANK_ORDER
        .iter()
        .position(|b| *b == backend)
        .map_or(4, |i| i as u8)
}

fn instance_for(backends: wgpu::Backends) -> wgpu::Instance {
    if backends == wgpu::Backends::all() {
        return wgpu::Instance::default();
    }
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

async fn adapters_for(selector: &DeviceSelector) -> Vec<wgpu::Adapter> {
    match selector {
        DeviceSelector::Runtime { api, .. } => {
            let b = wgpu::Backends::from(*api);
            instance_for(b).enumerate_adapters(b).await
        }
        DeviceSelector::Index(_) | DeviceSelector::Name(_) => {
            instance_for(wgpu::Backends::all())
                .enumerate_adapters(wgpu::Backends::all())
                .await
        }
        DeviceSelector::Auto => {
            let mut all = Vec::new();
            for api in API_RANK_ORDER {
                let b = wgpu::Backends::from(api);
                let mut found = instance_for(b).enumerate_adapters(b).await;
                let discrete = found
                    .iter()
                    .any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
                all.append(&mut found);
                if discrete {
                    break;
                }
            }
            all
        }
        DeviceSelector::Cpu => Vec::new(),
    }
}

fn is_real_gpu(info: &wgpu::AdapterInfo) -> bool {
    matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::VirtualGpu
    )
}

/// `auto` found no GPU. The aligner then uses the host tower.
#[derive(Debug)]
pub(crate) struct NoGpuError;

impl std::fmt::Display for NoGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no wgpu gpu")
    }
}

impl std::error::Error for NoGpuError {}

/// Discrete, then integrated, then virtual. A CPU adapter is not a GPU.
fn device_class(kind: wgpu::DeviceType) -> u8 {
    match kind {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    }
}

fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    (device_class(info.device_type), api_rank(info.backend))
}

fn list_names(adapters: &[wgpu::Adapter]) -> String {
    adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            format!(
                "#{i} {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// One enumerated adapter, for `--list-devices`.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub driver_info: String,
    pub max_binding_bytes: u64,
    pub max_workgroup_storage: u32,
}

impl DeviceInfo {
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}) | {} {} | workgroup storage {} B, binding {} MiB",
            self.name,
            self.backend,
            self.device_type,
            self.driver,
            self.driver_info,
            self.max_workgroup_storage,
            self.max_binding_bytes / (1024 * 1024),
        )
    }
}

/// How `--list-devices` orders what it found, and the same key `Auto` ranks
/// adapters by: device class first (the old code sorted the *debug string* of
/// the class, which put `Cpu` ahead of `DiscreteGpu`), then API, then name.
fn target_sort_key(d: &DeviceInfo) -> (u8, u8, String) {
    (
        device_class(d.device_type),
        api_rank(d.backend),
        d.name.clone(),
    )
}

pub async fn list_targets() -> Vec<String> {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let mut infos: Vec<DeviceInfo> = adapters
        .iter()
        .map(|a| {
            let i = a.get_info();
            let l = a.limits();
            DeviceInfo {
                name: i.name,
                backend: i.backend,
                device_type: i.device_type,
                driver: i.driver,
                driver_info: i.driver_info,
                max_binding_bytes: l.max_storage_buffer_binding_size,
                max_workgroup_storage: l.max_compute_workgroup_storage_size,
            }
        })
        .collect();
    infos.sort_by_key(target_sort_key);
    infos.iter().map(DeviceInfo::describe).collect()
}

impl Gpu {
    pub async fn new_with(selector: DeviceSelector) -> Result<Self> {
        if selector == DeviceSelector::Cpu {
            bail!("DeviceSelector::Cpu is the host backend, not a wgpu device");
        }
        let adapters = adapters_for(&selector).await;
        if adapters.is_empty() {
            if selector == DeviceSelector::Auto {
                return Err(NoGpuError.into());
            }
            bail!("no wgpu adapters found (selector {selector:?})");
        }

        // `Auto` walks every real GPU, discrete before integrated. A named
        // adapter or a runtime index stays on that one device.
        let candidates: Vec<&wgpu::Adapter> = match &selector {
            DeviceSelector::Index(i) => vec![adapters.get(*i).ok_or_else(|| {
                anyhow::anyhow!(
                    "device #{i} does not exist ({} adapter(s) visible: {})",
                    adapters.len(),
                    list_names(&adapters)
                )
            })?],
            sel => {
                let mut hits: Vec<&wgpu::Adapter> = adapters
                    .iter()
                    .filter(|a| sel.matches(&a.get_info()))
                    .filter(|a| !matches!(sel, DeviceSelector::Auto) || is_real_gpu(&a.get_info()))
                    .collect();
                if hits.is_empty() && matches!(sel, DeviceSelector::Auto) {
                    return Err(NoGpuError.into());
                }
                if hits.is_empty() {
                    bail!("no adapter (visible: {})", list_names(&adapters));
                }
                hits.sort_by_key(|a| rank(&a.get_info()));
                match sel {
                    DeviceSelector::Runtime { index, .. } => {
                        vec![hits.get(*index).copied().ok_or_else(|| {
                            anyhow::anyhow!(
                                "that runtime has {} device(s), index {index} is out of range",
                                hits.len()
                            )
                        })?]
                    }
                    DeviceSelector::Auto => hits,
                    _ => vec![hits[0]],
                }
            }
        };

        let mut opened = None;
        let mut skipped: Option<anyhow::Error> = None;
        for adapter in &candidates {
            let info = adapter.get_info();
            let features = adapter.features();
            let limits = adapter.limits();
            let desc = wgpu::DeviceDescriptor {
                label: Some("ctc-forced-aligner-wgpu"),
                required_features: features
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::PIPELINE_CACHE
                        | wgpu::Features::SUBGROUP
                        | wgpu::Features::SHADER_F16
                        | wgpu::Features::SHADER_F64),
                required_limits: limits.clone(),
                ..Default::default()
            };
            // A copy-only queue lets the logits download run while the next
            // encoder occupies the compute queue. Failure here is not fatal:
            // the same adapter opens again without the extra queue.
            #[cfg_attr(
                not(any(target_os = "linux", target_os = "windows")),
                allow(unused_mut)
            )]
            let mut xfer_family = None;
            // On Vulkan-capable platforms try for a dedicated transfer queue;
            // elsewhere (macOS) no adapter can report `Backend::Vulkan`, so
            // this branch is dead code and must not be compiled at all.
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            let requested = if info.backend == wgpu::Backend::Vulkan {
                match open_vulkan_transfer(adapter, &desc) {
                    Ok((device, queue, family)) => {
                        xfer_family = family;
                        Ok((device, queue))
                    }
                    Err(e) => {
                        eprintln!("ctc-aligner: no transfer queue on {} ({e:#})", info.name);
                        adapter.request_device(&desc).await
                    }
                }
            } else {
                adapter.request_device(&desc).await
            };
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            let requested = adapter.request_device(&desc).await;
            match requested {
                Ok((device, queue)) => {
                    opened = Some((info, device, queue, xfer_family));
                    break;
                }
                Err(e) => {
                    if candidates.len() > 1 {
                        eprintln!(
                            "ctc-aligner: skip {} ({:?}, {:?}): {e:#}",
                            info.name, info.backend, info.device_type
                        );
                    }
                    skipped = Some(e.into());
                }
            }
        }
        let Some((info, device, queue, xfer_family)) = opened else {
            if matches!(selector, DeviceSelector::Auto) {
                return Err(NoGpuError.into());
            }
            return Err(skipped.context("request_device")?);
        };

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("[wgpu uncaptured error] {e:?}");
        }));

        // Negotiated, not advertised: `request_device` is free to hand back
        // fewer features or tighter limits than the adapter offered, and a
        // capability query that reads the adapter's set can then name a path
        // the device refuses.
        let features = device.features();
        let limits = device.limits();
        let cache_supported = features.contains(wgpu::Features::PIPELINE_CACHE);
        let (pipeline_cache, cache_path, seed) = match pipeline_cache_path(&info) {
            Some(path) if cache_supported => {
                let seed = std::fs::read(&path).ok();
                let cache = unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("pipeline_cache"),
                        data: seed.as_deref(),
                        fallback: true,
                    })
                };
                // the seed is what is already on disk: writing it back after
                // a load that compiled nothing new would be a no-op
                (Some(cache), Some(path), seed.unwrap_or_default())
            }
            _ => (None, None, Vec::new()),
        };

        Ok(Self {
            device,
            queue,
            info,
            features,
            limits,
            pipeline_cache,
            pipeline_cache_path: cache_path,
            pipeline_cache_bytes: std::sync::Mutex::new(seed),
            xfer_family,
        })
    }

    pub fn describe(&self) -> String {
        let sg = match self.subgroup_mode() {
            SubgroupMode::None => "sg none".to_string(),
            mode => format!(
                "sg {}..{} ({mode:?})",
                self.info.subgroup_min_size, self.info.subgroup_max_size
            ),
        };
        format!(
            "{} ({:?}, {:?}) | {} {} | wg {} B, {sg}, f16={}, f64={}",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info,
            self.limits.max_compute_workgroup_storage_size,
            self.has_f16(),
            self.has_f64(),
        )
    }

    /// The limits the device was opened with.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    pub fn has_f16(&self) -> bool {
        self.features.contains(wgpu::Features::SHADER_F16)
    }

    pub fn has_f64(&self) -> bool {
        self.features.contains(wgpu::Features::SHADER_F64)
    }

    pub fn has_subgroups(&self) -> bool {
        self.features.contains(wgpu::Features::SUBGROUP)
    }

    /// The largest storage buffer this device can both allocate and bind.
    ///
    /// WebGPU lets `max_buffer_size` and `max_storage_buffer_binding_size`
    /// differ: a buffer is created against the first and bound against the
    /// second, so anything this crate allocates and dispatches on has to fit
    /// the smaller of the two. Every size check reads this one number.
    pub fn effective_max_binding(&self) -> u64 {
        self.limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size)
    }

    /// The subgroup shape the reduction kernels may assume. The range comes
    /// from the adapter info: wgpu 30 exposes `subgroup_min_size` /
    /// `subgroup_max_size` there, not on [`wgpu::Limits`].
    ///
    /// `CTC_SUBGROUPS=0|off|false` forces [`SubgroupMode::None`], so the
    /// shared-memory tree can be benchmarked against the subgroup path.
    pub fn subgroup_mode(&self) -> SubgroupMode {
        if env_flag_off(std::env::var("CTC_SUBGROUPS").ok().as_deref()) {
            return SubgroupMode::None;
        }
        SubgroupMode::from_limits(
            self.has_subgroups(),
            self.info.subgroup_min_size,
            self.info.subgroup_max_size,
        )
    }

    /// Whether a kernel may carry `enable f16;` and `f16` shared memory.
    ///
    /// `CTC_F16_KERNEL=1|true|on` opts in; `auto`, unset or empty leave it
    /// off, because the weights are native `f32` and the `f16` tiles have not
    /// been measured 0-diff on real hardware. It never returns `true` on a
    /// device without [`wgpu::Features::SHADER_F16`], because the shader would
    /// not compile.
    pub fn use_f16_kernel(&self) -> bool {
        resolve_f16_kernel(
            self.has_f16(),
            std::env::var("CTC_F16_KERNEL").ok().as_deref(),
        )
    }

    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn upload(&self, buf: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buf, 0, data);
    }

    pub fn flush(&self) -> Result<()> {
        self.pump()?;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for flush")?;
        Ok(())
    }

    pub fn pump(&self) -> Result<()> {
        let enc = self.device.create_command_encoder(&Default::default());
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    pub fn readback(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::with_capacity(bytes as usize);
        self.readback_map(buf, 0, bytes, &mut |piece| out.extend_from_slice(piece))?;
        Ok(out)
    }

    /// Copy `bytes` of `buf` at `offset` out in 16 MiB staging pieces, handing
    /// each mapped piece to `sink` in order.
    ///
    /// The staging stays chunked — one big MAP_READ staging can fail to map
    /// after a long dispatch sequence on some drivers; small maps keep
    /// working — but every copy is submitted before the first map, so the
    /// pieces no longer serialize behind one poll round trip each.
    fn readback_map(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        bytes: u64,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<()> {
        const CHUNK: u64 = 16 << 20;
        let mut pieces: Vec<(wgpu::Buffer, u64)> = Vec::new();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            let mut off: u64 = offset;
            while off < offset + bytes {
                let take = CHUNK.min(offset + bytes - off);
                let size = (take + 3) & !3;
                let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_buffer_to_buffer(buf, off, &staging, 0, take);
                pieces.push((staging, take));
                off += take;
            }
            self.queue.submit([enc.finish()]);
        }
        // Arm every map first: the copies are already on the queue, so one
        // wait retires them all and resolves every callback. Polling per
        // piece turned a 70 MB readback into five round trips.
        let mut waits = Vec::with_capacity(pieces.len());
        for (staging, _) in &pieces {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            waits.push(rx);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for readback")?;
        for ((staging, take), rx) in pieces.iter().zip(waits) {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    eprintln!("[map debug] inner error = {e:?}");
                    return Err(anyhow::Error::from(e).context("map buffer"));
                }
                Err(e) => anyhow::bail!("map callback dropped: {e}"),
            }
            let slice = staging.slice(..);
            let mapped = slice.get_mapped_range()?;
            sink(&mapped[..*take as usize]);
            drop(mapped);
            staging.unmap();
        }
        Ok(())
    }

    /// Compile a WGSL module + compute pipeline, surfacing validation errors.
    pub fn pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline> {
        self.pipeline_with(label, wgsl, entry, layout, true)
    }

    /// Same as [`pipeline`](Self::pipeline), but skip the WebGPU-required
    /// zeroing of workgroup memory. GEMM overwrites every shared slot it
    /// reads, and the zeroing pass is a serial prologue on this backend.
    pub fn pipeline_no_zero(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline> {
        self.pipeline_with(label, wgsl, entry, layout, false)
    }

    fn pipeline_with(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
        zero_init: bool,
    ) -> Result<wgpu::ComputePipeline> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipe = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: zero_init,
                },
                cache: self.pipeline_cache.as_ref(),
            });
        let err = pollster::block_on(guard.pop());
        if let Some(e) = err {
            bail!("pipeline {label} failed validation: {e}");
        }
        Ok(pipe)
    }

    /// Persist the pipeline cache, atomically and only when it changed.
    ///
    /// The model calls this after its 11 pipelines exist and the DP after its
    /// 4, so the next cold start compiles nothing. Failure is not the caller's
    /// problem: `let _ =` keeps an unwritable cache directory from failing a
    /// model load.
    pub fn save_pipeline_cache(&self) -> Result<()> {
        let (Some(cache), Some(path)) = (&self.pipeline_cache, &self.pipeline_cache_path) else {
            return Ok(());
        };
        let Some(data) = cache.get_data() else {
            return Ok(());
        };
        let mut last = self
            .pipeline_cache_bytes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        persist_cache_bytes(path, &data, &mut last)
            .with_context(|| format!("save pipeline cache {}", path.display()))?;
        Ok(())
    }
}

/// A copy-only queue family. Family 0 is the graphics queue wgpu always
/// opens. A compute family also sets TRANSFER and is not the DMA engine.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn transfer_family(hal: &wgpu::hal::vulkan::Adapter) -> Result<u32> {
    let props = unsafe {
        hal.shared_instance()
            .raw_instance()
            .get_physical_device_queue_family_properties(hal.raw_physical_device())
    };
    if let Some((index, _)) = props.iter().enumerate().find(|(i, fam)| {
        *i != 0
            && fam.queue_count > 0
            && fam.queue_flags.contains(ash::vk::QueueFlags::TRANSFER)
            && !fam.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
            && !fam.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
    }) {
        return Ok(index as u32);
    }
    let listed: Vec<String> = props
        .iter()
        .enumerate()
        .map(|(i, fam)| format!("{i}: {:?} x{}", fam.queue_flags, fam.queue_count))
        .collect();
    anyhow::bail!("no dedicated transfer queue ({})", listed.join(", "))
}

/// Open `adapter` with wgpu's graphics queue plus one transfer queue.
/// The caller falls back to [`wgpu::Adapter::request_device`] on `Err`,
/// so this must not leave a device behind when it fails.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn open_vulkan_transfer(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue, Option<u32>)> {
    let family = {
        let hal =
            unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }.context("vulkan hal adapter")?;
        transfer_family(&hal)?
    };
    // The hal guard borrows `adapter`. Drop it before `create_device_from_hal`.
    let hal_open = {
        let hal =
            unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }.context("vulkan hal adapter")?;
        unsafe {
            hal.open_with_callback(
                desc.required_features,
                &desc.required_limits,
                &desc.memory_hints,
                Some(Box::new(move |args| {
                    args.queue_create_infos.push(
                        ash::vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(family)
                            .queue_priorities(std::slice::from_ref(&XFER_Q_PRIO)),
                    );
                })),
            )
        }
        .map_err(|e| anyhow::anyhow!("vulkan device with transfer queue: {e:?}"))?
    };
    let (device, queue) =
        unsafe { adapter.create_device_from_hal::<wgpu::hal::api::Vulkan>(hal_open, desc) }
            .map_err(|e| anyhow::anyhow!("wrap vulkan device: {e:?}"))?;
    eprintln!("ctc-aligner: transfer queue family {family}");
    Ok((device, queue, Some(family)))
}

/// Replace every character a Windows filename cannot hold, and whitespace,
/// with `_`: the cache key ends up as a file name.
fn sanitize_cache_component(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_whitespace() => '_',
            c => c,
        })
        .collect()
}

fn pipeline_cache_path(info: &wgpu::AdapterInfo) -> Option<std::path::PathBuf> {
    let root = std::env::var_os("CTC_WGPU_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from))
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .or_else(|| Some(std::env::temp_dir()))?;
    let key = format!(
        "{:04x}-{:04x}-{:?}-{}",
        info.vendor,
        info.device,
        info.backend,
        sanitize_cache_component(&info.driver_info)
    );
    Some(
        root.join("ctc-forced-aligner-wgpu")
            .join(format!("{key}.pipeline_cache")),
    )
}

/// `true` for the values that switch an optional path off (`0`, `off`,
/// `false`, any case). Anything else, including unset, leaves it on.
fn env_flag_off(v: Option<&str>) -> bool {
    let Some(v) = v else { return false };
    let v = v.trim();
    v.eq_ignore_ascii_case("0") || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
}

/// Whether the f16 kernels may run, from the device and the `CTC_F16_KERNEL`
/// value (`None` when unset).
///
/// The checkpoint is native `f32` and the `f16` shared-tile GEMM has not been
/// measured 0-diff against it on real `f16` hardware, so the default is
/// precision first: `auto`, unset, empty and anything unrecognised answer
/// `false`. Only `1|true|on` opts in.
///
/// `has_f16` is a hard gate on that opt-in: no environment value can make an
/// `enable f16;` module compile on a device that never advertised
/// [`wgpu::Features::SHADER_F16`].
pub fn resolve_f16_kernel(has_f16: bool, env_val: Option<&str>) -> bool {
    if !has_f16 {
        return false;
    }
    matches!(
        env_val
            .map(str::trim)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on"
    )
}

/// Write `data` to `path` unless it is empty or byte-identical to `last`;
/// `last` is updated only after the rename lands. Returns whether the file
/// changed.
fn persist_cache_bytes(
    path: &std::path::Path,
    data: &[u8],
    last: &mut Vec<u8>,
) -> std::io::Result<bool> {
    if data.is_empty() || *last == data {
        return Ok(false);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // A unique sibling: the pid keeps two processes warming the same cache
    // from writing the same temporary file, and the rename that follows is
    // last-writer-wins.
    let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(format!(".{}.tmp", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    if let Err(e) = std::fs::write(&tmp, data) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    last.clear();
    last.extend_from_slice(data);
    Ok(true)
}

/// A top-level WGSL directive: `enable ...;`, `requires ...;` or
/// `diagnostic(...);`. WGSL requires these before any other declaration.
fn directive_line(line: &str) -> Option<&str> {
    let t = line.trim();
    let is = t.starts_with("enable ")
        || t.starts_with("requires ")
        || (t.starts_with("diagnostic(") && t.ends_with(';'));
    is.then_some(t)
}

/// `true` for `enable subgroups;`, which Naga accepts the feature for but
/// rejects as a directive: subgroup built-ins never need an `enable`.
fn enables_subgroups(directive: &str) -> bool {
    let mut words = directive.trim_end_matches(';').split_whitespace();
    matches!(words.next(), Some("enable")) && matches!(words.next(), Some("subgroups"))
}

/// Collect one directive line, unless it is `enable subgroups;` or a repeat.
/// Returns whether `line` was a directive at all.
fn take_directive<'a>(directives: &mut Vec<&'a str>, line: &'a str) -> bool {
    let Some(d) = directive_line(line) else {
        return false;
    };
    if !enables_subgroups(d) && !directives.contains(&d) {
        directives.push(d);
    }
    true
}

/// Put a shader together with its directives at the top, where WGSL wants
/// them, whatever order the pieces arrived in.
///
/// A prelude may carry `enable f16;` (or `requires` / `diagnostic`) in front
/// of the declarations it needs; a body may carry its own leading directives.
/// Both are collected, de-duplicated in first-seen order, and emitted before
/// the rest of the prelude and the rest of the body. `enable subgroups;` is
/// dropped on the way: Naga rejects it outright.
pub fn assemble_wgsl(prelude: &str, body: &str) -> String {
    let mut directives: Vec<&str> = Vec::new();
    let mut prelude_rest = String::new();
    for line in prelude.lines() {
        if !take_directive(&mut directives, line) {
            prelude_rest.push_str(line);
            prelude_rest.push('\n');
        }
    }
    // The body's own leading directives, ignoring blank lines between them.
    let mut start = 0usize;
    let mut blanks = 0usize;
    for line in body.split_inclusive('\n') {
        if line.trim().is_empty() {
            blanks += line.len();
            continue;
        }
        if !take_directive(&mut directives, line) {
            break;
        }
        start += blanks + line.len();
        blanks = 0;
    }
    if directives.is_empty() && prelude.is_empty() && start == 0 {
        return body.to_string();
    }
    let mut out = String::with_capacity(prelude.len() + body.len() + 32);
    for d in &directives {
        out.push_str(d);
        out.push('\n');
    }
    out.push_str(&prelude_rest);
    if !prelude_rest.is_empty() && !prelude_rest.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&body[start..]);
    out
}

/// Load-time upload session with bounded outstanding staging (256 MiB), so a
/// multi-GiB model load never overflows VRAM on WDDM.
const STAGING_BUDGET: u64 = 256 << 20;

pub(crate) struct BulkUpload<'a> {
    gpu: &'a Gpu,
    pending: u64,
}

impl Gpu {
    pub fn uploader(&self) -> BulkUpload<'_> {
        BulkUpload {
            gpu: self,
            pending: 0,
        }
    }
}

impl<'a> BulkUpload<'a> {
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.gpu.storage(label, bytes)
    }

    pub fn upload(&mut self, buf: &wgpu::Buffer, data: &[u8]) -> Result<()> {
        self.upload_at(buf, 0, data)
    }

    pub fn upload_at(&mut self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) -> Result<()> {
        anyhow::ensure!(
            offset.is_multiple_of(4),
            "upload offset {offset} is not 4-byte aligned"
        );
        let budget = (STAGING_BUDGET as usize).max(1);
        let mut off = offset;
        for piece in data.chunks(budget) {
            if self.pending + piece.len() as u64 > STAGING_BUDGET {
                self.pump()?;
            }
            self.gpu.queue.write_buffer(buf, off, piece);
            self.pending += piece.len() as u64;
            off += piece.len() as u64;
        }
        Ok(())
    }

    fn pump(&mut self) -> Result<()> {
        self.gpu.pump()?;
        self.pending = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if self.pending > 0 {
            self.gpu.flush()?;
        }
        self.pending = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        api_rank, assemble_wgsl, device_class, persist_cache_bytes, resolve_f16_kernel,
        target_sort_key, Backend, DeviceInfo, DeviceSelector, API_RANK_ORDER,
    };

    #[test]
    fn auto_orders_discrete_before_integrated() {
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, VirtualGpu};
        assert!(device_class(DiscreteGpu) < device_class(IntegratedGpu));
        assert!(device_class(IntegratedGpu) < device_class(VirtualGpu));
        assert!(device_class(VirtualGpu) < device_class(Cpu));
    }

    #[test]
    fn one_device_only() {
        assert!(DeviceSelector::parse("dual").is_err());
        assert!(DeviceSelector::parse("gpu").is_err());
        assert!(matches!(
            DeviceSelector::parse("auto").unwrap(),
            DeviceSelector::Auto
        ));
        assert!(matches!(
            DeviceSelector::parse("cpu").unwrap(),
            DeviceSelector::Cpu
        ));
        assert!(matches!(
            DeviceSelector::parse("vulkan:0").unwrap(),
            DeviceSelector::Runtime { index: 0, .. }
        ));
    }

    #[test]
    fn backend_policy_is_auto_cpu_or_required_gpu() {
        assert_eq!(Backend::parse("").unwrap(), Backend::Auto);
        assert_eq!(Backend::parse("auto").unwrap(), Backend::Auto);
        assert_eq!(Backend::parse("cpu").unwrap(), Backend::Cpu);
        assert_eq!(
            Backend::parse("GPU").unwrap(),
            Backend::Gpu(DeviceSelector::Auto)
        );
        assert!(matches!(
            Backend::parse("vulkan:1").unwrap(),
            Backend::Gpu(DeviceSelector::Runtime { index: 1, .. })
        ));
        assert_eq!(Backend::from_selector(DeviceSelector::Auto), Backend::Auto);
        assert_eq!(Backend::from_selector(DeviceSelector::Cpu), Backend::Cpu);
        assert_eq!(
            Backend::from_selector(DeviceSelector::Name("p104".into())),
            Backend::Gpu(DeviceSelector::Name("p104".into()))
        );
    }

    #[test]
    fn api_rank_is_the_order_table() {
        use wgpu::Backend as B;
        // the table is the ranking: listed backends rank 0..3 in its order
        for (i, b) in API_RANK_ORDER.iter().enumerate() {
            assert_eq!(api_rank(*b), i as u8, "{b:?}");
        }
        // anything unlisted ranks last
        assert_eq!(api_rank(B::BrowserWebGpu), 4);
        assert_eq!(api_rank(B::Noop), 4);
        // the platform's native API is preferred over its translation layer
        #[cfg(target_os = "macos")]
        {
            assert_eq!(API_RANK_ORDER[0], B::Metal);
            assert!(api_rank(B::Metal) < api_rank(B::Vulkan));
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(API_RANK_ORDER[0], B::Vulkan);
            assert!(api_rank(B::Vulkan) < api_rank(B::Metal));
        }
    }

    fn dev(name: &str, backend: wgpu::Backend, device_type: wgpu::DeviceType) -> DeviceInfo {
        DeviceInfo {
            name: name.to_string(),
            backend,
            device_type,
            driver: String::new(),
            driver_info: String::new(),
            max_binding_bytes: 0,
            max_workgroup_storage: 0,
        }
    }

    #[test]
    fn list_targets_sorts_class_then_api_then_name() {
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, VirtualGpu};
        let mut found = vec![
            dev("llvmpipe", wgpu::Backend::Vulkan, Cpu),
            dev("igpu", wgpu::Backend::Vulkan, IntegratedGpu),
            dev("vgpu", wgpu::Backend::Vulkan, VirtualGpu),
            dev("card-b", wgpu::Backend::Vulkan, DiscreteGpu),
            dev("card-a", wgpu::Backend::Vulkan, DiscreteGpu),
            dev("vivante", wgpu::Backend::Gl, DiscreteGpu),
        ];
        found.sort_by_key(target_sort_key);
        let names: Vec<&str> = found.iter().map(|d| d.name.as_str()).collect();
        // the old key sorted the debug string of the class, which put the CPU
        // adapter first; class ordering is numeric now
        assert_eq!(
            names,
            vec!["card-a", "card-b", "vivante", "igpu", "vgpu", "llvmpipe"]
        );
        assert!(target_sort_key(&found[0]).0 < target_sort_key(&found[5]).0);
        // API breaks ties inside one class, name breaks the rest
        let mut gpus = vec![
            dev("b", wgpu::Backend::Dx12, DiscreteGpu),
            dev("a", wgpu::Backend::Dx12, DiscreteGpu),
        ];
        gpus.sort_by_key(target_sort_key);
        assert_eq!(gpus[0].name, "a");
    }

    #[test]
    fn assemble_wgsl_puts_directives_first() {
        let body = "\nstruct Dims { m: u32 }\n\n@compute @workgroup_size(1) fn main() {}\n";
        let out = assemble_wgsl("enable f16;\nvar<workgroup> s: array<f32, 4u>;\n", body);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "enable f16;");
        assert_eq!(lines[1], "var<workgroup> s: array<f32, 4u>;");
        assert!(out.contains("struct Dims"));
        assert!(out.contains("fn main()"));

        // directives from both halves are collected, de-duplicated, and none
        // of them survives where the body had it
        let out = assemble_wgsl(
            "enable f16;\nenable f16;\n",
            "enable f16;\nstruct A { x: u32 }\n",
        );
        assert_eq!(out.matches("enable f16;").count(), 1, "{out}");
        assert!(out.starts_with("enable f16;\n"), "{out}");
        assert!(out.contains("struct A"));

        // `enable subgroups;` is never emitted
        let out = assemble_wgsl(
            "enable subgroups;\nstruct P { x: u32 }\n",
            "struct Cfg { n: u32 }\n",
        );
        assert!(!out.contains("subgroups"), "{out}");
        assert!(out.starts_with("struct P"), "{out}");
    }

    #[test]
    fn assemble_wgsl_leaves_a_plain_body_alone() {
        let body = "\nstruct Cfg { n: u32 }\n";
        assert_eq!(assemble_wgsl("", body), body);
        // with no directives anywhere, a prelude is simply prepended
        let out = assemble_wgsl("struct P { x: u32 }\n", "struct Cfg { n: u32 }\n");
        assert!(out.starts_with("struct P"), "{out}");
        assert!(out.contains("struct Cfg"), "{out}");
    }

    #[test]
    fn f16_kernel_follows_the_device_and_the_env() {
        // Precision first: `auto`, unset, empty and anything unknown stay on
        // the native f32 path, because the f16 tiles are unverified against
        // the f32 checkpoint.
        for v in [None, Some(""), Some(" "), Some("auto"), Some("Auto")] {
            assert!(!resolve_f16_kernel(true, v), "{v:?}");
        }
        for v in ["1", "true", "on", "TRUE", " On ", " True "] {
            assert!(resolve_f16_kernel(true, Some(v)), "{v}");
        }
        for v in ["0", "false", "off", "FALSE", "Off", " 0 ", "2", "yes"] {
            assert!(!resolve_f16_kernel(true, Some(v)), "{v}");
        }
        assert!(!resolve_f16_kernel(false, None));
        // no environment value can compile an f16 module on a device that
        // never advertised the feature
        for v in [None, Some("1"), Some("true"), Some("on"), Some("auto")] {
            assert!(!resolve_f16_kernel(false, v), "{v:?}");
        }
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ctc-aligner-test-{}-{nanos}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn pipeline_cache_writes_atomically_and_only_when_it_changed() {
        let dir = scratch_dir("cache");
        let path = dir.join("nested").join("dev.pipeline_cache");
        let mut last = Vec::new();

        // nothing to save: no file, and no directory either
        assert!(!persist_cache_bytes(&path, &[], &mut last).unwrap());
        assert!(!dir.join("nested").exists());
        assert!(last.is_empty());

        // the first real save creates the parents and lands the bytes
        assert!(persist_cache_bytes(&path, b"one", &mut last).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), b"one");
        assert_eq!(last, b"one");

        // identical bytes: skipped, so the file keeps its timestamp
        let before = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!persist_cache_bytes(&path, b"one", &mut last).unwrap());
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            before
        );

        // a change replaces the file in place, leaving no temp sibling
        assert!(persist_cache_bytes(&path, b"two", &mut last).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), b"two");
        assert_eq!(last, b"two");
        let left: Vec<_> = std::fs::read_dir(dir.join("nested"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left.len(), 1, "{left:?}");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
