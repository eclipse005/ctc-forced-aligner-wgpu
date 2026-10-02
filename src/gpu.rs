//! Device / queue wrapper and buffer plumbing (adapted from the
//! qwen-aligner-wgpu reference implementation).
//!
//! Storage buffers are always allocated 16-byte padded; activations here are
//! f32 (the model checkpoint is f32; a 16-bit storage mode can be added later
//! the same way the reference does).

use anyhow::{bail, Context, Result};

/// A wgpu device plus the queue, adapter info and negotiated limits.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
    pub pipeline_cache: Option<wgpu::PipelineCache>,
    pub pipeline_cache_path: Option<std::path::PathBuf>,
}

/// One device: `auto`, `cpu`, `vulkan[:i]`, `dx12[:i]`, `#n`, or a name substring.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    #[default]
    Auto,
    Runtime { api: wgpu::Backend, index: usize },
    Cpu,
    Index(usize),
    Name(String),
}

pub const RUNTIMES: &[(&str, wgpu::Backend)] = &[
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

const API_RANK_ORDER: [wgpu::Backend; 4] = [
    wgpu::Backend::Vulkan,
    wgpu::Backend::Metal,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
];

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
        DeviceSelector::Index(_) | DeviceSelector::Name(_) => instance_for(wgpu::Backends::all())
            .enumerate_adapters(wgpu::Backends::all())
            .await,
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
pub struct NoGpuError;

impl std::fmt::Display for NoGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no wgpu gpu")
    }
}

impl std::error::Error for NoGpuError {}

fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let class = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    };
    let api = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    };
    (class, api)
}

fn list_names(adapters: &[wgpu::Adapter]) -> String {
    adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            format!("#{i} {} ({:?}, {:?})", info.name, info.backend, info.device_type)
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
    infos.sort_by_key(|d| (format!("{:?}", d.device_type), d.name.clone()));
    infos.iter().map(DeviceInfo::describe).collect()
}

impl Gpu {
    pub async fn new(prefer: Option<&str>) -> Result<Self> {
        let sel = match prefer {
            Some(p) => DeviceSelector::parse(p)?,
            None => DeviceSelector::Auto,
        };
        Self::new_with(sel).await
    }

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

        let adapter = match &selector {
            DeviceSelector::Index(i) => adapters.get(*i).ok_or_else(|| {
                anyhow::anyhow!(
                    "device #{i} does not exist ({} adapter(s) visible: {})",
                    adapters.len(),
                    list_names(&adapters)
                )
            })?,
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
                    DeviceSelector::Runtime { index, .. } => hits.get(*index).copied().ok_or_else(
                        || {
                            anyhow::anyhow!(
                                "that runtime has {} device(s), index {index} is out of range",
                                hits.len()
                            )
                        },
                    )?,
                    _ => hits[0],
                }
            }
        };

        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ctc-forced-aligner-wgpu"),
                required_features: features
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::PIPELINE_CACHE
                        | wgpu::Features::SUBGROUP),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .context("request_device")?;

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("[wgpu uncaptured error] {e:?}");
        }));

        let cache_supported = features.contains(wgpu::Features::PIPELINE_CACHE);
        let (pipeline_cache, pipeline_cache_path) = match pipeline_cache_path(&info) {
            Some(path) if cache_supported => {
                let seed = std::fs::read(&path).ok();
                let cache = unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("pipeline_cache"),
                        data: seed.as_deref(),
                        fallback: true,
                    })
                };
                (Some(cache), Some(path))
            }
            _ => (None, None),
        };

        Ok(Self {
            device,
            queue,
            info,
            limits,
            features,
            pipeline_cache,
            pipeline_cache_path,
        })
    }

    pub fn save_pipeline_cache(&self) -> Result<()> {
        let (Some(cache), Some(path)) = (&self.pipeline_cache, &self.pipeline_cache_path) else {
            return Ok(());
        };
        let Some(data) = cache.get_data() else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, &data)?;
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}) | {} {}",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info,
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
        // chunked: one big MAP_READ staging can fail to map after a long
        // dispatch sequence on some drivers; small maps keep working
        const CHUNK: u64 = 4 << 20;
        let mut out: Vec<u8> = Vec::with_capacity(bytes as usize);
        let mut off: u64 = 0;
        while off < bytes {
            let take = CHUNK.min(bytes - off);
            let size = (take + 3) & !3;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            {
                let mut enc = self.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(buf, off, &staging, 0, take);
                self.queue.submit([enc.finish()]);
            }
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("poll for readback")?;
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    eprintln!("[map debug] inner error = {e:?}");
                    return Err(anyhow::Error::from(e).context("map buffer"));
                }
                Err(e) => anyhow::bail!("map callback dropped: {e}"),
            }
            let mut part = slice.get_mapped_range()?.to_vec();
            staging.unmap();
            part.truncate(take as usize);
            out.extend_from_slice(&part);
            off += take;
        }
        Ok(out)
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
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
        info.driver_info.replace(['\\', '/', ':', ' '], "_")
    );
    Some(root.join("ctc-forced-aligner-wgpu").join(format!("{key}.pipeline_cache")))
}

/// Load-time upload session with bounded outstanding staging (256 MiB), so a
/// multi-GiB model load never overflows VRAM on WDDM.
const STAGING_BUDGET: u64 = 256 << 20;

pub struct BulkUpload<'a> {
    gpu: &'a Gpu,
    pending: u64,
}

impl Gpu {
    pub fn uploader(&self) -> BulkUpload<'_> {
        BulkUpload { gpu: self, pending: 0 }
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
        anyhow::ensure!(offset % 4 == 0, "upload offset {offset} is not 4-byte aligned");
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
    use super::DeviceSelector;

    #[test]
    fn one_device_only() {
        assert!(DeviceSelector::parse("dual").is_err());
        assert!(matches!(DeviceSelector::parse("auto").unwrap(), DeviceSelector::Auto));
        assert!(matches!(DeviceSelector::parse("cpu").unwrap(), DeviceSelector::Cpu));
        assert!(matches!(
            DeviceSelector::parse("vulkan:0").unwrap(),
            DeviceSelector::Runtime { index: 0, .. }
        ));
    }
}
