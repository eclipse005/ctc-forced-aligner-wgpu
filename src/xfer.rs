//! Device-to-host copy of a large emission block on a dedicated Vulkan
//! transfer queue.
//!
//! A 66 MB copy takes ~80 ms on this P104 (PCIe gen1 x4, about 1 GB/s). The
//! compute queue copies the block into a device-local bounce (~1 ms) and
//! signals a binary semaphore; the transfer queue DMAs that bounce into cached
//! host memory. On this card the copy does not overlap the next encoder.
//! `CTC_READBACK=cached` keeps `MAP_READ`. Readback under [`MIN_BYTES`] (the
//! hidden encoder stream, about 7 MB) never comes here.

use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};

use crate::gpu::Gpu;

/// Below this, cached `MAP_READ` is already a few milliseconds. The hidden
/// readback is ~7 MB and stays on that path.
pub(crate) const MIN_BYTES: u64 = 16 << 20;

const GRAPHICS_WAIT: ash::vk::PipelineStageFlags = ash::vk::PipelineStageFlags::TRANSFER;

/// Command pool plus the memory heaps the bounce and the host buffer need.
/// Dropped after every [`XferSet`]: the sets wait their fences and free
/// buffers first, then this destroys the pool (which frees the command buffers).
pub(crate) struct Engine {
    vk: ash::Device,
    queue: ash::vk::Queue,
    family: u32,
    graphics_family: u32,
    pool: ash::vk::CommandPool,
    mem: ash::vk::PhysicalDeviceMemoryProperties,
    alloc_failed: AtomicBool,
}

/// One window of logits. Two of these are in flight: one DMAing, one being filled.
pub(crate) struct XferSet {
    pieces: Vec<Piece>,
    sem: ash::vk::Semaphore,
    fence: ash::vk::Fence,
    cmd: ash::vk::CommandBuffer,
    queue: ash::vk::Queue,
    vk: ash::Device,
    in_flight: Cell<bool>,
    /// The bounce submit signaled `sem` and the transfer submit did not wait
    /// for it. Drop has to idle the device before destroying the semaphore.
    broken: Cell<bool>,
}

struct Piece {
    bounce: wgpu::Buffer,
    bounce_raw: ash::vk::Buffer,
    host: HostMem,
    /// Byte capacity of the bounce. Copies may be shorter.
    size: u64,
}

struct HostMem {
    vk: ash::Device,
    buffer: ash::vk::Buffer,
    memory: ash::vk::DeviceMemory,
    ptr: NonNull<u8>,
}

// The pointer addresses memory this value unmaps on drop, and it is read only
// after the transfer fence on the encode thread.
unsafe impl Send for HostMem {}

impl Drop for HostMem {
    fn drop(&mut self) {
        unsafe {
            self.vk.unmap_memory(self.memory);
            self.vk.destroy_buffer(self.buffer, None);
            self.vk.free_memory(self.memory, None);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { self.vk.destroy_command_pool(self.pool, None) };
    }
}

impl Drop for XferSet {
    fn drop(&mut self) {
        if self.in_flight.get() {
            let _ = unsafe { self.vk.wait_for_fences(&[self.fence], true, u64::MAX) };
            self.in_flight.set(false);
        }
        if self.broken.get() {
            let _ = unsafe { self.vk.device_wait_idle() };
        }
        unsafe {
            self.vk.destroy_semaphore(self.sem, None);
            self.vk.destroy_fence(self.fence, None);
        }
    }
}

impl Engine {
    pub(crate) fn open(gpu: &Gpu) -> Option<Self> {
        let family = gpu.xfer_family?;
        let hal = unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() };
        let Some(hal) = hal else {
            eprintln!("ctc-aligner: transfer queue present but the vulkan device is not exposed");
            return None;
        };
        let graphics_family = hal.queue_family_index();
        if graphics_family == family {
            eprintln!(
                "ctc-aligner: transfer family {family} is the compute queue; keeping cached readback"
            );
            return None;
        }
        let vk = hal.raw_device().clone();
        let mem = unsafe {
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_memory_properties(hal.raw_physical_device())
        };
        let queue = unsafe { vk.get_device_queue(family, 0) };
        let pool = match unsafe {
            vk.create_command_pool(
                &ash::vk::CommandPoolCreateInfo::default()
                    .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(family),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(e) => {
                eprintln!("ctc-aligner: transfer command pool: {e:?}");
                return None;
            }
        };
        Some(Self {
            vk,
            queue,
            family,
            graphics_family,
            pool,
            mem,
            alloc_failed: AtomicBool::new(false),
        })
    }

    pub(crate) fn alloc_set(&self, device: &wgpu::Device, takes: &[u64]) -> Option<XferSet> {
        if self.alloc_failed.load(Ordering::Relaxed) || takes.is_empty() {
            return None;
        }
        match self.try_alloc(device, takes) {
            Ok(set) => Some(set),
            Err(e) => {
                self.alloc_failed.store(true, Ordering::Relaxed);
                eprintln!("ctc-aligner: transfer readback unavailable: {e:#}");
                None
            }
        }
    }

    fn try_alloc(&self, device: &wgpu::Device, takes: &[u64]) -> Result<XferSet> {
        let cmd = {
            let info = ash::vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(ash::vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { self.vk.allocate_command_buffers(&info) }
                .map_err(|e| anyhow::anyhow!("transfer command buffer: {e:?}"))?
                .pop()
                .context("transfer command buffer")?
        };
        let sem = match unsafe { self.vk.create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None) } {
            Ok(sem) => sem,
            Err(e) => bail!("transfer semaphore: {e:?}"),
        };
        let fence = match unsafe { self.vk.create_fence(&ash::vk::FenceCreateInfo::default(), None) } {
            Ok(fence) => fence,
            Err(e) => {
                unsafe { self.vk.destroy_semaphore(sem, None) };
                bail!("transfer fence: {e:?}");
            }
        };
        let mut pieces = Vec::with_capacity(takes.len());
        for &take in takes {
            match self.alloc_piece(device, take) {
                Ok(piece) => pieces.push(piece),
                Err(e) => {
                    drop(pieces);
                    unsafe {
                        self.vk.destroy_semaphore(sem, None);
                        self.vk.destroy_fence(fence, None);
                    }
                    return Err(e);
                }
            }
        }
        Ok(XferSet {
            pieces,
            sem,
            fence,
            cmd,
            queue: self.queue,
            vk: self.vk.clone(),
            in_flight: Cell::new(false),
            broken: Cell::new(false),
        })
    }

    fn alloc_piece(&self, device: &wgpu::Device, take: u64) -> Result<Piece> {
        let size = take.max(4).next_multiple_of(256);
        let families = [self.graphics_family, self.family];
        let bounce_info = ash::vk::BufferCreateInfo::default()
            .size(size)
            .usage(ash::vk::BufferUsageFlags::TRANSFER_SRC | ash::vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(ash::vk::SharingMode::CONCURRENT)
            .queue_family_indices(&families);
        // Device-local and not host-visible: the copy engine DMAs real VRAM.
        // The host-visible device heap on this card is a 214 MB BAR, and reading
        // it from the CPU is ~50 MB/s.
        let (bounce_raw, bounce_mem, bounce_alloc, bounce_ty) =
            alloc_bound(&self.vk, &bounce_info, |bits| {
                pick_type(&self.mem, bits, device_local).or_else(|| {
                    pick_type(&self.mem, bits, |flags| {
                        flags.contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    })
                })
            })?;
        let hal_buffer = unsafe {
            wgpu::hal::vulkan::Buffer::from_raw_managed(bounce_raw, bounce_mem, 0, bounce_alloc)
        };
        let bounce = unsafe {
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                hal_buffer,
                &wgpu::BufferDescriptor {
                    label: Some("logits-bounce"),
                    size,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        };

        let host_info = ash::vk::BufferCreateInfo::default()
            .size(size)
            .usage(ash::vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);
        let (host_raw, host_mem, host_alloc, host_ty) =
            alloc_bound(&self.vk, &host_info, |bits| pick_type(&self.mem, bits, host_cached))?;
        let mapped = unsafe {
            self.vk
                .map_memory(host_mem, 0, host_alloc, ash::vk::MemoryMapFlags::empty())
        };
        let ptr = match mapped.and_then(|p| {
            NonNull::new(p.cast::<u8>()).ok_or(ash::vk::Result::ERROR_MEMORY_MAP_FAILED)
        }) {
            Ok(ptr) => ptr,
            Err(e) => {
                unsafe {
                    self.vk.destroy_buffer(host_raw, None);
                    self.vk.free_memory(host_mem, None);
                }
                bail!("map transfer staging: {e:?}");
            }
        };
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "ctc-aligner: emission readback on transfer queue (bounce type {bounce_ty}, host type {host_ty})"
            );
        }
        Ok(Piece {
            bounce,
            bounce_raw,
            host: HostMem {
                vk: self.vk.clone(),
                buffer: host_raw,
                memory: host_mem,
                ptr,
            },
            size,
        })
    }
}

impl XferSet {
    pub(crate) fn fits(&self, regions: &[(u64, u64)]) -> bool {
        self.pieces.len() == regions.len()
            && self
                .pieces
                .iter()
                .zip(regions)
                .all(|(piece, &(_, take))| piece.size >= take)
    }

    /// Bounce `src` into device-local memory on the compute queue, then DMA
    /// that bounce on the transfer queue. The semaphore is signaled by the
    /// bounce submit, so the DMA can start when the bounce finishes.
    pub(crate) fn arm(&mut self, gpu: &Gpu, src: &wgpu::Buffer, regions: &[(u64, u64)]) -> Result<()> {
        if self.broken.get() {
            bail!("transfer readback is unusable after a failed submit");
        }
        if regions.len() != self.pieces.len() {
            bail!("transfer readback piece count changed");
        }
        if self.in_flight.get() {
            self.wait()?;
        }
        unsafe { self.vk.reset_fences(&[self.fence]) }
            .map_err(|e| anyhow::anyhow!("reset transfer fence: {e:?}"))?;
        unsafe {
            self.vk
                .reset_command_buffer(self.cmd, ash::vk::CommandBufferResetFlags::empty())
        }
        .map_err(|e| anyhow::anyhow!("reset transfer command: {e:?}"))?;

        let begin = ash::vk::CommandBufferBeginInfo::default()
            .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.vk.begin_command_buffer(self.cmd, &begin) }
            .map_err(|e| anyhow::anyhow!("begin transfer: {e:?}"))?;
        for (piece, &(_, take)) in self.pieces.iter().zip(regions) {
            let region = ash::vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: take,
            };
            unsafe {
                self.vk
                    .cmd_copy_buffer(self.cmd, piece.bounce_raw, piece.host.buffer, &[region]);
            }
        }
        unsafe { self.vk.end_command_buffer(self.cmd) }
            .map_err(|e| anyhow::anyhow!("end transfer: {e:?}"))?;

        {
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("logits-bounce"),
                });
            for (piece, &(src_off, take)) in self.pieces.iter().zip(regions) {
                enc.copy_buffer_to_buffer(src, src_off, &piece.bounce, 0, take);
            }
            let hal = unsafe { gpu.queue.as_hal::<wgpu::hal::api::Vulkan>() }
                .context("vulkan queue")?;
            hal.add_signal_semaphore(self.sem, None);
            drop(hal);
            gpu.queue.submit([enc.finish()]);
        }

        let waits = [self.sem];
        let stages = [GRAPHICS_WAIT];
        let cmds = [self.cmd];
        let info = ash::vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&cmds);
        if let Err(e) = unsafe { self.vk.queue_submit(self.queue, &[info], self.fence) } {
            // The bounce submit already consumed the semaphore signal. Idle
            // before the semaphore can be destroyed or signaled again.
            self.broken.set(true);
            self.in_flight.set(false);
            let _ = unsafe { self.vk.device_wait_idle() };
            bail!("transfer submit: {e:?}");
        }
        self.in_flight.set(true);
        Ok(())
    }

    pub(crate) fn wait(&self) -> Result<f64> {
        let t = std::time::Instant::now();
        unsafe { self.vk.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| anyhow::anyhow!("transfer fence: {e:?}"))?;
        self.in_flight.set(false);
        Ok(t.elapsed().as_secs_f64() * 1000.0)
    }

    pub(crate) fn copy_out(&self, regions: &[(u64, u64)], out: &mut Vec<f32>) {
        for (piece, &(_, take)) in self.pieces.iter().zip(regions) {
            let n = (take as usize) / 4;
            let start = out.len();
            out.resize(start + n, 0.0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    piece.host.ptr.as_ptr(),
                    out[start..].as_mut_ptr().cast::<u8>(),
                    take as usize,
                );
            }
        }
    }
}

fn device_local(flags: ash::vk::MemoryPropertyFlags) -> bool {
    flags.contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
        && !flags.contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
}

fn host_cached(flags: ash::vk::MemoryPropertyFlags) -> bool {
    flags.contains(
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE
            | ash::vk::MemoryPropertyFlags::HOST_COHERENT
            | ash::vk::MemoryPropertyFlags::HOST_CACHED,
    ) && !flags.contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
}

fn pick_type(
    props: &ash::vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    pred: impl Fn(ash::vk::MemoryPropertyFlags) -> bool,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        bits & (1u32 << i) != 0 && pred(props.memory_types[i as usize].property_flags)
    })
}

fn alloc_bound(
    vk: &ash::Device,
    info: &ash::vk::BufferCreateInfo<'_>,
    pick: impl FnOnce(u32) -> Option<u32>,
) -> Result<(ash::vk::Buffer, ash::vk::DeviceMemory, u64, u32)> {
    let buffer = unsafe { vk.create_buffer(info, None) }
        .map_err(|e| anyhow::anyhow!("vkCreateBuffer: {e:?}"))?;
    let reqs = unsafe { vk.get_buffer_memory_requirements(buffer) };
    let Some(type_index) = pick(reqs.memory_type_bits) else {
        unsafe { vk.destroy_buffer(buffer, None) };
        bail!("no memory type for a transfer buffer");
    };
    let memory = match unsafe {
        vk.allocate_memory(
            &ash::vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(type_index),
            None,
        )
    } {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { vk.destroy_buffer(buffer, None) };
            bail!("vkAllocateMemory: {e:?}");
        }
    };
    if let Err(e) = unsafe { vk.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            vk.destroy_buffer(buffer, None);
            vk.free_memory(memory, None);
        }
        bail!("vkBindBufferMemory: {e:?}");
    }
    Ok((buffer, memory, reqs.size, type_index))
}
