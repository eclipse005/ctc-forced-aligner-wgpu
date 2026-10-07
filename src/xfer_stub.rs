//! Placeholder where Vulkan cannot exist (macOS and friends): wgpu-hal does
//! not build a Vulkan backend there, so there is no dedicated transfer queue
//! and no [`Engine`]. Every readback takes the MAP_READ path; the API mirrors
//! [`crate::xfer`]'s real implementation one-to-one so callers need no cfg.

use anyhow::Result;

use crate::gpu::Gpu;

/// Same threshold as the real module: it only decides between readback
/// styles, and the style below never activates.
pub(crate) const MIN_BYTES: u64 = 16 << 20;

/// Never constructed: [`Engine::open`] always returns `None` where Vulkan is
/// absent, so every consumer falls back to `MAP_READ` staging.
pub(crate) struct Engine;

impl Engine {
    pub(crate) fn open(_gpu: &Gpu) -> Option<Self> {
        None
    }

    pub(crate) fn alloc_set(&self, _device: &wgpu::Device, _takes: &[u64]) -> Option<XferSet> {
        None
    }
}

/// Never constructed on this platform; the bodies exist only to satisfy the
/// callers' signatures.
pub(crate) struct XferSet;

impl XferSet {
    pub(crate) fn fits(&self, _regions: &[(u64, u64)]) -> bool {
        false
    }

    pub(crate) fn arm(
        &mut self,
        _gpu: &Gpu,
        _src: &wgpu::Buffer,
        _regions: &[(u64, u64)],
    ) -> Result<()> {
        unreachable!("no transfer engine where Vulkan is absent")
    }

    pub(crate) fn wait(&self) -> Result<f64> {
        unreachable!("no transfer engine where Vulkan is absent")
    }

    pub(crate) fn copy_out(&self, _regions: &[(u64, u64)], _out: &mut Vec<f32>) {
        unreachable!("no transfer engine where Vulkan is absent")
    }
}
