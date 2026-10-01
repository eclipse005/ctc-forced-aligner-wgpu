//! Two GEMM dispatches with different a_off at different uniform slots:
//! verifies per-dispatch uniform data actually reaches the kernel.
use ctc_forced_aligner_wgpu::gpu::{DeviceSelector, Gpu};
use ctc_forced_aligner_wgpu::shaders;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct D {
    m: u32, n: u32, k: u32,
    a_stride: u32, b_stride: u32, c_stride: u32,
    a_off: u32, b_off: u32, c_off: u32,
    scale: f32,
}

fn main() -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new_with(DeviceSelector::parse("auto").unwrap()))?;
    let pipe = gpu.pipeline("gemm", &shaders::gemm_bias(), "main", None)?;

    // A holds two DIFFERENT 8x8 matrices; each dispatch must read its own.
    let a: Vec<f32> = (0..128).map(|i| if i < 64 { 1.0 } else { 2.0 }).collect();
    let b: Vec<f32> = vec![1.0; 64];
    let bias: Vec<f32> = vec![0.0; 8];
    let ba = gpu.storage("a", 128 * 4);
    let bb = gpu.storage("b", 64 * 4);
    let bbias = gpu.storage("bias", 8 * 4);
    let bc = gpu.storage("c", 2 * 64 * 4);
    // per-dispatch uniform buffers, written synchronously via mapped_at_creation
    gpu.upload(&ba, bytemuck::cast_slice(&a));
    gpu.upload(&bb, bytemuck::cast_slice(&b));
    gpu.upload(&bbias, bytemuck::cast_slice(&bias));

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    for (_slot, a_off, c_off) in [(0u64, 0u32, 0u32), (256u64, 64u32, 64u32)] {

        let dims = D {
            m: 8, n: 8, k: 8, a_stride: 8, b_stride: 8, c_stride: 8,
            a_off, b_off: 0, c_off, scale: 1.0,
        };
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dims"),
            size: 40,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        match ubuf.slice(..).get_mapped_range_mut() { Ok(mut view) => view.copy_from_slice(bytemuck::bytes_of(&dims)), Err(_) => unreachable!() }
        ubuf.unmap();
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ba.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bbias.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bc.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue.submit([enc.finish()]);

    let bytes = gpu.readback(&bc, 2 * 64 * 4)?;
    let got: &[f32] = bytemuck::cast_slice(&bytes);
    // dispatch 0: sum of ones*1 = 8 per element. dispatch 1 (a_off=64): 2*8=16.
    println!("c[0..3] (want 8):   {:?}", &got[0..3]);
    println!("c[64..67] (want 16): {:?}", &got[64..67]);
    Ok(())
}
