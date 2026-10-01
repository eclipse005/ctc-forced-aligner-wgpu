//! conv0 kernel at full scale vs numpy-style CPU reference.
use ctc_forced_aligner_wgpu::gpu::{DeviceSelector, Gpu};
use ctc_forced_aligner_wgpu::shaders;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Cfg4 { a: u32, b: u32, c: u32, d: u32 }

fn main() -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new_with(DeviceSelector::parse("auto").unwrap()))?;
    let pipe = gpu.pipeline("conv0", &shaders::conv0(), "main", None)?;

    let n = 251840usize;
    let k = 10usize;
    let stride = 5usize;
    let chans = 512usize;
    let t_out = (n - k) / stride + 1;
    let x: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 * 0.16 - 7.7)).collect();
    let w: Vec<f32> = (0..chans * k).map(|i| ((i % 23) as f32 * 0.09 - 1.0)).collect();
    let bias: Vec<f32> = (0..chans).map(|i| ((i % 11) as f32 * 0.02 - 0.1)).collect();

    let bx = gpu.storage("x", (n * 4) as u64);
    let bw = gpu.storage("w", (chans * k * 4) as u64);
    let bb = gpu.storage("b", (chans * 4) as u64);
    let bo = gpu.storage("o", (t_out * chans * 4) as u64);
    let bu = gpu.uniform("u", 256);
    gpu.upload(&bx, bytemuck::cast_slice(&x));
    gpu.upload(&bw, bytemuck::cast_slice(&w));
    gpu.upload(&bb, bytemuck::cast_slice(&bias));
    gpu.upload(&bu, bytemuck::bytes_of(&Cfg4 { a: t_out as u32, b: chans as u32, c: k as u32, d: stride as u32 }));

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bo.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bu.as_entire_binding() },
            ],
        }), &[]);
        pass.dispatch_workgroups((t_out as u32).div_ceil(16), (chans as u32).div_ceil(16), 1);
    }
    gpu.queue.submit([enc.finish()]);
    let bytes = gpu.readback(&bo, (t_out * chans * 4) as u64)?;
    let got: &[f32] = bytemuck::cast_slice(&bytes);

    let mut max_err = 0.0f32;
    for j in 0..t_out {
        for c in 0..chans {
            let mut acc = bias[c];
            for t in 0..k { acc += x[j * stride + t] * w[c * k + t]; }
            max_err = max_err.max((got[j * chans + c] - acc).abs());
        }
    }
    println!("conv0 full-scale max_err = {max_err}");
    anyhow::ensure!(max_err < 1e-3, "CONV0 WRONG AT SCALE");
    println!("conv0 OK at scale");
    Ok(())
}
