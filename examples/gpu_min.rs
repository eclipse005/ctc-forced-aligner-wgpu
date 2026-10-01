//! Minimal GPU sanity: run the conv0 kernel on a tiny input and read back.
use ctc_forced_aligner_wgpu::gpu::{DeviceSelector, Gpu};
use ctc_forced_aligner_wgpu::shaders;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Cfg4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

fn main() -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new_with(DeviceSelector::parse("auto").unwrap()))?;
    println!("device: {}", gpu.describe());

    let pipe = gpu.pipeline("conv0", &shaders::conv0(), "main", None)?;

    // 32 samples -> (32-10)/5+1 = 5 frames, 4 channels
    let n = 32u32;
    let t_out = (n - 10) / 5 + 1;
    let chans = 4u32;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    // w[c][t] = 1 => out = sum of window + bias c
    let w: Vec<f32> = vec![1.0; (chans * 10) as usize];
    let bias: Vec<f32> = (0..chans).map(|c| 100.0 + c as f32).collect();

    let bx = gpu.storage("x", (n * 4) as u64);
    let bw = gpu.storage("w", (chans * 10 * 4) as u64);
    let bb = gpu.storage("b", (chans * 4) as u64);
    let bo = gpu.storage("o", (t_out * chans * 4) as u64);
    // mimic forward(): one big uniform buffer, per-dispatch offset binding
    let bu = gpu.uniform("u", 8192 * 256);
    let slot0: u64 = 3 * 256;
    gpu.upload(&bx, bytemuck::cast_slice(&x));
    gpu.upload(&bw, bytemuck::cast_slice(&w));
    gpu.upload(&bb, bytemuck::cast_slice(&bias));
    gpu.queue.write_buffer(&bu, slot0, bytemuck::bytes_of(&Cfg4 { a: t_out, b: chans, c: 10, d: 5 }));

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(
            0,
            &gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bw.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: bo.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bu, offset: slot0, size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    }) },
                ],
            }),
            &[],
        );
        pass.dispatch_workgroups(t_out, chans.div_ceil(16), 1);
    }
    gpu.queue.submit([enc.finish()]);

    let bytes = gpu.readback(&bo, (t_out * chans * 4) as u64)?;
    let vals: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    println!("out: {vals:?}");
    // expected: sums of 5-sample windows: [0..10): j0=0+5=... windows at 0,5,10,15,20
    // j0: 0+1+2+3+4=10, j1: 5..9=35, j2: 10..14=60, j3: 15..19=85, j4: 20..24=110
    Ok(())
}

#[test]
fn bulk_upload_roundtrip() -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new_with(DeviceSelector::parse("auto").unwrap()))?;
    let data: Vec<f32> = (0..5120).map(|i| i as f32).collect();
    let mut up = gpu.uploader();
    let buf = up.storage("test", (data.len() * 4) as u64);
    up.upload(&buf, bytemuck::cast_slice(&data))?;
    up.finish()?;
    let bytes = gpu.readback(&buf, (data.len() * 4) as u64)?;
    let back: &[f32] = bytemuck::cast_slice(&bytes);
    println!("bulk roundtrip: first5={:?} max_err={}", &back[..5],
        back.iter().zip(&data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max));
    Ok(())
}
