//! GEMM kernel vs a CPU reference, including partial tiles and strided slices,
//! then one FFN-shaped timing on the Vulkan device.
use ctc_forced_aligner_wgpu::gpu::{DeviceSelector, Gpu};
use ctc_forced_aligner_wgpu::shaders;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    b_stride: u32,
    c_stride: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    scale: f32,
}

fn pat(i: usize) -> f32 {
    ((i % 19) as i32 - 9) as f32 * 0.25
}

fn check(gpu: &Gpu, pipe: &wgpu::ComputePipeline, m: u32, n: u32, k: u32, a_stride: u32, b_stride: u32, c_stride: u32, a_off: u32, b_off: u32, c_off: u32, scale: f32) -> anyhow::Result<f32> {
    let a_len = (a_off + (m - 1) * a_stride + k) as usize;
    let b_len = (b_off + (k - 1) * b_stride + n) as usize;
    let c_len = (c_off + (m - 1) * c_stride + n) as usize;
    let a: Vec<f32> = (0..a_len).map(pat).collect();
    let b: Vec<f32> = (0..b_len).map(|i| pat(i + 3)).collect();
    let bias: Vec<f32> = (0..n as usize).map(|i| (i % 5) as f32 * 0.5).collect();
    let mut want = vec![0f32; c_len];
    for r in 0..m {
        for c in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += a[(a_off + r * a_stride + kk) as usize] * b[(b_off + kk * b_stride + c) as usize];
            }
            want[(c_off + r * c_stride + c) as usize] = s * scale + bias[c as usize];
        }
    }
    let (ba, bb, bbias, bc) = (
        gpu.storage("a", (a_len * 4) as u64),
        gpu.storage("b", (b_len * 4) as u64),
        gpu.storage("bias", (n as usize * 4) as u64),
        gpu.storage("c", (c_len * 4) as u64),
    );
    let bu = gpu.uniform("u", 256);
    gpu.upload(&ba, bytemuck::cast_slice(&a));
    gpu.upload(&bb, bytemuck::cast_slice(&b));
    gpu.upload(&bbias, bytemuck::cast_slice(&bias));
    gpu.upload(&bc, &vec![0u8; c_len * 4]);
    gpu.upload(&bu, bytemuck::bytes_of(&Dims { m, n, k, a_stride, b_stride, c_stride, a_off, b_off, c_off, scale }));
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ba.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bbias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bc.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bu.as_entire_binding() },
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(m.div_ceil(shaders::MT), n.div_ceil(shaders::NT), 1);
    }
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let bytes = gpu.readback(&bc, (c_len * 4) as u64)?;
    let got: &[f32] = bytemuck::cast_slice(&bytes);
    Ok(got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max))
}

fn main() -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new_with(DeviceSelector::parse("p104")?))?;
    println!("{}", gpu.describe());
    let pipe = gpu.pipeline_no_zero("gemm", &shaders::gemm_bias(), "main", None)?;
    let cases = [
        (37, 70, 45, 45, 70, 70, 0, 0, 0, 1.0f32),
        (33, 66, 13, 20, 1699, 80, 1, 3, 5, 0.125),
        (48, 64, 40, 40, 96, 128, 0, 0, 8, 1.0),
    ];
    for (m, n, k, a_stride, b_stride, c_stride, a_off, b_off, c_off, scale) in cases {
        let err = check(&gpu, &pipe, m, n, k, a_stride, b_stride, c_stride, a_off, b_off, c_off, scale)?;
        println!("m={m} n={n} k={k} max_err={err:.3e}");
        anyhow::ensure!(err < 1e-3, "GEMM wrong");
    }

    let (m, n, k) = (1699u32, 4096u32, 1024u32);
    let (ba, bb, bbias, bc) = (
        gpu.storage("a", (m as u64) * (k as u64) * 4),
        gpu.storage("b", (k as u64) * (n as u64) * 4),
        gpu.storage("bias", (n as u64) * 4),
        gpu.storage("c", (m as u64) * (n as u64) * 4),
    );
    let bu = gpu.uniform("u", 256);
    gpu.upload(&ba, &vec![0x3eu8; (m as usize) * (k as usize) * 4]);
    gpu.upload(&bb, &vec![0x3eu8; (k as usize) * (n as usize) * 4]);
    gpu.upload(&bbias, &vec![0u8; n as usize * 4]);
    gpu.upload(&bu, bytemuck::bytes_of(&Dims {
        m, n, k, a_stride: k, b_stride: n, c_stride: n, a_off: 0, b_off: 0, c_off: 0, scale: 1.0,
    }));
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ba.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bbias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bc.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bu.as_entire_binding() },
        ],
    });
    let (gx, gy) = (m.div_ceil(shaders::MT), n.div_ceil(shaders::NT));
    let launch = |reps: u32| -> anyhow::Result<()> {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[]);
            for _ in 0..reps {
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        gpu.queue.submit([enc.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    };
    launch(1)?;
    let reps = 8u32;
    let t0 = std::time::Instant::now();
    launch(reps)?;
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / per / 1e9;
    println!("ffn1 {m}x{n}x{k}  {per:.4}s  {gflops:.0} GFLOP/s  tile {}x{}", shaders::MT, shaders::NT);
    println!("GEMM OK");
    Ok(())
}
