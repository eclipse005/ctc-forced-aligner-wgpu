//! Compare the WGSL LayerNorm kernels against a CPU reference.
use bytemuck::Pod;
use ctc_forced_aligner_wgpu::gpu::{DeviceSelector, Gpu};
use ctc_forced_aligner_wgpu::shaders;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Cfg {
    rows: u32,
    cols: u32,
    eps_x_1e6: u32,
    do_gelu: u32,
}
use bytemuck::Zeroable;

fn cpu_ln(x: &[f32], w: &[f32], b: &[f32], rows: usize, cols: usize, gelu: bool) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / ((var as f64 + 1e-5).sqrt() as f32);
        for c in 0..cols {
            let mut o = (row[c] - mean) * inv * w[c] + b[c];
            if gelu {
                let z = o / 1.4142135623730951;
                let a = z.abs();
                let t = 1.0 / (1.0 + 0.3275911 * a);
                let e = 1.0
                    - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                        + 0.254829592)
                        * t
                        * (-a * a).exp();
                o = 0.5 * o * (1.0 + o.signum() * e);
            }
            out[r * cols + c] = o;
        }
    }
    out
}

fn run(
    gpu: &Gpu,
    pipe: &wgpu::ComputePipeline,
    sd: bool,
    src: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    cols: usize,
    gelu: bool,
) -> anyhow::Result<Vec<f32>> {
    let src_b = gpu.storage("src", (src.len() * 4) as u64);
    gpu.upload(&src_b, bytemuck::cast_slice(src));
    let dst_b = gpu.storage("dst", (src.len() * 4) as u64);
    let w_b = gpu.storage("w", (w.len() * 4) as u64);
    gpu.upload(&w_b, bytemuck::cast_slice(w));
    let b_b = gpu.storage("b", (b.len() * 4) as u64);
    gpu.upload(&b_b, bytemuck::cast_slice(b));
    let uni = gpu.uniform("cfg", 256);
    gpu.upload(
        &uni,
        bytemuck::bytes_of(&Cfg {
            rows: rows as u32,
            cols: cols as u32,
            eps_x_1e6: 10,
            do_gelu: gelu as u32,
        }),
    );
    let entries = if sd {
        vec![
            wgpu::BindGroupEntry { binding: 0, resource: src_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: uni.as_entire_binding() },
        ]
    } else {
        vec![
            wgpu::BindGroupEntry { binding: 0, resource: src_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
        ]
    };
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
        label: None,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(rows as u32, 1, 1);
    }
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let out_b = if sd { &dst_b } else { &src_b };
    let bytes = gpu.readback(out_b, (src.len() * 4) as u64)?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new_with(DeviceSelector::parse("auto")?))?;
    let (rows, cols) = (37usize, 1024usize);
    let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.123).sin() * 3.0).collect();
    let w: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32 * 0.01).cos() * 0.2).collect();
    let b: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.02).sin() * 0.1).collect();

    // src/dst kernel
    let p_sd = gpu.pipeline("ln_sd", &shaders::layernorm_sd(), "main", None)?;
    let got = run(&gpu, &p_sd, true, &src, &w, &b, rows, cols, false)?;
    let want = cpu_ln(&src, &w, &b, rows, cols, false);
    let worst: f32 = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("ln_sd  (1024): worst {worst:.3e} {}", if worst < 2e-4 { "OK" } else { "MISMATCH" });

    // in-place kernel with gelu, cols=512
    let (rows5, cols5) = (37usize, 512usize);
    let src5: Vec<f32> = (0..rows5 * cols5).map(|i| (i as f32 * 0.077).cos() * 2.0).collect();
    let w5: Vec<f32> = (0..cols5).map(|i| 1.0 + (i as f32 * 0.013).sin() * 0.3).collect();
    let b5: Vec<f32> = (0..cols5).map(|i| (i as f32 * 0.031).cos() * 0.1).collect();
    let p_ip = gpu.pipeline("ln", &shaders::layernorm(), "main", None)?;
    let got5 = run(&gpu, &p_ip, false, &src5, &w5, &b5, rows5, cols5, true)?;
    let want5 = cpu_ln(&src5, &w5, &b5, rows5, cols5, true);
    let worst5: f32 = got5.iter().zip(&want5).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("ln inplace+gelu (512): worst {worst5:.3e} {}", if worst5 < 2e-4 { "OK" } else { "MISMATCH" });

    if worst5 > 2e-4 {
        for r in [0usize, 1, 2] {
            println!("row {r} got[0..6]:  {:?}", &got5[r * cols5..r * cols5 + 6]);
            println!("row {r} want[0..6]: {:?}", &want5[r * cols5..r * cols5 + 6]);
        }
    }
    Ok(())
}
