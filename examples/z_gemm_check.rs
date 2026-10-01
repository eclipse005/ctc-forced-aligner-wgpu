//! z-batched GEMM (wid.z = head) vs a CPU reference on the scores shape.
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
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
    a_z: u32,
    b_z: u32,
    c_z: u32,
    scale: f32,
}

fn main() -> anyhow::Result<()> {
    let gpu = pollster::block_on(ctc_forced_aligner_wgpu::Gpu::new_with(
        ctc_forced_aligner_wgpu::DeviceSelector::parse("auto")?,
    ))?;
    let pipe = gpu.pipeline_no_zero("zgemm", &ctc_forced_aligner_wgpu::shaders::gemm_bias(), "main", None)?;

    let (t, heads, head_dim, hidden) = (130usize, 4usize, 16usize, 64usize);
    // qkv (t, 3h): head h's Q lives at columns h*head_dim..+head_dim
    let a: Vec<f32> = (0..t * 3 * hidden).map(|i| (i as f32 * 0.113).sin()).collect();
    // kt (hidden, t): head h's K^T rows at h*head_dim..+head_dim
    let kt: Vec<f32> = (0..hidden * t).map(|i| (i as f32 * 0.071).cos()).collect();

    let a_b = gpu.storage("a", (a.len() * 4) as u64);
    gpu.upload(&a_b, bytemuck::cast_slice(&a));
    let kt_b = gpu.storage("kt", (kt.len() * 4) as u64);
    gpu.upload(&kt_b, bytemuck::cast_slice(&kt));
    let zeros = gpu.storage("zeros", 4096 * 4);
    let c_b = gpu.storage("c", (heads * t * t * 4) as u64);
    let uni = gpu.uniform("dims", 256);
    gpu.upload(
        &uni,
        bytemuck::bytes_of(&Dims {
            m: t as u32, n: t as u32, k: head_dim as u32,
            a_stride: (3 * hidden) as u32, b_stride: t as u32, c_stride: t as u32,
            a_off: 0, b_off: 0, c_off: 0,
            a_z: head_dim as u32, b_z: (head_dim * t) as u32, c_z: (t * t) as u32,
            scale: 0.125,
        }),
    );
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: kt_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: zeros.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: c_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: uni.as_entire_binding() },
        ],
        label: None,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(t.div_ceil(128) as u32, t.div_ceil(64) as u32, heads as u32);
    }
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let bytes = gpu.readback(&c_b, (heads * t * t * 4) as u64)?;
    let got: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let mut worst = 0.0f32;
    'outer: for h in 0..heads {
        for i in 0..t {
            for j in 0..t {
                let mut acc = 0.0f32;
                for k in 0..head_dim {
                    acc += a[i * 3 * hidden + h * head_dim + k] * kt[(h * head_dim + k) * t + j];
                }
                let want = 0.125 * acc;
                let d = (got[h * t * t + i * t + j] - want).abs();
                worst = worst.max(d);
                if worst > 1e-3 {
                    println!("first big diff at h={h} i={i} j={j}: got {} want {want}", got[h * t * t + i * t + j]);
                    break 'outer;
                }
            }
        }
    }
    println!("z-gemm scores shape: worst {worst:.3e} {}", if worst < 1e-3 { "OK" } else { "MISMATCH" });
    Ok(())
}
