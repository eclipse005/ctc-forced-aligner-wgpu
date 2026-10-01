//! Isolate: map a 32 MB staging after a copy — no model involved.
use ctc_forced_aligner_wgpu::gpu::{DeviceSelector, Gpu};

fn main() -> anyhow::Result<()> {
    let gpu = pollster::block_on(Gpu::new_with(DeviceSelector::parse("auto").unwrap()))?;
    let n = 8_086_368usize; // 786 x 10288 floats = 31.7 MB
    let src = gpu.storage("src", (n * 4) as u64);
    let data: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    gpu.upload(&src, bytemuck::cast_slice(&data));

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&src, 0, &staging, 0, (n * 4) as u64);
    gpu.queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let got = slice.get_mapped_range()?;
    println!("map ok, first5 = {:?}", &got[..5].chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect::<Vec<_>>());
    staging.unmap();
    Ok(())
}
