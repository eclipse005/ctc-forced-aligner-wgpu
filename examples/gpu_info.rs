//! Print every wgpu adapter the selector can see, with backend + limits.
fn main() {
    pollster::block_on(async {
        for (i, t) in ctc_forced_aligner_wgpu::list_targets().await.iter().enumerate() {
            println!("{i}: {t}");
        }
    });
    let gpu = pollster::block_on(ctc_forced_aligner_wgpu::Gpu::new_with(
        ctc_forced_aligner_wgpu::DeviceSelector::parse("auto").unwrap(),
    ))
    .expect("no GPU");
    println!("auto -> {}", gpu.describe());
}
