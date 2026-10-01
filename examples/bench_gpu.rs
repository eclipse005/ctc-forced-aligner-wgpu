//! GPU tower steady-state RTF: repeated forwards on one 34 s chunk.
//!
//! `cargo run --release --example bench_gpu [-- <seconds>]`
//! Pass CTC_PROFILE=2 for the per-kernel timestamp split (first forward).
use std::time::Instant;

use ctc_forced_aligner_wgpu::audio::znorm;
use ctc_forced_aligner_wgpu::wav2vec2_gpu::GpuModel;
use ctc_forced_aligner_wgpu::DeviceSelector;

fn main() -> anyhow::Result<()> {
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(34.0);
    let dir = std::env::var_os("CTC_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf"));
    let t0 = Instant::now();
    let model = GpuModel::load(&dir, DeviceSelector::parse("auto")?)?;
    println!("load+upload: {:.2}s  ({})", t0.elapsed().as_secs_f64(), model.adapter_name());

    let n = (secs * 16_000.0) as usize;
    let mut input: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin() * 0.3) as f32).collect();
    znorm(&mut input);

    // production path: (T, S) gathered readback for the alignment Viterbi
    let labels: Vec<u32> = (1..=43u32).collect(); // 43 tokens -> S = 87
    let expanded: Vec<u32> = (0..labels.len() * 2 + 1)
        .map(|i| if i % 2 == 0 { 0 } else { labels[(i - 1) / 2] })
        .collect();

    // first forward records the dispatch graph; later ones replay it
    let t0 = Instant::now();
    let g = model.forward_gathered(&input, &expanded)?;
    println!(
        "gathered #1 (record): {:.3}s ({} frames x {} states)",
        t0.elapsed().as_secs_f64(),
        g.len() / expanded.len(),
        expanded.len()
    );

    let mut best = f64::MAX;
    for i in 2..=6 {
        let t0 = Instant::now();
        model.forward_gathered(&input, &expanded)?;
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        println!("gathered #{i} (replay): {dt:.3}s   RTF = {:.1}x", secs / dt);
    }
    println!("steady RTF = {:.1}x realtime ({secs}s audio / {best:.3}s)", secs / best);
    Ok(())
}
