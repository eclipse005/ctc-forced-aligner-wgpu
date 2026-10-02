//! Probe: record-vs-replay for the hidden AND logits readbacks.

use ctc_forced_aligner_wgpu::audio::{load_audio, znorm, TARGET_SR};
use ctc_forced_aligner_wgpu::wav2vec2_gpu::GpuModel;
use ctc_forced_aligner_wgpu::DeviceSelector;

fn model_dir() -> std::path::PathBuf {
    std::env::var_os("CTC_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf"))
}

fn chunk(wave: &[f32], start_s: usize) -> Vec<f32> {
    let sr = TARGET_SR as usize;
    let mut v = wave[start_s * sr..(start_s + 3) * sr].to_vec();
    znorm(&mut v);
    v
}

fn diff(a: &[f32], b: &[f32]) -> (usize, f32, usize) {
    let mut max = 0.0f32;
    let mut at = 0usize;
    let mut ndiff = 0usize;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > 1e-4 { ndiff += 1; }
        if d > max { max = d; at = i; }
    }
    (at, max, ndiff)
}

fn main() -> anyhow::Result<()> {
    let (wave, sr) = load_audio(&std::path::Path::new(r"D:\omnilingual-asr\fixtures\3m.wav"))?;
    assert_eq!(sr, TARGET_SR);
    let a = chunk(&wave, 0);
    let b = chunk(&wave, 30);
    let g = GpuModel::load(&model_dir(), DeviceSelector::parse("auto")?)?;

    // ---- hidden mode
    let a1 = g.forward_hidden(&a)?; // record
    let b1 = g.forward_hidden(&b)?; // replay
    let a3 = g.forward_hidden(&a)?; // replay
    let (at, d, n) = diff(&a1, &a3);
    println!("hidden  A record vs A replay : max {d:.4} at {at}, {n} elems differ of {}", a1.len());
    println!("  a1[0..4] {:?}", &a1[..4]);
    println!("  b1[0..4] {:?}", &b1[..4]);
    println!("  a3[0..4] {:?}", &a3[..4]);
    println!("  a1[row86][780..784] {:?}", &a1[86*1024+780..86*1024+784]);
    println!("  a3[row86][780..784] {:?}", &a3[86*1024+780..86*1024+784]);
    let (at, d, n) = diff(&a1, &b1);
    println!("hidden  A record vs B replay : max {d:.4} at {at}, {n} differ (sanity: should be large)");
    let (at, d, n) = diff(&a3, &b1);
    println!("hidden  A replay vs B replay : max {d:.4} at {at}, {n} differ (stale-buffer test)");

    // fresh model: B as the FIRST call (record) vs g's B (replay)
    let g2 = GpuModel::load(&model_dir(), DeviceSelector::parse("auto")?)?;
    let b2 = g2.forward_hidden(&b)?;
    let (at, d, n) = diff(&b1, &b2);
    println!("hidden  B replay(g) vs B record(g2): max {d:.4} at {at}, {n} differ");

    // ---- logits mode (control: known-good replay)
    let la1 = g.forward(&a)?; // re-records (mode change) — first logits call
    let la3 = g.forward(&a)?; // replay
    let (at, d, n) = diff(&la1, &la3);
    println!("logits  A record vs A replay : max {d:.6} at {at}, {n} differ of {}", la1.len());
    Ok(())
}
