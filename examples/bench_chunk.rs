//! Time repeated 34s-chunk forwards to separate cold-start from steady state.
use ctc_forced_aligner_wgpu::audio::znorm;
use ctc_forced_aligner_wgpu::wav2vec2::{Model, StageSet};

const MODEL: &str = r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf";

fn main() -> anyhow::Result<()> {
    let model = Model::load(std::path::Path::new(MODEL))?;
    // one 34s chunk: 544_000 samples at 16 kHz
    let mut input: Vec<f32> = (0..544_000)
        .map(|i| ((i as f32 * 0.01).sin() * 0.3) as f32)
        .collect();
    znorm(&mut input);
    let stages = StageSet::default();
    for it in 0..8 {
        let t0 = std::time::Instant::now();
        let (lp, _) = model.forward(&input, &stages)?;
        let dt = t0.elapsed().as_secs_f64();
        println!("forward {it}: {dt:.3}s  (log_probs {} frames)", lp.len() / 10288);
    }
    Ok(())
}
