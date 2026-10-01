//! CPU forward RTF baseline — the number every RTFX change is judged by.
//!
//! Run: `cargo bench --bench rtfx`  (model dir: `$CTC_MODEL_DIR`)
//!
//! RTF = 34 s of audio / mean iteration time (higher is better, 1.0 = realtime).
//! `34s_steady_scratch` is the production path (the aligner reuses one Scratch
//! across chunks); `34s_fresh_scratch` quantifies the buffer-reuse win.

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use ctc_forced_aligner_wgpu::audio::znorm;
use ctc_forced_aligner_wgpu::wav2vec2::{Model, Scratch, StageSet};

fn model_dir() -> PathBuf {
    std::env::var_os("CTC_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf"))
}

/// One 34 s chunk (544_000 samples @ 16 kHz) — the windowed-align shape.
fn bench_input() -> Vec<f32> {
    let mut input: Vec<f32> = (0..544_000)
        .map(|i| ((i as f32 * 0.01).sin() * 0.3) as f32)
        .collect();
    znorm(&mut input);
    input
}

fn bench_forward(c: &mut Criterion) {
    let model = Model::load(&model_dir()).expect("load model (set CTC_MODEL_DIR)");
    let input = bench_input();
    let stages = StageSet::default();

    let mut group = c.benchmark_group("cpu_forward");
    group.throughput(Throughput::Elements((input.len() / 320) as u64));

    // steady state: one Scratch reused across iterations (the aligner's path)
    let mut scratch = Scratch::default();
    model.forward_with(&input, &stages, &mut scratch).unwrap();
    group.bench_function("34s_steady_scratch", |b| {
        b.iter(|| model.forward_with(&input, &stages, &mut scratch).unwrap())
    });

    // fresh Scratch per iteration: pays the ~90 MB of buffer churn
    group.bench_function("34s_fresh_scratch", |b| {
        b.iter_batched(
            Scratch::default,
            |mut scratch| model.forward_with(&input, &stages, &mut scratch).unwrap(),
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
