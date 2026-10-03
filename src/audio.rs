//! Audio decoding: WAV → mono f32 @16 kHz, and the feature-extractor
//! z-normalisation. Matches the Python side (`omni_align/audio.py` +
//! Wav2Vec2FeatureExtractor) sample-for-sample on PCM16 input.

use anyhow::{Context, Result};

pub(crate) const TARGET_SR: u32 = 16000;

/// Decode a WAV file to mono f32, resampling to [`TARGET_SR`] when needed.
pub(crate) fn load_audio(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let mut samples: Vec<f32> = Vec::with_capacity(64 * 1024);
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>() {
                samples.push(s?);
            }
        }
        hound::SampleFormat::Int => {
            let max = match spec.bits_per_sample {
                8 => 128.0,
                16 => 32768.0,
                24 => 8388608.0,
                32 => 2147483648.0,
                b => anyhow::bail!("unsupported PCM bit depth {b}"),
            };
            for s in reader.samples::<i32>() {
                samples.push(s? as f32 / max);
            }
        }
    }

    // fold to mono by averaging (soundfile always_2d + mean(axis=1))
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if sr == TARGET_SR {
        Ok((mono, sr))
    } else {
        Ok((resample(&mono, sr, TARGET_SR), TARGET_SR))
    }
}

/// Wav2Vec2FeatureExtractor's z-normalisation: biased variance, eps inside
/// the sqrt, computed per chunk (the Python path normalises each chunk too).
pub(crate) fn znorm(w: &mut [f32]) {
    let n = w.len() as f32;
    let mean = w.iter().sum::<f32>() / n;
    let var = w.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-7).sqrt();
    for x in w.iter_mut() {
        *x = (*x - mean) * inv;
    }
}

/// Resample by sinc interpolation with a Hann window, torchaudio's
/// `sinc_interp_hann` default (lowpass_filter_width=6, rolloff=0.99).
pub(crate) fn resample(input: &[f32], orig_freq: u32, new_freq: u32) -> Vec<f32> {
    crate::resample_sinc::resample(input, orig_freq, new_freq)
}

// The resampler lives in its own module (generated from the torchaudio
// algorithm) — see resample_sinc.rs.
pub use crate::resample_sinc;
