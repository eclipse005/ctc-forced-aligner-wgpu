//! ctc-forced-aligner-wgpu — omniASR-CTC (300M v2) forced alignment in Rust,
//! on wgpu, with a CPU backend alongside it.
//!
//! Audio + transcript in, per-character (and word / segment) timestamps out.
//! The model is Meta's omniASR-CTC-300M-v2 (the HF conversion shipped in
//! `models/omniASR-CTC-300M-v2-hf`): a Wav2Vec2 encoder with a 10288-way
//! character CTC head, 50 fps, blank id 0.
//!
//! The behaviour reference is the Python implementation this was ported from
//! (`D:/omnilingual-asr/omni_align/`): same Viterbi (stay/advance/skip with
//! stay-wins ties, f64 path scores), same char→word→segment aggregation, same
//! output JSON schema. `golden/` carries per-stage activations dumped from
//! Python; `tests/golden.rs` diffs every stage and the final token timestamps.
//!
//! Layout: [`gpu`] is device plumbing, [`weights`] checkpoint access,
//! [`audio`] decoding + resampling + chunking, [`wav2vec2`] the CPU model
//! (types + checkpoint load + forward), [`wav2vec2_gpu`] its GPU twin,
//! [`viterbi`] the aligner, [`spans`] the aggregation, [`align_inference`]
//! the entry point.

pub mod align_inference;
pub mod fix_timestamp;
pub mod audio;
pub mod config;
pub mod gpu;
pub mod resample_sinc;
pub mod shaders;
pub mod simd;
pub mod spans;
pub mod viterbi;
pub mod views;
pub mod vocab;
pub mod wav2vec2;
pub mod wav2vec2_gpu;
pub mod weights;

pub use align_inference::Aligner;
pub use gpu::{list_targets, DeviceSelector, Gpu};
