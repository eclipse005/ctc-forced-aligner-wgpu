//! ctc-forced-aligner-wgpu — omniASR-CTC (300M v2) forced alignment in Rust,
//! on wgpu, with a CPU backend alongside it.
//!
//! Audio + transcript in, per-character (and word / sentence) timestamps out.
//! The model is Meta's omniASR-CTC-300M-v2 (the HF conversion shipped in
//! `models/omniASR-CTC-300M-v2-hf`): a Wav2Vec2 encoder with a 10288-way
//! character CTC head, 50 fps, blank id 0.
//!
//! The behaviour reference is the Python implementation this was ported from
//! (`D:/omnilingual-asr/omni_align/`): same Viterbi (stay/advance/skip with
//! stay-wins ties, f64 path scores), same char→word→sentence aggregation.
//! `golden/` carries per-stage activations dumped from Python and `src/golden.rs`
//! diffs every stage against them.
//!
//! # The API
//!
//! ```no_run
//! use ctc_forced_aligner_wgpu::{Aligner, DeviceSelector};
//! use std::path::Path;
//!
//! let aligner = Aligner::load_on(Path::new("models/omniASR-CTC-300M-v2-hf"),
//!                                DeviceSelector::parse("auto")?)?;
//! let out = aligner.align(Path::new("speech.wav"), "hello world", Some(30.0), 2.0, None)?;
//! for t in &out.tokens {
//!     println!("{:.3}s - {:.3}s  {}", t.start, t.end, t.piece);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! That is the whole of it: [`Aligner`] to load and run, [`AlignOutput`] to
//! read, [`WordSpan`] for one timed unit, [`TokenAlignment`] for one CTC
//! target, [`DeviceSelector`] to pick a device, [`list_targets`] to list them.
//! Subtitles are [`views::build_cues`] rendered by [`render::srt`] or
//! [`render::ass`]. Everything else is `pub(crate)` and free to change.
//!
//! # Which view to read
//!
//! `AlignOutput::words` is the alignment as units: one **character** where the
//! script writes without spaces between its words, one **word** where it does
//! not, decided from the characters themselves with no language table and no
//! flag. `你好` is two units, `alignment` is one, `你好Whisper` is three, and
//! [`space_before`](WordSpan::space_before) carries the transcript's own gaps so
//! the units can be rendered back into the text they came from.
//!
//! `AlignOutput::tokens` is the other view, and the one to reach for only when a
//! character is the unit you want: one row per CTC target, which for this
//! character-level checkpoint means one row per character for *every* script. A
//! consumer that joins those with spaces renders `Whisper` as `W h i s p e r`,
//! which is why [`tokens`](AlignOutput::tokens) serializes as units in JSON even
//! though it holds targets in memory.
//!
//! # How it is put together
//!
//! Four layers, each depending only on the ones above it:
//!
//! - **base** — [`config`], [`weights`], [`vocab`], [`simd`], [`shaders`],
//!   [`gpu`], [`resample_sinc`], [`audio`]. Checkpoints, kernels, devices. No
//!   idea what an alignment is.
//! - **model** — [`wav2vec2`] and [`wav2vec2_gpu`], the encoder tower and its
//!   CPU twin. Audio in, per-frame label scores out.
//! - **align** — [`viterbi`] reads the path into frame numbers, and
//!   [`timeline`] is the ONE place a timestamp is decided afterwards: a
//!   boundary sits in the middle of a pause that is short enough to be
//!   prosody and on the word's own evidence once the run is long enough to
//!   be silence, a mark has no sound and becomes a point on the sound it
//!   touches, a character with no target lies between its placed neighbours.
//!   sentences, [`views`] breaks them into subtitle lines.
//!
//! [`align_inference`] is the facade over all of it, and [`render`] only writes
//! out what the two layers below decided. Nothing after the Viterbi may write a
//! `start` or an `end`; that is what keeps `--format srt`, `--format ass` and
//! the JSON from ever disagreeing.

pub mod align_inference;
pub mod alloc_stats;
pub mod audio;
pub mod config;
pub mod gpu;
pub mod render;
pub mod resample_sinc;
pub mod shaders;
pub mod simd;
pub mod spans;
pub mod timeline;
// The Vulkan transfer-queue fast path only exists where Vulkan can: wgpu-hal
// does not build a Vulkan backend on macOS, so `wgpu::hal::api::Vulkan` is
// not even a type there. macOS and friends get a stub with the same API.
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod xfer;
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
#[path = "xfer_stub.rs"]
mod xfer;
pub mod viterbi;
mod viterbi_gpu;
pub mod views;
pub mod vocab;
pub mod wav2vec2;
pub mod wav2vec2_gpu;
pub mod weights;

#[cfg(test)]
mod golden;
#[cfg(test)]
mod testutil;

pub use align_inference::{AlignOutput, AlignProgress, Aligner};
pub use gpu::{list_targets, Backend, DeviceSelector};
pub use spans::{WordSpan, build_words};
pub use viterbi::TokenAlignment;
