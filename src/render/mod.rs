//! Rendering a [`CueDoc`](crate::views::CueDoc) to a subtitle file.
//!
//! One module per format, and neither of them decides a time: they are handed
//! the line breaking that [`crate::views::build_cues`] computed and only write
//! it out. That is why `--format srt` and the `cues` field of `json` cannot
//! disagree — they are the same computation, rendered twice.

pub mod ass;
pub mod srt;
