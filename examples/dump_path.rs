//! Dump the alignment the way the frame-level diff wants it.
//!
//! `D:/alignment_gold` compares this port against the Python reference one
//! CHARACTER span at a time, and a one- or two-frame difference in the output
//! is not interpretable on its own: it could be a wrong boundary, or the right
//! boundary taken over the wrong blank run. So this writes the blank runs too,
//! and the raw path, not just the spans.
//!
//! `examples/` rather than a `--flag` on the CLI: nothing here is a feature a
//! caller would ever want, and everything it prints is already in
//! `AlignOutput` as a `#[serde(skip)]` diagnostic field.
//!
//! ```text
//! dump_path <wav> <model-dir> <out.json> [device-spec]
//! ```

use ctc_forced_aligner_wgpu::{Aligner, DeviceSelector};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let mut a = std::env::args().skip(1);
    let (Some(wav), Some(model), Some(out)) = (a.next(), a.next(), a.next()) else {
        eprintln!("usage: dump_path <wav> <model-dir> <out.json> [device-spec]");
        std::process::exit(2);
    };
    let dev = DeviceSelector::parse(&a.next().unwrap_or_else(|| "auto".into()))?;

    let aligner = Aligner::load_on(Path::new(&model), dev)?;
    let out_path = Path::new(&out);
    let transcript = std::fs::read_to_string(out_path.with_extension("txt"))?;
    let out = aligner.align_with_path(Path::new(&wav), transcript.trim(), Some(30.0), 2.0)?;

    // The path is a sequence of EXPANDED state indices: even is blank, odd is
    // token `(state - 1) / 2`. `-1` is padding, which never won a frame.
    let token_per_frame: Vec<i32> = out
        .frame_path
        .iter()
        .map(|&s| if s % 2 == 1 && s >= 1 { (s - 1) / 2 } else { -1 })
        .collect();

    // One row per SOURCE WORD, grouped by the tokenizer's `word_id` -- the same
    // grouping the reference's `get_spans` produces, and the granularity the
    // frame diff compares at when the star placement is the variable.
    let mut word_spans: Vec<(usize, String, i64, i64)> = Vec::new();
    for t in out.tokens.iter().filter(|t| t.piece != "<star>") {
        match word_spans.last_mut() {
            Some(w) if w.0 == t.word_id => {
                w.1.push_str(&t.piece);
                w.3 = t.end_frame;
            }
            _ => word_spans.push((t.word_id, t.piece.clone(), t.start_frame, t.end_frame)),
        }
    }

    let doc = serde_json::json!({
        "wav": wav,
        "frames": out.frames,
        "n_frames": out.frames,
        "frame_rate": out.frame_rate,
        "pieces": out.tokens.iter().map(|t| &t.piece).collect::<Vec<_>>(),
        "char_spans": out.tokens.iter().map(|t| serde_json::json!({
            "piece": t.piece,
            "start_frame": t.start_frame,
            "end_frame": t.end_frame,
        })).collect::<Vec<_>>(),
        "word_spans": word_spans.iter().map(|(_, text, a, b)| serde_json::json!({
            "text": text,
            "start_frame": a,
            "end_frame": b,
        })).collect::<Vec<_>>(),
        "token_per_frame": token_per_frame,
        "blank_runs": out.blank_runs.iter().map(|(i, a, b)| serde_json::json!({
            "before_token": i,
            "first": a,
            "last": b,
            "len": b - a + 1,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(out_path, serde_json::to_string(&doc)?)?;
    eprintln!(
        "{}: {} frames, {} tokens, {} blank runs -> {}",
        out_path.display(),
        out.frames,
        out.tokens.len(),
        out.blank_runs.len(),
        out_path.display()
    );
    Ok(())
}
