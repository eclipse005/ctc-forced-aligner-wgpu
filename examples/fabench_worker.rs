//! FA-Bench worker: one process, one model load, JSONL in and out.
//!
//! Protocol (fa-bench `SubprocessAligner`, the same one the Python workers
//! speak):
//!
//! ```text
//! argv    <jobs.jsonl> <model-dir> [--device D]
//! in      {"item_id": ..., "audio_path": ..., "transcript": ...}   per line
//! out     {"item_id": ..., "words": [[text, start_s, end_s, conf], ...]}
//!         {"item_id": ..., "error": "..."}                         per line
//! ```
//!
//! Word tier: the aligner emits one token per CTC target; they are regrouped
//! by the tokenizer's `word_id` -- the same grouping `spans::build_words`
//! applies, and the granularity the Buckeye hand marks are scored at. The
//! confidence is the mean of the word's characters' frame scores, which is
//! how the Python baseline worker computes it.

use anyhow::Context;
use ctc_forced_aligner_wgpu::{Aligner, DeviceSelector};
use std::io::{BufRead, Write};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(jobs_path), Some(model_dir)) = (args.next(), args.next()) else {
        eprintln!("usage: fabench_worker <jobs.jsonl> <model-dir> [--device D]");
        std::process::exit(2);
    };
    let mut device = String::from("cpu");
    while let (Some(flag), Some(val)) = (args.next(), args.next()) {
        if flag == "--device" {
            device = val;
        }
    }

    let aligner = Aligner::load_on(Path::new(&model_dir), DeviceSelector::parse(&device)?)?;

    let jobs = std::fs::File::open(&jobs_path)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in std::io::BufReader::new(jobs).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let job: serde_json::Value = serde_json::from_str(&line)?;
        let item_id = job["item_id"].as_str().unwrap_or_default().to_string();
        let reply = match align_job(&aligner, &job) {
            Ok(words) => serde_json::json!({ "item_id": item_id, "words": words }),
            Err(e) => serde_json::json!({ "item_id": item_id, "error": format!("{e:#}") }),
        };
        writeln!(out, "{reply}")?;
    }
    Ok(())
}

fn align_job(
    aligner: &Aligner,
    job: &serde_json::Value,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let audio = Path::new(
        job["audio_path"]
            .as_str()
            .context("job has no audio_path")?,
    );
    let text = job["transcript"].as_str().context("job has no transcript")?;
    let o = aligner.align_with_path(audio, text, None, 2.0, None)?;

    // Regroup the CTC targets into source words by `word_id`, dropping the
    // stars: word text, first start, last end, mean frame score.
    let mut words: Vec<(String, f64, f64, f64, i32, usize)> = Vec::new();
    for t in &o.tokens {
        if t.piece == "<star>" {
            continue;
        }
        if matches!(words.last(), Some(w) if w.5 == t.word_id) {
            let w = words.last_mut().expect("checked non-empty");
            w.0.push_str(&t.piece);
            w.2 = t.end;
            if t.score.is_finite() {
                w.3 += t.score;
                w.4 += 1;
            }
        } else {
            words.push((
                t.piece.clone(),
                t.start,
                t.end,
                if t.score.is_finite() { t.score } else { 0.0 },
                if t.score.is_finite() { 1 } else { 0 },
                t.word_id,
            ));
        }
    }
    Ok(words
        .iter()
        .map(|(text, s, e, sum, n, _)| {
            serde_json::json!([
                text,
                s,
                e,
                if *n > 0 { sum / *n as f64 } else { -100.0 },
            ])
        })
        .collect())
}
