//! Per-word timing metadata for the gold-convention analysis.
//!
//! argv: <jobs.jsonl> <model-dir> <out.jsonl> [device-spec]
//!
//! For every source word: the word's own timing plus the blank run that sits
//! immediately before its first token, so the caller can ask where a gold
//! boundary falls INSIDE that run. Writes one JSON object per job.

use anyhow::Context;
use ctc_forced_aligner_wgpu::{Aligner, DeviceSelector};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(jobs_path), Some(model_dir)) = (args.next(), args.next()) else {
        eprintln!("usage: dump_word_runs <jobs.jsonl> <model-dir> <out.jsonl> [device]");
        std::process::exit(2);
    };
    let out_path = args.next().context("missing out path")?;
    let device = args.next().unwrap_or_else(|| "cpu".into());

    let aligner = Aligner::load_on(Path::new(&model_dir), DeviceSelector::parse(&device)?)?;
    let mut out = BufWriter::new(std::fs::File::create(&out_path)?);

    for line in BufReader::new(std::fs::File::open(&jobs_path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let job: serde_json::Value = serde_json::from_str(&line)?;
        let item_id = job["item_id"].as_str().unwrap_or_default().to_string();
        let reply = match align_job(&aligner, &job) {
            Ok(v) => serde_json::json!({ "item_id": item_id, "words": v.0, "meta": v.1 }),
            Err(e) => serde_json::json!({ "item_id": item_id, "error": format!("{e:#}") }),
        };
        writeln!(out, "{reply}")?;
    }
    out.flush()?;
    Ok(())
}

fn align_job(
    aligner: &Aligner,
    job: &serde_json::Value,
) -> anyhow::Result<(Vec<serde_json::Value>, Vec<serde_json::Value>)> {
    let audio = Path::new(job["audio_path"].as_str().context("no audio_path")?);
    let text = job["transcript"].as_str().context("no transcript")?;
    let o = aligner.align_with_path(audio, text, Some(30.0), 2.0)?;

    // Words grouped by word_id (stars dropped), remembering each word's first
    // token index; then the blank run whose `before_token_index` is that index.
    let mut words: Vec<(String, f64, f64, usize, usize)> = Vec::new();
    for t in &o.tokens {
        if t.piece == "<star>" {
            continue;
        }
        if matches!(words.last(), Some(w) if w.4 == t.word_id) {
            let w = words.last_mut().expect("checked non-empty");
            w.0.push_str(&t.piece);
            w.2 = t.end;
        } else {
            words.push((t.piece.clone(), t.start, t.end, t.index, t.word_id));
        }
    }
    let rows: Vec<serde_json::Value> = words
        .iter()
        .map(|(text, s, e, _, _)| serde_json::json!([text, s, e]))
        .collect();
    let meta: Vec<serde_json::Value> = words
        .iter()
        .map(|(_, _, _, first_idx, _)| {
            let run = o
                .blank_runs
                .iter()
                .find(|(idx, _, _)| idx == first_idx)
                .map(|(_, a, b)| serde_json::json!([a, b]));
            serde_json::json!({ "first_token": first_idx, "blank_before": run })
        })
        .collect();
    Ok((rows, meta))
}
