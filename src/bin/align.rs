//! CLI: align one audio file against its transcript.
//!
//! `--format json` (the default) is the tight character alignment.
//! `--format spans` tiles silence onto the neighbouring chunks.
//! `--format srt` / `cues` are subtitle breaks on the tight timestamps.

use std::path::PathBuf;

use anyhow::{Context, Result};
use ctc_forced_aligner_wgpu::align_inference::{AlignOutput, Aligner};
use ctc_forced_aligner_wgpu::views::{self, resolve_split};
use ctc_forced_aligner_wgpu::DeviceSelector;

const USAGE: &str = "\
usage: align --audio <wav> (--text <text|file>) [options]

  --model <dir>            model directory (default: $CTC_MODEL_DIR or the bundled checkpoint)
  --window <sec>           windowed encoding (default: 30; 0 = whole file)
  --context <sec>          context on each side of a window (default: 2)
  --device <spec>          auto (default), cpu, vulkan[:i], dx12[:i], #n, or a name substring
  --format <json|spans|srt|cues>
                           json is the tight alignment (default)
  --split <word|char|sentence>
                           spans only; default is char when the text is mostly CJK
  --preset <short|standard|loose>
                           subtitle preset (default: standard)
  --merge-threshold <sec>  spans: snap a gap shorter than this (default: 0)
  --output <path>          write here (default: print to stdout). spans also writes a .txt sidecar
  --list-devices           list wgpu adapters and exit
";

fn main() -> Result<()> {
    let mut audio: Option<PathBuf> = None;
    let mut text: Option<String> = None;
    let mut model_dir = default_model_dir();
    let mut window: Option<f64> = Some(30.0);
    let mut context: f64 = 2.0;
    let mut device = String::from("auto");
    let mut format = String::from("json");
    let mut split: Option<String> = None;
    let mut preset = String::from("standard");
    let mut merge_threshold: f64 = 0.0;
    let mut output: Option<PathBuf> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--audio" => audio = Some(PathBuf::from(it.next().context("--audio needs a value")?)),
            "--text" => text = Some(it.next().context("--text needs a value")?),
            "--model" => model_dir = PathBuf::from(it.next().context("--model needs a value")?),
            "--window" => {
                let w: f64 = it
                    .next()
                    .context("--window needs a value")?
                    .parse()
                    .context("--window must be a number of seconds")?;
                window = if w > 0.0 { Some(w) } else { None };
            }
            "--context" => {
                context = it
                    .next()
                    .context("--context needs a value")?
                    .parse()
                    .context("--context must be a number of seconds")?;
            }
            "--format" => format = it.next().context("--format needs a value")?,
            "--split" => split = Some(it.next().context("--split needs a value")?),
            "--preset" => preset = it.next().context("--preset needs a value")?,
            "--merge-threshold" => {
                merge_threshold = it
                    .next()
                    .context("--merge-threshold needs a value")?
                    .parse()
                    .context("--merge-threshold must be a number of seconds")?;
            }
            "--device" => device = it.next().context("--device needs a value")?,
            "--output" | "-o" => {
                output = Some(PathBuf::from(it.next().context("--output needs a value")?))
            }
            "--list-devices" => {
                for line in pollster::block_on(ctc_forced_aligner_wgpu::list_targets()) {
                    println!("{line}");
                }
                return Ok(());
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other:?}\n{USAGE}"),
        }
    }

    let audio = audio.context("--audio is required")?;
    let text_arg = text.context("--text is required (transcript text, or a path to it)")?;
    let text = if std::path::Path::new(&text_arg).is_file() {
        std::fs::read_to_string(&text_arg)
            .with_context(|| format!("read transcript {}", text_arg))?
    } else if text_arg.contains('\\') || text_arg.contains('/') || text_arg.ends_with(".txt") {
        anyhow::bail!("transcript file not found: {text_arg}");
    } else {
        text_arg
    };

    let t0 = std::time::Instant::now();
    let aligner = if device.eq_ignore_ascii_case("dual") {
        Aligner::load_dual(&model_dir)
    } else {
        let selector = DeviceSelector::parse(&device)?;
        Aligner::load_on(&model_dir, selector)
    }
    .with_context(|| format!("load model from {}", model_dir.display()))?;
    let load_s = t0.elapsed().as_secs_f64();

    let out = aligner.align(&audio, &text, window, context)?;
    let rtfx = out.duration / (out.encode_s + out.align_s).max(1e-9);

    let body = render(&out, &format, split.as_deref(), &preset, merge_threshold)?;
    match &output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(path, &body)?;
            if format == "spans" {
                let split_name = resolve_split(&out.text, split.as_deref());
                let spans = views::build_spans(
                    &out.text,
                    &out.tokens,
                    out.frames,
                    out.frame_rate,
                    &out.frame_scores,
                    &split_name,
                    merge_threshold,
                );
                let txt = path.with_extension("txt");
                std::fs::write(&txt, views::spans_to_txt(&spans))?;
            }
        }
        None => println!("{body}"),
    }
    eprintln!(
        "{}: {:.2}s audio, {} chars | encode {:.3}s, align {:.3}s, RTFx {:.1}x | load {:.1}s | {}{}",
        audio.display(),
        out.duration,
        out.chars.len(),
        out.encode_s,
        out.align_s,
        rtfx,
        load_s,
        aligner.device_desc(),
        output
            .as_ref()
            .map(|p| format!(" | -> {}", p.display()))
            .unwrap_or_default()
    );
    Ok(())
}

fn render(
    out: &AlignOutput,
    format: &str,
    split: Option<&str>,
    preset: &str,
    merge_threshold: f64,
) -> Result<String> {
    match format {
        "json" => Ok(serde_json::to_string_pretty(out)?),
        "spans" => {
            if let Some(s) = split {
                if !matches!(s, "word" | "char" | "sentence" | "auto") {
                    anyhow::bail!("--split must be word, char, or sentence");
                }
            }
            let split_name = resolve_split(&out.text, split);
            let spans = views::build_spans(
                &out.text,
                &out.tokens,
                out.frames,
                out.frame_rate,
                &out.frame_scores,
                &split_name,
                merge_threshold,
            );
            let segments: Vec<_> = spans
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "score": s.score,
                    })
                })
                .collect();
            Ok(serde_json::to_string_pretty(&serde_json::json!({
                "text": out.text,
                "segments": segments,
            }))?)
        }
        "srt" | "cues" => {
            if !matches!(preset, "short" | "standard" | "loose") {
                anyhow::bail!("--preset must be short, standard, or loose");
            }
            let doc = views::build_cues(&out.tokens, preset);
            if format == "srt" {
                Ok(views::cues_to_srt(&doc))
            } else {
                let cues: Vec<_> = doc
                    .cues
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "index": c.index,
                            "start": c.start,
                            "end": c.end,
                            "text": c.text,
                        })
                    })
                    .collect();
                Ok(serde_json::to_string_pretty(&serde_json::json!({
                    "preset": doc.preset,
                    "script": doc.script,
                    "cues": cues,
                }))?)
            }
        }
        other => anyhow::bail!("unknown --format {other:?}; expected json, spans, srt, or cues"),
    }
}

fn default_model_dir() -> PathBuf {
    std::env::var_os("CTC_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf"))
}
