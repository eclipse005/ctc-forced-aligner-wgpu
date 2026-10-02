//! CLI: align one audio file against its transcript.
//!
//! `--format json` (the default) is the tight character alignment.
//! `--format spans` tiles silence onto the neighbouring chunks.
//! `--format srt` / `cues` are subtitle breaks on the tight timestamps.
//!
//! The global allocator is the counting wrapper in `alloc_stats`, but only
//! with `--features alloc-stats` (the counters are three relaxed atomics per
//! allocation).  Build that way and set `CTC_ALLOC_STATS=1` to get the run's
//! live set and high-water mark per chunk on stderr; the forward pass makes
//! ~1e3 allocations per chunk, so the counters are noise.  A default build
//! forwards every call straight to the system allocator.

#[global_allocator]
static GLOBAL: ctc_forced_aligner_wgpu::alloc_stats::Stats =
    ctc_forced_aligner_wgpu::alloc_stats::Stats;

use std::path::PathBuf;

use anyhow::{Context, Result};
use ctc_forced_aligner_wgpu::align_inference::{AlignOutput, Aligner};
use ctc_forced_aligner_wgpu::views;
use ctc_forced_aligner_wgpu::DeviceSelector;

const USAGE: &str = "\
usage: align --audio <wav> (--text <text|file>) [options]

  --model <dir>            model directory (default: $CTC_MODEL_DIR or the bundled checkpoint)
  --window <sec>           memory/throughput only; it does not move timestamps (default: 30)
  --context <sec>          encoder context each side of a window, at least 1.3 (default: 2)
  --device <spec>          one device: auto (default, cpu if no gpu), cpu, vulkan[:i], dx12[:i], #n, or a name substring
  --format <json|srt|cues> json is the full alignment (default); srt and cues are subtitles
  --output <path>          write here (default: print to stdout)
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
    let selector = DeviceSelector::parse(&device)?;
    let aligner = Aligner::load_on(&model_dir, selector)
        .with_context(|| format!("load model from {}", model_dir.display()))?;
    let load_s = t0.elapsed().as_secs_f64();

    let out = aligner.align(&audio, &text, window, context)?;
    let rtfx = out.duration / (out.encode_s + out.align_s).max(1e-9);

    let body = render(&out, &format)?;
    match &output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(path, &body)?;
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

fn render(out: &AlignOutput, format: &str) -> Result<String> {
    match format {
        "json" => Ok(serde_json::to_string_pretty(out)?),
        "srt" | "cues" => {
            let doc = views::build_cues(&out.tokens);
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
                    "script": doc.script,
                    "cues": cues,
                }))?)
            }
        }
        other => anyhow::bail!("unknown --format {other:?}; expected json, srt, or cues"),
    }
}

fn default_model_dir() -> PathBuf {
    std::env::var_os("CTC_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf"))
}
