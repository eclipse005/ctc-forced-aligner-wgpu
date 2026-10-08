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

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::{Context, Result};
use ctc_forced_aligner_wgpu::align_inference::{AlignOutput, Aligner};
use ctc_forced_aligner_wgpu::render::{ass, srt};
use ctc_forced_aligner_wgpu::views;
use ctc_forced_aligner_wgpu::{Backend, Progress};

const USAGE: &str = "\
usage: align --audio <wav> (--text <text|file>) [options]

  --model <dir>            model directory (default: $CTC_MODEL_DIR or the bundled checkpoint)
  --window <sec>           memory/throughput only; it does not move timestamps (default: 30)
  --context <sec>          encoder context each side of a window, at least 1.3 (default: 2)
  --device <spec>          auto (default: gpu, or cpu when no gpu opens), cpu, gpu (error if no gpu),
                           vulkan[:i], dx12[:i], #n, or a name substring
  --format <json|srt|ass>
                           json is the full alignment (default); srt is subtitles,
                           ass is karaoke (one sweep per character)
  --ass-res <WxH>          ass only: the VIDEO's resolution (default 1920x1080).
                           The aligner only sees a waveform and cannot know it
  --ass-font <name>        ass only: font family (default Malgun Gothic); size is fixed at 64
  --output <path>          write here (default: print to stdout)
  --list-devices           list wgpu adapters and exit

Progress goes to stderr, never to stdout: one rewriting line on a terminal, or a
line every 10% when stderr is redirected to a file. The line reads
\"pct%  done/total  stage\" over the WHOLE run, so it reaches 100% only when
the alignment is finished — the windows count as they come back from the
device, not as they are queued. stdout stays byte-exactly what --format asked
for.
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
    let mut karaoke_font: Option<String> = None;
    let mut ass_res: Option<String> = None;

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
            "--ass-font" => karaoke_font = Some(it.next().context("--ass-font needs a value")?),
            "--ass-res" => ass_res = Some(it.next().context("--ass-res needs a value")?),
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
    let backend = Backend::parse(&device)?;
    // The weights are a fixed few seconds and they happen before `align` can
    // hand out a sink, so without this line the first thing a user sees is a
    // bar appearing at 10% — which reads as "it had already done 10%".
    eprintln!("[align] loading model from {}", model_dir.display());
    let aligner = Aligner::load_with(&model_dir, backend)
        .with_context(|| format!("load model from {}", model_dir.display()))?;
    let load_s = t0.elapsed().as_secs_f64();

    let out = {
        let ticker = Ticker::new();
        aligner.align(&audio, &text, window, context, Some(&|p| ticker.tick(p)))?
    };
    let rtfx = out.duration / (out.encode_s + out.align_s).max(1e-9);

    // The karaoke style carries only what a user might genuinely need to
    // change; everything else has a default that reads on normal footage.
    let mut karaoke = ass::KaraokeStyle {
        title: output
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Karaoke".to_string()),
        ..ass::KaraokeStyle::default()
    };
    if let Some(f) = karaoke_font {
        karaoke.font = f;
    }
    if let Some(res) = &ass_res {
        let mut it = res.splitn(2, 'x');
        let w: i64 = it.next().and_then(|s| s.trim().parse().ok()).context("--ass-res wants WxH, e.g. 1280x720")?;
        let h: i64 = it.next().and_then(|s| s.trim().parse().ok()).context("--ass-res wants WxH, e.g. 1280x720")?;
        karaoke.set_play_res(w, h);
    }
    let body = render(&out, &format, &karaoke)?;
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
        "{}: {:.2}s audio, {} tokens | encode {:.3}s, align {:.3}s, RTFx {:.1}x | load {:.1}s | {}{}",
        audio.display(),
        out.duration,
        // the transcript's characters, which is what the JSON's `tokens` array
        // holds; the `<star>` targets are not part of it
        out.tokens.iter().filter(|t| t.piece != "<star>").count(),
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

/// stderr 上的进度条，分两种接收端：
///
/// * **终端**里一行到底，用 `\r` 覆盖 —— 这是人盯着看的那种。
/// * **重定向到文件/管道**时按 10% 打点成独立行 —— `\r` 写进日志就是一串控制符，
///   而每个刻度打一行在日志里是一长串几乎一样的字。两个都不想要。
///
/// 打的是 `pct%  done/total  stage`：`pct` 是整条 run 的百分比（不是某个阶段的），
/// `stage` 说明这一刻在干什么 —— `viterbi` 那一段在 CPU 路径上是编码全部结束之后
/// 才开始的，没有阶段标签的话那段时间看上去像卡死了。
struct Ticker {
    tty: bool,
    last_pct: AtomicU8,
}

/// 非终端时每隔多少个百分点打一行。
const TICK_STEP: u8 = 10;

impl Ticker {
    fn new() -> Self {
        Self::with_tty(std::io::stderr().is_terminal())
    }

    fn with_tty(tty: bool) -> Self {
        Self { tty, last_pct: AtomicU8::new(0) }
    }

    /// 这一格要不要写；返回要写的百分比。终端恒为 `Some`（每次都重写一行），
    /// 非终端才按 [`TICK_STEP`] 打点。与打印分开，是为了能在没有终端的单测里
    /// 断言「该打几次、什么时候打」——`\r` 分支在管道里永远走不到。
    fn due(&self, done: usize, total: usize) -> Option<u8> {
        if total == 0 {
            return None;
        }
        let pct = ((done.min(total) * 100) / total).min(100) as u8;
        if self.tty {
            return Some(pct);
        }
        if pct == 100 || pct >= self.last_pct.load(Ordering::Relaxed) + TICK_STEP {
            self.last_pct.store(pct, Ordering::Relaxed);
            return Some(pct);
        }
        None
    }

    fn tick(&self, p: Progress) {
        let Some(pct) = self.due(p.done, p.total) else {
            return;
        };
        // 行尾多两个空格：百分比 9→100 变宽，不补的话上一次会露在后面。
        let line = format!(
            "[align] {pct:>3}%  {}/{}  {}   ",
            p.done.min(p.total),
            p.total,
            p.stage.label()
        );
        if self.tty {
            eprint!("\r{line}");
            if pct == 100 {
                eprintln!();
            }
        } else {
            eprintln!("[align] {pct:>3}%  {}/{}  {}", p.done.min(p.total), p.total, p.stage.label());
        }
    }
}

fn render(out: &AlignOutput, format: &str, karaoke: &ass::KaraokeStyle) -> Result<String> {
    match format {
        // Compact, not pretty. This is a machine-read format and a 73-minute
        // Japanese transcript spent 6.1 MB of its 16.4 on indentation alone --
        // 37% of the file, saying nothing. An editor can re-indent it.
        "json" => Ok(serde_json::to_string(out)?),
        "srt" => Ok(srt::cues_to_srt(&views::build_cues(&out.tokens))),
        "ass" => {
            let doc = views::build_cues(&out.tokens);
            Ok(ass::cues_to_karaoke(&doc, &out.tokens, karaoke))
        }
        other => anyhow::bail!("unknown --format {other:?}; expected json, srt, or ass"),
    }
}

fn default_model_dir() -> PathBuf {
    std::env::var_os("CTC_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf"))
}

#[cfg(test)]
mod tests {
    use super::{TICK_STEP, Ticker};
    use ctc_forced_aligner_wgpu::{Progress, Stage};

    /// 一次运行里实际会打出的每一格的百分比（非终端）。
    fn ticks(total: usize) -> Vec<u8> {
        let t = Ticker::with_tty(false);
        (1..=total).filter_map(|done| t.due(done, total)).collect()
    }

    /// 引擎报上来的 done 偶尔会越界（换了分母、旧值还在飞），夹住而不是画出
    /// 一个 140% 的条。
    #[test]
    fn a_done_past_the_total_clamps_instead_of_overshooting() {
        let t = Ticker::with_tty(false);
        assert_eq!(t.due(9, 3), Some(100));
    }

    /// 日志里应该是十来行，不是每个刻度一行几乎一样的字。
    #[test]
    fn a_redirected_run_reports_every_ten_percent() {
        let got = ticks(42);
        assert_eq!(got.first().copied(), Some(11));
        assert_eq!(*got.last().unwrap(), 100);
        // 每一步都至少跨了一个 TICK_STEP，且严格递增。
        assert!(
            got.windows(2).all(|w| w[1] > w[0]),
            "progress went backwards or repeated: {got:?}"
        );
        assert!(
            got.windows(2).all(|w| w[1] - w[0] >= TICK_STEP),
            "a step under {TICK_STEP}% slipped through: {got:?}"
        );
    }

    /// 终端里每一格都要重写，所以每一格都给 Some。
    #[test]
    fn a_terminal_rewrites_on_every_checkpoint() {
        let t = Ticker::with_tty(true);
        let got: Vec<Option<u8>> = (1..=42).map(|d| t.due(d, 42)).collect();
        assert_eq!(got.len(), 42, "the tty branch must never skip a tick");
        assert!(got.iter().all(|p| p.is_some()));
        assert_eq!(*got.last().unwrap(), Some(100));
    }

    /// 一条只有几个刻度的 run（比如 `--window 0`）不该只剩首尾两行——10% 的步长
    /// 比整段还粗。这种 run 的 `done` 也会很小，引擎照实报，这里照实打。
    #[test]
    fn a_short_run_reports_every_checkpoint() {
        assert_eq!(ticks(3), vec![33, 66, 100]);
    }

    /// 没有分母不是画一条永远停在 0% 的线，而是一个刻度都不发。
    #[test]
    fn no_denominator_is_no_progress_at_all() {
        assert_eq!(Ticker::with_tty(false).due(0, 0), None);
        assert_eq!(Ticker::with_tty(true).due(0, 0), None);
    }

    /// `Progress::pct()` 与 Ticker 自己算的是同一个数：库给百分比、CLI 也只
    /// 认百分比，两边不能各算一套。
    #[test]
    fn the_ticker_and_the_library_agree_on_the_percentage() {
        let t = Ticker::with_tty(true);
        for (done, total) in [(0, 42usize), (1, 42), (7, 42), (31, 31), (5, 9), (9, 3)] {
            let p = Progress { stage: Stage::Encode, done, total };
            assert_eq!(
                t.due(p.done, p.total),
                Some(p.pct()),
                "disagreed at {done}/{total}"
            );
        }
    }
}
