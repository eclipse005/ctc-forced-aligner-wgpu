//! The aligner's forward: (audio, text) → character / word / segment
//! timestamps.  Orchestration port of `omni_align/aligner.py` +
//! `backend.log_probs_chunked`, producing the same JSON schema.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::audio::{load_audio, znorm, TARGET_SR};
use crate::fix_timestamp::fix_timestamp;
use crate::config::Wav2Vec2Config;
use crate::gpu::DeviceSelector;
use crate::spans::{build_segments, build_words};
use crate::viterbi::{
    build_expanded_labels, ctc_forced_align_emissions, ctc_forced_align_gathered_chunks,
    Emissions, GatheredChunks, TokenAlignment,
};
use crate::views::skipped_chars;
use crate::vocab::Vocab;
use crate::wav2vec2::Model;
use crate::wav2vec2_gpu::GpuModel;

/// The model on one backend. Both towers are token-identical to the
/// Python reference (see tests/golden.rs).
pub enum Tower {
    Cpu(Model),
    Gpu(GpuModel),
}

pub const FRAME_RATE: f64 = 50.0;
pub const BLANK_ID: usize = 0;
const SUBSAMPLING: usize = 320; // samples per frame

pub struct Aligner {
    pub tower: Tower,
    pub vocab: Vocab,
    pub model_dir: PathBuf,
}

/// One output JSON, mirroring `omni_align.cli.to_json_obj` field for field.
#[derive(serde::Serialize)]
pub struct AlignOutput {
    pub audio: String,
    pub text: String,
    pub duration: f64,
    pub frames: usize,
    pub frame_rate: f64,
    pub mean_frame_score: f64,
    pub log_prob: f64,
    /// Transcript characters the vocabulary dropped, in order.
    pub skipped: Vec<String>,
    pub chars: Vec<serde_json::Value>,
    pub words: Vec<serde_json::Value>,
    pub segments: Vec<serde_json::Value>,
    /// Encoder time. Omitted from JSON so the schema stays the Python one.
    #[serde(skip)]
    pub encode_s: f64,
    /// Viterbi time. Omitted from JSON for the same reason.
    #[serde(skip)]
    pub align_s: f64,
    /// Tight character alignments. Not serialized; `chars` is the JSON form.
    #[serde(skip)]
    pub tokens: Vec<TokenAlignment>,
    /// Winning-label log-prob per frame, including blanks. Used by spans.
    #[serde(skip)]
    pub frame_scores: Vec<f64>,
}

fn r4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

impl Aligner {
    /// CPU backend (the reference twin).
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_on(model_dir, DeviceSelector::Cpu)
    }

    /// One backend. `Auto` uses a single wgpu GPU, or the CPU tower when none is present.
    pub fn load_on(model_dir: &Path, selector: DeviceSelector) -> Result<Self> {
        let tower = match selector {
            DeviceSelector::Cpu => Tower::Cpu(Model::load(model_dir)?),
            DeviceSelector::Auto => match GpuModel::load(model_dir, DeviceSelector::Auto) {
                Ok(gpu) => Tower::Gpu(gpu),
                Err(e) if e.downcast_ref::<crate::gpu::NoGpuError>().is_some() => {
                    Tower::Cpu(Model::load(model_dir)?)
                }
                Err(e) => return Err(e),
            },
            other => Tower::Gpu(GpuModel::load(model_dir, other)?),
        };
        let vocab = Vocab::load(model_dir)?;
        Ok(Self { tower, vocab, model_dir: model_dir.to_path_buf() })
    }

    pub fn backend_name(&self) -> &'static str {
        match self.tower {
            Tower::Cpu(_) => "cpu",
            Tower::Gpu(_) => "wgpu",
        }
    }

    pub fn device_desc(&self) -> String {
        match &self.tower {
            Tower::Cpu(_) => "cpu".to_string(),
            Tower::Gpu(g) => g.describe(),
        }
    }

    pub fn config(&self) -> &Wav2Vec2Config {
        match &self.tower {
            Tower::Cpu(m) => &m.cfg,
            Tower::Gpu(g) => &g.cfg,
        }
    }

    /// Public forward for tests/benchmarks: z-normalised waveform -> log_probs.
    pub fn forward_pub(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.forward(input)
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.forward_scratch(input, &mut crate::wav2vec2::Scratch::default())
    }

    /// Same, reusing the CPU tower's scratch buffers across calls.
    fn forward_scratch(
        &self,
        input: &[f32],
        scratch: &mut crate::wav2vec2::Scratch,
    ) -> Result<Vec<f32>> {
        match &self.tower {
            Tower::Cpu(m) => Ok(m.forward_with(input, &Default::default(), scratch)?.0),
            Tower::Gpu(g) => g.forward(input),
        }
    }

    /// Align one file against its transcript.
    ///
    /// `window_sec = None` encodes the whole file in one pass (matches the
    /// Python unchunked path); `Some(w)` uses w-second windows with
    /// `context_sec` of context on each side (matches `log_probs_chunked`).
    pub fn align(
        &self,
        audio_path: &Path,
        text: &str,
        window_sec: Option<f64>,
        context_sec: f64,
    ) -> Result<AlignOutput> {
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let (ids, pieces) = self.vocab.tokenise(&text);
        if ids.is_empty() {
            anyhow::bail!("no in-vocabulary characters in the transcript");
        }

        let (waveform, sr) = load_audio(audio_path)
            .with_context(|| format!("load {}", audio_path.display()))?;
        anyhow::ensure!(sr == TARGET_SR, "expected {TARGET_SR} Hz after decoding");
        let duration = waveform.len() as f64 / sr as f64;

        // Both towers gather the trellis labels' log-probs (T, S) instead of
        // handing the whole (T, V) matrix to the CPU — the GPU readback of a
        // 34 s chunk drops from ~70 MB to ~30 KB.  The gathered values are
        // the same f32s the full path would read, so the DP is unchanged.
        let expanded = build_expanded_labels(&ids, BLANK_ID);
        let t_enc = std::time::Instant::now();
        let trellis = self.log_probs_trellis(&waveform, window_sec, context_sec, &expanded)?;
        let encode_s = t_enc.elapsed().as_secs_f64();
        crate::wav2vec2::prof::dump(&format!(
            "{} [{}]",
            audio_path.display(),
            self.backend_name()
        ));
        let t_al = std::time::Instant::now();
        let mut res = match trellis {
            Trellis::Gathered(gathered) => {
                ctc_forced_align_gathered_chunks(&gathered, &ids, FRAME_RATE, Some(&pieces))?
            }
            Trellis::Logits(blocks) => {
                let gather = match &self.tower {
                    Tower::Cpu(m) => RowGather::Cpu(m),
                    Tower::Gpu(_) => RowGather::Gpu,
                };
                let em = LogitsEmissions::new(&blocks, gather, &expanded);
                let frames = em.total_frames();
                em.validate()?;
                ctc_forced_align_emissions(&em, frames, &ids, FRAME_RATE, Some(&pieces))?
            }
        };
        let align_s = t_al.elapsed().as_secs_f64();
        if crate::alloc_stats::enabled() {
            // the per-chunk prints stop at the end of the encode; the DP's
            // backpointers are allocated after that, so the run's real high
            // water mark only shows here
            let (live, peak) = crate::alloc_stats::stats();
            eprintln!(
                "[alloc] after dp: live {live:>12} peak {peak:>12}  ({} states x {} frames)",
                expanded.len(),
                res.frames
            );
        }

        fix_timestamp(&mut res.tokens);
        let words = build_words(&res.tokens);
        let segments = build_segments(&res.tokens, &words);
        let skipped = skipped_chars(&text, &pieces);
        let chars: Vec<serde_json::Value> = res
            .tokens
            .iter()
            .map(|t| {
                serde_json::json!({
                    "index": t.index,
                    "token_id": t.token_id,
                    "piece": t.piece,
                    "start": r4(t.start),
                    "end": r4(t.end),
                    "mid": r4(t.mid()),
                    "duration": r4(t.duration()),
                    "start_frame": t.start_frame,
                    "end_frame": t.end_frame,
                    "score": r4(t.score),
                })
            })
            .collect();

        Ok(AlignOutput {
            audio: audio_path.display().to_string(),
            text,
            duration: r4(duration),
            frames: res.frames,
            frame_rate: res.frame_rate,
            mean_frame_score: r4(res.mean_frame_score()),
            log_prob: r4(res.log_prob),
            skipped,
            chars,
            words: words
                .iter()
                .map(|w| {
                    serde_json::json!({
                        "index": w.index,
                        "text": w.text,
                        "start": r4(w.start),
                        "end": r4(w.end),
                        "duration": r4(w.duration()),
                        "char_start": w.char_start,
                        "char_end": w.char_end,
                    })
                })
                .collect(),
            segments: segments
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "index": s.index,
                        "text": s.text,
                        "start": r4(s.start),
                        "end": r4(s.end),
                        "duration": r4(s.end - s.start),
                        "words": s.words,
                    })
                })
                .collect(),
            encode_s,
            align_s,
            tokens: res.tokens,
            frame_scores: res.frame_scores,
        })
    }

    /// Trellis-label log-probabilities for a raw (un-normalised) waveform,
    /// in whichever of the two equivalent storage forms is smaller.
    fn log_probs_trellis(
        &self,
        waveform: &[f32],
        window_sec: Option<f64>,
        context_sec: f64,
        expanded: &[usize],
    ) -> Result<Trellis> {
        // The DP reads one f32 per (frame, expanded state), so the trellis is
        // T×(2·tokens+1).  The alternative is the lm head's T×vocab output,
        // and whichever is narrower wins: a short transcript's trellis
        // (3 m: 10500×6533) is well under the 10288-wide log-prob block, a
        // long one (15 m: 43500×31130 = 5.4 GB) is 2.8× over it.  Both hold
        // the identical f32 values, so this is a storage decision only.
        // `CTC_TRELLIS=auto|gathered|logits` forces either side of it.
        let vocab = self.vocab_size();
        let use_logits = match std::env::var("CTC_TRELLIS").ok().as_deref() {
            Some("gathered") => false,
            Some("logits") => true,
            _ => expanded.len() > vocab,
        };
        let win = window_sec.map(|w| (w * TARGET_SR as f64) as usize);
        match win {
            None => {
                let mut input = waveform.to_vec();
                znorm(&mut input);
                let mut scratch = crate::wav2vec2::Scratch::default();
                Ok(if use_logits {
                    let g = self.forward_logits(&input, &mut scratch)?;
                    let rows = g.len() / vocab;
                    Trellis::Logits(LogitsBlocks {
                        blocks: vec![g],
                        vocab,
                        num_states: expanded.len(),
                        row_offset: 0,
                        frames_per_chunk: rows.max(1),
                        spans: vec![(0usize, 0usize, rows)],
                    })
                } else {
                    let g = self.forward_gathered(&input, expanded, &mut scratch)?;
                    let frames = g.len() / expanded.len();
                    Trellis::Gathered(GatheredChunks {
                        chunks: vec![g],
                        frames_per_chunk: frames.max(1),
                        num_states: expanded.len(),
                    })
                })
            }
            Some(win) => self.log_probs_chunked(waveform, win, context_sec, expanded, use_logits),
        }
    }

    fn vocab_size(&self) -> usize {
        match &self.tower {
            Tower::Cpu(m) => m.vocab_size(),
            Tower::Gpu(g) => g.vocab_size(),
        }
    }

    /// The lm head's (frames, vocab) block — what the trellis is gathered
    /// from on demand in [`Trellis::Logits`].  The CPU tower's rows are
    /// bias-free logits, the GPU tower's are already log-softmaxed; the
    /// matching gather lives in [`RowGather`].
    fn forward_logits(
        &self,
        input: &[f32],
        scratch: &mut crate::wav2vec2::Scratch,
    ) -> Result<Vec<f32>> {
        match &self.tower {
            // the gathered path reuses a scratch logits buffer; here the GEMM
            // writes straight into the block that is kept
            Tower::Cpu(m) => m.forward_logits(input, scratch),
            // no gather kernel: (t, vocab) log-probs instead of (t, S) trellis
            Tower::Gpu(g) => g.forward(input),
        }
    }

    /// Gathered scores of one z-normalised chunk; the CPU tower gathers from
    /// the full matrix, the GPU tower gathers on-device before readback.
    fn forward_gathered(
        &self,
        input: &[f32],
        expanded: &[usize],
        scratch: &mut crate::wav2vec2::Scratch,
    ) -> Result<Vec<f32>> {
        match &self.tower {
            Tower::Cpu(m) => {
                // The gathered epilogue is fused into the forward's lm head:
                // the (t, vocab) log-prob matrix is never materialised.
                let (g, _) = m.forward_gathered_with(input, &Default::default(), scratch, expanded)?;
                Ok(g)
            }
            Tower::Gpu(gpu_model) => {
                let exp32: Vec<u32> = expanded.iter().map(|&x| x as u32).collect();
                gpu_model.forward_gathered(input, &exp32)
            }
        }
    }

    /// Windowed encoding, gathered (port of `backend.log_probs_chunked`):
    /// chunks of `win` samples carry `ctx` real samples on both sides; only
    /// the middle `win` frames of each chunk are kept, so the stream tiles
    /// the audio exactly.  Normalisation is per chunk, as the Python path
    /// does; the gather is applied per chunk before readback.
    ///
    /// The trellis comes back one window at a time, so its memory commits as
    /// the file is encoded instead of doubling a single 5 GB allocation on the
    /// way (see `GatheredChunks`), or — in `Logits` mode — never materialising
    /// the wide trellis at all.
    fn log_probs_chunked(
        &self,
        waveform: &[f32],
        win: usize,
        ctx_sec: f64,
        expanded: &[usize],
        use_logits: bool,
    ) -> Result<Trellis> {
        let ctx = (ctx_sec * TARGET_SR as f64) as usize;
        let states = expanded.len();
        let vocab = self.vocab_size();
        if waveform.len() < win {
            let mut input = waveform.to_vec();
            znorm(&mut input);
            let mut scratch = crate::wav2vec2::Scratch::default();
            if use_logits {
                let g = self.forward_logits(&input, &mut scratch)?;
                let rows = g.len() / vocab;
                return Ok(Trellis::Logits(LogitsBlocks {
                    blocks: vec![g],
                    vocab,
                    num_states: states,
                    row_offset: 0,
                    frames_per_chunk: rows.max(1),
                    spans: vec![(0usize, 0usize, rows)],
                }));
            }
            let g = self.forward_gathered(&input, expanded, &mut scratch)?;
            let frames = g.len() / states;
            return Ok(Trellis::Gathered(GatheredChunks {
                chunks: vec![g],
                frames_per_chunk: frames.max(1),
                num_states: states,
            }));
        }
        let ctx_frames = ctx / SUBSAMPLING;
        let win_frames = win / SUBSAMPLING;
        anyhow::ensure!(ctx_frames >= 64, "context must cover the ±64-frame positional conv");
        anyhow::ensure!(win_frames > 0, "window must span at least one frame");

        let n = waveform.len();
        let extension = n.div_ceil(win) * win - n;
        // padded = [ctx zeros | waveform | ctx+extension zeros]
        let padded_len = n + 2 * ctx + extension;
        let ext_frames = ((extension as f64 / TARGET_SR as f64 * FRAME_RATE).ceil()) as usize;

        // rows kept per window: the middle win_frames, clamped for a short one
        let kept_of = |rows: usize| (ctx_frames + win_frames).min(rows) - ctx_frames.min(rows);
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut scratch = crate::wav2vec2::Scratch::default();
        // one scratch for the whole file: no per-chunk buffer churn
        let mut start = 0usize; // chunk start inside `padded`
        while start + win + 2 * ctx <= padded_len {
            // chunk = [ctx zeros | win real samples | ctx zeros], gathered from
            // the waveform with zero fill at the file edges
            let w_lo = start as i64 - ctx as i64; // waveform index of chunk sample 0
            let mut chunk = vec![0.0f32; win + 2 * ctx];
            for (i, slot) in chunk.iter_mut().enumerate() {
                let wi = w_lo + i as i64;
                if wi >= 0 && (wi as usize) < n {
                    *slot = waveform[wi as usize];
                }
            }
            znorm(&mut chunk);

            if use_logits {
                // keep the whole lm-head block: no per-chunk copy, and the
                // DP gathers this window's columns when it gets there
                let g = self.forward_logits(&chunk, &mut scratch)?;
                let rows = g.len() / vocab;
                blocks.push(g);
                if crate::alloc_stats::enabled() {
                    let (live, peak) = crate::alloc_stats::stats();
                    eprintln!(
                        "[alloc] chunk {:>2}: live {live:>12} peak {peak:>12}  logits {} MB, {} kept rows",
                        blocks.len() - 1,
                        blocks[blocks.len() - 1].capacity() * 4 >> 20,
                        kept_of(rows),
                    );
                }
            } else {
                let g = self.forward_gathered(&chunk, expanded, &mut scratch)?;
                let rows = g.len() / states;
                let keep_lo = (ctx_frames * states).min(g.len());
                let keep_hi = ((ctx_frames + win_frames).min(rows)) * states;
                let mut block = Vec::with_capacity(keep_hi - keep_lo);
                block.extend_from_slice(&g[keep_lo..keep_hi]);
                blocks.push(block);
                if crate::alloc_stats::enabled() {
                    let (live, peak) = crate::alloc_stats::stats();
                    let c = blocks.len() - 1;
                    eprintln!(
                        "[alloc] chunk {:>2}: live {live:>12} peak {peak:>12}  block {} MB",
                        c,
                        blocks[c].capacity() * 4 >> 20
                    );
                }
            }
            start += win;
        }

        // the tail zero-padding contributed ext_frames of frames: drop them
        // from the end of the kept stream, which is the last window's last
        // rows.  (The gathered path truncates the concatenated blocks, which
        // lands on the same rows; the logits path cannot truncate the block —
        // the padding sits *before* its end, inside the kept range — so it
        // shortens the last span instead.)
        if ext_frames > 0 {
            if use_logits {
                // nothing to trim: the block keeps its context rows
            } else if let Some(last) = blocks.last_mut() {
                let keep = last.len().saturating_sub(ext_frames * states);
                last.truncate(keep);
            }
        }
        if use_logits {
            let mut spans: Vec<(usize, usize, usize)> = blocks
                .iter()
                .enumerate()
                .map(|(i, b)| (i, ctx_frames.min(b.len() / vocab), kept_of(b.len() / vocab)))
                .collect();
            if let Some(last) = spans.last_mut() {
                last.2 = last.2.saturating_sub(ext_frames);
            }
            Ok(Trellis::Logits(LogitsBlocks {
                frames_per_chunk: spans.first().map_or(1, |s| s.2.max(1)),
                row_offset: spans.first().map_or(0, |s| s.1),
                blocks,
                vocab,
                num_states: states,
                spans,
            }))
        } else {
            Ok(Trellis::Gathered(GatheredChunks {
                chunks: blocks,
                frames_per_chunk: win_frames,
                num_states: states,
            }))
        }
    }
}

/// How the aligner holds one file's trellis between the forward pass and the
/// Viterbi.  Both variants hold the same f32 values, so the alignment does
/// not depend on which one a run picks.
pub enum Trellis {
    /// One (kept_frames × S) block per window — used when the trellis is
    /// narrower than the vocabulary (a short transcript).
    Gathered(GatheredChunks),
    /// The lm head's (rows × vocab) block per window, narrower than the
    /// trellis once the transcript passes ~5 k characters (15 m: 2.2 GB of
    /// logits against a 5.4 GB trellis).  The DP gathers each window's
    /// columns on demand, so only one window's trellis is ever resident.
    Logits(LogitsBlocks),
}

/// Per-window lm-head blocks plus the row window the DP is allowed to read:
/// the same kept frames the gathered path keeps, addressed inside the wider
/// block.
pub struct LogitsBlocks {
    blocks: Vec<Vec<f32>>,
    /// columns per row (the vocabulary)
    vocab: usize,
    /// columns the DP reads (S = 2·tokens+1)
    num_states: usize,
    /// kept frames per window; the last window may be shorter
    frames_per_chunk: usize,
    /// first kept row inside every block
    row_offset: usize,
    /// per block: (index, first kept row, kept frames) — the invariant the
    /// `t / frames_per_chunk` addressing relies on
    spans: Vec<(usize, usize, usize)>,
}

impl LogitsBlocks {
    fn total_frames(&self) -> usize {
        self.spans.iter().map(|s| s.2).sum()
    }

    /// First frame of window `b` in the DP's frame numbering.
    fn frame_base(&self, b: usize) -> usize {
        b * self.frames_per_chunk
    }

    /// (window, first kept row, kept frames) holding frame `t`.
    fn span_of(&self, t: usize) -> (usize, usize, usize) {
        let b = t / self.frames_per_chunk;
        let s = self.spans[b];
        (b, s.1, s.2)
    }

    /// Every window but the last must contribute exactly `frames_per_chunk`
    /// frames from the same row offset, or the DP's arithmetic addressing
    /// would read the wrong row instead of failing.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.vocab > 0 && self.num_states > 0, "empty trellis");
        anyhow::ensure!(self.frames_per_chunk > 0, "frames_per_chunk must be positive");
        let last = self.spans.len().saturating_sub(1);
        for (i, &(b, row, kept)) in self.spans.iter().enumerate() {
            anyhow::ensure!(b == i, "block span {i} out of order");
            let rows = self.blocks.get(i).map_or(0, |x| x.len() / self.vocab);
            anyhow::ensure!(
                self.blocks[i].len() % self.vocab == 0,
                "block {i} is not a whole number of {0}-column rows",
                self.vocab
            );
            anyhow::ensure!(row + kept <= rows, "block {i} keeps {kept} of {rows} rows");
            if i != last {
                anyhow::ensure!(
                    kept == self.frames_per_chunk && row == self.row_offset,
                    "block {i} keeps {kept}@{row}, expected {}@{} (only the last may be short)",
                    self.frames_per_chunk,
                    self.row_offset
                );
            }
        }
        Ok(())
    }
}

/// Gathers one row of an lm-head block into the trellis columns, per tower:
/// the CPU's rows are bias-free logits, the GPU's are already log-softmaxed
/// (its gather kernel is a plain column copy, so this reproduces it exactly).
enum RowGather<'a> {
    Cpu(&'a Model),
    Gpu,
}

impl RowGather<'_> {
    /// Fill `dst` with the trellis columns of `src`, returning the row's
    /// log-softmax normaliser (`0` on the GPU tower, whose rows are already
    /// log-probs).
    fn row(&self, src: &[f32], cols: &[i32], dst: &mut [f32]) -> f32 {
        match self {
            RowGather::Cpu(m) => m.gather_logits_row(src, cols, dst),
            RowGather::Gpu => {
                for (j, &c) in cols.iter().enumerate() {
                    dst[j] = src[c as usize];
                }
                0.0
            }
        }
    }

    /// One trellis column of `src`, reusing a normaliser a previous pass
    /// recorded.  Bit-identical to the `row` call that produced it: same
    /// operands, same `x + bias - c` arithmetic.
    fn value(&self, src: &[f32], col: usize, c: f32) -> f32 {
        match self {
            RowGather::Cpu(m) => src[col] + m.lm_bias()[col] - c,
            RowGather::Gpu => src[col],
        }
    }
}

/// [`Emissions`] over [`Trellis::Logits`]: the DP walks forward, so each
/// window's columns are gathered the first time a frame inside it is reached
/// and dropped when the walk moves past it.  Peak cost is one window's
/// trellis (t×S) on top of the stored logits, not the whole file's.
struct LogitsEmissions<'a> {
    blocks: &'a LogitsBlocks,
    gather: RowGather<'a>,
    /// the expanded labels, as the gather kernel wants them
    cols: Vec<i32>,
    /// (block index, its trellis block), the one window currently gathered
    cache: std::cell::RefCell<Option<(usize, Vec<f32>)>>,
    /// per frame, the log-softmax normaliser recorded while the DP gathered
    /// its window (0 on the GPU tower, whose blocks are already log-probs).
    /// This is what lets the post-traceback pass read the path's column
    /// straight out of the stored logits instead of gathering every window a
    /// second time — 46 k reads instead of 5.8 GB of writes.
    norm: std::cell::RefCell<Vec<f32>>,
}

impl<'a> LogitsEmissions<'a> {
    fn new(blocks: &'a LogitsBlocks, gather: RowGather<'a>, expanded: &[usize]) -> Self {
        let frames = blocks.total_frames();
        LogitsEmissions {
            blocks,
            gather,
            cols: expanded.iter().map(|&x| x as i32).collect(),
            cache: std::cell::RefCell::new(None),
            norm: std::cell::RefCell::new(vec![0.0; frames]),
        }
    }

    fn total_frames(&self) -> usize {
        self.blocks.total_frames()
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.blocks.num_states == self.cols.len(),
            "trellis has {} states but {} columns",
            self.blocks.num_states,
            self.cols.len()
        );
        self.blocks.validate()
    }

    /// One window's trellis block, gathered on first use and then replaced as
    /// the DP walks into the next window.  The buffer is reused across windows
    /// (only the last one is shorter), so the 187 MB is faulted in once.
    fn window(&self, b: usize) -> std::cell::RefMut<'_, [f32]> {
        use std::cell::RefMut;
        let mut cache = self.cache.borrow_mut();
        let stale = cache.as_ref().map(|(i, _)| *i != b).unwrap_or(true);
        if stale {
            let s = self.blocks.num_states;
            let (_, row_offset, kept) = self.blocks.spans[b];
            let vocab = self.blocks.vocab;
            // slice to the kept rows first: same shape as the forward's fused
            // gather, and no `skip` adaptor in the parallel iterator
            let src = &self.blocks.blocks[b][row_offset * vocab..][..kept * vocab];
            let need = kept * s;
            let out = match cache.as_mut() {
                Some((_, buf)) if buf.len() == need => buf,
                _ => {
                    cache.replace((b, vec![0.0f32; need]));
                    &mut cache.as_mut().expect("just filled").1
                }
            };
            let gather = &self.gather;
            let cols = &self.cols;
            let t0 = b * self.blocks.frames_per_chunk;
            // record each row's normaliser so the post-traceback pass never
            // has to gather this window again
            let mut norm = self.norm.borrow_mut();
            let norm_row = &mut norm[t0..t0 + kept];
            src.par_chunks_exact(vocab)
                .zip(out.par_chunks_mut(s))
                .zip(norm_row.par_iter_mut())
                .for_each(|((row, dst), c)| *c = gather.row(row, cols, dst));
            drop(norm);
            cache.as_mut().expect("just filled").0 = b;
        }
        RefMut::map(cache, |c: &mut Option<(usize, Vec<f32>)>| {
            c.as_mut().expect("just filled").1.as_mut_slice()
        })
    }
}

impl Emissions for LogitsEmissions<'_> {
    fn fill_emit(&self, t: usize, emit: &mut [f64], _token_ids: &[usize]) {
        let s = self.blocks.num_states;
        let fpc = self.blocks.frames_per_chunk;
        let block = self.window(t / fpc);
        let row = &block[(t % fpc) * s..][..s];
        for (e, &v) in emit.iter_mut().zip(row) {
            *e = v as f64;
        }
    }

    /// One column of `t`'s row, straight from the stored logits.  The window
    /// holding `t` has to have been gathered at least once for its
    /// normaliser to exist — the DP reads frame 0's two states *before* its
    /// first `fill_emit`, so this may be what triggers the gather.  After the
    /// DP, every window is normalised and this is a single indexed read.
    fn score(&self, t: usize, st: usize) -> f32 {
        let b = self.blocks;
        let (block, row, kept) = b.span_of(t);
        let _ = self.window(block); // records this row's normaliser
        let col = self.cols[st] as usize;
        let r = t - b.frame_base(block);
        debug_assert!(r < kept);
        let src = &b.blocks[block][(row + r) * b.vocab..][..b.vocab];
        self.gather.value(src, col, self.norm.borrow()[t])
    }

    /// The path's per-frame scores, window by window, without re-gathering:
    /// `score_path` is only reached after the DP has normalised every row.
    fn score_path(&self, states: &[i32], out: &mut [f64]) {
        let b = self.blocks;
        let norm = self.norm.borrow();
        for (bi, &(_, row, kept)) in b.spans.iter().enumerate() {
            let base = bi * b.frames_per_chunk;
            let src = &b.blocks[bi][row * b.vocab..][..kept * b.vocab];
            for r in 0..kept {
                let t = base + r;
                let col = self.cols[states[t] as usize] as usize;
                out[t] = self.gather.value(&src[r * b.vocab..][..b.vocab], col, norm[t]) as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two trellis representations must be interchangeable: the lazy
    /// logits gather reads a window's columns out of the wider lm-head block,
    /// and any mistake in the `t / frames_per_chunk` + row-offset addressing
    /// — or in how the last, short window is trimmed — would shift timestamps
    /// instead of failing.  `RowGather::Gpu` is the pure column copy, so this
    /// needs no model.
    #[test]
    fn lazy_logits_trellis_matches_gathered_blocks() {
        let (vocab, l) = (23usize, 5usize);
        let blank = 0usize;
        let token_ids: Vec<usize> = (1..=l).map(|i| (i * 3) % vocab).collect();
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let expanded = crate::viterbi::build_expanded_labels(&token_ids, blank);
        let s = expanded.len();
        assert_eq!(s, 2 * l + 1);

        // three windows of `per` kept rows, the last short; context rows in
        // front of each block
        let (row_offset, kept_last, per) = (5usize, 7usize, 13usize);
        let rows_per = row_offset + per;
        let total_kept = 2 * per + kept_last;
        let total_rows = 3 * rows_per;

        // "log-prob" block: the gather is a column copy, values are arbitrary
        let block_all: Vec<f32> = (0..total_rows * vocab)
            .map(|i| -((i % 37) as f32) * 0.07 - 0.3)
            .collect();

        let mut blocks = Vec::new();
        let mut spans = Vec::new();
        for i in 0..3 {
            let kept = if i == 2 { kept_last } else { per };
            blocks.push(
                block_all[i * rows_per * vocab..(i * rows_per + rows_per) * vocab].to_vec(),
            );
            spans.push((i, row_offset, kept));
        }
        let lb = LogitsBlocks {
            blocks,
            vocab,
            num_states: s,
            frames_per_chunk: per,
            row_offset,
            spans,
        };

        // the same values, gathered: what the block-per-window path holds
        let mut flat = Vec::with_capacity(total_kept * s);
        for (b, &(_, row, kept)) in lb.spans.iter().enumerate() {
            for r in 0..kept {
                let src = &lb.blocks[b][(row + r) * vocab..][..vocab];
                for &c in &expanded {
                    flat.push(src[c]);
                }
            }
        }
        let gc = GatheredChunks {
            chunks: flat.chunks(per * s).map(|c| c.to_vec()).collect(),
            frames_per_chunk: per,
            num_states: s,
        };
        gc.validate().unwrap();

        let em = LogitsEmissions::new(&lb, RowGather::Gpu, &expanded);
        em.validate().unwrap();
        assert_eq!(em.total_frames(), total_kept);
        assert_eq!(gc.total_frames(), total_kept);

        let want = ctc_forced_align_gathered_chunks(&gc, &token_ids, 50.0, Some(&pieces)).unwrap();
        let got = ctc_forced_align_emissions(&em, total_kept, &token_ids, 50.0, Some(&pieces))
            .unwrap();

        assert_eq!(got.tokens.len(), want.tokens.len());
        for (g, w) in got.tokens.iter().zip(&want.tokens) {
            assert_eq!((g.start_frame, g.end_frame), (w.start_frame, w.end_frame));
            assert_eq!(g.score.to_bits(), w.score.to_bits(), "token score bits");
        }
        assert_eq!(got.log_prob.to_bits(), want.log_prob.to_bits());
        assert_eq!(got.frame_scores.len(), want.frame_scores.len());
        for (a, b) in got.frame_scores.iter().zip(&want.frame_scores) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
