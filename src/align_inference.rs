//! The aligner's forward: (audio, text) → character / word / segment
//! timestamps.  Orchestration port of `omni_align/aligner.py` +
//! `backend.log_probs_chunked`, producing the same JSON schema.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::audio::{load_audio, znorm, TARGET_SR};
use crate::fix_timestamp::fix_timestamp;
use crate::config::Wav2Vec2Config;
use crate::gpu::DeviceSelector;
use crate::spans::{build_segments, build_words};
use crate::viterbi::{build_expanded_labels, ctc_forced_align_gathered, TokenAlignment};
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
        let gathered = self.log_probs_gathered(&waveform, window_sec, context_sec, &expanded)?;
        let encode_s = t_enc.elapsed().as_secs_f64();
        let t_al = std::time::Instant::now();
        let states = expanded.len();
        let mut res = ctc_forced_align_gathered(
            &gathered,
            gathered.len() / states,
            states,
            &ids,
            FRAME_RATE,
            Some(&pieces),
        )?;
        let align_s = t_al.elapsed().as_secs_f64();

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

    /// Trellis-label log-probabilities for a raw (un-normalised) waveform:
    /// (frames, expanded.len()) row-major.
    fn log_probs_gathered(
        &self,
        waveform: &[f32],
        window_sec: Option<f64>,
        context_sec: f64,
        expanded: &[usize],
    ) -> Result<Vec<f32>> {
        let win = window_sec.map(|w| (w * TARGET_SR as f64) as usize);
        match win {
            None => {
                let mut input = waveform.to_vec();
                znorm(&mut input);
                self.forward_gathered(&input, expanded, &mut crate::wav2vec2::Scratch::default())
            }
            Some(win) => self.log_probs_chunked_gathered(waveform, win, context_sec, expanded),
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
                let (log_probs, _) = m.forward_with(input, &Default::default(), scratch)?;
                let v = self.config().vocab_size;
                let frames = log_probs.len() / v;
                let mut g = Vec::with_capacity(frames * expanded.len());
                for f in 0..frames {
                    let row = &log_probs[f * v..(f + 1) * v];
                    for &st in expanded {
                        g.push(row[st]);
                    }
                }
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
    fn log_probs_chunked_gathered(
        &self,
        waveform: &[f32],
        win: usize,
        ctx_sec: f64,
        expanded: &[usize],
    ) -> Result<Vec<f32>> {
        let ctx = (ctx_sec * TARGET_SR as f64) as usize;
        let states = expanded.len();
        if waveform.len() < win {
            let mut input = waveform.to_vec();
            znorm(&mut input);
            return self.forward_gathered(&input, expanded, &mut crate::wav2vec2::Scratch::default());
        }
        let ctx_frames = ctx / SUBSAMPLING;
        let win_frames = win / SUBSAMPLING;
        anyhow::ensure!(ctx_frames >= 64, "context must cover the ±64-frame positional conv");

        let n = waveform.len();
        let extension = n.div_ceil(win) * win - n;
        // padded = [ctx zeros | waveform | ctx+extension zeros]
        let padded_len = n + 2 * ctx + extension;

        let mut chunks = Vec::new();
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
            chunks.push(chunk);
            start += win;
        }

        let mut scratch = crate::wav2vec2::Scratch::default();
        // one scratch for the whole file: no per-chunk buffer churn
        let mut out: Vec<f32> = Vec::new();
        for chunk in &chunks {
            let g = self.forward_gathered(chunk, expanded, &mut scratch)?;
            let rows = g.len() / states;
            let keep_lo = (ctx_frames * states).min(g.len());
            let keep_hi = ((ctx_frames + win_frames).min(rows)) * states;
            out.extend_from_slice(&g[keep_lo..keep_hi]);
        }

        // drop the frames the tail padding contributed
        let ext_frames = ((extension as f64 / TARGET_SR as f64 * FRAME_RATE).ceil()) as usize;
        if ext_frames > 0 && out.len() >= ext_frames * states {
            let keep = out.len() - ext_frames * states;
            out.truncate(keep);
        }
        Ok(out)
    }
}
