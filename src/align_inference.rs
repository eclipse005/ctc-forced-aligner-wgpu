//! The aligner's forward: (audio, text) → character / word / segment
//! timestamps.  Orchestration port of `omni_align/aligner.py` +
//! `backend.log_probs_chunked`, producing the same JSON schema.

use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::audio::{load_audio, znorm, TARGET_SR};
use crate::timeline::{anchor_marks, place_unmeasured};
use crate::gpu::DeviceSelector;
use crate::spans::{build_segments, build_words};
use crate::viterbi::{
    build_expanded_labels, ctc_forced_align_emissions_with_word_ids,
    ctc_forced_align_gathered_with_word_ids,
    Emissions, GatheredChunks, TokenAlignment,
};
use crate::vocab::Vocab;
use crate::wav2vec2::{LmHeadCpu, Model};
use crate::wav2vec2_gpu::GpuModel;

/// The model on one backend. Both towers are token-identical to the
/// Python reference (see tests/golden.rs).
pub(crate) enum Tower {
    Cpu(Model),
    Gpu(GpuModel),
}


/// The score the reference gives `<star>`: the column it appends to its
/// emissions is `torch.cat([emissions, zeros(..., 1)], dim=1)` AFTER the
/// log_softmax, so every star frame scores exactly 0.0 — measured on a real
/// utterance, where the appended column is all zeros while the real column
/// beside it averages −2.457.
///
/// Scoring it 0.0 exactly, as the reference does, lets the path PARK on the
/// star: with a 0 to tie or beat the blank, the unbanded DP held it for eleven
/// consecutive frames where the reference held it for one, which moved every
/// later boundary with it. A small negative value keeps the star reachable —
/// it still has to be placed, the CTC path cannot skip a target — while making
/// the blank strictly preferable on any frame where the model is not sure the
/// frame is the star, which is the situation the reference's own banding
/// produces.
const CTC_STAR_SCORE: f32 = -1.0;

pub struct Aligner {
    pub(crate) tower: Tower,
    pub(crate) vocab: Vocab,
    /// Host-side lm head weights, loaded for the GPU tower only: its hidden
    /// form stores the encoder stream and re-runs the head per window
    /// ([`RowGather::Head`]).  `None` on the CPU tower, which owns a full
    /// [`Model`].
    pub(crate) gpu_lm_head: Option<LmHeadCpu>,
    /// Input samples per output frame, from the feature extractor's conv
    /// strides. Not a constant: it is a property of the checkpoint, and
    /// hardcoding it would silently mis-time every frame of a model whose
    /// extractor downsamples differently.
    pub(crate) subsampling: usize,
    /// Timestamps per second, i.e. `TARGET_SR / subsampling`.
    pub(crate) frame_rate: f64,
    /// The CTC blank, i.e. the checkpoint's `pad_token_id`.
    pub(crate) blank_id: usize,
}

/// One alignment, as JSON: the transcript, and where in the audio it is said.
///
/// Three things, all at the top level and none inside another:
/// `text` is the transcript, `tokens` is where each unit of it is said, and
/// `segments` is the same thing read a sentence at a time. None of them points
/// at another, so a consumer who only wants timings reads one array and stops,
/// and one who wants sentences reads the other. Carrying the intermediate
/// views as well -- words, subtitle cues, a gapless timeline -- made the file
/// 39% larger and added copies of every character to keep in step.
#[derive(serde::Serialize)]
pub struct AlignOutput {
    pub audio: String,
    /// The transcript as given, unchanged. Every character of it is in
    /// `tokens`, in this order.
    pub text: String,
    pub duration: f64,
    pub frames: usize,
    pub frame_rate: f64,
    pub mean_frame_score: f64,
    pub log_prob: f64,
    /// Whole sentences, each a `text` with a `start` and an `end`, cut at the
    /// transcript's own sentence-final punctuation.
    ///
    /// Readable, and derived from `tokens` -- a consumer whose idea of a
    /// sentence differs groups them again themselves.
    pub segments: Vec<serde_json::Value>,
    /// Encoder time. Omitted from JSON so the schema stays the Python one.
    #[serde(skip)]
    pub encode_s: f64,
    /// Viterbi time. Omitted from JSON for the same reason.
    #[serde(skip)]
    pub align_s: f64,
    /// The alignment itself, one entry per CTC target.
    ///
    /// In JSON this is not one row per target but one row per TIMED UNIT, and
    /// the unit is decided per word with no language table: a whole word where
    /// the script spaces its words, a single character where it does not. A
    /// mixed transcript gets both, in the same file. See
    /// `serialize_token_rows`.
    #[serde(serialize_with = "serialize_token_rows")]
    pub tokens: Vec<TokenAlignment>,
    /// Winning-label log-prob per frame, including blanks. Diagnostic only —
    /// `mean_frame_score` is the part worth reading.
    #[serde(skip)]
    pub frame_scores: Vec<f64>,
    /// Per-frame trellis state (even = blank, odd = token), the path the
    /// blank runs are cut from. Diagnostic only — not serialised.
    #[serde(skip)]
    pub frame_path: Vec<i32>,
    /// The blank runs the boundary padding consumed, as
    /// `(before_token_index, first_frame, last_frame)`, with `before_token_index
    /// == tokens.len()` for the trailing run.
    ///
    /// Diagnostic only, and only populated by `align_with_path`: a one- or
    /// two-frame difference in the output cannot otherwise be told apart from a
    /// midpoint taken over a different range, and the per-frame diff against the
    /// Python reference needs the ranges themselves. `examples/dump_path.rs`
    /// writes it out.
    #[serde(skip)]
    pub blank_runs: Vec<(usize, i64, i64)>,
}

/// The JSON form of [`AlignOutput::tokens`]: one row per timed unit.
///
/// A unit is a word, and which kind of word is decided per word with no
/// language table: where the script spaces its words a row is the whole word,
/// and where it does not a row is a single character -- a Chinese "word" is a
/// whole sentence, and the test applies inside it. A transcript with both gets
/// both, which is the point of deciding it per word rather than per file.
///
/// The `<star>` targets the tokenizer puts between words are not rows: they are
/// DP anchors rather than transcript text, they carry a constant score, and a
/// `<star>` printed into a subtitle is a bug.
///
/// Nothing else is written. `duration` is `end - start`, `mid` is
/// `(start + end) / 2`, a row's position is its index in the array, the frame
/// numbers are `start * frame_rate`, and `token_id` is a raw vocabulary index
/// whose only useful question -- was this measured? -- is what `inferred`
/// answers.
fn serialize_token_rows<S: serde::Serializer>(
    tokens: &[TokenAlignment],
    s: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let words = build_words(tokens);
    let mut seq = s.serialize_seq(Some(words.len()))?;
    for w in &words {
        let span = &tokens[w.char_start..=w.char_end];
        let measured: Vec<&TokenAlignment> = span.iter().filter(|t| !t.inferred).collect();
        seq.serialize_element(&serde_json::json!({
            "text": w.text,
            "start": r4(w.start),
            "end": r4(w.end),
            // the transcript had whitespace in front of this unit. Without it
            // the rows cannot be rendered back into the text they came from,
            // and the obvious guess -- join with spaces -- is what put a space
            // between every character of a Japanese or Chinese sentence.
            "space_before": w.space_before,
            // the mean per-frame log-probability of the characters that were
            // measured, and `null` when none of them was. A zero would not do:
            // an interpolated character carries no score, and a log-prob of 0
            // is a probability of 1 -- the best possible confidence, said about
            // the one row that has no evidence at all.
            "score": match measured.is_empty() {
                true => serde_json::Value::Null,
                false => {
                    let mean = measured.iter().map(|t| t.score).sum::<f64>() / measured.len() as f64;
                    serde_json::json!(r4(mean))
                }
            },
            // the vocabulary had no target for at least one character of this
            // unit, so part of its span is the midpoint of what the neighbours
            // leave open rather than something measured
            "inferred": span.iter().any(|t| t.inferred),
        }))?;
    }
    seq.end()
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
        let gpu_lm_head = match tower {
            Tower::Gpu(_) => Some(LmHeadCpu::load(model_dir)?),
            Tower::Cpu(_) => None,
        };
        let vocab = Vocab::load(model_dir)?;
        let cfg = crate::config::Wav2Vec2Config::load(model_dir)?;
        // The star's synthetic id must land outside the vocabulary, and the
        // vocabulary must be exactly as wide as the lm head: a checkpoint with
        // spare columns takes a different gather path than one without, and
        // guessing wrong mis-times the whole file.
        anyhow::ensure!(
            vocab.size == cfg.vocab_size,
            "vocab.json holds {} ids but config.json declares vocab_size {}",
            vocab.size,
            cfg.vocab_size
        );
        let subsampling = cfg.subsampling();
        let frame_rate = cfg.frame_rate(TARGET_SR);
        anyhow::ensure!(
            cfg.pad_token_id < vocab.size,
            "pad_token_id {} is outside the vocabulary",
            cfg.pad_token_id
        );
        Ok(Self {
            tower,
            vocab,
            gpu_lm_head,
            subsampling,
            frame_rate,
            blank_id: cfg.pad_token_id,
        })
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

    /// The checkpoint's own configuration. The golden test reads the frame
    /// rate and vocabulary size from here rather than hardcoding them, because
    /// hardcoding them is the bug the config module exists to prevent.
    #[cfg(test)]
    pub(crate) fn config(&self) -> &crate::config::Wav2Vec2Config {
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
        self.align_impl(audio_path, text, window_sec, context_sec, false)
    }

    /// `align`, but the per-frame trellis path is kept on the output.
    ///
    /// The path is what the blank runs are cut from, and a blank run is what the
    /// word boundary is padded into, so it is the only way to tell a midpoint
    /// computed over the wrong range from one computed over a different range.
    /// Diagnostic only — the production path leaves the traceback freed.
    pub fn align_with_path(
        &self,
        audio_path: &Path,
        text: &str,
        window_sec: Option<f64>,
        context_sec: f64,
    ) -> Result<AlignOutput> {
        self.align_impl(audio_path, text, window_sec, context_sec, true)
    }

    fn align_impl(
        &self,
        audio_path: &Path,
        text: &str,
        window_sec: Option<f64>,
        context_sec: f64,
        keep_path: bool,
    ) -> Result<AlignOutput> {
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        // `<star>` between words, the reference's per-script placement. See
        // `Vocab::tokenise_with_stars` for why this is not cosmetic: the star is
        // a real CTC target, so the DP places it and it consumes frames, which
        // is what the blank runs -- and therefore every word boundary -- are
        // measured against.
        let (ids, pieces, word_ids, src) = self.vocab.tokenise_with_word_ids(&text);
        if ids.is_empty() {
            anyhow::bail!("no in-vocabulary characters in the transcript");
        }
        // The star's id is one past the last real id, so a gather would index
        // past the end of the row. Gather it from the BLANK column instead --
        // the reference's own appended star column is a constant 0 (verified on
        // a real utterance), and the blank column is the only one guaranteed
        // present. The DP still treats the star as its own target, because the
        // skip rule compares ids and `<star>` never equals `<blank>`.
        let star = self.vocab.star_id;
        let gather_ids: Vec<usize> = ids.iter().map(|&c| if c == star { self.blank_id } else { c }).collect();

        let (waveform, sr) = load_audio(audio_path)
            .with_context(|| format!("load {}", audio_path.display()))?;
        anyhow::ensure!(sr == TARGET_SR, "expected {TARGET_SR} Hz after decoding");
        let duration = waveform.len() as f64 / sr as f64;

        // Both towers gather the trellis labels' log-probs (T, S) instead of
        // handing the whole (T, V) matrix to the CPU — the GPU readback of a
        // 34 s chunk drops from ~70 MB to ~30 KB.  The gathered values are
        // the same f32s the full path would read, so the DP is unchanged.
        // `expanded` is what the GATHER indexes with, so it must not contain
        // the out-of-range star id; `ids` is what the DP compares for the
        // repeat rule, so it keeps the real one. The star's ODD state index in
        // `expanded` is recorded so its gathered score can be forced to the 0.0
        // the reference gives it -- see `forward_gathered`.
        let expanded = build_expanded_labels(&gather_ids, self.blank_id);
        let star_state_idx: Vec<usize> =
            (0..ids.len()).filter(|&i| ids[i] == star).map(|i| 2 * i + 1).collect();
        let t_enc = std::time::Instant::now();
        let trellis = self.log_probs_trellis(
            &waveform, window_sec, context_sec, &expanded, &star_state_idx,
        )?;
        let encode_s = t_enc.elapsed().as_secs_f64();
        crate::wav2vec2::prof::dump(&format!(
            "{} [{}]",
            audio_path.display(),
            self.backend_name()
        ));
        let t_al = std::time::Instant::now();
        let mut res = match trellis {
            Trellis::Gathered(gathered) => {
                ctc_forced_align_gathered_with_word_ids(
                    &gathered, &ids, self.frame_rate, Some(&pieces), &word_ids, keep_path)?
            }
            Trellis::Lazy(blocks) => {
                let gather = match &self.tower {
                    Tower::Cpu(m) => RowGather::Cpu(m),
                    Tower::Gpu(_) => match blocks.kind {
                        // the GPU tower's logits blocks are already
                        // log-softmaxed on the device (plain column copy);
                        // its hidden blocks need the host-side head re-run
                        BlockKind::Hidden { .. } => RowGather::Head(self
                            .gpu_lm_head
                            .as_ref()
                            .context("GPU tower is missing the host-side lm head")?),
                        BlockKind::Logits { .. } => RowGather::Gpu,
                    },
                };
                let em = LazyEmissions::new(&blocks, gather, &expanded, &star_state_idx);
                let frames = em.total_frames();
                em.validate()?;
                ctc_forced_align_emissions_with_word_ids(
                    &em, frames, &ids, self.frame_rate, Some(&pieces), &word_ids)?
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

        anchor_marks(&mut res.tokens);
        res.tokens = place_unmeasured(
            &res.tokens,
            &text,
            &src,
            self.frame_rate,
            duration,
        );
        // `segments` is a reading of `tokens`, cut at the transcript's own
        // sentence-final punctuation, and a consumer whose idea of a sentence
        // differs groups them again themselves. It carries no index into
        // `tokens`: a row that has to be joined up to mean anything is not
        // parallel to one that does not.
        let segments = build_segments(&res.tokens, &build_words(&res.tokens));

        Ok(AlignOutput {
            audio: audio_path.display().to_string(),
            text,
            duration: r4(duration),
            frames: res.frames,
            frame_rate: res.frame_rate,
            mean_frame_score: r4(res.mean_frame_score()),
            log_prob: r4(res.log_prob),
            segments: segments
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "text": s.text,
                        "start": r4(s.start),
                        "end": r4(s.end),
                    })
                })
                .collect(),
            tokens: std::mem::take(&mut res.tokens),
            encode_s,
            align_s,
            frame_scores: res.frame_scores,
            frame_path: res.frame_path.unwrap_or_default(),
            blank_runs: res.blank_runs,
        })
    }

    /// Trellis-label log-probabilities for a raw (un-normalised) waveform,
    /// in whichever of the equivalent storage forms fits the memory budget.
    fn log_probs_trellis(
        &self,
        waveform: &[f32],
        window_sec: Option<f64>,
        context_sec: f64,
        expanded: &[usize],
        star_state_idx: &[usize],
    ) -> Result<Trellis> {
        // The DP reads one f32 per (frame, expanded state), so the trellis is
        // T×(2·tokens+1).  Two things can stand in for it, both holding the
        // identical f32s: the lm head's T×vocab output, and — if even that is
        // too much — the encoder's own T×hidden stream, from which the lm head
        // is re-run on demand.  Whichever of the three is the narrowest one
        // that fits, so RAM is bounded no matter how long the audio is.
        // `CTC_TRELLIS=auto|gathered|logits|hidden` forces any of them.
        let vocab = self.vocab_size();
        let hidden = self.hidden_size();
        let states = expanded.len();
        let frames = (waveform.len() / self.subsampling).max(1);
        let win = window_sec.map(|w| (w * TARGET_SR as f64) as usize);
        let form = match std::env::var("CTC_TRELLIS").ok().as_deref() {
            Some("gathered") => Form::Gathered,
            Some("logits") => Form::Lazy(BlockKind::Logits { vocab }),
            Some("hidden") => Form::Lazy(BlockKind::Hidden { hidden }),
            _ if fits(frames, states) => Form::Gathered,
            // past the gathered trellis, the encoder stream (4 KB/frame) beats
            // parking the lm head's logits (41 KB/frame) for a windowed run:
            // the per-window head re-run costs ~25 ms against a 34 s window,
            // while the logits form's extra 37 KB/frame sits resident for the
            // whole DP.  Measured on 15 m: hidden 2.24 GB / RTFx 17.8 vs
            // logits 4.19 GB / 17.1, outputs byte-identical.  An unchunked
            // forward has no windows to re-run over — the head would
            // materialise the whole (T, V) matrix anyway — so it keeps the
            // logits form.
            _ if win.is_some() => Form::Lazy(BlockKind::Hidden { hidden }),
            _ => Form::Lazy(BlockKind::Logits { vocab }),
        };
        match win {
            None => {
                let mut input = waveform.to_vec();
                znorm(&mut input);
                let mut scratch = crate::wav2vec2::Scratch::default();
                Ok(match form {
                    Form::Gathered => {
                        let g =
                            self.forward_gathered(&input, expanded, &star_state_idx, &mut scratch, None)?;
                        let f = g.len() / states;
                        Trellis::Gathered(GatheredChunks {
                            chunks: vec![g],
                            frames_per_chunk: f.max(1),
                            num_states: states,
                        })
                    }
                    Form::Lazy(kind) => {
                        let g = self.forward_lazy(&input, kind, &mut scratch)?;
                        let rows = g.len() / kind.width();
                        Trellis::Lazy(LazyBlocks {
                            blocks: vec![g],
                            kind,
                            num_states: states,
                            row_offset: 0,
                            frames_per_chunk: rows.max(1),
                            spans: vec![(0usize, 0usize, rows)],
                        })
                    }
                })
            }
            Some(win) => self.log_probs_chunked(
                waveform, win, context_sec, expanded, star_state_idx, form,
            ),
        }
    }

    fn vocab_size(&self) -> usize {
        match &self.tower {
            Tower::Cpu(m) => m.vocab_size(),
            Tower::Gpu(g) => g.vocab_size(),
        }
    }

    fn hidden_size(&self) -> usize {
        match &self.tower {
            Tower::Cpu(m) => m.hidden_size(),
            Tower::Gpu(g) => g.hidden_size(),
        }
    }

    /// The block a lazily gathered trellis is parked in, per
    /// [`Trellis::Lazy`] kind: the lm head's (frames, vocab) output, or the
    /// encoder's (frames, hidden) stream with the head not yet run.  The
    /// matching gather lives in [`RowGather`].
    fn forward_lazy(
        &self,
        input: &[f32],
        kind: BlockKind,
        scratch: &mut crate::wav2vec2::Scratch,
    ) -> Result<Vec<f32>> {
        match (&self.tower, kind) {
            (Tower::Cpu(m), BlockKind::Logits { .. }) => m.forward_logits(input, scratch),
            (Tower::Cpu(m), BlockKind::Hidden { .. }) => m.forward_hidden(input, scratch),
            // the hidden readback: post-final-LN stream, head re-run later
            (Tower::Gpu(g), BlockKind::Hidden { .. }) => g.forward_hidden(input),
            // no gather kernel: (t, vocab) log-probs instead of (t, S) trellis
            (Tower::Gpu(g), BlockKind::Logits { .. }) => g.forward(input),
        }
    }

    /// Gathered scores of one z-normalised chunk; the CPU tower gathers from
    /// the full matrix, the GPU tower gathers on-device before readback.
    ///
    /// `keep` restricts the result to a row range of the chunk (the windowed
    /// aligner keeps only the middle rows and drops the context rows on both
    /// sides; the range is clamped to what the chunk produced, since the last
    /// one may be short). Rows outside `keep` are never stamped, never copied
    /// and, on the GPU tower, never downloaded.
    fn forward_gathered(
        &self,
        input: &[f32],
        expanded: &[usize],
        star_state_idx: &[usize],
        scratch: &mut crate::wav2vec2::Scratch,
        keep: Option<(usize, usize)>,
    ) -> Result<Vec<f32>> {
        // The star's id is one past the last real column, so `expanded` carries
        // BLANK_ID there instead (see the caller). The reference scores that
        // column 0.0, not a real log-probability: `generate_emissions` does its
        // log_softmax and then `torch.cat([emissions, zeros(..., 1)], dim=1)`,
        // so `<star>` indexes an appended zero column -- verified on a real
        // utterance, where it is all zeros while the column beside it is about
        // -25.8. Gathering the blank column instead yields a real log-prob
        // (around -0.1 there), a different number, and the DP moves: the star
        // landed on frame 0 instead of frame 6 and every later boundary with
        // it. So the star states are overwritten back to 0 here.
        let states = expanded.len();
        let g = match &self.tower {
            Tower::Cpu(m) => {
                // The gathered epilogue is fused into the forward's lm head:
                // the (t, vocab) log-prob matrix is never materialised.
                let (mut g, _) =
                    m.forward_gathered_with(input, &Default::default(), scratch, expanded)?;
                if let Some((lo, hi)) = keep {
                    let lo = lo.min(g.len() / states);
                    let hi = hi.min(g.len() / states).max(lo);
                    g.drain(hi * states..);
                    g.drain(..lo * states);
                }
                // the star states' columns are the reference's appended zero
                // column, not a real log-prob; stamped row-major — one row is
                // `states * 4` bytes and stays in cache (the column-major
                // loop this replaced wrote `frames` values at a stride of
                // `states` floats per star, every write a fresh cache line).
                // `star_state_idx` is already ascending.
                let stars: Vec<usize> = star_state_idx
                    .iter()
                    .copied()
                    .filter(|&si| si < states)
                    .collect();
                if !stars.is_empty() {
                    for row in g.chunks_exact_mut(states) {
                        for &si in &stars {
                            row[si] = CTC_STAR_SCORE;
                        }
                    }
                }
                g
            }
            Tower::Gpu(gpu_model) => {
                let exp32: Vec<u32> = expanded.iter().map(|&x| x as u32).collect();
                let star32: Vec<u32> = star_state_idx.iter().map(|&x| x as u32).collect();
                // the gather kernel stamps the star columns on-device
                gpu_model.forward_gathered(input, &exp32, keep, &star32)?
            }
        };
        anyhow::ensure!(
            g.len() % states == 0,
            "gathered {} is not a whole number of {states}-state rows",
            g.len()
        );
        Ok(g)
    }
    /// Windowed encoding, gathered (port of `backend.log_probs_chunked`):
    /// chunks of `win` samples carry `ctx` real samples on both sides; only
    /// the middle `win` frames of each chunk are kept, so the stream tiles
    /// the audio exactly.  Normalisation is per chunk, as the Python path
    /// does; the gather is applied per chunk before readback.
    ///
    /// The trellis comes back one window at a time, so its memory commits as
    /// the file is encoded instead of doubling a single 5 GB allocation on the
    /// way (see `GatheredChunks`), or — in `Lazy` mode — never materialising
    /// the wide trellis at all.
    fn log_probs_chunked(
        &self,
        waveform: &[f32],
        win: usize,
        ctx_sec: f64,
        expanded: &[usize],
        star_state_idx: &[usize],
        form: Form,
    ) -> Result<Trellis> {
        let ctx = (ctx_sec * TARGET_SR as f64) as usize;
        let states = expanded.len();
        if waveform.len() < win {
            let mut input = waveform.to_vec();
            znorm(&mut input);
            let mut scratch = crate::wav2vec2::Scratch::default();
            return Ok(match form {
                Form::Lazy(kind) => {
                    let g = self.forward_lazy(&input, kind, &mut scratch)?;
                    let rows = g.len() / kind.width();
                    Trellis::Lazy(LazyBlocks {
                        blocks: vec![g],
                        kind,
                        num_states: states,
                        row_offset: 0,
                        frames_per_chunk: rows.max(1),
                        spans: vec![(0usize, 0usize, rows)],
                    })
                }
                Form::Gathered => {
                    let g = self.forward_gathered(
                        &input,
                        expanded,
                        &star_state_idx,
                        &mut scratch,
                        None,
                    )?;
                    let frames = g.len() / states;
                    Trellis::Gathered(GatheredChunks {
                        chunks: vec![g],
                        frames_per_chunk: frames.max(1),
                        num_states: states,
                    })
                }
            });
        }
        let ctx_frames = ctx / self.subsampling;
        let win_frames = win / self.subsampling;
        anyhow::ensure!(ctx_frames >= 64, "context must cover the ±64-frame positional conv");
        anyhow::ensure!(win_frames > 0, "window must span at least one frame");

        let n = waveform.len();
        let extension = n.div_ceil(win) * win - n;
        // padded = [ctx zeros | waveform | ctx+extension zeros]
        let padded_len = n + 2 * ctx + extension;
        let ext_frames = ((extension as f64 / TARGET_SR as f64 * self.frame_rate).ceil()) as usize;

        // rows kept per window: the middle win_frames, clamped for a short one
        let kept_of = |rows: usize| (ctx_frames + win_frames).min(rows) - ctx_frames.min(rows);
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut scratch = crate::wav2vec2::Scratch::default();
        // one scratch for the whole file: no per-chunk buffer churn
        //
        // GPU window pipeline: window N's begin arms window N-1's pending —
        // its staging copies are enqueued ahead of window N's compute, which
        // is what overwrites the result buffers — so collecting window N-1
        // below overlaps window N's compute instead of idling the GPU behind
        // a readback. The last window flushes after the loop.
        let mut pending: Option<crate::wav2vec2_gpu::GpuPending> = None;
        // the gather path takes u32s; built once, not per window
        let exp32: Vec<u32> = expanded.iter().map(|&x| x as u32).collect();
        let star32: Vec<u32> = star_state_idx.iter().map(|&x| x as u32).collect();
        let mut start = 0usize; // chunk start inside `padded`
        while start + win + 2 * ctx <= padded_len {
            // chunk = [ctx zeros | win real samples | ctx zeros], gathered from
            // the waveform with zero fill at the file edges
            let w_lo = start as i64 - ctx as i64; // waveform index of chunk sample 0
            let len = win + 2 * ctx;
            let mut chunk = vec![0.0f32; len];
            // one memcpy of the waveform overlap instead of a bounds-checked
            // copy per sample; the zero padding at the file edges stays put
            let (i_lo, i_hi) = (0i64.max(-w_lo), (len as i64).min(n as i64 - w_lo));
            if i_lo < i_hi {
                chunk[i_lo as usize..i_hi as usize]
                    .copy_from_slice(&waveform[(w_lo + i_lo) as usize..(w_lo + i_hi) as usize]);
            }
            znorm(&mut chunk);

            if let Form::Lazy(kind) = form {
                // keep the whole block: no per-chunk copy, and the DP gathers
                // this window's columns a slice at a time when it gets there
                let w = kind.width();
                match &self.tower {
                    Tower::Gpu(gpu) => {
                        let new = match kind {
                            BlockKind::Hidden { .. } => {
                                gpu.forward_hidden_begin(&chunk, pending.as_mut())?
                            }
                            BlockKind::Logits { .. } => {
                                gpu.forward_logits_begin(&chunk, pending.as_mut())?
                            }
                        };
                        if let Some(p) = pending.replace(new) {
                            let g = gpu.collect(p)?;
                            let rows = g.len() / w;
                            blocks.push(g);
                            if crate::alloc_stats::enabled() {
                                let (live, peak) = crate::alloc_stats::stats();
                                eprintln!(
                                    "[alloc] chunk {:>2}: live {live:>12} peak {peak:>12}  {kind:?} {} MB, {} kept rows",
                                    blocks.len() - 1,
                                    blocks[blocks.len() - 1].capacity() * 4 >> 20,
                                    kept_of(rows),
                                );
                            }
                        }
                    }
                    Tower::Cpu(_) => {
                        let g = self.forward_lazy(&chunk, kind, &mut scratch)?;
                        let rows = g.len() / w;
                        blocks.push(g);
                        if crate::alloc_stats::enabled() {
                            let (live, peak) = crate::alloc_stats::stats();
                            eprintln!(
                                "[alloc] chunk {:>2}: live {live:>12} peak {peak:>12}  {kind:?} {} MB, {} kept rows",
                                blocks.len() - 1,
                                blocks[blocks.len() - 1].capacity() * 4 >> 20,
                                kept_of(rows),
                            );
                        }
                    }
                }
            } else {
                // the kept rows only: the gather slices to the kept range
                // before it lands in `blocks`
                match &self.tower {
                    Tower::Gpu(gpu) => {
                        let new = gpu.forward_gathered_begin(
                            &chunk,
                            &exp32,
                            Some((ctx_frames, ctx_frames + win_frames)),
                            &star32,
                            pending.as_mut(),
                        )?;
                        if let Some(p) = pending.replace(new) {
                            blocks.push(gpu.collect(p)?);
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
                    }
                    Tower::Cpu(_) => {
                        let g = self.forward_gathered(
                            &chunk,
                            expanded,
                            star_state_idx,
                            &mut scratch,
                            Some((ctx_frames, ctx_frames + win_frames)),
                        )?;
                        blocks.push(g);
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
                }
            }
            start += win;
        }
        // flush the last window: no next begin to arm it, so its collect
        // waits for its own copy
        if let Some(p) = pending.take() {
            match &self.tower {
                Tower::Gpu(gpu) => blocks.push(gpu.collect(p)?),
                Tower::Cpu(_) => anyhow::bail!("pending result without the GPU tower"),
            }
        }

        // the tail zero-padding contributed ext_frames of frames: drop them
        // from the end of the kept stream, which is the last window's last
        // rows.  (The gathered path truncates the concatenated blocks, which
        // lands on the same rows; a lazily gathered block cannot truncate —
        // the padding sits *before* its end, inside the kept range — so it
        // shortens the last span instead.)
        if ext_frames > 0 {
            if matches!(form, Form::Lazy(_)) {
                // nothing to trim: the block keeps its context rows
            } else if let Some(last) = blocks.last_mut() {
                let keep = last.len().saturating_sub(ext_frames * states);
                last.truncate(keep);
            }
        }
        if let Form::Lazy(kind) = form {
            let w = kind.width();
            let mut spans: Vec<(usize, usize, usize)> = blocks
                .iter()
                .enumerate()
                .map(|(i, b)| (i, ctx_frames.min(b.len() / w), kept_of(b.len() / w)))
                .collect();
            if let Some(last) = spans.last_mut() {
                last.2 = last.2.saturating_sub(ext_frames);
            }
            Ok(Trellis::Lazy(LazyBlocks {
                frames_per_chunk: spans.first().map_or(1, |s| s.2.max(1)),
                row_offset: spans.first().map_or(0, |s| s.1),
                blocks,
                kind,
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
/// Which of the three equivalent trellis storage forms a run takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    Gathered,
    Lazy(BlockKind),
}

/// How many bytes the emissions may occupy between the forward pass and the
/// traceback.  Above this the next-narrower form is used instead: the
/// difference is invisible in the output (same f32s), only in what has to stay
/// resident, and re-deriving costs one lm head per slice.
fn emission_budget() -> usize {
    match std::env::var("CTC_TRELLIS_BUDGET_MB").ok().as_deref().map(str::parse::<usize>) {
        Some(Ok(mb)) => mb << 20,
        _ => 3 << 30, // 3 GiB
    }
}

fn fits(frames: usize, width: usize) -> bool {
    frames.saturating_mul(width).saturating_mul(4) <= emission_budget()
}

impl std::fmt::Debug for Trellis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trellis::Gathered(g) => f.debug_tuple("Gathered").field(&g.chunks.len()).finish(),
            Trellis::Lazy(b) => f
                .debug_struct("Lazy")
                .field("kind", &b.kind)
                .field("blocks", &b.blocks.len())
                .finish(),
        }
    }
}

impl std::fmt::Debug for LazyBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyBlocks")
            .field("kind", &self.kind)
            .field("blocks", &self.blocks.len())
            .field("frames", &self.total_frames())
            .finish()
    }
}

pub(crate) enum Trellis {
    /// One (kept_frames × S) block per window — used when the trellis is
    /// narrower than the vocabulary (a short transcript).
    Gathered(GatheredChunks),
    /// One wider (rows × width) block per window, gathered into S columns a
    /// slice at a time: the logits (41 KB/frame) once the trellis passes
    /// ~5 k characters, the encoder's own output (4 KB/frame) once even that
    /// does not fit.  The DP materialises one slice at a time, so the resident
    /// cost does not grow with the file.
    Lazy(LazyBlocks),
}

/// Per-window blocks plus the row window the DP is allowed to read: the same
/// kept frames the gathered path keeps, addressed inside the wider block.
pub(crate) struct LazyBlocks {
    blocks: Vec<Vec<f32>>,
    /// what a row of a block is, and how wide it is
    kind: BlockKind,
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

impl LazyBlocks {
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
        let w = self.kind.width();
        anyhow::ensure!(w > 0 && self.num_states > 0, "empty trellis");
        anyhow::ensure!(self.frames_per_chunk > 0, "frames_per_chunk must be positive");
        let last = self.spans.len().saturating_sub(1);
        for (i, &(b, row, kept)) in self.spans.iter().enumerate() {
            anyhow::ensure!(b == i, "block span {i} out of order");
            let rows = self.blocks.get(i).map_or(0, |x| x.len() / w);
            anyhow::ensure!(
                self.blocks[i].len() % w == 0,
                "block {i} is not a whole number of {w}-column rows"
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

/// Frames per gathered slice.  The DP walks frames in order, so exactly one
/// slice of trellis columns is resident whatever the file length — 256 rows of
/// S columns is 127 MB at an hour-long file's 124 k states.
const SLICE_FRAMES: usize = 256;

/// What a stored per-window block holds, and how its rows become trellis
/// columns.  All three widths produce the same f32s; they only differ in how
/// much memory has to stay resident between the forward pass and the traceback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlockKind {
    /// `(rows × vocab)` bias-free logits: 41 KB per frame.  The gather is a
    /// log-softmax sweep over the row, no GEMM.
    Logits { vocab: usize },
    /// `(rows × hidden)` normalised encoder stream: 4 KB per frame, ~40x
    /// smaller, and the lm head is re-run to get the columns back.
    Hidden { hidden: usize },
}

impl BlockKind {
    fn width(&self) -> usize {
        match self {
            BlockKind::Logits { vocab } => *vocab,
            BlockKind::Hidden { hidden } => *hidden,
        }
    }
}

/// Turns a stored block's rows into trellis columns, per tower: the CPU tower
/// sweeps the row (or re-runs the lm head), the GPU tower's logits rows are
/// already log-softmaxed (its gather kernel is a plain column copy, so this
/// reproduces it exactly), and the GPU tower's *hidden* rows go through the
/// host-side lm head ([`LmHeadCpu`]) the same way the CPU tower's do.
enum RowGather<'a> {
    Cpu(&'a Model),
    Head(&'a LmHeadCpu),
    Gpu,
}

impl RowGather<'_> {
    /// The `Hidden` kind's re-run of the lm head, over a whole window at
    /// once: the GEMM wants a few thousand rows to reach full throughput, so
    /// slicing the *gathered* columns (not the GEMM) is what keeps the buffer
    /// bounded.  One window's logits is 61 MB and lives in `logits`.
    fn window_logits(
        &self,
        hidden_rows: &[f32],
        rows: usize,
        logits: &mut Vec<f32>,
    ) {
        match self {
            RowGather::Cpu(m) => {
                let vocab = m.vocab_size();
                logits.resize(rows * vocab, 0.0);
                m.lm_head_into_logits(hidden_rows, rows, logits);
            }
            RowGather::Head(h) => {
                let vocab = h.vocab();
                logits.resize(rows * vocab, 0.0);
                h.into_logits(hidden_rows, rows, logits);
            }
            RowGather::Gpu => unreachable!("the GPU tower's logits rows need no head re-run"),
        }
    }

    /// `rows` trellis columns starting at `block_row_lo` of `src` (whose width
    /// the kind gives), with each row's log-softmax normaliser (0 on the GPU
    /// tower, and for the `Hidden` kind, whose rows are pre-lm-head).
    /// `window_logits` is the lazily recomputed `(kept × vocab)` block for that
    /// kind, indexed from the first *kept* row — hence the separate
    /// `window_row_lo`.
    fn slice(
        &self,
        kind: BlockKind,
        src: &[f32],
        block_row_lo: usize,
        window_row_lo: usize,
        rows: usize,
        window_logits: &[f32],
        cols: &[i32],
        out: &mut [f32],
        norm: &mut [f32],
    ) {
        let s = cols.len();
        match (self, kind) {
            // GPU logits rows are log-probs already: pure column copy, and the
            // recorded normaliser stays 0 (the value path adds nothing)
            (RowGather::Gpu, _) | (RowGather::Head(_), BlockKind::Logits { .. }) => {
                let w = kind.width();
                src[block_row_lo * w..][..rows * w]
                    .par_chunks_exact(w)
                    .zip(out.par_chunks_mut(s))
                    .for_each(|(row, dst)| {
                        for (j, &c) in cols.iter().enumerate() {
                            dst[j] = row[c as usize];
                        }
                    });
                norm.fill(0.0);
            }
            (RowGather::Cpu(m), BlockKind::Logits { vocab }) => {
                src[block_row_lo * vocab..][..rows * vocab]
                    .par_chunks_exact(vocab)
                    .zip(out.par_chunks_mut(s))
                    .zip(norm.par_iter_mut())
                    .for_each(|((row, dst), c)| *c = m.gather_logits_row(row, cols, dst));
            }
            (RowGather::Cpu(m), BlockKind::Hidden { .. }) => {
                let vocab = m.vocab_size();
                window_logits[window_row_lo * vocab..][..rows * vocab]
                    .par_chunks_exact(vocab)
                    .zip(out.par_chunks_mut(s))
                    .zip(norm.par_iter_mut())
                    .for_each(|((x, dst), c)| *c = m.gather_lp_row(x, cols, dst));
            }
            // the GPU tower's hidden rows: same gather, host-side head
            (RowGather::Head(h), BlockKind::Hidden { .. }) => {
                let vocab = h.vocab();
                window_logits[window_row_lo * vocab..][..rows * vocab]
                    .par_chunks_exact(vocab)
                    .zip(out.par_chunks_mut(s))
                    .zip(norm.par_iter_mut())
                    .for_each(|((x, dst), c)| *c = h.gather_lp_row(x, cols, dst));
            }
        }
    }

    /// Width of a computed lm-head logit row (the vocabulary), which is what
    /// `window_logits` holds — not `BlockKind::width`, which is the width of
    /// the *stored* block and is the encoder stream for `Hidden`.
    fn logits_width(&self) -> usize {
        match self {
            RowGather::Cpu(m) => m.vocab_size(),
            RowGather::Head(h) => h.vocab(),
            RowGather::Gpu => 0,
        }
    }

    /// One trellis column of a computed lm-head logit row, reusing a
    /// normaliser a previous pass recorded.  Bit-identical to the `slice`
    /// call that produced it: same operands, same `x + bias - c` arithmetic.
    fn lp_value(&self, logits: &[f32], col: usize, c: f32) -> f32 {
        match self {
            RowGather::Cpu(m) => logits[col] + m.lm_bias()[col] - c,
            RowGather::Head(h) => logits[col] + h.bias()[col] - c,
            RowGather::Gpu => logits[col],
        }
    }

    /// One trellis column of a stored `src` row, reusing a normaliser a
    /// previous pass recorded.  Only available when the stored rows *are* the
    /// logit row.
    fn value(&self, kind: BlockKind, src: &[f32], col: usize, c: f32) -> f32 {
        match (self, kind) {
            (RowGather::Cpu(m), BlockKind::Logits { .. }) => src[col] + m.lm_bias()[col] - c,
            // GPU-produced logits rows are log-probs already: no bias, no normaliser
            (RowGather::Head(_), BlockKind::Logits { .. })
            | (RowGather::Gpu, BlockKind::Logits { .. }) => src[col],
            // hidden rows are pre-lm-head: the column only exists once gathered
            (RowGather::Cpu(_), BlockKind::Hidden { .. })
            | (RowGather::Head(_), BlockKind::Hidden { .. })
            | (RowGather::Gpu, BlockKind::Hidden { .. }) => f32::NAN,
        }
    }
}

/// [`Emissions`] over a lazily gathered trellis ([`Trellis::Logits`] /
/// [`Trellis::Hidden`]): the DP walks forward, so each slice of trellis columns
/// is gathered the first time a frame inside it is reached and dropped when
/// the walk moves past it.  Peak cost is one slice, not the file's.
struct LazyEmissions<'a> {
    blocks: &'a LazyBlocks,
    kind: BlockKind,
    gather: RowGather<'a>,
    /// the expanded labels, as the gather kernel wants them
    cols: Vec<i32>,
    /// (window, first row, rows) of the one slice currently gathered
    cache: std::cell::RefCell<Option<(usize, usize, usize, Vec<f32>)>>,
    /// the `Hidden` kind's recomputed lm-head logits for the window in flight
    /// — one window, never the file, because the GEMM wants thousands of rows
    /// to be worth running and the gather is what gets sliced.
    window_logits: std::cell::RefCell<Option<(usize, Vec<f32>)>>,
    /// per frame, the log-softmax normaliser recorded while the DP gathered
    /// its slice (0 on the GPU tower, whose blocks are already log-probs).
    /// This is what lets the post-traceback pass read the path's column out of
    /// the stored rows instead of gathering every slice a second time — 46 k
    /// reads instead of 5.8 GB of writes.
    norm: std::cell::RefCell<Vec<f32>>,
    /// the trellis **state** indices the `<star>` sentinel occupies (each
    /// target's odd state), so both read paths can score them
    /// [`CTC_STAR_SCORE`].  The gathered form gets this for
    /// free (`forward_gathered` overwrites the column once, up front); a lazy
    /// block re-derives each slice from stored rows, so without this it would
    /// read the *blank* log-prob the star's column was gathered from — about
    /// -0.005 against the constant's -1.0.  A path that parks on the star for
    /// a different number of frames is a different path, and every boundary
    /// after it moves: measured on the dub fixture, 17 token starts drifted by
    /// up to 52 frames and the run's log_prob by 28.85.
    ///
    /// State indices, **not** vocabulary columns: `gather_ids` maps the star
    /// onto `BLANK_ID`, so `cols[star_state]` is the blank's own column, and a
    /// membership test against column ids would match every blank state as
    /// well.  `score_path` has to test the state.
    ///
    /// A set, not a list: `score_path` asks once per frame per window, and a
    /// linear scan over a few hundred stars is ~10^7 comparisons on an hour.
    star_cols: std::collections::HashSet<usize>,
}

impl<'a> LazyEmissions<'a> {
    fn new(
        blocks: &'a LazyBlocks,
        gather: RowGather<'a>,
        expanded: &[usize],
        star_state_idx: &[usize],
    ) -> Self {
        let frames = blocks.total_frames();
        LazyEmissions {
            blocks,
            kind: blocks.kind,
            gather,
            cols: expanded.iter().map(|&x| x as i32).collect(),
            cache: std::cell::RefCell::new(None),
            window_logits: std::cell::RefCell::new(None),
            norm: std::cell::RefCell::new(vec![0.0; frames]),
            star_cols: star_state_idx
                .iter()
                .copied()
                .filter(|&i| i < blocks.num_states)
                .collect(),        }
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

    /// The window's recomputed logits, running the lm head if this window is
    /// not the one in flight.  One window, never the file.
    fn window_logits_for(&self, b: usize) -> std::cell::RefMut<'_, [f32]> {
        use std::cell::RefMut;
        let blocks = self.blocks;
        let w = self.kind.width();
        let (_, row0, kept) = blocks.spans[b];
        let mut wl = self.window_logits.borrow_mut();
        if wl.as_ref().map(|(wb, _)| *wb != b).unwrap_or(true) {
            let src = &blocks.blocks[b][row0 * w..][..kept * w];
            match &mut *wl {
                // the buffer is reused between windows, but the index has to
                // follow it or every frame re-runs the head
                Some((wb, buf)) => {
                    self.gather.window_logits(src, kept, buf);
                    *wb = b;
                }
                None => {
                    let mut buf = Vec::new();
                    self.gather.window_logits(src, kept, &mut buf);
                    *wl = Some((b, buf));
                }
            }
        }
        RefMut::map(wl, |c: &mut Option<(usize, Vec<f32>)>| {
            c.as_mut().expect("just filled").1.as_mut_slice()
        })
    }

    /// Gather the slice of trellis columns `[row0+lo, +rows)` of window
    /// `block` unless it is already the cached one.  `window_logits` is the
    /// `Hidden` kind's recomputed logit rows (empty for the other kinds,
    /// which read their stored block instead).
    fn fill_slice(&self, block: usize, row0: usize, lo: usize, rows: usize, window_logits: &[f32]) {
        let b = self.blocks;
        let s = b.num_states;
        let need = rows * s;
        let t0 = b.frame_base(block) + lo;
        let mut cache = self.cache.borrow_mut();
        let fresh = match cache.as_mut() {
            Some((cb, cl, cr, buf)) => {
                if *cb == block && *cl == row0 + lo && *cr == rows && buf.len() == need {
                    false
                } else {
                    cache.replace((block, row0 + lo, rows, vec![0.0f32; need]));
                    true
                }
            }
            None => {
                cache.replace((block, row0 + lo, rows, vec![0.0f32; need]));
                true
            }
        };
        if !fresh {
            return;
        }
        let out = &mut cache.as_mut().expect("just filled").3;
        let mut norm = self.norm.borrow_mut();
        self.gather.slice(
            self.kind,
            &b.blocks[block],
            row0 + lo,
            lo,
            rows,
            window_logits,
            &self.cols,
            out,
            &mut norm[t0..t0 + rows],
        );
        // the star columns are the reference's appended zero column, not a real
        // log-prob; `forward_gathered` stamps them once and the lazy forms have
        // to match or the DP walks a different path (see `star_cols`).
        for &c in &self.star_cols {
            for row in out.chunks_exact_mut(s) {
                row[c] = CTC_STAR_SCORE;
            }
        }
    }

    /// The gathered trellis slice holding frame `t`, as (buffer, first row of
    /// the slice within the window, rows in it).  Gathers on first use;
    /// consecutive frames of the same slice are then a borrow, and the buffer
    /// is reused across slices.
    fn slice_at(&self, t: usize) -> (std::cell::RefMut<'_, [f32]>, usize, usize) {
        use std::cell::RefMut;
        let b = self.blocks;
        let (block, row0, kept) = b.span_of(t);
        let r = t - b.frame_base(block);
        let lo = (r / SLICE_FRAMES) * SLICE_FRAMES;
        let rows = (kept - lo).min(SLICE_FRAMES);
        if matches!(self.kind, BlockKind::Hidden { .. }) {
            let wl = self.window_logits_for(block);
            self.fill_slice(block, row0, lo, rows, wl.as_ref());
        } else {
            self.fill_slice(block, row0, lo, rows, &[]);
        }
        let cache = self.cache.borrow_mut();
        (
            RefMut::map(cache, |c: &mut Option<(usize, usize, usize, Vec<f32>)>| {
                c.as_mut().expect("just filled").3.as_mut_slice()
            }),
            lo,
            rows,
        )
    }
}

impl Emissions for LazyEmissions<'_> {
    fn fill_emit(&self, t: usize, emit: &mut [f64], _token_ids: &[usize]) {
        let s = self.blocks.num_states;
        let (block, lo, _) = self.slice_at(t);
        let r = t % self.blocks.frames_per_chunk - lo;
        let row = &block[r * s..][..s];
        for (e, &v) in emit.iter_mut().zip(row) {
            *e = v as f64;
        }
    }

    /// The band `[lo, hi]` only — on an hour's trellis the mid-file band is
    /// the whole row, but the two ramps shrink the copy with it.
    fn fill_emit_band(&self, t: usize, emit: &mut [f64], lo: usize, hi: usize, _token_ids: &[usize]) {
        let s = self.blocks.num_states;
        let (block, slice_lo, _) = self.slice_at(t);
        let r = t % self.blocks.frames_per_chunk - slice_lo;
        let row = &block[r * s..][..s];
        for (e, &v) in emit[lo..=hi].iter_mut().zip(row[lo..=hi].iter()) {
            *e = v as f64;
        }
    }

    /// One column of `t`'s row.  The slice holding `t` has to have been
    /// gathered at least once for its normaliser to exist — the DP reads frame
    /// 0's two states *before* its first `fill_emit`, so this may be what
    /// triggers the gather.  After the DP every slice is normalised, and for
    /// the `Logits` kind this is then a single indexed read out of the stored
    /// rows — no second gather of the whole file.
    fn score(&self, t: usize, st: usize) -> f32 {
        let b = self.blocks;
        if matches!(self.kind, BlockKind::Hidden { .. }) {
            // pre-lm-head rows: the column only exists in the gathered slice
            let (block, lo, _) = self.slice_at(t);
            let s = b.num_states;
            return block[(t % b.frames_per_chunk - lo) * s + st];
        }
        // The star's column was gathered from BLANK_ID, so the stored row's
        // own value is the blank log-prob, not the reference's constant.  This
        // read happens before the DP's first `fill_emit` (the DP seeds
        // `prev[0]`/`prev[1]` from frame 0 through here), so without the
        // check the first frames score the star on the blank and the path
        // parks on it from frame 0 instead of waiting for the model to make
        // it likely.
        if self.star_cols.contains(&st) {
            return CTC_STAR_SCORE;
        }
        let col = self.cols[st] as usize;
        let (block, row, kept) = b.span_of(t);
        let _ = self.slice_at(t); // records this row's normaliser
        let w = b.kind.width();
        let r = t - b.frame_base(block);
        debug_assert!(r < kept);
        let src = &b.blocks[block][(row + r) * w..][..w];
        self.gather.value(self.kind, src, col, self.norm.borrow()[t])
    }

    /// The path's per-frame scores, window by window, without re-gathering a
    /// window's S columns: both lazily stored forms still have the logit row
    /// at hand once the window's normalisers are known (`Hidden` by re-running
    /// the head, `Logits` from the stored block), so this is one indexed read
    /// per frame.
    fn score_path(&self, states: &[i32], out: &mut [f64]) {
        let b = self.blocks;
        let fpc = b.frames_per_chunk;
        let norm = self.norm.borrow();
        for (bi, &(_, row, kept)) in b.spans.iter().enumerate() {
            let base = bi * fpc;
            if matches!(self.kind, BlockKind::Hidden { .. }) {
                let wl = self.window_logits_for(bi);
                let vocab = self.gather.logits_width();
                for r in 0..kept {
                    let t = base + r;
                    let st = states[t] as usize;
                    if self.star_cols.contains(&st) {
                        // same constant `fill_slice` wrote, so the path's own
                        // score matches what the DP saw
                        out[t] = CTC_STAR_SCORE as f64;
                    } else {
                        let col = self.cols[st] as usize;
                        out[t] =
                            self.gather.lp_value(&wl[r * vocab..][..vocab], col, norm[t]) as f64;
                    }
                }
            } else {
                let w = b.kind.width();
                let src = &b.blocks[bi][row * w..][..kept * w];
                for r in 0..kept {
                    let t = base + r;
                    let st = states[t] as usize;
                    out[t] = if self.star_cols.contains(&st) {
                        CTC_STAR_SCORE as f64
                    } else {
                        let col = self.cols[st] as usize;
                        self.gather.value(self.kind, &src[r * w..][..w], col, norm[t]) as f64
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of putting a timestamp on text: the text must come out
    /// the other side untouched.
    ///
    /// Every property below is one way this has actually broken. A character
    /// the vocabulary had no target for went missing; a cue boundary handed one
    /// character to two cues and printed it twice; and an interpolated run split
    /// across a shared frame came out with a start later than the placed
    /// character that follows it, so the sequence ran backwards.
    #[test]
    fn unaligned_characters_are_restored_without_reordering() {
        // "拉蔻就" -- 蔻 is not in the vocabulary, and the neighbours share a
        // frame, which is the case that produced an inverted start.
        let text = "拉蔻就";
        let mut toks: Vec<TokenAlignment> = Vec::new();
        // 拉 spans 71.500..71.520 (padded forward past 就's onset),
        // 就 spans 71.500..71.620. 蔻 has no target at all.
        for (i, (piece, sf, ef)) in
            [("拉", 3548u64, 3575u64), ("就", 3575u64, 3580u64)].iter().enumerate()
        {
            toks.push(TokenAlignment {
                index: i,
                token_id: 7 + i,
                piece: piece.to_string(),
                word_id: 0,
                start: *sf as f64 * 0.02,
                end: *ef as f64 * 0.02 + 0.02,
                start_frame: *sf as i64,
                end_frame: *ef as i64,
                score: -0.1,
                inferred: false,
            });
        }
        // `src[i]` is the byte offset in `text` of the character that target
        // `i` came from, and `usize::MAX` marks a `<star>`, which has no
        // character. One entry per target, so the two lists are parallel.
        // These are BYTE offsets: every character here is three bytes of UTF-8,
        // so 拉 is at 0 and 就 at 6, not at 2.
        let src: Vec<usize> = vec![0, 6];
        let out = place_unmeasured(&toks, text, &src, 50.0, 4.0);

        let got: String = out.iter().map(|t| t.piece.as_str()).collect();
        assert_eq!(got, "拉蔻就", "the transcript comes back character for character");
        assert!(out[1].inferred && !out[0].inferred && !out[2].inferred);

        for w in out.windows(2) {
            assert!(
                w[1].start >= w[0].start - 1e-9,
                "start ran backwards: {} ({}) then {} ({})",
                w[0].piece, w[0].start, w[1].piece, w[1].start
            );
        }
        // and the interpolated span may not reach past the character it precedes
        assert!(out[1].end <= out[2].start + 1e-9, "interpolated span overran");
    }

    /// A character with no target at the START of a source word the rest of
    /// which was placed belongs to that word, not to a new one.
    ///
    /// `贅沢` is one word: `贅` is outside the vocabulary and `沢` is not. The
    /// star that opens the word is emitted while the tokenizer is still
    /// counting it, so it lands between the two, carrying the PREVIOUS word's
    /// id. Giving `贅` an id of its own and letting that star vote for a
    /// boundary both split the word, and `words` printed `贅 沢。` -- a space
    /// the transcript does not contain.
    #[test]
    fn a_missing_leading_character_joins_the_word_it_was_written_in() {
        let text = "。 贅沢。";
        let at: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        // at == [0, 3, 4, 7, 10] for 。, the space, 贅, 沢, 。
        let tok = |i: usize, piece: &str, word_id: usize, end: f64| TokenAlignment {
            index: i,
            token_id: 7 + i,
            piece: piece.to_string(),
            word_id,
            start: end - 0.2,
            end,
            start_frame: (end * 50.0) as i64,
            end_frame: (end * 50.0) as i64 + 9,
            score: -0.1,
            inferred: false,
        };
        // what the DP returned: the stars, then the three characters it had a
        // target for
        let toks = vec![
            tok(0, "<star>", usize::MAX, 0.0),
            tok(1, "。", 1, 0.2),
            tok(2, "<star>", 1, 0.2),
            tok(3, "沢", 2, 0.6),
            tok(4, "。", 2, 0.8),
        ];
        let src = vec![usize::MAX, at[0], usize::MAX, at[3], at[4]];
        let out = place_unmeasured(&toks, text, &src, 50.0, 1.0);
        let got: String = out.iter().map(|t| t.piece.as_str()).collect();
        assert_eq!(got, "<star>。<star>贅沢。", "every character comes back once");
        let zei = out.iter().position(|t| t.piece == "贅").unwrap();
        let sawa = out.iter().position(|t| t.piece == "沢").unwrap();
        assert_eq!(
            out[zei].word_id, out[sawa].word_id,
            "贅 was put in a word of its own, so 贅沢 renders as 贅 沢"
        );
        // and the two really are one word to `build_words`: `贅沢。` is a run of
        // characters, so it is reported a character at a time -- except that the
        // full stop rides in the character it follows, since a mark occupies no
        // frames and would otherwise be a row that is only a point in time.
        // One gap, the one the transcript has before `贅`.
        let words = crate::spans::build_words(&out);
        let rendered: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(rendered, ["。", "贅", "沢。"]);
        let mut text_back = String::new();
        for (i, w) in words.iter().enumerate() {
            if i > 0 && w.space_before {
                text_back.push(' ');
            }
            text_back.push_str(&w.text);
        }
        assert_eq!(text_back, "。 贅沢。", "the transcript is not the one we were given");
    }

    /// A whole word the vocabulary dropped keeps its own word id, or it would
    /// render glued to the word before it.
    #[test]
    fn a_wholly_unalignable_word_stays_separate() {
        let text = "好 野";
        let toks = vec![TokenAlignment {
            index: 0,
            token_id: 7,
            piece: "好".to_string(),
            word_id: 1,
            start: 0.0,
            end: 0.2,
            start_frame: 0,
            end_frame: 9,
            score: -0.1,
            inferred: false,
        }];
        // only 好 has a target, at text offset 0; 野 was dropped whole
        let src = vec![0];
        let out = place_unmeasured(&toks, text, &src, 50.0, 2.0);
        let got: String = out.iter().map(|t| t.piece.as_str()).collect();
        assert_eq!(got, "好野");
        assert_ne!(out[1].word_id, out[0].word_id, "the missing word joined its neighbour");
    }

    /// The GPU tower's hidden form: the stored rows are pre-lm-head, and
    /// `RowGather::Head` re-runs the host-side head per window.  The whole
    /// per-window machinery (context rows, short last window, one-window
    /// logit cache, normaliser recording) must land on the trellis a direct
    /// head re-run produces, bit for bit.  Like the test below it, this is a
    /// pure addressing test — a synthetic head, no model needed.
    #[test]
    fn lazy_hidden_trellis_matches_direct_head_gather() {
        let (hidden_dim, vocab, l) = (8usize, 23usize, 5usize);
        let blank = 0usize;
        let token_ids: Vec<usize> = (1..=l).map(|i| 1 + (i * 4) % (vocab - 1)).collect();
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let expanded = crate::viterbi::build_expanded_labels(&token_ids, blank);
        let s = expanded.len();
        let cols: Vec<i32> = expanded.iter().map(|&x| x as i32).collect();

        let head = crate::wav2vec2::LmHeadCpu {
            linear: crate::wav2vec2::Linear {
                w: (0..vocab * hidden_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect(),
                b: (0..vocab).map(|i| -0.01 * i as f32).collect(),
                out: vocab,
                in_: hidden_dim,
            },
        };

        // three windows of `per` kept rows, the last short; context rows in
        // front of each block
        let (row_offset, kept_last, per) = (5usize, 7usize, 13usize);
        let rows_per = row_offset + per;
        let block_all: Vec<f32> = (0..3 * rows_per * hidden_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.11)
            .collect();
        let mut blocks = Vec::new();
        let mut spans = Vec::new();
        for i in 0..3 {
            let kept = if i == 2 { kept_last } else { per };
            blocks.push(
                block_all[i * rows_per * hidden_dim..(i * rows_per + rows_per) * hidden_dim]
                    .to_vec(),
            );
            spans.push((i, row_offset, kept));
        }
        let lb = LazyBlocks {
            blocks,
            kind: BlockKind::Hidden { hidden: hidden_dim },
            num_states: s,
            frames_per_chunk: per,
            row_offset,
            spans,
        };

        // reference: head over each block's kept rows, gathered straight —
        // no windows, no slices, no caches
        let mut flat = Vec::new();
        for (bi, &(_, row, kept)) in lb.spans.iter().enumerate() {
            let src = &lb.blocks[bi][row * hidden_dim..(row + kept) * hidden_dim];
            let mut logits = vec![0f32; kept * vocab];
            head.into_logits(src, kept, &mut logits);
            for r in 0..kept {
                let mut out = vec![0f32; s];
                head.gather_lp_row(&logits[r * vocab..(r + 1) * vocab], &cols, &mut out);
                flat.extend_from_slice(&out);
            }
        }
        let gc = GatheredChunks {
            chunks: flat.chunks(per * s).map(|c| c.to_vec()).collect(),
            frames_per_chunk: per,
            num_states: s,
        };
        gc.validate().unwrap();

        let em = LazyEmissions::new(&lb, RowGather::Head(&head), &expanded, &[]);
        em.validate().unwrap();
        let total_kept = 2 * per + kept_last;
        assert_eq!(em.total_frames(), total_kept);

        // one target per word is what the synthetic paths mean
        let word_ids: Vec<usize> = (0..token_ids.len()).collect();
        let want = ctc_forced_align_gathered_with_word_ids(
            &gc, &token_ids, 50.0, Some(&pieces), &word_ids, false).unwrap();
        let got =
            ctc_forced_align_emissions_with_word_ids(&em, total_kept, &token_ids, 50.0, Some(&pieces), &word_ids).unwrap();

        assert_eq!(got.tokens.len(), want.tokens.len());
        for (g, w) in got.tokens.iter().zip(&want.tokens) {
            assert_eq!((g.start_frame, g.end_frame), (w.start_frame, w.end_frame));
            assert_eq!(g.score.to_bits(), w.score.to_bits(), "token score bits");
        }
        assert_eq!(got.log_prob.to_bits(), want.log_prob.to_bits());
        for (a, b) in got.frame_scores.iter().zip(&want.frame_scores) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// The `<star>` sentinel is a real target, and its column is the
    /// reference's *appended* zero column rather than a real log-prob.  The
    /// gathered form stamps it once up front; a lazy block re-derives every
    /// slice from stored rows, so it has to stamp it again or the DP reads the
    /// blank log-prob the column was gathered from (~-0.005) where the
    /// constant is -1.0 — the star then parks for a different number of frames
    /// and every later boundary moves with it.
    ///
    /// Both read paths matter: `fill_slice` feeds the DP forward, `score_path`
    /// rescores the chosen path afterwards.  A fix in only one of them leaves
    /// a token whose timestamp and whose score disagree.
    #[test]
    fn lazy_star_columns_score_the_reference_constant() {
        let (hidden_dim, vocab, l) = (8usize, 23usize, 5usize);
        let blank = 0usize;
        // target 1 is the star: its expanded odd state is column 3
        let token_ids: Vec<usize> = vec![1, 2, 3, 1, 4];
        let star = 1usize;
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let expanded = crate::viterbi::build_expanded_labels(&token_ids, blank);
        let s = expanded.len();
        let cols: Vec<i32> = expanded.iter().map(|&x| x as i32).collect();
        let star_state_idx: Vec<usize> = (0..token_ids.len())
            .filter(|&i| token_ids[i] == star)
            .map(|i| 2 * i + 1)
            .collect();
        assert_eq!(star_state_idx, vec![1, 7], "star sits on each target's odd state");

        let head = crate::wav2vec2::LmHeadCpu {
            linear: crate::wav2vec2::Linear {
                w: (0..vocab * hidden_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect(),
                b: (0..vocab).map(|i| -0.01 * i as f32).collect(),
                out: vocab,
                in_: hidden_dim,
            },
        };

        let (row_offset, kept_last, per) = (5usize, 7usize, 13usize);
        let rows_per = row_offset + per;
        let block_all: Vec<f32> = (0..3 * rows_per * hidden_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.11)
            .collect();
        let mut blocks = Vec::new();
        let mut spans = Vec::new();
        for i in 0..3 {
            let kept = if i == 2 { kept_last } else { per };
            blocks.push(
                block_all[i * rows_per * hidden_dim..(i * rows_per + rows_per) * hidden_dim]
                    .to_vec(),
            );
            spans.push((i, row_offset, kept));
        }
        let lb = LazyBlocks {
            blocks,
            kind: BlockKind::Hidden { hidden: hidden_dim },
            num_states: s,
            frames_per_chunk: per,
            row_offset,
            spans,
        };

        // reference: the gathered form, stamped exactly as forward_gathered does
        let mut flat = Vec::new();
        for (bi, &(_, row, kept)) in lb.spans.iter().enumerate() {
            let src = &lb.blocks[bi][row * hidden_dim..(row + kept) * hidden_dim];
            let mut logits = vec![0f32; kept * vocab];
            head.into_logits(src, kept, &mut logits);
            for r in 0..kept {
                let mut out = vec![0f32; s];
                head.gather_lp_row(&logits[r * vocab..(r + 1) * vocab], &cols, &mut out);
                for &si in &star_state_idx {
                    out[si] = CTC_STAR_SCORE;
                }
                flat.extend_from_slice(&out);
            }
        }
        let gc = GatheredChunks {
            chunks: flat.chunks(per * s).map(|c| c.to_vec()).collect(),
            frames_per_chunk: per,
            num_states: s,
        };
        gc.validate().unwrap();

        let em = LazyEmissions::new(&lb, RowGather::Head(&head), &expanded, &star_state_idx);
        em.validate().unwrap();
        let total_kept = 2 * per + kept_last;

        // every gathered slice carries the constant, never the blank score
        for t in 0..total_kept {
            let (buf, lo, _rows) = em.slice_at(t);
            let r = t % em.blocks.frames_per_chunk - lo;
            let frame = &buf[r * s..][..s];
            for &si in &star_state_idx {
                assert_eq!(
                    frame[si], CTC_STAR_SCORE,
                    "frame {t} star column {si} is not the reference constant"
                );
                // -1.0, not the ~-0.005 blank log-prob the column was
                // gathered from: below the guard, so a regression to the
                // ungathered value cannot pass as equal
                assert!(
                    frame[si] < -0.5,
                    "frame {t}: the constant must not read back as a blank log-prob"
                );
            }
        }

        // The DP seeds `prev[0]`/`prev[1]` from frame 0 through `score()`, which
        // for the `Logits` kind reads the stored row directly and so has to
        // apply the constant on its own — a slice gathered later cannot help
        // here, and the path ends up parking on the star from frame 0.
        for t in 0..total_kept.min(4) {
            for &si in &star_state_idx {
                assert_eq!(
                    em.score(t, si),
                    CTC_STAR_SCORE,
                    "score({t}, star state {si}) is not the reference constant"
                );
            }
        }

        // and the two forms have to agree on the whole alignment
        let word_ids: Vec<usize> = (0..token_ids.len()).collect();
        let want = ctc_forced_align_gathered_with_word_ids(
            &gc, &token_ids, 50.0, Some(&pieces), &word_ids, false).unwrap();
        let got = ctc_forced_align_emissions_with_word_ids(
            &em, total_kept, &token_ids, 50.0, Some(&pieces), &word_ids).unwrap();

        assert_eq!(got.tokens.len(), want.tokens.len());
        for (g, w) in got.tokens.iter().zip(&want.tokens) {
            assert_eq!(
                (g.start_frame, g.end_frame),
                (w.start_frame, w.end_frame),
                "token {} moved", w.piece
            );
            assert_eq!(g.score.to_bits(), w.score.to_bits(), "token score bits");
        }
        assert_eq!(got.log_prob.to_bits(), want.log_prob.to_bits());
        for (i, (a, b)) in got.frame_scores.iter().zip(&want.frame_scores).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "frame {i} score bits");
        }
        let _ = l;
    }

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
        let lb = LazyBlocks {
            blocks,
            kind: BlockKind::Logits { vocab },
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

        let em = LazyEmissions::new(&lb, RowGather::Gpu, &expanded, &[]);
        em.validate().unwrap();
        assert_eq!(em.total_frames(), total_kept);
        assert_eq!(gc.total_frames(), total_kept);

        // one target per word is what the synthetic paths mean
        let word_ids: Vec<usize> = (0..token_ids.len()).collect();
        let want = ctc_forced_align_gathered_with_word_ids(
            &gc, &token_ids, 50.0, Some(&pieces), &word_ids, false).unwrap();
        let got = ctc_forced_align_emissions_with_word_ids(&em, total_kept, &token_ids, 50.0, Some(&pieces), &word_ids)
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

    /// The `Logits` kind's `score()` is a third read path, and the one the DP
    /// hits first: it seeds `prev[0]`/`prev[1]` from frame 0 before any
    /// `fill_emit` has gathered a slice.  It reads the stored row directly, so
    /// the star constant has to be applied there too — otherwise the opening
    /// frames score the star on the blank column and the path parks on it from
    /// frame 0, which moves every boundary by the length of the leading blank
    /// run.  Measured on dub: 52 frames, and log_prob off by 0.98 even after
    /// `fill_slice` and `score_path` were already fixed.
    #[test]
    fn lazy_logits_star_seed_uses_the_reference_constant() {
        let (vocab, l) = (23usize, 5usize);
        let blank = 0usize;
        let token_ids: Vec<usize> = vec![1, 2, 3, 1, 4];
        let star = 1usize;
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let expanded = crate::viterbi::build_expanded_labels(&token_ids, blank);
        let s = expanded.len();
        let star_state_idx: Vec<usize> = (0..token_ids.len())
            .filter(|&i| token_ids[i] == star)
            .map(|i| 2 * i + 1)
            .collect();

        let (row_offset, kept_last, per) = (5usize, 7usize, 13usize);
        let rows_per = row_offset + per;
        // stored logit rows; the blank column is deliberately attractive
        // (-0.001) so an un-stamped star scores better than a real blank and
        // the path visibly prefers to sit on it
        let block_all: Vec<f32> = (0..3 * rows_per * vocab)
            .map(|i| if i % vocab == blank { -0.001 } else { -((i % 37) as f32) * 0.07 - 0.3 })
            .collect();
        let mut blocks = Vec::new();
        let mut spans = Vec::new();
        for i in 0..3 {
            let kept = if i == 2 { kept_last } else { per };
            blocks.push(
                block_all[i * rows_per * vocab..(i + 1) * rows_per * vocab].to_vec(),
            );
            spans.push((i, row_offset, kept));
        }
        let lb = LazyBlocks {
            blocks,
            kind: BlockKind::Logits { vocab },
            num_states: s,
            frames_per_chunk: per,
            row_offset,
            spans,
        };

        let em = LazyEmissions::new(&lb, RowGather::Gpu, &expanded, &star_state_idx);
        em.validate().unwrap();
        let total_kept = 2 * per + kept_last;

        // the stored blank really is the tempting value, so the assertions
        // below cannot pass by accident
        let raw_blank = em.blocks.blocks[0][(row_offset as usize) * vocab + blank];
        assert!(raw_blank > -0.01, "blank column should look attractive, got {raw_blank}");

        for t in 0..total_kept {
            for &si in &star_state_idx {
                assert_eq!(
                    em.score(t, si),
                    CTC_STAR_SCORE,
                    "score({t}, star state {si}) read the stored column instead of the constant"
                );
            }
        }

        // and the whole alignment has to match the gathered reference, which
        // got the same constant stamped up front
        let mut flat = Vec::with_capacity(total_kept * s);
        for (b, &(_, row, kept)) in lb.spans.iter().enumerate() {
            for r in 0..kept {
                let src = &lb.blocks[b][(row + r) * vocab..][..vocab];
                for (j, &c) in expanded.iter().enumerate() {
                    flat.push(if star_state_idx.contains(&j) {
                        CTC_STAR_SCORE
                    } else {
                        src[c]
                    });
                }
            }
        }
        let gc = GatheredChunks {
            chunks: flat.chunks(per * s).map(|c| c.to_vec()).collect(),
            frames_per_chunk: per,
            num_states: s,
        };
        gc.validate().unwrap();

        let word_ids: Vec<usize> = (0..l).collect();
        let want = ctc_forced_align_gathered_with_word_ids(
            &gc, &token_ids, 50.0, Some(&pieces), &word_ids, false).unwrap();
        let got = ctc_forced_align_emissions_with_word_ids(
            &em, total_kept, &token_ids, 50.0, Some(&pieces), &word_ids).unwrap();

        for (g, w) in got.tokens.iter().zip(&want.tokens) {
            assert_eq!(
                (g.start_frame, g.end_frame),
                (w.start_frame, w.end_frame),
                "token {} moved",
                w.piece
            );
        }
        assert_eq!(got.log_prob.to_bits(), want.log_prob.to_bits());
    }
}
