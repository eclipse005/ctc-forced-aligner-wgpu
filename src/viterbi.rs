//! CTC forced alignment: token-level timestamps from frame log-probabilities.
//!
//! Port of the Python reference (`omni_align/align.py`) with *identical*
//! semantics, down to the tie-breaking: numpy `argmax` over the stacked
//! `[stay, advance, skip]` candidates takes the first maximum, so on exact
//! ties stay wins over advance wins over skip; the path score accumulates in
//! f64 over f32 log-prob rows; the trellis may terminate on the last blank or
//! the last token.
//!
//! Graves et al., *Connectionist Temporal Classification* (2006), Sec. 4.1.

use anyhow::Context;
use rayon::prelude::*;

/// One aligned token and its time span.
#[derive(Debug, Clone)]
pub struct TokenAlignment {
    pub index: usize,
    pub token_id: usize,
    pub piece: String,
    /// Index of the source word this token came from. The target sequence is
    /// the letters of every word laid end to end, with nothing between the
    /// words — a space token there would be a target the reference never had
    /// and would push the sequence past what the frame count can carry. Word
    /// boundaries therefore have to come from the SIDE, not from a token, or
    /// `build_words` sees one long run of letters and emits a single word.
    pub word_id: usize,
    pub start: f64,
    pub end: f64,
    pub start_frame: i64,
    pub end_frame: i64,
    /// Mean per-frame log-probability over the token's frames.
    pub score: f64,
}

impl TokenAlignment {
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
    pub fn mid(&self) -> f64 {
        0.5 * (self.start + self.end)
    }
}

#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub tokens: Vec<TokenAlignment>,
    pub frames: usize,
    pub frame_rate: f64,
    /// Total path log-probability (higher = better match).
    pub log_prob: f64,
    /// Per-frame winning trellis state, when requested.
    pub frame_path: Option<Vec<i32>>,
    /// Per-frame log-prob of the winning label, including blanks.
    /// Empty when there are no tokens. Not part of the JSON.
    pub frame_scores: Vec<f64>,
    /// The blank runs the boundary padding consumed, as
    /// `(before_token_index, first_frame, last_frame)`. `before_token_index == l`
    /// is the trailing run. Exposed so the boundary rule can be diffed against
    /// the reference frame by frame -- a one- or two-frame output difference is
    /// otherwise indistinguishable between "the midpoint was taken over a
    /// different range" and "the range itself differs". Not part of the JSON.
    pub blank_runs: Vec<(usize, i64, i64)>,
}

impl AlignmentResult {
    pub fn mean_frame_score(&self) -> f64 {
        self.log_prob / self.frames.max(1) as f64
    }
    pub fn text(&self) -> String {
        self.tokens.iter().map(|t| t.piece.as_str()).collect()
    }
}

/// One trellis row. `skip_dead[st] == u64::MAX` forbids the skip arc.
/// Tie-break matches numpy `argmax` over `[stay, advance, skip]`: the first
/// maximum wins, so stay beats advance beats skip.
///
/// `back_row` holds that row's backpointers **2 bits per state** (stay 0,
/// advance 1, skip 2), four states to a byte — the packing the reference C++
/// aligner uses.  It is the one structure that grows with the audio *and* with
/// the transcript (T×(2·tokens+1)), so the 4x is what keeps an hour of audio
/// inside RAM: 15 m goes 1.44 GB → 360 MB.
#[inline]
fn put_back(row: &mut [u8], st: usize, choice: u8) {
    let i = st >> 2;
    let sh = (st & 3) * 2;
    row[i] = (row[i] & !(0b11u8 << sh)) | (choice << sh);
}

/// Store a whole packed byte — four states' choices at once.
///
/// Nothing here reads the buffer first: the DP fills a row before the
/// traceback reads it, and every byte the traceback can reach is written
/// wholesale (the vector loop by four states, the 0..3 and tail groups
/// below), so the allocation can stay uninitialised instead of costing a
/// 360 MB memset on the 15 m fixture.
#[inline]
fn put_back_byte(row: &mut [u8], byte_index: usize, byte: u8) {
    row[byte_index] = byte;
}

/// The backpointer choice recorded for `(t, st)`.
#[inline]
fn get_back(row: &[u8], st: usize) -> usize {
    ((row[st >> 2] >> ((st & 3) * 2)) & 0b11u8) as usize
}

/// Bytes one row of `states` states occupies in the packed store.
#[inline]
fn row_bytes(states: usize) -> usize {
    states.div_ceil(4)
}

/// Move each of the low four bits to bit `2·i`, so a movemask's four lanes
/// land in four 2-bit fields of one byte.
#[inline]
fn spread2(x: u8) -> u8 {
    let x = x & 0x0f;
    let x = (x | (x << 2)) & 0x33;
    (x | (x << 1)) & 0x55
}

/// One DP row, split across threads by state range.
///
/// A row has no cross-state dependency — `next[st]` reads only `prev[st-2..=st]`
/// and `emit[st]` — so any contiguous split gives bit-identical results to the
/// serial version.  The row is memory-bound (a 1 h file pushes ~900 GB through
/// it) and one core only reaches ~20 GB/s of that, which is where the
/// parallelism pays: 4-state groups stay together, so every chunk owns whole
/// backpointer bytes.
///
/// `back` is `None` on the alpha-only pass (linear space keeps checkpoints, not
/// choices), which skips the packing work entirely.
/// A row shorter than this stays serial: the parallel dispatch costs a few tens
/// of microseconds whatever the row size, so splitting only pays once a row
/// moves a few hundred KB.  Measured on this box: 3 m (S=6531) loses 26% when
/// split, 15 m (S=31129) gains 15%, an hour (S=124519) gains 6%.
const PAR_ROW_MIN_STATES: usize = 16_384;

fn dp_row_par(
    prev: &[f64],
    emit: &[f64],
    skip_dead: &[u64],
    next: &mut [f64],
    back: Option<&mut [u8]>,
    use_avx2: bool,
) {
    let s = prev.len();
    dp_row_range(prev, emit, skip_dead, next, back, 0, s - 1, use_avx2)
}

/// States `[lo, hi]` of one row — the *band* the DP computes for frame `t`.
/// The full-width driver is `dp_row_range(..., 0, s - 1, ...)`; the banded DP
/// narrows the range to the states frame `t` can reach and that can still
/// reach the end, which cuts the row work by ~1/3 on hour-long files while
/// leaving every in-band value — and therefore the alignment — bit-identical.
///
/// `next` and `back` are the full row slices; ranges are rounded to the
/// global 4-state byte grid, so every chunk owns whole backpointer bytes and
/// the traceback indexes them exactly as the full-width run packed them.
fn dp_row_range(
    prev: &[f64],
    emit: &[f64],
    skip_dead: &[u64],
    next: &mut [f64],
    mut back: Option<&mut [u8]>,
    lo: usize,
    hi: usize,
    use_avx2: bool,
) {
    let end = (hi + 1).min(next.len());
    // The vector kernel reads prev[st-2] and writes whole 4-state bytes, so
    // its ranges start on a 4-boundary.  The band may open mid-byte (the
    // lower bound moves in steps of two from an odd state), so the first
    // computed state rounds *down* to one: the fields below `lo` then hold
    // garbage choices computed from stale alphas, which is safe because the
    // traceback only ever visits in-band states.
    let first = (lo >> 2) << 2;
    if first >= 4 {
        // the band opens past the row head: no scalar prologue needed
        return dp_row_body(prev, emit, skip_dead, next, back, first, end, use_avx2);
    }
    // the band includes states 0..4: they run first and serially — the
    // vector kernel reads st-2, and 4 is also the first offset that owns a
    // whole backpointer byte
    let head = end.min(4);
    if head > 0 {
        let mut byte = 0u8;
        for k in 0..head {
            byte |= dp_one(k, prev, emit, skip_dead, &mut next[k..]) << (k * 2);
        }
        if let Some(back) = back.as_deref_mut() {
            put_back_byte(back, 0, byte);
        }
    }
    if head >= end {
        return;
    }
    dp_row_body(prev, emit, skip_dead, next, back, 4, end, use_avx2);
}

/// States `[st0, end)` of one row, split across threads when wide enough.
fn dp_row_body(
    prev: &[f64],
    emit: &[f64],
    skip_dead: &[u64],
    next: &mut [f64],
    back: Option<&mut [u8]>,
    st0: usize,
    end: usize,
    use_avx2: bool,
) {
    if end - st0 < PAR_ROW_MIN_STATES {
        // too small to be worth splitting: one serial range
        let bytes = back.map(|b| &mut b[st0 >> 2..]);
        return dp_range(prev, emit, skip_dead, &mut next[st0..end], bytes, st0, use_avx2);
    }
    let per = {
        // whole 4-groups, and enough of them to fill the pool
        let threads = rayon::current_num_threads().max(1);
        (end - st0).div_ceil(4).div_ceil(threads).max(1) * 4
    };
    let bytes = back.map(|b| &mut b[st0 >> 2..]);
    match bytes {
        // both halves are indexed by the chunk's own position; states are
        // 4-aligned, so the backpointer bytes are too
        Some(bytes) => next[st0..end]
            .par_chunks_mut(per)
            .zip(bytes.par_chunks_mut(per / 4))
            .enumerate()
            .for_each(|(ci, (chunk, row))| {
                dp_range(prev, emit, skip_dead, chunk, Some(row), st0 + ci * per, use_avx2)
            }),
        None => next[st0..end].par_chunks_mut(per).enumerate().for_each(
            |(ci, chunk)| {
                dp_range(prev, emit, skip_dead, chunk, None, st0 + ci * per, use_avx2)
            },
        ),
    }
}

/// States `[st0, st0 + next.len())` of one row, serial within.
fn dp_range(
    prev: &[f64],
    emit: &[f64],
    skip_dead: &[u64],
    next: &mut [f64],
    back: Option<&mut [u8]>,
    st0: usize,
    use_avx2: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        // SAFETY: use_avx2 is the runtime AVX2 check.
        unsafe { dp_range_avx2(prev, emit, skip_dead, next, back, st0) };
        return;
    }
    dp_range_scalar(prev, emit, skip_dead, next, back, st0);
}

fn dp_range_scalar(
    prev: &[f64],
    emit: &[f64],
    skip_dead: &[u64],
    next: &mut [f64],
    mut back: Option<&mut [u8]>,
    st0: usize,
) {
    let end = st0 + next.len();
    let mut st = st0;
    while st < end {
        let mut byte = 0u8;
        for k in 0..4 {
            if st + k >= end {
                break;
            }
            byte |= dp_one(st + k, prev, emit, skip_dead, &mut next[st + k - st0..]) << (k * 2);
        }
        if let Some(row) = back.as_deref_mut() {
            put_back_byte(row, (st - st0) >> 2, byte);
        }
        st += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dp_range_avx2(
    prev: &[f64],
    emit: &[f64],
    skip_dead: &[u64],
    next: &mut [f64],
    mut back: Option<&mut [u8]>,
    st0: usize,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_and_pd, _mm256_blendv_pd, _mm256_castsi256_pd, _mm256_cmp_pd,
        _mm256_loadu_pd, _mm256_loadu_si256, _mm256_movemask_pd, _mm256_set1_pd, _mm256_storeu_pd,
        _CMP_GE_OQ,
    };
    let end = st0 + next.len();
    debug_assert!(st0 >= 2, "the vector loop reads prev[st-2]; the driver starts at 4");
    debug_assert!(st0 % 4 == 0, "chunks are 4-aligned so they own whole back bytes");
    let neginf = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut st = st0;
    while st + 4 <= end {
        let stay = _mm256_loadu_pd(prev.as_ptr().add(st));
        let adv = _mm256_loadu_pd(prev.as_ptr().add(st - 1));
        let skip_raw = _mm256_loadu_pd(prev.as_ptr().add(st - 2));
        let kill = _mm256_loadu_si256(skip_dead.as_ptr().add(st) as *const _);
        let skip = _mm256_blendv_pd(skip_raw, neginf, _mm256_castsi256_pd(kill));
        let stay_ge_adv = _mm256_cmp_pd(stay, adv, _CMP_GE_OQ);
        let stay_ge_skip = _mm256_cmp_pd(stay, skip, _CMP_GE_OQ);
        let stay_wins = _mm256_and_pd(stay_ge_adv, stay_ge_skip);
        let adv_ge_skip = _mm256_cmp_pd(adv, skip, _CMP_GE_OQ);
        let adv_or_skip = _mm256_blendv_pd(skip, adv, adv_ge_skip);
        let best = _mm256_blendv_pd(adv_or_skip, stay, stay_wins);
        let out = _mm256_add_pd(best, _mm256_loadu_pd(emit.as_ptr().add(st)));
        _mm256_storeu_pd(next.as_mut_ptr().add(st - st0), out);
        if let Some(row) = back.as_deref_mut() {
            // 2 bits per state, four states to a byte: stay = 0b00, advance =
            // 0b01, skip = 0b10.  A field's high bit is the skip bit and its low
            // one the advance bit; `adv_ge_skip` is also true when stay wins
            // (the scalar tie-break checks stay first), so it only counts when
            // stay does not.
            let stay_mask = _mm256_movemask_pd(stay_wins) as u8 & 0x0f;
            let adv_mask = _mm256_movemask_pd(adv_ge_skip) as u8 & 0x0f;
            let skip_mask = !(stay_mask | adv_mask) & 0x0f;
            let adv_only = adv_mask & !stay_mask & 0x0f;
            // spread each lane's two bits into its own 2-bit field (lane L ->
            // bits 2L, 2L+1) before combining
            row[(st - st0) >> 2] = (spread2(skip_mask) << 1) | spread2(adv_only);
        }
        st += 4;
    }
    if st < end {
        // the chunk's tail: one byte, whose states past `end` are never read,
        // so it can be written whole without reading first
        let mut byte = 0u8;
        while st < end {
            byte |= dp_one(st, prev, emit, skip_dead, &mut next[st - st0..]) << ((st & 3) * 2);
            st += 1;
        }
        if let Some(row) = back.as_deref_mut() {
            put_back_byte(row, (st - st0) >> 2, byte);
        }
    }
}

/// One state of the row: returns its packed choice and writes the new score
/// to `next[0]` — callers pass the sub-slice for state `st`, which is what lets
/// a parallel chunk own a contiguous piece of the row.
fn dp_one(st: usize, prev: &[f64], emit: &[f64], skip_dead: &[u64], next: &mut [f64]) -> u8 {
    let stay = prev[st];
    let adv = if st >= 1 { prev[st - 1] } else { f64::NEG_INFINITY };
    let skip = if st >= 2 && skip_dead[st] == 0 {
        prev[st - 2]
    } else {
        f64::NEG_INFINITY
    };
    let (choice, best) = if stay >= adv && stay >= skip {
        (0u8, stay)
    } else if adv >= skip {
        (1, adv)
    } else {
        (2, skip)
    };
    next[0] = best + emit[st];
    choice
}

/// The CTC blank's id in the omniASR checkpoint.
///
/// It is also the id of `<s>`, which is the inter-word SPACE, so a token
/// carrying this id is a blank for every purpose the reference's own
/// `prev_seg.label == blank` test cares about. See the blank-run collection in
/// [`collapse`].
pub const BLANK_TOKEN_ID: usize = 0;

/// `l' = [blank, t1, blank, ..., tL, blank]`
pub fn build_expanded_labels(token_ids: &[usize], blank_id: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(2 * token_ids.len() + 1);
    for &t in token_ids {
        out.push(blank_id);
        out.push(t);
    }
    out.push(blank_id);
    out
}

/// Per-frame emission scores for the trellis. Two sources: the full
/// (T, V) log-prob matrix gathered through the expanded labels, and a
/// pre-gathered (T, S) matrix (the GPU gather kernel's output). Both
/// yield bit-identical f32 values, so the DP — and the timestamps — are
/// identical either way.
///
/// Public so the aligner can plug in its own source: for a long transcript it
/// keeps the lm-head logits per window instead of the trellis and gathers the
/// columns on demand ([`ctc_forced_align_emissions`]).
pub trait Emissions {
    /// Fill `emit[0..num_states]` with frame `t`'s state scores.
    fn fill_emit(&self, t: usize, emit: &mut [f64], token_ids: &[usize]);
    /// Fill `emit[lo..=hi]` — the band the banded DP reads.  The default
    /// fills the whole row (of which the band is a subset); sources that pay
    /// per-column work override it to touch the band only.
    fn fill_emit_band(&self, t: usize, emit: &mut [f64], lo: usize, hi: usize, token_ids: &[usize]) {
        let _ = (lo, hi);
        self.fill_emit(t, emit, token_ids)
    }
    /// Score of expanded state `st` at frame `t` (frame_scores, collapse).
    fn score(&self, t: usize, st: usize) -> f32;
    /// The path's per-frame scores, `out[t] = score(t, states[t])`.
    ///
    /// The default walks frame by frame.  A source that materialises a window
    /// on demand overrides this to visit each window once, since the traceback
    /// has to be finished before `states` exists and so the path pass is a
    /// *second* walk over the same windows.
    fn score_path(&self, states: &[i32], out: &mut [f64]) {
        for (t, &st) in states.iter().enumerate() {
            out[t] = self.score(t, st as usize) as f64;
        }
    }
}

impl<T: Emissions + ?Sized> Emissions for &T {
    fn fill_emit(&self, t: usize, emit: &mut [f64], token_ids: &[usize]) {
        (**self).fill_emit(t, emit, token_ids)
    }
    fn score(&self, t: usize, st: usize) -> f32 {
        (**self).score(t, st)
    }
}

struct FullRows<'a> {
    log_probs: &'a [f32],
    vocab: usize,
    labels: &'a [usize],
    blank_id: usize,
}

impl Emissions for FullRows<'_> {
    fn fill_emit(&self, t: usize, emit: &mut [f64], token_ids: &[usize]) {
        let row = &self.log_probs[t * self.vocab..(t + 1) * self.vocab];
        emit.fill(row[self.blank_id] as f64);
        for (i, &tok) in token_ids.iter().enumerate() {
            emit[2 * i + 1] = row[tok] as f64;
        }
    }
    /// Banded: blank over the band, then the token states that fall inside
    /// it — the same values the full row holds on `[lo, hi]`.
    fn fill_emit_band(&self, t: usize, emit: &mut [f64], lo: usize, hi: usize, token_ids: &[usize]) {
        let row = &self.log_probs[t * self.vocab..(t + 1) * self.vocab];
        emit[lo..=hi].fill(row[self.blank_id] as f64);
        for (i, &tok) in token_ids.iter().enumerate() {
            let st = 2 * i + 1;
            if st > hi {
                break;
            }
            if st >= lo {
                emit[st] = row[tok] as f64;
            }
        }
    }
    fn score(&self, t: usize, st: usize) -> f32 {
        self.log_probs[t * self.vocab + self.labels[st]]
    }
}

struct GatheredRows<'a> {
    gathered: &'a [f32],
    num_states: usize,
}

impl Emissions for GatheredRows<'_> {
    fn fill_emit(&self, t: usize, emit: &mut [f64], _token_ids: &[usize]) {
        let row = &self.gathered[t * self.num_states..(t + 1) * self.num_states];
        for (e, &v) in emit.iter_mut().zip(row) {
            *e = v as f64;
        }
    }
    fn fill_emit_band(&self, t: usize, emit: &mut [f64], lo: usize, hi: usize, _token_ids: &[usize]) {
        let s = self.num_states;
        let row = &self.gathered[t * s..(t + 1) * s];
        for (e, &v) in emit[lo..=hi].iter_mut().zip(row[lo..=hi].iter()) {
            *e = v as f64;
        }
    }
    fn score(&self, t: usize, st: usize) -> f32 {
        self.gathered[t * self.num_states + st]
    }
}

/// The gathered trellis as one (frames × S) block per window: the aligner
/// collects each chunk's gathered matrix as it is produced and runs the DP
/// straight over them, so the frames never sit in a second contiguous copy.
#[derive(Debug, Clone, Default)]
pub struct GatheredChunks {
    /// per window: (kept_frames × S) row-major
    pub chunks: Vec<Vec<f32>>,
    /// kept frames per window (the last window matches — the tail padding is
    /// trimmed from its block before the DP runs)
    pub frames_per_chunk: usize,
    pub num_states: usize,
}

impl GatheredChunks {
    pub fn total_frames(&self) -> usize {
        self.chunks.iter().map(|c| c.len() / self.num_states.max(1)).sum()
    }

    /// `fill_emit`/`score` locate a frame as `t / frames_per_chunk`, so every
    /// block but the last must hold exactly `frames_per_chunk` whole rows.  The
    /// aligner guarantees that by construction (every window is encoded from
    /// an identically sized chunk, and only the last one is trimmed); a short
    /// block anywhere else would silently shift every later frame, so refuse
    /// it rather than return a plausible wrong alignment.
    pub fn validate(&self) -> anyhow::Result<()> {
        let s = self.num_states;
        anyhow::ensure!(s > 0, "trellis has no states");
        anyhow::ensure!(
            self.frames_per_chunk > 0,
            "frames_per_chunk must be positive"
        );
        let last = self.chunks.len().saturating_sub(1);
        for (i, c) in self.chunks.iter().enumerate() {
            anyhow::ensure!(
                c.len() % s == 0,
                "block {i} holds {} values, not a whole number of {s}-state rows",
                c.len()
            );
            if i != last {
                anyhow::ensure!(
                    c.len() == self.frames_per_chunk * s,
                    "block {i} holds {} rows, expected {} (only the last block may be short)",
                    c.len() / s,
                    self.frames_per_chunk
                );
            }
        }
        Ok(())
    }
}

impl Emissions for GatheredChunks {
    fn fill_emit(&self, t: usize, emit: &mut [f64], _token_ids: &[usize]) {
        let s = self.num_states;
        let row = &self.chunks[t / self.frames_per_chunk][t % self.frames_per_chunk * s..(t % self.frames_per_chunk + 1) * s];
        for (e, &v) in emit.iter_mut().zip(row) {
            *e = v as f64;
        }
    }
    fn fill_emit_band(&self, t: usize, emit: &mut [f64], lo: usize, hi: usize, _token_ids: &[usize]) {
        let s = self.num_states;
        let row = &self.chunks[t / self.frames_per_chunk][t % self.frames_per_chunk * s..(t % self.frames_per_chunk + 1) * s];
        for (e, &v) in emit[lo..=hi].iter_mut().zip(row[lo..=hi].iter()) {
            *e = v as f64;
        }
    }
    fn score(&self, t: usize, st: usize) -> f32 {
        let s = self.num_states;
        self.chunks[t / self.frames_per_chunk][t % self.frames_per_chunk * s + st]
    }
}

/// Force-align `token_ids` against `log_probs` ((T, V) row-major, f32).
#[allow(clippy::too_many_arguments)]
pub fn ctc_forced_align(
    log_probs: &[f32],
    num_frames: usize,
    vocab: usize,
    token_ids: &[usize],
    blank_id: usize,
    frame_rate: f64,
    pieces: Option<&[String]>,
    return_path: bool,
) -> anyhow::Result<AlignmentResult> {
    let labels = build_expanded_labels(token_ids, blank_id);
    let em = FullRows { log_probs, vocab, labels: &labels, blank_id };
    align(&em, num_frames, &labels, blank_id, token_ids, frame_rate, pieces, return_path)
}

/// Force-align against a pre-gathered (T, S) score matrix with
/// S = 2·L+1 states in the expanded-label order: even states are the
/// blank, odd state 2i+1 emits `token_ids[i]`. `gathered[t * S + st]`
/// must equal the full matrix's `log_probs[t * V + labels[st]]`.
pub fn ctc_forced_align_gathered(
    gathered: &[f32],
    num_frames: usize,
    num_states: usize,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
) -> anyhow::Result<AlignmentResult> {
    anyhow::ensure!(
        num_states == 2 * token_ids.len() + 1,
        "gathered states {num_states} != 2·{}+1",
        token_ids.len()
    );
    anyhow::ensure!(
        gathered.len() == num_frames * num_states,
        "gathered matrix is {} values, expected {num_frames}×{num_states}",
        gathered.len()
    );
    // expanded-label structure: even states are blanks, odd states tokens;
    // a skip arc is illegal when it would repeat the same emission.  The
    // blank shows up as the usize::MAX sentinel, so blank_id = usize::MAX
    // makes the shared skip_dead rule fire on even states.
    let labels: Vec<usize> = (0..num_states)
        .map(|st| if st % 2 == 0 { usize::MAX } else { token_ids[(st - 1) / 2] })
        .collect();
    let em = GatheredRows { gathered, num_states };
    align(&em, num_frames, &labels, usize::MAX, token_ids, frame_rate, pieces, false)
}

/// Same DP over [`GatheredChunks`]: the per-window blocks the aligner
/// collects on the fly.  Values must match the contiguous gather exactly
/// (they are the same f32s, so the alignment is bit-identical).
pub fn ctc_forced_align_gathered_chunks(
    chunks: &GatheredChunks,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
) -> anyhow::Result<AlignmentResult> {
    let num_states = chunks.num_states;
    anyhow::ensure!(
        num_states == 2 * token_ids.len() + 1,
        "gathered states {num_states} != 2·{}+1",
        token_ids.len()
    );
    chunks.validate()?;
    let num_frames = chunks.total_frames();
    let labels: Vec<usize> = (0..num_states)
        .map(|st| if st % 2 == 0 { usize::MAX } else { token_ids[(st - 1) / 2] })
        .collect();
    align(&chunks, num_frames, &labels, usize::MAX, token_ids, frame_rate, pieces, false)
}

/// [`ctc_forced_align_gathered_chunks`], but keeps the per-frame trellis state.
///
/// The path is what the blank runs are cut from, and a blank run is what the
/// word boundary is padded into. Comparing outputs cannot tell "the midpoint
/// was taken over a different range" from "the range differs", so the range
/// itself has to be inspectable. Diagnostic only — the aligner does not use it.
pub fn ctc_forced_align_gathered_chunks_with_path(
    chunks: &GatheredChunks,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
) -> anyhow::Result<AlignmentResult> {
    let num_states = chunks.num_states;
    anyhow::ensure!(
        num_states == 2 * token_ids.len() + 1,
        "gathered states {num_states} != 2·{}+1",
        token_ids.len()
    );
    chunks.validate()?;
    let num_frames = chunks.total_frames();
    let labels: Vec<usize> = (0..num_states)
        .map(|st| if st % 2 == 0 { usize::MAX } else { token_ids[(st - 1) / 2] })
        .collect();
    align(&chunks, num_frames, &labels, usize::MAX, token_ids, frame_rate, pieces, true)
}

/// The same DP over any [`Emissions`] source.  The aligner uses it for the
/// lazy-logits trellis, which produces the identical f32 values the gathered
/// blocks hold — it just materialises a window's columns only when the DP
/// reaches it.
pub fn ctc_forced_align_emissions(
    em: &impl Emissions,
    num_frames: usize,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
) -> anyhow::Result<AlignmentResult> {
    let num_states = 2 * token_ids.len() + 1;
    let labels: Vec<usize> = (0..num_states)
        .map(|st| if st % 2 == 0 { usize::MAX } else { token_ids[(st - 1) / 2] })
        .collect();
    align(em, num_frames, &labels, usize::MAX, token_ids, frame_rate, pieces, false)
}

/// The Viterbi recursion over one emissions source: two alpha rows, the
/// emissions row, and the skip mask.  Split out of `align` so the same step
/// serves the single pass and the per-segment recompute of the linear-space
/// traceback — the two produce the same choices, since a recompute sees the
/// same alpha it started from.
///
/// With `band` on, each frame computes only the states it can reach and that
/// can still reach the end ([`Dp::band`]) — every in-band alpha, and hence
/// every backpointer choice and the final path, is bit-identical to the
/// full-width row, because a banded -inf sits exactly where the full row's
/// alpha is unreachable.
struct Dp<'a, E: Emissions + ?Sized> {
    em: &'a E,
    token_ids: &'a [usize],
    s: usize,
    rb: usize,
    t_len: usize,
    /// compute only each frame's reachable band (CTC_NO_BAND=1 turns it off
    /// for A/B); the buffers stay full-length either way
    band: bool,
    /// all-ones lane => the skip arc is illegal (forced to -inf)
    skip_dead: Vec<u64>,
    emit: Vec<f64>,
    prev: Vec<f64>,
    next: Vec<f64>,


    #[cfg(target_arch = "x86_64")]
    avx2: bool,
}

impl<'a, E: Emissions + ?Sized> Dp<'a, E> {
    /// `prev` starts as the alpha at frame 0: state 0 is the first blank and
    /// state 1 the first token of the expanded label sequence.
    #[allow(clippy::too_many_arguments)]
    fn new(
        em: &'a E,
        s: usize,
        rb: usize,
        labels: &[usize],
        blank_id: usize,
        token_ids: &'a [usize],
        t_len: usize,
        band: bool,
    ) -> Self {
        let mut skip_dead = vec![0u64; s];
        for st in 2..s {
            if labels[st] == blank_id || labels[st] == labels[st - 2] {
                skip_dead[st] = u64::MAX;
            }
        }
        let mut prev = vec![f64::NEG_INFINITY; s];
        prev[0] = em.score(0, 0) as f64;
        if s > 1 {
            prev[1] = em.score(0, 1) as f64;
        }
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2");
        Dp {
            em,
            token_ids,
            s,
            rb,
            t_len,
            band,
            skip_dead,
            emit: vec![0.0f64; s],
            prev,
            next: vec![0.0f64; s],

            #[cfg(target_arch = "x86_64")]
            avx2,
        }
    }

    /// The states frame `t` can reach and that can still reach the end.
    ///
    /// A path starts in {0, 1} at frame 0 and gains at most 2 states per
    /// frame (the skip arc), so nothing above `2t+1` is reachable.  It must
    /// end at the last blank (S-1) or the last token (S-2) — whichever the
    /// final argmax picks is unknown until the DP is done, so the lower bound
    /// follows from the *lower* of the two: a state that cannot reach S-2
    /// with the frames left cannot be on the path either.  (A bound derived
    /// from S-1 alone would clip the last token off paths that skip every
    /// remaining frame.)  Both bounds move right monotonically, which is why
    /// `run` can keep the buffers full-length, compute less of each row, and
    /// keep everything outside the band at -inf.
    fn band(&self, t: usize) -> (usize, usize) {
        let hi = (2 * t + 1).min(self.s - 1);
        let lo = (self.s - 2).saturating_sub(2 * (self.t_len - 1 - t));
        (lo, hi)
    }

    /// Frames `[t0, t1)`, writing row `t`'s backpointers at
    /// `back[(t - t0) * rb..]`.  `None` runs the pass for its alpha only: the
    /// linear-space forward keeps checkpoints, not choices, and the row the
    /// kernel would write is then dropped.
    fn run(&mut self, t0: usize, t1: usize, mut back: Option<&mut [u8]>) {
        let rb = self.rb;
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = self.avx2;
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;
        for t in t0..t1 {
            let (lo, hi) = if self.band { self.band(t) } else { (0, self.s - 1) };
            if self.band {
                self.em
                    .fill_emit_band(t, &mut self.emit, lo, hi, self.token_ids);
            } else {
                self.em.fill_emit(t, &mut self.emit, self.token_ids);
            }
            let row = match back.as_deref_mut() {
                Some(buf) => {
                    let at = (t - t0) * rb;
                    Some(&mut buf[at..at + rb])
                }
                None => None,
            };
            if self.band {
                dp_row_range(
                    &self.prev,
                    &self.emit,
                    &self.skip_dead,
                    &mut self.next,
                    row,
                    lo,
                    hi,
                    use_avx2,
                );
                // frame t+1's kernel reads up to hi+2 (the states the band
                // grows into); what it sees there must be -inf, not the alpha
                // this buffer held two frames ago.  Below the band nothing
                // needs clearing: the lowest read at t+1 is exactly lo.
                let hi_next = (hi + 2).min(self.s - 1);
                for st in hi + 1..=hi_next {
                    self.next[st] = f64::NEG_INFINITY;
                }
            } else {
                dp_row_par(
                    &self.prev,
                    &self.emit,
                    &self.skip_dead,
                    &mut self.next,
                    row,
                    use_avx2,
                );
            }
            std::mem::swap(&mut self.prev, &mut self.next);
        }
    }
}

/// Frames per linear-space segment: the largest length whose checkpoints plus
/// segment backpointers still fit the budget, or the whole file when the
/// backpointers fit on their own (no recompute at all) or nothing does.
///
/// The trade is `(t_len/seg)·S·8 + seg·S/4` bytes — U-shaped, so halving from
/// the whole file walks down the long side to the largest segment that fits.
/// Recomputing a segment replays the DP over every frame once whatever the
/// segment length, so a large one only saves the per-segment overhead.
fn segment_len(t_len: usize, s: usize, rb: usize) -> usize {
    let budget = match std::env::var("CTC_VITERBI_BUDGET_MB").ok().as_deref().map(str::parse::<usize>)
    {
        Some(Ok(mb)) => mb << 20,
        _ => 512 << 20,
    };
    let whole = t_len.max(1);
    let need = |seg: usize| (t_len.div_ceil(seg) * s * 8).saturating_add(seg * rb);
    let mut best = None;
    let mut seg = whole;
    loop {
        // keep the *largest* segment that fits: halving from the whole file
        // walks down the long side of the U, so the first hit is the best one
        if best.is_none() && need(seg) <= budget {
            best = Some(seg);
        }
        if seg <= 1 {
            break;
        }
        let next = seg.div_ceil(2);
        if next == seg {
            break;
        }
        seg = next;
    }
    // nothing fits: keep the plain single pass, whose backpointers are the
    // smallest of the options (a checkpoint per frame would be 8x worse)
    best.unwrap_or(whole)
}

fn align(
    em: &impl Emissions,
    t_len: usize,
    labels: &[usize],
    blank_id: usize,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
    return_path: bool,
) -> anyhow::Result<AlignmentResult> {
    align_with(em, t_len, labels, blank_id, token_ids, frame_rate, pieces, return_path, None, None)
}

/// [ctc_forced_align_gathered_chunks], additionally told which source word
/// each target came from.
pub fn ctc_forced_align_gathered_with_word_ids(
    chunks: &GatheredChunks,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
    word_ids: &[usize],
    keep_path: bool,
) -> anyhow::Result<AlignmentResult> {
    let num_states = chunks.num_states;
    anyhow::ensure!(
        num_states == 2 * token_ids.len() + 1,
        "gathered states {num_states} != 2*token_ids+1"
    );
    chunks.validate()?;
    let num_frames = chunks.total_frames();
    let labels: Vec<usize> = (0..num_states)
        .map(|st| if st % 2 == 0 { usize::MAX } else { token_ids[(st - 1) / 2] })
        .collect();
    with_word_ids(word_ids, || {
        align(&chunks, num_frames, &labels, usize::MAX, token_ids, frame_rate,
              pieces, keep_path)
    })
}

/// [ctc_forced_align_emissions], additionally told which source word each
/// target came from.
pub fn ctc_forced_align_emissions_with_word_ids(
    em: &impl Emissions,
    t_len: usize,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
    word_ids: &[usize],
) -> anyhow::Result<AlignmentResult> {
    let expanded = build_expanded_labels(token_ids, 0);
    let num_states = expanded.len();
    with_word_ids(word_ids, || {
        align(em, t_len, &expanded, 0, token_ids, frame_rate, pieces, false)
    })
}

/// [lign], additionally told which source word each target came from.
///
/// The target sequence lays every word's letters end to end with nothing
/// between the words, so the DP cannot recover where one word stopped and the
/// next began. uild_words needs that information and it has to arrive from
/// the side: a space token between words would be a target the reference never
/// had, and it lengthens the sequence past what the frame count can carry.
pub fn align_with_word_ids(
    em: &impl Emissions,
    t_len: usize,
    labels: &[usize],
    blank_id: usize,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
    word_ids: &[usize],
) -> anyhow::Result<AlignmentResult> {
    with_word_ids(word_ids, || {
        align_with(em, t_len, labels, blank_id, token_ids, frame_rate,
                   pieces, false, None, None)
    })
}

/// Install word_ids for the duration of , then restore the previous value
/// so a nested or repeated call cannot leak one run's words into another's.
fn with_word_ids<T>(word_ids: &[usize], f: impl FnOnce() -> T) -> T {
    WORD_IDS.with(|c| {
        let prev = c.replace(Some(word_ids.to_vec()));
        let out = f();
        c.replace(prev);
        out
    })
}

thread_local! {
    /// Set by [lign_with_word_ids] for the duration of one DP. A side
    /// channel rather than a parameter on every helper, because the word ids
    /// are only read once, in collapse, and threading them through the
    /// chunked / lazy / banded call tree would touch a dozen signatures for a
    /// value one function uses.
    static WORD_IDS: std::cell::RefCell<Option<Vec<usize>>> = const { std::cell::RefCell::new(None) };
}

/// [`align`], with the linear-space segment length forced (`None` = the
/// budget's choice, so `Some(t_len)` is the single pass) and banding forced
/// (`None` = on, unless `CTC_NO_BAND=1` — the same-binary A/B switch).  The
/// segment length and the band must not change the answer, only the work and
/// memory it takes to get there.
#[allow(clippy::too_many_arguments)]
fn align_with(
    em: &impl Emissions,
    t_len: usize,
    labels: &[usize],
    blank_id: usize,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
    return_path: bool,
    force_seg: Option<usize>,
    force_band: Option<bool>,
) -> anyhow::Result<AlignmentResult> {
    let l = token_ids.len();
    let s = labels.len();
    if l == 0 {
        return Ok(AlignmentResult {
            tokens: vec![],
            frames: t_len,
            frame_rate,
            log_prob: 0.0,
            frame_path: None,
            frame_scores: vec![],
            blank_runs: vec![],
        });
    }
    if t_len < l {
        anyhow::bail!(
            "Audio too short: {t_len} frames cannot hold {l} tokens."
        );
    }

    // 0 = stay, 1 = from s-1, 2 = from s-2, two bits each (see `put_back`).
    // Row 0 is never read (traceback starts at t >= 1); every later row is
    // written in full before that.
    let rb = row_bytes(s);
    let band = force_band.unwrap_or_else(|| {
        std::env::var("CTC_NO_BAND").ok().as_deref() != Some("1")
    });
    let mut dp = Dp::new(em, s, rb, labels, blank_id, token_ids, t_len, band);
    let seg = force_seg.unwrap_or_else(|| segment_len(t_len, s, rb)).max(1);

    // Linear space: keep the alpha every `seg` frames and no backpointers at
    // all, then walk the file backwards one segment at a time, recomputing
    // just that segment's backpointers from its checkpoint.  Memory is
    // (frames/seg)·S·8 + seg·S/4 instead of frames·S/2, so it stops growing
    // with the audio — at the cost of a second DP pass.  When the whole
    // file's backpointers fit the budget, `seg` is the file and this is the
    // single-pass path unchanged.
    let linear = seg < t_len;
    let mut checkpoints: Vec<f64> = Vec::new();
    let mut back: Vec<u8> = Vec::new();
    if linear {
        checkpoints.reserve((t_len / seg + 2) * s);
        checkpoints.extend_from_slice(&dp.prev); // alpha at frame 0
    } else {
        // one row per frame 1..t_len (row 0 has no predecessor), each rb bytes
        let nback = (t_len - 1).checked_mul(rb).context("backpointer size")?;
        back.try_reserve_exact(nback).context("backpointer alloc")?;
        // SAFETY: u8 has no destructor and no invalid bit patterns, and every
        // row the traceback reads is written whole by the run below —
        // `put_back_byte` stores bytes without reading them first.
        unsafe { back.set_len(nback) };
    }
    // Frames 1..t_len in one call: the alpha-only run (linear space) throws
    // the choices away and keeps only the checkpoints, the single pass keeps
    // them all.
    if linear {
        for t in 1..t_len {
            dp.run(t, t + 1, None);
            // after the step, `prev` is the alpha *at* frame t, which is what
            // the segment ending there resumes from
            if t % seg == 0 {
                checkpoints.extend_from_slice(&dp.prev);
            }
        }
    } else {
        dp.run(1, t_len, Some(&mut back[..]));
    }

    let score = dp.prev.clone();

    // termination: last blank or last token, whichever scores higher
    let mut s_end = s - 1;
    if s >= 2 && score[s - 2] > score[s - 1] {
        s_end = s - 2;
    }
    let total = score[s_end];

    let mut states = vec![0i32; t_len];
    states[t_len - 1] = s_end as i32;
    let mut cur = s_end;
    if !linear {
        for t in (1..t_len).rev() {
            // a feasible path's states stay inside each frame's band, so the
            // packed field it reads was always written; `saturating` only
            // matters for a transcript the audio cannot hold at all (the DP
            // is all -inf and the walk degenerates) — keep it from panicking
            cur = cur.saturating_sub(get_back(&back[(t - 1) * rb..t * rb], cur));
            states[t - 1] = cur as i32;
        }
    } else {
        // Segment k replays frames k·seg+1 ..= (k+1)·seg from the alpha at
        // frame k·seg, producing rows k·seg+1 ..= (k+1)·seg — row r comes
        // from frame r, so a segment's starting alpha is the one *before* its
        // first row.  The last segment stops at the last frame.
        let mut seg_back = vec![0u8; seg * rb];
        let mut b = ((t_len - 1) / seg) * seg; // alpha frame this segment starts from
        loop {
            let k = b / seg;
            let lo = b + 1; // first row of this segment
            let hi = (b + seg).min(t_len - 1); // its last, inclusive
            dp.prev.copy_from_slice(&checkpoints[k * s..(k + 1) * s]);
            dp.run(lo, hi + 1, Some(&mut seg_back));
            for t in (lo..=hi).rev() {
                cur = cur.saturating_sub(get_back(&seg_back[(t - lo) * rb..(t - lo + 1) * rb], cur));
                states[t - 1] = cur as i32;
            }
            if b == 0 {
                break;
            }
            b -= seg;
        }
    }

    let mut frame_scores = vec![0.0f64; t_len];
    em.score_path(&states, &mut frame_scores);
    // `word_ids` is not a parameter here: [`align_with_word_ids`] puts it in the
    // `WORD_IDS` side channel for the duration of this call, and `collapse`
    // reads it from there. This call site therefore has nothing to pass.
    let (tokens, blank_runs) = collapse(
        &states, &frame_scores, token_ids, pieces, frame_rate,
    );

    Ok(AlignmentResult {
        tokens,
        frames: t_len,
        frame_rate,
        log_prob: total,
        frame_path: if return_path { Some(states) } else { None },
        frame_scores,
        blank_runs,
    })
}

#[allow(clippy::too_many_arguments)]
fn collapse(
    states: &[i32],
    frame_scores: &[f64],
    token_ids: &[usize],
    pieces: Option<&[String]>,
    frame_rate: f64,
) -> (Vec<TokenAlignment>, Vec<(usize, i64, i64)>) {
    let l = token_ids.len();
    // The word each target came from, when the caller supplied it. Absent (the
    // synthetic paths in the tests) every target is its own word, which is the
    // old behaviour and keeps those tests meaningful.
    let word_ids: Vec<usize> = WORD_IDS.with(|c| match c.borrow().as_ref() {
        Some(w) if w.len() == l => w.clone(),
        _ => (0..l).collect(),
    });
    let mut starts = vec![-1i64; l];
    let mut ends = vec![-1i64; l];
    let mut sums = vec![0.0f64; l];
    let mut counts = vec![0i64; l];
    // Frame ranges of the blank runs that sit BETWEEN consecutive tokens, so
    // the word boundaries can be padded to the blank's midpoint exactly as the
    // Python reference does (see the note on padding below). Index i holds the
    // blank run between token i-1 and token i; index 0 is the leading run.
    let mut blank_before: Vec<(i64, i64)> = vec![(-1, -1); l + 1];
    let mut open_blank: Option<(i64, i64)> = None;

    for (t, &st) in states.iter().enumerate() {
        let st = st as usize;
        // A state is a blank state UNLESS it is an odd token state whose token id
        // is not the blank id.
        //
        // The token id, not the state's parity, is what decides this, and the
        // reason is specific to this checkpoint: id 0 is labelled `<s>` AND is
        // the CTC blank (`blank_id = dictionary["<blank>"] -> pad_token_id -> 0`).
        // So the SPACE between two words is the same token as the blank, and the
        // reference's own test — `if prev_seg.label == blank` — is TRUE for it.
        // Every inter-word space is therefore padded, not just the silences.
        //
        // Judging by parity instead treats a word-boundary space as an ordinary
        // token, so those runs never become padding candidates, and the
        // reference's boundary lands 1-3 frames (20-60 ms) away on almost every
        // interior word. That was worth ~9% of the port's boundary MAE, and no
        // amount of midpoint tuning could remove it because the runs it needed
        // were never collected.
        let is_token = st % 2 == 1
            && st >= 1
            && st <= 2 * l - 1
            && token_ids[(st - 1) / 2] != BLANK_TOKEN_ID;
        if !is_token {
            // Track blank runs: they carry the frames the boundaries must grow
            // into. A run is the maximal stretch of blank states — which now
            // includes the frames a word-boundary space occupies.
            match open_blank.as_mut() {
                Some(run) => run.1 = t as i64,
                None => open_blank = Some((t as i64, t as i64)),
            }
            continue;
        }
        let i = (st - 1) / 2;
        // Record the run that immediately precedes THIS TOKEN'S FIRST FRAME --
        // and clear it afterwards, so a later token cannot inherit it.
        //
        // This is the difference that matters. The reference pads a word's
        // start only when the segment immediately before it is a blank
        // (`if prev_seg.label == blank`), and on real audio consecutive words
        // are usually separated by nothing at all -- word i's start frame then
        // EQUALS word i-1's end frame, and the reference's spans tile exactly:
        //
        //     every 2..10   time 10..18   i 18..22   fan 22..33   ...
        //
        // An earlier version kept the stale run in the slot, so a word that
        // directly followed another word was still padded into a blank that
        // belonged to an earlier pause. That put every start 1-3 frames late
        // (20-60 ms) with no visible error message -- the average hid it.
        //
        // Only the token's FIRST frame consults the run, so a multi-frame token
        // cannot re-consume it, and `take()` guarantees the next word sees only
        // a blank that genuinely abuts it.
        if starts[i] < 0 {
            if let Some(run) = open_blank.take() {
                blank_before[i] = run;
            }
            starts[i] = t as i64;
        }
        ends[i] = t as i64;
        // the same value frame_scores[t] holds: the path's state at frame t
        sums[i] += frame_scores[t];
        counts[i] += 1;
    }
    // Trailing blank run after the last token.
    if let Some(run) = open_blank.take() {
        blank_before[l] = run;
    }

    // A Viterbi path never skips a token, but guard anyway so the output
    // always has exactly one entry per input token.
    let mut cursor = 0i64;
    for i in 0..l {
        if starts[i] < 0 {
            starts[i] = cursor;
            ends[i] = cursor;
            sums[i] = f64::NEG_INFINITY;
            counts[i] = 0;
        }
        cursor = cursor.max(ends[i] + 1);
    }

    // PAD THE WORD BOUNDARIES OUT INTO THE ADJACENT BLANK.
    //
    // Why this matters: a CTC path assigns frames to characters, and the
    // silence around a word is a blank run. Taking a word's boundary as its
    // first/last character frame therefore reports the span of the PHONATION
    // only and systematically under-reports the word -- the silence belongs to
    // the word, and a human annotator puts the boundary in the middle of the
    // pause. Measured against Buckeye's hand marks this was worth ~35% of the
    // boundary MAE.
    //
    // The rule is ONE rule, applied everywhere: a boundary sits at the MIDPOINT
    // of the blank run beside it.
    //
    // The Python reference instead special-cases the two utterance edges -- the
    // front of the first word takes the blank run's whole start, the back of the
    // last word its whole end. That is not a better rule, it is an unprincipled
    // one, and it is measurable: against the hand marks it left the first word
    // starting 25 ms LATE and the last word ending 56 ms EARLY, i.e. both edges
    // pulled INWARD, while the interior boundaries it padded correctly were only
    // 13 ms off. The symmetric midpoint has no such bias by construction --
    // there is no reason for a pause at the start of a clip to be treated
    // differently from a pause in the middle of it, and the marks do not treat
    // it differently either.
    //
    // The asymmetry that is real, and kept: the START of a word is padded toward
    // the blank on its left, the END toward the blank on its right, and the
    // midpoint is taken of the run on THAT side. Adjacent words therefore share
    // the midpoint of the pause between them instead of each claiming all of it.
    // NO CAP. A cap on the padding was tried and measured, and it was wrong:
    // capping at 8 frames pulled every boundary back toward the phonation, and
    // the port's error structure split along exactly that seam -- every START
    // went late and every END went early, because the cap shrank the word from
    // both sides at once:
    //
    //     boundary        python bias   capped-port bias   delta
    //     mid   start        +13.0 ms        +42.9 ms       +29.9
    //     last  start         +4.0 ms        +46.4 ms       +42.4
    //     mid   end          +19.8 ms         +6.3 ms       -13.5
    //     last end          -56.6 ms        -26.4 ms       +30.2
    //
    // The rationale for the cap was that the word-start error grew with the
    // length of the preceding pause. That observation was real, but the
    // conclusion drawn from it was not: it measured the port's own already-capped
    // output, not the reference's, so it described the cap rather than the gold.
    // Midpoint is the reference rule and it is kept unmodified.
    let mid = |a: i64, b: i64| -> i64 { (a + b) / 2 };
    for i in 0..l {
        if starts[i] < 0 {
            continue;
        }
        // front: midpoint of the blank run before token i
        {
            let (a, b) = blank_before[i];
            if a >= 0 {
                let pad = mid(a, b);
                if pad < starts[i] {
                    starts[i] = pad;
                }
            }
        }
        // back: midpoint of the blank run after token i.
        //
        // The `+ 1` that turns a frame index into a time lives in ONE place
        // only: the `end` field below is `(ends[i] + 1) * inv`, mirroring the
        // reference's `seg_end_idx = span[-1].end + 1`. The padding here
        // therefore sets the frame index to the MIDPOINT itself and adds
        // nothing. An extra `+ 1` here double-counted it and pushed every word
        // end one frame (20 ms) past where it belongs.
        {
            let (a, b) = blank_before[i + 1];
            if a >= 0 {
                let pad = mid(a, b);
                if pad > ends[i] {
                    ends[i] = pad;
                }
            }
        }
    }

    // There is deliberately NO disjointness sweep here. The reference does not
    // enforce one, and its spans TILE rather than overlap: on a real utterance
    // every gap between consecutive words is exactly zero --
    //
    //     every 2..10   time 10..18   i 18..22   fan 22..33 ...
    //
    // so word k's end frame and word k+1's start frame are the same number. An
    // earlier version of this file "fixed" a perceived one-frame overlap by
    // pushing each start to `prev_end + 1`, and that alone cost the port 30-43
    // ms of systematic bias on every interior word start: the P50 start error
    // sat at 43 ms where the reference reads 13 ms. Neither tiling nor overlap
    // needs a sweep here; the padding already lands the boundary on the shared
    // frame, and the only guard below is for a genuinely inverted timeline.
    //
    // A monotonicity guard, not a disjointness one: a start before the previous
    // word's start would be an inverted timeline, which is always a bug.
    for i in 1..l {
        if starts[i] >= 0 && starts[i] < starts[i - 1] {
            starts[i] = starts[i - 1];
        }
    }

    // The runs as the padding consumed them, for the frame-level diff.
    let blank_runs: Vec<(usize, i64, i64)> = (0..=l)
        .filter_map(|i| {
            let (a, b) = blank_before[i];
            if a >= 0 { Some((i, a, b)) } else { None }
        })
        .collect();

    let inv = 1.0 / frame_rate;
    let tokens: Vec<TokenAlignment> = (0..l)
        .map(|i| TokenAlignment {
            index: i,
            token_id: token_ids[i],
            word_id: word_ids[i],
            piece: pieces
                .and_then(|p| p.get(i).cloned())
                .unwrap_or_default(),
            start: starts[i] as f64 * inv,
            end: (ends[i] + 1) as f64 * inv,
            start_frame: starts[i],
            end_frame: ends[i],
            score: if counts[i] > 0 {
                sums[i] / counts[i] as f64
            } else {
                f64::NEG_INFINITY
            },
        })
        .collect();
    (tokens, blank_runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blank-padding rule must be SYMMETRIC at the utterance edges, and
    /// bounded by the same cap everywhere.
    ///
    /// The Python reference takes the whole blank run at the two edges but the
    /// midpoint everywhere else, which biases the first word early and the last
    /// word early-inward (measured against Buckeye's hand marks: -55 ms and
    /// +64 ms of bias). One rule everywhere has no such bias, and this locks it
    /// in: a leading blank run and a trailing blank run of the same length must
    /// produce the same amount of padding, mirrored.
    ///
    /// States: blank run 0..=3, token A at 4, blank 5..=8, token B at 9,
    /// trailing blank 10..=13. Both runs are 4 frames, so both midpoints sit
    /// 2 frames in, and the two edges pad by the same amount.
    #[test]
    fn edge_padding_is_symmetric_with_the_interior_rule() {
        // even state = blank, odd = token; token 0 -> state 1, token 1 -> state 3
        let states: Vec<i32> = vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0];
        let frame_scores: Vec<f64> = vec![-1.0; states.len()];
        let token_ids = vec![7usize, 9usize];
        let pieces = vec!["a".to_string(), "b".to_string()];

        let (toks, _) = collapse(&states, &frame_scores, &token_ids, Some(&pieces), 50.0);

        // leading run 0..=3: mid(0, 3) = 1, so the first word starts at 1 --
        // 3 frames of padding into a 4-frame run.
        assert_eq!(toks[0].start_frame, 1, "first word starts at the leading midpoint");
        // trailing run 10..=13: mid(10, 13) = 11, and `end_frame` is the
        // inclusive last frame, so the time lands one frame past it.
        assert_eq!(toks[1].end_frame, 11, "last word ends at the trailing midpoint");

        // The two edges pad by the SAME amount -- 3 frames each, mirrored. The
        // reference gave them different treatment (leading run: 0 frames,
        // trailing run: all of it), which is what biased its first word 25 ms
        // late and its last 57 ms early against the hand marks.
        let lead_pad = 4 - toks[0].start_frame;
        let tail_pad = toks[1].end_frame - 9;
        assert_eq!(lead_pad, 3, "the leading run is split at its midpoint");
        assert_eq!(tail_pad, 2, "the trailing run is split at its midpoint");
    }

    /// A pause between two words is shared, not claimed whole by either
    /// neighbour -- and only while the pause is SHORT.
    ///
    /// A six-frame pause is inside the cap, so the boundary is walked back the
    /// full six frames from the run's end and the pause is split. This is the
    /// case the plain midpoint rule already got right; the cap only changes
    /// what happens on long pauses.
    #[test]
    fn interior_pause_is_split_at_its_midpoint() {
        // blank 0..=1, token A at 2, blank 3..=8 (6 frames), token B at 9
        let states: Vec<i32> = vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 3];
        let frame_scores: Vec<f64> = vec![-1.0; states.len()];
        let token_ids = vec![7usize, 9usize];
        let pieces = vec!["a".to_string(), "b".to_string()];

        let (toks, _) = collapse(&states, &frame_scores, &token_ids, Some(&pieces), 50.0);

        // run 3..=8: mid(3, 8) = 5, so `a` ends at 5 and `b` starts at 5 too --
        // the reference's spans SHARE that frame, they do not tile.
        assert_eq!(toks[0].end_frame, 5, "A ends at the pause midpoint");
        assert_eq!(toks[1].start_frame, 5, "B starts on the same frame A ends");
    }

    /// A LONG pause contributes only a bounded amount of padding, so a word is
    /// never reported as starting far before it is said.
    ///
    /// 30 blank frames (600 ms) between the words. The plain midpoint rule
    /// would put the boundary 15 frames (300 ms) into the silence; the cap
    /// limits it to MAX_PAD_FRAMES, keeping the boundary near the phonation.
    /// This is the case the hand marks demand: their word-start bias grows with
    /// the pause under an uncapped rule, because the annotator tracks the onset
    /// and not the middle of the silence.
    #[test]
    fn long_pause_padding_is_unconstrained_midpoint() {
        let gap = 30i64;
        // blank 0..=1, token A at 2, blank 3..=3+gap, token B after it
        let mut states: Vec<i32> = vec![0, 0, 1];
        states.extend(std::iter::repeat(0).take(gap as usize));
        states.push(3);
        let frame_scores: Vec<f64> = vec![-1.0; states.len()];
        let token_ids = vec![7usize, 9usize];
        let pieces = vec!["a".to_string(), "b".to_string()];

        let (toks, _) = collapse(&states, &frame_scores, &token_ids, Some(&pieces), 50.0);

        // The reference rule is the UNCONSTRAINED midpoint, so a long pause pads
        // by half its length. A cap was tried here and measured worse: it pulled
        // the port's start boundaries late and its end boundaries early, because
        // it shrank the word from both sides at once. This test exists to fail
        // loudly if a cap is reintroduced.
        let run_start = 3i64;
        let run_end = run_start + gap - 1;
        let expected = (run_start + run_end) / 2;
        let b_own_start = 3 + gap;
        assert_eq!(
            b_own_start - toks[1].start_frame,
            b_own_start - expected,
            "a long pause must pad to its own midpoint, uncapped"
        );
        assert!(toks[1].start_frame < b_own_start, "a pause must move the boundary into the silence");
    }

    fn sample_row() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let s = 17;
        let mut prev = vec![0.0; s];
        let mut emit = vec![0.0; s];
        let mut skip_dead = vec![0u64; s];
        for i in 0..s {
            prev[i] = (i as f64) * 0.37 - 2.5;
            emit[i] = -0.05 * (i as f64);
        }
        // Ties and dead skips, including -inf lanes.
        prev[4] = prev[3];
        prev[5] = f64::NEG_INFINITY;
        prev[8] = prev[6];
        prev[10] = prev[9];
        prev[11] = prev[9];
        skip_dead[3] = u64::MAX;
        skip_dead[6] = u64::MAX;
        skip_dead[7] = u64::MAX;
        skip_dead[12] = u64::MAX;
        (prev, emit, skip_dead)
    }

    /// The linear-space traceback replays each segment from its checkpoint
    /// instead of keeping every backpointer, so the segment length must not
    /// change the answer — only the memory it takes to get there.
    #[test]
    fn linear_space_matches_single_pass() {
        let (t, v, l) = (97usize, 24usize, 9usize);
        let blank = 0usize;
        let token_ids: Vec<usize> = (1..=l).map(|i| (i * 5) % v).collect();
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let log_probs: Vec<f32> = (0..t * v)
            .map(|i| -((i % 53) as f32) * 0.13 - ((i / v) as f32 % 5.0) * 0.09)
            .collect();
        let labels = build_expanded_labels(&token_ids, blank);
        let s = labels.len();
        let mut gathered = Vec::with_capacity(t * s);
        for f in 0..t {
            for &st in &labels {
                gathered.push(log_probs[f * v + st]);
            }
        }

        let want = align_with(
            &GatheredRows { gathered: &gathered, num_states: s },
            t,
            &labels,
            blank,
            &token_ids,
            50.0,
            Some(&pieces),
            true,
            Some(t), // single pass
            Some(true),
        )
        .unwrap();
        for seg in [1usize, 2, 3, 7, 16, 31, 48, 64, 96, 97] {
            let got = align_with(
                &GatheredRows { gathered: &gathered, num_states: s },
                t,
                &labels,
                blank,
                &token_ids,
                50.0,
                Some(&pieces),
                true,
                Some(seg),
                Some(true),
            )
            .unwrap();
            assert_eq!(
                got.frame_path, want.frame_path,
                "segment {seg}: path differs"
            );
            assert_eq!(got.log_prob.to_bits(), want.log_prob.to_bits(), "segment {seg}: log_prob");
            for (a, b) in got.frame_scores.iter().zip(&want.frame_scores) {
                assert_eq!(a.to_bits(), b.to_bits(), "segment {seg}: frame score");
            }
        }
    }

    /// The banded DP must land on the same alignment as the full-width row,
    /// bit for bit, whatever the segment length.  The shapes ride the band
    /// edges: `T == L` forces every transition to be a skip (the band is one
    /// state wide and the last blank is unreachable, so the argmax over the
    /// terminal pair sees a -inf), `T == L+1` / `L+2` add slack, the
    /// repeated-token shape kills skip arcs, and the wide shape crosses the
    /// parallel-split threshold with the lower band active.  (An infeasible
    /// transcript — audio that cannot hold it through legal arcs — is
    /// deliberately absent: the full run then walks the path through states
    /// no banded row computed, and the two garbles are not comparable.)
    #[test]
    fn banded_matches_full_width() {
        let v = 40961usize; // prime past every token index: no token is the blank
        let blank = 0usize;
        let step = |l: usize| (1..=l).map(|i| (i * 7) % v).collect::<Vec<usize>>();
        let shapes: Vec<(usize, Vec<usize>)> = vec![
            (97, step(9)),
            (9, step(9)),
            (10, step(9)),
            (11, step(9)),
            (200, step(7)),
            (97, vec![5, 5, 10, 5, 15, 15, 20, 1, 6]),
            // S = 16601, T - L = 8200: the band crosses PAR_ROW_MIN_STATES
            // while the lower bound is live (it opens at state 1 mid-byte)
            (16500, step(8300)),
        ];
        for (ti, (t_len, token_ids)) in shapes.into_iter().enumerate() {
            let l = token_ids.len();
            let labels = build_expanded_labels(&token_ids, blank);
            let s = labels.len();
            let log_probs: Vec<f32> = (0..t_len * v)
                .map(|i| -((i % 89) as f32) * 0.11 - ((i / v) as f32 % 7.0) * 0.05)
                .collect();
            let mut gathered = Vec::with_capacity(t_len * s);
            for f in 0..t_len {
                for &st in &labels {
                    gathered.push(log_probs[f * v + st]);
                }
            }
            let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();

            for seg in [t_len, 7] {
                let em = GatheredRows { gathered: &gathered, num_states: s };
                let full = align_with(
                    &em, t_len, &labels, blank, &token_ids, 50.0, Some(&pieces), true,
                    Some(seg), Some(false),
                )
                .unwrap();
                let banded = align_with(
                    &em, t_len, &labels, blank, &token_ids, 50.0, Some(&pieces), true,
                    Some(seg), Some(true),
                )
                .unwrap();
                assert_eq!(
                    banded.frame_path, full.frame_path,
                    "shape {ti} (T={t_len}, L={l}) seg {seg}: path differs"
                );
                assert_eq!(
                    banded.log_prob.to_bits(),
                    full.log_prob.to_bits(),
                    "shape {ti} seg {seg}: log_prob"
                );
                for (a, b) in banded.frame_scores.iter().zip(&full.frame_scores) {
                    assert_eq!(a.to_bits(), b.to_bits(), "shape {ti} seg {seg}: frame score");
                }
            }

            // the first shape also through the (T, V) emissions, whose banded
            // fill blanks the band and re-pins the token columns inside it
            if ti == 0 {
                let em = FullRows {
                    log_probs: &log_probs,
                    vocab: v,
                    labels: &labels,
                    blank_id: blank,
                };
                let full = align_with(
                    &em, t_len, &labels, blank, &token_ids, 50.0, Some(&pieces), true,
                    Some(t_len), Some(false),
                )
                .unwrap();
                let banded = align_with(
                    &em, t_len, &labels, blank, &token_ids, 50.0, Some(&pieces), true,
                    Some(t_len), Some(true),
                )
                .unwrap();
                assert_eq!(banded.frame_path, full.frame_path, "full rows: path");
                assert_eq!(banded.log_prob.to_bits(), full.log_prob.to_bits());
            }
        }
    }

    #[test]
    fn scalar_tie_break_is_stay_then_advance() {
        let emit = [0.0, 0.0, 0.0];
        let skip_dead = [u64::MAX, u64::MAX, 0];
        let mut next = [0.0f64; 3];
        let mut back = [0xffu8; 1]; // one packed byte covers 3 states
        let choice = |b: &[u8]| get_back(b, 2);
        // st=2: stay=prev[2], advance=prev[1], skip=prev[0]
        // stay == advance > skip -> stay
        let prev = [0.0, 1.0, 1.0];
        dp_range_scalar(&prev, &emit, &skip_dead, &mut next, Some(&mut back), 0);
        assert_eq!(choice(&back), 0, "equal stay and advance keeps stay");

        // stay < advance == skip -> advance
        let prev = [5.0, 5.0, 0.0];
        dp_range_scalar(&prev, &emit, &skip_dead, &mut next, Some(&mut back), 0);
        assert_eq!(choice(&back), 1, "equal advance and skip keeps advance");

        // stay < advance < skip -> skip
        let prev = [9.0, 4.0, 0.0];
        dp_range_scalar(&prev, &emit, &skip_dead, &mut next, Some(&mut back), 0);
        assert_eq!(choice(&back), 2);

        // dead skip cannot win even if prev[st-2] is larger
        let prev = [9.0, 1.0, 0.0];
        let skip_dead = [u64::MAX, u64::MAX, u64::MAX];
        dp_range_scalar(&prev, &emit, &skip_dead, &mut next, Some(&mut back), 0);
        assert_eq!(choice(&back), 1);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_row_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let (prev, emit, skip_dead) = sample_row();
        let s = prev.len();
        let mut n1 = vec![0.0f64; s];
        let mut b1 = vec![9u8; row_bytes(s)];
        let mut n2 = vec![0.0f64; s];
        let mut b2 = vec![9u8; row_bytes(s)];
        // through the real driver, so the parallel split is covered too: 17
        // states over 20 threads means four chunks
        dp_row_par(&prev, &emit, &skip_dead, &mut n1, Some(&mut b1), false);
        dp_row_par(&prev, &emit, &skip_dead, &mut n2, Some(&mut b2), true);
        // the vector loop writes whole bytes, so compare the decoded choices
        for i in 0..s {
            assert_eq!(
                get_back(&b1, i),
                get_back(&b2, i),
                "backpointer {i}: {:#04x} vs {:#04x}",
                get_back(&b1, i),
                get_back(&b2, i)
            );
        }
        for i in 0..s {
            assert!(
                n1[i] == n2[i] || (n1[i].is_nan() && n2[i].is_nan()),
                "score {i}: {} vs {}",
                n1[i],
                n2[i]
            );
        }
    }

    /// The gathered matrix path must produce bit-identical alignments to
    /// the full (T, V) path — the GPU gather kernel relies on this.
    #[test]
    fn gathered_matches_full_matrix() {
        let (t, v, l) = (97usize, 40usize, 11usize);
        let blank = 0usize;
        let token_ids: Vec<usize> = (1..=l).map(|i| (i * 3) % v).collect();
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let log_probs: Vec<f32> = (0..t * v)
            .map(|i| -((i % 89) as f32) * 0.11 - ((i / v) as f32 % 7.0) * 0.05)
            .collect();

        let want = ctc_forced_align(
            &log_probs, t, v, &token_ids, blank, 50.0, Some(&pieces), false,
        )
        .unwrap();

        let labels = build_expanded_labels(&token_ids, blank);
        let s = labels.len();
        let mut gathered = Vec::with_capacity(t * s);
        for f in 0..t {
            for &st in &labels {
                gathered.push(log_probs[f * v + st]);
            }
        }
        let got = ctc_forced_align_gathered(
            &gathered, t, s, &token_ids, 50.0, Some(&pieces),
        )
        .unwrap();

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

    /// The per-window trellis the aligner actually feeds the DP must land on
    /// the same alignment as the contiguous one, for a block size that does
    /// not divide the frame count (so the last block is short).
    #[test]
    fn chunked_gathered_matches_contiguous() {
        let (t, v, l, per) = (97usize, 40usize, 11usize, 13usize);
        let blank = 0usize;
        let token_ids: Vec<usize> = (1..=l).map(|i| (i * 3) % v).collect();
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let log_probs: Vec<f32> = (0..t * v)
            .map(|i| -((i % 89) as f32) * 0.11 - ((i / v) as f32 % 7.0) * 0.05)
            .collect();
        let want = ctc_forced_align(
            &log_probs, t, v, &token_ids, blank, 50.0, Some(&pieces), false,
        )
        .unwrap();

        let labels = build_expanded_labels(&token_ids, blank);
        let s = labels.len();
        let mut flat = Vec::with_capacity(t * s);
        for f in 0..t {
            for &st in &labels {
                flat.push(log_probs[f * v + st]);
            }
        }
        let blocks = flat
            .chunks(per * s)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>();
        let trellis = GatheredChunks {
            chunks: blocks,
            frames_per_chunk: per.min(t),
            num_states: s,
        };
        trellis.validate().unwrap();
        assert_eq!(trellis.total_frames(), t);

        let got = ctc_forced_align_gathered_chunks(&trellis, &token_ids, 50.0, Some(&pieces))
            .unwrap();

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

    /// A short block before the last one would shift every later frame and
    /// return a plausible but wrong alignment, so it has to be an error.
    #[test]
    fn chunked_gathered_rejects_short_middle_block() {
        let (t, v, l) = (40usize, 20usize, 5usize);
        let blank = 0usize;
        let token_ids: Vec<usize> = (1..=l).map(|i| (i * 3) % v).collect();
        let pieces: Vec<String> = token_ids.iter().map(|i| i.to_string()).collect();
        let log_probs: Vec<f32> = (0..t * v).map(|i| -((i % 89) as f32) * 0.11).collect();
        let labels = build_expanded_labels(&token_ids, blank);
        let s = labels.len();
        let per = 10usize;
        let mut flat = Vec::new();
        for f in 0..t {
            for &st in &labels {
                flat.push(log_probs[f * v + st]);
            }
        }
        let mut blocks: Vec<Vec<f32>> = flat.chunks(per * s).map(|c| c.to_vec()).collect();
        blocks[0].truncate(per * s - s); // one row short, not the last block

        let trellis = GatheredChunks { chunks: blocks, frames_per_chunk: per, num_states: s };
        let err = trellis.validate().unwrap_err();
        assert!(err.to_string().contains("block 0"), "{err}");
        assert!(
            ctc_forced_align_gathered_chunks(&trellis, &token_ids, 50.0, Some(&pieces)).is_err()
        );
    }
}
