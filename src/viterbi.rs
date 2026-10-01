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

/// One aligned token and its time span.
#[derive(Debug, Clone)]
pub struct TokenAlignment {
    pub index: usize,
    pub token_id: usize,
    pub piece: String,
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
fn dp_row_scalar(prev: &[f64], emit: &[f64], skip_dead: &[u64], next: &mut [f64], back: &mut [u8]) {
    let s = prev.len();
    for st in 0..s {
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
        back[st] = choice;
        next[st] = best + emit[st];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dp_row_avx2(prev: &[f64], emit: &[f64], skip_dead: &[u64], next: &mut [f64], back: &mut [u8]) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_and_pd, _mm256_blendv_pd, _mm256_castsi256_pd, _mm256_cmp_pd,
        _mm256_loadu_pd, _mm256_loadu_si256, _mm256_movemask_pd, _mm256_set1_pd, _mm256_storeu_pd,
        _CMP_GE_OQ,
    };
    let s = prev.len();
    // States 0 and 1 have no legal skip; the vector loop starts where st-2 is in range.
    if s > 0 {
        dp_one(0, prev, emit, skip_dead, next, back);
    }
    if s > 1 {
        dp_one(1, prev, emit, skip_dead, next, back);
    }
    let neginf = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut st = 2;
    while st + 4 <= s {
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
        _mm256_storeu_pd(next.as_mut_ptr().add(st), out);
        let stay_mask = _mm256_movemask_pd(stay_wins);
        let adv_mask = _mm256_movemask_pd(adv_ge_skip);
        for lane in 0..4 {
            let bit = 1 << lane;
            back[st + lane] = if stay_mask & bit != 0 {
                0
            } else if adv_mask & bit != 0 {
                1
            } else {
                2
            };
        }
        st += 4;
    }
    while st < s {
        dp_one(st, prev, emit, skip_dead, next, back);
        st += 1;
    }
}

fn dp_one(st: usize, prev: &[f64], emit: &[f64], skip_dead: &[u64], next: &mut [f64], back: &mut [u8]) {
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
    back[st] = choice;
    next[st] = best + emit[st];
}

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
trait Emissions {
    /// Fill `emit[0..num_states]` with frame `t`'s state scores.
    fn fill_emit(&self, t: usize, emit: &mut [f64], token_ids: &[usize]);
    /// Score of expanded state `st` at frame `t` (frame_scores, collapse).
    fn score(&self, t: usize, st: usize) -> f32;
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
    fn score(&self, t: usize, st: usize) -> f32 {
        self.gathered[t * self.num_states + st]
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
    align(em, num_frames, &labels, blank_id, token_ids, frame_rate, pieces, return_path)
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
    align(em, num_frames, &labels, usize::MAX, token_ids, frame_rate, pieces, false)
}

fn align(
    em: impl Emissions,
    t_len: usize,
    labels: &[usize],
    blank_id: usize,
    token_ids: &[usize],
    frame_rate: f64,
    pieces: Option<&[String]>,
    return_path: bool,
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
        });
    }
    if t_len < l {
        anyhow::bail!(
            "Audio too short: {t_len} frames cannot hold {l} tokens."
        );
    }

    // state 0 = first blank, state 1 = first token.
    // Expanded labels are [blank, tok, blank, tok, ..., blank], so even states
    // all emit the blank and odd states emit token_ids in order.
    let mut prev = vec![f64::NEG_INFINITY; s];
    {
        prev[0] = em.score(0, 0) as f64;
        if s > 1 {
            prev[1] = em.score(0, 1) as f64;
        }
    }

    // 0 = stay, 1 = from s-1, 2 = from s-2. Row 0 is never read (traceback
    // starts at t >= 1); every later row is written in full before that.
    let nback = t_len.checked_mul(s).context("backpointer size")?;
    let mut back: Vec<u8> = Vec::new();
    back.try_reserve_exact(nback).context("backpointer alloc")?;
    // SAFETY: u8 has no destructor and no invalid bit patterns. Traceback
    // reads only rows t >= 1, and the DP writes each of those rows completely.
    unsafe { back.set_len(nback) };

    // All-ones lane => skip is illegal (force -inf). Zero => skip is allowed.
    let mut skip_dead = vec![0u64; s];
    for st in 2..s {
        if labels[st] == blank_id || labels[st] == labels[st - 2] {
            skip_dead[st] = u64::MAX;
        }
    }

    let mut emit = vec![0.0f64; s];
    let mut next = vec![0.0f64; s];
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2");
    for t in 1..t_len {
        em.fill_emit(t, &mut emit, token_ids);
        let back_row = &mut back[t * s..(t + 1) * s];
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            // SAFETY: use_avx2 is the runtime AVX2 check.
            unsafe { dp_row_avx2(&prev, &emit, &skip_dead, &mut next, back_row) };
        } else {
            dp_row_scalar(&prev, &emit, &skip_dead, &mut next, back_row);
        }
        #[cfg(not(target_arch = "x86_64"))]
        dp_row_scalar(&prev, &emit, &skip_dead, &mut next, back_row);
        std::mem::swap(&mut prev, &mut next);
    }
    let score = prev;

    // termination: last blank or last token, whichever scores higher
    let mut s_end = s - 1;
    if s >= 2 && score[s - 2] > score[s - 1] {
        s_end = s - 2;
    }
    let total = score[s_end];

    let mut states = vec![0i32; t_len];
    states[t_len - 1] = s_end as i32;
    let mut cur = s_end;
    for t in (1..t_len).rev() {
        cur -= back[t * s + cur] as usize;
        states[t - 1] = cur as i32;
    }

    let frame_scores: Vec<f64> = (0..t_len)
        .map(|t| em.score(t, states[t] as usize) as f64)
        .collect();
    let tokens = collapse(&states, &em, token_ids, pieces, frame_rate);

    Ok(AlignmentResult {
        tokens,
        frames: t_len,
        frame_rate,
        log_prob: total,
        frame_path: if return_path { Some(states) } else { None },
        frame_scores,
    })
}

#[allow(clippy::too_many_arguments)]
fn collapse(
    states: &[i32],
    em: &impl Emissions,
    token_ids: &[usize],
    pieces: Option<&[String]>,
    frame_rate: f64,
) -> Vec<TokenAlignment> {
    let l = token_ids.len();
    let mut starts = vec![-1i64; l];
    let mut ends = vec![-1i64; l];
    let mut sums = vec![0.0f64; l];
    let mut counts = vec![0i64; l];

    for (t, &st) in states.iter().enumerate() {
        let st = st as usize;
        let is_token = st % 2 == 1 && st >= 1 && st <= 2 * l - 1;
        if !is_token {
            continue;
        }
        let i = (st - 1) / 2;
        if starts[i] < 0 {
            starts[i] = t as i64;
        }
        ends[i] = t as i64;
        sums[i] += em.score(t, st) as f64;
        counts[i] += 1;
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

    let inv = 1.0 / frame_rate;
    (0..l)
        .map(|i| TokenAlignment {
            index: i,
            token_id: token_ids[i],
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
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn scalar_tie_break_is_stay_then_advance() {
        let emit = [0.0, 0.0, 0.0];
        let skip_dead = [u64::MAX, u64::MAX, 0];
        let mut next = [0.0; 3];
        let mut back = [9u8; 3];
        // st=2: stay=prev[2], advance=prev[1], skip=prev[0]
        // stay == advance > skip -> stay
        let prev = [0.0, 1.0, 1.0];
        dp_row_scalar(&prev, &emit, &skip_dead, &mut next, &mut back);
        assert_eq!(back[2], 0, "equal stay and advance keeps stay");

        // stay < advance == skip -> advance
        let prev = [5.0, 5.0, 0.0];
        dp_row_scalar(&prev, &emit, &skip_dead, &mut next, &mut back);
        assert_eq!(back[2], 1, "equal advance and skip keeps advance");

        // stay < advance < skip -> skip
        let prev = [9.0, 4.0, 0.0];
        dp_row_scalar(&prev, &emit, &skip_dead, &mut next, &mut back);
        assert_eq!(back[2], 2);

        // dead skip cannot win even if prev[st-2] is larger
        let prev = [9.0, 1.0, 0.0];
        let skip_dead = [u64::MAX, u64::MAX, u64::MAX];
        dp_row_scalar(&prev, &emit, &skip_dead, &mut next, &mut back);
        assert_eq!(back[2], 1);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_row_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let (prev, emit, skip_dead) = sample_row();
        let mut n1 = vec![0.0; prev.len()];
        let mut b1 = vec![9u8; prev.len()];
        let mut n2 = vec![0.0; prev.len()];
        let mut b2 = vec![9u8; prev.len()];
        dp_row_scalar(&prev, &emit, &skip_dead, &mut n1, &mut b1);
        unsafe { dp_row_avx2(&prev, &emit, &skip_dead, &mut n2, &mut b2) };
        assert_eq!(b1, b2);
        for i in 0..prev.len() {
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
}
