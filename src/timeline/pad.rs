//! Rule 1: a boundary sits in the middle of the pause beside it — unless the
//! pause is long enough to be silence, in which case the word starts where
//! its own evidence starts.
//!
//! Operates on FRAME indices, not seconds. `collapse` reads the path into
//! frame numbers, this decides which frame each boundary lands on, and
//! `collapse` converts to seconds afterwards — the `+ 1` that turns a frame
//! index into a time therefore lives in exactly one place, and it is not
//! here.

/// How far past its own last frame a token may claim the following blank run.
///
/// Measured, not guessed, and not taken from another model: over 15,722
/// aligned characters spanning seven languages, the depth a token's tail
/// reaches past the last detected speech has median 0.00 s, p99 0.00 s and a
/// maximum of 1.57 s. Read speech simply does not produce long tails, so a
/// bound set at the top of that range is nearly free, and on material that
/// DOES have long pauses it is the difference between a subtitle that ends
/// when the speaker stops and one that ends halfway through the silence that
/// follows.
///
/// A multiple of the mean token duration cannot do this job. Measured both
/// ways, a 1.5x-mean rule fired on 7.9% of this corpus and cut more real
/// speech than it released, because in read speech the long spans it caught
/// were held vowels. FireRedASR2 caps by a multiple; its model is not this
/// one and neither is its number.
///
/// 1.0 s is where the two sets of evidence meet. On broadcast material it
/// takes the worst end error from 8.4 s to 1.7 s and the end MAE from 896 ms
/// to 486 ms, with every start boundary bit-identical -- the bound is
/// one-sided and never reaches a start. On an 89 s Japanese variety-show
/// clip, dense speech with no gap over 1.85 s, it is inert and produces
/// output identical to no cap at all.
pub(crate) const MAX_PADDING_SEC: f64 = 1.0;

/// The longest blank run that still counts as a pause BESIDE a word rather
/// than silence BETWEEN words.
///
/// Below the bound the front boundary sits at the run's midpoint — the pause
/// is prosody, it belongs to the words on both sides, and a human annotator
/// splits it down the middle (the measurement that chose the midpoint rule is
/// in [`pad_into_silence`]). Above it the run is silence, and silence belongs
/// to no word: a word begins where its own evidence begins.
///
/// Where the two regimes must meet: the midpoint puts a start `run/2` frames
/// early, so its error grows with the pause, while the evidence edge costs
/// one frame of coarticulation regardless. Past half a second a read-speech
/// gap is inter-sentence silence, not coarticulation, and the midpoint's
/// error has already passed 250 ms — ten times the evidence edge's.
///
/// Measured on 10 clean Mandarin sentences (FLEURS cmn test, leading
/// silences 0.9–3.7 s): the unsplit midpoint put the first character 235 ms
/// before the acoustic onset at the median and 740 ms at the worst — clip07
/// dumps it exactly: the path sits on 它 at frame 179, the blank run before
/// it is frames 114–178, and the midpoint moved the boundary to frame 146,
/// 33 frames of invented early start. Qwen3-ForcedAligner, which claims no
/// silence at all, held 85 ms at the median on the same clips — and most of
/// THAT is the energy-onset detector lagging on aspirated onsets, not
/// placement error. With the split, every long-run case lands on the
/// evidence frame.
///
/// The bound is deliberately generous towards the midpoint: Buckeye — where
/// the midpoint rule was measured against hand marks — is dense
/// conversational speech whose word-fronting runs sit far below it, so
/// English behaviour is unchanged wherever the original rule was actually
/// doing its job. The one-sidedness of [`MAX_PADDING_SEC`] (ends may claim,
/// starts may not) is therefore no longer unbounded: both sides of a
/// boundary now have a stated limit.
pub(crate) const MAX_PAUSE_SEC: f64 = 0.5;

/// Move every boundary into the silence beside it, then bound how much of
/// that silence one token may keep.
///
/// `own_ends[i]` is where the path last sat on token `i`, before the midpoint
/// rule reached it forward. `blank_before[i]` is the blank run immediately
/// before token `i` as `(first_frame, last_frame)`, or `(-1, -1)` when the
/// token abuts the previous one with no blank at all.
///
/// This is rule 1 of [`crate::timeline`] and it runs first, because
/// [`super::anchor_marks`] and [`super::place_unmeasured`] both read the ends
/// it leaves.
pub(crate) fn pad_into_silence(
    starts: &mut [i64],
    ends: &mut [i64],
    own_ends: &[i64],
    blank_before: &[(i64, i64)],
    frame_rate: f64,
) {
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
    // 13 ms off. The symmetric midpoint has no such bias by construction -- there
    // is no reason for a pause at the start of a clip to be treated differently
    // from a pause in the middle of it, and the marks do not treat it
    // differently either.
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
    for i in 0..starts.len() {
        if starts[i] < 0 {
            continue;
        }
        // front: the blank run before token i.
        //
        // A SHORT run is a prosodic pause: it belongs to the word, and the
        // boundary sits at its midpoint — the annotator habit, measured
        // against Buckeye's hand marks in the essay above. A LONG run is
        // silence, and silence is nobody's: the word starts where its own
        // evidence starts, one frame of grace for coarticulation. One rule
        // cannot serve both — the midpoint's error is run/2 and unbounded in
        // the pause length, which is how a 3.7 s clip-opening silence put a
        // Mandarin first character 1.86 s early (see MAX_PAUSE_SEC).
        {
            let (a, b) = blank_before[i];
            if a >= 0 {
                let run_sec = (b - a + 1) as f64 / frame_rate;
                let pad = if run_sec <= MAX_PAUSE_SEC {
                    mid(a, b)
                } else {
                    (starts[i] - 1).max(0)
                };
                if pad < starts[i] {
                    starts[i] = pad;
                }
            }
        }
        // back: midpoint of the blank run after token i.
        //
        // The `+ 1` that turns a frame index into a time lives in ONE place
        // only: the `end` field the caller writes is `(ends[i] + 1) * inv`,
        // mirroring the reference's `seg_end_idx = span[-1].end + 1`. The
        // padding here therefore sets the frame index to the MIDPOINT itself
        // and adds nothing. An extra `+ 1` here double-counted it and pushed
        // every word end one frame (20 ms) past where it belongs.
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

    let max_pad = (MAX_PADDING_SEC * frame_rate).round() as i64;
    for i in 0..starts.len() {
        if starts[i] < 0 {
            continue;
        }
        // `own_ends[i]` is where the path last sat on this token, before the
        // midpoint rule reached it forward. Clamping to that plus a bound
        // leaves the token's own timing untouched and only limits how much of
        // the pause it may absorb.
        if ends[i] - own_ends[i] > max_pad {
            ends[i] = own_ends[i] + max_pad;
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
    for i in 1..starts.len() {
        if starts[i] >= 0 && starts[i] < starts[i - 1] {
            starts[i] = starts[i - 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `blank_before` holds one entry per token PLUS one for the trailing run,
    /// and `(-1, -1)` is what "there is no blank here" looks like.
    const NONE: (i64, i64) = (-1, -1);

    #[test]
    fn a_boundary_lands_in_the_middle_of_its_pause() {
        // 拉 on frames 10..14, 就 on 22..26, so the pause is 15..21 and its
        // midpoint is 18: 拉's end and 就's start must BOTH move there.
        let mut starts = vec![10, 22];
        let mut ends = vec![14, 26];
        let own = ends.clone();
        pad_into_silence(&mut starts, &mut ends, &own, &[NONE, (15, 21), NONE], 50.0);
        assert_eq!((starts[0], ends[0]), (10, 18), "start left alone, end moved to the middle");
        assert_eq!((starts[1], ends[1]), (18, 26), "the next start met it exactly");
    }

    #[test]
    fn words_with_no_blank_between_them_still_meet() {
        // Real speech: word k's end frame IS word k+1's start frame.
        let mut starts = vec![10, 14];
        let mut ends = vec![14, 20];
        let own = ends.clone();
        pad_into_silence(&mut starts, &mut ends, &own, &[NONE, NONE, NONE], 50.0);
        assert_eq!((starts[0], ends[0]), (10, 14), "nothing to pad into");
        assert_eq!((starts[1], ends[1]), (14, 20), "nor here");
    }

    #[test]
    fn a_token_cannot_claim_more_than_the_bound() {
        // 拉's own end is 14 and the pause runs to 1000, so the midpoint is far
        // past the 1.0 s bound (50 frames) and the end is clamped to 64.
        let mut starts = vec![10];
        let mut ends = vec![14];
        pad_into_silence(&mut starts, &mut ends, &[14], &[NONE, (15, 1000), NONE], 50.0);
        assert_eq!(ends[0], 64, "own end 14 plus the 1.0 s bound");
    }

    #[test]
    fn a_long_leading_run_is_silence_and_the_word_starts_on_its_evidence() {
        // A single token, so its front pause is blank_before[0]. The pause is
        // 800 frames -- 16 s, inter-utterance silence, nobody's property --
        // and the start holds at the token's own first frame minus the one
        // coarticulation frame, instead of the old midpoint (499), which
        // invented 400 frames of early start out of pure silence.
        let mut starts = vec![900];
        let mut ends = vec![950];
        pad_into_silence(&mut starts, &mut ends, &[950], &[(100, 899), NONE], 50.0);
        assert_eq!(starts[0], 899, "long run: the evidence edge, not the midpoint");
    }

    #[test]
    fn the_two_regimes_meet_at_the_bound() {
        // 25 frames = exactly MAX_PAUSE_SEC at 50 fps: still a pause, still
        // the midpoint. One frame more and it is silence: the evidence edge.
        // Own start 30, run (0, 24) -> mid 12; run (0, 25) -> 29.
        let mut starts = vec![30];
        let mut ends = vec![35];
        pad_into_silence(&mut starts, &mut ends, &[35], &[(0, 24), NONE], 50.0);
        assert_eq!(starts[0], 12, "a pause at the bound is still split by midpoint");
        let mut starts = vec![30];
        let mut ends = vec![35];
        pad_into_silence(&mut starts, &mut ends, &[35], &[(0, 25), NONE], 50.0);
        assert_eq!(starts[0], 29, "one frame past the bound it is silence");
    }

    #[test]
    fn an_inverted_timeline_is_pulled_back() {
        let mut starts = vec![30, 10];
        let mut ends = vec![30, 10];
        pad_into_silence(&mut starts, &mut ends, &[30, 10], &[NONE, NONE, NONE], 50.0);
        assert_eq!(starts[1], 30, "a start before the previous start is always a bug");
    }
}
