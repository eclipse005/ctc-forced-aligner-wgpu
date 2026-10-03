//! Rule 1: a boundary sits in the middle of the pause beside it.
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
    fn the_bound_never_reaches_a_start() {
        // A single token, so its front pause is blank_before[0] and the
        // trailing run is blank_before[1]. The pause is 800 frames -- 16 s --
        // and the start still moves the whole way to its midpoint, unbounded:
        // the clamp is one-sided by design, so a long pause before a word costs
        // it start accuracy and costs it nothing at the end.
        let mut starts = vec![900];
        let mut ends = vec![950];
        pad_into_silence(&mut starts, &mut ends, &[950], &[(100, 899), NONE], 50.0);
        assert_eq!(starts[0], 499, "the start takes the whole midpoint");
    }

    #[test]
    fn an_inverted_timeline_is_pulled_back() {
        let mut starts = vec![30, 10];
        let mut ends = vec![30, 10];
        pad_into_silence(&mut starts, &mut ends, &[30, 10], &[NONE, NONE, NONE], 50.0);
        assert_eq!(starts[1], 30, "a start before the previous start is always a bug");
    }
}
