//! Rule 2: a mark is a point at the end of the sound before it.

use crate::spans::PUNCT;
use crate::viterbi::TokenAlignment;

/// Whether a character has a sound of its own.
///
/// False for every mark and for the spaces the tokenizer drops, which is what
/// makes this the test for "is there anything here to be timed by".
fn owns_time(piece: &str) -> bool {
    // `<star>` holds no time of its own — it is the reference's marker between
    // words, and it must not become the anchor that a following space is
    // snapped back to, or the mark lands on the star instead of on the speech
    // that precedes it.
    if piece == "<star>" {
        return false;
    }
    piece.chars().any(|ch| !PUNCT.contains(&ch))
}

/// Anchor word-boundary marks to a timed token, in place.
///
/// The CTC path gives every vocabulary character a frame. Punctuation and
/// spaces have no phone, so the path parks them at the end of the following
/// pause; a mark with a timed token before it moves backward onto that
/// token's end — a mark with nothing timed before it moves forward onto the
/// next timed token's start, the mirror case. A token that contains anything
/// other than a mark keeps its frames, so word spans built from those tokens
/// do not move. The two passes write disjoint prefixes, and a mark is always
/// a point: `start == end` equals the anchored token's boundary, and the
/// frame fields name it. Applying this twice changes nothing, which is what
/// lets it sit in a pipeline rather than being a correction somebody has to
/// remember to apply.
pub(crate) fn anchor_marks(tokens: &mut [TokenAlignment]) {
    let mut anchor: Option<(f64, i64)> = None;
    for token in tokens.iter_mut() {
        if owns_time(&token.piece) {
            anchor = Some((token.end, token.end_frame));
            continue;
        }
        let Some((end, frame)) = anchor else {
            continue;
        };
        token.start = end;
        token.end = end;
        token.start_frame = frame;
        token.end_frame = frame;
    }
    // LEADING MARKS. The forward pass above cannot anchor a mark that has no
    // timed token before it — `<star>` does not own time, so a quotation mark
    // opening the stream kept whatever frames the path left near it, and on
    // material that opens into silence that span is pure fiction: measured on
    // a Mandarin FLEURS clip, an opening `“` held frames 90–146 (1.1 s) over
    // a blank run it had no sound in, dragging its host word's cue a second
    // early with it. A mark with no sound before it belongs to the sound that
    // follows: a point at the next timed token's start, the mirror of the
    // forward rule. The prefix before the first timed token is exactly the
    // set of marks the forward pass had to skip, so the two passes never
    // touch the same token twice.
    let Some(first_timed) = tokens.iter().position(|t| owns_time(&t.piece)) else {
        return;
    };
    let (start, frame) = (tokens[first_timed].start, tokens[first_timed].start_frame);
    for token in tokens[..first_timed].iter_mut() {
        token.start = start;
        token.end = start;
        token.start_frame = frame;
        token.end_frame = frame;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(piece: &str, end: f64) -> TokenAlignment {
        TokenAlignment {
            index: 0,
            token_id: 7,
            piece: piece.to_string(),
            word_id: 1,
            start: end - 0.2,
            end,
            start_frame: (end * 50.0) as i64,
            end_frame: (end * 50.0) as i64,
            score: -0.1,
            inferred: false,
        }
    }

    #[test]
    fn a_mark_lands_on_the_sound_before_it() {
        let mut t = vec![tok("あ", 0.2), tok("、", 0.6), tok("い", 0.8)];
        anchor_marks(&mut t);
        // `end - 0.2` is not exactly 0.0 in f64, so compare with a tolerance
        let near = |a: f64, b: f64| assert!((a - b).abs() < 1e-9, "{a} != {b}");
        near(t[0].start, 0.0);
        near(t[0].end, 0.2);
        assert_eq!((t[1].start, t[1].end), (0.2, 0.2), "the mark is a point there");
        assert_eq!((t[1].start_frame, t[1].end_frame), (10, 10), "and names that frame");
        near(t[2].start, 0.6);
        near(t[2].end, 0.8);
    }

    #[test]
    fn a_star_is_never_the_anchor() {
        // The path parks a space after the star; snapping it to the star would
        // put the mark on a target rather than on the speech.
        let mut t = vec![tok("あ", 0.2), tok("<star>", 0.2), tok(" ", 0.6), tok("い", 0.8)];
        anchor_marks(&mut t);
        assert_eq!((t[2].start, t[2].end), (0.2, 0.2), "onto the speech, not the star");
    }

    #[test]
    fn a_leading_mark_belongs_to_the_sound_that_follows() {
        // An opening quotation mark has no timed token before it (the star
        // holds no time), so the forward pass skips it and the path's fiction
        // would survive. It becomes a point at the first sound's start --
        // the mirror of the trailing rule.
        let mut t = vec![tok("“", 1.4), tok("<star>", 1.4), tok("它", 1.8), tok("们", 2.0)];
        anchor_marks(&mut t);
        assert_eq!((t[0].start, t[0].end), (1.6, 1.6), "a point at the first sound");
        assert_eq!((t[1].start, t[1].end), (1.6, 1.6), "the star too");
        assert!((t[2].start - 1.6).abs() < 1e-9, "the sound itself does not move");
        assert!((t[2].end - 1.8).abs() < 1e-9);
    }

    #[test]
    fn running_it_twice_changes_nothing() {
        let once = {
            let mut t = vec![tok("a", 0.2), tok(",", 0.6), tok("b", 0.8)];
            anchor_marks(&mut t);
            t
        };
        let mut twice = once.clone();
        anchor_marks(&mut twice);
        for (a, b) in once.iter().zip(&twice) {
            assert_eq!((a.start, a.end, a.start_frame, a.end_frame),
                       (b.start, b.end, b.start_frame, b.end_frame));
        }
    }
}
