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

/// Anchor word-boundary marks to the previous timed token's end, in place.
///
/// The CTC path gives every vocabulary character a frame. Punctuation and
/// spaces have no phone, so the path parks them at the end of the following
/// pause; this moves only those marks, and only backward onto the token
/// already placed before them. A token that contains anything other than a
/// mark keeps its frames, so word spans built from those tokens do not move.
/// Nothing is attached to a later token.
///
/// Anchored marks are points: `start == end` equals the previous token's end,
/// and the frame fields name its end frame. Applying this twice changes
/// nothing, which is what lets it sit in a pipeline rather than being a
/// correction somebody has to remember to apply.
pub fn anchor_marks(tokens: &mut [TokenAlignment]) {
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
