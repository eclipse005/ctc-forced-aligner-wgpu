//! Point word-boundary marks at the previous timed token.
//!
//! The CTC path gives every vocabulary character a frame. Punctuation and
//! spaces have no phone, so the path parks them at the end of the following
//! pause. This moves only those marks, and only backward onto the token
//! already placed before them. A token that contains anything other than a
//! word boundary keeps its frames, so word spans built from those tokens do
//! not move. Nothing is attached to a later token.
//!
//! Anchored marks are points: `start == end` equals the previous token's end,
//! and the frame fields name its end frame. Applying this twice changes nothing.

use crate::spans::PUNCT;
use crate::viterbi::TokenAlignment;

/// False when every character is already a word boundary.
pub fn owns_time(piece: &str) -> bool {
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
pub fn fix_timestamp(tokens: &mut [TokenAlignment]) {
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
    use crate::viterbi::TokenAlignment;

    fn tok(piece: &str, start: f64, end: f64, frame: i64) -> TokenAlignment {
        TokenAlignment {
            index: 0,
            token_id: 0,
            word_id: 0,
            piece: piece.to_string(),
            start,
            end,
            start_frame: frame,
            end_frame: frame,
            score: 0.0,
        }
    }

    #[test]
    fn letters_stay_period_and_tilde_anchor_backward() {
        let mut tokens = vec![
            tok("t", 11.38, 11.40, 569),
            tok("h", 11.40, 11.42, 570),
            tok("e", 11.42, 11.44, 571),
            tok(".", 12.40, 12.42, 620),
            tok("啊", 183.00, 183.02, 9150),
            tok("~", 206.96, 206.98, 10348),
            tok("같", 266.48, 266.50, 13324),
            tok("아", 266.50, 266.52, 13325),
            tok("요", 266.52, 266.54, 13326),
            tok("%", 815.18, 815.20, 40759),
        ];
        fix_timestamp(&mut tokens);
        assert_eq!(tokens[2].start, 11.42);
        assert_eq!(tokens[2].end, 11.44);
        assert_eq!(tokens[2].end_frame, 571);
        assert_eq!(tokens[3].start, 11.44);
        assert_eq!(tokens[3].end, 11.44);
        assert_eq!(tokens[3].end_frame, 571);
        assert_eq!(tokens[4].start, 183.00);
        assert_eq!(tokens[4].end, 183.02);
        assert_eq!(tokens[5].start, 183.02);
        assert_eq!(tokens[5].end, 183.02);
        assert_eq!(tokens[5].end_frame, 9150);
        assert_eq!((tokens[6].start, tokens[8].end), (266.48, 266.54));
        assert_eq!((tokens[9].start, tokens[9].end), (815.18, 815.20));
        fix_timestamp(&mut tokens);
        assert_eq!(tokens[3].start, 11.44);
        assert_eq!(tokens[5].end_frame, 9150);
    }

    #[test]
    fn leading_mark_has_no_earlier_speech() {
        let mut tokens = vec![tok(".", 1.0, 1.02, 50), tok("A", 1.10, 1.12, 55)];
        fix_timestamp(&mut tokens);
        assert_eq!((tokens[0].start, tokens[0].end_frame), (1.0, 50));
        assert_eq!((tokens[1].start, tokens[1].end), (1.10, 1.12));
    }
}
