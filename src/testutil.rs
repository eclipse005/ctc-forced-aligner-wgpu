//! Token builders shared by the tests of several modules.
//!
//! `views` and `render::ass` both need tokens with a given word id and frame
//! range, and duplicating them meant a fix to one copy could leave the other
//! testing the wrong thing.

#![cfg(test)]

use crate::viterbi::TokenAlignment;

/// A token whose word id is its own index.
pub fn tok(i: usize, piece: &str, sf: i64, ef: i64) -> TokenAlignment {
    tokw(i, piece, i, sf, ef)
}

/// A token that belongs to word `w`. The word id is what the tokenizer assigns
/// per whitespace-delimited word, and it is what `cue_tokens` groups on -- a
/// star is the CTC anchor, not the word boundary.
pub fn tokw(i: usize, piece: &str, w: usize, sf: i64, ef: i64) -> TokenAlignment {
    TokenAlignment {
        index: i,
        token_id: 1,
        word_id: w,
        piece: piece.to_string(),
        start: sf as f64 / 50.0,
        end: (ef + 1) as f64 / 50.0,
        start_frame: sf,
        end_frame: ef,
        score: -0.2,
        inferred: false,
    }
}

/// A run of tokens laid out as words: `[["<star>", "你", "好"], ...]`.
pub fn words_from(groups: &[&[&str]]) -> Vec<TokenAlignment> {
    let mut out = Vec::new();
    for (w, g) in groups.iter().enumerate() {
        for piece in *g {
            out.push(tokw(
                out.len(),
                piece,
                w,
                10 + 10 * out.len() as i64,
                12 + 10 * out.len() as i64,
            ));
        }
    }
    out
}
