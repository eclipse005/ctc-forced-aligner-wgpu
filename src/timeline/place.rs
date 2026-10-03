//! Rule 3: a character with no target lies between its placed neighbours.

use crate::viterbi::TokenAlignment;
use std::collections::HashMap;

/// Put back the transcript characters the vocabulary had no target for.
///
/// The aligner exists to put times on a transcript, not to edit one. A
/// character outside the checkpoint's vocabulary has no CTC target, so it never
/// reaches the DP -- and if it is simply left out, the subtitle goes on saying
/// something the speaker did not. A 3,793-character Chinese transcript that
/// mentions 悍 four times would ship a subtitle with the word mangled, and the
/// only trace would be a flag on a character nobody was looking for.
///
/// The span is not a guess dressed up as a measurement. Forced alignment is a
/// monotone path, so a character sitting between two placed ones MUST fall
/// between them; the midpoint is the only value available without more
/// evidence, and it satisfies that constraint exactly. [`TokenAlignment::inferred`]
/// records that nothing was measured, so a consumer can tell the two apart.
///
/// It runs after [`super::anchor_marks`] and before anything reads the spans,
/// so the only thing that can own time here is speech.
pub fn place_unmeasured(
    tokens: &[TokenAlignment],
    text: &str,
    src: &[usize],
    frame_rate: f64,
    duration: f64,
) -> Vec<TokenAlignment> {
    let star = |t: &TokenAlignment| t.piece == "<star>";
    let mut placed: HashMap<usize, usize> = HashMap::with_capacity(src.len());
    let mut t = 0usize;
    for &at in src {
        while t < tokens.len() && star(&tokens[t]) {
            t += 1;
        }
        if at != usize::MAX && t < tokens.len() {
            placed.entry(at).or_insert(t);
            t += 1;
        }
    }

    let mut out: Vec<TokenAlignment> = Vec::with_capacity(tokens.len() + 8);
    // the last placed token, and the next one, as indices into `tokens`
    let mut prev: Option<&TokenAlignment> = None;
    let mut next = 0usize;
    // unplaced characters waiting for a right-hand bound
    let mut gap: Vec<(char, bool)> = Vec::new();  // (character, starts a word)
    // Ids for the words the vocabulary dropped whole, which have to be ones no
    // real word holds. The leading star's `usize::MAX` is a sentinel saying
    // "belongs to no word", not an id, and taking the maximum over it wrapped
    // the counter to 0 in a release build -- so a dropped word was handed the
    // first real word's id and fused with it, and a debug build panicked on the
    // overflow instead of showing it.
    let mut next_synthetic_word = tokens
        .iter()
        .filter(|t| t.word_id != usize::MAX)
        .map(|t| t.word_id)
        .max()
        .unwrap_or(0)
        + 1;
    let mut after_space = true;

    let mut flush = |gap: &mut Vec<(char, bool)>, prev: Option<&TokenAlignment>,
                     upto: Option<&TokenAlignment>, out: &mut Vec<TokenAlignment>| {
        if gap.is_empty() {
            return;
        }
        // The monotone constraint: everything in the gap lies between prev's
        // end and the next placed character's start.
        //
        // That upper bound is the next character's START, never prev's end.
        // The midpoint rule pads a token's end forward to the middle of the
        // pause, so a neighbour can start BEFORE that padded end -- measured:
        // `拉` ends at frame 3575 and `就` starts at 3575, with a two-character
        // run between them. Splitting [next.start, prev.end] between those two
        // put the second one's start a frame past `就`, and the sequence ran
        // backwards. Capping at the next onset costs a degenerate span in that
        // case and nothing at all when there is room.
        let (lo, hi) = match (prev, upto) {
            (Some(p), Some(n)) => (p.end.min(n.start), n.start),
            (Some(p), None) => (p.end, duration),
            (None, Some(n)) => (0.0, n.start),
            (None, None) => (0.0, duration),
        };
        let n = gap.len() as f64;
        let run: Vec<(char, bool)> = std::mem::take(gap);
        // The word a character belongs to is the word it was WRITTEN in, and the
        // only way to know that is to look at what closes it. A run that ends at
        // a placed character with no whitespace in between is inside that
        // character's word and takes its id -- `贅沢` is one word even though
        // `贅` had no target and `沢` did, and giving `贅` an id of its own
        // split the word in two and printed a space the transcript does not
        // have. A run the source closed with whitespace is a word the
        // vocabulary dropped whole, and it keeps an id of its own so that it
        // does not fuse with the word after it.
        let mut word = prev.map(|p| p.word_id);
        for (i, (ch, starts_word)) in run.iter().enumerate() {
            if *starts_word {
                let closed_by_next = upto.is_some() && !run[i + 1..].iter().any(|(_, s)| *s);
                word = if closed_by_next {
                    upto.map(|n| n.word_id)
                } else {
                    let id = next_synthetic_word;
                    next_synthetic_word += 1;
                    Some(id)
                };
            }
            // split the available interval evenly across the run
            let a = lo + (hi - lo) * (i as f64 / n);
            let b = lo + (hi - lo) * ((i as f64 + 1.0) / n);
            let frame = (a * frame_rate).round() as i64;
            out.push(TokenAlignment {
                index: out.len(),
                token_id: usize::MAX,
                piece: ch.to_string(),
                word_id: word.unwrap_or(next_synthetic_word),
                start: a,
                end: b,
                start_frame: frame,
                end_frame: frame,
                score: 0.0,
                inferred: true,
            });
        }
    };

    for (at, ch) in text.char_indices() {
        if ch.is_whitespace() {
            after_space = true;
            continue;
        }
        let starts_word = after_space;
        after_space = false;
        match placed.get(&at).copied() {
            Some(ti) => {
                while next < ti {
                    out.push(tokens[next].clone());
                    next += 1;
                }
                flush(&mut gap, prev, Some(&tokens[ti]), &mut out);
                out.push(tokens[ti].clone());
                prev = Some(&tokens[ti]);
                next = ti + 1;
            }
            None => gap.push((ch, starts_word)),
        }
    }
    while next < tokens.len() {
        out.push(tokens[next].clone());
        next += 1;
    }
    flush(&mut gap, prev, None, &mut out);
    for (i, tok) in out.iter_mut().enumerate() {
        tok.index = i;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A token with a real span, placed.
    fn tok(piece: &str, word_id: usize, start: f64, end: f64) -> TokenAlignment {
        TokenAlignment {
            index: 0,
            token_id: 7,
            piece: piece.to_string(),
            word_id,
            start,
            end,
            start_frame: (start * 50.0).round() as i64,
            end_frame: (end * 50.0).round() as i64,
            score: -0.1,
            inferred: false,
        }
    }

    #[test]
    fn a_missing_leading_character_joins_the_word_it_was_written_in() {
        // 。 贅沢。 -- `贅` is outside the vocabulary, `沢` is not, and they are
        // ONE source word. An id of its own for `贅` split the word in two and
        // printed a space the transcript does not contain.
        let text = "。 贅沢。";
        let at: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        // at == [0, 3, 4, 7, 10] for 。, the space, 贅, 沢, 。
        let toks = vec![
            tok("<star>", usize::MAX, 0.0, 0.0),
            tok("。", 1, 0.0, 0.2),
            tok("<star>", 1, 0.2, 0.2),
            tok("沢", 2, 0.2, 0.6),
            tok("。", 2, 0.6, 0.8),
        ];
        let src = vec![usize::MAX, at[0], usize::MAX, at[3], at[4]];
        let out = place_unmeasured(&toks, text, &src, 50.0, 2.0);
        let got: String = out.iter().map(|t| t.piece.as_str()).collect();
        assert_eq!(got, "<star>。<star>贅沢。", "every character comes back once");
        let zei = out.iter().position(|t| t.piece == "贅").unwrap();
        let sawa = out.iter().position(|t| t.piece == "沢").unwrap();
        assert_eq!(out[zei].word_id, out[sawa].word_id, "one word, not two");
        assert!(out[zei].inferred, "and nothing was measured for it");
    }

    #[test]
    fn a_wholly_dropped_word_does_not_fuse_with_the_next_one() {
        // `好 野` -- only 好 has a target, so 野 is a whole dropped word and has
        // to keep an id of its own or the two render glued together.
        let text = "好 野";
        let at: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        let toks = vec![tok("好", 1, 0.0, 0.2)];
        let src = vec![at[0]];
        let out = place_unmeasured(&toks, text, &src, 50.0, 2.0);
        let got: String = out.iter().map(|t| t.piece.as_str()).collect();
        assert_eq!(got, "好野");
        assert_ne!(out[1].word_id, out[0].word_id, "the missing word joined its neighbour");
    }

    #[test]
    fn the_run_sits_inside_the_pause_its_neighbours_leave() {
        // `拉蔻就` -- 蔻 is missing, and it must land between 拉's end and
        // 就's start, never outside them.
        let text = "拉蔻就";
        let at: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        let toks = vec![tok("拉", 1, 0.0, 0.4), tok("就", 1, 0.8, 1.0)];
        let src = vec![at[0], at[2]];
        let out = place_unmeasured(&toks, text, &src, 50.0, 2.0);
        let got: String = out.iter().map(|t| t.piece.as_str()).collect();
        assert_eq!(got, "拉蔻就", "the transcript comes back character for character");
        assert!(out[1].inferred && !out[0].inferred && !out[2].inferred);
        assert!(out[1].start >= 0.4 && out[1].end <= 0.8, "inside the neighbours");
    }

    #[test]
    fn the_order_never_runs_backwards() {
        // A token whose end was padded forward past the next one's start --
        // the midpoint rule can do that. The interpolated character is capped
        // at the next onset rather than pushed past it.
        let text = "aXb";
        let at: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        let toks = vec![tok("a", 1, 0.0, 1.0), tok("b", 1, 1.0, 1.2)];
        let src = vec![at[0], at[2]];
        let out = place_unmeasured(&toks, text, &src, 50.0, 4.0);
        for w in out.windows(2) {
            assert!(w[1].start >= w[0].start - 1e-9,
                    "start ran backwards: {} then {}", w[0].piece, w[1].piece);
        }
        assert!(out[1].end <= out[2].start + 1e-9, "interpolated span overran");
    }
}
