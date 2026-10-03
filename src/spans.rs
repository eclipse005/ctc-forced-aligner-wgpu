//! Aggregate character alignments into word and segment levels.
//!
//! Port of the Python reference (`omni_align/spans.py`), which splits on
//! `text.split()`: a *word* is one whitespace-delimited word of the
//! TRANSCRIPT, punctuation included; a *segment* ends at sentence-final
//! punctuation. All levels come from the same Viterbi path — a word's span is
//! the union of its characters' spans, nothing interpolated.
//!
//! A word is cut where the transcript's next word begins and nowhere else, and
//! nothing is set aside on the way. Punctuation used to be held in a side
//! buffer on the theory that a mark belongs to the word before it, and that
//! buffer was emptied by the next letter — so `さあ、始まりました` lost its `、`,
//! and 1,116 marks vanished from a 26,552 character Japanese transcript while
//! the cue view, which has no such buffer, kept every one. The aligner exists
//! to put times on a transcript, not to edit one.

use crate::viterbi::TokenAlignment;

/// Characters that are marks rather than letters.
///
/// Word boundaries do NOT come from this list — they come from the tokenizer's
/// `word_id`, which is the transcript's own whitespace. A mark here is
/// therefore part of the word it was written in, and is never a reason to cut
/// one. What this list is for is the other question: whether a character is a
/// *sound* or a *mark*, which decides whether the karaoke renderer gives it a
/// `\k` sweep of its own.
///
/// The apostrophe is deliberately NOT here. It is word-INTERNAL in English
/// (`it's`, `i'm`, `don't`, `we're`) and the Python reference splits words on
/// whitespace, not on punctuation, so those are single words there. Listing
/// `'` as punctuation made `it's` come out as two words `it` + `s` -- on
/// Buckeye dev that turned 135 gold words into 146 predictions, and only 85 of
/// them matched a gold label, because the stray `s`/`t`/`m` fragments had
/// nothing to pair with.
pub(crate) const PUNCT: &[char] = &[
    ' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '"', '(', ')', '[', ']', '{', '}', '<',
    '>', '«', '»', '„', '“', '”', '‘', '’', '…', '、', '，', '。', '！', '？', '；', '：', '（',
    '）', '【', '】', '《', '》', '〈', '〉', '·', '～', '~', '-', '—', '–',
];

/// Characters that terminate a segment.
pub(crate) const SENTENCE_END: &[char] = &['.', '!', '?', '…', '。', '！', '？', '；', ';'];

#[derive(Debug, Clone)]
pub(crate) struct WordSpan {
    pub text: String,
    pub start: f64,
    pub end: f64,
    /// Inclusive range of character indices covered by this word.
    pub char_start: usize,
    pub char_end: usize,
    /// Whether the source had whitespace in front of this word.
    ///
    /// This is a property of the TRANSCRIPT, not of the words: `hello world`
    /// and `你好世界` are both built from a list of words, and only the first
    /// separates them. Deriving the gap by joining words with a space put one
    /// between every character of an unspaced script, which is why a 3,793
    /// character Chinese transcript came out as `我 现 在 在 天 海 酒 吧`.
    pub space_before: bool,
}

/// A sentence: a run of words, with the span its first and last character
/// cover.
#[derive(Debug, Clone)]
pub(crate) struct SegmentSpan {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Whether a run of characters belongs to a script that writes without spaces
/// between words, and so may be broken between its characters.
///
/// This is the ONE place the question is asked. The tokenizer's `word_id`
/// marks a whitespace-delimited word, which for a script that does not space
/// its words is a whole line: a 3,793-character Chinese transcript arrives as
/// a single word, and a consumer that reads `words` gets one unusable entry.
/// Deciding from the characters themselves instead of from the file turns that
/// into 3,793 single-character words, and leaves `hello` alone.
///
/// It was decided per file once and per word now, and the two copies drifted
/// the way duplicated code does -- the rendering copy had lost two of the
/// blocks the tokenizer copy still had, so a text of extension-B ideographs
/// could be called Chinese by one and not by the other. One function, one
/// answer.
pub(crate) fn splits_between_characters(run: &[&TokenAlignment]) -> bool {
    use crate::vocab::is_cjk;
    let letters = run
        .iter()
        .filter(|t| !t.piece.chars().all(|c| c.is_whitespace()))
        .count();
    letters > 0 && run.iter().filter(|t| t.piece.chars().any(is_cjk)).count() * 2 > letters
}

/// Split a run of tokens into timed units: a sound, and the marks that trail it.
///
/// A mark is in the vocabulary and a real CTC target, but `fix_timestamp` snaps
/// it onto the preceding token's end, so it occupies no frames: `start == end`.
/// As a row of its own it is a point in time with a number attached, which is
/// not a timing. Measured on a 73-minute Japanese transcript, 2,490 of its
/// 26,355 rows were exactly that -- 9.5% of the output saying nothing. Folded
/// into the sound it trails, the row keeps the text and gains a duration it
/// actually has, and the mark is still printed.
///
/// The one rule: a mark joins the unit in front of it only when the transcript
/// wrote them in the SAME word. `ね。 (笑い声)` has a space before the bracket,
/// so that bracket opens a word of its own and must not be glued to `。`.
pub(crate) fn mark_groups(run: &[&TokenAlignment]) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    // the word the current group belongs to, which is the one that opened it
    let mut word = usize::MAX;
    for (i, t) in run.iter().enumerate() {
        if !out.is_empty() && is_mark(&t.piece) && word == t.word_id {
            if let Some(g) = out.last_mut() {
                g.end = i + 1;
            }
        } else {
            out.push(i..i + 1);
        }
        word = t.word_id;
    }
    out
}

/// Whether a piece is a mark rather than something with a sound.
pub(crate) fn is_mark(piece: &str) -> bool {
    !piece.is_empty() && piece.chars().all(|c| PUNCT.contains(&c))
}

pub(crate) fn build_words(tokens: &[TokenAlignment]) -> Vec<WordSpan> {
    let mut words: Vec<WordSpan> = Vec::new();
    let mut buf: Vec<&TokenAlignment> = Vec::new();

    macro_rules! flush {
        () => {
            if !buf.is_empty() {
                // A word of an unspaced script is reported per character -- the
                // same unit the line-breaker cuts on, so `words` and `cues` can
                // never disagree about what a unit is -- except that a mark
                // rides in the character it follows, because it has no time of
                // its own to be a unit.
                let space_before = !words.is_empty();
                if splits_between_characters(&buf) {
                    for (n, g) in mark_groups(&buf).iter().enumerate() {
                        let span = &buf[g.clone()];
                        words.push(WordSpan {
                            text: span.iter().map(|t| t.piece.as_str()).collect(),
                            start: span[0].start,
                            end: span[span.len() - 1].end,
                            char_start: span[0].index,
                            char_end: span[span.len() - 1].index,
                            // only the first unit of the word carries the
                            // gap; the rest follow it directly
                            space_before: space_before && n == 0,
                        });
                    }
                } else {
                    let text: String = buf.iter().map(|t| t.piece.as_str()).collect();
                    words.push(WordSpan {
                        text,
                        start: buf[0].start,
                        end: buf[buf.len() - 1].end,
                        char_start: buf[0].index,
                        char_end: buf[buf.len() - 1].index,
                        space_before,
                    });
                }
                buf.clear();
            }
        };
    }

    for t in tokens {
        // A star is not text and not a boundary. It has to be skipped before
        // the boundary test rather than after it, because the id it carries is
        // the id of the word BEFORE it -- the tokenizer emits a word's opening
        // star while still counting that word. So a word whose first character
        // had no target gets its star in the middle of the word, the ids around
        // it run backwards, and a test that let the star vote split `贅沢` in
        // two and printed a space the transcript does not have.
        if t.piece == "<star>" {
            continue;
        }
        // The word boundary comes from `word_id`, not from the piece. The
        // target sequence is every word's letters laid end to end with nothing
        // between the words, so a piece test cannot see where one word stopped:
        // "andhesat" looks like one long run and the whole utterance came out as
        // a single word. A space token would have carried the boundary, but it
        // would also be a CTC target the reference never had, and it lengthens
        // the sequence past what the frame count can carry.
        if t.word_id != buf.first().map(|b| b.word_id).unwrap_or(t.word_id) {
            flush!();
        }
        // Everything else, punctuation included, belongs to the word it was
        // written in. Holding marks in a side buffer until the next flush was
        // meant to stop a mark from fusing the following word, but the word is
        // already cut by `word_id` above, so the buffer bought nothing and cost
        // the mark: emptying it on the next letter dropped every `、` that was
        // not the last character of its word. Measured on a 26,552 character
        // Japanese transcript: 1,116 marks gone from `words` and `segments`,
        // and none from `cues`, which has no such buffer. `fix_timestamp` has
        // already snapped these marks onto the preceding token's end, so
        // attaching one invents no time of its own.
        buf.push(t);
    }
    flush!();
    words
}

/// Whether `word` carries a sentence boundary.
///
/// The mark now belongs to the word it follows rather than sitting in the gap
/// after it, so the boundary is in the word's own tail. A decimal point is in
/// that tail too and is not a boundary.
fn ends_sentence(word: &str) -> bool {
    let mut rev = word.chars().rev();
    let Some(last) = rev.next() else { return false };
    if !SENTENCE_END.contains(&last) {
        return false;
    }
    !rev.next().is_some_and(|c| c.is_ascii_digit())
}

pub(crate) fn build_segments(tokens: &[TokenAlignment], words: &[WordSpan]) -> Vec<SegmentSpan> {
    let mut segments: Vec<SegmentSpan> = Vec::new();
    let mut cur: Vec<&WordSpan> = Vec::new();

    macro_rules! flush {
        () => {
            if !cur.is_empty() {
                // The gap between words is the transcript's, not the format's:
                // an unspaced script has none, and inventing one there put a
                // space between every character of `え確かに。`
                let mut text = String::new();
                for (n, w) in cur.iter().enumerate() {
                    // The first word of a segment carries no leading gap: the
                    // segment boundary already separates it from the last one.
                    if w.space_before && n > 0 {
                        text.push(' ');
                    }
                    text.push_str(&w.text);
                }
                segments.push(SegmentSpan {
                    text,
                    start: cur[0].start,
                    end: cur[cur.len() - 1].end,
                });
                cur.clear();
            }
        };
    }

    for (i, w) in words.iter().enumerate() {
        cur.push(w);
        // characters strictly between this word and the next one
        let gap_end = words.get(i + 1).map(|n| n.char_start).unwrap_or(tokens.len());
        let gap: String = tokens[w.char_end + 1..gap_end.min(tokens.len())]
            .iter()
            .map(|t| t.piece.as_str())
            .collect();
        if ends_sentence(&w.text) || gap.chars().any(|ch| SENTENCE_END.contains(&ch)) {
            flush!();
        }
    }
    flush!();
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token, two frames, at a position derived from its index.
    fn tok(index: usize, piece: &str, word_id: usize) -> TokenAlignment {
        let f = 10 + 10 * index as i64;
        TokenAlignment {
            index,
            token_id: 0,
            piece: piece.to_string(),
            word_id,
            start: f as f64 / 50.0,
            end: (f + 2) as f64 / 50.0,
            start_frame: f,
            end_frame: f + 2,
            score: -0.2,
            inferred: false,
        }
    }

    /// A transcript as the tokenizer hands it over: one group per source word,
    /// each opened by the `<star>` that separates it from the one before.
    fn tokens(groups: &[&[&str]]) -> Vec<TokenAlignment> {
        let mut out = Vec::new();
        for (w, g) in groups.iter().enumerate() {
            for piece in *g {
                out.push(tok(out.len(), piece, w + 1));
            }
        }
        out
    }

    /// The words as a reader sees them: the transcript's own gaps, and only
    /// those.
    fn rendered(words: &[WordSpan]) -> String {
        let mut s = String::new();
        for (i, w) in words.iter().enumerate() {
            if i > 0 && w.space_before {
                s.push(' ');
            }
            s.push_str(&w.text);
        }
        s
    }

    #[test]
    fn a_mark_inside_a_word_stays_in_it() {
        // さあ、始まりました is ONE source word -- there is no whitespace in it
        // anywhere -- so the mark is a character of that word. Holding marks in
        // a side buffer and emptying that buffer on the next letter dropped it,
        // and 1,116 marks with it across a 26,552 character transcript.
        let t = tokens(&[&["さ", "あ", "、", "始", "ま", "り", "ま", "し", "た", "。"]]);
        assert_eq!(rendered(&build_words(&t)), "さあ、始まりました。");
    }

    #[test]
    fn an_unspaced_script_renders_tight() {
        // Two sentences with a real space between them. The gap is the
        // transcript's; a gap inside either sentence is not, and inventing one
        // per character is what shipped `私 も 思 っ て た 。` to a subscriber.
        let t = tokens(&[&["私", "も", "思", "っ", "て", "た", "。"], &["え", "確", "か", "に", "。"]]);
        let words = build_words(&t);
        assert_eq!(rendered(&words), "私も思ってた。 え確かに。");
        let segments = build_segments(&t, &words);
        let seg: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(seg, ["私も思ってた。", "え確かに。"]);
    }

    #[test]
    fn a_mark_rides_in_the_unit_it_follows() {
        // One source word, marks in the minority the way real text has them:
        // は、 す。 ね こ い。 あ -- the marks have no frames of their own
        // (`fix_timestamp` snapped each onto the preceding end), so as rows of
        // their own they are points in time, not timings.
        let t = tokens(&[&["は", "、", "す", "。", "ね", "こ", "い", "。", "あ"]]);
        let words = build_words(&t);
        let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(text, ["は、", "す。", "ね", "こ", "い。", "あ"], "a mark is not a unit");
        assert!(
            words.iter().all(|w| w.end > w.start),
            "every unit now has a duration: {:?}",
            words.iter().map(|w| (w.text.as_str(), w.start, w.end)).collect::<Vec<_>>()
        );
        // The unit now runs to the end of the mark it absorbed. In real
        // material that is the same instant the sound ends, because
        // `fix_timestamp` snapped the mark onto it -- here the mark has a span
        // of its own, so the difference is visible.
        assert_eq!((words[0].start, words[0].end), (0.2, 0.44));
        assert_eq!(rendered(&words), "は、す。ねこい。あ");

        // A mark the transcript put after a space opens a word of its own, or
        // `ね。 (笑い声)` would glue the bracket to the previous sentence.
        let t = tokens(&[&["ね", "こ", "そ", "。"], &["（", "笑", "い", "声", "）"]]);
        let words = build_words(&t);
        let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(&text[..3], ["ね", "こ", "そ。"], "the sentence closed on its mark");
        assert_eq!(text[3], "（", "the bracket opened its own word");
    }

    #[test]
    fn a_latin_word_keeps_its_comma_and_the_gap_after_it() {
        let t = tokens(&[&["h", "e", "l", "l", "o", ","], &["w", "o", "r", "l", "d", "!"]]);
        let words = build_words(&t);
        let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(text, ["hello,", "world!"]);
        assert!(!words[0].space_before, "the first word has nothing before it");
        assert!(words[1].space_before, "but the source had a space there");
        let segments = build_segments(&t, &words);
        assert_eq!(segments[0].text, "hello, world!");
    }
}
