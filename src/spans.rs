//! Aggregate character alignments into word and segment levels.
//!
//! Port of the Python reference (`omni_align/spans.py`): a *word* is a maximal
//! run of non-space, non-punctuation characters; a *segment* ends at
//! sentence-final punctuation found in the gap after a word.  All levels come
//! from the same Viterbi path — a word's span is the union of its characters'
//! spans, nothing interpolated.

use crate::viterbi::TokenAlignment;

/// Characters treated as a word boundary rather than word content.
///
/// The apostrophe is deliberately NOT here. It is word-INTERNAL in English
/// (`it's`, `i'm`, `don't`, `we're`) and the Python reference splits words on
/// whitespace, not on punctuation, so those are single words there. Listing
/// `'` as punctuation made `it's` come out as two words `it` + `s` -- on
/// Buckeye dev that turned 135 gold words into 146 predictions, and only 85 of
/// them matched a gold label, because the stray `s`/`t`/`m` fragments had
/// nothing to pair with.
///
/// It stays excluded from the WORD-BREAKING set below; a standalone quote
/// surrounded by spaces is still a separator there, because word boundaries
/// come from the whitespace run, not from this character.
pub const PUNCT: &[char] = &[
    ' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '"', '(', ')', '[', ']', '{', '}', '<',
    '>', '«', '»', '„', '“', '”', '‘', '’', '…', '、', '，', '。', '！', '？', '；', '：', '（',
    '）', '【', '】', '《', '》', '〈', '〉', '·', '～', '~', '-', '—', '–',
];

/// Characters that terminate a segment.
pub const SENTENCE_END: &[char] = &['.', '!', '?', '…', '。', '！', '？', '；', ';'];

#[derive(Debug, Clone)]
pub struct WordSpan {
    pub index: usize,
    pub text: String,
    pub start: f64,
    pub end: f64,
    /// Inclusive range of character indices covered by this word.
    pub char_start: usize,
    pub char_end: usize,
}

impl WordSpan {
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

#[derive(Debug, Clone)]
pub struct SegmentSpan {
    pub index: usize,
    pub text: String,
    pub start: f64,
    pub end: f64,
    /// Indices into the word list.
    pub words: Vec<usize>,
}

fn is_word_char(piece: &str) -> bool {
    if piece == "<star>" {
        // `<star>` is the CTC target the reference inserts between words, not a
        // character of one. It must BREAK the word: treating its pieces (`<`,
        // `s`, `t`, `a`, `r`, `>`) as word characters fused a whole utterance
        // into a single word, because none of them appear in PUNCT either.
        return false;
    }
    !piece.is_empty() && !piece.chars().any(|c| PUNCT.contains(&c))
}

pub fn build_words(tokens: &[TokenAlignment]) -> Vec<WordSpan> {
    let mut words: Vec<WordSpan> = Vec::new();
    let mut buf: Vec<&TokenAlignment> = Vec::new();
    // Punctuation that follows the word it belongs to. It is held aside rather
    // than pushed, so that it joins this word without also fusing the next one:
    // "안녕.하세요" is the word "안녕." followed by the word "하세요", not one
    // word. `fix_timestamp` already snapped these marks onto the preceding
    // token's end, so attaching one invents no time of its own.
    let mut tail: Vec<&TokenAlignment> = Vec::new();

    macro_rules! flush {
        () => {
            if !buf.is_empty() || !tail.is_empty() {
                buf.extend(tail.drain(..));
                let text: String = buf.iter().map(|t| t.piece.as_str()).collect();
                words.push(WordSpan {
                    index: words.len(),
                    text,
                    start: buf[0].start,
                    end: buf[buf.len() - 1].end,
                    char_start: buf[0].index,
                    char_end: buf[buf.len() - 1].index,
                });
                buf.clear();
            }
        };
    }

    for t in tokens {
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
        if is_word_char(&t.piece) {
            buf.push(t);
            tail.clear();
        } else if t.piece == "<star>" {
            // A star is a boundary and not text: it closes the word, and the
            // marks that trailed it go with it.
            flush!();
        } else if !buf.is_empty() {
            tail.push(t);
        } else {
            // Leading punctuation has no word to join, so it opens one.
            buf.push(t);
        }
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

pub fn build_segments(tokens: &[TokenAlignment], words: &[WordSpan]) -> Vec<SegmentSpan> {
    let mut segments: Vec<SegmentSpan> = Vec::new();
    let mut cur: Vec<&WordSpan> = Vec::new();

    macro_rules! flush {
        () => {
            if !cur.is_empty() {
                let text = cur
                    .iter()
                    .map(|w| w.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                segments.push(SegmentSpan {
                    index: segments.len(),
                    text,
                    start: cur[0].start,
                    end: cur[cur.len() - 1].end,
                    words: cur.iter().map(|w| w.index).collect(),
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
