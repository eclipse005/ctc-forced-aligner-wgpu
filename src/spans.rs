//! Aggregate character alignments into word and segment levels.
//!
//! Port of the Python reference (`omni_align/spans.py`): a *word* is a maximal
//! run of non-space, non-punctuation characters; a *segment* ends at
//! sentence-final punctuation found in the gap after a word.  All levels come
//! from the same Viterbi path — a word's span is the union of its characters'
//! spans, nothing interpolated.

use crate::viterbi::TokenAlignment;

/// Characters treated as a word boundary rather than word content.
pub const PUNCT: &[char] = &[
    ' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '<',
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
    !piece.is_empty() && !piece.chars().any(|c| PUNCT.contains(&c))
}

pub fn build_words(tokens: &[TokenAlignment]) -> Vec<WordSpan> {
    let mut words: Vec<WordSpan> = Vec::new();
    let mut buf: Vec<&TokenAlignment> = Vec::new();

    macro_rules! flush {
        () => {
            if !buf.is_empty() {
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
        if is_word_char(&t.piece) {
            buf.push(t);
        } else {
            flush!();
        }
    }
    flush!();
    words
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
        if gap.chars().any(|ch| SENTENCE_END.contains(&ch)) {
            flush!();
        }
    }
    flush!();
    segments
}
