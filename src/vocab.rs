//! Character vocabulary of the omniASR-CTC checkpoint (vocab.json).
//!
//! blank = id 0 (`<s>`), unk = `<unk>`; tokenisation is per character and
//! skips out-of-vocabulary characters, mirroring the Python `_tokenise`, so
//! every timestamp maps to one visible character.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Scripts written without spaces between words: Chinese (Han) and Japanese
/// (Kana). Those stay character-level units. Hangul is *not* included — Korean
/// is whitespace-delimited like Latin (eojeol), so a transcript space keeps
/// `이번` as one unit rather than splitting it into `이` `번`.
///
/// Deliberately narrower than "not ASCII": a transcript mixing Latin and Han
/// ("hello 你好") still needs the unspaced-script rule for the Han run.
pub(crate) fn is_cjk(c: char) -> bool {
    let n = c as u32;
    (0x3040..=0x30FF).contains(&n)      // kana
        || (0x3400..=0x4DBF).contains(&n)  // CJK ext A
        || (0x4E00..=0x9FFF).contains(&n)  // CJK
        || (0xF900..=0xFAFF).contains(&n)  // compatibility
        || (0x20000..=0x2A6DF).contains(&n) // CJK ext B
}

pub(crate) struct Vocab {
    pub char_to_id: HashMap<char, usize>,
    pub unk_id: usize,
    pub size: usize,
    /// The reference's synthetic `<star>` id, `len(dictionary)`, which is one
    /// past the last real id. `<star>` is NOT in vocab.json -- the reference
    /// adds it to its own dictionary after loading -- so it cannot come from
    /// `char_to_id`. It is a real target in the CTC sequence (see
    /// [`tokenise_with_stars`]) and therefore occupies frames.
    pub star_id: usize,
}

impl Vocab {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("vocab.json")).context("read vocab.json")?,
        )
        .context("parse vocab.json")?;
        let mut char_to_id = HashMap::new();
        let mut size = 0usize;
        let mut unk_id = 3usize;
        for (tok, id) in raw.as_object().context("vocab.json object")? {
            let id = id.as_u64().context("vocab id")? as usize;
            size = size.max(id + 1);
            if tok == "<unk>" {
                unk_id = id;
            }
            let mut chars = tok.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                char_to_id.insert(c, id);
            }
        }
        // The reference builds `dictionary = {k.lower(): v for k, v in vocab}`
        // and then appends `<star>`, so the synthetic id is the dictionary
        // SIZE -- one past the largest real id, which `size` already is.
        let star_id = size;
        Ok(Self { char_to_id, unk_id, size, star_id })
    }

#[cfg(test)]
    pub fn tokenise(&self, text: &str) -> (Vec<usize>, Vec<String>) {
        let mut ids = Vec::new();
        let mut pieces = Vec::new();
        for c in text.chars() {
            if let Some(&id) = self.char_to_id.get(&c) {
                if id == self.unk_id {
                    continue;
                }
                ids.push(id);
                pieces.push(c.to_string());
            }
        }
        (ids, pieces)
    }

    /// [`tokenise`], with one `<star>` in front of every word.
    ///
    /// This is not cosmetic. `<star>` is a real entry of the CTC target
    /// sequence, so the DP has to place it and it consumes frames: on
    /// "every time i fan myself it goes up" the reference's per-frame path
    /// opens with `[10022, 0, 0, 0, 2802, ...]`, frame 0 being the star, and
    /// its first blank run is `(1, 3)` where a starless path's is `(0, 3)`.
    /// Those shifted runs are what the word boundary is padded into, which is
    /// why every boundary landed one to three frames away from the
    /// reference's without any single rule looking wrong.
    ///
    /// The reference offers a second placement, `edges`, which puts one star
    /// at each end of the whole file and none in between, and that is its
    /// DEFAULT. It is not used here. A star is a DP anchor, and the DP on this
    /// checkpoint is under-constrained wherever the acoustic evidence is weak:
    /// with two stars the path can slide a whole phrase, and with one per word
    /// it cannot.
    ///
    /// Over the 180-clip multilingual set, 124 of whose clips FireRedVAD and a
    /// short-time-energy detector agree about (VAD alone misses most of the
    /// speech in some FLEURS English clips, and believing it inverts the
    /// result), the per-frame path score:
    ///
    /// ```text
    ///     Spanish   -0.1308 vs -0.3896
    ///     French    -0.2605 vs -0.5315
    ///     German    -0.3322 vs -0.5347
    ///     English   -0.5936 vs -0.7137
    ///     Japanese  -0.2348 vs -0.2346
    ///     Chinese   -0.1993 vs -0.1987
    ///     all 124   -0.2570 vs -0.4301      (one star per word wins 91)
    /// ```
    ///
    /// The gap is exactly where the mechanism says it should be: large on
    /// scripts that space their words, nil on the ones that do not, where a
    /// "word" is a whole line and both placements put about the same number of
    /// stars anyway. On material with reference timings the same ordering
    /// holds -- Korean broadcast word-start MAE 277.5 ms vs 442.0 ms, and on
    /// the sung Chinese of the same file 82.8 ms vs 795.5 ms.
    ///
    /// The whole-file log-prob cannot compare the two: it sums over the path,
    /// and two stars is fewer terms than one per word. The per-frame mean
    /// divides by the same frame count either way, which is why it is the
    /// number quoted.
    ///
    /// The stars are an ANCHOR, not a word marker. Word boundaries come from
    /// `word_id`, which the tokenizer assigns per whitespace-delimited word and
    /// which is correct for every script. Reading boundaries off the stars
    /// instead put the whole English transcript into one cue.
    /// The source word index of every target, plus the index in `text` of the
    /// character each one came from.
    ///
    /// The character index is what lets a character the vocabulary has no id
    /// for be put BACK into the output afterwards. The aligner exists to time
    /// text, not to edit it: dropping an unknown character silently rewrites
    /// the transcript, and the subtitle ends up saying something the speaker
    /// did not. See [`crate::align_inference::fill_unaligned_characters`].
    ///
    /// A `<star>` has no source character and is recorded as [`usize::MAX`].
    pub fn tokenise_with_word_ids(
        &self,
        text: &str,
    ) -> (Vec<usize>, Vec<String>, Vec<usize>, Vec<usize>) {
        let mut ids: Vec<usize> = Vec::new();
        let mut pieces: Vec<String> = Vec::new();
        let mut word_ids: Vec<usize> = Vec::new();
        let mut src: Vec<usize> = Vec::new();
        let star = self.star_id;
        let push_star = |ids: &mut Vec<usize>, pieces: &mut Vec<String>,
                             word_ids: &mut Vec<usize>, src: &mut Vec<usize>,
                             w: usize, star: usize| {
            ids.push(star);
            pieces.push("<star>".to_string());
            word_ids.push(w);
            src.push(usize::MAX);
        };
        // a leading star belongs to no word; use a sentinel nothing else takes
        push_star(&mut ids, &mut pieces, &mut word_ids, &mut src, usize::MAX, star);
        let mut wi = 0usize;
        // Words are the maximal runs of non-whitespace. Each character is
        // matched against the vocabulary here, so a word the vocabulary dropped
        // entirely contributes nothing -- and the word counter must skip it, or
        // the ids would not be consecutive and the boundary would fall in the
        // wrong place. Its characters are recovered afterwards from `text`.
        let mut word: Vec<(usize, char)> = Vec::new();
        let take = |word: &mut Vec<(usize, char)>,
                        wi: &mut usize,
                        ids: &mut Vec<usize>,
                        pieces: &mut Vec<String>,
                        word_ids: &mut Vec<usize>,
                        src: &mut Vec<usize>| {
            let letters: Vec<(usize, String, usize)> = word
                .iter()
                .filter_map(|(at, c)| match self.char_to_id.get(c) {
                    Some(&id) if id != self.unk_id => Some((id, c.to_string(), *at)),
                    _ => None,
                })
                .collect();
            if letters.is_empty() {
                return;
            }
            if *wi > 0 {
                push_star(ids, pieces, word_ids, src, *wi, star);
            }
            *wi += 1;
            for (id, c, at) in letters {
                ids.push(id);
                pieces.push(c);
                word_ids.push(*wi);
                src.push(at);
            }
        };
        for (at, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if !word.is_empty() {
                    take(&mut word, &mut wi, &mut ids, &mut pieces, &mut word_ids, &mut src);
                    word.clear();
                }
            } else {
                word.push((at, ch));
            }
        }
        if !word.is_empty() {
            take(&mut word, &mut wi, &mut ids, &mut pieces, &mut word_ids, &mut src);
        }
        (ids, pieces, word_ids, src)
    }

#[cfg(test)]
    pub fn tokenise_with_stars(&self, text: &str) -> (Vec<usize>, Vec<String>) {

        // Words are split on whitespace and their letters concatenated, with a
        // `<star>` at each end (`edges`) or between words (`segment`).
        //
        // Two things that look like they belong here and do not:
        //
        //   * a space token between WORDS. It reads as the obvious word
        //     boundary, but the reference's flat target list has the letters
        //     and the stars adjacent, and adding one lengthens the sequence
        //     past what the frame count can carry — the DP then compresses the
        //     path and every boundary drifts earlier, cumulatively.
        //   * a space token between LETTERS of a word. `preprocess_text` writes
        //     `" ".join(list(word))` per word, but `get_alignments` flattens it
        //     again and keeps only dictionary entries, and the space is not one
        //     (`if c in dictionary`). It never becomes a target.
        //
        // What DOES separate words is the two stars plus `build_words`' own
        // word-character test, and what pads the boundaries is the blank run
        // around each letter — see [`crate::viterbi::collapse`].
        let words: Vec<Vec<(usize, String)>> = text
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter_map(|c| {
                        let id = *self.char_to_id.get(&c)?;
                        if id == self.unk_id {
                            None
                        } else {
                            Some((id, c.to_string()))
                        }
                    })
                    .collect::<Vec<(usize, String)>>()
            })
            .filter(|w: &Vec<(usize, String)>| !w.is_empty())
            .collect();

        let mut ids: Vec<usize> = Vec::new();
        let mut pieces: Vec<String> = Vec::new();
        let star = self.star_id;
        let push_star = |ids: &mut Vec<usize>, pieces: &mut Vec<String>| {
            ids.push(star);
            pieces.push("<star>".to_string());
        };

        push_star(&mut ids, &mut pieces);
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                push_star(&mut ids, &mut pieces);
            }
            // The reference's `preprocess_text` builds `" ".join(list(word))`
            // per word, but `get_alignments` flattens that on
            // `" ".join(tokens).split(" ")` and then keeps only entries the
            // dictionary knows -- and the space is NOT one of them:
            // `token_indices = [dictionary[c] for c in ... if c in dictionary]`.
            // The letters are adjacent in the target sequence; the space only
            // ever existed to be split away again.
            //
            // Emitting a space token between letters is therefore wrong twice
            // over: it adds a target the reference never had, and it pushes the
            // sequence past what the frame count can carry. On a 118-frame clip
            // the DP then has to compress the path, and every boundary drifts
            // earlier cumulatively -- up to 60 frames by the last word.
            for (id, c) in w {
                ids.push(*id);
                pieces.push(c.clone());
            }
        }
        (ids, pieces)
    }
}
