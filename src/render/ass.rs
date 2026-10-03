//! Karaoke (Advanced SubStation Alpha, `.ass`) output.
//!
//! The same cue list as the SRT, with a `\k` sweep per character. It is a
//! renderer and nothing more: where a line breaks, which characters form a
//! unit, and whether a mark carries time all come from `crate::timeline` and
//! `crate::views`, so the three cannot drift apart.
//!
//! Two things about the format are worth knowing before reading the code.
//! `\k<n>` runs for `n` centiseconds, so the durations have to be differences
//! between successive ONSETS rather than each token's own span -- the tag
//! before the first syllable is the delay until it starts, and each later one
//! runs until the next begins. And `\h` is a hard space: a real space in a
//! transcript becomes `\h` so the renderer draws it and does not break the line
//! there, which is the only way a gap survives into the picture.

use crate::spans::is_mark;
use crate::views::CueDoc;
use crate::viterbi::TokenAlignment;


/// Subtitle appearance for the karaoke output.
///
/// ASS colours are `AABBGGRR`, and the karaoke sweep runs from `secondary` to
/// `primary`: a syllable sits in `secondary` until its `{\k}` elapses, then
/// stays `primary`. The defaults are white-then-amber on a dark outline, which
/// reads on both bright and dark footage.
#[derive(Clone, Debug)]
pub struct KaraokeStyle {
    pub title: String,
    pub font: String,
    pub font_size: f64,
    /// The colour a syllable ends on, after its sweep.
    pub primary: u32,
    /// The colour a syllable sits in before its sweep.
    pub secondary: u32,
    pub outline: u32,
    pub back: u32,
    pub outline_width: f64,
    pub shadow: f64,
    pub margin_l: i64,
    pub margin_r: i64,
    pub margin_v: i64,
    pub play_res_x: i64,
    pub play_res_y: i64,
}

impl Default for KaraokeStyle {
    fn default() -> Self {
        Self {
            title: "Karaoke".to_string(),
            font: "Malgun Gothic".to_string(),
            font_size: 64.0,
            primary: 0x0000E5FF,  // BGR: amber
            secondary: 0x00FFFFFF, // BGR: white
            outline: 0x00202020,
            back: 0x80000000,
            outline_width: 3.2,
            shadow: 1.0,
            margin_l: 80,
            margin_r: 80,
            margin_v: 90,
            play_res_x: 1920,
            play_res_y: 1080,
        }
    }
}

impl KaraokeStyle {
    /// ASS draws in its own coordinate space and the player maps that onto the
    /// video, so the resolution has to be the VIDEO's or the player scales the
    /// whole thing and the type comes out wrong. The aligner only ever sees a
    /// waveform and cannot know it, hence the option.
    ///
    /// Nothing else is derived from it. Font size, margins and outline are
    /// left exactly as they are: guessing a scaling rule from the frame width
    /// shrinks a 9:16 portrait video's type to a quarter of what it was, and
    /// nobody asked for that. Set the size you want.
    pub fn set_play_res(&mut self, w: i64, h: i64) {
        self.play_res_x = w.max(16);
        self.play_res_y = h.max(16);
    }
}

fn ass_colour(v: u32) -> String {
    format!("&H{v:08X}")
}

/// `H:MM:SS.cc`, the timestamp format ASS dialogue lines use.
fn fmt_ass_time(sec: f64) -> String {
    let cs = (sec.max(0.0) * 100.0).round() as i64;
    let (cs, s) = (cs % 100, cs / 100);
    let (s, m) = (s % 60, s / 60);
    format!("{}:{:02}:{:02}.{:02}", s / 3600, m, s, cs)
}

/// The karaoke ASS rendering of `doc`, one `\k` per character.
///
/// A `\k<n>` sweep runs for `n` centiseconds, so the durations have to be
/// differences between successive ONSETS, not the token's own span: the tag
/// before the first syllable is the delay until that syllable starts, and each
/// later one runs until the next begins. The line's own end carries the tail.
///
/// Two details that are easy to get wrong, and were:
///
///   * Cues TILE. Two adjacent cues share their boundary instant rather than
///     splitting it, so a character spanning that instant lies inside both. An
///     interval-overlap test assigns it to both and prints it twice -- a
///     duplicated character that reads as a transcription error. Characters are
///     assigned on their own start against half-open cue ranges instead.
///   * Punctuation is a real CTC target that `fix_timestamp` snapped onto the
///     preceding token's end, so it is a point occupying no frames. Given a
///     `\k` of zero it would flash for nothing; emitted without a tag it rides
///     along inside the syllable's sweep and stays on screen.
pub fn cues_to_karaoke(doc: &CueDoc, tokens: &[TokenAlignment], st: &KaraokeStyle) -> String {
    let chars: Vec<&TokenAlignment> =
        tokens.iter().filter(|t| t.piece != "<star>").collect();

    let mut lines: Vec<Vec<&TokenAlignment>> = Vec::with_capacity(doc.cues.len());
    for (i, c) in doc.cues.iter().enumerate() {
        let last = i + 1 == doc.cues.len();
        lines.push(
            chars
                .iter()
                .copied()
                .filter(|t| t.start >= c.start - 1e-9 && (t.start < c.end - 1e-9 || last))
                .collect(),
        );
    }
    // Nothing may be dropped. A character no cue claimed goes to the cue whose
    // range is nearest, so the karaoke still covers the whole transcript.
    let claimed: std::collections::HashSet<usize> =
        lines.iter().flatten().map(|t| t.index).collect();
    let orphans: Vec<&TokenAlignment> =
        chars.iter().copied().filter(|t| !claimed.contains(&t.index)).collect();
    for t in orphans {
        if lines.is_empty() {
            break;
        }
        let best = (0..lines.len())
            .min_by(|&a, &b| {
                let d = |i: usize| {
                    let c = &doc.cues[i];
                    (t.start - c.start).abs().min((t.start - c.end).abs())
                };
                d(a).partial_cmp(&d(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        lines[best].push(t);
    }
    for l in lines.iter_mut() {
        // Transcript order, not time order. An interpolated character has no
        // room of its own -- the midpoint rule makes its neighbours share a
        // frame -- so it sits at the same instant, or a hair past the next one,
        // and sorting by time would print the word backwards.
        l.sort_by_key(|t| t.index);
    }

    let mut out = String::new();
    out.push_str("[Script Info]\n");
    out.push_str(&format!("Title: {}\n", st.title));
    out.push_str("ScriptType: v4.00+\n");
    out.push_str(&format!("PlayResX: {}\n", st.play_res_x));
    out.push_str(&format!("PlayResY: {}\n", st.play_res_y));
    // 0 = smart wrapping. An SRT carries no line breaks either, and the player
    // supplies them; `WrapStyle: 2` means "never wrap, only \\N does", which
    // sends a long cue off the side of the frame. The line breaks are the
    // player's job here exactly as they are for SRT.
    out.push_str("WrapStyle: 0\n");
    out.push_str("ScaledBorderAndShadow: yes\n");
    out.push_str("YCbCr Matrix: TV.709\n\n");
    out.push_str("[V4+ Styles]\n");
    out.push_str(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, \
         OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, \
         ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
         Alignment, MarginL, MarginR, MarginV, Encoding\n",
    );
    out.push_str(&format!(
        "Style: Karaoke,{},{},{},{},{},{},0,0,0,0,100,100,1,0,1,{},{},2,{},{},{},1\n",
        st.font,
        st.font_size as i64,
        ass_colour(st.primary),
        ass_colour(st.secondary),
        ass_colour(st.outline),
        ass_colour(st.back),
        st.outline_width,
        st.shadow,
        st.margin_l,
        st.margin_r,
        st.margin_v,
    ));
    out.push_str("\n[Events]\n");
    out.push_str(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    );

    for (i, line) in lines.iter().enumerate() {
        let Some(cue) = doc.cues.get(i.min(doc.cues.len().saturating_sub(1))) else {
            break;
        };
        if line.is_empty() {
            continue;
        }
        // A sound and the marks that trail it, which is what `build_words`
        // calls a unit too -- so this and `words` cannot disagree about where a
        // unit begins. A mark that opens a word of its own stays its own group,
        // because it has no sound to be timed with.
        let groups = crate::spans::mark_groups(line);
        let mut body = String::new();
        let mut at = cue.start;
        for (gi, g) in groups.iter().enumerate() {
            let span = &line[g.clone()];
            let (text, word) = (
                span.iter().map(|t| t.piece.as_str()).collect::<String>(),
                span[0].word_id,
            );
            let nxt = groups
                .get(gi + 1)
                .map(|nx| line[nx.start].start)
                .unwrap_or(cue.end)
                .max(at);
            // A gap the transcript has, and only those. The first group of a
            // line never takes one: the line break already stands where the
            // space was, exactly as it does in the SRT.
            if gi > 0 {
                let prev = &line[groups[gi - 1].start..groups[gi - 1].end];
                if prev[0].word_id != word {
                    body.push_str("\\h");
                }
            }
            // A character the vocabulary had no target for has no frames of its
            // own -- the midpoint rule makes its neighbours share one -- so the
            // sweep computes to zero. A `\k0` is not merely a zero sweep: mpv,
            // libass and xy-VSFilter each draw it differently from a positive
            // one, and a character given one comes out in the unsung colour and
            // sometimes with its neighbours' metrics dropped. Emitting no tag at
            // all lets it ride inside the previous character's sweep, which is
            // both the colour and the rendering every player agrees on.
            if !is_mark(&text) {
                let cs = ((nxt - at) * 100.0).round() as i64;
                if cs > 0 {
                    body.push_str(&format!("{{\\k{cs}}}"));
                }
                at = nxt;
            }
            body.push_str(&text);
        }
        if let Some(cue) = doc.cues.get(i) {
            let tail = ((cue.end - at) * 100.0).round() as i64;
            if tail > 0 {
                body.push_str(&format!("{{\\k{tail}}}"));
            }
        }
        out.push_str(&format!(
            "Dialogue: 0,{},{},Karaoke,,0,0,0,,{}\n",
            fmt_ass_time(cue.start),
            fmt_ass_time(cue.end),
            body
        ));
    }
    out
}

/// The text of the karaoke file with its markup removed, for checking it
/// against the transcript: every character the alignment produced must appear
/// exactly once, and the words must come back in order.
pub fn karaoke_plain_text(ass: &str) -> String {
    let mut out = String::new();
    for line in ass.lines().filter(|l| l.starts_with("Dialogue:")) {
        // ten comma-separated fields, and the text may itself contain commas
        let Some(body) = line.splitn(10, ',').nth(9) else { continue };
        for block in body.split('{').skip(1) {
            // strip a leading tag such as `\k120`, keep the text after it
            let Some((tag, rest)) = block.split_once('}') else { continue };
            if tag.trim_start_matches('\\').starts_with('k') {
                out.push_str(rest);
            }
        }
        out.push(' ');
    }
    out.replace("\\h", " ").split_whitespace().collect::<Vec<_>>().join("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::tok;
    use crate::views::{build_cues, CueOut};


    /// Cues TILE, so a character spanning two adjacent cues is inside both.
    /// Assigning on interval overlap printed it twice, which reads as a
    /// transcription error -- this is the regression.
    #[test]
    fn karaoke_does_not_duplicate_a_character_on_a_cue_boundary() {
        // Two cues sharing the instant 1.0, and one character sitting on it.
        let cue1 = CueOut { index: 1, start: 0.0, end: 1.0, text: "我的情".into() };
        let cue2 = CueOut { index: 2, start: 1.0, end: 2.0, text: "不移".into() };
        let doc = CueDoc { script: "cjk".into(), cues: vec![cue1, cue2] };
        // 我 0.0-.3   的 .3-.6   情 .6-1.0   不 1.0-1.4   移 1.4-2.0
        let names = ["我", "的", "情", "不", "移"];
        let starts = [0.0f64, 0.3, 0.6, 1.0, 1.4];
        let tokens: Vec<TokenAlignment> = names
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut t = tok(i, p, (starts[i] * 50.0) as i64, (starts[i] * 50.0) as i64 + 1);
                t.word_id = 0;
                t
            })
            .collect();
        let ass = cues_to_karaoke(&doc, &tokens, &KaraokeStyle::default());
        let plain = karaoke_plain_text(&ass);
        assert_eq!(plain, "我的情不移", "every character exactly once, in order");
        // one sweep per character group; no tail here, because each cue's last
        // onset already lands on the cue's end
        assert_eq!(ass.matches("\\k").count(), plain.chars().count());
    }


    /// The whole point: strip the markup back off and get the transcript.
    #[test]
    fn karaoke_round_trips_to_the_transcript() {
        let names = ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"];
        let mut tokens: Vec<TokenAlignment> = Vec::new();
        for (i, p) in names.iter().enumerate() {
            if *p == " " {
                continue;                    // spaces are not targets
            }
            let word = if i < 5 { 0 } else { 1 };
            let mut t = tok(i, p, 20 + 10 * i as i64, 28 + 10 * i as i64);
            t.word_id = word;
            tokens.push(t);
        }
        let doc = build_cues(&tokens);
        let ass = cues_to_karaoke(&doc, &tokens, &KaraokeStyle::default());
        assert_eq!(karaoke_plain_text(&ass), "helloworld");

        // and the words come back separated, because the space is recovered
        // from the word id rather than from a token
        assert!(ass.contains("\\h"), "a word boundary becomes a hard space");
        // and each sweep is a difference between onsets, so a tag exists per char
        let tags = ass.matches("\\k").count();
        assert!(tags >= 10, "one sweep per character, got {tags}");
    }


    /// A gap belongs to the character that FOLLOWS it, and that character may
    /// not be the one at the same index once the marks have been folded into
    /// their syllable. Reading the word id off the ungrouped token list put
    /// every `\h` one group late, so the karaoke said `...あの の謙...` where
    /// the transcript says `かぶって、 あの謙虚`.
    #[test]
    fn a_gap_lands_before_the_character_that_follows_it() {
        // か ぶ っ て 、  [gap]  あ の 謙 虚  -- one cue, so the gap is inside it
        let names = ["か", "ぶ", "っ", "て", "、", "あ", "の", "謙", "虚"];
        let words = [0, 0, 0, 0, 0, 1, 1, 1, 1];
        let mut tokens: Vec<TokenAlignment> = names
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut t = tok(i, p, 20 + 10 * i as i64, 28 + 10 * i as i64);
                t.word_id = words[i];
                t
            })
            .collect();
        // the star the tokenizer puts in front of each word
        tokens.insert(0, {
            let mut s = tok(0, "<star>", 10, 18);
            s.word_id = usize::MAX;
            s
        });
        let doc = build_cues(&tokens);
        assert_eq!(doc.cues.len(), 1, "one cue: the gap cannot fall on a line break");
        let ass = cues_to_karaoke(&doc, &tokens, &KaraokeStyle::default());
        let body = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        // the gap, then the sweep for the character it belongs in front of
        let h = body.find("\\h").expect("the transcript's space is there");
        let rest = &body[h + 2..];
        let tag = rest.find("}").expect("a sweep tag");
        assert!(
            rest.starts_with("{\\k") && rest[tag + 1..].starts_with('あ'),
            "the gap must sit immediately before あ, not before {}: {body}",
            rest[tag + 1..].chars().next().unwrap_or('?')
        );
        assert_eq!(karaoke_plain_text(&ass), "かぶって、あの謙虚");
    }


    /// A mark joins the syllable before it only inside the same word. The
    /// transcript says `元 CA 。`, and folding the full stop onto the `A` printed
    /// `元 CA。` -- four cues out of 1,555 on the Japanese material, and the
    /// gap simply gone.
    #[test]
    fn a_mark_does_not_swallow_the_gap_in_front_of_it() {
        let names = ["元", "C", "A", "。"];
        let words = [0, 1, 1, 2];
        let tokens: Vec<TokenAlignment> = names
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut t = tok(i, p, 20 + 10 * i as i64, 28 + 10 * i as i64);
                t.word_id = words[i];
                t
            })
            .collect();
        let doc = CueDoc {
            script: "cjk".into(),
            cues: vec![CueOut { index: 1, start: 0.4, end: 1.6, text: "元 CA 。".into() }],
        };
        let ass = cues_to_karaoke(&doc, &tokens, &KaraokeStyle::default());
        let body = ass.lines().find(|l| l.starts_with("Dialogue:")).unwrap();
        assert!(body.contains("A\\h。"), "the gap before 。 is gone: {body}");
        // and the mark still rides inside A's sweep rather than getting a tag
        // of its own: 元, C, and A。 are the three groups, and the line ends on
        // the last onset so there is no tail
        assert_eq!(karaoke_plain_text(&ass), "元CA。");
        // `元 CA 。` has two gaps and gets two: one at the word boundary the
        // group loop sees, one for the mark that opened a word of its own --
        // and not a third for the syllable after it
        assert_eq!(
            body.matches("\\h").count(),
            2,
            "one \\h per gap, no more: {body}"
        );
        // the mark rides inside A's sweep rather than being given one of its own
        assert!(!body.contains("\\k0"), "no zero sweep was invented: {body}");
    }


    /// `\k` durations are differences between onsets, and they must add up to
    /// the line: a sweep that overruns or underruns drifts every later mark.
    #[test]
    fn karaoke_sweeps_close_on_the_line() {
        let doc = CueOut { index: 1, start: 1.0, end: 3.0, text: "abc".into() };
        let doc = CueDoc { script: "latin".into(), cues: vec![doc] };
        let mut tokens = Vec::new();
        // onsets at 1.0, 1.5 and 2.0 inside a line that runs 1.0 -> 3.0
        for (i, p) in ["a", "b", "c"].iter().enumerate() {
            let mut t = tok(i, p, 50 + 25 * i as i64, 74 + 25 * i as i64);
            t.word_id = i;
            tokens.push(t);
        }
        let ass = cues_to_karaoke(&doc, &tokens, &KaraokeStyle::default());
        let sum: i64 = ass
            .match_indices("\\k")
            .map(|(i, _)| {
                let rest = &ass[i + 2..];
                let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
                rest[..end].parse::<i64>().unwrap()
            })
            .sum();
        assert_eq!(sum, 200, "200 cs of sweeps across a 2 s line");
    }
}
