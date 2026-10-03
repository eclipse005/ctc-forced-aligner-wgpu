//! Derived views on one character alignment.
//!
//! `spans` tiles the timeline: silence is split at the blank-run midpoint, the
//! first span starts at 0, and the last span ends at the audio duration.
//! `cues` are subtitle lines whose times stay on the spoken characters
//! (CrisperWhisper / OneAsr standard cost model).

use crate::spans::{splits_between_characters, PUNCT};
use crate::viterbi::TokenAlignment;

#[derive(Clone, Debug)]
pub struct SpanOut {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub score: f64,
}

#[derive(Clone, Debug)]
pub struct CueOut {
    pub index: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Advance `ti` past any `<star>` targets and return the next real piece.
///
/// `<star>` is the reference's synthetic word-boundary marker, not a character
/// of the transcript (see [`crate::vocab::Vocab::tokenise_with_stars`]), so a
/// walk that pairs transcript characters with token pieces in lockstep has to
/// step over it. It does not: `pieces[0]` is a star in BOTH placements — the
/// `edges` star and the `segment` star — so an unskipped walk compares the
/// transcript's first character against `"<star>"` for the whole file, never
/// advances, and reports every character as unmatched.
fn skip_stars(pieces: &[String], mut ti: usize) -> usize {
    while ti < pieces.len() && pieces[ti] == "<star>" {
        ti += 1;
    }
    ti
}

/// Whether `piece` is exactly the one character `ch` — the shape a kept
/// transcript character has once the vocabulary has emitted it.
fn piece_is_char(piece: &str, ch: char) -> bool {
    piece.chars().next() == Some(ch) && piece.chars().nth(1).is_none()
}

/// Whether `text` is written without spaces between words.
///
/// The question this answers is not "is this CJK" — that asks about Unicode
/// blocks and cannot tell a space-delimited script (Hangul, Kana) from an
/// unspaced one (Han, Thai), even though they call for opposite treatment.
/// What matters is whether splitting on whitespace yields *words*, and the
/// length of what comes out answers it without a language table.
///
/// Measured on real transcripts: space-delimited scripts average 3.9-4.5
/// characters per whitespace token, unspaced ones 33.4. A transcript mixes
/// both, and so does the verdict — it takes the majority, which is what a
/// single granularity for the whole file has to do.
pub fn text_is_unspaced(text: &str) -> bool {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return false;
    }
    let chars: usize = tokens.iter().map(|t| t.chars().count()).sum();
    chars * 8 > tokens.len() * 100
}

/// `word` where the transcript separates words with spaces, `char` where it
/// does not. One verdict for the whole text, chosen by [`text_is_unspaced`].
pub fn auto_split(text: &str) -> &'static str {
    if text_is_unspaced(text) {
        "char"
    } else {
        "word"
    }
}

fn r4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn chunk_ranges(text: &str, split: &str) -> Vec<(usize, usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    // byte offsets are not needed; we index by char.
    if split == "char" {
        return chars
            .iter()
            .enumerate()
            .filter(|(_, ch)| !ch.is_whitespace())
            .map(|(i, ch)| (i, i + 1, ch.to_string()))
            .collect();
    }
    if split == "word" {
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }
            let a = i;
            while i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
            }
            out.push((a, i, chars[a..i].iter().collect()));
        }
        return out;
    }
    // sentence, keeping a decimal point inside a number
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut i = 0usize;
    let closers: Vec<char> = "»”’）】》]'\"）".chars().collect();
    while i < chars.len() {
        let ch = chars[i];
        if ch == '.' && i > 0 && i + 1 < chars.len() && chars[i - 1].is_ascii_digit() && chars[i + 1].is_ascii_digit()
        {
            i += 1;
            continue;
        }
        if " .!?…。！？；;".contains(ch) && ch != ' ' {
            let mut j = i + 1;
            while j < chars.len() && closers.contains(&chars[j]) {
                j += 1;
            }
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let raw: String = chars[pos..j].iter().collect();
            if raw.chars().any(|c| !c.is_whitespace()) {
                out.push((pos, j, raw.trim().to_string()));
            }
            pos = j;
            i = j;
            continue;
        }
        i += 1;
    }
    if pos < chars.len() {
        let raw: String = chars[pos..].iter().collect();
        if raw.chars().any(|c| !c.is_whitespace()) {
            out.push((pos, chars.len(), raw.trim().to_string()));
        }
    }
    out
}

fn token_index(text: &str, pieces: &[String]) -> Vec<i64> {
    let mut ti = 0usize;
    let mut out = Vec::new();
    for ch in text.chars() {
        ti = skip_stars(pieces, ti);
        if ti < pieces.len() && piece_is_char(&pieces[ti], ch) {
            out.push(ti as i64);
            ti += 1;
        } else {
            out.push(-1);
        }
    }
    out
}

fn mid_start(prev_end_frame: i64, this_start: i64) -> i64 {
    let blank_lo = prev_end_frame + 1;
    let blank_hi = this_start - 1;
    if blank_hi >= blank_lo {
        (blank_lo + blank_hi) / 2
    } else {
        this_start
    }
}

fn mid_end(this_end_frame: i64, next_start: i64) -> i64 {
    let blank_lo = this_end_frame + 1;
    let blank_hi = next_start - 1;
    if blank_hi >= blank_lo {
        (blank_lo + blank_hi) / 2 + 1
    } else {
        this_end_frame + 1
    }
}

/// Segments of `text` over the whole file, with the leading and trailing
/// silence folded in so the spans tile the timeline and no stretch of audio
/// belongs to nobody.
///
/// `split` is the unit: `char` for a transcript that does not space its words,
/// `word` for one that does. The caller normally passes [`auto_split`]; it
/// stays a parameter because a library user with a mixed transcript may want
/// to force one, and because a two-word test string has no statistical
/// evidence either way.
pub fn build_spans(
    text: &str,
    tokens: &[TokenAlignment],
    frames: usize,
    frame_rate: f64,
    frame_scores: &[f64],
    split: &str,
) -> Vec<SpanOut> {
    // `json` already carries `words` and `segments`, so the only thing spans
    // add is the gapless timeline; the granularity is not a user choice.
    let pieces: Vec<String> = tokens.iter().map(|t| t.piece.clone()).collect();
    let owned = token_index(text, &pieces);
    let mut rows: Vec<(usize, usize, String, Vec<usize>)> = Vec::new();
    for (a, b, display) in chunk_ranges(text, split) {
        let idxs: Vec<usize> = (a..b).filter_map(|i| if owned[i] >= 0 { Some(owned[i] as usize) } else { None }).collect();
        if idxs.is_empty() {
            continue;
        }
        let lo = *idxs.first().unwrap();
        let hi = *idxs.last().unwrap();
        rows.push((lo, hi, display, idxs));
    }
    let mut segments = Vec::new();
    for i in 0..rows.len() {
        let lo = rows[i].0;
        let hi = rows[i].1;
        let display = rows[i].2.clone();
        let idxs = rows[i].3.clone();
        let start_f = if i == 0 {
            0
        } else {
            mid_start(tokens[rows[i - 1].1].end_frame, tokens[lo].start_frame)
        };
        let end_f = if i + 1 == rows.len() {
            frames as i64
        } else {
            mid_end(tokens[hi].end_frame, tokens[rows[i + 1].0].start_frame)
        };
        let score = if !frame_scores.is_empty() && end_f > start_f {
            let a = start_f.max(0) as usize;
            let b = (end_f as usize).min(frame_scores.len());
            if b > a {
                frame_scores[a..b].iter().sum::<f64>() / (b - a) as f64
            } else {
                0.0
            }
        } else {
            idxs.iter().map(|&k| tokens[k].score).sum::<f64>() / idxs.len() as f64
        };
        segments.push(SpanOut {
            start: start_f as f64 / frame_rate,
            end: end_f as f64 / frame_rate,
            text: display.clone(),
            score,
        });
    }
    // Spans already meet at the midpoint of the blank between them, so the
    // only thing left to do is clamp the sub-frame overlaps that rounding
    // leaves behind and make the timeline strictly contiguous. A user-tunable
    // threshold used to sit here; the default did this and nothing else, and
    // every other value was a guess about somebody's subtitle style.
    for i in 0..segments.len().saturating_sub(1) {
        if segments[i + 1].start - segments[i].end < 0.0 {
            segments[i + 1].start = segments[i].end;
        }
    }
    for seg in &mut segments {
        seg.start = r4(seg.start);
        seg.end = r4(seg.end);
        seg.score = r4(seg.score);
    }
    segments
}

pub fn spans_to_txt(spans: &[SpanOut]) -> String {
    let mut s = String::new();
    for sp in spans {
        s.push_str(&format!("{}-{}: {}\n", sp.start, sp.end, sp.text));
    }
    s
}

// --- cues -----------------------------------------------------------------

struct Tok {
    start: f64,
    end: f64,
    token: String,
    space_before: bool,
    /// Whether this token is a single character of a script that writes without
    /// spaces, and so may be split further. Decided per token, not per file: a
    /// file with one Han character in it was previously taken to be Chinese
    /// throughout, and the line-breaker duly cut `alignment` into `alignm` and
    /// `ent`.
    splittable: bool,
}

const GRACE: f64 = 11.0;
const LENGTH_W: f64 = 0.3;
const SHORT_PENALTY: f64 = 0.8;
const PAUSE_SEC: f64 = 0.35;
const MAX_CUE_SEC: f64 = 7.0;
const WORD_COST: f64 = 2.5;

fn is_terminal(c: char) -> bool {
    matches!(c, '.' | '?' | '!' | '。' | '！' | '？' | '…')
}
fn is_soft(c: char) -> bool {
    matches!(c, ';' | ':' | '；' | '：')
}
fn is_comma(c: char) -> bool {
    matches!(c, ',' | '，' | '、')
}
fn is_open(c: char) -> bool {
    matches!(c, '(' | '[' | '{' | '（' | '【' | '「' | '『' | '“' | '‘')
}
fn is_close(c: char) -> bool {
    matches!(c, ',' | '.' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '，' | '。' | '；' | '：' | '！' | '？' | '）' | '】' | '」' | '』' | '”' | '’')
}
fn is_strip_char(c: char) -> bool {
    is_terminal(c) || is_soft(c) || is_comma(c) || is_open(c) || is_close(c) || matches!(c, '"' | '\'' | '`' | '~' | '—' | '-' | '*' | '_')
}

fn strip_tok(tok: &str) -> &str {
    tok.trim_matches(is_strip_char)
}

fn is_decimal(tok: &str) -> bool {
    let b = tok.as_bytes();
    if b.len() < 3 {
        return false;
    }
    let mut dot = None;
    for (i, c) in b.iter().enumerate() {
        if *c == b'.' {
            if dot.is_some() {
                return false;
            }
            dot = Some(i);
        } else if !c.is_ascii_digit() {
            return false;
        }
    }
    matches!(dot, Some(i) if i > 0 && i + 1 < b.len())
}

fn has(tok: &str, f: fn(char) -> bool) -> bool {
    tok.chars().any(f)
}

fn function_left(word: &str) -> bool {
    matches!(
        word,
        "a" | "an" | "the" | "of" | "to" | "in" | "on" | "at" | "for" | "and" | "or" | "but"
            | "is" | "was" | "are" | "were" | "be" | "been" | "being" | "with" | "from" | "by"
            | "as" | "that" | "this" | "these" | "those" | "my" | "your" | "his" | "her" | "its"
            | "our" | "their" | "into" | "about" | "than" | "so" | "if" | "when" | "while"
            | "up" | "out" | "down" | "over" | "just" | "very" | "too" | "also" | "not" | "no"
    )
}

fn connector(word: &str) -> bool {
    matches!(
        word,
        "and" | "but" | "so" | "or" | "because" | "then" | "when" | "if" | "though" | "although"
            | "while" | "since" | "after" | "before" | "yet" | "plus" | "anyway"
    )
}

fn discourse(word: &str) -> bool {
    matches!(
        word,
        "okay" | "ok" | "now" | "well" | "so" | "right" | "yeah" | "oh" | "hey" | "come" | "look"
            | "listen" | "alright" | "fine" | "and"
    )
}

/// The line-breaking unit, decided per word.
///
/// A word whose characters are mostly from a script that writes without spaces
/// is split into its characters; every other word is one unit. Deciding this
/// per word rather than per file is what lets `你好这个` break between
/// characters while `alignment` stays whole -- the previous one-flag-per-file
/// test called any transcript containing a Han character Chinese throughout,
/// and cut Latin words in half at the cue boundaries.
fn cue_tokens(tokens: &[TokenAlignment]) -> Vec<Tok> {
    let mut out: Vec<Tok> = Vec::new();
    let mut buf: Vec<&TokenAlignment> = Vec::new();
    let mut pending = false;

    let flush = |buf: &mut Vec<&TokenAlignment>, pending: &mut bool, out: &mut Vec<Tok>| {
        if buf.is_empty() {
            return;
        }
        let text: String = buf.iter().map(|t| t.piece.as_str()).collect();
        let (start, end) = (buf[0].start, buf[buf.len() - 1].end);
        let space_before = *pending && !out.is_empty();
        if splits_between_characters(&buf) {
            // One unit per character, so the splitter can break between them.
            for (n, t) in buf.iter().enumerate() {
                out.push(Tok {
                    start: t.start,
                    end: t.end,
                    token: t.piece.clone(),
                    space_before: space_before && n == 0,
                    splittable: true,
                });
            }
        } else {
            out.push(Tok { start, end, token: text, space_before, splittable: false });
        }
        buf.clear();
        *pending = false;
    };

    for t in tokens {
        // Two different boundary signals, and they arrive in different places.
        //
        // `<star>` and whitespace are markers that sit BETWEEN words, so
        // discarding them loses nothing -- and the star has to go, or it is
        // printed in the subtitle.
        //
        // A change of `word_id` arrives ON the first character of the next
        // word: that character is real text. Treating it as a boundary threw
        // away the first letter of every word, which is what turned
        // "All right, we purse welcome" into "All ight, e urse elcome". Flush
        // on the change and keep the token.
        //
        // The word id is the signal that works for every script. The star
        // placement does not: English takes `edges`, which has exactly two
        // stars in a whole file, so grouping on stars made the entire
        // transcript one 3-minute cue with every space gone.
        if t.piece == "<star>" {
            // Skipped, but NOT a boundary. The star a word opens with carries
            // the id of the word before it, so a word whose first character
            // had no target has its star in the middle of the word; letting it
            // vote here cut that word in two and printed a space the
            // transcript does not have. `word_id` alone draws the boundary.
            pending = true;
            continue;
        }
        if t.piece.trim().is_empty() {
            flush(&mut buf, &mut pending, &mut out);
            pending = true;
            continue;
        }
        if t.word_id != buf.first().map(|b| b.word_id).unwrap_or(t.word_id) {
            flush(&mut buf, &mut pending, &mut out);
            pending = true;
        }
        buf.push(t);
    }
    flush(&mut buf, &mut pending, &mut out);
    merge_decimals(out)
}

fn merge_decimals(tokens: Vec<Tok>) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if i + 2 < tokens.len()
            && tokens[i].token.chars().all(|c| c.is_ascii_digit())
            && tokens[i + 1].token == "."
            && !tokens[i + 1].space_before
            && tokens[i + 2].token.chars().all(|c| c.is_ascii_digit())
            && !tokens[i + 2].space_before
        {
            out.push(Tok {
                start: tokens[i].start,
                end: tokens[i + 2].end,
                token: format!("{}{}{}", tokens[i].token, tokens[i + 1].token, tokens[i + 2].token),
                space_before: tokens[i].space_before,
                splittable: false,
            });
            i += 3;
            continue;
        }
        out.push(Tok {
            start: tokens[i].start,
            end: tokens[i].end,
            token: tokens[i].token.clone(),
            space_before: tokens[i].space_before,
            splittable: tokens[i].splittable,
        });
        i += 1;
    }
    out
}

/// Whether `text` carries anything a viewer would read as speech.
fn says_something(text: &str) -> bool {
    text.chars().any(|c| !c.is_whitespace() && !PUNCT.contains(&c))
}

/// Fold cues that are nothing but punctuation into the cue they belong to.
///
/// These are splitter artifacts, not subtitles. A sentence-final mark gets
/// `fix_timestamp`-snapped back onto the end of the token before it, which
/// makes it a *point* (`start == end`); when it lands in a span of its own that
/// span is a millisecond wide, and the SRT grows a cue reading `.` on screen
/// for no time at all.
///
/// The mark is kept — dropping it would silently edit the transcript — and no
/// duration is invented for it: it simply stops being a cue of its own. How
/// long a subtitle should stay up is the player's business, not the aligner's.
fn fold_bare_marks(raw: Vec<(f64, f64, String)>) -> Vec<(f64, f64, String)> {
    let mut out: Vec<(f64, f64, String)> = Vec::with_capacity(raw.len());
    for (s, e, t) in raw {
        match out.last_mut() {
            Some(last) if !says_something(&t) => {
                last.1 = last.1.max(e);
                last.2.push_str(&t);
            }
            _ => out.push((s, e, t)),
        }
    }
    // A run of marks at the very start has no cue behind it to join; hand it
    // forward to the first one that says something.
    let head = out.iter().take_while(|(_, _, t)| !says_something(t)).count();
    if head > 0 && head < out.len() {
        let marks: String = out[..head].iter().map(|(_, _, t)| t.as_str()).collect();
        out[head].0 = out[head - 1].0;
        out[head].2.insert_str(0, &marks);
        out.drain(..head);
    }
    out
}

fn sentence_end(words: &[Tok], i: usize) -> bool {    let tok = &words[i].token;
    if is_decimal(tok) {
        return false;
    }
    if tok == "." && i > 0 && i + 1 < words.len() {
        let prev = words[i - 1].token.chars().last().unwrap_or(' ');
        let next = words[i + 1].token.chars().next().unwrap_or(' ');
        if prev.is_ascii_digit() && next.is_ascii_digit() {
            return false;
        }
    }
    has(strip_tok(tok), is_terminal) || has(tok, is_terminal)
}

fn boundary(words: &[Tok], i: usize) -> f64 {
    let left = &words[i].token;
    let right = &words[i + 1].token;
    let decimal = is_decimal(left);
    let ls = strip_tok(left).to_lowercase();
    let rs = strip_tok(right).to_lowercase();
    if left.chars().last().is_some_and(is_open) {
        return f64::INFINITY;
    }
    if right.chars().next().is_some_and(is_close) {
        return f64::INFINITY;
    }
    if !decimal && (has(strip_tok(left), is_terminal) || has(left, is_terminal)) {
        return 0.0;
    }
    if has(left, is_soft) {
        return 0.5;
    }
    if has(left, is_comma) {
        let bare = ls.trim_matches(|c| c == ',' || c == '，');
        if discourse(bare) {
            return WORD_COST;
        }
        return 1.0;
    }
    if function_left(&ls) {
        return f64::INFINITY;
    }
    if connector(&rs) && !connector(&ls) {
        return 0.8;
    }
    if words[i + 1].start - words[i].end >= PAUSE_SEC {
        return 1.5;
    }
    WORD_COST
}

/// Budget weight of one unit: a character of an unspaced script costs its
/// character count, a space-delimited word costs one.
fn units(t: &Tok) -> f64 {
    let s = strip_tok(&t.token);
    if s.is_empty() {
        0.0
    } else if t.splittable {
        s.chars().count() as f64
    } else {
        1.0
    }
}

/// The budget a span is measured against, and the character ceiling that goes
/// with it. A span of Chinese wants a character count and a span of Latin wants
/// a word count; a mixed transcript has both, so the majority of the span's own
/// units picks the budget rather than a flag decided for the whole file.
fn span_budget(span: &[&Tok], latin_words: f64, latin_chars: f64, cjk_chars: f64)
    -> (f64, Option<f64>)
{
    let splittable = span.iter().filter(|t| t.splittable).count();
    if splittable * 2 > span.len() {
        (cjk_chars, None)
    } else {
        (latin_words, Some(latin_chars))
    }
}

fn dchars(words: &[Tok], a: usize, b: usize) -> usize {
    words[a..=b].iter().map(|w| w.token.chars().count()).sum()
}

fn dur(words: &[Tok], a: usize, b: usize) -> f64 {
    words[b].end - words[a].start
}

fn dp_split(words: &[Tok], target: f64, char_limit: Option<f64>) -> Vec<usize> {
    let n = words.len();
    if n < 2 {
        return Vec::new();
    }
    let mut pre = vec![0.0; n + 1];
    for (k, w) in words.iter().enumerate() {
        pre[k + 1] = pre[k] + units(w);
    }
    let fits_chars = match char_limit {
        Some(lim) => dchars(words, 0, n - 1) as f64 <= lim,
        None => true,
    };
    if pre[n] <= target && fits_chars && dur(words, 0, n - 1) <= MAX_CUE_SEC {
        return Vec::new();
    }
    let cost: Vec<f64> = (0..n - 1).map(|k| boundary(words, k)).collect();
    let mut dp = vec![f64::INFINITY; n + 1];
    let mut prev = vec![0usize; n + 1];
    dp[0] = 0.0;
    for i in 1..=n {
        let mut j = i;
        while j > 0 {
            j -= 1;
            let seg_u = pre[i] - pre[j];
            if i - j > 1 && seg_u > target + GRACE {
                break;
            }
            let seg_c = dchars(words, j, i - 1);
            if let Some(lim) = char_limit {
                if i - j > 1 && seg_c as f64 > lim + GRACE {
                    continue;
                }
            }
            if i - j > 1 && dur(words, j, i - 1) > MAX_CUE_SEC {
                break;
            }
            if dp[j].is_infinite() {
                continue;
            }
            if j > 0 && cost[j - 1].is_infinite() {
                continue;
            }
            let mut c = dp[j] + if j > 0 { cost[j - 1] } else { 0.0 };
            c += LENGTH_W * (seg_u - target).abs() / target;
            if let Some(lim) = char_limit {
                c += LENGTH_W * 0.5 * (seg_c as f64 - lim).abs() / lim;
            }
            if seg_u > 0.0 && seg_u <= 2.0 {
                c += SHORT_PENALTY;
            }
            // Ties resolve toward the later cut in Latin and toward the earlier
            // one in Chinese, which is what the two scripts' line shapes want:
            // Latin words are short, so an even split wins; a Han sentence has
            // no natural break, so the greedy one that stays near the target
            // does. `char_limit` is the signal for which regime this is.
            let better = if char_limit.is_some() { c < dp[i] } else { c <= dp[i] };
            if better {
                dp[i] = c;
                prev[i] = j;
            }
        }
    }
    if dp[n].is_infinite() {
        return greedy(words, target, char_limit, &pre);
    }
    let mut cuts = Vec::new();
    let mut cur = n;
    while cur > 0 {
        let p = prev[cur];
        if p > 0 {
            cuts.push(p);
        }
        cur = p;
    }
    cuts.sort_unstable();
    cuts
}

fn greedy(words: &[Tok], target: f64, char_limit: Option<f64>, pre: &[f64]) -> Vec<usize> {
    let n = words.len();
    let mut cuts = Vec::new();
    let mut j = 0usize;
    while j + 1 < n {
        let mut k = j + 1;
        while k < n
            && pre[k + 1] - pre[j] <= target + GRACE
            && char_limit.map(|lim| dchars(words, j, k) as f64 <= lim + GRACE).unwrap_or(true)
            && dur(words, j, k) <= MAX_CUE_SEC
        {
            k += 1;
        }
        k -= 1;
        while k > j && boundary(words, k - 1).is_infinite() {
            k -= 1;
        }
        if k == j {
            k = j + 1;
        }
        if k >= n {
            break;
        }
        cuts.push(k);
        j = k;
    }
    cuts
}

/// Render a segment. One rule for both scripts: a space goes where the
/// tokenizer put a word boundary. A CJK run has none inside a word, so it
/// renders tight; Latin words each carry one, so they render separated.
fn join_seg(seg: &[Tok]) -> String {
    let mut s = String::new();
    for (i, t) in seg.iter().enumerate() {
        if i > 0 && t.space_before {
            s.push(' ');
        }
        s.push_str(&t.token);
    }
    s
}

fn wrap_line(text: &str) -> String {
    if text.chars().count() <= 42 {
        return text.to_string();
    }
    let half = text.chars().count() / 2;
    let mut best: Option<usize> = None;
    let mut bestd = usize::MAX;
    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() {
            let at = text[..i].chars().count();
            let d = at.abs_diff(half);
            if d < bestd {
                bestd = d;
                best = Some(i);
            }
        }
    }
    match best {
        Some(i) => {
            let a = text[..i].trim_end();
            let b = text[i..].trim_start();
            if a.is_empty() || b.is_empty() {
                text.to_string()
            } else {
                format!("{a}\n{b}")
            }
        }
        None => text.to_string(),
    }
}

fn fmt_ms(mut total: i64) -> String {
    if total < 0 {
        total = 0;
    }
    let h = total / 3_600_000;
    let r = total % 3_600_000;
    let m = r / 60_000;
    let r = r % 60_000;
    let s = r / 1000;
    let x = r % 1000;
    format!("{h:02}:{m:02}:{s:02},{x:03}")
}

pub struct CueDoc {
    pub script: String,
    pub cues: Vec<CueOut>,
}

pub fn build_cues(tokens: &[TokenAlignment]) -> CueDoc {
    // Subtitle geometry. These are the long-form broadcast norms -- two lines
    // of 44 columns, at most 7 s on screen -- and a CJK glyph is two columns
    // wide, so 22 characters is the same width as 44 Latin ones. The
    // short/loose variants that used to be selectable measured as a no-op on
    // real material: a transcript that already has sentence punctuation, or
    // simply many pauses, is split by those long before a length target
    // binds, so the knob chose between 54 and 53 cues and nothing more.
    let (latin_words, latin_chars, cjk_chars) = (16.0, 88.0, 22.0);
    let words = cue_tokens(tokens);

    let mut spans: Vec<Vec<Tok>> = Vec::new();
    let mut cur = Vec::new();
    for i in 0..words.len() {
        let end = sentence_end(&words, i);
        cur.push(Tok {
            start: words[i].start,
            end: words[i].end,
            token: words[i].token.clone(),
            space_before: words[i].space_before,
            splittable: words[i].splittable,
        });
        if end {
            spans.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        spans.push(cur);
    }
    let mut raw: Vec<(f64, f64, String)> = Vec::new();
    for span in &spans {
        let refs: Vec<&Tok> = span.iter().collect();
        let (target, char_limit) = span_budget(&refs, latin_words, latin_chars, cjk_chars);
        let cuts = dp_split(span, target, char_limit);
        let mut bounds = vec![0];
        bounds.extend(cuts.iter().copied());
        bounds.push(span.len());
        for w in bounds.windows(2) {
            let seg = &span[w[0]..w[1]];
            if seg.is_empty() {
                continue;
            }
            let text = join_seg(seg);
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            raw.push((seg[0].start, seg[seg.len() - 1].end, wrap_line(text)));
        }
    }
    raw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)));
    let raw = fold_bare_marks(raw);
    let mut ms: Vec<(i64, i64, String)> = raw
        .into_iter()
        .map(|(s, e, t)| ((s * 1000.0).round() as i64, (e * 1000.0).round() as i64, t))
        .collect();
    for i in 0..ms.len() {
        if ms[i].1 <= ms[i].0 {
            ms[i].1 = ms[i].0 + 1;
        }
        if i + 1 < ms.len() && ms[i].1 > ms[i + 1].0 {
            ms[i].1 = ms[i + 1].0.max(ms[i].0);
        }
    }
    let cues = ms
        .into_iter()
        .enumerate()
        .map(|(i, (s, e, t))| CueOut {
            index: i + 1,
            start: s as f64 / 1000.0,
            end: e as f64 / 1000.0,
            text: t,
        })
        .collect();
    CueDoc {
        script: if words.iter().any(|t| t.splittable) { "cjk" } else { "latin" }.to_string(),
        cues,
    }
}

pub fn cues_to_srt(doc: &CueDoc) -> String {
    let mut parts = Vec::new();
    for c in &doc.cues {
        parts.push(format!(
            "{}\n{} --> {}\n{}\n",
            c.index,
            fmt_ms((c.start * 1000.0).round() as i64),
            fmt_ms((c.end * 1000.0).round() as i64),
            c.text
        ));
    }
    parts.join("\n")
}

// --- karaoke (ASS) ---------------------------------------------------------

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

/// Whether `piece` is a mark rather than something with a sound.
fn is_mark(piece: &str) -> bool {
    !piece.is_empty() && piece.chars().all(|c| PUNCT.contains(&c))
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

    fn tok(i: usize, piece: &str, sf: i64, ef: i64) -> TokenAlignment {
        tokw(i, piece, i, sf, ef)
    }

    /// A token that belongs to word `w`. The word id is what the tokenizer
    /// assigns per whitespace-delimited word, and it is what `cue_tokens` now
    /// groups on -- a star is the CTC anchor, not the word boundary.
    fn tokw(i: usize, piece: &str, w: usize, sf: i64, ef: i64) -> TokenAlignment {
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
    fn words_from(groups: &[&[&str]]) -> Vec<TokenAlignment> {
        let mut out = Vec::new();
        for (w, g) in groups.iter().enumerate() {
            for piece in *g {
                if *piece == "<star>" {
                    out.push(tokw(out.len(), piece, w, 10 + 10 * out.len() as i64, 12 + 10 * out.len() as i64));
                } else {
                    out.push(tokw(out.len(), piece, w, 10 + 10 * out.len() as i64, 12 + 10 * out.len() as i64));
                }
            }
        }
        out
    }

    #[test]
    fn spans_tile_like_python() {
        let tokens = vec![tok(0, "a", 10, 12), tok(1, "b", 20, 21)];
        let scores: Vec<f64> = (0..40).map(|t| t as f64).collect();
        let segs = build_spans("ab", &tokens, 40, 50.0, &scores, "char");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "a");
        assert_eq!(segs[0].start, 0.0);
        assert_eq!(segs[0].end, 0.34);
        assert_eq!(segs[1].start, 0.34);
        assert_eq!(segs[1].end, 0.8);
        assert_eq!(segs[0].score, 8.0);
        assert_eq!(segs[1].score, 27.5);
    }

    #[test]
    fn cues_keep_decimal() {
        let pieces = ["达", "到", "2", ".", "5", "。"];
        let tokens: Vec<_> = pieces.iter().enumerate().map(|(i, p)| tokw(i, p, 0, 10 + i as i64, 10 + i as i64)).collect();
        let doc = build_cues(&tokens);
        assert_eq!(doc.cues.len(), 1);
        assert_eq!(doc.cues[0].text, "达到2.5。");
    }

    /// Regression: the walk pairs transcript characters with token pieces, and
    /// the real tokenizer interleaves `<star>` targets the transcript never
    /// contains. Hand-built star-free pieces cannot catch that — they are what
    /// let a desynchronising walk pass as green.
    #[test]
    fn stars_do_not_desync_the_transcript_walk() {
        // "edges" placement: a star at each end of a space-delimited script.
        let en: Vec<String> = ["<star>", "h", "i", "<star>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(token_index("hi there", &en), vec![1, 2, -1, -1, -1, -1, -1, -1]);

        // "segment" placement: a star before every word after the first.
        let cjk: Vec<String> = ["<star>", "你", "<star>", "好", "<star>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(token_index("你好", &cjk), vec![1, 3]);
    }

    #[test]
    fn a_bare_period_is_not_a_cue() {
        // What the real run produced: the sentence carries one syllable that
        // absorbed a long silence, `fix_timestamp` snapped the period onto its
        // end, and the splitter gave the period a span of its own.
        let raw = vec![
            (60.82, 64.32, "아무래도 곡을 잘 아시니까 좀 더 날카롭게 보시겠".to_string()),
            (64.32, 64.46, "네".to_string()),
            (64.46, 72.94, "요".to_string()),
            (72.94, 72.94, ".".to_string()),
        ];
        let out = fold_bare_marks(raw);
        assert_eq!(out.len(), 3, "the lone period must not survive as a cue");
        assert_eq!(out[2].2, "요.");
        assert_eq!((out[2].0, out[2].1), (64.46, 72.94), "the span is untouched");
        for (s, e, _) in &out {
            assert!(e > s, "no cue may collapse to a point");
        }
    }

    #[test]
    fn leading_bare_marks_move_to_the_first_speaking_cue() {
        let raw = vec![
            (0.10, 0.10, "(".to_string()),
            (0.10, 0.20, "안".to_string()),
            (0.20, 0.30, "녕".to_string()),
        ];
        let out = fold_bare_marks(raw);
        assert_eq!(out.len(), 2, "only the bare mark is folded, not the syllables");
        assert_eq!(out[0].0, 0.10, "the leading mark does not shift the start");
        assert_eq!(out[0].2, "(안");
        assert_eq!(out[1].2, "녕");
    }

    #[test]
    fn real_cue_text_is_never_rewritten() {
        let raw = vec![(1.0, 2.0, "hello world".to_string())];
        assert_eq!(fold_bare_marks(raw), vec![(1.0, 2.0, "hello world".to_string())]);
    }

    #[test]
    fn the_granularity_comes_from_whitespace_not_unicode() {
        // Space-delimited: the whitespace split yields words, whatever script.
        assert!(!text_is_unspaced("hello world, this is a test"));
        assert!(!text_is_unspaced("안녕하세요 반갑습니다 저는 학생이에요"));
        // Unspaced: the split yields whole lines.
        assert!(text_is_unspaced("你问我爱你有多深我爱你有几分我的情也真我的爱也真"));
        // The point of the test: a space-delimited Hangul transcript and an
        // unspaced Han one land on opposite sides, which a Unicode-block test
        // cannot do -- both are "CJK" by block.
        assert_eq!(auto_split("안녕하세요 반갑습니다 저는 학생이에요"), "word");
        assert_eq!(auto_split("你问我爱你有多深我爱你有几分"), "char");
        // Empty and single-token inputs must not panic or divide by zero.
        assert_eq!(auto_split(""), "word");
        assert_eq!(auto_split("   \n  "), "word");
        assert_eq!(auto_split("hi"), "word");
    }

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

    #[test]
    fn stars_never_reach_the_cue_text() {
        // The `segment` placement the port picks for Hangul/Chinese puts a star
        // in front of every word. It is the only word-boundary signal left
        // (the tokenizer drops the transcript's spaces), so it has to become
        // the boundary AND stay out of the rendered text.
        let tokens = words_from(&[&["<star>", "你", "好"], &["<star>", "世", "界"]]);
        let words = cue_tokens(&tokens);
        assert!(words.iter().all(|w| w.splittable));
        let text: String = words.iter().map(|w| w.token.as_str()).collect();
        assert!(!text.contains("<star>"), "marker leaked into {text:?}");
        assert_eq!(text, "你好世界");
        assert!(!words[0].space_before, "the leading star is not a space");
        assert!(!words[1].space_before, "no boundary inside a word");
        assert!(words[2].space_before, "the inter-word star is a boundary");
        assert!(!words[3].space_before, "no boundary inside a word");
        // rendered the way join_seg renders a cue
        let mut s = String::new();
        for (i, w) in words.iter().enumerate() {
            if i > 0 && w.space_before {
                s.push(' ');
            }
            s.push_str(&w.token);
        }
        assert_eq!(s, "你好 世界");
    }

    /// A transcript that mixes scripts: the unit is decided per word, so the
    /// Han run breaks between characters and the Latin words stay whole.
    ///
    /// This is the case a single file-wide flag got wrong. Any file containing
    /// one Han character was taken to be Chinese throughout, and the
    /// line-breaker duly cut `alignment` into `alignm` and `ent`.
    #[test]
    fn mixed_script_breaks_between_characters_and_never_inside_a_word() {
        let tokens = words_from(&[&["<star>", "你", "好", "这", "个"], &["<star>", "a", "l", "i", "g", "n"], &["<star>", "工", "具"]]);
        let words = cue_tokens(&tokens);
        let text: String = join_seg(&words);
        assert_eq!(text, "你好这个 align 工具");
        assert!(!text.contains("<star>"));

        let han: Vec<&Tok> = words.iter().filter(|t| t.splittable).collect();
        let latin: Vec<&Tok> = words.iter().filter(|t| !t.splittable).collect();
        assert_eq!(han.len(), 6, "the Han words are one unit per character");
        assert_eq!(latin.len(), 1, "the Latin word is a single unit");
        assert_eq!(latin[0].token, "align");

        // The budget follows the span, not the file: a mostly-Han span is
        // measured in characters, a mostly-Latin one in words.
        let all: Vec<&Tok> = words.iter().collect();
        let (t, lim) = span_budget(&all, 16.0, 88.0, 22.0);
        assert_eq!((t, lim), (22.0, None));
        let only_latin: Vec<&Tok> = words.iter().filter(|t| !t.splittable).collect();
        let (t2, lim2) = span_budget(&only_latin, 16.0, 88.0, 22.0);
        assert_eq!((t2, lim2), (16.0, Some(88.0)));
    }

    /// End to end: the line-breaker must not split a Latin word even when the
    /// budget forces a cut nearby.
    #[test]
    fn cue_cutting_leaves_latin_words_intact() {
        // One very long Latin sentence: the splitter has to break it somewhere,
        // and every break must land between words.
        let sentence = "the quick brown fox jumps over the lazy dog and then it keeps running \
                        for a while until somebody finally calls it back to the yard again";
        let words: Vec<String> = sentence.split_whitespace().map(String::from).collect();
        // the layout the tokenizer actually produces: every word carries a
        // word id, and the star placement is whatever the script gets
        let groups: Vec<Vec<&str>> = vec![vec!["<star>"]]
            .into_iter()
            .chain(words.iter().map(|w| vec!["<star>", w.as_str()]))
            .collect();
        let borrowed: Vec<&[&str]> = groups.iter().map(|g| g.as_slice()).collect();
        let tokens = words_from(&borrowed);
        let chars: usize = words.iter().map(|w| w.chars().count()).sum();
        let doc = build_cues(&tokens);
        assert!(doc.cues.len() > 1, "a long sentence must be cut");
        let mut rejoined = String::new();
        for (i, c) in doc.cues.iter().enumerate() {
            if i > 0 {
                rejoined.push(' ');
            }
            rejoined.push_str(&c.text.replace('\n', " "));
        }
        for w in &words {
            assert!(
                rejoined.contains(w.as_str()),
                "cue text lost the word {w:?}: {rejoined:?}"
            );
        }
        assert!(chars > 0);
    }

    #[test]
    fn leading_and_trailing_stars_stay_out_of_latin_words() {
        // `edges` placement: one star at each end, both must vanish.
        let names = ["<star>", "h", "i", "<star>"];
        let tokens: Vec<TokenAlignment> = names
            .iter()
            .enumerate()
            .map(|(i, p)| tokw(i, p, 0, 10 + 10 * i as i64, 12 + 10 * i as i64))
            .collect();
        let words = cue_tokens(&tokens);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].token, "hi");
        assert!(!words[0].space_before);
        assert!(!words[0].splittable, "a Latin word is one unit");
    }

    #[test]
    fn spans_survive_star_targets() {
        // The full shape: a leading star, letters, an inter-word star, letters,
        // a trailing star -- what `tokenise_with_stars` actually emits.
        let names = ["<star>", "你", "<star>", "好", "<star>"];
        let tokens: Vec<TokenAlignment> = names
            .iter()
            .enumerate()
            .map(|(i, p)| tok(i, p, 10 + 10 * i as i64, 12 + 10 * i as i64))
            .collect();
        let scores: Vec<f64> = (0..80).map(|t| t as f64).collect();
        let segs = build_spans("你好", &tokens, 80, 50.0, &scores, "char");
        assert_eq!(segs.len(), 2, "star targets must not empty the span list");
        assert_eq!(segs[0].text, "你");
        assert_eq!(segs[1].text, "好");
        assert!(segs[0].end <= segs[1].start);
    }
}
