//! Derived views on one character alignment.
//!
//! `spans` tiles the timeline: silence is split at the blank-run midpoint, the
//! first span starts at 0, and the last span ends at the audio duration.
//! `cues` are subtitle lines whose times stay on the spoken characters
//! (CrisperWhisper / OneAsr standard cost model).

use crate::spans::PUNCT;
// The CJK ranges come from the tokenizer, so the alignment and the rendering
// can never disagree about which script a text is in. They used to be spelled
// out twice, and the two copies had drifted apart.
use crate::vocab::is_cjk;
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

pub fn skipped_chars(text: &str, pieces: &[String]) -> Vec<String> {
    let mut ti = 0usize;
    let mut skipped = Vec::new();
    for ch in text.chars() {
        ti = skip_stars(pieces, ti);
        if ti < pieces.len() && piece_is_char(&pieces[ti], ch) {
            ti += 1;
        } else {
            skipped.push(ch.to_string());
        }
    }
    skipped
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
        let letters = buf.iter().filter(|t| !t.piece.chars().all(|c| c.is_whitespace())).count();
        let cjk = letters > 0
            && buf.iter().filter(|t| t.piece.chars().any(is_cjk)).count() * 2 > letters;
        let (start, end) = (buf[0].start, buf[buf.len() - 1].end);
        let space_before = *pending && !out.is_empty();
        if cjk {
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
        if t.piece == "<star>" || t.piece.trim().is_empty() {
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

    #[test]
    fn skipped_reports_missing_char() {
        let pieces = vec!["精".to_string(), "，".to_string()];
        assert_eq!(skipped_chars("精悍，", &pieces), vec!["悍".to_string()]);
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
        assert_eq!(
            skipped_chars("hi there", &en),
            [" ", "t", "h", "e", "r", "e"].map(String::from)
        );
        assert_eq!(token_index("hi there", &en), vec![1, 2, -1, -1, -1, -1, -1, -1]);

        // "segment" placement: a star before every word after the first.
        let cjk: Vec<String> = ["<star>", "你", "<star>", "好", "<star>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(skipped_chars("你好", &cjk), Vec::<String>::new());
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
