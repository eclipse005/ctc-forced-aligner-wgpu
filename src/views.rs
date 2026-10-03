//! Derived views on one character alignment.
//!
//! `spans` tiles the timeline: silence is split at the blank-run midpoint, the
//! first span starts at 0, and the last span ends at the audio duration.
//! `cues` are subtitle lines whose times stay on the spoken characters
//! (CrisperWhisper / OneAsr standard cost model).

use crate::spans::{splits_between_characters, PUNCT};
use crate::viterbi::TokenAlignment;


#[derive(Clone, Debug)]
pub struct CueOut {
    pub index: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}












// --- cues -----------------------------------------------------------------

pub(crate) struct Tok {
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
pub(crate) fn cue_tokens(tokens: &[TokenAlignment]) -> Vec<Tok> {
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{tokw, words_from};


    #[test]
    fn cues_keep_decimal() {
        let pieces = ["达", "到", "2", ".", "5", "。"];
        let tokens: Vec<_> = pieces.iter().enumerate().map(|(i, p)| tokw(i, p, 0, 10 + i as i64, 10 + i as i64)).collect();
        let doc = build_cues(&tokens);
        assert_eq!(doc.cues.len(), 1);
        assert_eq!(doc.cues[0].text, "达到2.5。");
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

}
