//! Derived views on one character alignment.
//!
//! `spans` tiles the timeline: silence is split at the blank-run midpoint, the
//! first span starts at 0, and the last span ends at the audio duration.
//! `cues` are subtitle lines whose times stay on the spoken characters
//! (CrisperWhisper / OneAsr standard cost model).

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

pub fn skipped_chars(text: &str, pieces: &[String]) -> Vec<String> {
    let mut ti = 0usize;
    let mut skipped = Vec::new();
    for ch in text.chars() {
        if ti < pieces.len() && pieces[ti].chars().next() == Some(ch) && pieces[ti].chars().nth(1).is_none()
        {
            ti += 1;
        } else {
            skipped.push(ch.to_string());
        }
    }
    skipped
}

pub fn resolve_split(text: &str, split: Option<&str>) -> String {
    match split {
        None | Some("auto") => {
            if is_mostly_cjk(text) {
                "char".to_string()
            } else {
                "word".to_string()
            }
        }
        Some(s) if s == "word" || s == "char" || s == "sentence" => s.to_string(),
        Some(s) => panic!("split must be word|char|sentence, got {s}"),
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x3040..=0x30FF | 0xAC00..=0xD7AF)
}

fn is_mostly_cjk(text: &str) -> bool {
    let cjk = text.chars().filter(|c| is_cjk(*c)).count();
    let letters = text.chars().filter(|c| c.is_alphabetic()).count().max(1);
    cjk > 0 && cjk * 10 > letters * 3
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
        if ti < pieces.len() && pieces[ti].chars().next() == Some(ch) && pieces[ti].chars().nth(1).is_none() {
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

pub fn build_spans(
    text: &str,
    tokens: &[TokenAlignment],
    frames: usize,
    frame_rate: f64,
    frame_scores: &[f64],
    split: &str,
    merge_threshold: f64,
) -> Vec<SpanOut> {
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
    for i in 0..segments.len().saturating_sub(1) {
        if segments[i + 1].start - segments[i].end < merge_threshold {
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

fn cue_tokens(tokens: &[TokenAlignment]) -> (Vec<Tok>, bool) {
    let cjk = tokens.iter().any(|t| t.piece.chars().any(is_cjk));
    let mut out = Vec::new();
    if cjk {
        let mut pending = false;
        for t in tokens {
            if t.piece.trim().is_empty() {
                pending = true;
                continue;
            }
            out.push(Tok {
                start: t.start,
                end: t.end,
                token: t.piece.clone(),
                space_before: pending && !out.is_empty(),
            });
            pending = false;
        }
        return (merge_decimals(out), true);
    }
    let mut buf: Vec<&TokenAlignment> = Vec::new();
    let mut pending = false;
    let flush = |buf: &mut Vec<&TokenAlignment>, pending: &mut bool, out: &mut Vec<Tok>| {
        if buf.is_empty() {
            return;
        }
        out.push(Tok {
            start: buf[0].start,
            end: buf[buf.len() - 1].end,
            token: buf.iter().map(|t| t.piece.as_str()).collect(),
            space_before: *pending && !out.is_empty(),
        });
        buf.clear();
        *pending = false;
    };
    for t in tokens {
        if t.piece.trim().is_empty() {
            flush(&mut buf, &mut pending, &mut out);
            pending = true;
            continue;
        }
        buf.push(t);
    }
    flush(&mut buf, &mut pending, &mut out);
    (out, false)
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
            });
            i += 3;
            continue;
        }
        out.push(Tok {
            start: tokens[i].start,
            end: tokens[i].end,
            token: tokens[i].token.clone(),
            space_before: tokens[i].space_before,
        });
        i += 1;
    }
    out
}

fn sentence_end(words: &[Tok], i: usize) -> bool {
    let tok = &words[i].token;
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

fn units(tok: &str, cjk: bool) -> f64 {
    let s = strip_tok(tok);
    if s.is_empty() {
        0.0
    } else if cjk {
        s.chars().count() as f64
    } else {
        1.0
    }
}

fn dchars(words: &[Tok], a: usize, b: usize) -> usize {
    words[a..=b].iter().map(|w| w.token.chars().count()).sum()
}

fn dur(words: &[Tok], a: usize, b: usize) -> f64 {
    words[b].end - words[a].start
}

fn dp_split(words: &[Tok], target: f64, char_limit: Option<f64>, cjk: bool) -> Vec<usize> {
    let n = words.len();
    if n < 2 {
        return Vec::new();
    }
    let mut pre = vec![0.0; n + 1];
    for (k, w) in words.iter().enumerate() {
        pre[k + 1] = pre[k] + units(&w.token, cjk);
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
            let better = if cjk { c < dp[i] } else { c <= dp[i] };
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

fn join_seg(seg: &[Tok], cjk: bool) -> String {
    if !cjk {
        return seg.iter().map(|t| t.token.as_str()).collect::<Vec<_>>().join(" ");
    }
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
    pub preset: String,
    pub script: String,
    pub cues: Vec<CueOut>,
}

pub fn build_cues(tokens: &[TokenAlignment], preset: &str) -> CueDoc {
    let (latin_words, latin_chars, cjk_chars) = match preset {
        "short" => (12.0, 66.0, 16.0),
        "standard" => (16.0, 88.0, 22.0),
        "loose" => (20.0, 110.0, 28.0),
        other => panic!("preset must be short|standard|loose, got {other}"),
    };
    let (words, cjk) = cue_tokens(tokens);
    let target = if cjk { cjk_chars } else { latin_words };
    let char_limit = if cjk { None } else { Some(latin_chars) };
    let mut spans: Vec<Vec<Tok>> = Vec::new();
    let mut cur = Vec::new();
    for i in 0..words.len() {
        let end = sentence_end(&words, i);
        cur.push(Tok {
            start: words[i].start,
            end: words[i].end,
            token: words[i].token.clone(),
            space_before: words[i].space_before,
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
        let cuts = dp_split(span, target, char_limit, cjk);
        let mut bounds = vec![0];
        bounds.extend(cuts.iter().copied());
        bounds.push(span.len());
        for w in bounds.windows(2) {
            let seg = &span[w[0]..w[1]];
            if seg.is_empty() {
                continue;
            }
            let text = join_seg(seg, cjk);
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            raw.push((seg[0].start, seg[seg.len() - 1].end, wrap_line(text)));
        }
    }
    raw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)));
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
        preset: preset.to_string(),
        script: if cjk { "cjk" } else { "latin" }.to_string(),
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
        TokenAlignment {
            index: i,
            token_id: 1,
            word_id: i,
            piece: piece.to_string(),
            start: sf as f64 / 50.0,
            end: (ef + 1) as f64 / 50.0,
            start_frame: sf,
            end_frame: ef,
            score: -0.2,
        }
    }

    #[test]
    fn spans_tile_like_python() {
        let tokens = vec![tok(0, "a", 10, 12), tok(1, "b", 20, 21)];
        let scores: Vec<f64> = (0..40).map(|t| t as f64).collect();
        let segs = build_spans("ab", &tokens, 40, 50.0, &scores, "char", 0.0);
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
        let tokens: Vec<_> = pieces.iter().enumerate().map(|(i, p)| tok(i, p, 10 + i as i64, 10 + i as i64)).collect();
        let doc = build_cues(&tokens, "standard");
        assert_eq!(doc.script, "cjk");
        assert_eq!(doc.cues.len(), 1);
        assert_eq!(doc.cues[0].text, "达到2.5。");
    }

    #[test]
    fn skipped_reports_missing_char() {
        let pieces = vec!["精".to_string(), "，".to_string()];
        assert_eq!(skipped_chars("精悍，", &pieces), vec!["悍".to_string()]);
    }
}
