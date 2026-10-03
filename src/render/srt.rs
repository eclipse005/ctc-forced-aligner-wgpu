//! SubRip (`.srt`) output.
//!
//! The format carries no styling and no karaoke, so this is the whole of it:
//! an index, a timestamp range, and the cue text. The line breaking is the
//! player's -- an SRT cue has no width budget of its own, so nothing here
//! decides where a line ends.

use crate::views::CueDoc;

/// `H:MM:SS,mmm`, the timestamp format SubRip uses.
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

/// The cue list as a SubRip file.
///
/// This is `views::build_cues` rendered as text, which is why the `cues` field
/// of `json` and `--format srt` can never disagree: they are one computation.
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
    use crate::testutil::words_from;
    use crate::views::build_cues;

    #[test]
    fn an_srt_is_an_index_a_range_and_the_text() {
        let doc = build_cues(&words_from(
            &[&["<star>", "你", "好"], &["<star>", "世", "界"]],
        ));
        let srt = cues_to_srt(&doc);
        assert!(srt.starts_with("1\n00:00:"), "index then a range: {srt}");
        assert!(srt.contains(" --> "), "the range is a SubRip arrow");
        for line in srt.lines().filter(|l| l.contains(" --> ")) {
            let t = line.split(" --> ").collect::<Vec<_>>();
            assert_eq!(t[0].len(), 12, "H:MM:SS,mmm is 12 characters");
            assert!(t[1].len() == 12, "and so is the end");
        }
    }

    #[test]
    fn a_negative_time_does_not_print_as_a_minus() {
        assert_eq!(fmt_ms(-1), "00:00:00,000");
    }
}
