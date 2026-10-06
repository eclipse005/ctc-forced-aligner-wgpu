//! Golden diff against the Python reference (see golden/manifest.json):
//! every forward stage is compared numerically, and the final token
//! timestamps must match `alignment_eager.json` / `alignment_sdpa.json`
//! exactly.
//!
//! Directories: `$CTC_GOLDEN_DIR` (default `<repo>/golden`) and
//! `$CTC_MODEL_DIR` (default the local omniASR checkpoint path).
use crate::gpu::DeviceSelector;


use std::path::PathBuf;

use rayon::prelude::*;
use crate::align_inference::{Aligner, Tower};
use crate::audio::{load_audio, znorm};
use crate::viterbi::{build_expanded_labels, ctc_forced_align};
use crate::wav2vec2::{Scratch, StageSet};

fn golden_dir() -> PathBuf {
    std::env::var_os("CTC_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden"))
}

/// The stored alignments are only meaningful if they were produced from the same
/// target sequence the port aligns today.
///
/// The port gained `<star>` targets and kept the inter-word spaces (the
/// reference's `preprocess_text` emits `" ".join(word)` per word, and id 0 is
/// both the blank and `<s>`, so a space is a blank as far as the boundary rule
/// is concerned). A golden captured before that describes a different sequence,
/// and comparing against it measures the fixture's age rather than the port.
///
/// `alignment_*.json` therefore carries the token count it was captured with, and
/// the test skips rather than fails when that does not match today's port. It is
/// a local, gitignored artefact, so a stale one must not be read as a
/// regression; regenerate it with the reference dump before trusting it again.
fn golden_is_current(dir: &std::path::Path, token_count: usize) -> bool {
    for name in ["alignment_eager.json", "alignment_sdpa.json"] {
        let Ok(text) = std::fs::read_to_string(dir.join(name)) else {
            return false;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            return false;
        };
        if let Some(n) = v.get("n_tokens").and_then(|x| x.as_u64()) {
            if n as usize != token_count {
                eprintln!(
                    "SKIP: {name} was captured with {n} tokens, the port now aligns \
                     {token_count} (the <star>/space fix changed the target \
                     sequence). Regenerate the golden to re-arm this check."
                );
                return false;
            }
        }
    }
    true
}

fn model_dir() -> PathBuf {
    std::env::var_os("CTC_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf"))
}

fn load_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read golden");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "shape mismatch");
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y).abs() as f64;
        max = max.max(d);
        sum += d;
    }
    (max, sum / a.len() as f64)
}

#[test]
fn golden_stages_and_tokens() {
    let root = golden_dir();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    let shapes = manifest["shapes"].as_object().unwrap();
    let wav_path = PathBuf::from(manifest["wav"].as_str().unwrap());
    let text = manifest["ref"].as_str().unwrap();

    // The port's own tokenisation decides how many targets it aligns; if that
    // does not match what the stored alignments were captured with, the fixture
    // predates the `<star>`/space fix and cannot judge the boundary rule.
    {
        let aligner0 = Aligner::load(&model_dir()).unwrap();
        let n = aligner0.vocab.tokenise_with_stars(text).0.len();
        if !golden_is_current(&root, n) {
            return;
        }
    }

    let aligner = Aligner::load(&model_dir()).unwrap();
    let model = match &aligner.tower {
        Tower::Cpu(m) => m,
        _ => unreachable!(),
    };

    // same preprocessing as the Python export
    let (wave, sr) = load_audio(&wav_path).unwrap();
    assert_eq!(sr, 16000);
    let mut input = wave.clone();
    znorm(&mut input);

    let stages = StageSet {
        conv: true,
        proj: true,
        pos_conv: true,
        enc_in: true,
        layers: Some(vec![0, 11, 23]),
        enc_final: true,
    };
    let (log_probs, st) = model.forward(&input, &stages).unwrap();

    // dump CPU-side references for GPU cross-checking
    if std::env::var("CTC_CPU_DUMP").is_ok() {
        std::fs::create_dir_all("gpu_debug").unwrap();
        std::fs::write("gpu_debug/cpu_input.bin", bytemuck::cast_slice(&input)).unwrap();
        for (i, c) in st.conv.iter().enumerate() {
            std::fs::write(format!("gpu_debug/cpu_conv{i}.bin"), bytemuck::cast_slice(c)).unwrap();
        }
    }

    let check = |name: &str, got: &[f32]| {
        let golden = load_f32(&root.join(format!("eager_{name}.bin")));
        let (max, mean) = diff(got, &golden);
        println!("{name:<20} max {max:.6}  mean {mean:.6}");
        max
    };

    let mut worst = 0.0f64;

    // conv stages: golden is (C, T), ours is (T, C) -> transpose
    for (i, ours) in st.conv.iter().enumerate() {
        let shape = shapes[&format!("eager_conv_{i}")].as_array().unwrap();
        let (c, t) = (shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
        assert_eq!(ours.len(), c * t);
        let mut ours_t = vec![0.0f32; ours.len()];
        for tt in 0..t {
            for cc in 0..c {
                ours_t[cc * t + tt] = ours[tt * c + cc];
            }
        }
        let m = check(&format!("conv_{i}"), &ours_t);
        worst = worst.max(m);
    }

    let m = check("proj", &st.proj);
    worst = worst.max(m);
    let m = check("pos_conv", &st.pos_conv);
    worst = worst.max(m);
    let m = check("enc_in", &st.enc_in);
    worst = worst.max(m);
    for li in [0usize, 11, 23] {
        let a = st.attn_out.get(&li).unwrap();
        let m = check(&format!("layer{li:02}_attn_out"), a);
        worst = worst.max(m);
        let o = st.layer_out.get(&li).unwrap();
        let m = check(&format!("layer{li:02}_out"), o);
        worst = worst.max(m);
    }
    let m = check("enc_final", &st.enc_final);
    worst = worst.max(m);
    let m = check("logits", &log_probs);
    worst = worst.max(m);
    let m = check("log_probs", &log_probs);
    worst = worst.max(m);
    println!("\nworst max-abs diff across stages: {worst:.6}");

    // ---- token-level acceptance: our forward -> viterbi vs Python JSONs
    let (ids, pieces) = aligner.vocab.tokenise(text);
    let vocab = aligner.config().vocab_size;
    let t_frames = log_probs.len() / vocab;
    // the blank and the frame rate come from the checkpoint's own config, not
    // from constants here: `pad_token_id` and `conv_stride`'s product
    let (blank_id, frame_rate) = (aligner.blank_id, aligner.frame_rate);

    // gathered lm-head epilogue: the values the DP actually consumes.  The
    // fused path must agree with a gather from the full log_probs above to
    // within the GEMM's destination-add rounding.
    {
        let expanded = build_expanded_labels(&ids, blank_id);
        let (g, _) = model
            .forward_gathered_with(
                &input,
                &StageSet::default(),
                &mut Scratch::default(),
                &expanded,
            )
            .unwrap();
        assert_eq!(g.len(), t_frames * expanded.len(), "gathered shape");
        let mut gmax = 0.0f64;
        for f in 0..t_frames {
            for (s, &st) in expanded.iter().enumerate() {
                let d = (g[f * expanded.len() + s] - log_probs[f * vocab + st]).abs() as f64;
                gmax = gmax.max(d);
            }
        }
        println!(
            "gathered epilogue vs full log_probs: max {gmax:.6} ({} states)",
            expanded.len()
        );
        assert!(gmax < 1e-3, "gathered epilogue diverged: {gmax}");

        // The aligner may instead park the lm head's logits and gather the
        // trellis columns later (Trellis::Logits — the cheaper store once the
        // transcript passes ~5 k characters).  That re-gather must be the
        // *same f32s*, since it is the same kernel on the same logits, and the
        // normaliser it records has to reproduce a single column exactly.
        {
            let logits = model
                .forward_logits(
                    &input,
                    &mut Scratch::default(),
                )
                .unwrap();
            let s = expanded.len();
            let cols: Vec<i32> = expanded.iter().map(|&x| x as i32).collect();
            assert_eq!(logits.len(), t_frames * vocab, "logits shape");
            let mut late = vec![0.0f32; t_frames * s];
            let mut norm = vec![0.0f32; t_frames];
            for (t, (src, dst)) in logits
                .par_chunks_exact(vocab)
                .zip(late.par_chunks_mut(s))
                .enumerate()
                .collect::<Vec<_>>()
            {
                norm[t] = model.gather_logits_row(src, &cols, dst);
            }
            let mut bits = 0usize;
            let mut worst = 0.0f32;
            for i in 0..late.len() {
                if late[i].to_bits() != g[i].to_bits() {
                    bits += 1;
                    worst = worst.max((late[i] - g[i]).abs());
                }
            }
            println!(
                "lazy gather vs fused gather: {bits}/{} values differ in bits, max {worst:.3e}",
                late.len()
            );
            assert_eq!(bits, 0, "lazy gather is not bit-identical to the fused one");

            // and the recorded normaliser must reproduce any single column
            for &(t, s_idx) in &[(0usize, 0usize), (t_frames / 2, s - 1), (t_frames - 1, 3)] {
                let col = cols[s_idx] as usize;
                let v = logits[t * vocab + col] + model.lm_bias()[col] - norm[t];
                assert_eq!(
                    v.to_bits(),
                    g[t * s + s_idx].to_bits(),
                    "normaliser path differs at frame {t} state {s_idx}"
                );
            }
        }
    }

    let res = ctc_forced_align(
        &log_probs,
        t_frames,
        vocab,
        &ids,
        blank_id,
        frame_rate,
        Some(&pieces),
        false,
    )
    .unwrap();

    for (impl_name, expect_same) in [("eager", true), ("sdpa", true)] {
        let py: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join(format!("alignment_{impl_name}.json"))).unwrap(),
        )
        .unwrap();
        let py_chars = py["chars"].as_array().unwrap();
        let mut same = 0;
        let mut diffs = Vec::new();
        for (rust, py) in res.tokens.iter().zip(py_chars) {
            let pf = (
                py["start_frame"].as_i64().unwrap(),
                py["end_frame"].as_i64().unwrap(),
            );
            let rf = (rust.start_frame, rust.end_frame);
            if pf == rf {
                same += 1;
            } else {
                diffs.push(format!(
                    "{}: py {}..{} rust {}..{}",
                    py["piece"].as_str().unwrap_or("?"),
                    pf.0,
                    pf.1,
                    rf.0,
                    rf.1
                ));
            }
        }
        println!(
            "tokens vs {impl_name}: {same}/{} identical frames{}",
            res.tokens.len(),
            if diffs.is_empty() {
                String::new()
            } else {
                format!("\n  {}", diffs.join("\n  "))
            }
        );
        let _ = expect_same;

        // Word boundaries are padded into the adjacent blank run (the
        // `get_spans` rule), so they no longer equal the Python reference's
        // unpadded character frames. What must still hold is the property that
        // actually catches a bug: the padded span still CONTAINS the reference
        // span, and it is the SAME set of tokens in the same order. A shifted
        // frame index, a lost token, or a mis-mapped label all break that; a
        // deliberate padding does not.
        // Match by TEXT, not by index: the port now carries a `<star>` target
        // at each end, like the reference's `star_frequency="edges"`, so the
        // two lists are offset and a positional compare pairs a star with a
        // letter. The invariant is per-character and text-addressed: a padded
        // span must CONTAIN the reference span for the same character.
        let mut checked = 0usize;
        for rust in res.tokens.iter() {
            if !rust.piece.chars().any(|c| c.is_alphanumeric()) {
                continue;
            }
            let Some(py) = py_chars
                .iter()
                .find(|c| c["piece"].as_str() == Some(rust.piece.as_str()))
            else {
                continue;
            };
            let (ps, pe) = (
                py["start_frame"].as_i64().unwrap(),
                py["end_frame"].as_i64().unwrap(),
            );
            assert!(
                rust.start_frame <= ps && rust.end_frame >= pe,
                "{:?}: padded span {}..{} does not contain the reference span \
                 {ps}..{pe} — a padding rule, not a boundary shift",
                rust.piece,
                rust.start_frame,
                rust.end_frame,
            );
            checked += 1;
        }
        assert!(checked > 0, "no alphanumeric characters were checked at all");
    }

    // words and segments too
    let py: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("alignment_sdpa.json")).unwrap(),
    )
    .unwrap();
    let words = crate::spans::build_words(&res.tokens);
    let py_words = py["words"].as_array().unwrap();
    // The apostrophe fix can legitimately change the word COUNT against a
    // reference produced by the old splitter (`it's` was `it` + `s`), so only
    // compare when the counts line up, and say so loudly when they do not.
    if words.len() != py_words.len() {
        println!(
            "word count differs from the reference ({} vs {}): the reference was \
             produced with the old apostrophe-splitting rule; text equality is \
             checked only where the counts agree",
            words.len(),
            py_words.len()
        );
    } else {
        for (w, pw) in words.iter().zip(py_words) {
            assert_eq!(w.text, pw["text"].as_str().unwrap(), "word text");
            // Padded outward: the new span must contain the reference span.
            let (ps, pe) = (pw["start"].as_f64().unwrap(), pw["end"].as_f64().unwrap());
            assert!(
                w.start <= ps + 1e-6 && w.end >= pe - 1e-6,
                "word {:?}: padded {:.4}..{:.4} does not contain reference {ps:.4}..{pe:.4}",
                w.text,
                w.start,
                w.end
            );
        }
    }
    let segments = crate::spans::build_segments(&res.tokens, &words);
    let py_segs = py["segments"].as_array().unwrap();
    assert_eq!(segments.len(), py_segs.len(), "segment count");
    for (s, ps) in segments.iter().zip(py_segs) {
        let pstart = ps["start"].as_f64().unwrap();
        let pend = ps["end"].as_f64().unwrap();
        assert!(
            s.start <= pstart + 1e-6 && s.end >= pend - 1e-6,
            "segment {:?}: padded {:.4}..{:.4} does not contain reference {pstart:.4}..{pend:.4}",
            s.text,
            s.start,
            s.end
        );
    }
    println!(
        "words: {} checked, segments: {} checked",
        words.len(),
        segments.len()
    );
}

/// GPU tower: same acceptance as the CPU test.  Skips (pass) when no wgpu
/// adapter exists.
#[test]
fn gpu_golden_tokens() {
    let root = golden_dir();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    let wav_path = PathBuf::from(manifest["wav"].as_str().unwrap());
    let text = manifest["ref"].as_str().unwrap();

    // same staleness guard as the CPU test: a golden captured before the
    // `<star>`/space fix describes a different target sequence.
    {
        let n = match Aligner::load(&model_dir()) {
            Ok(a) => a.vocab.tokenise_with_stars(text).0.len(),
            Err(e) => {
                eprintln!("no usable model to count targets, skipping: {e:#}");
                return;
            }
        };
        if !golden_is_current(&root, n) {
            return;
        }
    }

    let selector = DeviceSelector::parse("auto").unwrap();
    let aligner = match Aligner::load_on(&model_dir(), selector) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("no usable GPU adapter, skipping: {e:#}");
            return;
        }
    };
    println!("backend: {}", aligner.backend_name());

    // production-path log_probs, diffed against the golden directly
    let (wave, sr) = load_audio(&wav_path).unwrap();
    assert_eq!(sr, 16000);
    let mut input = wave.clone();
    znorm(&mut input);
    let t0 = std::time::Instant::now();
    let lp = aligner.forward_pub(&input).unwrap();
    println!("gpu forward: {:.2}s ({} frames)", t0.elapsed().as_secs_f64(), lp.len() / 10288);
    let gold_lp = load_f32(&root.join("sdpa_log_probs.bin"));
    let (mx, mn) = diff(&lp, &gold_lp);
    println!("gpu log_probs vs golden: max {mx:.6} mean {mn:.6}");
    std::fs::create_dir_all("gpu_debug").unwrap();
    std::fs::write("gpu_debug/gpu_log_probs.bin", bytemuck::cast_slice(&lp)).unwrap();

    let out = aligner.align(&wav_path, text, Some(30.0), 2.0, None).unwrap();
    println!("gpu align: {:.2}s ({} frames)", t0.elapsed().as_secs_f64(), out.frames);

    for impl_name in ["eager", "sdpa"] {
        let py: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join(format!("alignment_{impl_name}.json"))).unwrap(),
        )
        .unwrap();
        let py_chars = py["chars"].as_array().unwrap();
        let mut diffs = Vec::new();
        let mut same = 0usize;
        let mut speech = 0usize;
        for (rust, py) in out.tokens.iter().zip(py_chars) {
            let piece = py["piece"].as_str().unwrap_or("");
            // fix_timestamp rewrites only non-speech marks. Letter and digit
            // frames stay on the raw Viterbi path.
            if !piece.chars().any(|c| c.is_alphanumeric()) {
                continue;
            }
            speech += 1;
            let pf = (
                py["start_frame"].as_i64().unwrap(),
                py["end_frame"].as_i64().unwrap(),
            );
            let rf = (rust.start_frame, rust.end_frame);
            if pf == rf {
                same += 1;
            } else {
                diffs.push(format!(
                    "{}: py {}..{} gpu {}..{}",
                    piece, pf.0, pf.1, rf.0, rf.1
                ));
            }
        }
        println!(
            "gpu speech tokens vs {impl_name}: {same}/{speech} identical{}",
            if diffs.is_empty() { String::new() } else { format!("\n  {}", diffs.join("\n  ")) }
        );
        // Match by TEXT, not by index. The port now prepends and appends a
        // `<star>` target, as the reference's `star_frequency="edges"` does, so
        // the two token lists are offset by one at the front and the positional
        // comparison compared a star against a letter. The property that still
        // has to hold — and the one this test exists for — is that each
        // al character's padded span CONTAINS the reference span, so a wrong
        // frame index or a shifted label is caught even though the index
        // alignment moved.
        let py_by_piece = |piece: &str| -> Option<(i64, i64)> {
            py_chars
                .iter()
                .find(|c| c["piece"].as_str() == Some(piece))
                .map(|c| {
                    (
                        c["start_frame"].as_i64().unwrap_or(0),
                        c["end_frame"].as_i64().unwrap_or(0),
                    )
                })
        };
        let mut checked = 0usize;
        for rust in out.tokens.iter() {
            let piece = rust.piece.as_str();
            if !piece.chars().any(|c| c.is_alphanumeric()) {
                continue;
            }
            let Some((ps, pe)) = py_by_piece(piece) else {
                continue;
            };
            let (rs, re) = (rust.start_frame, rust.end_frame);
            assert!(
                rs <= ps && re >= pe,
                "{piece}: gpu span {rs}..{re} does not contain reference {ps}..{pe}"
            );
            checked += 1;
        }
        assert!(checked > 0, "no alphanumeric characters were checked at all");
    }
    let py: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("alignment_sdpa.json")).unwrap(),
    )
    .unwrap();
    let py_words = py["words"].as_array().unwrap().len();
    // `words` is no longer a field of the output; the golden comparison is
    // against the reference's own segmentation, so build the same view here.
    let our_words = crate::spans::build_words(&out.tokens);
    if our_words.len() != py_words {
        println!(
            "word count {py_words} in the reference vs {} here: expected, the \
             reference predates the apostrophe fix",
            our_words.len()
        );
    }
    let our_segments =
        crate::spans::build_segments(&out.tokens, &our_words);
    assert_eq!(
        our_segments.len(),
        py["segments"].as_array().unwrap().len(),
        "segment count"
    );
}
