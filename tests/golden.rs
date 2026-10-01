//! Golden diff against the Python reference (see golden/manifest.json):
//! every forward stage is compared numerically, and the final token
//! timestamps must match `alignment_eager.json` / `alignment_sdpa.json`
//! exactly.

use std::path::PathBuf;

use ctc_forced_aligner_wgpu::align_inference::{Aligner, BLANK_ID, FRAME_RATE};
use ctc_forced_aligner_wgpu::audio::{load_audio, znorm};
use ctc_forced_aligner_wgpu::viterbi::ctc_forced_align;
use ctc_forced_aligner_wgpu::wav2vec2::StageSet;

const GOLDEN: &str = r"D:\ctc-forced-aligner-wgpu\golden";
const MODEL: &str = r"D:\omnilingual-asr\models\omniASR-CTC-300M-v2-hf";

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
    let root = PathBuf::from(GOLDEN);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    let shapes = manifest["shapes"].as_object().unwrap();
    let wav_path = PathBuf::from(manifest["wav"].as_str().unwrap());
    let text = manifest["ref"].as_str().unwrap();

    let aligner = Aligner::load(std::path::Path::new(MODEL)).unwrap();
    let model = match &aligner.tower {
        ctc_forced_aligner_wgpu::align_inference::Tower::Cpu(m) => m,
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
    let res = ctc_forced_align(
        &log_probs,
        t_frames,
        vocab,
        &ids,
        BLANK_ID,
        FRAME_RATE,
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
        assert_eq!(
            same,
            res.tokens.len(),
            "token timestamps must match the Python reference exactly"
        );
    }

    // words and segments too
    let py: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("alignment_sdpa.json")).unwrap(),
    )
    .unwrap();
    let words = ctc_forced_aligner_wgpu::spans::build_words(&res.tokens);
    let py_words = py["words"].as_array().unwrap();
    assert_eq!(words.len(), py_words.len(), "word count");
    for (w, pw) in words.iter().zip(py_words) {
        assert_eq!(w.text, pw["text"].as_str().unwrap(), "word text");
        assert_eq!(
            (w.start * 1e4).round() / 1e4,
            pw["start"].as_f64().unwrap(),
            "word start vs python"
        );
        assert_eq!((w.end * 1e4).round() / 1e4, pw["end"].as_f64().unwrap(), "word end");
    }
    let segments = ctc_forced_aligner_wgpu::spans::build_segments(&res.tokens, &words);
    let py_segs = py["segments"].as_array().unwrap();
    assert_eq!(segments.len(), py_segs.len(), "segment count");
    for (s, ps) in segments.iter().zip(py_segs) {
        assert_eq!((s.start * 1e4).round() / 1e4, ps["start"].as_f64().unwrap(), "seg start");
        assert_eq!((s.end * 1e4).round() / 1e4, ps["end"].as_f64().unwrap(), "seg end");
    }
    println!("words: {} identical, segments: {} identical", words.len(), segments.len());
}

/// GPU tower: same acceptance as the CPU test.  Skips (pass) when no wgpu
/// adapter exists.
#[test]
fn gpu_golden_tokens() {
    let root = PathBuf::from(GOLDEN);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    let wav_path = PathBuf::from(manifest["wav"].as_str().unwrap());
    let text = manifest["ref"].as_str().unwrap();

    let selector = ctc_forced_aligner_wgpu::DeviceSelector::parse("auto").unwrap();
    let aligner = match ctc_forced_aligner_wgpu::Aligner::load_on(std::path::Path::new(MODEL), selector)
    {
        Ok(a) => a,
        Err(e) => {
            eprintln!("no usable GPU adapter, skipping: {e:#}");
            return;
        }
    };
    println!("backend: {}", aligner.backend_name());

    // production-path log_probs, diffed against the golden directly
    let (wave, sr) = ctc_forced_aligner_wgpu::audio::load_audio(&wav_path).unwrap();
    assert_eq!(sr, 16000);
    let mut input = wave.clone();
    ctc_forced_aligner_wgpu::audio::znorm(&mut input);
    let t0 = std::time::Instant::now();
    let lp = aligner.forward_pub(&input).unwrap();
    println!("gpu forward: {:.2}s ({} frames)", t0.elapsed().as_secs_f64(), lp.len() / 10288);
    let gold_lp = load_f32(&root.join("sdpa_log_probs.bin"));
    let (mx, mn) = diff(&lp, &gold_lp);
    println!("gpu log_probs vs golden: max {mx:.6} mean {mn:.6}");
    std::fs::write("gpu_debug/gpu_log_probs.bin", bytemuck::cast_slice(&lp)).unwrap();

    let out = aligner.align(&wav_path, text, Some(30.0), 2.0).unwrap();
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
        for (rust, py) in out.chars.iter().zip(py_chars) {
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
            let rf = (rust["start_frame"].as_i64().unwrap(), rust["end_frame"].as_i64().unwrap());
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
        assert_eq!(same, speech, "GPU speech tokens must match Python exactly");
    }
    let py: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("alignment_sdpa.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(out.words.len(), py["words"].as_array().unwrap().len(), "word count");
    assert_eq!(out.segments.len(), py["segments"].as_array().unwrap().len(), "segment count");
}
