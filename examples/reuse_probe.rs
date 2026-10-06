//! 探针：同一 Aligner 实例连续 align 两个文件，第二个文件的结果
//! 是否与"新实例 align 同一文件"一致？（管线按块复用实例的前提检验）
use std::path::Path;

use ctc_forced_aligner_wgpu::{Aligner, DeviceSelector};

fn summarize(out: &ctc_forced_aligner_wgpu::AlignOutput) -> (usize, f64, f64, f64) {
    let n = out.tokens.len();
    let first = out.tokens.first().map(|t| t.start).unwrap_or(0.0);
    let last = out.tokens.last().map(|t| t.end).unwrap_or(0.0);
    let mid = out.tokens.get(n / 2).map(|t| t.start).unwrap_or(0.0);
    (n, first, mid, last)
}

fn main() -> anyhow::Result<()> {
    let model = Path::new("D:/omnilingual-asr/models/omniASR-CTC-300M-v2-hf");
    let chunk1 = Path::new("D:/OneAsr/tmp/segtest/out/_ko_chunk1.wav");
    let chunk2 = Path::new("D:/OneAsr/tmp/segtest/out/_ko_chunk2.wav");
    let text1 = std::fs::read_to_string("D:/OneAsr/tmp/segtest/out/_ko_c1_asr.txt")?;
    let text2 = std::fs::read_to_string("D:/OneAsr/tmp/segtest/out/_ko_c2_asr.txt")?;

    // A) 复用实例：先 chunk1，再 chunk2
    let shared = Aligner::load_on(model, DeviceSelector::parse("nvidia")?)?;
    let a1 = shared.align(chunk1, &text1, Some(30.0), 2.0, None)?;
    let a2 = shared.align(chunk2, &text2, Some(30.0), 2.0, None)?;
    println!("reuse  chunk1: {:?}", summarize(&a1));
    println!("reuse  chunk2: {:?}", summarize(&a2));

    // B) 全新实例：只 align chunk2
    let fresh = Aligner::load_on(model, DeviceSelector::parse("nvidia")?)?;
    let b2 = fresh.align(chunk2, &text2, Some(30.0), 2.0, None)?;
    println!("fresh  chunk2: {:?}", summarize(&b2));

    let same = a2.tokens.len() == b2.tokens.len()
        && a2
            .tokens
            .iter()
            .zip(b2.tokens.iter())
            .all(|(x, y)| x.start == y.start && x.end == y.end);
    println!("reuse-chunk2 == fresh-chunk2 ? {same}");
    anyhow::ensure!(same, "a reused Aligner must align file 2 exactly like a fresh one");
    Ok(())
}
