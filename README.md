# CTC Forced Aligner wgpu

**omniASR-CTC forced alignment in Rust with wgpu.**

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of CTC forced alignment for [OmniLingual / omniASR-CTC](https://huggingface.co/aadel4/omniASR-CTC-300M-v2) checkpoints, using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: given audio **and** its transcript, produce the start and end time of every character — **locally and natively**, without Python or vendor-specific GPU runtimes. The alignment is token-identical to the Python reference it was ported from, verified frame by frame.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Cross-platform GPU support
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback
* 📦 Offline local inference
* ⏱️ Character- and word-level start / end timestamps
* 🗣️ Multilingual — whatever the checkpoint's vocabulary covers
* 📄 JSON / spans / SRT / Cues output
* 🧩 CLI + Rust library

### Install

As a Cargo dependency:

```toml
[dependencies]
ctc-forced-aligner-wgpu = { git = "https://github.com/eclipse005/ctc-forced-aligner-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/ctc-forced-aligner-wgpu.git
cd ctc-forced-aligner-wgpu
cargo build --release        # target/release/align
```

| Feature | Description |
|---------|-------------|
| `alloc-stats` | Counts allocations in `alloc_stats::Stats` and reports a per-chunk live / peak figure. Off by default: the counters are three relaxed atomics per allocation, which a production run does not need. Build with `--features alloc-stats` and set `CTC_ALLOC_STATS=1` |

### Model download

Weights are **not** included in this repository. Download the checkpoint (rights remain with the original authors):

```python
from huggingface_hub import snapshot_download
snapshot_download('aadel4/omniASR-CTC-300M-v2',
                  local_dir='models/omniASR-CTC-300M-v2-hf')
```

Point the CLI at it with `--model`, or set `CTC_MODEL_DIR`.

The default storage format is `f32`; `f16` and `bf16` checkpoints are accepted and widened exactly.

### Quick Start

```bash
align --audio speech.wav --text "hello world" --model models/omniASR-CTC-300M-v2-hf
align --audio speech.wav --text transcript.txt --format srt --output out.srt
```

| Option | Description |
|--------|-------------|
| `--audio <file>` | Audio file (WAV) |
| `--text <file/text>` | Transcript: an existing path is read as a file, otherwise the argument is used as text |
| `--model <dir>` | Model directory, or the `CTC_MODEL_DIR` environment variable |
| `--device <spec>` | `auto` (default, CPU when no GPU), `cpu`, `vulkan[:i]`, `dx12[:i]`, `#n`, or a name substring |
| `--window <sec>` | Memory and throughput; **does not move timestamps** (default `30`). Capped near 90 s by the single-forward VRAM limit |
| `--context <sec>` | Encoder overlap on each side of a window (default `2`, floor near 1.3 — it must cover the positional conv) |
| `--format <json\|srt>` | `json` is the full alignment (default); `srt` is subtitles |
| `--output <path>` | Write here (default: print to stdout) |
| `--list-devices` | List the wgpu adapters usable on this machine |

`--window` and `--context` were measured: across their legal range they move no timestamp metric (30/2 against 60/4 on the same material differ inside the noise on start MAE), and 60/4 encodes 47% slower. They are memory knobs, not accuracy knobs.

`json` carries `chars` (per character, with frame indices and confidence), `words`, `segments`, `spans` (silence folded onto neighbours so the timeline is gapless) and `cues` (exactly the line breaking `--format srt` renders). `--format srt` is those cues as text, so the two cannot disagree.

### Library

```rust
use ctc_forced_aligner_wgpu::align_inference::Aligner;
use ctc_forced_aligner_wgpu::gpu::DeviceSelector;

let aligner = Aligner::load_on(std::path::Path::new("model"),
                               DeviceSelector::parse("auto")?)?;
let out = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some(30.0), 2.0)?;
for t in &out.tokens {
    println!("{:.3}s - {:.3}s  {}", t.start, t.end, t.piece);
}
```

`align` takes the window and context lengths in seconds (`None` for the window runs the whole file in one pass). `TokenAlignment` carries `piece`, `start`, `end`, `start_frame`, `end_frame`, `word_id` and a per-frame mean `score`. `align_with_path` additionally returns the per-frame state path, which is what the reference comparison is made against. See `cargo doc` for the full API.

### Audio input

16 kHz mono WAV is recommended — the model's native rate, used without resampling. Any other sample rate in a WAV is converted automatically; convert it yourself first when full control matters:

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

Only WAV is currently supported.

### Timestamp granularity

The model's convolutional frontend subsamples by 320, so **timestamps land on a 20 ms grid** — that is the model's limit, not this implementation's. The frame rate, the subsampling factor and the blank id all come from `config.json` (the product of `conv_stride`, and `pad_token_id`), none of them is a constant.

**What gets aligned is the character. Spaces do not** — a space is a tokenizer separator, and the audio has no such acoustic event. **Punctuation does, but occupies no time**: it is in the vocabulary and a real CTC target, but it is snapped onto the preceding token's end, so `start == end` and it takes zero frames. The model has no opinion about it either — measured confidence runs −12 to −24 — and giving it a duration would be inventing one.

Whether the output is per character or per word is decided **per word**, with no language table: a word whose characters are mostly CJK is split into characters, every other word is one unit the line-breaker cannot cut inside. So Korean and Japanese come out per character, Chinese likewise (a Chinese "word" is a whole sentence, and the test applies inside it), and English, Spanish and French per word.

Characters outside the checkpoint's vocabulary are dropped, and listed in the `skipped` field of `json`.

### Deliberate departures from the reference

The algorithm is ported from `MahmoudAshraf97/ctc-forced-aligner`, but that implementation was written for its own MMS checkpoint. The following five were changed **on measurement, on this checkpoint**. They are not bugs:

1. **One `<star>` per word, not the reference's `edges` placement (one at each end of the file).**
   A star is a DP anchor, not a word marker. This checkpoint's DP is under-constrained wherever the acoustic evidence is weak: two stars let the whole path slide, one per word does not. Over a 180-clip multilingual set — 124 of which pass a FireRedVAD / short-time-energy cross-check, since VAD alone misses most of the speech in some FLEURS English clips and believing it inverts the result:

   | Language | one per word | `edges` |
   |---|---:|---:|
   | Spanish | **−0.1308** | −0.3896 |
   | French | **−0.2605** | −0.5315 |
   | German | **−0.3322** | −0.5347 |
   | English | **−0.5936** | −0.7137 |
   | Japanese | −0.2348 | −0.2346 |
   | Chinese | −0.1993 | −0.1987 |
   | **all 124** | **−0.2570** | −0.4301 |

   The gap lands exactly where the mechanism says it should: large on scripts that space their words, nil on the ones that do not, where a "word" is a whole line and both placements insert about as many stars. The whole-file `log_prob` cannot compare the two — it sums over the path, and two stars is fewer terms — so the per-frame mean is the number quoted.

2. **A token may claim at most 1.0 s of the following silence.**
   The midpoint rule hands a whole pause to the token in front of it, and one syllable was measured swallowing 8.4 s. The bound comes from the same corpus: over 15,722 characters the depth a token's tail reaches past the last detected speech has median 0.00 s, p99 0.00 s and a maximum of 1.57 s, so it is nearly free on read speech, and on material with long pauses it takes the worst end error from 8.4 s to 1.7 s and the end MAE from 896 ms to 486 ms — with **every start boundary bit-identical**, because the bound is one-sided and never reaches a start.
   FireRedASR2 caps by a multiple of the mean duration instead. On this corpus that rule fired on 7.9% of tokens and cut more real speech than it released: a multiple of a duration and a bound in seconds are not the same thing.

3. **The line-breaker groups on `word_id`, not on `<star>`.**
   Recovering word boundaries from stars only works for scripts whose placement puts one in front of every word. English takes `edges`, with two stars in a whole file, so the entire transcript arrived as one cue and every word was cut in half.

4. **The breaking unit is decided per word, not per file.**
   The old code asked once per file whether the text was Chinese; one Han character made the whole transcript Chinese, and `alignment` came out as `alignm` and `ent`.

5. **`--format cues` is now a field of `json`.**
   It was the same `build_cues()` call as `--format srt` with the numbers serialised as JSON, verified identical cue for cue. Two formats could only ever disagree with each other.

### Long audio

Memory is bounded no matter how long the file is. The trellis is held in whichever of three equivalent forms is the narrowest that fits: the gathered `(T, S)` columns, the lm head's `(T, vocab)` output, or the encoder's `(T, hidden)` stream with the head re-run on demand. A one-hour file runs in a few GB rather than growing without limit; the Viterbi is banded to the states each frame can reach, and its backpointers are packed 2 bits per state.

### Accuracy

Measured against the [Buckeye](https://buckeye.shoup.informatics.cmu.edu/) hand-marked corpus (speaker s32), word-level start and end against the manual annotation, via fa-bench's matching rule:

| | MAE | ≤20 ms | ≤50 ms |
|---|---:|---:|---:|
| **This implementation (300M)** | **29.8 ms** | 45.3% | 84.7% |
| MMS-FA (published, fa-bench Track-1) | 30.5 / 31.1 ms | — | — |

The two tower implementations (CPU and GPU) and all three trellis forms are bit-identical on the same input, so the numbers above are a property of the checkpoint, not of the backend.

**The scope of that number is worth stating.** Buckeye is read English with short inter-word gaps. On broadcast material with long pauses, start boundaries stay accurate — 277.5 ms word-start MAE (median 65 ms) on a Korean broadcast, 82.8 ms on the sung Chinese of the same file — but **end boundaries reach into the silence that follows**, because the midpoint rule says so. The bound above in the departures list reduces it; a systematic lateness remains.

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as a unified GPU abstraction.

This makes it possible to build a single Rust-based alignment runtime for different platforms and GPU vendors.

### Project Status

🚧 **Active development**

Performance and hardware compatibility are still being actively optimized and tested across different GPUs.

### Related

* [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) — the model family
* [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) — the Python reference this was ported from
* [fa-bench](https://github.com/olewave/fa-bench) — a forced-alignment benchmark with published numbers
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — speech recognition (transcription)
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) — forced alignment for Qwen3

### License

Apache-2.0.

The algorithm is a port of [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner), whose code is BSD 2-Clause.

This repository is an **independent Rust inference implementation** for loading and running the officially released omniASR-CTC weights — not an official Meta release, and not affiliated with the original authors. Model weights remain under the terms of their respective owners.
