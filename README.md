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
| `--window <sec>` | Windowed encoding in seconds (default `30`); `0` runs the whole file in one forward pass |
| `--context <sec>` | Overlap carried on each side of a window (default `2`) |
| `--format <json\|spans\|srt\|cues>` | `json` is the tight character alignment (default); `spans` tiles silence onto neighbouring segments; `srt` / `cues` are subtitle breaks |
| `--split <word\|char\|sentence>` | `spans` only; defaults to `char` for mostly-CJK text |
| `--preset <short\|standard\|loose>` | Subtitle preset for `srt` / `cues` (default `standard`) |
| `--merge-threshold <sec>` | `spans`: snap a gap shorter than this (default `0`) |
| `--output <path>` | Write here (default: print to stdout). `spans` also writes a `.txt` sidecar |
| `--list-devices` | List the wgpu adapters usable on this machine |

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

The model's convolutional frontend subsamples by 320, so **timestamps land on a 20 ms grid** — that is the model's limit, not this implementation's. Tokenization is whitespace-split for word-delimited scripts and per-character for CJK; characters outside the checkpoint's vocabulary are dropped.

### Long audio

Memory is bounded no matter how long the file is. The trellis is held in whichever of three equivalent forms is the narrowest that fits: the gathered `(T, S)` columns, the lm head's `(T, vocab)` output, or the encoder's `(T, hidden)` stream with the head re-run on demand. A one-hour file runs in a few GB rather than growing without limit; the Viterbi is banded to the states each frame can reach, and its backpointers are packed 2 bits per state.

### Accuracy

Measured against the [Buckeye](https://buckeye.shoup.informatics.cmu.edu/) hand-marked corpus (speaker s32), word-level start and end against the manual annotation, via fa-bench's matching rule:

| | MAE | ≤20 ms | ≤50 ms |
|---|---:|---:|---:|
| **This implementation (300M)** | **29.8 ms** | 45.3% | 84.7% |
| MMS-FA (published, fa-bench Track-1) | 30.5 / 31.1 ms | — | — |

The two tower implementations (CPU and GPU) and all three trellis forms are bit-identical on the same input, so the numbers above are a property of the checkpoint, not of the backend.

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
