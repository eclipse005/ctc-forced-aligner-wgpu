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
* 📄 JSON / SRT / ASS karaoke output
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
| `--format <json\|srt\|ass>` | `json` is the full alignment (default); `srt` is subtitles; `ass` is karaoke, one sweep per character |
| `--ass-res <WxH>` | `ass` only: the **video's** resolution (default `1920x1080`) — the aligner sees a waveform and cannot know it |
| `--ass-font <name>` | `ass` only: font family (default `Malgun Gothic`); the size is fixed |
| `--output <path>` | Write here (default: print to stdout) |
| `--list-devices` | List the wgpu adapters usable on this machine |

`--window` and `--context` were measured: across their legal range they move no timestamp metric (30/2 against 60/4 on the same material differ inside the noise on start MAE), and 60/4 encodes 47% slower. They are memory knobs, not accuracy knobs.

`json` is three things, all at the top level and none inside another:

| field | what it is |
|---|---|
| `text` | the transcript, exactly as given |
| `tokens` | where each **timed unit** is said: `text`, `start`, `end`, `space_before`, `score`, `inferred` |
| `segments` | the same read a sentence at a time: `text`, `start`, `end` |

A unit is a word, and which kind is decided **per word** with no language table: a whole word where the script spaces its words, a single character where it does not. A Chinese "word" is a whole sentence and the test applies inside it, so a mixed transcript gets both kinds in the same file. Where a script is cut per character, a **mark rides in the character it follows** — `は、` is one unit, not two — because a mark is snapped onto the preceding token's end and occupies no frames of its own, so a row that was only a mark was a point in time rather than a timing. On the 73-minute Japanese transcript that was 2,490 rows of 26,355 saying nothing; 4 remain, the marks the transcript put after a space, which open a word of their own.

Two of the six fields are there so the row cannot lie:

- **`space_before`** is the transcript's own whitespace in front of this unit, and it is what makes the rows renderable back into `text`. The obvious alternative — join the rows with spaces — is what put a space between every character of a Japanese sentence. On that transcript it is set on 1,780 rows, which is exactly the number of spaces in the source.
- **`inferred`** marks a unit containing a character the vocabulary had no target for, so part of its span is the midpoint of what its neighbours leave open rather than something measured. The character is still in `text` and still in the row. **`score` is the mean over the characters that were measured, and `null` when none of them was** — an interpolated character has no score, and a log-probability of 0 is a probability of 1, which would be the best possible confidence said about the one row that has no evidence at all.

Nothing points at anything else. A consumer who wants timings reads `tokens` and stops; one who wants sentences reads `segments` and regroups if their idea of a sentence differs. The subtitle line breaking is this program's opinion and lives in `--format srt` and `--format ass`, not in the alignment — carrying it as data too made the file 39% larger and added copies of every character to keep in step. `<star>` is not in the output at all: it is a DP anchor rather than transcript text, it carries a constant score, and a `<star>` printed into a subtitle is a bug.

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

Characters outside the checkpoint's vocabulary have no CTC target and so never reach the DP, but they are **not** dropped from the output: a forced alignment is a monotone path, so a character written between two placed ones must fall between them, and it is placed at the midpoint of that interval with `inferred: true` to say so. A transcript that mentions a rare han character four times used to ship a subtitle missing all four, and the only trace was a line in a `skipped` list that nobody read. There is no such list now — the character is in `tokens` with the flag set, which is the same information without the risk of it reading as "this was thrown away". (That list was 96% spaces anyway: a space is a tokenizer separator and was never a target to begin with.)

### Deliberate departures from the reference

The algorithm is ported from `MahmoudAshraf97/ctc-forced-aligner`, but that implementation was written for its own MMS checkpoint. The following eight were changed **on measurement, on this checkpoint**. They are not bugs:

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

5. **`json` is three parallel fields, and none of them is a second copy of the line breaking.**
   It used to carry the cues as data too, so a consumer could see the subtitle line breaking without asking for the format — verified identical cue for cue, which is the one thing that argued for it. The cost was larger than the benefit: the file grew 39%, and every character of the transcript was written five times over, so five things had to stay in step. `--format srt` and `--format ass` are one `build_cues()` call either way, so they cannot disagree with each other, and a consumer who wants the cues as data can group `tokens` however it likes.

6. **A word is a whitespace-delimited run of the transcript, punctuation included, and only `word_id` draws its boundary.**
   The reference splits on `text.split()`, so a mark is a character of the word it was written in. Holding marks in a side buffer "so they join the word before them" emptied that buffer on the next letter, and dropped every `、` that was not the last character of its word: **1,116 marks gone** from a 26,552 character Japanese transcript, while the cue view — which has no such buffer — kept every one. The aligner exists to put times on a transcript, not to edit one.
   A `<star>` is skipped, never treated as a boundary: the star that opens a word carries the id of the word *before* it, so a word whose first character had no target has its star in the middle of the word and the ids around it run backwards. `贅沢` came out as `贅 沢` — a space the transcript does not contain. `json` reports `space_before` per word, so `words` can be rendered back into the text it came from.

7. **Characters with no target are interpolated back in, not dropped.**
   See [Timestamp granularity](#timestamp-granularity). The reference drops them and reports them in a `skipped` list; here they get the midpoint of the interval their neighbours leave open and an `inferred` flag on the `tokens` row, so the subtitle says what the speaker said and the row is still there.

8. **`--format ass` exists.**
   The karaoke sweep is the same cue list with a `\k` duration per character, so it cannot disagree with the SRT. A gap in the transcript becomes a `\h` on the character it stands in front of, and none at the start of a line — the line break is already there, exactly as in the SRT. `WrapStyle: 0` and the play resolution come from `--ass-res`: the line breaking is libass's, and inserting `\N` here would hard-code a guess about the player's font fallback and safe area.

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
