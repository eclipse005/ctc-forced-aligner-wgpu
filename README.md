# CTC Forced Aligner wgpu

**omniASR-CTC forced alignment in Rust with wgpu.**

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of CTC forced alignment for [OmniLingual / omniASR-CTC](https://huggingface.co/aadel4/omniASR-CTC-300M-v2) checkpoints, using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: given audio **and** its transcript, produce the start and end time of every token — **locally and natively**, without Python or vendor-specific GPU runtimes. The Viterbi and the boundary rules are ported from the Python reference and verified against it frame by frame; where this differs on purpose, it says so and says why.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Cross-platform GPU support
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback
* 📦 Offline local inference
* ⏱️ Per-token and per-sentence start / end timestamps
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
| `--device <spec>` | `auto` (default: discrete GPU, then integrated, else CPU), `cpu`, `gpu` (discrete then integrated, error when none opens), `vulkan[:i]`, `dx12[:i]`, `#n`, or a name substring |
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

A unit is a word, and which kind is decided **per run of characters** with no language table: a whole word where the script spaces its words, a single character where it does not. A Chinese "word" is a whole sentence and the test applies inside it, so a mixed transcript gets both kinds in the same file. The run is the unit of the decision, not the word: `word_id` marks the transcript's whitespace, so `原版Whisper` — a word with no space in it anywhere — arrives as one group of nine characters, and asked once, "mostly Han?" said yes and reported all nine separately, which renders as `原 版 W h i s p e r`. Decided per run it is `原` `版` `Whisper`. Where a script is cut per character, a **mark rides in the character it follows** — `は、` is one unit, not two — because a mark is snapped onto the preceding token's end and occupies no frames of its own, so a row that was only a mark was a point in time rather than a timing. On the 73-minute Japanese transcript that was 2,490 rows of 26,355 saying nothing; 4 remain, the marks the transcript put after a space, which open a word of their own.

Two of the six fields are there so the row cannot lie:

- **`space_before`** is the transcript's own whitespace in front of this unit, and it is what makes the rows renderable back into `text`. The obvious alternative — join the rows with spaces — is what put a space between every character of a Japanese sentence. On that transcript it is set on 1,780 rows, which is exactly the number of spaces in the source.
- **`inferred`** marks a unit containing a character the vocabulary had no target for, so part of its span is the midpoint of what its neighbours leave open rather than something measured. The character is still in `text` and still in the row. **`score` is the mean over the characters that were measured, and `null` when none of them was** — an interpolated character has no score, and a log-probability of 0 is a probability of 1, which would be the best possible confidence said about the one row that has no evidence at all.

Nothing points at anything else. A consumer who wants timings reads `tokens` and stops; one who wants sentences reads `segments` and regroups if their idea of a sentence differs. The subtitle line breaking is this program's opinion and lives in `--format srt` and `--format ass`, not in the alignment — carrying it as data too made the file 39% larger and added copies of every character to keep in step. `<star>` is not in the output at all: it is a DP anchor rather than transcript text, it carries a constant score, and a `<star>` printed into a subtitle is a bug.

### Library

```rust
use ctc_forced_aligner_wgpu::{Aligner, DeviceSelector};

let aligner = Aligner::load_on(std::path::Path::new("model"),
                               DeviceSelector::parse("auto")?)?;
let out = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some(30.0), 2.0, None)?;
for w in &out.words {
    println!("{:.3}s - {:.3}s  {}", w.start, w.end, w.text);
}
```

That is the whole of the API: `Aligner` to load and run, `AlignOutput` to read, `WordSpan` for one timed unit, `TokenAlignment` for one CTC target, `DeviceSelector` to pick a device, `list_targets` to list them. Subtitles are `views::build_cues` rendered by `render::srt` or `render::ass`. **Everything else is `pub(crate)`** — the model, the Viterbi, the boundary rules, the sentence grouping, all of it, because none of it is a promise to anyone but this crate. The unit grouping is now a promise, which is why `words` is public: it is the one thing every consumer has to agree on, and agreeing on it twice is how `W h i s p e r` happens.

`align` takes the window and context lengths in seconds (`None` for the window runs the whole file in one pass). Its last argument is an optional `AlignProgress` sink — `&(dyn Fn(Progress) + Send + Sync)` — fired every time a checkpoint is **finished**, counted across the whole run: the encoder's windows, the Viterbi, the traceback, the score replay, the timeline. `Progress` carries `stage`, `done`, `total` and `pct()`, so a caller that wants one number reads `pct()` and a caller that wants a labelled bar reads `stage`. The unit is **one window of the file**: countable before a sample is read, so the denominator is never a guess. A window counts when its encoder rows and DP choices are back in RAM, not when the work is queued — otherwise the windows still in flight on the device are exactly the ones the bar has already claimed, and it sits at 100% with a fifth of the run left to do. Pass `None` and no callback runs. `TokenAlignment` carries `piece`, `start`, `end`, `start_frame`, `end_frame`, `word_id` and a per-frame mean `score`. `align_with_path` additionally returns the per-frame state path, which is what the reference comparison is made against. See `cargo doc` for the full API.

### How it is put together

Four layers, each depending only on the ones above it:

| layer | modules | what it knows |
|---|---|---|
| base | `config` `weights` `vocab` `simd` `shaders` `gpu` `resample_sinc` `audio` | checkpoints, kernels, devices — no idea what an alignment is |
| model | `wav2vec2` `wav2vec2_gpu` | the encoder tower and its CPU twin: audio in, per-frame scores out |
| align | `viterbi` `timeline` | the path, and **the one place a timestamp is decided** |
| text | `spans` `views` | the transcript's own words and sentences, and where a subtitle line breaks |

`timeline` is the part worth knowing: after the Viterbi picks frames, three rules turn that into the spans a subtitle shows — a boundary sits in the middle of a pause short enough to be prosody and on the word's own evidence once the run is long enough to be silence, a mark has no sound so it becomes a point on the sound it touches — the one before it, or, at a stream opening, the one after — and a character the vocabulary had no target for lies between its placed neighbours. They used to live in three files with the order between them held up by comments; they are now one list, in order, each with the measurement that chose it. Nothing downstream of the Viterbi may write a `start` or an `end`, which is what keeps `--format srt`, `--format ass` and the JSON from ever disagreeing.

### Audio input

16 kHz mono WAV is recommended — the model's native rate, used without resampling. Any other sample rate in a WAV is converted automatically; convert it yourself first when full control matters:

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

Only WAV is currently supported.

### Timestamp granularity

The model's convolutional frontend subsamples by 320, so **timestamps land on a 20 ms grid** — that is the model's limit, not this implementation's. The frame rate, the subsampling factor and the blank id all come from `config.json` (the product of `conv_stride`, and `pad_token_id`), none of them is a constant.

**What gets aligned is the character. Spaces do not** — a space is a tokenizer separator, and the audio has no such acoustic event. **Punctuation does, but occupies no time**: it is in the vocabulary and a real CTC target, but it is snapped onto the preceding token's end, so `start == end` and it takes zero frames. The model has no opinion about it either — measured confidence runs −12 to −24 — and giving it a duration would be inventing one.

Whether the output is per character or per word is decided **per run of characters**, with no language table: a run from a script that writes without spaces is split into characters, every other run is one unit the line-breaker cannot cut inside. So Korean and Japanese come out per character, Chinese likewise (a Chinese "word" is a whole sentence, and the test applies inside it), and English, Spanish and French per word. A word holding both scripts is cut where the script changes rather than judged once, so `原版Whisper` is `原` `版` `Whisper` and not nine single characters.

Characters outside the checkpoint's vocabulary have no CTC target and so never reach the DP, but they are **not** dropped from the output: a forced alignment is a monotone path, so a character written between two placed ones must fall between them, and it is placed at the midpoint of that interval with `inferred: true` to say so. A transcript that mentions a rare han character four times used to ship a subtitle missing all four, and the only trace was a line in a `skipped` list that nobody read. There is no such list now — the character is in `tokens` with the flag set, which is the same information without the risk of it reading as "this was thrown away". (That list was 96% spaces anyway: a space is a tokenizer separator and was never a target to begin with.)

### Deliberate departures from the reference

The algorithm is ported from `MahmoudAshraf97/ctc-forced-aligner`, but that implementation was written for its own MMS checkpoint. The following ten were changed **on measurement, on this checkpoint**. They are not bugs:

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

   **Measuring this against the Python reference needs care, and getting it wrong is easy.** The reference's own dump script hard-codes `edges`, so comparing this port against a reference dump compares two different target sequences and the number that comes out is about the star placement rather than about the port. On two Buckeye utterances, the port against an `edges` reference reads 1.91 and 1.59 frames of mean |d_start| — and the reference against *itself* with only the placement changed reads 1.36 and 1.71. Re-dumped on the port's own placement, the same comparison is 0.55 and 0.59. Anyone re-measuring this should re-dump the reference first, and `examples/dump_path.rs` writes the word-level spans that comparison needs.

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

9. **A blank run longer than 0.5 s is silence, not a pause: the word starts on its own evidence.**
   The midpoint rule's error is half the run it is applied over — fine for the prosodic gaps of continuous speech, where a human annotator really does split the pause, unbounded where the run is silence that belongs to nobody. Measured on 10 clean Mandarin sentences (FLEURS cmn test) with leading silences of 0.9–3.7 s: the unsplit midpoint placed the first character 235 ms before the acoustic onset at the median and 740 ms at the worst. Buckeye — where the midpoint rule was measured against hand marks — has no word-fronting run past 500 ms, so English behaviour is bit-identical: every start boundary the old rule ever exercised is unchanged. With this, both sides of a boundary have a stated limit (the end side's is departure 2).

10. **A mark with no timed token before it is a point on the sound that follows.**
    The forward anchor rule cannot reach a mark that opens the stream — a star holds no time, so an opening `“` kept whatever frames the path left near it, and on material that opens into silence that span is pure fiction: measured on a Mandarin FLEURS clip, an opening `“` held 1.1 s over a blank run it had no sound in, dragging its host word's cue a second early with it. It is now a point at the first timed token's start — the mirror of the rule that anchors every other mark to the sound before it.

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
