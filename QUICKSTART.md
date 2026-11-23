# Quick Start - Apple Silicon Optimized Version

## 5-Minute Getting Started Guide

### Step 1: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Step 2: Clone and Build

```bash
git clone https://github.com/eclipse005/ctc-forced-aligner-wgpu.git
cd ctc-forced-aligner-wgpu
cargo build --release
```

Wait 3-5 minutes for compilation to complete.

### Step 3: Download Model

Create Python script `download_model.py`:

```python
from huggingface_hub import snapshot_download
snapshot_download('aadel4/omniASR-CTC-300M-v2',
                  local_dir='models/omniASR-CTC-300M-v2-hf')
```

Run download:
```bash
pip install huggingface_hub
python download_model.py
```

Model size is about 1.2GB, takes a few minutes to download.

### Step 4: Prepare Test Audio

If you have a WAV file, use it directly. For other formats:

```bash
# Install ffmpeg (if not already installed)
brew install ffmpeg

# Convert audio to WAV
ffmpeg -i your_audio.mp3 -ar 16000 -ac 1 -c:a pcm_f32le test.wav
```

### Step 5: Run First Alignment

```bash
./target/release/align \
    --audio test.wav \
    --text "your transcript text" \
    --model models/omniASR-CTC-300M-v2-hf
```

Output:
```json
{
  "text": "your transcript text",
  "tokens": [
    {"piece": "your", "start": 0.5, "end": 0.8, ...},
    {"piece": "transcript", "start": 0.8, "end": 1.1, ...},
    ...
  ]
}
```

## Common Commands

### Generate SRT Subtitles

```bash
./target/release/align \
    --audio video.wav \
    --text transcript.txt \
    --format srt \
    --output output.srt
```

### Generate Karaoke Subtitles

```bash
./target/release/align \
    --audio song.wav \
    --text "lyrics content" \
    --format ass \
    --output karaoke.ass
```

### Force GPU Usage

```bash
./target/release/align \
    --audio speech.wav \
    --text "text" \
    --device gpu
```

### Process Long Audio

```bash
./target/release/align \
    --audio long_audio.wav \
    --text long_transcript.txt \
    --window 30 \
    --context 2
```

## Performance Testing

Run benchmark:

```bash
chmod +x benchmark_m_series.sh
./benchmark_m_series.sh test.wav "test text"
```

## Troubleshooting

### Issue 1: Model Not Found

```bash
export CTC_MODEL_DIR=/path/to/your/model
./target/release/align --audio test.wav --text "text"
```

### Issue 2: GPU Unavailable

```bash
# Check devices
./target/release/align --list-devices

# Should see:
# 0: gpu "Apple M1 Max" (Metal)
```

If no Metal device found, use CPU:
```bash
./target/release/align --audio test.wav --text "text" --device cpu
```

### Issue 3: Audio Format Not Supported

Only WAV format is supported, convert other formats:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

## Complete Examples

### Example 1: Create Subtitle File

```bash
# 1. Extract audio from video
ffmpeg -i video.mp4 -ar 16000 -ac 1 output.wav

# 2. Get transcript (or input manually)
echo "This is the complete video transcript" > transcript.txt

# 3. Generate subtitles
./target/release/align \
    --audio output.wav \
    --text transcript.txt \
    --format srt \
    --output video.srt

# 4. Add subtitles to video
ffmpeg -i video.mp4 -vf subtitles=video.srt output_with_subs.mp4
```

### Example 2: Batch Processing

Create script `batch_align.sh`:

```bash
#!/bin/bash
for audio in *.wav; do
    base=$(basename "$audio" .wav)
    ./target/release/align \
        --audio "$audio" \
        --text "${base}.txt" \
        --format srt \
        --output "${base}.srt"
done
```

Run:
```bash
chmod +x batch_align.sh
./batch_align.sh
```

### Example 3: Real-time Monitoring

```bash
# Use watch to monitor progress
watch -n 1 'tail -20 align.log'

# In another terminal
./target/release/align \
    --audio long_audio.wav \
    --text transcript.txt 2>&1 | tee align.log
```

## Next Steps

- 📖 Read complete documentation: `README.Apple-Silicon.md`
- 🔍 Learn optimization details: `OPTIMIZATION_M_SERIES.md`
- 📊 Check performance report: `PERFORMANCE_SUMMARY.md`
- 🐛 Issues? Submit to GitHub

## Get Help

```bash
# View all options
./target/release/align --help

# List available devices
./target/release/align --list-devices
```

## Performance Tips

✅ **Recommended settings (M1 Max):**
- Short audio (<2 min): `--device gpu --window 60`
- Medium (2-10 min): `--device gpu --window 30`
- Long audio (>10 min): `--device gpu --window 20`

✅ **Memory constrained:**
- Reduce window: `--window 15`
- Or use CPU: `--device cpu`

✅ **Maximum speed:**
```bash
# Use release-fast build
cargo build --profile release-fast
./target/release-fast/align ...
```

---

**Start using it now! Submit issues if you encounter problems!** 🚀
