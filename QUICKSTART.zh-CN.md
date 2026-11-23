# 快速开始 - Apple Silicon优化版

## 5分钟上手指南

### 第一步：安装Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 第二步：克隆并编译

```bash
git clone https://github.com/eclipse005/ctc-forced-aligner-wgpu.git
cd ctc-forced-aligner-wgpu
cargo build --release
```

等待3-5分钟编译完成。

### 第三步：下载模型

创建Python脚本 `download_model.py`:

```python
from huggingface_hub import snapshot_download
snapshot_download('aadel4/omniASR-CTC-300M-v2',
                  local_dir='models/omniASR-CTC-300M-v2-hf')
```

运行下载：
```bash
pip install huggingface_hub
python download_model.py
```

模型大小约1.2GB，需要几分钟下载。

### 第四步：准备测试音频

如果你有WAV文件，直接使用。如果是其他格式：

```bash
# 安装ffmpeg（如果还没有）
brew install ffmpeg

# 转换音频为WAV
ffmpeg -i your_audio.mp3 -ar 16000 -ac 1 -c:a pcm_f32le test.wav
```

### 第五步：运行第一次对齐

```bash
./target/release/align \
    --audio test.wav \
    --text "这是你的转录文本" \
    --model models/omniASR-CTC-300M-v2-hf
```

输出：
```json
{
  "text": "这是你的转录文本",
  "tokens": [
    {"piece": "这", "start": 0.5, "end": 0.8, ...},
    {"piece": "是", "start": 0.8, "end": 1.1, ...},
    ...
  ]
}
```

## 常用命令

### 生成SRT字幕

```bash
./target/release/align \
    --audio video.wav \
    --text transcript.txt \
    --format srt \
    --output output.srt
```

### 生成卡拉OK字幕

```bash
./target/release/align \
    --audio song.wav \
    --text "歌词内容" \
    --format ass \
    --output karaoke.ass
```

### 强制使用GPU

```bash
./target/release/align \
    --audio speech.wav \
    --text "文本" \
    --device gpu
```

### 处理长音频

```bash
./target/release/align \
    --audio long_audio.wav \
    --text long_transcript.txt \
    --window 30 \
    --context 2
```

## 性能测试

运行基准测试：

```bash
chmod +x benchmark_m_series.sh
./benchmark_m_series.sh test.wav "测试文本"
```

## 故障排除

### 问题1：找不到模型

```bash
export CTC_MODEL_DIR=/path/to/your/model
./target/release/align --audio test.wav --text "文本"
```

### 问题2：GPU不可用

```bash
# 检查设备
./target/release/align --list-devices

# 应该看到：
# 0: gpu "Apple M1 Max" (Metal)
```

如果没有Metal设备，使用CPU：
```bash
./target/release/align --audio test.wav --text "文本" --device cpu
```

### 问题3：音频格式不支持

只支持WAV格式，转换其他格式：

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

## 完整示例

### 示例1：制作字幕文件

```bash
# 1. 从视频提取音频
ffmpeg -i video.mp4 -ar 16000 -ac 1 output.wav

# 2. 获取转录（或手动输入）
echo "这是视频的完整转录文本" > transcript.txt

# 3. 生成字幕
./target/release/align \
    --audio output.wav \
    --text transcript.txt \
    --format srt \
    --output video.srt

# 4. 添加字幕到视频
ffmpeg -i video.mp4 -vf subtitles=video.srt output_with_subs.mp4
```

### 示例2：批量处理

创建脚本 `batch_align.sh`:

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

运行：
```bash
chmod +x batch_align.sh
./batch_align.sh
```

### 示例3：实时监控

```bash
# 使用watch监控进度
watch -n 1 'tail -20 align.log'

# 在另一个终端运行
./target/release/align \
    --audio long_audio.wav \
    --text transcript.txt 2>&1 | tee align.log
```

## 下一步

- 📖 阅读完整文档：`README.Apple-Silicon.zh-CN.md`
- 🔍 了解优化细节：`OPTIMIZATION_M_SERIES.md`
- 📊 查看性能报告：`PERFORMANCE_SUMMARY.md`
- 🐛 遇到问题？提交Issue到GitHub

## 获取帮助

```bash
# 查看所有选项
./target/release/align --help

# 列出可用设备
./target/release/align --list-devices
```

## 性能提示

✅ **推荐配置（M1 Max）：**
- 短音频（<2分钟）：`--device gpu --window 60`
- 中等（2-10分钟）：`--device gpu --window 30`
- 长音频（>10分钟）：`--device gpu --window 20`

✅ **内存受限时：**
- 减小窗口：`--window 15`
- 或使用CPU：`--device cpu`

✅ **追求最快速度：**
```bash
# 使用release-fast编译
cargo build --profile release-fast
./target/release-fast/align ...
```

---

**开始使用吧！遇到问题欢迎提Issue！** 🚀
