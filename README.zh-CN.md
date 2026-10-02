# CTC Forced Aligner wgpu

**基于 wgpu 的 omniASR-CTC 强制对齐（Rust 实现）。**

[English](README.md) · **简体中文**

[OmniLingual / omniASR-CTC](https://huggingface.co/aadel4/omniASR-CTC-300M-v2) 检查点的轻量级跨平台 Rust 实现，使用 [wgpu](https://github.com/gfx-rs/wgpu) 进行 GPU 加速。

目标很简单：给定一段音频**和**对应的转录文本，给出每个字在音频中的起止时间——**在本地原生运行**，不依赖 Python，也不依赖特定厂商的 GPU 运行时。对齐结果与移植所依据的 Python 参考实现逐 token 一致，并已逐帧验证。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 跨平台 GPU 支持
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退
* 📦 离线本地推理
* ⏱️ 字级 / 词级起止时间戳
* 🗣️ 多语言——取决于检查点词表的覆盖范围
* 📄 JSON / spans / SRT / Cues 输出
* 🧩 CLI + Rust 库

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
ctc-forced-aligner-wgpu = { git = "https://github.com/eclipse005/ctc-forced-aligner-wgpu.git" }
```

或从源码构建 CLI：

```bash
git clone https://github.com/eclipse005/ctc-forced-aligner-wgpu.git
cd ctc-forced-aligner-wgpu
cargo build --release        # target/release/align
```

| Feature | 说明 |
|---------|------|
| `alloc-stats` | 统计 `alloc_stats::Stats` 的分配，输出每块音频的 live / 峰值。默认关闭：每次分配要三次原子读改写，生产运行并不需要。以 `--features alloc-stats` 构建并设置 `CTC_ALLOC_STATS=1` 启用 |

### 模型下载

权重**不在**本仓库内。请下载检查点（版权归原作者）：

```python
from huggingface_hub import snapshot_download
snapshot_download('aadel4/omniASR-CTC-300M-v2',
                  local_dir='models/omniASR-CTC-300M-v2-hf')
```

用 `--model` 指向它，或设置 `CTC_MODEL_DIR`。

默认存储格式为 `f32`；`f16` 与 `bf16` 检查点同样接受，且为**无损**扩展。

### 快速上手

```bash
align --audio speech.wav --text "hello world" --model models/omniASR-CTC-300M-v2-hf
align --audio speech.wav --text transcript.txt --format srt --output out.srt
```

| 选项 | 说明 |
|------|------|
| `--audio <file>` | 音频文件（WAV） |
| `--text <file/text>` | 转录文本：若为已存在的路径则读文件，否则当作文本本身 |
| `--model <dir>` | 模型目录，或 `CTC_MODEL_DIR` 环境变量 |
| `--device <spec>` | `auto`（默认，无 GPU 时用 CPU）、`cpu`、`vulkan[:i]`、`dx12[:i]`、`#n`，或名称子串 |
| `--window <sec>` | 分窗编码的秒数（默认 `30`）；`0` 表示整文件一次前向 |
| `--context <sec>` | 每个窗口两侧携带的重叠秒数（默认 `2`） |
| `--format <json\|spans\|srt\|cues>` | `json` 为紧凑字级对齐（默认）；`spans` 把静音归到相邻片段；`srt` / `cues` 是字幕断点 |
| `--split <word\|char\|sentence>` | 仅 `spans`；以 CJK 为主的文本默认为 `char` |
| `--preset <short\|standard\|loose>` | `srt` / `cues` 的字幕预设（默认 `standard`） |
| `--merge-threshold <sec>` | `spans`：把短于该值的间隙合并（默认 `0`） |
| `--output <path>` | 写入此路径（默认打印到 stdout）。`spans` 另写一份 `.txt` |
| `--list-devices` | 列出本机可用的 wgpu 适配器 |

### 库

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

`align` 的窗口与上下文长度以秒为单位（窗口传 `None` 表示整文件一次前向）。`TokenAlignment` 携带 `piece`、`start`、`end`、`start_frame`、`end_frame`、`word_id` 以及逐帧均值 `score`。`align_with_path` 额外返回逐帧状态路径——与参考实现对拍时比对的就是它。完整 API 见 `cargo doc`。

### 音频输入

推荐 16 kHz 单声道 WAV——模型的原生采样率，不做重采样直接使用。WAV 内的其他采样率会自动转换；需要完全控制时请先自行转换：

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

目前只支持 WAV。

### 时间戳粒度

模型的卷积前端下采样 320 倍，因此**时间戳落在 20 ms 的网格上**——这是模型的限制，不是本实现的。分词对有词分隔的文字按空白切分，对 CJK 逐字符；不在检查点词表内的字符会被丢弃。

### 长音频

无论文件多长，内存都是有界的。trellis 以三种等价形式中最窄且放得下的那一种保存：gather 后的 `(T, S)` 列、lm head 的 `(T, vocab)` 输出，或编码器 `(T, hidden)` 流（按需重跑 head）。一小时的音频占用在几 GB 量级而不是无限增长；Viterbi 只计算每帧可达的带状区域，backpointer 每状态打包 2 bit。

### 准确度

以 [Buckeye](https://buckeye.shoup.informatics.cmu.edu/) 人工标注语料（说话人 s32）为基准，词级起止时间按 fa-bench 的匹配规则与人工标注对比：

| | MAE | ≤20 ms | ≤50 ms |
|---|---:|---:|---:|
| **本实现（300M）** | **29.8 ms** | 45.3% | 84.7% |
| MMS-FA（fa-bench Track-1 已发表） | 30.5 / 31.1 ms | — | — |

两种后端（CPU 塔与 GPU 塔）以及三种 trellis 形态在同一输入上逐位一致，所以上面的数字是检查点的属性，与后端无关。

### 为什么用 wgpu？

不依赖 CUDA、ROCm 或其他特定厂商的运行时，本项目用 **wgpu** 作为统一的 GPU 抽象层。

这样可以为不同平台和不同 GPU 厂商构建同一套 Rust 对齐运行时。

### 项目状态

🚧 **积极开发中**

性能和硬件兼容性仍在针对不同 GPU 持续优化与测试。

### 相关项目

* [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) — 模型家族
* [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) — 本项目移植所依据的 Python 参考实现
* [fa-bench](https://github.com/olewave/fa-bench) — 有公开数字的强制对齐基准
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — 语音识别（转录）
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) — Qwen3 强制对齐

### 许可

Apache-2.0。

算法移植自 [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)，其代码为 BSD 2-Clause 许可。

本仓库是加载并运行已公开发布的 omniASR-CTC 权重的**独立 Rust 推理实现**——并非 Meta 官方发布，与原作者无隶属关系。模型权重版权归其各自所有者。
