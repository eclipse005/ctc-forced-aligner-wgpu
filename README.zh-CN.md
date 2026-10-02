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
| `--window <sec>` | 显存与吞吐；**不影响时间戳**（默认 `30`）。上限约 90 s，受单次前向的显存限制 |
| `--context <sec>` | 窗口两侧的编码器重叠（默认 `2`，下限约 1.3，须覆盖位置卷积感受野） |
| `--format <json\|srt>` | `json` 为完整对齐（默认）；`srt` 为字幕 |
| `--output <path>` | 写入此路径（默认打印到 stdout） |
| `--list-devices` | 列出本机可用的 wgpu 适配器 |

`--window` 与 `--context` 测过：在合法区间内改动不改变任何时间戳指标（同一份素材 30/2 与 60/4 的 start MAE 差异在噪声内），而 60/4 编码慢 47%。它们是显存旋钮，不是精度旋钮。

`json` 里同时给出 `chars`（逐字 + 帧号 + 置信度）、`words`、`segments`、`spans`（静音归入相邻段、时间轴铺满）和 `cues`（与 `--format srt` 完全同一套断句结果）。`--format srt` 就是这些 cue 的渲染，两条路不可能对不上。

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

模型的卷积前端下采样 320 倍，因此**时间戳落在 20 ms 的网格上**——这是模型的限制，不是本实现的。帧率、下采样倍率与 blank id 都从 `config.json` 读（`conv_stride` 的乘积、`pad_token_id`），不写死。

**对齐的对象是字符。空格不参与**——它是分词器的分隔符，音频里没有"空格"这个声学事件。**标点参与但不占时长**：它在词表内，是真实的 CTC 目标，但被钉在前一个 token 的结束点上，`start == end`，占 0 帧；模型对它的置信度也一贯很差（实测 −12 到 −24），给它硬塞时长等于伪造。

输出的"字还是词"**逐词判定**，不用语言表：一个词的字符里 CJK 字符占多数就拆成字符，否则整词一个单位、断句器碰不到内部。所以韩文和日文按字、中文按字（中文的"词"本来就是整句，判据在词内部生效）、英文西法按词。

不在检查点词表内的字符会被丢弃，并列在 `json` 的 `skipped` 字段里。

### 与参考实现的有意偏离

算法移植自 `MahmoudAshraf97/ctc-forced-aligner`，但那个实现是为它自己的 MMS 检查点写的。以下五处是**在本检查点上实测后有意改掉的**，不是 bug：

1. **每个词前一个 `<star>`，不用参考实现的 `edges`（全文件首尾各一个）。**
   `<star>` 是 DP 锚点，不是词边界。这个检查点的 DP 在声学证据弱的地方是欠约束的：两个星会让整条路径滑动，每词一个不会。180 条多语料实测（其中 124 条经 VAD 与能量检测交叉验证——单信 VAD 会在部分 FLEURS 英语上漏掉整段语音，信了会把结论反过来）：

   | 语言 | 每词一星 | `edges` |
   |---|---:|---:|
   | 西语 | **−0.1308** | −0.3896 |
   | 法语 | **−0.2605** | −0.5315 |
   | 德语 | **−0.3322** | −0.5347 |
   | 英语 | **−0.5936** | −0.7137 |
   | 日语 | −0.2348 | −0.2346 |
   | 中文 | −0.1993 | −0.1987 |
   | **合计** | **−0.2570** | −0.4301 |

   差距恰好落在机制该起作用的地方：分词语言上大胜，不分词语言上持平——那里"词"本来就是整行，两种插法数量差不多。整文件 `log_prob` 不能用来比（对路径求和，星少则项少），所以用按帧归一的平均分。

2. **一个 token 最多向静音借用 1.0 秒。**
   中点规则会把整段静音分给前一个 token，实测有单字因此吃掉 8.4 秒。阈值来自同一批语料：15,722 个字符里尾部探入静音的深度中位 0.00 s、p99 0.00 s、最大 1.57 s，所以这个上限在朗读语料上几乎不起作用，而在有长停顿的素材上把最差结束误差从 8.4 s 压到 1.7 s、结束 MAE 从 896 ms 压到 486 ms，**且所有起始时间戳逐位不变**（这个上限只碰 `end`，单边）。
   FireRedASR2 用的是"平均时长的 N 倍"，在本语料上会误切 7.9% 的 token 且切掉的语音多于释放的静音——倍数和时长是两回事。

3. **断句按 word_id 逐词分组，不按 `<star>`。**
   参考实现按星恢复词边界，而英语走 `edges`、全文件只有两个星，结果整个转录变成一个 cue、单词全被劈开。

4. **断句单位逐词判定，不逐文件。**
   原来整篇只问一次"是不是中文"，一个汉字就让英文按字符断，`alignment` 被切成 `alignm` + `ent`。

5. **`--format cues` 并入 `json`。**
   它和 `--format srt` 是同一次 `build_cues()` 调用、逐条时间戳一致，只差序列化方式；分成两个 format 只会让它们有机会对不上。

### 长音频

无论文件多长，内存都是有界的。trellis 以三种等价形式中最窄且放得下的那一种保存：gather 后的 `(T, S)` 列、lm head 的 `(T, vocab)` 输出，或编码器 `(T, hidden)` 流（按需重跑 head）。一小时的音频占用在几 GB 量级而不是无限增长；Viterbi 只计算每帧可达的带状区域，backpointer 每状态打包 2 bit。

### 准确度

以 [Buckeye](https://buckeye.shoup.informatics.cmu.edu/) 人工标注语料（说话人 s32）为基准，词级起止时间按 fa-bench 的匹配规则与人工标注对比：

| | MAE | ≤20 ms | ≤50 ms |
|---|---:|---:|---:|
| **本实现（300M）** | **29.8 ms** | 45.3% | 84.7% |
| MMS-FA（fa-bench Track-1 已发表） | 30.5 / 31.1 ms | — | — |

两种后端（CPU 塔与 GPU 塔）以及三种 trellis 形态在同一输入上逐位一致，所以上面的数字是检查点的属性，与后端无关。

**这个数字的适用范围要说清楚**：Buckeye 是英语朗读语料、词间停顿短。在有长停顿的广播素材上，起始边界依然准（韩文广播词级 start MAE 277.5 ms、中位 65 ms；中文唱歌段 start MAE 82.8 ms），但**结束边界会按中点规则伸进后面的静音**——上面的静音上限缓解了这一点，剩余的系统性偏晚仍在。

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
