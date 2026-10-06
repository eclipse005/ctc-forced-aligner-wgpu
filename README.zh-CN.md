# CTC Forced Aligner wgpu

**基于 wgpu 的 omniASR-CTC 强制对齐（Rust 实现）。**

[English](README.md) · **简体中文**

[OmniLingual / omniASR-CTC](https://huggingface.co/aadel4/omniASR-CTC-300M-v2) 检查点的轻量级跨平台 Rust 实现，使用 [wgpu](https://github.com/gfx-rs/wgpu) 进行 GPU 加速。

目标很简单：给定一段音频**和**对应的转录文本，给出每个字在音频中的起止时间——**在本地原生运行**，不依赖 Python，也不依赖特定厂商的 GPU 运行时。Viterbi 与边界规则移植自 Python 参考实现并与之逐帧对拍验证；有意偏离的地方会说清楚偏离了什么、为什么。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 跨平台 GPU 支持
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退
* 📦 离线本地推理
* ⏱️ 逐时间单元 / 逐句起止时间戳
* 🗣️ 多语言——取决于检查点词表的覆盖范围
* 📄 JSON / SRT / ASS 卡拉OK 输出
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
| `--format <json\|srt\|ass>` | `json` 为完整对齐（默认）；`srt` 为字幕；`ass` 为卡拉OK，逐字扫光 |
| `--ass-res <WxH>` | 仅 `ass`：**视频**的分辨率（默认 `1920x1080`）——对齐器只看到波形，猜不出来 |
| `--ass-font <name>` | 仅 `ass`：字体名（默认 `Malgun Gothic`）；字号固定 |
| `--output <path>` | 写入此路径（默认打印到 stdout） |
| `--list-devices` | 列出本机可用的 wgpu 适配器 |

`--window` 与 `--context` 测过：在合法区间内改动不改变任何时间戳指标（同一份素材 30/2 与 60/4 的 start MAE 差异在噪声内），而 60/4 编码慢 47%。它们是显存旋钮，不是精度旋钮。

`json` 就是三样东西，全部平级、谁也不在谁里面：

| 字段 | 是什么 |
|---|---|
| `text` | 原文，一字未改 |
| `tokens` | 每个**时间单元**出现在哪儿：`text`、`start`、`end`、`space_before`、`score`、`inferred` |
| `segments` | 同样的内容按句子读：`text`、`start`、`end` |

单元就是词，而**是哪一种词逐词判定**，不用语言表：分词语言里一整词，不分词语言里一个字。中文的"词"本来就是整句，判据在词内部生效，所以中英混排的稿子同一篇里两种都有。按字切分时，**标点跟着它前面那个字走**——`は、` 是一个单元而不是两个——因为标点被钉在前一个 token 的结束点上、自己不占帧，所以单拎一个标点出来的那一行只是个时间点，不是时长。73 分钟的日语转录里这样的行有 2,490 / 26,355；现在只剩 4 行，是原文里跟在空格后面的标点——它们自己开启了一个词。

六个字段里有两个是为了**让这一行不能骗人**：

- **`space_before`** 是原文在这个单元前面自带的空白，正是它让这些行能还原回 `text`。最顺手的替代做法是「按空格 join」——那正是给日文每个字之间都塞空格的那个 bug。那份转录里它标了 1,780 行，正好等于原文的空格数。
- **`inferred`** 标出含有词表外字符的单元，那部分时长是前后字留出的空隙中点，不是实测出来的。字本身还在 `text` 里、也还在这一行里。**`score` 是那些实测字符的均值，一个都没实测到时为 `null`**——插值出来的字根本没有分数，而 log 概率 0 等于概率 1，那是"置信度满分"，偏偏用在了唯一没有任何证据的那一行上。

谁也不指向谁：要时间戳的人读 `tokens` 就完事；要句子的人读 `segments`，断句逻辑和自己不一样的可以自己重新分组。字幕断行是本程序的观点，所以只活在 `--format srt` 和 `--format ass` 里，不进对齐结果——之前也当数据带着，文件大 39%，还多出一份要同步的字符副本。`<star>` 根本不进输出：它是 DP 锚点不是转录文本，分数是个常数，而把 `<star>` 打进字幕正是曾经出过的 bug。

### 库

```rust
use ctc_forced_aligner_wgpu::{Aligner, DeviceSelector};

let aligner = Aligner::load_on(std::path::Path::new("model"),
                               DeviceSelector::parse("auto")?)?;
let out = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some(30.0), 2.0, None)?;
for t in &out.tokens {
    println!("{:.3}s - {:.3}s  {}", t.start, t.end, t.piece);
}
```

API 就这些：`Aligner` 加载与运行、`AlignOutput` 读结果、`TokenAlignment` 一个时间单元、`DeviceSelector` 选设备、`list_targets` 列设备。字幕是 `views::build_cues` 经 `render::srt` 或 `render::ass` 渲染。**其余一律 `pub(crate)`**——模型、Viterbi、边界规则、词句切分，全都不是对 crate 之外的承诺。

`align` 的窗口与上下文长度以秒为单位（窗口传 `None` 表示整文件一次前向）。最后一个参数是可选的 `AlignProgress` 回调——`&mut dyn FnMut(done, total)`，每编完一个窗口在**调用方线程**上触发一次；分母是**窗口数**不是秒，因为窗口数开跑前就数得出来，秒只能估。传 `None` 的代价是每窗口一个分支。`TokenAlignment` 携带 `piece`、`start`、`end`、`start_frame`、`end_frame`、`word_id` 以及逐帧均值 `score`。`align_with_path` 额外返回逐帧状态路径——与参考实现对拍时比对的就是它。完整 API 见 `cargo doc`。

### 代码怎么分层

四层，每层只依赖它上面的层：

| 层 | 模块 | 它知道什么 |
|---|---|---|
| base | `config` `weights` `vocab` `simd` `shaders` `gpu` `resample_sinc` `audio` | 权重、核、设备——完全不知道对齐是什么 |
| model | `wav2vec2` `wav2vec2_gpu` | 编码器塔及其 CPU 孪生：音频进，逐帧打分出 |
| align | `viterbi` `timeline` | 路径，以及**唯一决定时间戳的地方** |
| text | `spans` `views` | 原文自己的词与句，以及字幕在哪里断行 |

`timeline` 是值得了解的那部分：Viterbi 选定帧之后，三条规则把它变成字幕显示的区间——短到还算韵律停顿的，边界落在其中点；长到成了静音的，字从自己的声学证据开始；标点没有音素，所以是它所贴声音上的一个点——开头的标点贴后面的声音，其余贴前面的声音；词表里没有的字落在它已定位的邻居之间。它们原本散在三个文件里，顺序靠注释维持；现在是一份有序清单，每条都带着选出它的实测。Viterbi 之后谁都不许再写 `start` 或 `end`——这就是 `--format srt`、`--format ass` 和 json 三者永远不会对不上的原因。

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

不在检查点词表内的字符**不会被丢弃**：强制对齐是单调路径，写在两个已定位字符之间的字必然落在它们之间，因此它取该区间的中点，并带 `inferred: true` 标记说明这里没有任何实测。出现过 4 次的生僻字曾经 4 次都没进字幕，唯一的痕迹是 `skipped` 里一行没人看的记录。现在没有这个列表了——字就在 `tokens` 里、标记就在那一行上，信息一样，但不会被读成"这些字被扔了"。（顺带一提，那个列表 96% 是空格：空格是分词器分隔符，本来就不是 target。）

### 与参考实现的有意偏离

算法移植自 `MahmoudAshraf97/ctc-forced-aligner`，但那个实现是为它自己的 MMS 检查点写的。以下十处是**在本检查点上实测后有意改掉的**，不是 bug：

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

   **拿 Python 参考来量这件事要小心，搞错很容易。** 参考自己的 dump 脚本把星位写死成 `edges`，所以拿本项目去比那份 dump，比的是两套不同的 target 序列，出来的数是星位的差别而不是本项目的差别。Buckeye 两句上，本项目对 `edges` 参考读出 1.91 / 1.59 帧 mean|d_start|，而参考**自己**只换星位就读出 1.36 / 1.71；把参考按本项目的星位重新 dump 之后，同一个比较是 0.55 / 0.59。要重测这件事的人请先重 dump 参考，`examples/dump_path.rs` 会写出这个比较需要的词级 span。

2. **一个 token 最多向静音借用 1.0 秒。**
   中点规则会把整段静音分给前一个 token，实测有单字因此吃掉 8.4 秒。阈值来自同一批语料：15,722 个字符里尾部探入静音的深度中位 0.00 s、p99 0.00 s、最大 1.57 s，所以这个上限在朗读语料上几乎不起作用，而在有长停顿的素材上把最差结束误差从 8.4 s 压到 1.7 s、结束 MAE 从 896 ms 压到 486 ms，**且所有起始时间戳逐位不变**（这个上限只碰 `end`，单边）。
   FireRedASR2 用的是"平均时长的 N 倍"，在本语料上会误切 7.9% 的 token 且切掉的语音多于释放的静音——倍数和时长是两回事。

3. **断句按 word_id 逐词分组，不按 `<star>`。**
   参考实现按星恢复词边界，而英语走 `edges`、全文件只有两个星，结果整个转录变成一个 cue、单词全被劈开。

4. **断句单位逐词判定，不逐文件。**
   原来整篇只问一次"是不是中文"，一个汉字就让英文按字符断，`alignment` 被切成 `alignm` + `ent`。

5. **`json` 只剩三个平级字段，其中没有一个是断句结果的副本。**
   之前也把 cue 当数据带着，消费方不指定 format 就能看到字幕断行——逐条时间戳验证一致，这是唯一支持它的理由。但代价大于收益：文件大 39%，而且转录里每个字符被写了五遍，于是有五样东西要互相对齐。`--format srt` 和 `--format ass` 无论如何都是同一次 `build_cues()` 调用，它们之间不可能对不上；想要 cue 数据的消费方按自己的方式给 `tokens` 分组即可。

6. **一个词 = 原文里按空白切出的一段，标点算词内；边界只由 `word_id` 决定。**
   参考实现就是 `text.split()`，所以标点是它所在那个词的字符。原来为了"让标点归到前一个词"开了一个旁路缓冲，结果在下一个字母到来时把缓冲清空——凡是后面还跟着字的 `、` 全丢了：26,552 字的日语转录里**少了 1,116 个标点**，而没有那个缓冲的 cue 视图一个没少。对齐器是给文本打时间戳的，不是改文本的。
   `<star>` 只跳过、不当边界：一个词开头的星带的是**前一个词**的 id，所以首字没有 target 的词，星会落在词中间、id 序列还会倒流。`贅沢` 曾被切成 `贅 沢`——凭空多出一个原文没有的空格。`json` 在每个 word 上给出 `space_before`，`words` 因此能还原成它原来的文本。

7. **词表外的字符按中点插回，而不是丢弃。**
   见[时间戳粒度](#时间戳粒度)。参考实现直接丢、只在 `skipped` 里报一行；这里给它前后字符留出的区间中点，并在 `tokens` 的那一行上标 `inferred`，字幕才不会说出说话人没说过的话，而那一行本身也还在。

8. **新增 `--format ass`。**
   卡拉OK 就是同一份 cue 列表，每个字符一个 `\k`，所以不可能和 SRT 对不上。原文里的空格落在**它后面那个字符**上（写成 `\h`），行首不加——那里换行已经把位置占了，和 SRT 一样。`WrapStyle: 0` 和播放分辨率来自 `--ass-res`：断行交给 libass，在这里插 `\N` 等于把"播放器的字体回退和安全区"这一猜测写死。

9. **超过 0.5 秒的 blank 段是静音而不是停顿：字从自己的声学证据开始。**
   中点规则的误差是被套用的那段长度的一半——对连续语流里的韵律间隙没问题，人工标注员确实是从停顿中间切的；但对不属于任何词的静音，这个误差没有上界。在 10 句干净的普通话（FLEURS cmn test，前导静音 0.9–3.7 秒）上实测：不分段的中点规则把第一个字放在声学起点之前，中位 235 毫秒、最差 740 毫秒。Buckeye——中点规则当初对着人工标注实测的语料——根本不存在超过 500 毫秒的词前 blank 段，所以英语行为逐位不变：旧规则实际作用过的每一条起始边界都保持原样。从此边界的两侧都有了明说的界限（end 侧的那条就是偏离 2）。

10. **前面没有任何带声音 token 的标点，是后随声音上的一个点。**
    前向锚定规则够不到开头就出现的标点——星不带时间，所以开头的 `“` 会保留路径留在它附近的帧；在以静音开头的素材上这段跨度纯属虚构：某句普通话 FLEURS 上实测，开头的 `“` 在一个它没有任何声音的 blank 段上占了 1.1 秒，把它所在词的 cue 拖早了一秒。现在它是第一个带声音 token 起点上的一个点——正是"其余标点锚定到前面声音末尾"这条规则的镜像。

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
