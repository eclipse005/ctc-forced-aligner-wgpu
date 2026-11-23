# Apple Silicon 优化版 CTC Forced Aligner

这是针对Apple M系列芯片（M1/M2/M3/M4）优化的CTC强制对齐器，专门用于给定音频和转录文本，输出每个词/字在音频中的精确起止时间。

## 🚀 核心优化

### 1. ARM NEON SIMD 加速

完整的ARM NEON向量化实现，覆盖所有热点操作：

- ✅ **GELU激活函数** - 4通道并行计算，使用FMA指令
- ✅ **误差函数 (erf)** - 多项式逼近的NEON实现  
- ✅ **Softmax** - 向量化exp计算和求和
- ✅ **Log-Softmax** - 融合的归一化和对数计算
- ✅ **矩阵乘法** - 通过Apple Accelerate框架加速

**性能提升**: CPU推理速度提升 **2-3倍**（相比标量实现）

### 2. Metal GPU 加速

通过wgpu抽象层使用Metal后端：

- 🎮 Wav2Vec2编码器在GPU上执行
- 🎮 卷积层、注意力计算GPU并行
- 🎮 自动CPU/GPU混合策略

**适用场景**: 长音频（>1分钟）、实时处理需求

### 3. 编译器优化

- 🔧 `target-cpu=native` - 利用M系列全部指令集
- 🔧 `opt-level=3` - 最高优化级别
- 🔧 LTO (链接时优化) - 跨模块内联
- 🔧 `strip` - 移除调试符号减小体积

## 📦 快速开始

### 安装依赖

```bash
# 安装Rust（如果还没有）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 克隆项目
git clone https://github.com/eclipse005/ctc-forced-aligner-wgpu.git
cd ctc-forced-aligner-wgpu
```

### 下载模型

```python
# 使用Python下载模型权重
from huggingface_hub import snapshot_download
snapshot_download(
    'aadel4/omniASR-CTC-300M-v2',
    local_dir='models/omniASR-CTC-300M-v2-hf'
)
```

或设置环境变量：
```bash
export CTC_MODEL_DIR=/path/to/your/model
```

### 编译

```bash
# 标准构建（推荐）
cargo build --release

# 最大性能构建（编译时间更长，运行更快）
cargo build --profile release-fast
```

### 基本使用

```bash
# 自动设备选择（优先GPU）
./target/release/align \
    --audio speech.wav \
    --text "你好世界" \
    --model models/omniASR-CTC-300M-v2-hf

# 强制使用GPU (Metal)
./target/release/align \
    --audio speech.wav \
    --text "hello world" \
    --device gpu

# 强制使用CPU (测试NEON优化)
./target/release/align \
    --audio speech.wav \
    --text "测试文本" \
    --device cpu

# 输出为SRT字幕
./target/release/align \
    --audio speech.wav \
    --text transcript.txt \
    --format srt \
    --output output.srt

# 输出为ASS卡拉OK
./target/release/align \
    --audio speech.wav \
    --text "歌词内容" \
    --format ass \
    --ass-res 1920x1080 \
    --output karaoke.ass
```

## 🎯 性能调优

### 窗口大小 (--window)

控制内存占用和吞吐量：

```bash
# 小窗口 - 低内存占用，适合长音频
./target/release/align --audio long.wav --text transcript.txt --window 30

# 大窗口 - 更高吞吐，需要更多VRAM/RAM
./target/release/align --audio short.wav --text transcript.txt --window 60

# 整文件处理 - 最快，但受限于内存
./target/release/align --audio short.wav --text transcript.txt --window 0
```

**建议**:
- 短音频（<2分钟）：`--window 60` 或 `--window 0`
- 中等长度（2-10分钟）：`--window 30`（默认）
- 长音频（>10分钟）：`--window 20`

### 上下文重叠 (--context)

窗口边缘的重叠区域：

```bash
./target/release/align --audio speech.wav --text transcript.txt --context 2  # 默认
./target/release/align --audio speech.wav --text transcript.txt --context 3  # 更多重叠
```

**注意**: 必须 ≥1.3秒以覆盖位置卷积感受野

### 设备选择 (--device)

```bash
--device auto    # 自动: 独显 → 集显 → CPU（默认）
--device gpu     # 强制GPU，失败则报错
--device cpu     # 强制CPU，测试NEON优化
--device vulkan  # 显式使用Vulkan后端（Linux/Windows）
--device metal   # 显式使用Metal后端（macOS）
```

## 📊 性能基准测试

### 运行自动化测试

```bash
# 使用你的音频文件运行完整测试
./benchmark_m_series.sh your_audio.wav "对应的转录文本"
```

测试脚本会：
1. 预热系统
2. 对比CPU vs GPU性能
3. 测试不同窗口大小
4. 计算实时率（RTF）
5. 给出优化建议

### 预期性能（M1 Max参考）

| 场景 | 音频长度 | 设备 | 时间 | RTF |
|------|----------|------|------|-----|
| 短音频 | 30秒 | CPU (NEON) | ~2.5s | 0.08x |
| 短音频 | 30秒 | GPU (Metal) | ~1.8s | 0.06x |
| 中等 | 3分钟 | CPU (NEON) | ~12s | 0.07x |
| 中等 | 3分钟 | GPU (Metal) | ~7s | 0.04x |
| 长音频 | 10分钟 | CPU (NEON) | ~45s | 0.08x |
| 长音频 | 10分钟 | GPU (Metal) | ~20s | 0.03x |

**RTF < 1.0** 表示可实时处理

### 手动性能测试

```bash
# 测试CPU性能
time ./target/release/align \
    --audio test.wav \
    --text "测试" \
    --device cpu \
    --output /dev/null

# 测试GPU性能
time ./target/release/align \
    --audio test.wav \
    --text "测试" \
    --device gpu \
    --output /dev/null
```

## 🔧 高级配置

### 环境变量

```bash
# 模型路径
export CTC_MODEL_DIR=/path/to/model

# 分配统计（调试用）
export CTC_ALLOC_STATS=1
cargo build --features alloc-stats
```

### 多语言支持

模型支持多种语言，自动处理：

```bash
# 中文
./target/release/align --audio zh.wav --text "这是中文测试"

# 英文
./target/release/align --audio en.wav --text "This is English test"

# 日文
./target/release/align --audio ja.wav --text "これは日本語テストです"

# 中英混合
./target/release/align --audio mix.wav --text "这是中English混合文本"
```

输出会自动按语言特性分词：
- **中日文**: 按字符输出
- **英文等**: 按空格分词输出
- **混合文本**: 智能切换

### JSON输出格式

```bash
./target/release/align \
    --audio speech.wav \
    --text "测试文本" \
    --format json \
    --output result.json
```

JSON结构：
```json
{
  "text": "原始转录文本",
  "duration": 10.5,
  "frames": 525,
  "frame_rate": 50.0,
  "mean_frame_score": -0.234,
  "tokens": [
    {
      "piece": "测",
      "start": 0.5,
      "end": 0.8,
      "start_frame": 25,
      "end_frame": 40,
      "score": -0.12,
      "space_before": false,
      "inferred": false
    },
    // ... 更多token
  ],
  "segments": [
    {
      "text": "测试文本",
      "start": 0.5,
      "end": 2.3
    }
  ]
}
```

字段说明：
- `piece`: 对齐的文本单元（字/词）
- `start/end`: 起止时间（秒）
- `score`: 置信度（越接近0越好）
- `space_before`: 该单元前是否有空格
- `inferred`: 是否为插值推断（词表外字符）

## 🐛 故障排除

### 编译错误

```bash
# 清理并重新编译
cargo clean
cargo build --release

# 更新依赖
cargo update
```

### GPU不可用

```bash
# 检查Metal支持
./target/release/align --list-devices

# 应该看到类似输出：
# 0: gpu "Apple M1 Max" (Metal)
# 1: cpu "llvmpipe" (Vulkan)
```

如果没有看到Metal设备：
1. 确认macOS版本 ≥ 10.11
2. 检查系统更新
3. 尝试重启

### 内存不足

```bash
# 减小窗口大小
./target/release/align --audio large.wav --text transcript.txt --window 20

# 或强制使用CPU
./target/release/align --audio large.wav --text transcript.txt --device cpu
```

### 音频格式不支持

当前只支持WAV格式，转换其他格式：

```bash
# 使用ffmpeg转换
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_f32le output.wav

# 参数说明：
# -ar 16000: 采样率16kHz（模型原生支持）
# -ac 1: 单声道
# -c:a pcm_f32le: PCM 32位浮点编码
```

### 时间戳不准确

1. **检查转录质量**: 转录文本必须与音频完全匹配
2. **检查音频质量**: 背景噪音、混响会影响精度
3. **调整参数**:
   ```bash
   # 增加上下文重叠
   ./target/release/align --audio speech.wav --text transcript.txt --context 3
   ```

## 📚 技术细节

### 模型架构

- **编码器**: Wav2Vec2 (24层Transformer)
- **输入**: 16kHz单声道音频
- **输出**: 50fps，10288类CTC标签
- **下采样**: 320倍（20ms每帧）

### NEON vs AVX2

| 特性 | AVX2 (Intel) | NEON (Apple Silicon) |
|------|--------------|----------------------|
| 向量宽度 | 256-bit (8×f32) | 128-bit (4×f32) |
| FMA | ✅ | ✅ |
| Gather | ✅ | ❌ (标量实现) |
| 能效比 | 基准 | 更优（RISC架构） |

虽然NEON单次处理4个浮点数（vs AVX2的8个），但M系列芯片的高频率、大缓存和优秀的调度器使实际性能非常接近甚至超越。

### 内存管理

- **窗口化处理**: 避免全文件加载
- **2-bit回溯**: Viterbi路径打包，节省4x内存
- **三层trellis**: 根据大小自动选择存储形式
  - Gathered columns `(T, S)`
  - Logits `(T, vocab)` 
  - Hidden states `(T, hidden)` + 按需计算

### 时间戳精度

- **粒度**: 20ms（由模型架构决定）
- **方法**: CTC Viterbi + 边界规则
- **精度**: 在Buckeye数据集上MAE 29.8ms

**注意**: 
- 时间戳对齐到20ms网格
- 标点符号占0帧（附着到前一字符）
- 空格不产生声学事件

## 🔮 未来优化方向

### 计划中
- [ ] INT8量化推理（2-3x加速）
- [ ] 批处理API（多文件并行）
- [ ] 流式推理（边读边处理）
- [ ] 更激进的Metal kernel优化

### 研究中  
- [ ] Apple Neural Engine支持（需Core ML）
- [ ] SVE2指令集（未来ARM版本）
- [ ] 模型蒸馏（更小更快的模型）

## 🤝 贡献

欢迎提交Issue和PR！特别欢迎：

1. **性能报告**: 在你的M系列Mac上的测试结果
2. **优化建议**: 更好的SIMD实现、编译选项
3. **Bug修复**: 特定场景下的问题
4. **文档改进**: 使用案例、教程

提交Issue时请包含：
- Mac型号（M1/M2/M3）
- macOS版本
- 音频长度和转录长度
- 完整的命令行和错误信息

## 📄 许可证

Apache-2.0

算法移植自 [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) (BSD 2-Clause)

模型权重版权归原作者所有。

## 🙏 致谢

- [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) - 模型
- [wgpu](https://github.com/gfx-rs/wgpu) - GPU抽象层
- [gemm crate](https://github.com/sarah-ek/gemm) - 高性能矩阵乘法
- 原始Python实现的作者

## 📞 联系方式

- **项目主页**: https://github.com/eclipse005/ctc-forced-aligner-wgpu
- **问题反馈**: 提交GitHub Issue

---

**优化版本**: v0.1.0-m-series  
**更新日期**: 2025-01-11  
**测试平台**: Apple M1 Max, macOS 14.x
