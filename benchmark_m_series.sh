#!/bin/bash
# M系列芯片性能测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CTC Forced Aligner - Apple Silicon 性能测试 ===${NC}\n"

# 检查是否在Apple Silicon上运行
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}警告: 此脚本专为Apple Silicon (ARM64)设计${NC}"
    echo -e "${YELLOW}当前架构: $(uname -m)${NC}\n"
fi

# 显示系统信息
echo -e "${GREEN}系统信息:${NC}"
echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
echo "核心数: $(sysctl -n hw.ncpu)"
echo "物理内存: $(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)GB"
echo "macOS版本: $(sw_vers -productVersion)"
echo ""

# 检查是否已编译
if [ ! -f "./target/release/align" ]; then
    echo -e "${YELLOW}未找到编译好的二进制文件，开始编译...${NC}"
    cargo build --release
    echo ""
fi

# 检查模型和测试音频
MODEL_DIR="${CTC_MODEL_DIR:-models/omniASR-CTC-300M-v2-hf}"
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}错误: 未找到模型目录: $MODEL_DIR${NC}"
    echo -e "${YELLOW}请设置 CTC_MODEL_DIR 环境变量或下载模型到默认路径${NC}"
    exit 1
fi

# 创建测试音频（如果不存在）
TEST_AUDIO="${1:-test_audio.wav}"
if [ ! -f "$TEST_AUDIO" ]; then
    echo -e "${YELLOW}警告: 未找到测试音频: $TEST_AUDIO${NC}"
    echo -e "${YELLOW}请提供测试音频文件作为第一个参数${NC}"
    echo -e "${YELLOW}用法: $0 <audio_file> [transcript]${NC}"
    exit 1
fi

TEST_TEXT="${2:-这是一段测试文本用于性能评估}"

echo -e "${GREEN}测试配置:${NC}"
echo "音频文件: $TEST_AUDIO"
echo "转录文本: $TEST_TEXT"
echo "模型路径: $MODEL_DIR"
echo ""

# 函数：运行单次测试
run_test() {
    local device=$1
    local name=$2
    local window=${3:-30}
    
    echo -e "${BLUE}--- 测试: $name (device=$device, window=${window}s) ---${NC}"
    
    # 预热运行（不计时）
    ./target/release/align \
        --audio "$TEST_AUDIO" \
        --text "$TEST_TEXT" \
        --device "$device" \
        --window "$window" \
        --model "$MODEL_DIR" \
        --output /dev/null 2>/dev/null || true
    
    # 正式测试（3次取平均）
    local total_time=0
    for i in {1..3}; do
        echo -n "  运行 $i/3... "
        local start=$(date +%s.%N)
        ./target/release/align \
            --audio "$TEST_AUDIO" \
            --text "$TEST_TEXT" \
            --device "$device" \
            --window "$window" \
            --model "$MODEL_DIR" \
            --output /dev/null 2>&1 > /dev/null
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc)
        total_time=$(echo "$total_time + $duration" | bc)
        echo -e "${GREEN}${duration}s${NC}"
    done
    
    local avg_time=$(echo "scale=3; $total_time / 3" | bc)
    echo -e "  ${GREEN}平均时间: ${avg_time}s${NC}"
    echo ""
    
    # 返回平均时间（通过全局变量）
    LAST_AVG_TIME=$avg_time
}

# 运行测试
echo -e "${BLUE}=== 开始性能测试 ===${NC}\n"

# 测试1: CPU推理（NEON优化）
run_test "cpu" "CPU推理 (NEON优化)" 30
CPU_TIME=$LAST_AVG_TIME

# 测试2: GPU推理（Metal）
run_test "gpu" "GPU推理 (Metal)" 30
GPU_TIME=$LAST_AVG_TIME

# 测试3: 自动选择
run_test "auto" "自动设备选择" 30
AUTO_TIME=$LAST_AVG_TIME

# 测试4: 大窗口GPU
if [ -n "$GPU_TIME" ]; then
    run_test "gpu" "GPU推理 (大窗口)" 60
    GPU_LARGE_TIME=$LAST_AVG_TIME
fi

# 总结
echo -e "${BLUE}=== 性能测试总结 ===${NC}\n"
echo -e "${GREEN}测试结果:${NC}"
echo "  CPU (NEON):       ${CPU_TIME}s"
echo "  GPU (Metal):      ${GPU_TIME}s"
echo "  Auto:             ${AUTO_TIME}s"
[ -n "$GPU_LARGE_TIME" ] && echo "  GPU (60s窗口):    ${GPU_LARGE_TIME}s"
echo ""

# 计算加速比
if [ -n "$CPU_TIME" ] && [ -n "$GPU_TIME" ]; then
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    echo -e "${GREEN}GPU相对CPU加速比: ${SPEEDUP}x${NC}"
fi

echo ""
echo -e "${GREEN}优化建议:${NC}"

# 根据结果给出建议
if (( $(echo "$GPU_TIME < $CPU_TIME" | bc -l) )); then
    echo "  ✓ GPU推理更快，建议使用 --device gpu"
else
    echo "  ✓ CPU推理更快（NEON优化有效），建议使用 --device cpu"
fi

if [ -n "$GPU_LARGE_TIME" ] && [ -n "$GPU_TIME" ]; then
    if (( $(echo "$GPU_LARGE_TIME < $GPU_TIME * 1.1" | bc -l) )); then
        echo "  ✓ 大窗口性能良好，可考虑使用 --window 60 减少开销"
    fi
fi

# 获取音频时长（如果有ffprobe）
if command -v ffprobe &> /dev/null; then
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TEST_AUDIO" 2>/dev/null || echo "unknown")
    if [ "$DURATION" != "unknown" ]; then
        RTF=$(echo "scale=3; $GPU_TIME / $DURATION" | bc)
        echo ""
        echo -e "${GREEN}实时率 (RTF):${NC}"
        echo "  音频时长: ${DURATION}s"
        echo "  处理时间: ${GPU_TIME}s"
        echo "  RTF: ${RTF}x (越小越好，<1.0表示可实时处理)"
    fi
fi

echo ""
echo -e "${BLUE}测试完成！${NC}"
echo -e "${YELLOW}提示: 更长的音频文件可能显示出更明显的优化效果${NC}"
