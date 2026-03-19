#!/bin/bash

# 調試 Text Alignment Loss 的腳本

echo "========================================"
echo "開始調試 Text Alignment Loss"
echo "========================================"
echo ""

# 切換到正確的目錄
cd /mnt/backups/andycw/UDA-AI

# 激活 conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ATSSL

echo "運行調試程序..."
echo ""

python debug_text_loss.py

echo ""
echo "========================================"
echo "調試完成！"
echo "========================================"
echo ""
echo "根據調試結果調整參數："
echo "  如果平均最高機率 < 0.1："
echo "    --text_threshold 0.01"
echo "    --text_temp 0.5"
echo ""
echo "  如果平均最高機率在 0.1-0.5："
echo "    --text_threshold 0.1"
echo "    --text_temp 0.3"
echo ""
echo "  如果平均最高機率 > 0.5："
echo "    --text_threshold 0.3"
echo "    --text_temp 0.07"
