#!/bin/bash
# Pure CLIP Source Model 訓練腳本 (獨立版本)
# 切換資料集：把 DATASET 改成 m58 或 visda

set -e

# ========== 設定參數 ==========
# DATASET="m58"
DATASET="visda"

BACKBONE="RN101"
SEED=1
EPOCHS=50
BATCH_SIZE=32
BASE_LR=0.01

# 預設值（會由下方 case 覆蓋）
DATA_DIR=""
CLASSNAME_FILE=""
INFERENCE_DATA_DIR=""

case "$DATASET" in
    m58)
        DATA_DIR="/mnt/backups/andycw/M58/CAD_ratioFilter_StableDiffusion_37classes_new"
        CLASSNAME_FILE="/mnt/backups/andycw/M58/classname37.txt"
        INFERENCE_DATA_DIR="/mnt/backups/andycw/M58/Real_all_nobg"
        ;;
    visda)
        DATA_DIR="/mnt/backups/andycw/dataset/VISDA-C/train"
        CLASSNAME_FILE=""
        INFERENCE_DATA_DIR="/mnt/backups/andycw/dataset/VISDA-C/validation"
        ;;
    *)
        echo "❌ 不支援的 DATASET: $DATASET (只支援 m58 或 visda)"
        exit 1
        ;;
esac

# 輸出目錄
OUTPUT_DIR="output/${DATASET}/PureCLIP_Source_Model/${BACKBONE}_ep${EPOCHS}_lr${BASE_LR}_New"

set -- \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --backbone "${BACKBONE}" \
    --epochs "${EPOCHS}" \
    --lr "${BASE_LR}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --num-workers 4 \
    --inference-data-dir "${INFERENCE_DATA_DIR}" \
    --dataset-name "${DATASET}" \
    --use-cuda

if [ -n "${CLASSNAME_FILE}" ]; then
    set -- "$@" --classname-file "${CLASSNAME_FILE}"
fi

# 🚀 開始訓練
echo "Dataset: ${DATASET}"
echo "Data dir: ${DATA_DIR}"
python train_source.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 訓練完成！(已在訓練流程內完成 Real_all_nobg 推論+CSV 與 centroid 提取)"
else
    echo ""
    echo "❌ 訓練失敗"
    exit $EXIT_CODE
fi
