#!/bin/bash

# Script to run limitation visualization analysis
# Usage: ./run_limitation_analysis.sh

echo "Running Limitation Visualization Analysis..."

# Set parameters
MODEL_PATH="/mnt/backups/andycw/UDA-AI/ckps/source/uda/M58/C"
DATA_PATH="/mnt/backups/andycw/UDA-AI/data/M58/Real_all_nobg_augmented_79_list.txt"
SAVE_DIR="/mnt/backups/andycw/UDA-AI/visual/results"

# Create results directory
mkdir -p $SAVE_DIR

# Run the analysis
cd /mnt/backups/andycw/UDA-AI/visual

python limitation_visualize.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --net resnet101 \
    --class_num 79 \
    --batch_size 32 \
    --save_dir $SAVE_DIR \
    --num_workers 4

echo "Analysis complete! Check results in: $SAVE_DIR"