#!/bin/bash
# 运行 Pseudo Label 质量检查脚本

echo "开始检查 Pseudo Label 质量..."
echo "================================"

python check_pseudo_label_quality.py \
    --dset M58 \
    --s 0 \
    --t 1 \
    --batch_size 64 \
    --conf_thres 0.67 \
    --logit_scale 8.1 \
    --centroid_path /mnt/backups/andycw/UDA-AI/class_centroids_37.pth \
    --projector_path /mnt/backups/andycw/CLIP/output/m58/PureCLIP_Source_Model/rn101_projector_originCLIP_ep50_LR0.01_randomAug_37classes/source_projector.pt \
    --gpu_id 0 \
    --worker 4

echo ""
echo "检查完成！"
