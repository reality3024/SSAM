# 運行訓練腳本（Resume 模式）
# Resume 自: /mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/37_class_100epoch_noPropagation
# 新訓練設定:
#   - con_par 固定為 0.1 (降低對比學習權重)
#   - prop_par 從 0 線性增長到 15 (逐步增強 Propagation Loss)
#   - 訓練 50 個 epoch
# --resume_dir /mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/37_class_100epoch_noPropagation \
# --start_epoch 100 \
# --phase1_epochs 100 \
python contrast_clip_projector_ema.py \
    --dset M58 \
    --s 0 \
    --t 1 \
    --max_epoch 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --conf_thres 0.9 \
    --cls_par 1.0 \
    --prop_par 0.1 \
    --con_par 1.0 \
    --ent_par 1.0 \
    --tt 0.05 \
    --ema_m 0.99 \
    --centroid_logitScale 10.0 \
    --phase1_epoch 101 \
    --centroid_path /mnt/backups/andycw/UDA-AI/class_centroids37_Jeannie.pth \
    --projector_path /mnt/backups/andycw/UDA-AI/ckps/source/uda/M58_37_Final/source_projector.pt \
    --gpu_id 0 \
    --worker 8 \
    --seed 2021