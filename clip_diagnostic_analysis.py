"""
Source-Free Domain Adaptation (SFDA) 診斷分析腳本
功能：
1. 特徵提取與全域置中 (Global Centering)
2. 動態 Threshold 計算與 Reliability Mask
3. Per-class Accuracy 與 Confusion Matrix
4. t-SNE 特徵空間視覺化 (Ground Truth & Reliability & Correctness)
5. 信心度分佈直方圖 (Confidence Distribution Histogram)
6. 預測 vs 真實類別分佈比較 (Predicted vs. True Class Distribution)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random

# 導入 CLIP 和 data_list
sys.path.append('/mnt/backups/andycw/CLIP')
sys.path.append('/mnt/backups/andycw/UDA-AI')
from clip import clip
from data_list import ImageList_idx


class MLPProjector(nn.Module):
    """MLP Projector: 512 → 1024 → 512"""
    def __init__(self, in_dim=512, out_dim=512, hidden_dim=1024):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def load_clip_to_cpu(backbone):
    """載入 CLIP 模型到 CPU"""
    model, preprocess = clip.load(backbone, device="cpu", download_root=os.path.expanduser("~/.cache/clip"))
    return model, preprocess


def create_target_dataloader(target_list_path, clip_preprocess, batch_size=64, root_path='/mnt/backups/andycw/'):
    """創建 Target Domain Test DataLoader（使用 CLIP 官方 preprocess）"""
    print(f"\n{'='*60}")
    print("創建 Target DataLoader")
    print(f"{'='*60}")

    txt_target = open(target_list_path).readlines()

    # 使用 CLIP 官方的 preprocess（沒有任何增強）
    target_dataset = ImageList_idx(txt_target, transform=clip_preprocess, root=root_path)
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    print(f"  ✓ Target 數據: {len(target_dataset)} 張")
    print(f"  ✓ Batch size: {batch_size}")
    print(f"  ✓ 使用 CLIP 官方 preprocess (無增強)")
    print(f"{'='*60}\n")

    return target_loader


def load_projector(projector_path, device='cuda'):
    """載入訓練好的 Teacher Projector"""
    print(f"\n{'='*60}")
    print("載入 Teacher Projector")
    print(f"{'='*60}")

    projector = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).to(device)
    state_dict = torch.load(projector_path, map_location=device)
    projector.load_state_dict(state_dict)
    projector.eval()

    print(f"  ✓ Projector 載入成功")
    print(f"  ✓ 架構: Linear(512, 1024) -> BatchNorm1d -> ReLU -> Linear(1024, 512)")
    print(f"  ✓ 模式: eval() (已凍結)")
    print(f"  ✓ 路徑: {projector_path}")
    print(f"{'='*60}\n")

    return projector


def load_source_centroids(centroids_path):
    """載入 Source Centroids (37 classes, [37, 512])"""
    print(f"\n{'='*60}")
    print("載入 Source Centroids")
    print(f"{'='*60}")

    data = torch.load(centroids_path)
    source_centroids = data['centroids']  # [37, 512]

    # 嘗試載入類別名稱（如果存在）
    classnames = data.get('classnames', None)
    if classnames is None:
        # 如果沒有類別名稱，使用 Class 0, Class 1, ... 作為默認名稱
        classnames = [f'Class {i}' for i in range(len(source_centroids))]
        print(f"  ⚠️  未找到類別名稱，使用默認名稱")

    # L2 正規化
    source_centroids = F.normalize(source_centroids, dim=-1)

    print(f"  ✓ Source Centroids 載入成功")
    print(f"  ✓ Shape: {source_centroids.shape}")
    print(f"  ✓ 類別數量: {len(classnames)}")
    print(f"  ✓ 已 L2 正規化")
    print(f"  ✓ 路徑: {centroids_path}")
    print(f"{'='*60}\n")

    return source_centroids.cuda(), classnames


def extract_features_with_centering(clip_model, projector, target_loader, source_centroids, logit_scale=7.1):
    """
    提取 Target 特徵並執行全域置中 (Global Centering)

    數學邏輯（嚴格執行）：
    1. target_global_mean = target_features.mean(dim=0, keepdim=True)
    2. source_global_mean = source_centroids.mean(dim=0, keepdim=True)
    3. centered_target = F.normalize(target_features - target_global_mean, dim=-1)
    4. centered_source = F.normalize(source_centroids - source_global_mean, dim=-1)
    5. logits = logit_scale * (centered_target @ centered_source.T)

    Returns:
        target_features: [N, 512] 原始 Target 特徵 (L2 normalized)
        centered_target: [N, 512] 置中後的 Target 特徵 (L2 normalized)
        centered_source: [37, 512] 置中後的 Source Centroids (L2 normalized)
        true_labels: [N] GT labels
        pseudo_labels: [N] 預測的偽標籤
        max_probs: [N] 預測的最大機率
        reliable_mask: [N] 可靠性遮罩 (bool)
        conf_threshold: float 動態門檻值
    """
    print(f"\n{'='*60}")
    print("提取特徵並執行全域置中 (Global Centering)")
    print(f"{'='*60}")
    print(f"  - Logit Scale (寫死): {logit_scale}")
    print(f"  - 底線保護門檻: 0.60")
    print(f"{'='*60}\n")

    clip_model.eval()
    projector.eval()

    # 預先分配空間
    dataset_size = len(target_loader.dataset)
    all_features = torch.zeros(dataset_size, 512)
    all_true_labels = torch.zeros(dataset_size, dtype=torch.long)

    print("步驟 1: 提取所有 Target 特徵...")
    with torch.no_grad():
        for imgs, labels, indices, _ in tqdm(target_loader, desc="  提取特徵"):
            imgs = imgs.cuda()

            # CLIP Encoder → Projector → L2 Normalize
            feat_clip = clip_model.visual(imgs.type(clip_model.dtype))
            feat_projected = projector(feat_clip)
            feat_norm = F.normalize(feat_projected, dim=-1)

            # 使用 indices 確保對應關係正確
            all_features[indices] = feat_norm.cpu()
            all_true_labels[indices] = labels.cpu()

    print(f"  ✓ 特徵提取完成: {all_features.shape}")
    print(f"  ✓ 使用明確索引 (indices) 確保樣本對應正確\n")

    # 步驟 2: 全域置中 (Global Centering) - 嚴格遵守數學邏輯
    print("步驟 2: 執行全域置中 (Global Centering)...")

    # 2.1 計算全域平均
    target_global_mean = all_features.mean(dim=0, keepdim=True)  # [1, 512]
    source_global_mean = source_centroids.cpu().mean(dim=0, keepdim=True)  # [1, 512]

    print(f"  ✓ target_global_mean: {target_global_mean.shape}")
    print(f"  ✓ source_global_mean: {source_global_mean.shape}")

    # 2.2 消除全域偏差（置中）
    centered_target_before_norm = all_features - target_global_mean  # [N, 512]
    centered_source_before_norm = source_centroids.cpu() - source_global_mean  # [37, 512]

    # 2.3 重新 L2 正規化
    centered_target = F.normalize(centered_target_before_norm, dim=-1)  # [N, 512]
    centered_source = F.normalize(centered_source_before_norm, dim=-1)  # [37, 512]

    print(f"  ✓ centered_target: {centered_target.shape}")
    print(f"  ✓ centered_source: {centered_source.shape}")
    print(f"  ✓ 全域置中完成\n")

    # 步驟 3: 計算 Logits 與動態 Threshold (使用黃金參數)
    print("步驟 3: 計算 Logits 與動態 Threshold...")

    # 3.1 計算 logits
    logits = logit_scale * (centered_target @ centered_source.T)  # [N, 37]
    print(f"  ✓ logits shape: {logits.shape}")

    # 3.2 計算機率與預測
    probs = F.softmax(logits, dim=1)  # [N, 37]
    max_probs, pseudo_labels = torch.max(probs, dim=1)  # [N], [N]
    print(f"  ✓ max_probs: {max_probs.shape}")
    print(f"  ✓ pseudo_labels: {pseudo_labels.shape}")

    # 3.3 計算動態門檻
    conf_threshold = max(max_probs.mean().item(), 0.90)
    print(f"  ✓ 動態門檻 = max(max_probs.mean(), 0.90)")
    print(f"  ✓ max_probs.mean() = {max_probs.mean().item():.4f}")
    print(f"  ✓ conf_threshold = {conf_threshold:.4f}")

    # 3.4 產生遮罩
    reliable_mask = max_probs > conf_threshold  # [N], boolean

    # 統計數據
    total_samples = len(all_features)
    reliable_samples = reliable_mask.sum().item()
    unreliable_samples = total_samples - reliable_samples

    print(f"\n  【可靠性統計】")
    print(f"  - 總樣本數: {total_samples}")
    print(f"  - Reliable 樣本: {reliable_samples} ({reliable_samples/total_samples*100:.2f}%)")
    print(f"  - Unreliable 樣本: {unreliable_samples} ({unreliable_samples/total_samples*100:.2f}%)")

    # 計算 Reliable 樣本的純度
    reliable_correct = ((pseudo_labels == all_true_labels) & reliable_mask).sum().item()
    if reliable_samples > 0:
        purity = reliable_correct / reliable_samples * 100
        print(f"  - Reliable 樣本純度: {purity:.2f}%")

    print(f"{'='*60}\n")

    return (all_features, centered_target, centered_source,
            all_true_labels, pseudo_labels, max_probs,
            reliable_mask, conf_threshold)


def save_class_mapping(classnames, num_classes=37, output_dir='./output'):
    """
    保存 ClassID 到 ClassName 的映射到 CSV 文件

    Args:
        classnames: list[str] 類別名稱列表
        num_classes: int 類別數量
        output_dir: str 輸出目錄
    """
    print(f"\n{'='*60}")
    print("生成 ClassID 到 ClassName 的映射表")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 創建 DataFrame
    df = pd.DataFrame({
        'ClassID': list(range(num_classes)),
        'ClassName': classnames
    })

    # 保存到 CSV
    csv_path = os.path.join(output_dir, 'class_id_to_name.csv')
    df.to_csv(csv_path, index=False)

    print(f"  ✓ ClassID 到 ClassName 映射表已保存: {csv_path}")
    print(f"  ✓ 總共 {num_classes} 個類別")
    print(f"\n  【前 10 個類別】")
    print(df.head(10).to_string(index=False))
    print(f"{'='*60}\n")

    return csv_path


def compute_confusion_matrix_and_accuracy(true_labels, pseudo_labels, num_classes=37, output_dir='./output'):
    """
    計算 Per-class Accuracy 和 Confusion Matrix

    Args:
        true_labels: [N] Ground truth labels
        pseudo_labels: [N] Predicted pseudo labels
        num_classes: int 類別數量
        output_dir: str 輸出目錄
    """
    print(f"\n{'='*60}")
    print("計算 Per-class Accuracy & Confusion Matrix")
    print(f"{'='*60}\n")

    # 轉換為 numpy
    true_labels_np = true_labels.numpy()
    pseudo_labels_np = pseudo_labels.numpy()

    # 計算混淆矩陣
    cm = confusion_matrix(true_labels_np, pseudo_labels_np, labels=list(range(num_classes)))

    # 計算 Per-class Accuracy (Recall)
    per_class_acc = []
    severe_tail_classes = []

    print("Per-class Accuracy (Recall):")
    print(f"{'Class':<8} {'Accuracy (%)':<15} {'Samples':<10} {'Status':<20}")
    print("-" * 60)

    for i in range(num_classes):
        total = cm[i, :].sum()
        if total > 0:
            acc = cm[i, i] / total * 100
        else:
            acc = 0.0

        per_class_acc.append(acc)

        # 標記嚴重長尾類別 (< 10%)
        status = ""
        if acc < 10.0:
            status = "⚠️  嚴重長尾"
            severe_tail_classes.append(i)

        print(f"{i:<8} {acc:<15.2f} {total:<10} {status:<20}")

    # 計算平均準確率
    mean_acc = np.mean(per_class_acc)
    print("-" * 60)
    print(f"{'Mean':<8} {mean_acc:<15.2f}")
    print(f"\n嚴重長尾類別 (Accuracy < 10%): {severe_tail_classes}")
    print(f"共 {len(severe_tail_classes)} 個類別")

    # 保存 CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'per_class_accuracy.csv')

    df = pd.DataFrame({
        'Class': list(range(num_classes)),
        'Accuracy (%)': per_class_acc,
        'Total Samples': [cm[i, :].sum() for i in range(num_classes)]
    })
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Per-class Accuracy 已保存: {csv_path}")

    # 繪製混淆矩陣 (Confusion Matrix Heatmap)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True,
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title(f'Confusion Matrix (37x37)\nMean Accuracy: {mean_acc:.2f}%', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion Matrix 已保存: {cm_path}")

    print(f"{'='*60}\n")

    return per_class_acc, cm


def separate_close_points(points, min_distance=3.0, max_iterations=100, repulsion_strength=0.5):
    """
    使用排斥力算法分開距離太近的點

    Args:
        points: numpy array, 形狀為 [N, 2]，需要調整的點
        min_distance: float, 最小允許距離（小於此距離的點會被推開）
        max_iterations: int, 最大迭代次數
        repulsion_strength: float, 排斥力強度（控制每次移動的幅度）

    Returns:
        adjusted_points: numpy array, 形狀為 [N, 2]，調整後的點
        num_iterations: int, 實際迭代次數
    """
    from scipy.spatial.distance import pdist, squareform

    points = points.copy()  # 避免修改原始數據
    n_points = len(points)

    for iteration in range(max_iterations):
        # 計算所有點之間的距離
        distances = squareform(pdist(points))

        # 記錄是否有移動
        has_movement = False

        # 對每個點施加排斥力
        for i in range(n_points):
            # 找出太近的點（排除自己）
            close_mask = (distances[i] < min_distance) & (distances[i] > 0)
            close_indices = np.where(close_mask)[0]

            if len(close_indices) == 0:
                continue

            has_movement = True

            # 計算排斥力向量
            repulsion_vector = np.zeros(2)
            for j in close_indices:
                # 從 j 到 i 的向量
                diff = points[i] - points[j]
                dist = distances[i, j]

                if dist < 1e-6:  # 避免除零，如果兩點完全重疊
                    # 給一個隨機方向
                    angle = np.random.rand() * 2 * np.pi
                    diff = np.array([np.cos(angle), np.sin(angle)])
                    dist = 1e-6

                # 歸一化方向向量，並根據距離計算排斥力
                # 距離越近，排斥力越大
                force = (min_distance - dist) / min_distance
                repulsion_vector += (diff / dist) * force

            # 應用排斥力（移動當前點）
            points[i] += repulsion_vector * repulsion_strength

        # 如果沒有移動，提前結束
        if not has_movement:
            return points, iteration + 1

    return points, max_iterations


def visualize_tsne(centered_target, centered_source, true_labels, pseudo_labels,
                   reliable_mask, num_classes=37, output_dir='./output',
                   separate_centroids=True, min_centroid_distance=3.0):
    """
    使用 t-SNE 進行特徵空間視覺化

    生成兩張圖：
    - 圖 A (Ground Truth): Source Centroids (黑色星號) + Target 樣本 (依 true_labels 上色)
    - 圖 B (Reliability): Source Centroids (黑色星號) + Unreliable (灰色) + Reliable (依 pseudo_labels 上色)

    Args:
        centered_target: [N, 512] 置中後的 Target 特徵
        centered_source: [37, 512] 置中後的 Source Centroids
        true_labels: [N] Ground truth labels
        pseudo_labels: [N] Predicted pseudo labels
        reliable_mask: [N] Reliability mask (bool)
        num_classes: int 類別數量
        output_dir: str 輸出目錄
        separate_centroids: bool 是否自動分開太近的 Source Centroids（默認 True）
        min_centroid_distance: float Source Centroids 之間的最小距離（默認 3.0）
    """
    print(f"\n{'='*60}")
    print("執行 t-SNE 特徵空間視覺化")
    print(f"{'='*60}\n")

    # 合併 Source 和 Target 特徵
    print("步驟 1: 合併特徵並執行 t-SNE 降維...")
    combined_features = torch.cat([centered_source, centered_target], dim=0).numpy()  # [37+N, 512]
    print(f"  ✓ 合併特徵 shape: {combined_features.shape}")
    print(f"    - Source Centroids: {centered_source.shape[0]}")
    print(f"    - Target 樣本: {centered_target.shape[0]}")

    # t-SNE 降維
    print(f"\n  ⏳ 執行 t-SNE (這可能需要一些時間)...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(combined_features)  # [37+N, 2]
    print(f"  ✓ t-SNE 降維完成: {X_embedded.shape}\n")

    # 分離 Source 和 Target 的降維結果
    source_embedded = X_embedded[:num_classes, :]  # [37, 2]
    target_embedded = X_embedded[num_classes:, :]  # [N, 2]

    # 【新增】自動分開距離太近的 Source Centroids
    if separate_centroids:
        print("步驟 2: 自動分開距離太近的 Source Centroids...")
        from scipy.spatial.distance import pdist, squareform

        # 檢查初始狀態
        initial_distances = squareform(pdist(source_embedded))
        close_pairs_before = np.sum((initial_distances < min_centroid_distance) & (initial_distances > 0)) // 2

        print(f"  - 最小距離閾值: {min_centroid_distance:.2f}")
        print(f"  - 分離前：{close_pairs_before} 對點距離 < {min_centroid_distance:.2f}")

        # 應用排斥力算法
        source_embedded, num_iterations = separate_close_points(
            source_embedded,
            min_distance=min_centroid_distance,
            max_iterations=100,
            repulsion_strength=0.5
        )

        # 檢查分離後的狀態
        final_distances = squareform(pdist(source_embedded))
        close_pairs_after = np.sum((final_distances < min_centroid_distance) & (final_distances > 0)) // 2

        print(f"  - 迭代次數: {num_iterations}")
        print(f"  - 分離後：{close_pairs_after} 對點距離 < {min_centroid_distance:.2f}")
        print(f"  ✓ 成功分開 {close_pairs_before - close_pairs_after} 對重疊點\n")
    else:
        print("步驟 2: 跳過 Source Centroids 分離（separate_centroids=False）\n")

    # 為 Source Centroids 標註計算智能偏移（避免星星之間的標註重疊）
    distances = squareform(pdist(source_embedded))

    # 計算每個星星的偏移方向
    star_offsets = {}
    for i in range(num_classes):
        # 找出與當前星星距離很近的其他星星
        close_stars = np.where((distances[i] < 5.0) & (distances[i] > 0))[0]

        if len(close_stars) > 0:
            # 如果有很近的星星，使用放射狀偏移
            angle = (i % 8) * (2 * np.pi / 8)  # 8 個方向
            offset_x = 10 * np.cos(angle)
            offset_y = 10 * np.sin(angle)
        else:
            # 沒有重疊，使用默認偏移
            offset_x, offset_y = 0, 0

        star_offsets[i] = (offset_x, offset_y)

    # 設定顏色 palette (使用 37 種顏色)
    # 使用多個 palette 組合以獲得足夠的顏色
    colors_tab20 = plt.cm.tab20.colors  # 20 colors
    colors_tab20b = plt.cm.tab20b.colors  # 20 colors
    colors_tab20c = plt.cm.tab20c.colors  # 20 colors
    all_colors = list(colors_tab20) + list(colors_tab20b) + list(colors_tab20c)
    color_palette = all_colors[:num_classes]  # 取前 37 種顏色

    os.makedirs(output_dir, exist_ok=True)

    # ========== 圖 A: Ground Truth 分佈 ==========
    print("步驟 3: 繪製圖 A (Ground Truth 分佈)...")

    plt.figure(figsize=(16, 14))

    # 繪製 Target 樣本 (依 true_labels 上色)
    for class_id in range(num_classes):
        mask = (true_labels.numpy() == class_id)
        if mask.sum() > 0:
            plt.scatter(target_embedded[mask, 0], target_embedded[mask, 1],
                       c=[color_palette[class_id]], alpha=0.6, s=30,
                       label=f'Class {class_id}')

    # 在 Source Centroids 位置標註類別 ID（黑色圓形背景，使用智能偏移避免重疊）
    for i in range(num_classes):
        offset_x, offset_y = star_offsets[i]
        plt.annotate(str(i),
                    xy=(source_embedded[i, 0], source_embedded[i, 1]),
                    xytext=(offset_x, offset_y),
                    textcoords='offset points',
                    fontsize=8, ha='center', va='center',
                    color='white', weight='bold', zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.5', facecolor='black',
                             edgecolor='white', linewidth=1.5, alpha=0.9))

    # 添加 Source Centroids 圖例（使用虛擬點）
    plt.scatter([], [], c='black', marker='o', s=100,
               edgecolors='white', linewidths=1.5, label='Source Centroids')

    # 為每個 Class 隨機挑一個點標註 Class ID
    for class_id in range(num_classes):
        mask = (true_labels.numpy() == class_id)
        if mask.sum() > 0:
            # 獲取該類別的所有點的索引
            class_indices = np.where(mask)[0]
            # 隨機選擇一個索引
            random_idx = random.choice(class_indices)
            # 標註 Class ID（添加偏移量避免與星星重疊）
            plt.annotate(f'C{class_id}',
                        xy=(target_embedded[random_idx, 0], target_embedded[random_idx, 1]),
                        xytext=(0, 15),  # 向上偏移 15 個點（避免與星星重疊）
                        textcoords='offset points',
                        fontsize=9, ha='center', va='bottom',
                        color=color_palette[class_id], weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=color_palette[class_id], alpha=0.8),
                        arrowprops=dict(arrowstyle='-', color=color_palette[class_id],
                                       lw=0.5, alpha=0.6))

    plt.title("t-SNE Visualization: Ground Truth Distribution", fontsize=18, weight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(loc='upper right', fontsize=8, ncol=3, bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    plt.tight_layout()

    fig_a_path = os.path.join(output_dir, 'tsne_ground_truth.png')
    plt.savefig(fig_a_path, dpi=200)
    plt.close()
    print(f"  ✓ 圖 A 已保存: {fig_a_path}\n")

    # ========== 圖 B: Reliability 與預測分佈 ==========
    print("步驟 4: 繪製圖 B (Reliability 與預測分佈)...")

    plt.figure(figsize=(16, 14))

    # 繪製 Unreliable 樣本 (灰色半透明小點)
    unreliable_mask_np = ~reliable_mask.numpy()
    if unreliable_mask_np.sum() > 0:
        plt.scatter(target_embedded[unreliable_mask_np, 0],
                   target_embedded[unreliable_mask_np, 1],
                   c='gray', alpha=0.3, s=20, label='Unreliable')

    # 繪製 Reliable 樣本 (依 pseudo_labels 上色，加深顏色)
    reliable_mask_np = reliable_mask.numpy()
    for class_id in range(num_classes):
        mask = (pseudo_labels.numpy() == class_id) & reliable_mask_np
        if mask.sum() > 0:
            plt.scatter(target_embedded[mask, 0], target_embedded[mask, 1],
                       c=[color_palette[class_id]], alpha=0.8, s=50,
                       label=f'Reliable Class {class_id}',
                       edgecolors='black', linewidths=0.5)

    # 在 Source Centroids 位置標註類別 ID（黑色圓形背景，使用智能偏移避免重疊）
    for i in range(num_classes):
        offset_x, offset_y = star_offsets[i]
        plt.annotate(str(i),
                    xy=(source_embedded[i, 0], source_embedded[i, 1]),
                    xytext=(offset_x, offset_y),
                    textcoords='offset points',
                    fontsize=8, ha='center', va='center',
                    color='white', weight='bold', zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.5', facecolor='black',
                             edgecolor='white', linewidth=1.5, alpha=0.9))

    # 添加 Source Centroids 圖例（使用虛擬點）
    plt.scatter([], [], c='black', marker='o', s=100,
               edgecolors='white', linewidths=1.5, label='Source Centroids')

    # 為每個 Reliable Class 隨機挑一個點標註 Class ID
    for class_id in range(num_classes):
        mask = (pseudo_labels.numpy() == class_id) & reliable_mask_np
        if mask.sum() > 0:
            # 獲取該類別的所有點的索引
            class_indices = np.where(mask)[0]
            # 隨機選擇一個索引
            random_idx = random.choice(class_indices)
            # 標註 Class ID（添加偏移量避免與星星重疊）
            plt.annotate(f'C{class_id}',
                        xy=(target_embedded[random_idx, 0], target_embedded[random_idx, 1]),
                        xytext=(0, 15),  # 向上偏移 15 個點（避免與星星重疊）
                        textcoords='offset points',
                        fontsize=9, ha='center', va='bottom',
                        color=color_palette[class_id], weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=color_palette[class_id], alpha=0.8),
                        arrowprops=dict(arrowstyle='-', color=color_palette[class_id],
                                       lw=0.5, alpha=0.6))

    # 統計信息
    reliable_count = reliable_mask_np.sum()
    unreliable_count = unreliable_mask_np.sum()
    total_count = len(reliable_mask_np)

    plt.title(f"t-SNE Visualization: Reliability & Prediction Distribution\n"
             f"Reliable: {reliable_count}/{total_count} ({reliable_count/total_count*100:.2f}%), "
             f"Unreliable: {unreliable_count}/{total_count} ({unreliable_count/total_count*100:.2f}%)",
             fontsize=18, weight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(loc='upper right', fontsize=8, ncol=3, bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    plt.tight_layout()

    fig_b_path = os.path.join(output_dir, 'tsne_reliability.png')
    plt.savefig(fig_b_path, dpi=200)
    plt.close()
    print(f"  ✓ 圖 B 已保存: {fig_b_path}")

    print(f"{'='*60}\n")


def visualize_tsne_correctness(centered_target, centered_source, true_labels, pseudo_labels,
                               num_classes=37, output_dir='./output',
                               separate_centroids=True, min_centroid_distance=3.0):
    """
    使用 t-SNE 進行特徵空間視覺化 - 按預測正確性上色

    顯示模型預測的正確性分佈：
    - 正確預測 (有顏色): pseudo_labels == true_labels，按類別上色
    - 錯誤預測 (灰色): pseudo_labels != true_labels

    Args:
        centered_target: [N, 512] 置中後的 Target 特徵
        centered_source: [37, 512] 置中後的 Source Centroids
        true_labels: [N] Ground truth labels
        pseudo_labels: [N] Predicted pseudo labels
        num_classes: int 類別數量
        output_dir: str 輸出目錄
        separate_centroids: bool 是否自動分開太近的 Source Centroids（默認 True）
        min_centroid_distance: float Source Centroids 之間的最小距離（默認 3.0）
    """
    print(f"\n{'='*60}")
    print("執行 t-SNE 特徵空間視覺化 (按預測正確性)")
    print(f"{'='*60}\n")

    # 合併 Source 和 Target 特徵
    print("步驟 1: 合併特徵並執行 t-SNE 降維...")
    combined_features = torch.cat([centered_source, centered_target], dim=0).numpy()  # [37+N, 512]
    print(f"  ✓ 合併特徵 shape: {combined_features.shape}")
    print(f"    - Source Centroids: {centered_source.shape[0]}")
    print(f"    - Target 樣本: {centered_target.shape[0]}")

    # t-SNE 降維
    print(f"\n  ⏳ 執行 t-SNE (這可能需要一些時間)...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(combined_features)  # [37+N, 2]
    print(f"  ✓ t-SNE 降維完成: {X_embedded.shape}\n")

    # 分離 Source 和 Target 的降維結果
    source_embedded = X_embedded[:num_classes, :]  # [37, 2]
    target_embedded = X_embedded[num_classes:, :]  # [N, 2]

    # 自動分開距離太近的 Source Centroids
    if separate_centroids:
        print("步驟 2: 自動分開距離太近的 Source Centroids...")
        from scipy.spatial.distance import pdist, squareform

        # 檢查初始狀態
        initial_distances = squareform(pdist(source_embedded))
        close_pairs_before = np.sum((initial_distances < min_centroid_distance) & (initial_distances > 0)) // 2

        print(f"  - 最小距離閾值: {min_centroid_distance:.2f}")
        print(f"  - 分離前：{close_pairs_before} 對點距離 < {min_centroid_distance:.2f}")

        # 應用排斥力算法
        source_embedded, num_iterations = separate_close_points(
            source_embedded,
            min_distance=min_centroid_distance,
            max_iterations=100,
            repulsion_strength=0.5
        )

        # 檢查分離後的狀態
        final_distances = squareform(pdist(source_embedded))
        close_pairs_after = np.sum((final_distances < min_centroid_distance) & (final_distances > 0)) // 2

        print(f"  - 迭代次數: {num_iterations}")
        print(f"  - 分離後：{close_pairs_after} 對點距離 < {min_centroid_distance:.2f}")
        print(f"  ✓ 成功分開 {close_pairs_before - close_pairs_after} 對重疊點\n")
    else:
        print("步驟 2: 跳過 Source Centroids 分離（separate_centroids=False）\n")

    # 為 Source Centroids 標註計算智能偏移
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(source_embedded))

    # 計算每個星星的偏移方向
    star_offsets = {}
    for i in range(num_classes):
        # 找出與當前星星距離很近的其他星星
        close_stars = np.where((distances[i] < 5.0) & (distances[i] > 0))[0]

        if len(close_stars) > 0:
            # 如果有很近的星星，使用放射狀偏移
            angle = (i % 8) * (2 * np.pi / 8)  # 8 個方向
            offset_x = 10 * np.cos(angle)
            offset_y = 10 * np.sin(angle)
        else:
            # 沒有重疊，使用默認偏移
            offset_x, offset_y = 0, 0

        star_offsets[i] = (offset_x, offset_y)

    # 設定顏色 palette (使用 37 種顏色)
    colors_tab20 = plt.cm.tab20.colors  # 20 colors
    colors_tab20b = plt.cm.tab20b.colors  # 20 colors
    colors_tab20c = plt.cm.tab20c.colors  # 20 colors
    all_colors = list(colors_tab20) + list(colors_tab20b) + list(colors_tab20c)
    color_palette = all_colors[:num_classes]  # 取前 37 種顏色

    os.makedirs(output_dir, exist_ok=True)

    # ========== 繪製預測正確性視覺化 ==========
    print("步驟 3: 繪製預測正確性視覺化...")

    plt.figure(figsize=(16, 14))

    # 計算正確性遮罩
    correct_mask = (pseudo_labels == true_labels).numpy()
    incorrect_mask = ~correct_mask

    # 統計數據
    correct_count = correct_mask.sum()
    incorrect_count = incorrect_mask.sum()
    total_count = len(correct_mask)
    accuracy = correct_count / total_count * 100

    print(f"  - 正確預測: {correct_count}/{total_count} ({accuracy:.2f}%)")
    print(f"  - 錯誤預測: {incorrect_count}/{total_count} ({100-accuracy:.2f}%)")

    # 繪製錯誤預測的樣本 (灰色)
    if incorrect_mask.sum() > 0:
        plt.scatter(target_embedded[incorrect_mask, 0],
                   target_embedded[incorrect_mask, 1],
                   c='gray', alpha=0.4, s=25, label='Incorrect Prediction',
                   edgecolors='none')

    # 繪製正確預測的樣本 (依 true_labels 上色)
    for class_id in range(num_classes):
        mask = (true_labels.numpy() == class_id) & correct_mask
        if mask.sum() > 0:
            plt.scatter(target_embedded[mask, 0], target_embedded[mask, 1],
                       c=[color_palette[class_id]], alpha=0.8, s=50,
                       label=f'Correct Class {class_id}',
                       edgecolors='black', linewidths=0.5)

    # 在 Source Centroids 位置標註類別 ID（黑色圓形背景）
    for i in range(num_classes):
        offset_x, offset_y = star_offsets[i]
        plt.annotate(str(i),
                    xy=(source_embedded[i, 0], source_embedded[i, 1]),
                    xytext=(offset_x, offset_y),
                    textcoords='offset points',
                    fontsize=8, ha='center', va='center',
                    color='white', weight='bold', zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.5', facecolor='black',
                             edgecolor='white', linewidth=1.5, alpha=0.9))

    # 添加 Source Centroids 圖例
    plt.scatter([], [], c='black', marker='o', s=100,
               edgecolors='white', linewidths=1.5, label='Source Centroids')

    # 為每個正確預測的類別隨機挑一個點標註 Class ID
    for class_id in range(num_classes):
        mask = (true_labels.numpy() == class_id) & correct_mask
        if mask.sum() > 0:
            # 獲取該類別的所有點的索引
            class_indices = np.where(mask)[0]
            # 隨機選擇一個索引
            random_idx = random.choice(class_indices)
            # 標註 Class ID
            plt.annotate(f'C{class_id}',
                        xy=(target_embedded[random_idx, 0], target_embedded[random_idx, 1]),
                        xytext=(0, 15),
                        textcoords='offset points',
                        fontsize=9, ha='center', va='bottom',
                        color=color_palette[class_id], weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=color_palette[class_id], alpha=0.8),
                        arrowprops=dict(arrowstyle='-', color=color_palette[class_id],
                                       lw=0.5, alpha=0.6))

    plt.title(f"t-SNE Visualization: Prediction Correctness\n"
             f"Correct: {correct_count}/{total_count} ({accuracy:.2f}%), "
             f"Incorrect: {incorrect_count}/{total_count} ({100-accuracy:.2f}%)",
             fontsize=18, weight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(loc='upper right', fontsize=8, ncol=3, bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'tsne_correctness.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  ✓ 預測正確性視覺化已保存: {fig_path}")

    print(f"{'='*60}\n")


def plot_confidence_histogram(max_probs, conf_threshold, output_dir='./output'):
    """
    繪製信心度分佈直方圖

    顯示所有樣本的最大預測機率分佈，並標示出當前的信心度閾值。
    包含詳細的統計信息（均值、中位數、標準差等）。

    Args:
        max_probs: torch.Tensor or numpy array, 形狀為 [N]，包含所有 Target 樣本的最大預測機率
        conf_threshold: float, 當前的動態或固定閾值
        output_dir: str, 輸出目錄
    """
    print(f"\n{'='*60}")
    print("繪製信心度分佈直方圖")
    print(f"{'='*60}\n")

    # 轉換為 numpy
    max_probs_np = max_probs.numpy() if isinstance(max_probs, torch.Tensor) else max_probs

    # 計算統計數據
    mean_conf = np.mean(max_probs_np)
    median_conf = np.median(max_probs_np)
    std_conf = np.std(max_probs_np)
    min_conf = np.min(max_probs_np)
    max_conf = np.max(max_probs_np)

    print(f"  信心度統計：")
    print(f"    - 平均值: {mean_conf:.4f}")
    print(f"    - 中位數: {median_conf:.4f}")
    print(f"    - 標準差: {std_conf:.4f}")
    print(f"    - 最小值: {min_conf:.4f}")
    print(f"    - 最大值: {max_conf:.4f}")
    print(f"    - Threshold: {conf_threshold:.4f}")

    # 計算高於/低於閾值的樣本數
    above_threshold = (max_probs_np > conf_threshold).sum()
    below_threshold = (max_probs_np <= conf_threshold).sum()
    total_samples = len(max_probs_np)

    print(f"\n  樣本分佈：")
    print(f"    - 高於閾值 (Reliable): {above_threshold} ({above_threshold/total_samples*100:.2f}%)")
    print(f"    - 低於閾值 (Unreliable): {below_threshold} ({below_threshold/total_samples*100:.2f}%)")

    # 繪圖
    plt.figure(figsize=(12, 7))

    # 繪製直方圖與 KDE
    sns.histplot(max_probs_np, bins=50, kde=True, color='#3498db',
                 edgecolor='black', alpha=0.7, linewidth=0.5, stat='count')

    # 繪製閾值線
    plt.axvline(x=conf_threshold, color='#e74c3c', linestyle='--', linewidth=2.5,
                label=f'Threshold = {conf_threshold:.4f}', zorder=10)

    # 繪製平均值線
    plt.axvline(x=mean_conf, color='#2ecc71', linestyle='-.', linewidth=2,
                label=f'Mean = {mean_conf:.4f}', zorder=10, alpha=0.8)

    # 繪製中位數線
    plt.axvline(x=median_conf, color='#9b59b6', linestyle=':', linewidth=2,
                label=f'Median = {median_conf:.4f}', zorder=10, alpha=0.8)

    # 添加統計文本框
    textstr = (f'Statistic Abstract:\n'
              f'Mean:     {mean_conf:.4f}\n'
              f'Median:   {median_conf:.4f}\n'
              f'Std Dev:  {std_conf:.4f}\n'
              f'Range:    [{min_conf:.4f}, {max_conf:.4f}]\n'
              f'\n'
              f'Reliable:   {above_threshold} ({above_threshold/total_samples*100:.1f}%)\n'
              f'Unreliable: {below_threshold} ({below_threshold/total_samples*100:.1f}%)')

    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    plt.title('Confidence Score Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Max Probability (Confidence)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xlim(0, 1.0)
    plt.legend(fontsize=12, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'confidence_histogram.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ 信心度直方圖已保存: {save_path}")
    print(f"{'='*60}\n")


def plot_class_distribution(true_labels, pseudo_labels, num_classes=37, output_dir='./output'):
    """
    繪製預測分佈 vs 真實分佈長條圖

    並排比較每個類別的真實樣本數量與預測樣本數量，
    並計算分佈相似度指標（余弦相似度、KL散度）。

    Args:
        true_labels: torch.Tensor or numpy array, 形狀為 [N]
        pseudo_labels: torch.Tensor or numpy array, 形狀為 [N]
        num_classes: int, 類別數量
        output_dir: str, 輸出目錄
    """
    print(f"\n{'='*60}")
    print("繪製預測 vs 真實類別分佈")
    print(f"{'='*60}\n")

    # 轉換為 numpy
    true_labels_np = true_labels.numpy() if isinstance(true_labels, torch.Tensor) else true_labels
    pseudo_labels_np = pseudo_labels.numpy() if isinstance(pseudo_labels, torch.Tensor) else pseudo_labels

    # 計算每個類別的數量
    true_counts = np.bincount(true_labels_np, minlength=num_classes)
    pred_counts = np.bincount(pseudo_labels_np, minlength=num_classes)

    # 計算差異
    diff = np.abs(true_counts - pred_counts)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    std_diff = np.std(diff)

    # 動態設定警告閾值（使用均值 + 1.5 倍標準差）
    warning_threshold = mean_diff + 1.5 * std_diff

    print(f"  分佈統計：")
    print(f"    - 總樣本數: {len(true_labels_np)}")
    print(f"    - 平均差異: {mean_diff:.2f}")
    print(f"    - 最大差異: {max_diff} (Class {np.argmax(diff)})")
    print(f"    - 標準差: {std_diff:.2f}")
    print(f"    - 警告閾值: {warning_threshold:.2f} (mean + 1.5*std)")

    # 找出差異大的類別
    warning_classes = np.where(diff > warning_threshold)[0]
    print(f"    - 差異過大的類別: {list(warning_classes)} (共 {len(warning_classes)} 個)")

    # 計算分佈相似度指標
    # 1. 余弦相似度
    cos_sim = np.dot(true_counts, pred_counts) / (np.linalg.norm(true_counts) * np.linalg.norm(pred_counts) + 1e-8)

    # 2. KL 散度（加入平滑以避免除零）
    true_dist = (true_counts + 1) / (true_counts.sum() + num_classes)
    pred_dist = (pred_counts + 1) / (pred_counts.sum() + num_classes)
    kl_div = np.sum(true_dist * np.log(true_dist / pred_dist + 1e-8))

    # 3. JS 散度（更對稱的版本）
    m = 0.5 * (true_dist + pred_dist)
    js_div = 0.5 * np.sum(true_dist * np.log(true_dist / m + 1e-8)) + 0.5 * np.sum(pred_dist * np.log(pred_dist / m + 1e-8))

    print(f"\n  分佈相似度指標：")
    print(f"    - 余弦相似度 (Cosine Similarity): {cos_sim:.4f} (越接近 1 越好)")
    print(f"    - KL 散度 (KL Divergence): {kl_div:.4f} (越接近 0 越好)")
    print(f"    - JS 散度 (JS Divergence): {js_div:.4f} (越接近 0 越好)")

    # 繪圖
    x = np.arange(num_classes)
    width = 0.38

    plt.figure(figsize=(20, 7))

    # 繪製並排長條圖
    bars1 = plt.bar(x - width/2, true_counts, width,
                    label='Ground Truth', color='#2ecc71', alpha=0.85,
                    edgecolor='black', linewidth=0.5)
    bars2 = plt.bar(x + width/2, pred_counts, width,
                    label='Pseudo Labels', color='#e74c3c', alpha=0.85,
                    edgecolor='black', linewidth=0.5)

    # 標記差異過大的類別
    for class_id in warning_classes:
        y_max = max(true_counts[class_id], pred_counts[class_id])
        plt.text(class_id, y_max + max(true_counts) * 0.02, '⚠️',
                ha='center', color='#e67e22', fontsize=14)
        # 添加差異數字
        plt.text(class_id, y_max + max(true_counts) * 0.08, f'Δ{int(diff[class_id])}',
                ha='center', fontsize=7, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))

    # 添加統計文本框
    textstr = (f'Distribution Similarity Metrics:\n'
              f'Cosine Similarity: {cos_sim:.4f}\n'
              f'KL Divergence:     {kl_div:.4f}\n'
              f'JS Divergence:     {js_div:.4f}\n'
              f'\n'
              f'Difference Statistics:\n'
              f'Mean Diff: {mean_diff:.2f}\n'
              f'Max Diff:  {max_diff} (Class {np.argmax(diff)})')

    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5)
    plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    plt.title('Predicted vs. True Class Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(x, fontsize=9)
    plt.legend(fontsize=14, loc='upper left', framealpha=0.95)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'class_distribution_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ 類別分佈比較圖已保存: {save_path}")
    print(f"{'='*60}\n")

    return cos_sim, kl_div, js_div


def main():
    """主函數：執行完整的診斷分析"""

    # 設置隨機種子以確保結果可重現
    random.seed(42)
    np.random.seed(42)

    # ==================== 路徑配置 (方便修改) ====================
    # CLIP 模型
    CLIP_BACKBONE = 'RN101'

    # Teacher Projector 權重路徑
    # PROJECTOR_PATH = '/mnt/backups/andycw/CLIP/output/m58/PureCLIP_Source_Model/rn101_projector_originCLIP_ep50_LR0.01_37classesFinal/source_projector.pt'
    PROJECTOR_PATH = '/mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/37_class_100epoch_noPropagation_retrain/target_ProjectorC_best.pt'
    # PROJECTOR_PATH = '/mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/37_class_15epoch_noPropagation_retrain/target_ProjectorC_ema_current.pt'
    # PROJECTOR_PATH = '/mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/37_class_100epoch_noPropagation_retrain/target_ProjectorC_ema_current.pt'

    # Source Centroids 路徑 (37 classes, [37, 512])
    SOURCE_CENTROIDS_PATH = '/mnt/backups/andycw/UDA-AI/class_centroids_37classes.pth'

    # Target Domain Test 資料集路徑
    TARGET_LIST_PATH = '/mnt/backups/andycw/M58/Real_all_nobg_37_list.txt'
    ROOT_PATH = '/mnt/backups/andycw/'

    # 輸出目錄
    OUTPUT_DIR = '/mnt/backups/andycw/UDA-AI/diagnostic_output'

    # 超參數 (黃金參數)
    LOGIT_SCALE = 9.9  # 寫死
    BATCH_SIZE = 64
    NUM_CLASSES = 37

    # ==================== 開始執行 ====================
    print("\n" + "="*60)
    print("Source-Free Domain Adaptation (SFDA) 診斷分析")
    print("="*60)
    print(f"輸出目錄: {OUTPUT_DIR}")
    print("="*60 + "\n")

    # 1. 載入 CLIP 模型 (使用官方 preprocess)
    print(f"\n{'='*60}")
    print("載入 CLIP 模型")
    print(f"{'='*60}")
    clip_model, clip_preprocess = load_clip_to_cpu(backbone=CLIP_BACKBONE)
    clip_model.float()
    clip_model.cuda()
    clip_model.eval()
    print(f"  ✓ CLIP 模型已載入: {CLIP_BACKBONE}")
    print(f"  ✓ 模式: eval() (已凍結)")
    print(f"  ✓ 使用 CLIP 官方 preprocess")
    print(f"{'='*60}\n")

    # 2. 載入 Teacher Projector
    projector = load_projector(PROJECTOR_PATH, device='cuda')

    # 3. 載入 Source Centroids
    source_centroids, classnames = load_source_centroids(SOURCE_CENTROIDS_PATH)

    # 4. 生成 ClassID 到 ClassName 的映射表
    save_class_mapping(classnames, num_classes=NUM_CLASSES, output_dir=OUTPUT_DIR)

    # 5. 創建 Target DataLoader (使用 CLIP 官方 preprocess)
    target_loader = create_target_dataloader(
        target_list_path=TARGET_LIST_PATH,
        clip_preprocess=clip_preprocess,
        batch_size=BATCH_SIZE,
        root_path=ROOT_PATH
    )

    # 6. 提取特徵並執行全域置中
    (target_features, centered_target, centered_source,
     true_labels, pseudo_labels, max_probs,
     reliable_mask, conf_threshold) = extract_features_with_centering(
        clip_model=clip_model,
        projector=projector,
        target_loader=target_loader,
        source_centroids=source_centroids,
        logit_scale=LOGIT_SCALE
    )

    print(f"\n{'='*60}")
    print("特徵提取與置中完成 - 數據摘要")
    print(f"{'='*60}")
    print(f"  - Target 特徵: {target_features.shape}")
    print(f"  - Centered Target: {centered_target.shape}")
    print(f"  - Centered Source: {centered_source.shape}")
    print(f"  - True Labels: {true_labels.shape}")
    print(f"  - Pseudo Labels: {pseudo_labels.shape}")
    print(f"  - Max Probs: {max_probs.shape}")
    print(f"  - Reliable Mask: {reliable_mask.shape}")
    print(f"  - Confidence Threshold: {conf_threshold:.4f}")
    print(f"{'='*60}\n")

    # 7. 計算 Per-class Accuracy 和 Confusion Matrix
    per_class_acc, cm = compute_confusion_matrix_and_accuracy(
        true_labels=true_labels,
        pseudo_labels=pseudo_labels,
        num_classes=NUM_CLASSES,
        output_dir=OUTPUT_DIR
    )

    # 8. t-SNE 視覺化
    visualize_tsne(
        centered_target=centered_target,
        centered_source=centered_source,
        true_labels=true_labels,
        pseudo_labels=pseudo_labels,
        reliable_mask=reliable_mask,
        num_classes=NUM_CLASSES,
        output_dir=OUTPUT_DIR
    )

    # 8b. t-SNE 視覺化 - 按預測正確性
    visualize_tsne_correctness(
        centered_target=centered_target,
        centered_source=centered_source,
        true_labels=true_labels,
        pseudo_labels=pseudo_labels,
        num_classes=NUM_CLASSES,
        output_dir=OUTPUT_DIR
    )

    # 9. 繪製信心度分佈直方圖
    plot_confidence_histogram(
        max_probs=max_probs,
        conf_threshold=conf_threshold,
        output_dir=OUTPUT_DIR
    )

    # 10. 繪製預測 vs 真實類別分佈
    cos_sim, kl_div, js_div = plot_class_distribution(
        true_labels=true_labels,
        pseudo_labels=pseudo_labels,
        num_classes=NUM_CLASSES,
        output_dir=OUTPUT_DIR
    )

    # ==================== 完成 ====================
    print("\n" + "="*60)
    print("✅ 診斷分析完成！")
    print("="*60)
    print(f"\n輸出檔案：")
    print(f"  - {OUTPUT_DIR}/class_id_to_name.csv (ClassID 到 ClassName 映射表)")
    print(f"  - {OUTPUT_DIR}/per_class_accuracy.csv (每類別準確率)")
    print(f"  - {OUTPUT_DIR}/confusion_matrix.png (混淆矩陣)")
    print(f"  - {OUTPUT_DIR}/tsne_ground_truth.png (t-SNE Ground Truth 分佈)")
    print(f"  - {OUTPUT_DIR}/tsne_reliability.png (t-SNE Reliability 分佈)")
    print(f"  - {OUTPUT_DIR}/tsne_correctness.png (t-SNE 預測正確性分佈)")
    print(f"  - {OUTPUT_DIR}/confidence_histogram.png (信心度分佈直方圖)")
    print(f"  - {OUTPUT_DIR}/class_distribution_comparison.png (預測 vs 真實類別分佈)")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
