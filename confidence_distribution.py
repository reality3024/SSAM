#!/usr/bin/env python3
"""
生成 Target 資料在 Source Class Centroid 上的信心度分布圖
- 載入現成的 Source Class Centroid
- 載入 CLIP Visual Encoder + Projector
- 提取 Target 特徵並計算信心度（不使用 logit scale）
- 繪製信心度 Distribution 圖（範圍 0~1）
"""

import os
import sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端

# 讓 UDA-AI 的模組可以被 import
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, '/mnt/nas_40901/andycw/UDA-AI')

from clip import clip


# ============================================================
# 資料集定義
# ============================================================

class M58(Dataset):
    """M58資料集類別，用於載入CAD零件圖片資料"""

    def __init__(self, root_dir, transform=None):
        """
        初始化M58資料集

        Args:
            root_dir: 資料集根目錄路徑
            transform: 圖片預處理轉換函數
        """
        self.root_dir = root_dir
        self.transform = transform

        # 載入類別名稱
        classname_file = os.path.join(os.path.dirname(root_dir), 'classname37.txt')
        with open(classname_file, 'r', encoding='utf-8') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 建立類別到索引的映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集所有圖片路徑和標籤
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """載入所有樣本的路徑和標籤"""
        loaded_classes = 0

        # 只處理classname37.txt中列出的類別
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[class_name]
                class_images = 0
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
                        class_images += 1

                if class_images > 0:
                    loaded_classes += 1
            else:
                print(f"  警告: 找不到類別資料夾: {class_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        取得指定索引的樣本

        Args:
            idx: 樣本索引

        Returns:
            tuple: (image, class_idx)
        """
        img_path, class_idx = self.samples[idx]

        # 載入圖片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"載入圖片失敗: {img_path}, 錯誤: {e}")
            # 如果圖片載入失敗，建立一個黑色圖片作為替代
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 應用轉換
        if self.transform:
            image = self.transform(image)

        return image, class_idx


# ============================================================
# 模型定義
# ============================================================

class MLPProjector(nn.Module):
    """MLP Projector: 與訓練時相同的結構"""
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


# ============================================================
# 主要功能
# ============================================================

def load_class_centroids(centroid_path, device):
    """
    載入已儲存的 Class Centroid

    Args:
        centroid_path: centroid 檔案路徑
        device: torch device

    Returns:
        centroids: [num_classes, feature_dim] tensor
        classnames: list of class names
        num_classes: 類別數量
    """
    print(f"\n{'='*60}")
    print(f"載入 Source Class Centroid")
    print(f"{'='*60}")

    if not osp.exists(centroid_path):
        raise FileNotFoundError(f"找不到 Centroid 檔案: {centroid_path}")

    saved_data = torch.load(centroid_path, map_location='cpu')
    centroids = saved_data['centroids'].to(device)
    classnames = saved_data['classnames']
    num_classes = saved_data['num_classes']

    print(f"✅ Centroid 載入成功")
    print(f"   類別數        : {num_classes}")
    print(f"   特徵維度      : {saved_data['feature_dim']}")
    print(f"   總樣本數      : {saved_data['class_counts'].sum().item()}")
    print(f"   有效類別數    : {(saved_data['class_counts'] > 0).sum().item()}/{num_classes}")
    print(f"{'='*60}\n")

    return centroids, classnames, num_classes


def load_models(projector_path, clip_dim, backbone='RN101', device='cuda'):
    """
    載入 CLIP Visual Encoder 和 Projector

    Args:
        projector_path: projector 權重路徑
        clip_dim: CLIP 特徵維度
        backbone: CLIP backbone 名稱
        device: torch device

    Returns:
        clip_model: CLIP visual encoder
        projector: MLP projector
        preprocess: CLIP preprocess transform
    """
    print(f"\n{'='*60}")
    print(f"載入模型")
    print(f"{'='*60}")

    # 載入 CLIP 模型
    clip_model, preprocess = clip.load(
        backbone, device="cpu",
        download_root=os.path.expanduser("~/.cache/clip")
    )
    clip_model = clip_model.float().to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print(f"✅ CLIP Visual Encoder 載入成功: {backbone}")

    # 載入 Projector
    if not osp.exists(projector_path):
        raise FileNotFoundError(f"找不到 Projector 權重: {projector_path}")

    projector = MLPProjector(in_dim=clip_dim, out_dim=clip_dim).to(device)
    projector.load_state_dict(torch.load(projector_path, map_location='cpu'), strict=True)
    projector.eval()
    for param in projector.parameters():
        param.requires_grad = False
    print(f"✅ Projector 載入成功: {projector_path}")
    print(f"{'='*60}\n")

    return clip_model.visual, projector, preprocess


def extract_target_features(clip_visual, projector, target_loader, clip_dim, device):
    """
    提取 Target 資料集的所有特徵

    Args:
        clip_visual: CLIP visual encoder
        projector: MLP projector
        target_loader: Target data loader
        clip_dim: 特徵維度
        device: torch device

    Returns:
        all_features: [N, clip_dim] tensor (normalized)
        all_labels: [N] tensor
    """
    print(f"\n{'='*60}")
    print(f"提取 Target 特徵")
    print(f"{'='*60}")

    dataset_size = len(target_loader.dataset)
    all_features = torch.zeros(dataset_size, clip_dim)
    all_labels = torch.zeros(dataset_size, dtype=torch.long)

    clip_visual.eval()
    projector.eval()

    idx = 0
    with torch.no_grad():
        for images, labels in tqdm(target_loader, desc="提取 Target 特徵"):
            images = images.to(device)
            batch_size = images.size(0)

            # 提取特徵
            clip_feats = clip_visual(images.type(clip_visual.conv1.weight.dtype))  # [B, D]
            projected_feats = projector(clip_feats)  # [B, D]
            projected_feats = F.normalize(projected_feats, dim=-1)  # L2 normalize

            all_features[idx:idx+batch_size] = projected_feats.cpu()
            all_labels[idx:idx+batch_size] = labels
            idx += batch_size

    print(f"✅ Target 特徵提取完成: {all_features.shape}")
    print(f"{'='*60}\n")

    return all_features, all_labels


def compute_confidence_distribution(centroids, target_features, target_labels,
                                    logit_scale_1=1.0, logit_scale_2=11.0):
    """
    計算 Target 樣本對應其真實類別的信心度分布（使用全域置中 + softmax）
    比較兩種不同 logit scale 的分布

    Args:
        centroids: [num_classes, D] tensor (normalized)
        target_features: [N, D] tensor (normalized)
        target_labels: [N] tensor
        logit_scale_1: 第一種方法的 logit scale (預設 1.0，接近原始分布)
        logit_scale_2: 第二種方法的 logit scale (預設 11.0，訓練時使用)

    Returns:
        confidences_1: [N] numpy array (logit_scale_1 的 softmax 機率)
        confidences_2: [N] numpy array (logit_scale_2 的 softmax 機率)
        predicted_labels_1: [N] numpy array (方法1的預測標籤)
        predicted_labels_2: [N] numpy array (方法2的預測標籤)
    """
    print(f"\n{'='*60}")
    print(f"計算信心度分布（使用全域置中 + Softmax）")
    print(f"{'='*60}")

    # 1. 全域置中
    target_mean = target_features.mean(dim=0, keepdim=True)
    source_mean = centroids.mean(dim=0, keepdim=True).cpu()

    centered_target = F.normalize(target_features - target_mean, dim=-1)
    centered_source = F.normalize(centroids.cpu() - source_mean, dim=-1)

    print(f"  Target 特徵平均值: {target_mean.norm():.4f}")
    print(f"  Source 特徵平均值: {source_mean.norm():.4f}")

    # 2. 計算相似度矩陣
    similarities = centered_target @ centered_source.T  # [N, C], 範圍 [-1, 1]

    # 3. 方法1：logit_scale_1 + softmax
    print(f"\n  === 方法1 (logit_scale={logit_scale_1}) ===")
    logits_1 = logit_scale_1 * similarities
    probs_1 = F.softmax(logits_1, dim=1)
    confidences_1, predicted_labels_1 = torch.max(probs_1, dim=1)

    correct_1 = (predicted_labels_1 == target_labels).sum().item()
    accuracy_1 = correct_1 / len(target_labels) * 100

    print(f"  預測準確率  : {accuracy_1:.2f}% ({correct_1}/{len(target_labels)})")
    print(f"  信心度範圍  : [{confidences_1.min():.4f}, {confidences_1.max():.4f}]")
    print(f"  平均信心度  : {confidences_1.mean():.4f}")
    print(f"  中位信心度  : {confidences_1.median():.4f}")

    # 4. 方法2：logit_scale_2 + softmax
    print(f"\n  === 方法2 (logit_scale={logit_scale_2}) ===")
    logits_2 = logit_scale_2 * similarities
    probs_2 = F.softmax(logits_2, dim=1)
    confidences_2, predicted_labels_2 = torch.max(probs_2, dim=1)

    correct_2 = (predicted_labels_2 == target_labels).sum().item()
    accuracy_2 = correct_2 / len(target_labels) * 100

    print(f"  預測準確率  : {accuracy_2:.2f}% ({correct_2}/{len(target_labels)})")
    print(f"  信心度範圍  : [{confidences_2.min():.4f}, {confidences_2.max():.4f}]")
    print(f"  平均信心度  : {confidences_2.mean():.4f}")
    print(f"  中位信心度  : {confidences_2.median():.4f}")
    print(f"{'='*60}\n")

    return (confidences_1.numpy(), confidences_2.numpy(),
            predicted_labels_1.numpy(), predicted_labels_2.numpy())


def plot_confidence_distribution(confidences_1, confidences_2, predicted_labels_1, predicted_labels_2,
                                   target_labels, logit_scale_1=1.0, logit_scale_2=11.0,
                                   save_path='confidence_distribution.png'):
    """
    繪製信心度分布圖（比較兩種 logit scale 的分布）

    Args:
        confidences_1: [N] numpy array (方法1的信心度)
        confidences_2: [N] numpy array (方法2的信心度)
        predicted_labels_1: [N] numpy array (方法1的預測標籤)
        predicted_labels_2: [N] numpy array (方法2的預測標籤)
        target_labels: [N] numpy array (真實標籤)
        logit_scale_1: 方法1的 logit scale
        logit_scale_2: 方法2的 logit scale
        save_path: 圖片儲存路徑
    """
    print(f"\n{'='*60}")
    print(f"繪製信心度分布圖")
    print(f"{'='*60}")

    target_labels = target_labels.numpy() if torch.is_tensor(target_labels) else target_labels

    # 分離正確和錯誤的預測
    correct_mask_1 = predicted_labels_1 == target_labels
    correct_mask_2 = predicted_labels_2 == target_labels

    conf_1_correct = confidences_1[correct_mask_1]
    conf_1_incorrect = confidences_1[~correct_mask_1]
    conf_2_correct = confidences_2[correct_mask_2]
    conf_2_incorrect = confidences_2[~correct_mask_2]

    # 設定繪圖參數
    fig = plt.figure(figsize=(18, 10))

    # ========== 第一行：整體分布比較 ==========
    # 左圖：方法1分布
    plt.subplot(2, 3, 1)
    plt.hist(confidences_1, bins=50, range=(0, 1), alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Confidence (Softmax Probability)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'Distribution with logit_scale={logit_scale_1}', fontsize=12, fontweight='bold')
    plt.axvline(confidences_1.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={confidences_1.mean():.3f}')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 中圖：方法2分布
    plt.subplot(2, 3, 2)
    plt.hist(confidences_2, bins=50, range=(0, 1), alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Confidence (Softmax Probability)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'Distribution with logit_scale={logit_scale_2}', fontsize=12, fontweight='bold')
    plt.axvline(confidences_2.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={confidences_2.mean():.3f}')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 右圖：直接疊加比較
    plt.subplot(2, 3, 3)
    plt.hist(confidences_1, bins=50, range=(0, 1), alpha=0.5, color='blue',
             label=f'logit_scale={logit_scale_1} (μ={confidences_1.mean():.3f})', edgecolor='black')
    plt.hist(confidences_2, bins=50, range=(0, 1), alpha=0.5, color='orange',
             label=f'logit_scale={logit_scale_2} (μ={confidences_2.mean():.3f})', edgecolor='black')
    plt.xlabel('Confidence (Softmax Probability)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Comparison', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # ========== 第二行：正確 vs 錯誤預測 ==========
    # 左圖：方法1的正確/錯誤
    plt.subplot(2, 3, 4)
    plt.hist(conf_1_correct, bins=50, range=(0, 1), alpha=0.6, color='green',
             label=f'Correct ({len(conf_1_correct)})', edgecolor='black')
    plt.hist(conf_1_incorrect, bins=50, range=(0, 1), alpha=0.6, color='red',
             label=f'Incorrect ({len(conf_1_incorrect)})', edgecolor='black')
    plt.xlabel('Confidence', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'logit_scale={logit_scale_1}: Correct vs Incorrect', fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 中圖：方法2的正確/錯誤
    plt.subplot(2, 3, 5)
    plt.hist(conf_2_correct, bins=50, range=(0, 1), alpha=0.6, color='green',
             label=f'Correct ({len(conf_2_correct)})', edgecolor='black')
    plt.hist(conf_2_incorrect, bins=50, range=(0, 1), alpha=0.6, color='red',
             label=f'Incorrect ({len(conf_2_incorrect)})', edgecolor='black')
    plt.xlabel('Confidence', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'logit_scale={logit_scale_2}: Correct vs Incorrect', fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 右圖：CDF 比較
    plt.subplot(2, 3, 6)
    sorted_1 = np.sort(confidences_1)
    sorted_2 = np.sort(confidences_2)
    cdf_1 = np.arange(1, len(sorted_1) + 1) / len(sorted_1)
    cdf_2 = np.arange(1, len(sorted_2) + 1) / len(sorted_2)

    plt.plot(sorted_1, cdf_1, color='blue', linewidth=2, label=f'logit_scale={logit_scale_1}')
    plt.plot(sorted_2, cdf_2, color='orange', linewidth=2, label=f'logit_scale={logit_scale_2}')
    plt.xlabel('Confidence Threshold', fontsize=11)
    plt.ylabel('Cumulative Probability', fontsize=11)
    plt.title('Cumulative Distribution (CDF)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 信心度分布圖已儲存至: {save_path}")

    # ========== 顯示統計資訊 ==========
    print(f"\n{'='*60}")
    print(f"統計資訊比較")
    print(f"{'='*60}")

    accuracy_1 = (predicted_labels_1 == target_labels).sum() / len(target_labels) * 100
    accuracy_2 = (predicted_labels_2 == target_labels).sum() / len(target_labels) * 100

    print(f"\n方法1 (logit_scale={logit_scale_1}):")
    print(f"  準確率    : {accuracy_1:.2f}%")
    print(f"  正確預測  : 平均={conf_1_correct.mean():.4f}, 中位={np.median(conf_1_correct):.4f}")
    print(f"  錯誤預測  : 平均={conf_1_incorrect.mean():.4f}, 中位={np.median(conf_1_incorrect):.4f}")

    print(f"\n方法2 (logit_scale={logit_scale_2}):")
    print(f"  準確率    : {accuracy_2:.2f}%")
    print(f"  正確預測  : 平均={conf_2_correct.mean():.4f}, 中位={np.median(conf_2_correct):.4f}")
    print(f"  錯誤預測  : 平均={conf_2_incorrect.mean():.4f}, 中位={np.median(conf_2_incorrect):.4f}")

    # ========== Threshold 分析 ==========
    print(f"\n{'='*60}")
    print(f"不同 Threshold 下的統計比較")
    print(f"{'='*60}")

    thresholds = np.arange(0.5, 1.0, 0.05)

    print(f"\n方法1 (logit_scale={logit_scale_1}):")
    print(f"  {'Threshold':<12} {'Retention (%)':<15} {'Accuracy (%)':<15}")
    print(f"  {'-'*42}")
    for thresh in thresholds:
        mask = confidences_1 >= thresh
        retention = mask.sum() / len(confidences_1) * 100
        acc = (predicted_labels_1[mask] == target_labels[mask]).sum() / mask.sum() * 100 if mask.sum() > 0 else 0
        print(f"  {thresh:<12.2f} {retention:<15.2f} {acc:<15.2f}")

    print(f"\n方法2 (logit_scale={logit_scale_2}):")
    print(f"  {'Threshold':<12} {'Retention (%)':<15} {'Accuracy (%)':<15}")
    print(f"  {'-'*42}")
    for thresh in thresholds:
        mask = confidences_2 >= thresh
        retention = mask.sum() / len(confidences_2) * 100
        acc = (predicted_labels_2[mask] == target_labels[mask]).sum() / mask.sum() * 100 if mask.sum() > 0 else 0
        print(f"  {thresh:<12.2f} {retention:<15.2f} {acc:<15.2f}")

    print(f"{'='*60}\n")


# ============================================================
# 主程式
# ============================================================

def main():
    # ---- 參數設定 ----
    centroid_path = '/mnt/backups/andycw/UDA-AI/class_centroids37_Jeannie.pth'
    projector_path = '/mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/37_class_50epoch_noPropagation_retrain_Jeannie/target_ProjectorC_ema_current.pt'
    target_data_root = '/mnt/backups/andycw/M58/Real_all_nobg'
    save_path = '/mnt/backups/andycw/UDA-AI/target_confidence_distribution.png'

    backbone = 'RN101'
    clip_dim = 512  # RN101 的特徵維度
    batch_size = 64
    logit_scale_1 = 1.0   # 接近原始分布
    logit_scale_2 = 10.0  # 訓練時使用的 logit scale (Phase 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"信心度分布分析（比較不同 Logit Scale 的效果）")
    print(f"{'='*60}")
    print(f"  Backbone        : {backbone}")
    print(f"  Feature Dim     : {clip_dim}")
    print(f"  Logit Scale 1   : {logit_scale_1} (接近原始分布)")
    print(f"  Logit Scale 2   : {logit_scale_2} (訓練時使用)")
    print(f"  Device          : {device}")
    print(f"  Centroid Path   : {centroid_path}")
    print(f"  Projector Path  : {projector_path}")
    print(f"  Target Data     : {target_data_root}")
    print(f"  Save Path       : {save_path}")
    print(f"{'='*60}\n")

    # Step 1: 載入 Source Class Centroid
    centroids, classnames, num_classes = load_class_centroids(centroid_path, device)

    # Step 2: 載入 CLIP Visual Encoder 和 Projector
    clip_visual, projector, preprocess = load_models(projector_path, clip_dim, backbone, device)

    # Step 3: 載入 Target 資料
    print(f"\n{'='*60}")
    print(f"載入 Target 資料")
    print(f"{'='*60}")
    target_dataset = M58(root_dir=target_data_root, transform=preprocess)
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"✅ Target 資料集載入完成: {len(target_dataset)} 張, {num_classes} 個類別")
    print(f"{'='*60}\n")

    # Step 4: 提取 Target 特徵
    target_features, target_labels = extract_target_features(
        clip_visual, projector, target_loader, clip_dim, device
    )

    # Step 5: 計算信心度分布（比較兩種 logit scale）
    confidences_1, confidences_2, predicted_labels_1, predicted_labels_2 = compute_confidence_distribution(
        centroids, target_features, target_labels,
        logit_scale_1=logit_scale_1, logit_scale_2=logit_scale_2
    )

    # Step 6: 繪製信心度分布圖
    plot_confidence_distribution(
        confidences_1, confidences_2, predicted_labels_1, predicted_labels_2, target_labels,
        logit_scale_1=logit_scale_1, logit_scale_2=logit_scale_2, save_path=save_path
    )

    print(f"\n{'='*60}")
    print(f"✅ 完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
