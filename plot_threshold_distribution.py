#!/usr/bin/env python3
"""
繪製 Logit Threshold 對 Pseudo Label 品質的影響
固定 logit_scale = 8.1，測試不同 logit threshold 對 Purity 和選取數量的影響

Note: 這裡使用 logit 值（而非 probability）作為信心分數，
      所以 threshold 範圍應該在 1-20 之間（而不是 0-1）
"""

import argparse
import os
import sys
import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 讓 UDA-AI 的 loss / data_list 可以被 import
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import loss
from data_list import ImageList_idx
from clip import clip


# ============================================================
# M58 Dataset 類別定義
# ============================================================

class M58(Dataset):
    """M58資料集類別，用於載入CAD零件圖片資料"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 載入類別名稱
        classname_file = os.path.join(os.path.dirname(root_dir), 'classname37.txt')
        if not os.path.exists(classname_file):
            raise FileNotFoundError(f"找不到類別名稱檔案: {classname_file}")
        
        with open(classname_file, 'r', encoding='utf-8') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # 載入圖片路徑與標籤
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================
# MLP Projector
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
# Step 1: Source Centroid 提取
# ============================================================

def extract_class_centroids_from_source(args, device):
    """從 Source 資料集提取 Class Centroid"""
    
    print(f"\n{'='*60}")
    print(f"Step 1: 提取 Source Class Centroid")
    print(f"{'='*60}")
    
    # 檢查是否已存在 Centroid 檔案
    if os.path.exists(args.centroid_save_path):
        print(f"載入已存在的 Centroid: {args.centroid_save_path}")
        checkpoint = torch.load(args.centroid_save_path, map_location='cpu')
        source_visual_prototypes = checkpoint['centroids'].to(device)
        classnames = checkpoint['classnames']
        print(f"✅ Centroid 形狀: {source_visual_prototypes.shape}")
        return source_visual_prototypes, classnames
    
    # 如果不存在，則重新提取
    print(f"未找到現有 Centroid，開始從 Source 資料提取...")
    
    # 載入 CLIP + Projector
    clip_model_cpu, preprocess = clip.load(
        args.net, device="cpu",
        download_root=os.path.expanduser("~/.cache/clip")
    )
    clip_model_cpu.float()
    netF = clip_model_cpu.visual.to(device)
    netF.eval()
    
    netP = MLPProjector(in_dim=args.clip_dim, out_dim=args.clip_dim).to(device)
    netP.load_state_dict(torch.load(args.projector_path, map_location='cpu'))
    netP.eval()
    
    print(f"✅ CLIP + Projector 載入完成")
    
    # 載入 Source 資料集
    source_dataset = M58(args.src_data_root, transform=preprocess)
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False
    )
    
    print(f"✅ Source 資料集載入: {len(source_dataset)} 張圖片")
    print(f"   類別數: {len(source_dataset.classes)}")
    
    # 提取所有特徵
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(source_loader, desc="提取 Source 特徵"):
            images = images.to(device)
            feat_512 = netF(images.type(netF.conv1.weight.dtype))
            feat_projected = netP(feat_512)
            feat_norm = F.normalize(feat_projected, dim=-1)
            
            all_features.append(feat_norm.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"✅ 特徵提取完成: {all_features.shape}")
    
    # 計算每個類別的 Centroid
    num_classes = len(source_dataset.classes)
    class_centroids = torch.zeros(num_classes, args.clip_dim)
    
    for c in range(num_classes):
        class_mask = (all_labels == c)
        if class_mask.sum() > 0:
            class_centroids[c] = all_features[class_mask].mean(dim=0)
            class_centroids[c] = F.normalize(class_centroids[c], dim=-1)
    
    print(f"✅ Class Centroid 計算完成: {class_centroids.shape}")
    
    # 儲存 Centroid
    torch.save({
        'class_centroids': class_centroids,
        'classnames': source_dataset.classes,
        'num_classes': num_classes,
    }, args.centroid_save_path)
    print(f"✅ Centroid 已儲存至: {args.centroid_save_path}")
    
    # 釋放記憶體
    del clip_model_cpu, netF, netP
    torch.cuda.empty_cache()
    
    return class_centroids.to(device), source_dataset.classes


# ============================================================
# Step 2: Target 資料載入
# ============================================================

def image_test(resize_size=256, crop_size=224):
    """測試/Target 推論用 transform"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    """載入 Target 資料集"""
    dsets = {}
    dset_loaders = {}
    txt_tar = open(args.t_dset_path).readlines()
    
    root_path = f'data/{args.dset}/'
    dsets["target"] = ImageList_idx(txt_tar, transform=image_test(), root=root_path)
    
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False
    )
    return dset_loaders


def extract_target_features(netF, netP, target_loader, clip_dim, device):
    """提取 Target 資料集的所有特徵"""
    dataset_size = len(target_loader.dataset)
    all_features = torch.zeros(dataset_size, clip_dim)
    all_true_labels = torch.zeros(dataset_size, dtype=torch.long)
    
    netF.eval()
    netP.eval()
    
    with torch.no_grad():
        for data in tqdm(target_loader, desc="提取 Target 特徵"):
            inputs, labels, idx, _ = data
            inputs = inputs.to(device)
            
            feat_512 = netF(inputs.type(netF.conv1.weight.dtype))
            feat_projected = netP(feat_512)
            feat_norm = F.normalize(feat_projected, dim=-1)
            
            all_features[idx] = feat_norm.cpu()
            all_true_labels[idx] = labels.long()
    
    print(f"✅ Target 特徵提取完成: {all_features.shape}")
    return all_features, all_true_labels


# ============================================================
# 計算 Pseudo Label 品質
# ============================================================

def compute_pseudo_label_quality(source_visual_prototypes, all_features, all_true_labels,
                                 threshold=0.9, logit_scale=8.1):
    """
    計算給定 threshold 下的 Pseudo Label 品質
    
    Note: 使用 softmax 後的機率值（0-1 範圍）作為信心分數
    
    Returns:
        purity: Pseudo Label 純度 (%)
        retained_count: 選取的樣本數量
    """
    dataset_size = len(all_features)
    
    # 1. 全域置中
    target_mean = all_features.mean(dim=0, keepdim=True)
    source_mean = source_visual_prototypes.mean(dim=0, keepdim=True).cpu()
    
    centered_target = F.normalize(all_features - target_mean, dim=-1)  # cpu
    centered_source = F.normalize(source_visual_prototypes.cpu() - source_mean, dim=-1)  # cpu
    
    # 2. 計算 Logits
    logits = logit_scale * (centered_target @ centered_source.T)  # [N, C]
    
    # 【關鍵修改】：使用 Softmax 將 Logits 轉為機率 (Confidence Score)
    probs = F.softmax(logits, dim=1)  # [N, C]
    max_probs, pseudo_labels = torch.max(probs, dim=1)  # max_probs 範圍 [0, 1]
    
    # threshold 作用於機率值（0-1 範圍）
    confidence_mask = max_probs > threshold
    retained_count = confidence_mask.sum().item()
    
    # 3. 計算純度
    if retained_count > 0:
        correct_count = ((pseudo_labels == all_true_labels) & confidence_mask).sum().item()
        purity = (correct_count / retained_count) * 100
    else:
        purity = 0.0
    
    return purity, retained_count


def compute_confidence_distribution(source_visual_prototypes, all_features, all_true_labels, logit_scale=8.1):
    """
    計算所有樣本的信心度分布，並區分正確與錯誤預測
    
    Args:
        source_visual_prototypes: [C, D] Source prototypes
        all_features: [N, D] Target 特徵
        all_true_labels: [N] Target 真實標籤
        logit_scale: Logit scale 值
    
    Returns:
        confidences_correct: 正確預測樣本的信心度列表
        confidences_failed: 錯誤預測樣本的信心度列表
    """
    # 1. 全域置中
    target_mean = all_features.mean(dim=0, keepdim=True)
    source_mean = source_visual_prototypes.mean(dim=0, keepdim=True).cpu()
    
    centered_target = F.normalize(all_features - target_mean, dim=-1)
    centered_source = F.normalize(source_visual_prototypes.cpu() - source_mean, dim=-1)
    
    # 2. 計算 Logits 並轉為機率
    logits = logit_scale * (centered_target @ centered_source.T)
    probs = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)
    
    # 3. 區分正確與錯誤預測
    correct_mask = (pseudo_labels == all_true_labels)
    
    confidences_correct = max_probs[correct_mask].numpy()
    confidences_failed = max_probs[~correct_mask].numpy()
    
    return confidences_correct, confidences_failed


# ============================================================
# 繪製分布圖
# ============================================================

def plot_threshold_distribution(thresholds, purities, retained_counts, save_path='threshold_distribution.png'):
    """
    繪製 Threshold vs Purity & Retained Count 的分布圖（柱狀圖 + 折線圖）
    
    Args:
        thresholds: threshold 列表
        purities: 對應的 purity 列表 (%)
        retained_counts: 對應的選取樣本數列表
        save_path: 圖片儲存路徑
    """
    # 設定中文字體（如果需要）
    # rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows
    # rcParams['axes.unicode_minus'] = False
    
    # 創建圖表
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # 計算柱狀圖寬度
    bar_width = (thresholds[1] - thresholds[0]) * 0.8 if len(thresholds) > 1 else 0.08
    
    # 設定第一個 y 軸（Purity - 折線圖）
    color1 = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold (Probability)', fontsize=12)
    ax1.set_ylabel('Pseudo-label Purity (%)', color=color1, fontsize=12)
    line1 = ax1.plot(thresholds, purities, color=color1, marker='o', linewidth=2.5, 
                     markersize=5, label='Purity (Line)', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, zorder=0)
    
    # 設定第二個 y 軸（Retained Count - 柱狀圖 + 折線圖）
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    color2_light = 'lightsalmon'
    ax2.set_ylabel('Number of Selected Samples', color=color2, fontsize=12)
    
    # 柱狀圖
    bars = ax2.bar(thresholds, retained_counts, width=bar_width, color=color2_light, 
                   alpha=0.6, label='Selected Samples (Bar)', zorder=1)
    
    # 折線圖
    line2 = ax2.plot(thresholds, retained_counts, color=color2, marker='s', linewidth=2,
                     markersize=4, linestyle='--', label='Selected Samples (Line)', zorder=2)
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 設定標題
    plt.title('Impact of Confidence Threshold on Pseudo-label Quality\n(Logit Scale = 8.1)', 
              fontsize=14, fontweight='bold')
    
    # 合併圖例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # 添加柱狀圖到圖例
    handles = [line1[0], bars, line2[0]]
    labels = ['Purity (Line)', 'Selected Samples (Bar)', 'Selected Samples (Line)']
    ax1.legend(handles, labels, loc='upper right', fontsize=10)
    
    # 調整佈局
    fig.tight_layout()
    
    # 儲存圖片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 圖表已儲存至: {save_path}")
    
    # 顯示圖片
    plt.show()


def plot_confidence_histogram(confidences_correct, confidences_failed, 
                             save_path='confidence_histogram.png'):
    """
    繪製信心度直方圖，診斷模型是否過度自信
    
    Args:
        confidences_correct: 正確預測樣本的信心度數組
        confidences_failed: 錯誤預測樣本的信心度數組
        save_path: 圖片儲存路徑
    """
    # 創建圖表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 設定 bins（使用更高解析度以觀察 0.9-1.0 之間的分佈）
    bins = np.linspace(0, 1.0, 51)  # 50 個 bins
    
    # 【關鍵修改點】：計算反向累計數值 (即大於該 threshold 的樣本總數)
    # 這樣柱子的平均高度就會直接等於你表格中的 "Selected Samples"
    def get_cumulative_counts(data, bins):
        counts = []
        for b in bins[:-1]: # 排除最後一個 1.0 的邊界
            counts.append(np.sum(data >= b))
        return counts
    
    cum_correct = get_cumulative_counts(confidences_correct, bins)
    cum_failed = get_cumulative_counts(confidences_failed, bins)

    # 繪製柱狀圖 (使用 plt.bar 代替 plt.hist，因為我們已經手動算好累計值了)
    width = 0.02 # 每個 bin 的寬度
    ax.bar(bins[:-1], cum_correct, width=width, alpha=0.7, color='green', 
           label='Correct (Cumulative)', align='edge', edgecolor='black', linewidth=0.5)
    ax.bar(bins[:-1], cum_failed, width=width, alpha=0.7, color='red', 
           bottom=cum_correct, # 堆疊上去，這樣總高度就是總選取樣本數
           label='Failed (Cumulative)', align='edge', edgecolor='black', linewidth=0.5)
    
    # 設定標籤和標題
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Total Selected Samples (Cumulative)', fontsize=12)
    ax.set_title('Cumulative Pseudo-label Selection Analysis\n(Total Samples = Correct + Failed at each Threshold)', 
                 fontsize=14, fontweight='bold')
    
    # 設定 Y 軸上限為總樣本數並留白
    total_samples = len(confidences_correct) + len(confidences_failed)
    ax.set_ylim(0, total_samples * 1.2) 
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    # 添加統計信息文字框
    total_samples = len(confidences_correct) + len(confidences_failed)
    accuracy = len(confidences_correct) / total_samples * 100 if total_samples > 0 else 0
    
    if len(confidences_correct) > 0:
        mean_conf_correct = confidences_correct.mean()
        std_conf_correct = confidences_correct.std()
    else:
        mean_conf_correct = 0
        std_conf_correct = 0
    
    if len(confidences_failed) > 0:
        mean_conf_failed = confidences_failed.mean()
        std_conf_failed = confidences_failed.std()
    else:
        mean_conf_failed = 0
        std_conf_failed = 0
    
    # stats_text = (
    #     f'Overall Accuracy: {accuracy:.2f}%\n'
    #     f'Correct: μ={mean_conf_correct:.3f}, σ={std_conf_correct:.3f}\n'
    #     f'Failed: μ={mean_conf_failed:.3f}, σ={std_conf_failed:.3f}'
    # )
    
    # ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
    #         fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 調整佈局
    fig.tight_layout()
    
    # 儲存圖片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 信心度直方圖已儲存至: {save_path}")
    
    # 顯示圖片
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='繪製 Logit Threshold 分布圖 (固定 logit_scale=8.1)'
    )
    # ---- 通用設定 ----
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--net', type=str, default='RN101',
                        choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--worker', type=int, default=0)
    
    # ---- Source (Centroid 提取) ----
    parser.add_argument('--src_data_root', type=str,
                        default='/mnt/backups/andycw/M58/CAD_ratioFilter_StableDiffusion_random')
    parser.add_argument('--projector_path', type=str,
                        default='/mnt/backups/andycw/CLIP/output/m58/PureCLIP_Source_Model/rn101_projector_originCLIP_ep50_LR0.01_randomAug_37classes/source_projector.pt')
    parser.add_argument('--centroid_save_path', type=str,
                        default='./class_centroids_37.pth')
    
    # ---- Target ----
    parser.add_argument('--dset', type=str, default='M58')
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--t', type=int, default=1)
    
    # ---- 測試參數 ----
    parser.add_argument('--logit_scale', type=float, default=8.1,
                        help='固定 Logit Scale 值')
    parser.add_argument('--threshold_min', type=float, default=0.0,
                        help='Confidence Threshold 最小值（0-1 範圍）')
    parser.add_argument('--threshold_max', type=float, default=1.0,
                        help='Confidence Threshold 最大值（0-1 範圍）')
    parser.add_argument('--threshold_step', type=float, default=0.01,
                        help='Confidence Threshold 步長')
    
    # ---- 輸出 ----
    parser.add_argument('--save_path', type=str, default='threshold_distribution.png',
                        help='Threshold 分布圖儲存路徑')
    parser.add_argument('--confidence_hist_path', type=str, default='confidence_histogram.png',
                        help='信心度直方圖儲存路徑')
    
    args = parser.parse_args()
    
    # ---- 衍生設定 ----
    args.clip_dim = 1024 if args.net == 'RN50' else 512
    
    if args.dset == 'M58':
        names = ['CAD_ratioFilter_StableDiffusion_random', 'Real_all_nobg']
        args.class_num = 37
    
    folder = 'data/'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
    
    # ---- 環境初始化 ----
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Logit Threshold 分布圖繪製")
    print(f"{'='*60}")
    print(f"  Backbone      : {args.net}  (dim={args.clip_dim})")
    print(f"  Dataset       : {args.dset}  (classes={args.class_num})")
    print(f"  Target        : {names[args.t]}")
    print(f"  Logit Scale   : {args.logit_scale} (固定)")
    print(f"  Confidence Threshold 範圍: {args.threshold_min} ~ {args.threshold_max} (步長 {args.threshold_step})")
    print(f"  Device        : {device}")
    print(f"{'='*60}\n")
    
    # ============================================================
    # Step 1: 提取 Source Class Centroid
    # ============================================================
    source_visual_prototypes, classnames = extract_class_centroids_from_source(args, device)
    
    # ============================================================
    # Step 2: 載入 Target 資料與模型
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 2: 載入 Target 資料與模型")
    print(f"{'='*60}")
    
    # 載入 CLIP Visual Encoder
    clip_model_cpu, preprocess = clip.load(
        args.net, device="cpu",
        download_root=os.path.expanduser("~/.cache/clip")
    )
    clip_model_cpu.float()
    netF = clip_model_cpu.visual.to(device)
    netF.eval()
    for param in netF.parameters():
        param.requires_grad = False
    print(f"✅ CLIP Visual Encoder 載入並凍結")
    
    # 載入 Projector
    netP = MLPProjector(in_dim=args.clip_dim, out_dim=args.clip_dim).to(device)
    netP.load_state_dict(torch.load(args.projector_path, map_location='cpu'))
    netP.eval()
    print(f"✅ Projector 載入完成")
    
    # 載入 Target 資料
    dset_loaders = data_load(args)
    print(f"✅ Target 資料集載入完成: {len(dset_loaders['target'].dataset)} 張\n")
    
    # ============================================================
    # Step 3: 提取 Target 特徵
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 3: 提取 Target 特徵")
    print(f"{'='*60}")
    all_features, all_true_labels = extract_target_features(
        netF, netP, dset_loaders["target"], args.clip_dim, device
    )
    
    # ============================================================
    # Step 4: 測試不同 Confidence Threshold
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 4: 測試不同 Confidence Threshold (固定 logit_scale={args.logit_scale})")
    print(f"{'='*60}\n")
    
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    purities = []
    retained_counts = []
    
    print(f"{'Confidence Threshold':<20} {'Purity (%)':<15} {'Selected Samples':<20}")
    print(f"{'-'*60}")
    
    for threshold in thresholds:
        purity, retained_count = compute_pseudo_label_quality(
            source_visual_prototypes,
            all_features,
            all_true_labels,
            threshold=threshold,
            logit_scale=args.logit_scale
        )
        
        purities.append(purity)
        retained_counts.append(retained_count)
        
        print(f"{threshold:<20.3f} {purity:<15.2f} {retained_count:<20}")
    
    print(f"{'-'*60}\n")
    
    # ============================================================
    # Step 5: 繪製分布圖
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 5: 繪製分布圖")
    print(f"{'='*60}")
    
    plot_threshold_distribution(thresholds, purities, retained_counts, save_path=args.save_path)
    
    # ============================================================
    # Step 6: 繪製信心度直方圖
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 6: 繪製信心度直方圖 (診斷模型自信度)")
    print(f"{'='*60}")
    
    confidences_correct, confidences_failed = compute_confidence_distribution(
        source_visual_prototypes,
        all_features,
        all_true_labels,
        logit_scale=args.logit_scale
    )
    
    print(f"\n信心度分布統計:")
    print(f"  正確預測樣本數: {len(confidences_correct)}")
    print(f"  錯誤預測樣本數: {len(confidences_failed)}")
    print(f"  總體準確率: {len(confidences_correct)/(len(confidences_correct)+len(confidences_failed))*100:.2f}%")
    
    if len(confidences_correct) > 0:
        print(f"\n  正確預測信心度: 平均={confidences_correct.mean():.4f}, 標準差={confidences_correct.std():.4f}")
        print(f"                    最小={confidences_correct.min():.4f}, 最大={confidences_correct.max():.4f}")
    
    if len(confidences_failed) > 0:
        print(f"  錯誤預測信心度: 平均={confidences_failed.mean():.4f}, 標準差={confidences_failed.std():.4f}")
        print(f"                    最小={confidences_failed.min():.4f}, 最大={confidences_failed.max():.4f}")
    
    plot_confidence_histogram(confidences_correct, confidences_failed, 
                             save_path=args.confidence_hist_path)
    
    print(f"\n{'='*60}")
    print(f"✅ 所有步驟完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
