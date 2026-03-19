#!/usr/bin/env python3
"""
整合版: Text Prompt 生成 + Pseudo Label 品質檢查
Step 1: 使用 CLIP Text Encoder 從類別名稱生成 Text Features
Step 2: 以 Text Features 對 Target 資料產生 Pseudo Label，並分析純度
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
from sklearn.metrics import confusion_matrix
from PIL import Image

# 讓 UDA-AI 的 loss / data_list 可以被 import
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import loss
from data_list import ImageList_idx
from clip import clip


# ============================================================
# 共用模型定義
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
# Step 1: Text Features 生成 (使用 CLIP Text Encoder)
# ============================================================

def generate_text_features(classnames, clip_model, device):
    """
    使用 CLIP Text Encoder 生成 text features
    Template: "A photo of a [CLASS_NAME]"
    
    Args:
        classnames: 類別名稱列表
        clip_model: CLIP 模型 (包含 text encoder)
        device: 運算設備
    
    Returns:
        text_features: [C, D] 正規化的 text features
    """
    print(f"\n{'='*60}")
    print(f"Step 1: 生成 Text Features (使用 CLIP Text Encoder)")
    print(f"{'='*60}")
    print(f"  Template: 'A photo of a [CLASS_NAME]'")
    print(f"  類別數: {len(classnames)}")
    
    # 生成 prompts
    prompts = [f"A photo of a {classname}" for classname in classnames]
    
    # Tokenize
    prompts_tokens = clip.tokenize(prompts).to(device)
    
    # 編碼為 text features
    with torch.no_grad():
        text_features = clip_model.encode_text(prompts_tokens)
        text_features = F.normalize(text_features, dim=-1)
    
    print(f"✅ Text Features 生成完成: {text_features.shape}")
    print(f"✅ 已正規化: {(text_features.norm(dim=-1) - 1.0).abs().max().item() < 0.01}")
    
    return text_features


# ============================================================
# Step 2: Target 資料載入與 Pseudo Label 分析
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
    txt_tar  = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    root_path = f'data/{args.dset}/'
    dsets["target"] = ImageList_idx(txt_tar,  transform=image_test(), root=root_path)
    dsets["test"]   = ImageList_idx(txt_test, transform=image_test(), root=root_path)

    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=args.batch_size * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False
    )
    return dset_loaders


def extract_target_features(netF, netP, target_loader, clip_dim, device):
    """提取 Target 資料集的所有特徵（使用 indices 確保對應正確）"""
    dataset_size = len(target_loader.dataset)
    all_features    = torch.zeros(dataset_size, clip_dim)
    all_true_labels = torch.zeros(dataset_size, dtype=torch.long)

    netF.eval()
    netP.eval()

    with torch.no_grad():
        for data in tqdm(target_loader, desc="提取 Target 特徵", leave=False):
            imgs, labels, indices, _ = data
            imgs = imgs.to(device)

            feat_512      = netF(imgs.type(netF.conv1.weight.dtype))
            feat_projected = netP(feat_512)
            feat_norm      = F.normalize(feat_projected, dim=-1)

            all_features[indices]    = feat_norm.cpu()
            all_true_labels[indices] = labels.cpu()

    print(f"✅ Target 特徵提取完成: {all_features.shape}")
    return all_features, all_true_labels


def update_pseudo_labels(source_visual_prototypes, all_features, all_true_labels,
                         threshold=0.9, momentum=0.99, logit_scale=15.0):
    """
    以 Global-Mean-Centered 相似度產生 Pseudo Label，並以 EMA 更新 Source Prototypes

    Returns:
        pseudo_labels          : [N] cuda tensor
        confidence_mask        : [N] cuda tensor (bool)
        source_visual_prototypes: [C, D] cuda tensor（已更新）
    """
    dataset_size = len(all_features)

    # 1. 全域置中
    target_mean = all_features.mean(dim=0, keepdim=True)
    source_mean = source_visual_prototypes.mean(dim=0, keepdim=True).cpu()

    centered_target = F.normalize(all_features - target_mean,              dim=-1)  # cpu
    centered_source = F.normalize(source_visual_prototypes.cpu() - source_mean, dim=-1)  # cpu

    # 2. 相似度 → Pseudo Labels
    logits        = logit_scale * (centered_target @ centered_source.T)           # [N, C]
    probs         = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)

    confidence_mask = max_probs > threshold
    retained_count  = confidence_mask.sum().item()
    correct_count   = ((pseudo_labels == all_true_labels) & confidence_mask).sum().item()

    print(f"  保留樣本數 : {retained_count}/{dataset_size} "
          f"({retained_count/dataset_size*100:.2f}%)")
    if retained_count > 0:
        print(f"  Pseudo Label 純度: {correct_count/retained_count*100:.2f}%")
    else:
        print("  警告: 無樣本通過信心門檻！")

    # 3. EMA 更新 Source Prototypes（使用 centered_source 的 GPU 版本）
    centered_source_gpu = centered_source.to(source_visual_prototypes.device)
    for c in range(source_visual_prototypes.shape[0]):
        class_mask = (pseudo_labels == c) & confidence_mask
        if class_mask.sum() > 0:
            target_class_feat = F.normalize(
                all_features[class_mask].mean(dim=0).to(source_visual_prototypes.device),
                dim=-1
            )
            centered_source_gpu[c] = (
                momentum * centered_source_gpu[c] + (1.0 - momentum) * target_class_feat
            )

    updated_prototypes = F.normalize(centered_source_gpu, dim=-1)
    return pseudo_labels.cuda(), confidence_mask.cuda(), updated_prototypes


def search_optimal_parameters_topk(source_visual_prototypes, all_features, all_true_labels,
                                  k_range=(1, 2), logit_scale_range=(5, 15),
                                  k_step=1, logit_scale_step=1.0,
                                  target_samples=(100, 200), target_purity=(75, 85),
                                  target_coverage=85):
    """
    使用 Class-wise Top-K 策略搜索最佳參數組合
    
    對於每個類別，選擇模型預測為該類別中「最有信心的前 K 張」樣本作為 Pseudo Label
    
    Args:
        source_visual_prototypes: [C, D] Source prototypes
        all_features: [N, D] Target 特徵
        all_true_labels: [N] Target 真實標籤
        k_range: (min, max) 每個類別選取的樣本數範圍
        logit_scale_range: (min, max) logit_scale 範圍
        k_step: K 步長
        logit_scale_step: logit_scale 步長
        target_samples: (min, max) 目標保留樣本數範圍
        target_purity: (min, max) 目標純度範圍 (%)
        target_coverage: 最小類別覆蓋率 (%)
    
    Returns:
        results: list of dict，包含符合條件的參數組合
    """
    print(f"\n{'='*80}")
    print(f"開始參數搜索 (Class-wise Top-K 策略)")
    print(f"{'='*80}")
    print(f"  K 範圍            : {k_range[0]} ~ {k_range[1]} (步長 {k_step})")
    print(f"  Logit Scale 範圍  : {logit_scale_range[0]} ~ {logit_scale_range[1]} (步長 {logit_scale_step})")
    print(f"  目標保留樣本數    : {target_samples[0]} ~ {target_samples[1]}")
    print(f"  目標純度          : {target_purity[0]}% ~ {target_purity[1]}%")
    print(f"  最小類別覆蓋率    : {target_coverage}%")
    print(f"{'='*80}\n")
    
    dataset_size = len(all_features)
    num_classes = source_visual_prototypes.shape[0]
    
    # 生成參數組合
    k_values = np.arange(k_range[0], k_range[1] + k_step, k_step, dtype=int)
    logit_scales = np.arange(logit_scale_range[0], logit_scale_range[1] + logit_scale_step, logit_scale_step)
    
    all_results = []
    sweet_spot_results = []
    
    print(f"正在測試 {len(k_values)} x {len(logit_scales)} = {len(k_values) * len(logit_scales)} 種參數組合...\n")
    print(f"{'K':<8} {'LogitScale':<12} {'保留樣本':<15} {'純度(%)':<12} {'覆蓋率(%)':<12} {'覆蓋類別':<12} {'狀態':<10}")
    print(f"{'-'*95}")
    
    for logit_scale in logit_scales:
        for k in k_values:
            # 1. 全域置中
            target_mean = all_features.mean(dim=0, keepdim=True)
            source_mean = source_visual_prototypes.mean(dim=0, keepdim=True).cpu()
            
            centered_target = F.normalize(all_features - target_mean, dim=-1)
            centered_source = F.normalize(source_visual_prototypes.cpu() - source_mean, dim=-1)
            
            # 2. 計算相似度和預測
            logits = logit_scale * (centered_target @ centered_source.T)
            probs = F.softmax(logits, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            
            # 3. Class-wise Top-K 選擇
            selected_indices = []
            class_sample_counts = {}
            
            for c in range(num_classes):
                # 找出所有預測為類別 c 的樣本
                class_mask = (pseudo_labels == c)
                class_indices = torch.where(class_mask)[0]
                
                if len(class_indices) > 0:
                    # 獲取這些樣本的信心度
                    class_probs = max_probs[class_indices]
                    
                    # 選擇信心度最高的前 K 個
                    k_to_select = min(k, len(class_indices))
                    top_k_local_indices = torch.topk(class_probs, k_to_select).indices
                    top_k_global_indices = class_indices[top_k_local_indices]
                    
                    selected_indices.extend(top_k_global_indices.tolist())
                    class_sample_counts[c] = k_to_select
            
            # 4. 計算指標
            if len(selected_indices) > 0:
                selected_indices = torch.tensor(selected_indices, dtype=torch.long)
                retained_count = len(selected_indices)
                
                # 計算純度
                correct_count = (pseudo_labels[selected_indices] == all_true_labels[selected_indices]).sum().item()
                purity = (correct_count / retained_count) * 100
                
                # 計算類別覆蓋率
                unique_classes = torch.unique(pseudo_labels[selected_indices])
                coverage_count = len(unique_classes)
                coverage_rate = (coverage_count / num_classes) * 100
            else:
                retained_count = 0
                purity = 0.0
                coverage_count = 0
                coverage_rate = 0.0
            
            # 5. 記錄結果
            result = {
                'k': int(k),
                'logit_scale': logit_scale,
                'retained_samples': retained_count,
                'retention_rate': (retained_count / dataset_size) * 100,
                'purity': purity,
                'coverage_count': coverage_count,
                'coverage_rate': coverage_rate,
                'class_sample_counts': class_sample_counts
            }
            all_results.append(result)
            
            # 6. 檢查是否符合甜蜜點條件
            is_sweet_spot = (
                target_samples[0] <= retained_count <= target_samples[1] and
                target_purity[0] <= purity <= target_purity[1] and
                coverage_rate >= target_coverage
            )
            
            status = "✓ 甜蜜點" if is_sweet_spot else ""
            if is_sweet_spot:
                sweet_spot_results.append(result)
            
            # 打印結果
            print(f"{k:<8} {logit_scale:<12.1f} {retained_count:<15} "
                  f"{purity:<12.2f} {coverage_rate:<12.2f} {coverage_count:<12} {status:<10}")
    
    print(f"{'-'*95}\n")
    
    # 顯示摘要
    print(f"\n{'='*80}")
    print(f"搜索結果摘要 (Class-wise Top-K)")
    print(f"{'='*80}")
    print(f"  總測試組合數      : {len(all_results)}")
    print(f"  符合甜蜜點組合數  : {len(sweet_spot_results)}")
    
    if sweet_spot_results:
        print(f"\n{'='*80}")
        print(f"符合條件的甜蜜點參數組合 (共 {len(sweet_spot_results)} 組):")
        print(f"{'='*80}")
        print(f"{'排名':<6} {'K':<8} {'LogitScale':<12} {'保留樣本':<12} {'純度(%)':<12} {'覆蓋率(%)':<12}")
        print(f"{'-'*70}")
        
        # 按照純度排序（優先考慮純度）
        sweet_spot_results.sort(key=lambda x: (-x['purity'], -x['coverage_rate']))
        
        for idx, result in enumerate(sweet_spot_results, 1):
            print(f"{idx:<6} {result['k']:<8} {result['logit_scale']:<12.1f} "
                  f"{result['retained_samples']:<12} {result['purity']:<12.2f} {result['coverage_rate']:<12.2f}")
        
        # 推薦最佳參數
        best_result = sweet_spot_results[0]
        print(f"\n{'='*80}")
        print(f"🎯 推薦最佳參數組合 (Class-wise Top-K):")
        print(f"{'='*80}")
        print(f"  K (每類樣本數)    : {best_result['k']}")
        print(f"  Logit Scale       : {best_result['logit_scale']:.1f}")
        print(f"  保留樣本數        : {best_result['retained_samples']} / {dataset_size} ({best_result['retention_rate']:.2f}%)")
        print(f"  Pseudo Label 純度 : {best_result['purity']:.2f}%")
        print(f"  類別覆蓋率        : {best_result['coverage_count']}/{num_classes} ({best_result['coverage_rate']:.2f}%)")
        
        # 顯示每個類別的樣本數分佈
        print(f"\n  各類別選取樣本數:")
        class_counts = best_result['class_sample_counts']
        if class_counts:
            counts_list = [class_counts.get(c, 0) for c in range(num_classes)]
            print(f"    平均: {np.mean([v for v in counts_list if v > 0]):.1f}")
            print(f"    最大: {max(counts_list)}")
            print(f"    最小: {min([v for v in counts_list if v > 0]) if any(v > 0 for v in counts_list) else 0}")
            print(f"    有樣本的類別數: {sum(1 for v in counts_list if v > 0)}/{num_classes}")
        print(f"{'='*80}")
    else:
        print(f"\n⚠️  警告: 沒有找到符合所有條件的參數組合！")
        print(f"\n建議調整目標條件或擴大搜索範圍。")
        
        # 顯示最接近的幾組結果
        print(f"\n以下是最接近目標的前 5 組參數:")
        print(f"{'K':<8} {'LogitScale':<12} {'保留樣本':<15} {'純度(%)':<12} {'覆蓋率(%)':<12}")
        print(f"{'-'*70}")
        
        # 按照綜合得分排序
        all_results.sort(key=lambda x: (
            abs(x['retained_samples'] - (target_samples[0] + target_samples[1]) / 2) / dataset_size +
            abs(x['purity'] - (target_purity[0] + target_purity[1]) / 2) / 100 +
            max(0, target_coverage - x['coverage_rate']) / 100
        ))
        
        for result in all_results[:5]:
            print(f"{result['k']:<8} {result['logit_scale']:<12.1f} "
                  f"{result['retained_samples']:<15} {result['purity']:<12.2f} {result['coverage_rate']:<12.2f}")
    
    print(f"\n{'='*80}\n")
    
    return sweet_spot_results if sweet_spot_results else all_results[:5]


def search_optimal_parameters(source_visual_prototypes, all_features, all_true_labels,
                             threshold_range=(0.7, 0.99), logit_scale_range=(5, 15),
                             threshold_step=0.01, logit_scale_step=0.5,
                             target_samples=(100, 200), target_purity=(75, 85),
                             target_coverage=85):
    """
    搜索最佳的 threshold 和 logit_scale 參數組合（使用信心門檻策略）
    
    Args:
        source_visual_prototypes: [C, D] Source prototypes
        all_features: [N, D] Target 特徵
        all_true_labels: [N] Target 真實標籤
        threshold_range: (min, max) threshold 範圍
        logit_scale_range: (min, max) logit_scale 範圍
        threshold_step: threshold 步長
        logit_scale_step: logit_scale 步長
        target_samples: (min, max) 目標保留樣本數範圍
        target_purity: (min, max) 目標純度範圍 (%)
        target_coverage: 最小類別覆蓋率 (%)
    
    Returns:
        results: list of dict，包含符合條件的參數組合
    """
    print(f"\n{'='*80}")
    print(f"開始參數搜索 (信心門檻策略)")
    print(f"{'='*80}")
    print(f"  Threshold 範圍    : {threshold_range[0]:.2f} ~ {threshold_range[1]:.2f} (步長 {threshold_step})")
    print(f"  Logit Scale 範圍  : {logit_scale_range[0]} ~ {logit_scale_range[1]} (步長 {logit_scale_step})")
    print(f"  目標保留樣本數    : {target_samples[0]} ~ {target_samples[1]}")
    print(f"  目標純度          : {target_purity[0]}% ~ {target_purity[1]}%")
    print(f"  最小類別覆蓋率    : {target_coverage}%")
    print(f"{'='*80}\n")
    
    dataset_size = len(all_features)
    num_classes = source_visual_prototypes.shape[0]
    
    # 生成參數組合
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    logit_scales = np.arange(logit_scale_range[0], logit_scale_range[1] + logit_scale_step, logit_scale_step)
    
    all_results = []
    sweet_spot_results = []
    
    print(f"正在測試 {len(thresholds)} x {len(logit_scales)} = {len(thresholds) * len(logit_scales)} 種參數組合...\n")
    print(f"{'Threshold':<12} {'LogitScale':<12} {'保留樣本':<15} {'純度(%)':<12} {'覆蓋率(%)':<12} {'覆蓋類別':<12} {'狀態':<10}")
    print(f"{'-'*95}")
    
    for logit_scale in logit_scales:
        for threshold in thresholds:
            # 1. 全域置中
            target_mean = all_features.mean(dim=0, keepdim=True)
            source_mean = source_visual_prototypes.mean(dim=0, keepdim=True).cpu()
            
            centered_target = F.normalize(all_features - target_mean, dim=-1)
            centered_source = F.normalize(source_visual_prototypes.cpu() - source_mean, dim=-1)
            
            # 2. 計算相似度和 Pseudo Labels
            logits = logit_scale * (centered_target @ centered_source.T)
            probs = F.softmax(logits, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            
            # 3. 應用信心門檻
            confidence_mask = max_probs > threshold
            retained_count = confidence_mask.sum().item()
            
            # 4. 計算純度
            if retained_count > 0:
                correct_count = ((pseudo_labels == all_true_labels) & confidence_mask).sum().item()
                purity = (correct_count / retained_count) * 100
            else:
                purity = 0.0
            
            # 5. 計算類別覆蓋率
            if retained_count > 0:
                unique_classes = torch.unique(pseudo_labels[confidence_mask])
                coverage_count = len(unique_classes)
                coverage_rate = (coverage_count / num_classes) * 100
            else:
                coverage_count = 0
                coverage_rate = 0.0
            
            # 6. 記錄結果
            result = {
                'threshold': threshold,
                'logit_scale': logit_scale,
                'retained_samples': retained_count,
                'retention_rate': (retained_count / dataset_size) * 100,
                'purity': purity,
                'coverage_count': coverage_count,
                'coverage_rate': coverage_rate,
            }
            all_results.append(result)
            
            # 7. 檢查是否符合甜蜜點條件
            is_sweet_spot = (
                target_samples[0] <= retained_count <= target_samples[1] and
                target_purity[0] <= purity <= target_purity[1] and
                coverage_rate >= target_coverage
            )
            
            status = "✓ 甜蜜點" if is_sweet_spot else ""
            if is_sweet_spot:
                sweet_spot_results.append(result)
            
            # 打印結果
            print(f"{threshold:<12.2f} {logit_scale:<12.1f} {retained_count:<15} "
                  f"{purity:<12.2f} {coverage_rate:<12.2f} {coverage_count:<12} {status:<10}")
    
    print(f"{'-'*95}\n")
    
    # 顯示摘要
    print(f"\n{'='*80}")
    print(f"搜索結果摘要 (信心門檻策略)")
    print(f"{'='*80}")
    print(f"  總測試組合數      : {len(all_results)}")
    print(f"  符合甜蜜點組合數  : {len(sweet_spot_results)}")
    
    if sweet_spot_results:
        print(f"\n{'='*80}")
        print(f"符合條件的甜蜜點參數組合 (共 {len(sweet_spot_results)} 組):")
        print(f"{'='*80}")
        print(f"{'排名':<6} {'Threshold':<12} {'LogitScale':<12} {'保留樣本':<12} {'純度(%)':<12} {'覆蓋率(%)':<12}")
        print(f"{'-'*70}")
        
        # 按照純度排序（優先考慮純度）
        sweet_spot_results.sort(key=lambda x: (-x['purity'], -x['coverage_rate']))
        
        for idx, result in enumerate(sweet_spot_results, 1):
            print(f"{idx:<6} {result['threshold']:<12.2f} {result['logit_scale']:<12.1f} "
                  f"{result['retained_samples']:<12} {result['purity']:<12.2f} {result['coverage_rate']:<12.2f}")
        
        # 推薦最佳參數
        best_result = sweet_spot_results[0]
        print(f"\n{'='*80}")
        print(f"🎯 推薦最佳參數組合 (信心門檻策略):")
        print(f"{'='*80}")
        print(f"  Threshold         : {best_result['threshold']:.2f}")
        print(f"  Logit Scale       : {best_result['logit_scale']:.1f}")
        print(f"  保留樣本數        : {best_result['retained_samples']} / {dataset_size} ({best_result['retention_rate']:.2f}%)")
        print(f"  Pseudo Label 純度 : {best_result['purity']:.2f}%")
        print(f"  類別覆蓋率        : {best_result['coverage_count']}/{num_classes} ({best_result['coverage_rate']:.2f}%)")
        print(f"{'='*80}")
    else:
        print(f"\n⚠️  警告: 沒有找到符合所有條件的參數組合！")
        print(f"\n建議調整目標條件或擴大搜索範圍。")
        
        # 顯示最接近的幾組結果
        print(f"\n以下是最接近目標的前 5 組參數:")
        print(f"{'Threshold':<12} {'LogitScale':<12} {'保留樣本':<15} {'純度(%)':<12} {'覆蓋率(%)':<12}")
        print(f"{'-'*70}")
        
        # 按照綜合得分排序
        all_results.sort(key=lambda x: (
            abs(x['retained_samples'] - (target_samples[0] + target_samples[1]) / 2) / dataset_size +
            abs(x['purity'] - (target_purity[0] + target_purity[1]) / 2) / 100 +
            max(0, target_coverage - x['coverage_rate']) / 100
        ))
        
        for result in all_results[:5]:
            print(f"{result['threshold']:<12.2f} {result['logit_scale']:<12.1f} "
                  f"{result['retained_samples']:<15} {result['purity']:<12.2f} {result['coverage_rate']:<12.2f}")
    
    print(f"\n{'='*80}\n")
    
    return sweet_spot_results if sweet_spot_results else all_results[:5]


def cal_acc(loader, netF, netP, source_visual_prototypes, flag=False, logit_scale=100.0):
    """使用 CLIP + Projector + Source Prototypes 計算 Target 準確率"""
    all_output, all_label = None, None

    netF.eval()
    netP.eval()

    with torch.no_grad():
        for data in loader:
            inputs = data[0].cuda()
            labels = data[1].float()

            feat_512       = netF(inputs.type(netF.conv1.weight.dtype))
            feat_projected = netP(feat_512)
            feat_norm      = F.normalize(feat_projected, dim=-1)

            src_norm = F.normalize(source_visual_prototypes, dim=-1)
            outputs  = logit_scale * (feat_norm @ src_norm.T)           # [B, C]

            outputs_cpu = outputs.float().cpu()
            if all_output is None:
                all_output = outputs_cpu
                all_label  = labels
            else:
                all_output = torch.cat((all_output, outputs_cpu), 0)
                all_label  = torch.cat((all_label, labels), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = (torch.squeeze(predict).float() == all_label).float().mean().item()
    mean_ent  = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc     = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc    = acc.mean()
        aa      = [str(np.round(i, 2)) for i in acc]
        return aacc, ' '.join(aa)
    else:
        return accuracy * 100, mean_ent


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='整合: Text Prompt 生成 + Pseudo Label 品質檢查'
    )
    # ---- 通用設定 ----
    parser.add_argument('--gpu_id',     type=str,   default='0')
    parser.add_argument('--seed',       type=int,   default=2022)
    parser.add_argument('--net',        type=str,   default='RN101',
                        choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'],
                        help='CLIP backbone')
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--worker',     type=int,   default=0)

    # ---- Text Prompt 設定 ----
    parser.add_argument('--classname_file', type=str,
                        default='/mnt/backups/andycw/M58/classname37.txt',
                        help='類別名稱檔案路徑')
    parser.add_argument('--projector_path', type=str,
                        default='/mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/37_class_100epoch_noPropagation/target_ProjectorC_ema_current.pt',
                        help='訓練好的 Source Projector 路徑')

    # ---- Target (Pseudo Label 分析) ----
    parser.add_argument('--dset',   type=str, default='M58',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'M58'])
    parser.add_argument('--s',      type=int, default=0, help='source domain index')
    parser.add_argument('--t',      type=int, default=1, help='target domain index')
    parser.add_argument('--da',     type=str, default='uda', choices=['uda', 'pda'])

    # ---- Pseudo Label 參數 ----
    parser.add_argument('--threshold',        type=float, default=0.95,
                        help='信心門檻 (0~1)')
    parser.add_argument('--momentum',         type=float, default=0.99,
                        help='EMA 更新動量')
    parser.add_argument('--pseudo_iterations', type=int,  default=1,
                        help='Prototype EMA 迭代次數')
    parser.add_argument('--logit_scale',      type=float, default=15,
                        help='cal_acc 用的 Logit Scale')

    args = parser.parse_args()

    # ---- 衍生設定 ----
    args.clip_dim = 1024 if args.net == 'RN50' else 512

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    elif args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    elif args.dset == 'M58':
        names = ['CAD_ratioFilter_StableDiffusion_random', 'Real_all_nobg']
        args.class_num = 37

    folder = 'data/'
    if args.dset == 'M58':
        args.t_dset_path    = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
    else:
        args.t_dset_path    = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    # ---- 環境初始化 ----
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"整合版: Text Prompt + Pseudo Label 品質檢查")
    print(f"{'='*60}")
    print(f"  Backbone   : {args.net}  (dim={args.clip_dim})")
    print(f"  Dataset    : {args.dset}  (classes={args.class_num})")
    print(f"  Target     : {names[args.t]}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Momentum   : {args.momentum}")
    print(f"  Iterations : {args.pseudo_iterations}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    # ============================================================
    # Step 1: 生成 Text Features (使用 CLIP Text Encoder)
    # ============================================================
    
    # 載入完整 CLIP 模型（包含 text encoder）
    clip_model_full, _ = clip.load(
        args.net, device=device,
        download_root=os.path.expanduser("~/.cache/clip")
    )
    clip_model_full.float()
    clip_model_full.eval()
    
    # 讀取類別名稱
    if not os.path.exists(args.classname_file):
        raise FileNotFoundError(f"找不到類別名稱檔案: {args.classname_file}")
    
    with open(args.classname_file, 'r', encoding='utf-8') as f:
        classnames = [line.strip() for line in f.readlines()]
    
    print(f"✅ 類別名稱載入: {len(classnames)} 個類別")
    
    # 驗證類別一致性
    if len(classnames) != args.class_num:
        raise ValueError(
            f"類別名稱檔案的類別數 ({len(classnames)}) "
            f"與 args.class_num ({args.class_num}) 不一致！"
        )
    
    # 生成 Text Features
    text_features = generate_text_features(classnames, clip_model_full, device)

    # ============================================================
    # Step 2: 載入 Target 資料 + CLIP Visual Encoder + Projector
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 2: 載入 Target 資料與模型")
    print(f"{'='*60}")

    # 使用已載入的 CLIP 模型的 Visual Encoder
    netF = clip_model_full.visual
    netF.eval()
    for param in netF.parameters():
        param.requires_grad = False
    print(f"✅ CLIP Visual Encoder 載入並凍結")

    netP = MLPProjector(in_dim=args.clip_dim, out_dim=args.clip_dim).to(device)
    netP.load_state_dict(torch.load(args.projector_path, map_location='cpu'))
    netP.eval()
    print(f"✅ Projector 載入完成: {args.projector_path}")

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
    # Step 4: 參數搜索（找到最佳 threshold 和 logit_scale）
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 4: 參數搜索 - 尋找最佳參數組合")
    print(f"{'='*60}")
    
    # 選擇搜索策略
    print(f"\n請選擇搜索策略:")
    print(f"  1. 信心門檻策略 (Confidence Threshold)")
    print(f"  2. Class-wise Top-K 策略")
    print(f"  3. 兩者都測試並比較\n")
    
    # 預設使用策略 3（兩者都測試）
    strategy_choice = 1
    
    optimal_params_threshold = None
    optimal_params_topk = None
    
    if strategy_choice in [1, 3]:
        print(f"\n{'─'*80}")
        print(f"策略 1: 信心門檻策略")
        print(f"{'─'*80}")
        optimal_params_threshold = search_optimal_parameters(
            source_visual_prototypes=text_features,
            all_features=all_features,
            all_true_labels=all_true_labels,
            threshold_range=(0.71, 0.71),
            logit_scale_range=(8, 8.1),
            threshold_step=0.01,
            logit_scale_step=0.1,
            target_samples=(150, 200),
            target_purity=(75, 90),
            target_coverage=70
        )
    
    if strategy_choice in [2, 3]:
        print(f"\n{'─'*80}")
        print(f"策略 2: Class-wise Top-K 策略")
        print(f"{'─'*80}")
        optimal_params_topk = search_optimal_parameters_topk(
            source_visual_prototypes=text_features,
            all_features=all_features,
            all_true_labels=all_true_labels,
            k_range=(1, 1),
            logit_scale_range=(5, 15),
            k_step=1,
            logit_scale_step=1.0,
            target_samples=(100, 200),
            target_purity=(75, 85),
            target_coverage=85
        )
    
    # 比較兩種策略
    if strategy_choice == 3 and optimal_params_threshold and optimal_params_topk:
        print(f"\n{'='*80}")
        print(f"策略比較")
        print(f"{'='*80}")
        
        best_threshold = optimal_params_threshold[0]
        best_topk = optimal_params_topk[0]
        
        print(f"\n信心門檻策略最佳結果:")
        print(f"  Threshold: {best_threshold['threshold']:.2f}, Logit Scale: {best_threshold['logit_scale']:.1f}")
        print(f"  保留樣本: {best_threshold['retained_samples']}, 純度: {best_threshold['purity']:.2f}%, 覆蓋率: {best_threshold['coverage_rate']:.2f}%")
        
        print(f"\nClass-wise Top-K 策略最佳結果:")
        print(f"  K: {best_topk['k']}, Logit Scale: {best_topk['logit_scale']:.1f}")
        print(f"  保留樣本: {best_topk['retained_samples']}, 純度: {best_topk['purity']:.2f}%, 覆蓋率: {best_topk['coverage_rate']:.2f}%")
        
        # 根據純度選擇更好的策略
        if best_threshold['purity'] > best_topk['purity']:
            print(f"\n✨ 推薦使用「信心門檻策略」(純度更高)")
            optimal_params = optimal_params_threshold
            strategy_name = "信心門檻策略"
        else:
            print(f"\n✨ 推薦使用「Class-wise Top-K 策略」(純度更高)")
            optimal_params = optimal_params_topk
            strategy_name = "Class-wise Top-K"
        print(f"{'='*80}")
    elif optimal_params_threshold:
        optimal_params = optimal_params_threshold
        strategy_name = "信心門檻策略"
    elif optimal_params_topk:
        optimal_params = optimal_params_topk
        strategy_name = "Class-wise Top-K"
    else:
        optimal_params = None
        strategy_name = None
    
    # ============================================================
    # Step 5: 使用最佳參數進行 Pseudo Label 迭代更新
    # ============================================================
    if optimal_params:
        best_param = optimal_params[0]
        print(f"\n{'='*60}")
        print(f"Step 5: 使用最佳參數進行 Pseudo Label 迭代分析 ({args.pseudo_iterations} 次)")
        print(f"{'='*60}")
        print(f"  選用策略: {strategy_name}")
        
        # 根據策略顯示不同的參數
        if 'threshold' in best_param:
            print(f"  使用參數: Threshold={best_param['threshold']:.2f}, Logit Scale={best_param['logit_scale']:.1f}\n")
            threshold_to_use = best_param['threshold']
        else:
            print(f"  使用參數: K={best_param['k']}, Logit Scale={best_param['logit_scale']:.1f}\n")
            # 對於 Top-K 策略，這裡暫時使用一個較低的 threshold 來配合
            threshold_to_use = 0.0  # Top-K 不需要 threshold
        
        for i in range(args.pseudo_iterations):
            print(f"--- Iteration {i+1}/{args.pseudo_iterations} ---")
            _, _, text_features = update_pseudo_labels(
                text_features,
                all_features,
                all_true_labels,
                threshold=threshold_to_use if 'threshold' in best_param else 0.0,
                momentum=args.momentum,
                logit_scale=best_param['logit_scale']
            )
    else:
        print(f"\n⚠️  跳過迭代更新（未找到合適參數）")

    # ============================================================
    # Step 5: 最終評估
    # ============================================================
    # print(f"\n{'='*60}")
    # print(f"Step 5: 最終 Target 準確率評估")
    # print(f"{'='*60}")

    # if args.dset == 'VISDA-C':
    #     acc, acc_list = cal_acc(
    #         dset_loaders['test'], netF, netP,
    #         source_visual_prototypes, flag=True,
    #         logit_scale=args.logit_scale
    #     )
    #     print(f"Per-class Accuracy: {acc:.2f}%")
    #     print(acc_list)
    # else:
    #     acc, mean_ent = cal_acc(
    #         dset_loaders['test'], netF, netP,
    #         source_visual_prototypes, flag=False,
    #         logit_scale=args.logit_scale
    #     )
    #     print(f"Target Accuracy  : {acc:.2f}%")
    #     print(f"Mean Entropy     : {mean_ent:.4f}")

    print(f"\n{'='*60}")
    print(f"✅ 所有步驟完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
