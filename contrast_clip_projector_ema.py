"""
CLIP + MLPProjector EMA Teacher 的 SFDA 訓練腳本
架構：
  CLIP RN101 Visual Encoder (完全凍結)
  + Student MLP ProjectorC (可訓練，用於分類)
  + Student MLP ProjectorH (可訓練，串接在 ProjectorC 後，只用於對比學習)
  + Teacher MLP ProjectorC_ema (EMA，隨 Student ProjectorC 被動更新)

損失函數（四個）：
  [A] CE Loss          - 高信心 Pseudo-label 樣本的 cross-entropy（reliable）
  [B] Propagation Loss - 低信心樣本的 MSE 對齊 Teacher logits（unreliable）
  [C] Information Loss - 全樣本的熵最小化 + diversity（im_loss）
  [D] Contrastive Loss - 弱/強增強特徵的 SimCLR 對比學習

特徵使用原則：
  - Student 使用 Strong augmented image → ProjectorC → 用於 CE、Propagation Loss
  - Student 使用 Original image → ProjectorC → 用於 IM Loss
  - Teacher 使用 Original image → ProjectorC_ema → 用於 Pseudo Label 生成
  - Contrastive Loss 使用 Weak + Strong → ProjectorC → ProjectorH（串聯）

訓練策略：
  - Phase 1 (Epoch 0-19):
    * CE=1.0, Con=1.0, Prop=0.0 (快速拓荒)
    * logitScale=11, Threshold=固定(args.conf_thres)
  - Phase 2 (Epoch 20+):
    * CE=1.0, Con=1.0, Prop 平滑增長至 1.0 (軟知識蒸餾)
    * logitScale=5, Threshold=動態平均信心度（參考 C-SFDA）
  - Pseudo Label 計算方式: 每個 batch 即時計算（用最新 Teacher EMA）
  - 每個 Epoch 結束時使用 obtain_label_clip 監控整體 Pseudo Label 質量

參考來源：
  - contrast_feature_micro.py （主骨架、Contrastive Loss、IM Loss）
  - C-SFDA/target_csfda.py （EMA Teacher、CE Loss、Propagation Loss、動態 Threshold）
  - clip_adaptation_projector.py （Projector 架構、Text Feature 生成）
"""

import argparse
import os
import datetime
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import copy
import random
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import clip
import math


# ============================================================
# 模型定義
# ============================================================

class MLPProjector(nn.Module):
    """MLP Projector（與 Source Model 訓練時完全相同的架構）"""
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


class Augmentation(object):
    """
    返回三個版本的圖片：
    1. weak augmentation (Aug 1)：輕度幾何變換 → Teacher 輸入
    2. strong augmentation (Aug 2)：SimCLR風格 (光影/模糊/幾何) → Student 輸入
    3. original：只做 Resize + CenterCrop，用於 Pseudo Label 監控（訓練時不使用）
    """
    def __init__(self, resize_size=256, crop_size=224):
        # 這是 OpenAI CLIP 官方標準的 Mean 和 Std，絕對不能用 ImageNet 的！
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # Original: 完全對齊 CLIP 官方 preprocess（與提取 centroid 時一致）
        # CLIP 官方使用 Resize(224) 而非 Resize((224, 224))，這會保持長寬比
        self.original = transforms.Compose([
            transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),  # 短邊縮放
            transforms.CenterCrop(crop_size)
        ])

        # Weak: 輕度的幾何變換 (平移、旋轉、翻轉)，不改變光影
        self.weak = transforms.Compose([
            transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])

        # Strong: SimCLR 風格 (加入強烈的光影擾動與模糊，強迫模型學習形狀輪廓)
        self.strong = transforms.Compose([
            transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(crop_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(0.1 * crop_size) | 1,
                                                             sigma=(0.1, 2.0))], p=0.5)
        ])

    def __call__(self, x):
        weak     = self.weak(x)
        strong   = self.strong(x)
        original = self.original(x)
        # 最後統一套用 CLIP 的 Normalization
        return self.normalize(weak), self.normalize(strong), self.normalize(original)


# ============================================================
# 工具函數
# ============================================================

def smoothed_cross_entropy(logits, labels, num_classes=37, epsilon=0.1):
    """
    epsilon=0.1 代表信任度為 90%，剩下的 10% 機率均分給其他類別。
    這能極大程度防止模型對錯誤的 Pseudo-label 產生 Overfitting。
    """
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        # 建立 One-hot targets
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        # 混入均勻分佈
        targets = (1 - epsilon) * targets + epsilon / num_classes
    
    # 計算 Soft Cross Entropy
    loss = (-targets * log_probs).sum(dim=1).mean()
    return loss

def load_clip_to_cpu(backbone):
    """載入 CLIP 模型（CPU 版，後續移至 GPU）"""
    model, _ = clip.load(backbone, device="cpu",
                         download_root=os.path.expanduser("~/.cache/clip"))
    return model


def op_copy(optimizer):
    """記錄初始學習率，供 lr_scheduler 使用"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """反向衰減學習率排程器"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr']           = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum']     = 0.9
        param_group['nesterov']     = True
    return optimizer


def get_clip_preprocess_transforms(crop_size=224):
    """
    返回 CLIP 官方 preprocess 的 Resize + CenterCrop 部分（不含 ToTensor 和 Normalize）
    用於 Augmentation 的 original 分支
    """
    return transforms.Compose([
        transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size)
    ])


def generate_text_features(classnames_path, clip_model, device, template="a photo of a {}"):
    """
    從 classname 檔案生成 CLIP Text Features

    Args:
        classnames_path: 類別名稱檔案路徑（每行一個類別名稱）
        clip_model:      已移至 device 的完整 CLIP 模型
        device:          計算裝置字串
        template:        提示詞模板，預設 "a photo of a {}"

    Returns:
        text_features: [C, 512] L2 正規化後的 Text Features (float32, on device)
        classnames:    類別名稱 list
    """
    with open(classnames_path, 'r', encoding='utf-8') as f:
        classnames = [line.strip() for line in f if line.strip()]

    print(f'生成 Text Features')
    print(f'  - 類別數量  : {len(classnames)}')
    print(f'  - 提示詞模板: "{template}"')

    clip_model.eval()
    with torch.no_grad():
        texts        = [template.format(name) for name in classnames]
        tokens       = clip.tokenize(texts).to(device)
        text_features = clip_model.encode_text(tokens)        # [C, D]
        text_features = F.normalize(text_features.float(), dim=-1)

    print(f'Text Features 生成完畢: shape = {text_features.shape}')

    return text_features, classnames


def data_load(args, preprocess):
    """載入資料集（Target 訓練 + 評估）
    
    Args:
        preprocess: CLIP 官方的 preprocess（用於 test loader）
    """
    dsets       = {}
    dset_loaders = {}
    train_bs    = args.batch_size

    txt_tar  = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    root_path = f'data/{args.dset}/'

    # 訓練 Loader：每個樣本回傳 (weak, strong, original) 三張增強圖
    dsets["target"] = ImageList_idx(txt_tar, transform=Augmentation(), root=root_path)
    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False
    )

    # 測試 Loader：使用 CLIP 官方 preprocess（完全對齊 extract_class_centroid.py）
    dsets["test"] = ImageList_idx(txt_test, transform=preprocess, root=root_path)
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=train_bs * 3, shuffle=False,
        num_workers=args.worker, drop_last=False
    )

    return dset_loaders


def cal_acc(loader, netF, netP, class_centroids, target_global_mean, source_global_mean, args, flag=False):
    """
    使用 Student Projector 評估準確率

    Args:
        class_centroids: [C, D] 從 Source 資料集提取的類別中心特徵（已正規化）
        target_global_mean: [1, D] Target 全域均值
        source_global_mean: [1, D] Source 全域均值
    """
    netP.eval()
    clip_dtype = next(netF.parameters()).dtype

    # 預計算置中後的 source centroids
    centered_source_centroids = F.normalize(class_centroids - source_global_mean, dim=-1)

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data   = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]

            feat_512  = netF(inputs.type(clip_dtype))       # [B, 512]，凍結
            feat_proj = netP(feat_512)                       # [B, 512]
            feat_norm = F.normalize(feat_proj, dim=-1)

            # 【修改】使用全域置中的方式計算 logits
            centered_features = F.normalize(feat_norm - target_global_mean, dim=-1)
            outputs = args.centroid_logitScale * (centered_features @ centered_source_centroids.t())  # [B, C]

            if start_test:
                all_output = outputs.float().cpu()
                all_label  = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label  = torch.cat((all_label, labels.float()), 0)

    netP.train()

    _, predict = torch.max(all_output, 1)
    accuracy   = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent   = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix   = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc      = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc     = acc.mean()
        aa       = [str(np.round(i, 2)) for i in acc]
        acc_str  = ' '.join(aa)
        return aacc, acc_str
    else:
        return accuracy * 100, mean_ent


def obtain_label_clip(loader, netF, netP_ema, class_centroids, args, current_epoch=0):
    """
    使用 Teacher Projector (EMA, eval mode) 產生全資料集的 Pseudo Labels
    【注意】此函數現在只用於監控整體 Pseudo Label 質量，每個 Epoch 結束時調用一次
    訓練時的 Pseudo Labels 是在每個 batch 即時計算的（在主訓練循環中）

    修改：現在使用穩定的 Teacher 而非震盪的 Student
    修改：使用 class_centroids（從 Source 提取的類別中心）而非 text features
    修改：使用全域置中（Global Centering）+ logit_scale（和 plot_threshold_distribution.py 一致）
    修改：支援動態 threshold（參考 C-SFDA，epoch >= 20 使用平均信心度）

    Args:
        class_centroids: [C, D] 從 Source 資料集提取的類別中心特徵（已正規化）
        current_epoch: 當前 epoch 數（用於決定使用固定或動態 threshold）

    Returns:
        pseudo_labels : [N] (CPU LongTensor) 每個樣本的偽標籤
        max_probs     : [N] (CPU FloatTensor) 最大 softmax 信心值
    """
    netP_ema.eval()
    clip_dtype = next(netF.parameters()).dtype

    # 第一輪：提取所有 Target 特徵（用於全域置中）
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data   = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]

            feat_512  = netF(inputs.type(clip_dtype))
            feat_proj = netP_ema(feat_512)  # 使用 Teacher EMA
            feat_norm = F.normalize(feat_proj, dim=-1)
            
            all_features.append(feat_norm.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)  # [N, D]
    all_labels = torch.cat(all_labels, dim=0)      # [N]
    
    # 【關鍵】使用全域置中（和 plot_threshold_distribution.py 完全相同）
    target_mean = all_features.mean(dim=0, keepdim=True)
    source_mean = class_centroids.mean(dim=0, keepdim=True).cpu()
    
    centered_target = F.normalize(all_features - target_mean, dim=-1)
    centered_source = F.normalize(class_centroids.cpu() - source_mean, dim=-1)

    # 【新增】根據 epoch 決定 logitScale（與訓練邏輯一致）
    use_logit_scale = args.centroid_logitScale  # Phase 1: 11

    # calculate Logits
    logits = use_logit_scale * (centered_target @ centered_source.T)  # [N, C]

    # Softmax Turn Logits to probabiltiy
    all_probs = F.softmax(logits, dim=1)           # [N, C]
    max_probs, pseudo_labels = torch.max(all_probs, dim=1)  # [N]
    
    # 【新增】根據 epoch 決定 threshold（與訓練邏輯一致）
    if current_epoch + 1 < args.phase1_epoch:
        # Phase 1: 使用固定 threshold
        conf_threshold = args.conf_thres
        threshold_mode = f"Fixed({conf_threshold})"
    else:
        # Phase 2: 使用動態平均信心度（參考 C-SFDA）
        conf_threshold = max_probs.mean().item()
        # 加上底線保護（與訓練邏輯一致）
        conf_threshold = max(conf_threshold, 0.90)
        threshold_mode = f"Dynamic({conf_threshold:.4f})"

    # 記錄統計資訊（使用當前實際的 threshold）
    accuracy       = torch.sum(pseudo_labels.float() == all_labels.float()).item() / float(all_labels.size()[0])
    reliable_ratio = (max_probs > conf_threshold).float().mean().item()

    # 計算 reliable 樣本的 Purity（高信心樣本中真正正確的比例）
    reliable_mask = max_probs > conf_threshold
    if reliable_mask.sum() > 0:
        reliable_correct = torch.sum((pseudo_labels[reliable_mask].float() == all_labels[reliable_mask].float())).item()
        reliable_total = reliable_mask.sum().item()
        reliable_purity = reliable_correct / reliable_total
    else:
        reliable_purity = 0.0

    log_str = (
        f'[Pseudo Label via Teacher Centroids + Global Centering] 準確率 = {accuracy * 100:.2f}%  |  '
        f'logitScale={use_logit_scale}  |  '
        f'Threshold={threshold_mode}\n'
        f'  Reliable (conf > {conf_threshold:.4f}) = {reliable_ratio * 100:.2f}%  |  '
        f'Reliable 樣本數 = {int((max_probs > conf_threshold).sum())}/{len(pseudo_labels)}  |  '
        f'Reliable Purity = {reliable_purity * 100:.2f}%'
    )
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)

    return pseudo_labels, max_probs


def compute_target_global_mean(netF, netP_C_ema, data_loader, clip_dtype):
    """
    計算 Target 資料集的全域均值（使用當前的 Teacher ProjectorC_ema）

    Args:
        netF: CLIP Visual Encoder（凍結）
        netP_C_ema: Teacher ProjectorC（EMA）
        data_loader: Target data loader
        clip_dtype: CLIP 模型的 dtype

    Returns:
        target_global_mean: [1, D] cuda tensor
    """
    netF.eval()
    netP_C_ema.eval()
    all_target_features = []

    with torch.no_grad():
        for data in data_loader:
            imgs, _, _, _ = data
            _, _, inputs_origin = imgs
            inputs_origin = inputs_origin.cuda()

            # 使用 weak augmentation
            feat_512 = netF(inputs_origin.type(clip_dtype))
            feat_projected = netP_C_ema(feat_512)
            feat_norm = F.normalize(feat_projected, dim=-1)
            all_target_features.append(feat_norm.cpu())

    all_target_features = torch.cat(all_target_features, dim=0)  # [N, D]
    target_global_mean = all_target_features.mean(dim=0, keepdim=True).cuda()  # [1, D]

    return target_global_mean


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


# ============================================================
# 主訓練函數
# ============================================================

def train_target(args):
    # =====================================================
    # 載入並凍結 CLIP Visual Encoder（保留 preprocess）
    # =====================================================
    print(f'載入 CLIP Model: {args.net}')
    clip_model, preprocess = clip.load(args.net, device="cpu",
                                        download_root=os.path.expanduser("~/.cache/clip"))
    clip_model.float()
    clip_model = clip_model.cuda()
    print(f'✅ 同時載入 CLIP 官方 preprocess（與 extract_class_centroid.py 完全一致）')
    
    # 載入資料集（使用 CLIP 官方 preprocess）
    dset_loaders = data_load(args, preprocess)

    netF = clip_model.visual       # 只取 Visual Encoder
    netF.eval()
    for param in netF.parameters():
        param.requires_grad = False

    clip_dtype = next(netF.parameters()).dtype
    print(f'CLIP Visual Encoder 已載入並凍結 (eval mode)')
    print(f'  - dtype: {clip_dtype}')

    # =====================================================
    # 生成固定的 CLIP Text Features（用於訓練時的分類）
    # =====================================================
    text_features, classnames = generate_text_features(
        classnames_path=args.classnames_path,
        clip_model=clip_model,
        device='cuda',
        template="a photo of a {}"
    )
    text_features = text_features.cuda()   # [C, 512]，已 L2 正規化
    text_norm     = F.normalize(text_features, dim=-1)

    if len(classnames) != args.class_num:
        raise ValueError(
            f'class_num 不一致：classname 檔案有 {len(classnames)} 類，'
            f'但 args.class_num = {args.class_num}，請確認設定。'
        )

    # =====================================================
    # 載入 Class Centroids（用於 Pseudo Label 生成）
    # =====================================================
    print(f'\n載入 Class Centroids from: {args.centroid_path}')
    if not os.path.exists(args.centroid_path):
        raise FileNotFoundError(
            f'找不到 Class Centroids 檔案：{args.centroid_path}\n'
            f'請先執行 extract_class_centroid.py 從 Source 資料集提取類別中心。'
        )
    
    centroid_data = torch.load(args.centroid_path)
    class_centroids = centroid_data['centroids'].cuda()  # [C, 512]
    print(f'✅ Class Centroids 載入完成: {class_centroids.shape}')
    print(f'   來自類別數: {centroid_data.get("num_classes", "未知")}')
    
    if class_centroids.size(0) != args.class_num:
        raise ValueError(
            f'class_num 不一致：Centroid 檔案有 {class_centroids.size(0)} 類，'
            f'但 args.class_num = {args.class_num}，請確認設定。'
        )

    # =====================================================
    # 初始化混合串聯 Projector 架構並載入 Source 預訓練權重
    # =====================================================
    print(f'初始化混合串聯 Projector 架構：')
    print(f'  - ProjectorC：用於分類任務 (CE/Prop/IM Loss)')
    print(f'  - ProjectorH：串接在 ProjectorC 後，只用於對比學習 (Contrastive Loss)')
    print(f'  - Teacher 只有 ProjectorC_ema，不需要 ProjectorH_ema')

    netP_C = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).cuda()
    netP_H = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).cuda()

    # 優先使用直接指定路徑，否則從 output_dir_src 推導
    if args.projector_path is not None:
        projector_path = args.projector_path
    else:
        projector_path = osp.join(args.output_dir_src, 'source_P.pt')

    if not osp.exists(projector_path):
        raise FileNotFoundError(
            f'找不到 Projector 權重：{projector_path}\n'
            f'請確認路徑或透過 --projector_path 直接指定。'
        )

    # 兩個 Projector 都載入相同的預訓練權重
    state_dict = torch.load(projector_path)
    netP_C.load_state_dict(state_dict)
    netP_H.load_state_dict(state_dict)
    netP_C.train()
    netP_H.train()
    print(f'✅ ProjectorC 與 ProjectorH 已載入：{projector_path}')

    # =====================================================
    # 建立 Teacher Projector（EMA，只針對 ProjectorC）
    # =====================================================
    netP_C_ema = copy.deepcopy(netP_C)
    for param in netP_C_ema.parameters():
        param.requires_grad = False   # Teacher 完全凍結，只透過 EMA 更新
    netP_C_ema.eval()
    print(f'Teacher ProjectorC (EMA) 已建立（不需要 ProjectorH_ema）')

    # =====================================================
    # 設定優化器（同時優化 ProjectorC + ProjectorH）
    # =====================================================
    param_group = []
    for k, v in netP_C.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netP_H.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    print(f'優化器設定')
    print(f'  - 可訓練模組         : ProjectorC + ProjectorH (串聯)')
    print(f'  - 初始 LR            : {args.lr}')
    print(f'  - EMA momentum       : {args.ema_m} (只更新 Teacher ProjectorC)')
    print(f'  - conf_thres         : {args.conf_thres}')
    print(f'  - centroid_logitScale: {args.centroid_logitScale}  (監控用 Centroids)')
    print(f'\n  Loss 設定:')
    print(f'  [A] CE Loss         : Phase1=1.0, Phase2=1.0  (高信心 Pseudo-label CE)')
    print(f'  [B] Propagation Loss: Phase1=0.0, Phase2=0→1 (低信心 MSE to Teacher)')
    print(f'  [C] IM Loss         : ent_par  = {args.ent_par}  (熵最小化 + diversity)')
    print(f'  [D] Contrastive Loss: Phase1=1.0, Phase2=1.0  (SimCLR Weak+Strong, 動態熵權重)')
    print(f'\n  訓練策略:')
    print(f'  - Phase 1 (Epoch 0-19)  : CE=1.0, Con=1.0, Prop=0.0, logitScale=11, Threshold=固定({args.conf_thres})')
    print(f'  - Phase 2 (Epoch 20+)   : CE=1.0, Con=1.0, Prop=0→1, logitScale=5, Threshold=動態平均（參考 C-SFDA）')
    print(f'  - Pseudo Label 計算方式 : 每個 batch 即時計算 (用最新 Teacher EMA)')
    print(f'\n  輸入圖像分配:')
    print(f'  - CE / Propagation Loss : Strong augmented (Student)')
    print(f'  - IM Loss               : Original image (Student)')
    print(f'  - Contrastive Loss      : Weak + Strong (Student, 串聯至 ProjectorH)')
    print(f'  - Teacher (Pseudo Label): Original image (所有 epoch，參考 C-SFDA)')

    # =====================================================
    # Resume 訓練（如果指定了 resume_dir）
    # =====================================================
    start_epoch_offset = 0
    if args.resume_dir is not None:
        print(f'\n🔄 Resume 訓練模式啟動')
        print(f'   從目錄載入模型: {args.resume_dir}')

        # 檢查檔案是否存在（混合架構：3 個模型）
        projectorC_path = osp.join(args.resume_dir, "target_ProjectorC_best.pt")
        projectorH_path = osp.join(args.resume_dir, "target_ProjectorH_best.pt")
        projectorC_ema_path = osp.join(args.resume_dir, "target_ProjectorC_ema_current.pt")

        if not osp.exists(projectorC_path):
            raise FileNotFoundError(f'找不到 ProjectorC checkpoint: {projectorC_path}')
        if not osp.exists(projectorH_path):
            raise FileNotFoundError(f'找不到 ProjectorH checkpoint: {projectorH_path}')
        if not osp.exists(projectorC_ema_path):
            raise FileNotFoundError(f'找不到 ProjectorC_ema checkpoint: {projectorC_ema_path}')

        # 載入模型權重（不需要 netP_H_ema）
        netP_C.load_state_dict(torch.load(projectorC_path))
        netP_H.load_state_dict(torch.load(projectorH_path))
        netP_C_ema.load_state_dict(torch.load(projectorC_ema_path))

        print(f'   ✅ ProjectorC 已載入: {projectorC_path}')
        print(f'   ✅ ProjectorH 已載入: {projectorH_path}')
        print(f'   ✅ ProjectorC_ema 已載入: {projectorC_ema_path}')
        print(f'   (混合架構：不需要 ProjectorH_ema)')
        print(f'   📍 起始 Epoch: {args.start_epoch}')

        start_epoch_offset = args.start_epoch

        if args.con_par is not None:
            print(f'   🎯 固定 Contrastive Loss 權重: {args.con_par}')

    max_iter         = (args.max_epoch - args.start_epoch) * len(dset_loaders["target"])
    interval_iter    = max_iter // args.interval
    iter_num         = 0

    print(f'\nBatch size: {args.batch_size}')
    print(f'Total iterations: {max_iter}, Pseudo-label 更新間隔: {interval_iter} iters')
    print(f'Seed: {args.seed}')
    print(f'📂 Target數據路徑: {args.t_dset_path}')
    print(f'📂 Test數據路徑 : {args.test_dset_path}\n')

    # =====================================================
    # 【新增】預計算 Source 全域均值（固定）和初始 Target 全域均值
    # =====================================================
    print(f'預計算全域均值用於全域置中...')

    # Source 全域均值（固定，整個訓練期間不變）
    source_global_mean = class_centroids.mean(dim=0, keepdim=True)  # [1, D]

    # 初始 Target 全域均值（使用初始的 Teacher）
    target_global_mean = compute_target_global_mean(
        netF, netP_C_ema, dset_loaders["target"], clip_dtype
    )

    # 預計算置中後的 source centroids（會在每個 batch 後更新）
    centered_source_centroids = F.normalize(class_centroids - source_global_mean, dim=-1)  # [C, D]

    print(f'✅ 全域均值計算完成')
    print(f'   Source 全域均值: {source_global_mean.shape}（固定）')
    print(f'   Target 全域均值: {target_global_mean.shape}（初始值，每個 batch 後更新）')
    print(f'   置中後 Source Centroids: {centered_source_centroids.shape}')

    # 初始化最佳準確率
    acc_init = 0

    # Epoch 追蹤變數
    batches_per_epoch   = len(dset_loaders["target"])
    current_epoch       = start_epoch_offset  # 從 resume 的 epoch 開始
    epoch_start_iter    = 0
    epoch_total_loss    = 0.0
    epoch_ce_loss       = 0.0
    epoch_prop_loss     = 0.0
    epoch_entropy_loss  = 0.0
    epoch_contrast_loss = 0.0
    epoch_batches       = 0

    # 建立初始進度條
    iter_test = iter(dset_loaders["target"])
    pbar      = tqdm(total=batches_per_epoch,
                     desc=f'Epoch {current_epoch+1}/{args.max_epoch}',
                     leave=False)

    while iter_num < max_iter:

        # =================================================
        # 取得 Batch（三種增強）
        # =================================================
        try:
            imgs, true_labels, tar_idx, path = next(iter_test)
        except StopIteration:
            iter_test = iter(dset_loaders["target"])
            imgs, true_labels, tar_idx, path = next(iter_test)

        inputs_weak, inputs_strong, inputs_original = imgs

        if inputs_weak.size(0) == 1:
            iter_num += 1
            continue
        
        # 根據 epoch 切換 logitScale
        if current_epoch+1 > args.phase1_epoch:
            args.centroid_logitScale = 9.9     # Phase 2: 降低 logitScale
            
        # =================================================
        # 學習率排程 + 梯度清零
        # =================================================
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        optimizer.zero_grad()

        # GPU 移動
        inputs_weak     = inputs_weak.cuda()
        inputs_strong   = inputs_strong.cuda()
        inputs_original = inputs_original.cuda()  # IM Loss 需要使用 Original

        if iter_num % 100 == 0:
            torch.cuda.empty_cache()

        # --------------------------------------------------
        # 特徵提取（CLIP Encoder 凍結，不計梯度）
        # --------------------------------------------------
        with torch.no_grad():
            # Student 用 Strong，Teacher 用 Weak，IM Loss 用 Original
            feat_512_weak     = netF(inputs_weak.type(clip_dtype))      # [B, 512]
            feat_512_strong   = netF(inputs_strong.type(clip_dtype))    # [B, 512]
            feat_512_original = netF(inputs_original.type(clip_dtype))  # [B, 512]

        # --------------------------------------------------
        # Student: ProjectorC Forward (Strong aug) → CE / Propagation Loss
        # --------------------------------------------------
        feat_C_strong = netP_C(feat_512_strong)                           # [B, 512]，有梯度
        feat_stu_norm = F.normalize(feat_C_strong, dim=-1)

        # 【修改】使用全域置中的方式計算 Student logits（與 Teacher 對齊）
        centered_student_features = F.normalize(feat_stu_norm - target_global_mean, dim=-1)  # [B, D]
        logits_student = args.centroid_logitScale * (centered_student_features @ centered_source_centroids.t())  # [B, C]

        # --------------------------------------------------
        # Student: ProjectorC Forward (Original aug) → IM Loss
        # --------------------------------------------------
        feat_C_original = netP_C(feat_512_original)                       # [B, 512]，有梯度
        feat_orig_norm  = F.normalize(feat_C_original, dim=-1)

        # 【修改】使用全域置中的方式計算 Original logits
        centered_original_features = F.normalize(feat_orig_norm - target_global_mean, dim=-1)  # [B, D]
        logits_original = args.centroid_logitScale * (centered_original_features @ centered_source_centroids.t())  # [B, C]

        # --------------------------------------------------
        # Student: ProjectorC → ProjectorH (Weak + Strong) → Contrastive Loss
        # --------------------------------------------------
        feat_C_weak   = netP_C(feat_512_weak)                             # [B, 512]
        feat_H_weak   = netP_H(feat_C_weak)                               # [B, 512]，串聯
        feat_H_strong = netP_H(feat_C_strong)                             # [B, 512]，串聯（複用）

        # --------------------------------------------------
        # Teacher: ProjectorC_ema Forward (Original) → 即時計算 Pseudo Labels
        # --------------------------------------------------
        # 所有 epoch 都使用 Original（無增強），Phase 2 切換 logitScale（參考 C-SFDA）
        with torch.no_grad():
            # 所有 epoch 都使用 Original image
            feat_tea_input = netP_C_ema(feat_512_original)                # [B, 512]
            feat_tea_norm = F.normalize(feat_tea_input, dim=-1)

            # 【修改】使用 batch EMA 更新全域均值（避免每個 batch 遍歷整個 dataset）
            batch_mean = feat_tea_norm.mean(dim=0, keepdim=True)
            target_global_mean = F.normalize(target_global_mean * args.ema_m + batch_mean * (1 - args.ema_m), dim=-1)

            # 使用更新後的 target_global_mean 進行全域置中
            centered_batch_features = F.normalize(feat_tea_norm - target_global_mean, dim=-1)  # [B, D]

            # 使用置中後的 source centroids 計算相似度（logitScale 根據 Phase 切換）
            logits_teacher = args.centroid_logitScale * (centered_batch_features @ centered_source_centroids.t())  # [B, C]

            # 【關鍵】即時計算當前 batch 的 Pseudo Labels
            probs_teacher = F.softmax(logits_teacher, dim=1)              # [B, C]
            batch_max_probs, batch_pseudo_labels = torch.max(probs_teacher, dim=1)  # [B]

            # 動態計算當前 batch 的熵權重（基於 Teacher 的最新預測）
            entropy_batch = loss.Entropy(probs_teacher)                   # [B]
            batch_mas     = 1.0 - torch.exp(-entropy_batch)               # [B]，高熵→高權重

        # --------------------------------------------------
        # Confidence Mask（基於即時計算的 Pseudo Labels）
        # --------------------------------------------------
        # 【新增】Epoch >= 20 使用動態平均信心度作為 Threshold（參考 C-SFDA）
        if current_epoch +1 < args.phase1_epoch:
            # Phase 1: 使用固定 threshold
            conf_threshold = args.conf_thres
        else:
            # Phase 2: 使用動態平均信心度（參考 C-SFDA）
            current_batch_mean = batch_max_probs.mean()  # 當前 batch 的平均最大概率
            # 最後加上 clamp 保護底線 (例如 0.5 或你設定的 conf_thres)
            conf_threshold = torch.clamp(current_batch_mean, min=0.9).item()

        # 1. 高信心 (Reliable)：直接做 CE Loss
        reliable_mask = batch_max_probs > conf_threshold   # [B] bool

        # 2. 低信心 (Unreliable)：所有非高信心樣本
        unreliable_mask = ~reliable_mask                   # [B] bool

        # =================================================
        # 分段學習權重調整
        # =================================================
        if current_epoch + 1 < args.phase1_epoch:  # Phase 1 (Epoch 0-7)
            current_cls_par  = args.cls_par   # CE Loss = 1
            current_con_par  = args.con_par   # Contrastive Loss = 1
            current_prop_par = 0.0   # Prop Loss = 0
        else:
            # Phase 2 (Epoch 8+): 平滑啟動 Prop Loss + 進度綁定指數衰減 Contrastive Loss
            progress = (current_epoch - args.start_epoch) / (args.max_epoch - args.start_epoch)
            # [3] ✨ Contrastive Loss 進度綁定指數衰減 ✨
            # 當 progress 從 0 走到 1，exp(-1.0 * progress) 會從 1.0 平滑降至 0.368
            # 這完美避開了 Dataset Size 和 Batch Size 帶來的干擾！
            current_con_par  = args.con_par * math.exp(-1.0 * progress)
            current_prop_par = args.prop_par * progress   # Prop Loss 從 0 → 1 平滑增長
            current_cls_par  = 1.0 - current_prop_par              # CE Loss 保持 1 → 0.5

        losses = torch.tensor(0.0).cuda()

        # --------------------------------------------------
        # [A] CE Loss：高信心 Pseudo-label 樣本
        # --------------------------------------------------
        ce_loss = torch.tensor(0.0).cuda()
        reliable_count = reliable_mask.sum().item()

        if reliable_count > 0:
            ce_loss = smoothed_cross_entropy(
                logits_student[reliable_mask],
                batch_pseudo_labels[reliable_mask],
                num_classes=args.class_num,
                epsilon=0.15
            ) * current_cls_par
        losses += ce_loss

        # --------------------------------------------------
        # [B] Propagation Loss：低信心樣本靠近 Teacher logit 分布（MSE）
        # --------------------------------------------------
        prop_loss = torch.tensor(0.0).cuda()
        if unreliable_mask.sum() > 0 and current_prop_par > 0:
            prop_loss = F.mse_loss(
                logits_student[unreliable_mask],
                logits_teacher[unreliable_mask].detach()
            ) * current_prop_par  # 使用動態調整的 prop_par

        losses += prop_loss

        # --------------------------------------------------
        # [C] Information Loss：熵最小化（全部樣本，使用 Original image）
        # --------------------------------------------------
        im_loss = torch.tensor(0.0).cuda()
        if args.ent:
            softmax_out  = F.softmax(logits_original, dim=1)  # 改用 Original image 的 logits
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax      = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss  -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
        losses += im_loss

        # --------------------------------------------------
        # [D] Contrastive Loss：Weak + Strong 特徵的 SimCLR（加熵值權重）
        # --------------------------------------------------
        out_1 = F.normalize(feat_H_weak,   dim=-1)   # [B, 512]，經過 netP_C → netP_H
        out_2 = F.normalize(feat_H_strong, dim=-1)   # [B, 512]，經過 netP_C → netP_H
        out   = torch.cat([out_1, out_2], dim=0)      # [2B, 512]

        # 相似度矩陣（去除對角線自我相似度）
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.tt)  # [2B, 2B]
        diag_mask  = (torch.ones_like(sim_matrix) -
                      torch.eye(out.shape[0], device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(diag_mask).view(out.shape[0], -1)  # [2B, 2B-1]

        # 正樣本對（Weak → Strong 與 Strong → Weak）
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.tt)  # [B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)                    # [2B]

        # 熵值加權（高熵樣本 → 需要更多對比監督）
        micro_mas = torch.cat([batch_mas, batch_mas])                     # [2B]
        contrast_loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1)) * micro_mas).mean()
        contrast_loss = contrast_loss * current_con_par  # 使用動態調整的 con_par
        losses += contrast_loss

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------
        losses.backward()

        # 累積 Epoch 損失（for 結尾統計）
        epoch_ce_loss       += ce_loss.item()
        epoch_prop_loss     += prop_loss.item()
        epoch_entropy_loss  += im_loss.item()
        epoch_contrast_loss += contrast_loss.item()
        epoch_total_loss    += losses.item()
        epoch_batches       += 1

        # =================================================
        # 更新 Student 參數
        # =================================================
        optimizer.step()

        # =================================================
        # EMA 更新 Teacher ProjectorC（只更新分類用 Projector）
        # =================================================
        with torch.no_grad():
            for param_q, param_k in zip(netP_C.parameters(), netP_C_ema.parameters()):
                param_k.data = param_k.data * args.ema_m + param_q.data * (1.0 - args.ema_m)

        # =================================================
        # 進度條更新
        # =================================================
        pbar.update(1)
        pbar.set_postfix({
            'Total': f'{losses.item():.4f}',
            'CE':    f'{ce_loss.item():.4f}',
            'Prop':  f'{prop_loss.item():.4f}',
            'Ent':   f'{im_loss.item():.4f}',
            'Con':   f'{contrast_loss.item():.4f}'
        })

        # =================================================
        # Epoch 結束：評估 + 儲存
        # =================================================
        if (iter_num - epoch_start_iter + 1) % batches_per_epoch == 0:
            pbar.close()

            avg_total    = epoch_total_loss    / epoch_batches if epoch_batches > 0 else 0
            avg_ce       = epoch_ce_loss       / epoch_batches if epoch_batches > 0 else 0
            avg_prop     = epoch_prop_loss     / epoch_batches if epoch_batches > 0 else 0
            avg_entropy  = epoch_entropy_loss  / epoch_batches if epoch_batches > 0 else 0
            avg_contrast = epoch_contrast_loss / epoch_batches if epoch_batches > 0 else 0

            # 使用 ProjectorC（eval）評估準確率
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netP_C,
                                  class_centroids, target_global_mean, source_global_mean, args,
                                  flag=False)

            # 【監控】每個 Epoch 統計一次 Pseudo Label 質量（使用 Centroids）
            if current_epoch % 1 == 0:  # 每個 epoch 都記錄
                _, _ = obtain_label_clip(
                    dset_loaders['test'], netF, netP_C_ema, class_centroids, args, current_epoch
                )
                # 返回值可以忽略（只用於日誌輸出）

            # 計算當前 epoch 的動態參數（用於日誌顯示）
            current_ent_display = args.ent_par  # 熵權重保持不變
            if current_epoch + 1 < args.phase1_epoch:
                current_cls_display = args.cls_par
                current_con_display = args.con_par
                current_prop_display = 0.0
                teacher_logit_scale = args.centroid_logitScale  # 11
                threshold_mode = f"Fixed({args.conf_thres})"
            else:
                progress = (current_epoch - args.start_epoch) / (args.max_epoch - args.start_epoch)
                # ✨ 日誌顯示也用相同的進度綁定指數衰減公式
                current_con_display = args.con_par * math.exp(-1.0 * progress)
                current_prop_display = args.prop_par * progress
                current_cls_display = 1.0 - current_prop_display
                teacher_logit_scale = args.centroid_logitScale
                threshold_mode = "Dynamic(Avg)"  # 參考 C-SFDA
            log_str = (
                f'Epoch: {current_epoch+1}/{args.max_epoch}  |  Accuracy = {acc_s_te:.2f}%\n'
                f'  Avg Loss: {avg_total:.4f}  '
                f'CE: {avg_ce:.4f}  '
                f'Prop: {avg_prop:.4f}  '
                f'Ent: {avg_entropy:.4f}  '
                f'Con: {avg_contrast:.4f}\n'
                f'  Dynamic Params: cls_par = {current_cls_display:.4f}, ent_par = {current_ent_display:.4f}, con_par = {current_con_display:.4f}, prop_par = {current_prop_display:.4f}\n'
                f'  Teacher Strategy: Aug=Original, logitScale={teacher_logit_scale}, Threshold={conf_threshold:.4f}'
            )
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()

            # 儲存 Current Model（每個 Epoch 覆蓋）
            torch.save(netP_C.state_dict(),
                       osp.join(args.output_dir, "target_ProjectorC_current.pt"))
            torch.save(netP_H.state_dict(),
                       osp.join(args.output_dir, "target_ProjectorH_current.pt"))
            torch.save(netP_C_ema.state_dict(),
                       osp.join(args.output_dir, "target_ProjectorC_ema_current.pt"))

            # 儲存 Best Model（準確率創歷史新高時）
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                torch.save(netP_C.state_dict(),
                           osp.join(args.output_dir, "target_ProjectorC_best.pt"))
                torch.save(netP_H.state_dict(),
                           osp.join(args.output_dir, "target_ProjectorH_best.pt"))
                torch.save(netP_C_ema.state_dict(),
                           osp.join(args.output_dir, "target_ProjectorC_ema_best.pt"))
                print(f'✨ New best accuracy: {acc_s_te:.2f}%，已儲存 Best Model')

            # 重置 Epoch 追蹤變數
            current_epoch       += 1
            epoch_start_iter     = iter_num + 1
            epoch_total_loss    = 0.0
            epoch_ce_loss       = 0.0
            epoch_prop_loss     = 0.0
            epoch_entropy_loss  = 0.0
            epoch_contrast_loss = 0.0
            epoch_batches       = 0

            if current_epoch < args.max_epoch:
                pbar = tqdm(total=batches_per_epoch,
                            desc=f'Epoch {current_epoch+1}/{args.max_epoch}',
                            leave=False)

        iter_num += 1

    # 關閉最後一個進度條
    if 'pbar' in locals():
        pbar.close()

    return netP_C, netP_H, netP_C_ema


# ============================================================
# 主程式入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP Projector EMA SFDA')

    # 基本設定
    parser.add_argument('--gpu_id',   type=str, nargs='?', default='0',  help='device id to run')
    parser.add_argument('--s',        type=int,  default=0,   help='source domain index')
    parser.add_argument('--t',        type=int,  default=1,   help='target domain index')
    parser.add_argument('--max_epoch',type=int,  default=5,   help='最大訓練 epoch 數')
    parser.add_argument('--interval', type=int,  default=20,  help='每隔幾個 interval 更新一次 pseudo-label')
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--worker',   type=int,  default=4)
    parser.add_argument('--seed',     type=int,  default=2022)

    # 資料集
    parser.add_argument('--dset', type=str, default='M58',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'M58'])

    # CLIP 設定
    parser.add_argument('--net',           type=str,   default='RN101',
                        help='CLIP backbone（RN50、RN101、ViT-B/32 等）')
    parser.add_argument('--centroid_logitScale',type=float, default=8.1,
                        help='Centroid similarity 的 logit scale（用於 Pseudo Label 生成，建議與 plot 一致）')
    parser.add_argument('--classnames_path',type=str,
                        default='/mnt/backups/andycw/M58/classname37.txt',
                        help='類別名稱檔案路徑（每行一個類別名稱）')
    parser.add_argument('--centroid_path',type=str,
                        default='/mnt/backups/andycw/UDA-AI/class_centroids_37.pth',
                        help='Class Centroids 檔案路徑（從 Source 提取的類別中心特徵）')

    # Projector 路徑（直接指定可覆蓋 output_dir_src/source_P.pt）
    parser.add_argument('--projector_path', type=str,
                        default='/mnt/backups/andycw/CLIP/output/m58/PureCLIP_Source_Model/'
                                'rn101_projector_originCLIP_ep50_LR0.01_randomAug_37classes/'
                                'source_projector.pt',
                        help='直接指定 Projector 預訓練權重路徑')

    # EMA
    parser.add_argument('--ema_m', type=float, default=0.99,
                        help='Teacher EMA momentum（越大 Teacher 越穩定）')

    # Loss 權重
    parser.add_argument('--gent',     type=bool,  default=True,
                        help='是否使用 diversity loss（IM Loss 的 gentropy 項）')
    parser.add_argument('--ent',      type=bool,  default=True,
                        help='是否使用 Information Loss（熵最小化）')
    parser.add_argument('--cls_par',  type=float, default=0.3,
                        help='[A] CE Loss 權重')
    parser.add_argument('--prop_par', type=float, default=0.5,
                        help='[B] Propagation Loss 權重')
    parser.add_argument('--ent_par',  type=float, default=1.0,
                        help='[C] Information Loss 權重')

    # Contrastive Loss 溫度
    parser.add_argument('--tt',       type=float, default=0.05,
                        help='[D] Contrastive Loss temperature（SimCLR）')

    # Pseudo-label 信心門檻
    parser.add_argument('--conf_thres',type=float, default=0.5,
                        help='Pseudo-label 信心門檻（> conf_thres 為 reliable）')

    # 輸出路徑
    parser.add_argument('--output',     type=str, default='ckps/target/')
    parser.add_argument('--output_src', type=str, default='ckps/source/')
    parser.add_argument('--da',         type=str, default='uda',
                        choices=['uda', 'pda'])

    # 其他
    parser.add_argument('--epsilon',            type=float, default=1e-5,
                        help='Entropy 計算的數值穩定 epsilon')
    parser.add_argument('--lr',                 type=float, default=1e-3,
                        help='初始學習率')
    parser.add_argument('--issave',             type=bool,  default=True)

    # Resume 相關參數
    parser.add_argument('--resume_dir',         type=str,   default=None,
                        help='Resume 訓練的模型目錄路徑（包含 _current.pt 檔案）')
    parser.add_argument('--start_epoch',        type=int,   default=0,
                        help='Resume 時的起始 epoch（通常從 log 中查看）')
    
    # 動態參數調整
    parser.add_argument('--con_par',            type=float, default=None,
                        help='Contrastive Loss 的權重（如果指定則覆蓋原始計算）')
    parser.add_argument('--phase1_epoch',    type=float, default=None,
                        help='Phase 1 的 epoch 數（用於動態調整參數）')

    args = parser.parse_args()

    # ===== 資料集對應設定 =====
    if args.dset == 'M58':
        names          = ['CAD_ratioFilter', 'Real_all_nobg']
        args.class_num = 37
    elif args.dset == 'office-home':
        names          = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    elif args.dset == 'office':
        names          = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names          = ['train', 'validation']
        args.class_num = 12
    elif args.dset == 'office-caltech':
        names          = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    # ===== 環境與隨機種子 =====
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ===== 資料路徑 =====
    folder = 'data/'
    if args.dset == 'M58':
        args.s_dset_path    = folder + args.dset + '/' + names[args.s] + '_37_list.txt'
        args.t_dset_path    = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
    else:
        args.s_dset_path    = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path    = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    # ===== 輸出目錄 =====
    args.output_dir_src = osp.join(
        args.output_src, args.da, args.dset, names[args.s][0].upper()
    )
    print("output_dir_src:", args.output_dir_src)

    timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M")
    args.output_dir = osp.join(
        args.output, args.da, args.dset,
        names[args.s][0].upper() + names[args.t][0].upper() + timestamp
    )
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    os.makedirs(args.output_dir, exist_ok=True)

    # ===== 日誌檔案 =====
    args.savename = (
        f'conf_{args.conf_thres}_cls_{args.cls_par}_prop_{args.prop_par}_ema_{args.ema_m}'
    )
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    train_target(args)
