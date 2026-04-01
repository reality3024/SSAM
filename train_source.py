#!/usr/bin/env python3
"""
Pure CLIP Source Model - Standalone Trainer
完全獨立於dassl的訓練程式
只訓練MLP Projector，其他組件凍結
"""

import os
import argparse
import random
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, _LRScheduler
from tqdm import tqdm

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dataset_controller import create_dataset, create_train_val_test_datasets, resolve_dataset_dirs

_tokenizer = _Tokenizer()


# ==================== dassl 的 LR Scheduler 實現 ====================
AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


class _BaseWarmupScheduler(_LRScheduler):
    """基礎 Warmup Scheduler"""

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    """常數 Warmup Scheduler"""

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):
    """線性 Warmup Scheduler"""

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def build_lr_scheduler(optimizer, optim_cfg):
    """構建學習率調度器 (來自 dassl)

    Args:
        optimizer (Optimizer): 優化器
        optim_cfg (dict): 優化配置
    """
    lr_scheduler = optim_cfg.get('LR_SCHEDULER', 'cosine')
    stepsize = optim_cfg.get('STEPSIZE', 10)
    gamma = optim_cfg.get('GAMMA', 0.1)
    max_epoch = optim_cfg.get('MAX_EPOCH', 50)
    warmup_epoch = optim_cfg.get('WARMUP_EPOCH', 1)
    warmup_type = optim_cfg.get('WARMUP_TYPE', 'linear')
    warmup_min_lr = optim_cfg.get('WARMUP_MIN_LR', 1e-6)
    warmup_cons_lr = optim_cfg.get('WARMUP_CONS_LR', 1e-6)
    warmup_recount = optim_cfg.get('WARMUP_RECOUNT', True)

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    if warmup_epoch > 0:
        if not warmup_recount:
            scheduler.last_epoch = warmup_epoch

        if warmup_type == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, warmup_epoch,
                warmup_cons_lr
            )

        elif warmup_type == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, warmup_epoch,
                warmup_min_lr
            )

        else:
            raise ValueError(f"Unknown warmup type: {warmup_type}")

    return scheduler


# ==================== 模型定義 ====================
class MLPProjector(nn.Module):
    """MLP Projector: 唯一可訓練的組件"""
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


class PureCLIP_SourceModel(nn.Module):
    """
    Pure CLIP Source Model
    - 使用原始 CLIP image encoder (frozen)
    - 使用原始 CLIP text encoder (frozen)
    - Text embeddings 用簡單模板動態生成
    - 只訓練 MLP Projector
    """

    def __init__(self, classnames, clip_model, backbone_name='RN50'):
        super().__init__()
        print("🎯 初始化 Pure CLIP Source Model (Projector Only)")

        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames

        # 確定特徵維度
        if backbone_name == 'RN50':
            self.clip_dim = 1024
        else:  # RN101, ViT
            self.clip_dim = 512

        print(f'  - Backbone: {backbone_name}')
        print(f'  - Feature dim: {self.clip_dim}')
        print(f'  - Num classes: {len(classnames)}')

        # ===== 建立 MLP Projector（唯一可訓練的組件）=====
        print(f'\n✅ 建立 Trainable MLP Projector: {self.clip_dim} → 1024 → {self.clip_dim}')
        self.projector = MLPProjector(in_dim=self.clip_dim, out_dim=self.clip_dim)

        # ===== 動態生成 Text Features（使用簡單模板）=====
        print(f'\n🔒 生成 Text Features（使用簡單模板）')
        text_features = self._generate_text_features(classnames, clip_model)
        self.register_buffer('text_features', text_features)

        print(f'  ✓ Text features shape: {text_features.shape}')
        print(f'  ✓ 已正規化: {(text_features.norm(dim=-1) - 1.0).abs().max().item() < 0.01}')

    def _generate_text_features(self, classnames, clip_model):
        """
        使用簡單模板生成 text features
        Template: "A photo of a [CLASS_NAME]"
        """
        print('  - Template: "A photo of a [CLASS_NAME]"')

        # 生成 prompts
        prompts = [f"A photo of a {classname}" for classname in classnames]

        # Tokenize
        prompts_tokens = clip.tokenize(prompts)

        # 確定 clip_model 的設備
        device = next(clip_model.parameters()).device

        # 編碼為 text features
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts_tokens.to(device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(self, image):
        """
        Forward pass

        Args:
            image: [B, 3, 224, 224]

        Returns:
            logits_clip: CLIP similarity logits [B, num_classes]
        """
        # ===== Path: Image → CLIP Encoder → Projector =====
        feat_clip = self.image_encoder(image.type(self.dtype))  # [B, 512]
        feat_projected = self.projector(feat_clip)  # [B, 512]

        # ===== 計算 CLIP Similarity =====
        # Normalize
        feat_projected_norm = feat_projected / feat_projected.norm(dim=-1, keepdim=True)
        text_features_norm = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_clip = logit_scale * feat_projected_norm @ text_features_norm.t()

        return logits_clip


# ==================== 訓練類 ====================
class Trainer:
    """獨立訓練器"""

    def __init__(self, args):
        self.args = args
        # 初始化輸出目錄
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.output_dir / "log.txt"
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write("epoch,train_loss,train_acc,val_acc\n")

        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
        print(f"使用設備: {self.device}")

        # 初始化結果目錄
        self.result_dir = Path("/mnt/backups/andycw/UDA-AI/pretrained_result")
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # 設定隨機種子
        self._set_random_seed(args.seed)

        # 載入CLIP模型
        print(f"\n載入 CLIP (backbone: {args.backbone})")
        self.clip_model = self._load_clip(args.backbone)

        # 載入資料集
        print(f"\n載入資料集")
        self.source_data_dir, _ = resolve_dataset_dirs(
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = create_train_val_test_datasets(
            data_dir=args.data_dir,
            classname_file=args.classname_file,
            train_transform=self._get_transform(is_train=True),
            eval_transform=self._get_transform(is_train=False),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            dataset_name=args.dataset_name,
        )

        # 建立資料載入器
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        # 建立模型
        print(f"\n建立模型")
        self.model = PureCLIP_SourceModel(
            self.train_dataset.classnames, self.clip_model,
            backbone_name=args.backbone
        ).to(self.device)

        # 凍結CLIP編碼器
        self._freeze_clip_encoders()

        # 建立優化器和調度器
        self.optimizer = SGD(
            self.model.projector.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True
        )

        # 使用 dassl 的 build_lr_scheduler（包含 warmup）
        optim_cfg = {
            'LR_SCHEDULER': 'cosine',        # Cosine Annealing
            'MAX_EPOCH': args.epochs,
            'WARMUP_EPOCH': 1,              # 1個 epoch 的 warmup
            'WARMUP_TYPE': 'constant',      # 常數 warmup（不是線性）
            'WARMUP_CONS_LR': 1e-5,         # Warmup 常數 LR
            'WARMUP_RECOUNT': True,
        }
        self.scheduler = build_lr_scheduler(self.optimizer, optim_cfg)

        # 使用多GPU
        if torch.cuda.device_count() > 1:
            print(f"檢測到{torch.cuda.device_count()}個GPU，使用DataParallel")
            self.model = nn.DataParallel(self.model)

        # 追蹤指標
        self.best_acc = -1.0
        self.best_epoch = -1

    def _log_epoch_result(self, epoch, train_loss, train_acc, val_acc):
        """僅記錄每個 epoch 的訓練結果"""
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc*100:.4f},{val_acc*100:.4f}\n")

    def _set_random_seed(self, seed):
        """設定隨機種子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"隨機種子已設定為: {seed}")

    def _load_clip(self, backbone_name):
        """載入CLIP模型"""
        model, _ = clip.load(
            backbone_name, device=str(self.device),
            download_root=os.path.expanduser("~/.cache/clip")
        )
        model.float()
        return model

    def _get_transform(self, is_train=False):
        """取得資料轉換"""
        from torchvision import transforms

        if is_train:
            # 訓練時使用數據增強
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # ← 隨機裁剪
                transforms.RandomHorizontalFlip(),                    # ← 隨機翻轉
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            # 測試時使用標準處理 (resize + center crop)
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),                           # ← 中心裁剪
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def _freeze_clip_encoders(self):
        """凍結CLIP編碼器"""
        print(f'\n{"="*60}')
        print(f'參數凍結策略')
        print(f'{"="*60}')

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # 凍結Image Encoder
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        print('🔒 CLIP Image Encoder: 凍結 (Frozen)')

        # Projector自動可訓練
        for param in model.projector.parameters():
            param.requires_grad = True
        print('✅ MLP Projector: 解凍 (Trainable)')

        # 設置訓練/評估模式
        model.projector.train()
        model.image_encoder.eval()
        print('='*60 + '\n')

    def train_epoch(self, epoch):
        """訓練一個epoch"""
        self.model.train()
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.image_encoder.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            img = batch['img'].to(self.device)
            label = batch['label'].to(self.device)

            # Forward pass
            logits = self.model(img)
            loss = F.cross_entropy(logits, label)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total_samples += label.size(0)

            # 更新進度
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = 100.0 * total_correct / total_samples
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.2f}%'})

        self.scheduler.step()

        return total_loss / len(self.train_loader), total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, data_loader, split_name='Val'):
        """評估模型"""
        self.model.eval()

        total_correct = 0
        total_samples = 0

        pbar = tqdm(data_loader, desc=f"{split_name}")

        for batch in pbar:
            img = batch['img'].to(self.device)
            label = batch['label'].to(self.device)

            logits = self.model(img)
            pred = logits.argmax(dim=1)
            correct = (pred == label).sum().item()

            total_correct += correct
            total_samples += label.size(0)

            acc = 100.0 * total_correct / total_samples
            pbar.set_postfix({'acc': f'{acc:.2f}%'})

        accuracy = total_correct / total_samples
        return accuracy

    def load_model(self, model_dir):
        """載入訓練好的模型"""
        if not model_dir:
            print("❌ 錯誤: 未指定模型目錄。請提供 --model-dir 參數")
            return False

        model_path = Path(model_dir) / 'source_projector.pt'

        print(f'\n{"="*60}')
        print(f'載入 Source Model')
        print(f'目錄: {model_dir}')
        print(f'{"="*60}')

        if not model_path.exists():
            print(f'❌ 找不到模型檔案: {model_path}')
            print(f'請確認模型已訓練並保存')
            return False

        # 載入 Projector
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state_dict = torch.load(model_path, map_location='cpu')
        model.projector.load_state_dict(state_dict, strict=True)
        print(f'✓ source_projector.pt: ✅ Trainable MLP Projector 已載入')

        print(f'{"="*60}\n')
        return True

    @torch.no_grad()
    def test(self, save_csv=True):
        """測試模型並保存結果"""
        self.model.eval()

        total_correct = 0
        total_samples = 0
        detailed_results = []
        class_stats = {}

        pbar = tqdm(self.test_loader, desc="Testing")

        for batch in pbar:
            img = batch['img'].to(self.device)
            label = batch['label'].to(self.device)
            impath = batch['impath']
            classname = batch['classname']

            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct = (pred == label).sum().item()

            # 收集詳細結果
            for i in range(label.size(0)):
                true_idx = label[i].item()
                pred_idx = pred[i].item()
                confidence = probs[i, pred_idx].item()
                is_correct = (pred_idx == true_idx)

                true_class = self.train_dataset.classnames[true_idx]
                pred_class = self.train_dataset.classnames[pred_idx]

                detailed_results.append({
                    'image_path': impath[i],
                    'class_id': true_idx,
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'confidence': f"{confidence:.4f}",
                    'is_correct': is_correct
                })

                if true_idx not in class_stats:
                    class_stats[true_idx] = {'total': 0, 'correct': 0}
                class_stats[true_idx]['total'] += 1
                if is_correct:
                    class_stats[true_idx]['correct'] += 1

            total_correct += correct
            total_samples += label.size(0)
            acc = 100.0 * total_correct / total_samples
            pbar.set_postfix({'acc': f'{acc:.2f}%'})

        accuracy = total_correct / total_samples

        if save_csv:
            self._save_results_csv(detailed_results, class_stats)

        return accuracy, detailed_results, class_stats

    @torch.no_grad()
    def run_inference_on_real_data(self, data_dir, save_csv=True):
        """在真實資料集上推論（可選擇保存 CSV）"""
        print(f"\n{'='*60}")
        print("Real Data Inference")
        print(f"Data Dir: {data_dir}")
        print(f"{'='*60}")

        real_dataset = create_dataset(
            data_dir=data_dir,
            dataset_name=self.args.dataset_name,
            classname_file=self.args.classname_file,
            split='all',
            transform=self._get_transform(is_train=False),
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            test_ratio=self.args.test_ratio,
            split_seed=self.args.split_seed,
        )
        real_loader = DataLoader(
            real_dataset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available() and self.args.use_cuda
        )

        self.model.eval()
        total_correct = 0
        total_samples = 0
        detailed_results = []
        class_stats = {}

        pbar = tqdm(real_loader, desc='Real-All')
        for batch in pbar:
            img = batch['img'].to(self.device)
            label = batch['label'].to(self.device)
            impath = batch['impath']

            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct = (pred == label).sum().item()

            for i in range(label.size(0)):
                true_idx = label[i].item()
                pred_idx = pred[i].item()
                confidence = probs[i, pred_idx].item()
                is_correct = (pred_idx == true_idx)

                true_class = real_dataset.classnames[true_idx]
                pred_class = real_dataset.classnames[pred_idx]
                detailed_results.append({
                    'image_path': impath[i],
                    'class_id': true_idx,
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'confidence': f"{confidence:.4f}",
                    'is_correct': is_correct
                })

                if true_idx not in class_stats:
                    class_stats[true_idx] = {'total': 0, 'correct': 0}
                class_stats[true_idx]['total'] += 1
                if is_correct:
                    class_stats[true_idx]['correct'] += 1

            total_correct += correct
            total_samples += label.size(0)
            acc = 100.0 * total_correct / total_samples
            pbar.set_postfix({'acc': f'{acc:.2f}%'})

        real_acc = total_correct / total_samples

        if save_csv:
            self._save_results_csv(
                detailed_results,
                class_stats,
                dataset_name='M58_PureCLIP_RealAll_Inference',
                classnames=real_dataset.classnames
            )

        print(f"Real-All 準確率: {real_acc*100:.2f}%")
        print(f"樣本數量: {len(real_dataset)}")
        print(f"{'='*60}\n")
        return real_acc

    @torch.no_grad()
    def extract_and_save_class_centroids(self, data_dir):
        """提取 source image 的 class centroid 並保存在模型目錄"""
        print(f"\n{'='*60}")
        print("提取 Source Class Centroid")
        print(f"Source Data Dir: {data_dir}")
        print(f"{'='*60}")

        source_dataset = create_dataset(
            data_dir=data_dir,
            dataset_name=self.args.dataset_name,
            classname_file=self.args.classname_file,
            split='all',
            transform=self._get_transform(is_train=False),
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            test_ratio=self.args.test_ratio,
            split_seed=self.args.split_seed,
        )
        source_loader = DataLoader(
            source_dataset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available() and self.args.use_cuda
        )

        num_classes = len(source_dataset.classnames)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.eval()

        class_sums = torch.zeros(num_classes, model.clip_dim, device=self.device)
        class_counts = torch.zeros(num_classes, dtype=torch.long, device=self.device)

        pbar = tqdm(source_loader, desc='Extract Centroids')
        for batch in pbar:
            img = batch['img'].to(self.device)
            label = batch['label'].to(self.device)

            clip_features = model.image_encoder(img.type(model.dtype))
            projected_features = model.projector(clip_features)
            projected_features = F.normalize(projected_features, dim=-1)

            for i in range(label.size(0)):
                class_id = label[i].item()
                class_sums[class_id] += projected_features[i]
                class_counts[class_id] += 1

        class_centroids = torch.zeros(num_classes, model.clip_dim, device=self.device)
        for class_id in range(num_classes):
            if class_counts[class_id] > 0:
                class_centroids[class_id] = class_sums[class_id] / class_counts[class_id]
                class_centroids[class_id] = F.normalize(
                    class_centroids[class_id].unsqueeze(0), dim=-1
                ).squeeze(0)

        centroid_path = self.output_dir / 'source_class_centroids.pth'
        centroid_data = {
            'centroids': class_centroids.cpu(),
            'class_counts': class_counts.cpu(),
            'classnames': source_dataset.classnames,
            'feature_dim': model.clip_dim,
            'num_classes': num_classes,
        }
        torch.save(centroid_data, centroid_path)

        print(f"Centroid 已保存: {centroid_path}")
        print(f"總樣本數: {class_counts.sum().item()}")
        print(f"有效類別數: {(class_counts > 0).sum().item()}/{num_classes}")
        print(f"{'='*60}\n")

    def _save_results_csv(self, detailed_results, class_stats,
                          dataset_name="M58_PureCLIP_Source_Model", classnames=None):
        """保存結果到CSV檔案"""
        try:
            import pandas as pd
        except ImportError:
            print("警告: 無法導入pandas，跳過CSV保存")
            return

        if classnames is None:
            classnames = self.train_dataset.classnames

        # 保存詳細預測結果
        output_df = pd.DataFrame(detailed_results)
        output_df = output_df.sort_values('class_id').reset_index(drop=True)
        output_filename = self.result_dir / f"{dataset_name}_output.csv"
        output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"詳細預測結果已保存到: {output_filename}")

        # 保存類別統計結果
        result_data = []
        total_images_all = 0
        total_correct_all = 0

        for class_id in range(len(classnames)):
            class_name = classnames[class_id]

            if class_id in class_stats:
                stats = class_stats[class_id]
                total_img = stats['total']
                correct = stats['correct']
                acc = (correct / total_img * 100) if total_img > 0 else 0.0

                total_images_all += total_img
                total_correct_all += correct
            else:
                total_img = 0
                correct = 0
                acc = 0.0

            result_data.append({
                'class_id': class_id,
                'class_name': class_name,
                'total_imgNum': total_img,
                'top1_correct': correct,
                'top1_acc': f"{acc:.2f}%"
            })

        # 添加平均行
        avg_acc = (total_correct_all / total_images_all * 100) if total_images_all > 0 else 0.0
        result_data.append({
            'class_id': '',
            'class_name': 'AVERAGE',
            'total_imgNum': total_images_all,
            'top1_correct': total_correct_all,
            'top1_acc': f"{avg_acc:.2f}%"
        })

        # 保存
        result_df = pd.DataFrame(result_data)
        result_filename = self.result_dir / f"{dataset_name}_result.csv"
        result_df.to_csv(result_filename, index=False, encoding='utf-8-sig')
        print(f"類別統計結果已保存到: {result_filename}")

    def save_model(self):
        """保存模型"""
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'儲存 Source Model')
        print(f'{"="*60}')

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(model.projector.state_dict(),
                  output_dir / "source_projector.pt")
        print(f'✓ source_projector.pt: ✅ Trainable MLP Projector 已儲存')
        print(f'模型已儲存到: {output_dir}')
        print(f'{"="*60}\n')

    def train(self):
        """完整訓練循環"""
        print(f'\n{"="*60}')
        print(f'開始訓練')
        print(f'{"="*60}')
        print(f'Epochs: {self.args.epochs}')
        print(f'Learning Rate: {self.args.lr}')
        print(f'Batch Size: {self.args.batch_size}')
        print(f'{"="*60}\n')

        for epoch in range(self.args.epochs):
            # 訓練
            train_loss, train_acc = self.train_epoch(epoch)

            # 驗證
            val_acc = self.evaluate(self.val_loader, split_name="Val")

            print(f"Epoch {epoch+1}/{self.args.epochs} - "
                  f"Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
                  f"Val Acc: {val_acc*100:.2f}%\n")
            self._log_epoch_result(epoch + 1, train_loss, train_acc, val_acc)

            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                self.save_model()
                print(f"✅ 新的最佳驗證準確率: {val_acc*100:.2f}% (Epoch {epoch+1})\n")

        # 載入最佳模型後，再做最終評估與後處理
        self.load_model(str(self.output_dir))

        # 測試
        print(f'\n{"="*60}')
        print(f'最終測試')
        print(f'{"="*60}')
        test_acc, _, _ = self.test(save_csv=False)
        print(f"測試準確率: {test_acc*100:.2f}%\n")

        # 訓練後在真實資料集推論並輸出 CSV
        self.run_inference_on_real_data(self.args.inference_data_dir, save_csv=True)

        # 訓練後提取 source class centroid 並保存到模型目錄
        self.extract_and_save_class_centroids(self.source_data_dir)


def main():
    parser = argparse.ArgumentParser(description='Pure CLIP Source Model - Standalone Trainer')
    parser.add_argument('--data-dir', type=str, required=True, help='資料集根目錄')
    parser.add_argument('--classname-file', type=str, default='',
                        help='類別名稱檔案 (可選，未提供時從子資料夾名稱自動推斷)')
    parser.add_argument('--output-dir', type=str, required=True, help='輸出目錄')
    parser.add_argument('--backbone', type=str, default='RN101', help='CLIP backbone (RN50, RN101, ViT-B/32)')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--seed', type=int, default=1, help='隨機種子')
    parser.add_argument('--num-workers', type=int, default=4, help='資料載入線程數')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='train split 比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='val split 比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='test split 比例')
    parser.add_argument('--split-seed', type=int, default=42, help='資料切分隨機種子')
    parser.add_argument('--dataset-name', type=str, default='m58',
                        choices=['m58', 'visda'],
                        help='資料集名稱: m58 或 visda')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='使用CUDA')
    parser.add_argument('--inference-data-dir', type=str,
                        default='/mnt/backups/andycw/M58/Real_all_nobg',
                        help='訓練後推論資料集路徑')
    parser.add_argument('--eval-only', action='store_true', help='僅評估模式 (推論)')
    parser.add_argument('--model-dir', type=str, default='', help='載入模型的目錄 (用於推論)')

    args = parser.parse_args()

    # 檢查訓練還是推論模式
    if args.eval_only:
        print(f'\n{"="*60}')
        print('🔍 推論模式 (Inference Only)')
        print(f'{"="*60}\n')

        trainer = Trainer(args)
        trainer.load_model(args.model_dir)
        test_acc, _, _ = trainer.test(save_csv=False)
        print(f"\n推論準確率: {test_acc*100:.2f}%\n")
    else:
        print(f'\n{"="*60}')
        print('🚀 訓練模式 (Training)')
        print(f'{"="*60}\n')

        trainer = Trainer(args)
        trainer.train()


if __name__ == '__main__':
    main()
