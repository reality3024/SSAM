import wandb
wandb.login()

import argparse
import os, sys
import datetime
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from randaugment import RandAugmentMC
from gaussian_blur import GaussianBlur
import loss
import clip


class MLPProjector(nn.Module):
    """MLP Projector - 與 clip_source_model_projectOnly.py 完全相同"""
    def __init__(self, in_dim=512, out_dim=512, hidden_dim=1024):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     # nn.BatchNorm1d(hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        return x
    
class Augmentation(object):
    """
    返回三個版本的圖片：
    1. weak augmentation：用於一般訓練
    2. strong augmentation：用於對比學習
    3. original：只經過基本 normalize，用於 text alignment loss
    """
    def __init__(self, resize_size=256, crop_size=224):
        self.weak = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip()
        ])
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.strong = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 原始圖片只做 resize + center crop + normalize（不做 random augmentation）
        self.original = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        original = self.original(x)
        return self.normalize(weak), self.normalize(strong), original


def load_clip_to_cpu(backbone):
    model, _ = clip.load(backbone, device="cpu", download_root=os.path.expanduser("~/.cache/clip"))
    return model

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    # Determine root path based on dataset
    root_path = f'data/{args.dset}/'

    dsets["target"] = ImageList_idx(txt_tar, transform=Augmentation(), root=root_path)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), root=root_path)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netP, text_features, flag=False):
    """使用 CLIP + Projector + Text Similarity 計算準確率"""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            # CLIP visual encoder → 512維
            feat_512 = netF(inputs.type(netF.conv1.weight.dtype))
            # Projector → 512維
            feat_projected = netP(feat_512)
            
            # 計算與 text features 的相似度
            feat_norm = feat_projected / feat_projected.norm(dim=-1, keepdim=True)
            text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            outputs = feat_norm @ text_norm.t()  # [B, num_classes]
            
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def calculate_gaussian_kl_divergence(m1, m2, v1, v2):
    return torch.log(v2 / v1) * 0.5 + torch.div(torch.add(v1, torch.square(m1 - m2)), 2 * v2) - 0.5


# ===== 以下函數已不需要（不使用 Bottleneck/Classifier）=====
# def load_feature_extractor_weights(...)
# def load_bottleneck_weights(...)
# def load_classifier_weights(...)
# def ShotHook(...)


def _deprecated_load_feature_extractor_weights(model, checkpoint_path):
    """
    載入 Feature Extractor (ResNet) 權重
    自動跳過 CLIP 特有的層（conv2, bn2, conv3, bn3, attnpool 等）
    
    Args:
        model: 標準 ResNet Feature Extractor
        checkpoint_path: CLIP Source Model 的 F 層權重路徑
    """
    checkpoint = torch.load(checkpoint_path)
    model_state = model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    shape_mismatch_keys = []
    
    for key, value in checkpoint.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.append(key)
            else:
                shape_mismatch_keys.append(
                    f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}"
                )
        else:
            skipped_keys.append(key)
    
    model.load_state_dict(model_state)
    
    print(f"\n{'='*60}")
    print(f"[Feature Extractor] 載入權重: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"✓ 成功載入: {len(loaded_keys)} 層")
    print(f"⊘ 跳過 CLIP 特有層: {len(skipped_keys)} 層")
    if len(skipped_keys) <= 10:
        for key in skipped_keys:
            print(f"  - {key}")
    else:
        for key in skipped_keys[:10]:
            print(f"  - {key}")
        print(f"  ... 還有 {len(skipped_keys) - 10} 層")
    
    if shape_mismatch_keys:
        print(f"⚠ 形狀不匹配: {len(shape_mismatch_keys)} 層")
        for msg in shape_mismatch_keys:
            print(f"  - {msg}")
    print(f"{'='*60}\n")


def load_bottleneck_weights(model, checkpoint_path):
    """
    載入 Bottleneck 權重
    處理 CLIP Source Model (Sequential: 0, 1) → Standard Model (bottleneck, bn) 的映射
    
    Args:
        model: 標準 Bottleneck 模型
        checkpoint_path: CLIP Source Model 的 B 層權重路徑
    """
    checkpoint = torch.load(checkpoint_path)
    model_state = model.state_dict()
    
    # 名稱映射表
    mapping = {
        '0.weight': 'bottleneck.weight',
        '0.bias': 'bottleneck.bias',
        '1.weight': 'bn.weight',
        '1.bias': 'bn.bias',
        '1.running_mean': 'bn.running_mean',
        '1.running_var': 'bn.running_var',
        '1.num_batches_tracked': 'bn.num_batches_tracked',
    }
    
    loaded_keys = []
    mapped_info = []
    
    for key, value in checkpoint.items():
        if key in mapping:
            target_key = mapping[key]
            if target_key in model_state and model_state[target_key].shape == value.shape:
                model_state[target_key] = value
                loaded_keys.append(target_key)
                mapped_info.append(f"{key} → {target_key}")
    
    model.load_state_dict(model_state)
    
    print(f"\n{'='*60}")
    print(f"[Bottleneck] 載入權重: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"✓ 成功載入: {len(loaded_keys)} 層")
    print(f"🔄 名稱映射:")
    for info in mapped_info:
        print(f"  - {info}")
    print(f"{'='*60}\n")


def load_classifier_weights(model, checkpoint_path):
    """
    載入 Classifier 權重
    處理 CLIP Source Model (Linear: weight, bias) → Standard Model (WeightNorm: weight_g, weight_v, bias) 的轉換
    
    Args:
        model: 使用 Weight Normalization 的 Classifier
        checkpoint_path: CLIP Source Model 的 C 層權重路徑
    """
    checkpoint = torch.load(checkpoint_path)
    model_state = model.state_dict()
    
    # 檢查是否需要轉換（從 Linear 到 WeightNorm）
    if 'weight' in checkpoint and 'fc.weight_g' in model_state:
        # CLIP Source Model 使用普通 Linear，需要轉換為 Weight Normalization
        weight = checkpoint['weight']
        bias = checkpoint['bias']
        
        # Weight Normalization: weight = weight_g * (weight_v / ||weight_v||)
        # 反向計算：
        # weight_v = weight (方向)
        # weight_g = ||weight|| (每行的 L2 norm，大小)
        weight_norm = torch.norm(weight, p=2, dim=1, keepdim=True)  # [num_classes, 1]
        
        model_state['fc.weight_v'] = weight
        model_state['fc.weight_g'] = weight_norm
        model_state['fc.bias'] = bias
        
        model.load_state_dict(model_state)
        
        print(f"\n{'='*60}")
        print(f"[Classifier] 載入權重: {checkpoint_path}")
        print(f"{'='*60}")
        print(f"✓ 成功載入並轉換: 3 層")
        print(f"🔄 轉換 Linear → Weight Normalization:")
        print(f"  - weight ({weight.shape}) → fc.weight_v ({weight.shape})")
        print(f"  - weight ({weight.shape}) → fc.weight_g ({weight_norm.shape})")
        print(f"  - bias ({bias.shape}) → fc.bias ({bias.shape})")
        print(f"{'='*60}\n")
    
    elif 'fc.weight_g' in checkpoint:
        # 已經是 Weight Normalization 格式，直接載入
        model.load_state_dict(checkpoint)
        print(f"\n{'='*60}")
        print(f"[Classifier] 載入權重: {checkpoint_path}")
        print(f"{'='*60}")
        print(f"✓ 直接載入 Weight Normalization 格式")
        print(f"{'='*60}\n")
    
    else:
        raise ValueError(f"Unsupported classifier format in checkpoint: {checkpoint.keys()}")


class ShotHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 or KL 散度 as a loss
    '''

    def __init__(self, module1, module2):
        self.hook = module1.register_forward_hook(self.hook_fn)
        self.mean_orignal = module2.running_mean
        self.var_orignal = module2.running_var

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        if isinstance(module, nn.BatchNorm2d):
            mean = input[0].mean([0, 2, 3])
            # Memory efficient variance calculation - process in smaller chunks
            x = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1])
            if x.size(1) > 10000:  # If too large, compute variance in chunks
                chunk_size = 5000
                var_chunks = []
                for i in range(0, x.size(1), chunk_size):
                    chunk = x[:, i:i+chunk_size]
                    var_chunks.append(chunk.var(1, unbiased=False, keepdim=True))
                var = torch.cat(var_chunks, dim=1).mean(dim=1)
            else:
                var = x.var(1, unbiased=False)

        if isinstance(module, nn.BatchNorm1d):
            mean = input[0].mean([0])
            x = input[0].permute(1, 0).contiguous().view([nch, -1])
            if x.size(1) > 10000:  # If too large, compute variance in chunks
                chunk_size = 5000
                var_chunks = []
                for i in range(0, x.size(1), chunk_size):
                    chunk = x[:, i:i+chunk_size]
                    var_chunks.append(chunk.var(1, unbiased=False, keepdim=True))
                var = torch.cat(var_chunks, dim=1).mean(dim=1)
            else:
                var = x.var(1, unbiased=False)

        klc = 0.0
        for i in range(mean.size()[0]):
            klc += calculate_gaussian_kl_divergence(self.mean_orignal[i], mean[i], self.var_orignal[i], var[i]) # Equation 9
        r_feature = klc / mean.size()[0]

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def train_target(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dset_loaders = data_load(args)
    
    # ===== 載入 CLIP Model =====
    print(f'\n{"="*60}')
    print(f'載入 CLIP Model: {args.net}')
    clip_model = load_clip_to_cpu(backbone=args.net)
    clip_model.float()
    
    # 只使用 visual encoder（不需要 wrapper）
    netF = clip_model.visual.cuda()
    netF.eval()  # 🔒 凍結 CLIP Encoder
    for param in netF.parameters():
        param.requires_grad = False
    print(f'✅ CLIP Visual Encoder: 已載入並凍結 (eval mode)')
    print(f'{"="*60}\n')
    
    # ===== 載入 Text Embeddings =====
    text_emb_path = osp.join(args.output_dir_src, 'M58_text_embeddings.pt')
    print(f'\n{"="*60}')
    print(f'載入 Text Embeddings: {text_emb_path}')
    
    if not osp.exists(text_emb_path):
        raise FileNotFoundError(
            f'找不到 text embeddings 檔案: {text_emb_path}\n'
            f'請先執行 clip_source_model.py 訓練以生成此檔案'
        )
    
    embeddings_dict = torch.load(text_emb_path, map_location='cuda')
    text_features = embeddings_dict['text_features'].cuda()  # [num_classes, 512]
    loaded_classnames = embeddings_dict['classnames']
    
    print(f'✓ Text embeddings 載入成功！')
    print(f'  - shape: {text_features.shape}')
    print(f'  - num_classes: {len(loaded_classnames)}')
    print(f'  - 已正規化: {(text_features.norm(dim=-1) - 1.0).abs().max().item() < 0.01}')
    print(f'{"="*60}\n')
    
    # 驗證類別數量一致性
    if len(loaded_classnames) != args.class_num:
        raise ValueError(
            f'Text embeddings 的類別數 ({len(loaded_classnames)}) '
            f'與 args.class_num ({args.class_num}) 不一致！'
        )
    
    # ===== 初始化並載入 Projector =====
    print(f'\n{"="*60}')
    print(f'初始化 MLP Projector')
    netP = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).cuda()
    
    # 載入預訓練的 Projector 權重
    projector_path = osp.join(args.output_dir_src, 'source_P.pt')
    if not osp.exists(projector_path):
        raise FileNotFoundError(
            f'找不到 Projector 權重: {projector_path}\n'
            f'請先執行 clip_source_model_projectOnly.py 訓練'
        )
    
    netP.load_state_dict(torch.load(projector_path))
    netP.train()  # ✅ Projector 設為訓練模式
    print(f'✅ Projector 已載入: {projector_path}')
    print(f'✅ Projector: train() 模式（可訓練）')
    print(f'{"="*60}\n')
    
    # ===== 設定優化器（只優化 Projector）=====
    param_group = []
    for k, v in netP.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)
    
    print(f'\n{"="*60}')
    print(f'優化器設定:')
    print(f'  - 只優化 Projector')
    print(f'  - Learning Rate: {args.lr}')
    print(f'  - Momentum: 0.9')
    print(f'  - Weight Decay: 1e-3')
    print(f'{"="*60}\n')

    max_iter = args.max_epoch * len(dset_loaders["target"])
    iter_num = 0
    
    # Initialize wandb
    wandb.init(project="SFDA", 
               name=f"ATSSL_{args.dset}_{args.s}_to_{args.t}_projector",
               config=vars(args))
    
    print(f"Batch size: {args.batch_size}, Accumulation steps: {args.accumulation_steps}")

    # Initialize best accuracy for saving
    acc_init = 0

    # Training loop variables for epoch tracking
    batches_per_epoch = len(dset_loaders["target"])
    current_epoch = 0
    epoch_start_iter = 0
    epoch_total_loss = 0.0
    epoch_entropy_loss = 0.0
    epoch_contrast_loss = 0.0
    epoch_text_loss = 0.0
    epoch_batches = 0
    
    print(f"\n{'='*60}")
    print(f"開始訓練: Epoch {current_epoch+1}/{args.max_epoch}")
    print(f"Loss 設定:")
    print(f"  - im_loss (entropy): weight = {args.ent_par}")
    print(f"  - text_loss: weight = {args.text_par}, threshold = 0.95")
    print(f"  - contrast_loss: 所有樣本都計算（無 mask）")
    print(f"{'='*60}\n")
    
    # Create initial progress bar
    pbar = tqdm(total=batches_per_epoch, desc=f'Epoch {current_epoch+1}/{args.max_epoch}', leave=False)

    while iter_num < max_iter:
        try:
            imgs, true_labels, tar_idx, path = next(iter_test)
            inputs_test, inputs_s, inputs_original = imgs
        except:
            iter_test = iter(dset_loaders["target"])
            imgs, true_labels, tar_idx, path = next(iter_test)
            inputs_test, inputs_s, inputs_original = imgs
            # imgs: (weak, strong, original) augmentations
            # true_labels: ground truth labels [B]
            # tar_idx: sample indices [B]
            # inputs_test: weakly augmented
            # inputs_s: strongly augmented
            # inputs_original: original image (只經過基本 normalize，用於 text loss)

        if inputs_test.size(0) == 1:
            continue

        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Move to GPU
        inputs_test = inputs_test.cuda()
        inputs_s = inputs_s.cuda()
        inputs_original = inputs_original.cuda()
        true_labels = true_labels.cuda()
        
        # Accumulate gradients over micro batches
        total_loss = 0
        total_entropy_loss = 0
        total_contrast_loss = 0
        total_text_loss = 0
        current_batch_size = inputs_test.size(0)
        
        for micro_step in range(args.accumulation_steps):
            start_idx = (micro_step * current_batch_size) // args.accumulation_steps
            end_idx = ((micro_step + 1) * current_batch_size) // args.accumulation_steps
            
            if start_idx >= end_idx:
                continue
                
            # Get micro batch
            micro_inputs_test = inputs_test[start_idx:end_idx]
            micro_inputs_s = inputs_s[start_idx:end_idx]
            micro_inputs_original = inputs_original[start_idx:end_idx]
            micro_true_labels = true_labels[start_idx:end_idx]
            micro_paths = path[start_idx:end_idx]
            
            # Clear cache periodically
            if micro_step == 0 and iter_num % 100 == 0:
                torch.cuda.empty_cache()
            
            # ===== 特徵提取：CLIP Encoder + Projector =====
            # Weak augmentation - 用於 entropy loss
            with torch.no_grad():
                feat_512_weak = netF(micro_inputs_test.type(netF.conv1.weight.dtype))  # [B, 512]
            feat_projected_weak = netP(feat_512_weak)  # [B, 512]
            
            # Strong augmentation - 用於 contrastive loss
            with torch.no_grad():
                feat_512_strong = netF(micro_inputs_s.type(netF.conv1.weight.dtype))  # [B, 512]
            feat_projected_strong = netP(feat_512_strong)  # [B, 512]
            
            # Original (no augmentation) - 用於 text loss
            with torch.no_grad():
                feat_512_original = netF(micro_inputs_original.type(netF.conv1.weight.dtype))  # [B, 512]
            feat_projected_original = netP(feat_512_original)  # [B, 512]

            # Calculate losses for micro batch
            losses = torch.tensor(0.0).cuda()
            
            # ===== Loss 1: Entropy Loss (im_loss) =====
            # 使用 text similarity 的 softmax 計算 entropy
            im_loss = torch.tensor(0.0).cuda()
            if args.ent:
                # 正規化特徵
                feat_weak_norm = feat_projected_weak / feat_projected_weak.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 計算與 text features 的相似度作為 logits
                logits_for_entropy = feat_weak_norm @ text_features_norm.t()  # [B, K]
                softmax_out = nn.Softmax(dim=1)(logits_for_entropy)
                
                # Entropy minimization
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                
                # Diversity loss (可選)
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
                
                im_loss = entropy_loss * args.ent_par
                losses += im_loss
                
            # ===== Loss 2: Text Alignment Loss (threshold=0.95 嚴格) =====
            text_loss = torch.tensor(0.0).cuda()
            if args.text_par > 0:
                # 正規化特徵
                feat_original_norm = feat_projected_original / feat_projected_original.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 計算相似度
                similarities = feat_original_norm @ text_features_norm.t()  # [B, K]
                logits_text = similarities * args.text_logitScale
                
                # 獲取偽標籤（直接用 softmax）
                with torch.no_grad():
                    sim_probs = F.softmax(logits_text, dim=1)  # [B, K]
                    max_sim_probs, pseudo_labels = torch.max(sim_probs, dim=1)  # [B]
                    
                    # ⚠️ 嚴格的 threshold = 0.95
                    confidence_mask = (max_sim_probs > 0.95).float()
                    
                    # Debug 資訊
                    if iter_num % 100 == 0 and micro_step == 0:
                        passed = int(confidence_mask.sum().item())
                        total = len(confidence_mask)
                        log_str = f"[Text Loss] Iter {iter_num}: "
                        log_str += f"通過樣本 {passed}/{total} (threshold=0.95), "
                        log_str += f"平均信心: {max_sim_probs.mean().item():.4f}\n"
                        
                        if confidence_mask.any():
                            passed_idx = confidence_mask.bool()
                            acc = (pseudo_labels[passed_idx] == micro_true_labels[passed_idx]).float().mean()
                            log_str += f"  通過樣本的偽標籤準確率: {acc.item()*100:.2f}%\n"
                        
                        print(log_str)
                        args.out_file.write(log_str)
                        args.out_file.flush()
                
                # 計算 loss（只對高信心樣本）
                per_sample_loss = F.cross_entropy(logits_text, pseudo_labels, reduction='none')
                text_loss = (per_sample_loss * confidence_mask).sum() / (confidence_mask.sum() + 1e-8)
                text_loss *= args.text_par
                losses += text_loss
            
            # ===== Loss 3: Contrastive Loss（所有樣本都計算，不用 mask）=====
            contrast_loss = torch.tensor(0.0).cuda()
            
            # 正規化特徵
            feat_weak_norm = feat_projected_weak / feat_projected_weak.norm(dim=-1, keepdim=True)  # [B, 512]
            feat_strong_norm = feat_projected_strong / feat_projected_strong.norm(dim=-1, keepdim=True)  # [B, 512]
            
            # 合併 weak 和 strong features
            out = torch.cat([feat_weak_norm, feat_strong_norm], dim=0)  # [2B, 512]
            
            # 計算相似度矩陣 (SimCLR)
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.tt)  # [2B, 2B]
            
            # 移除對角線（自我相似度）
            mask = (torch.ones_like(sim_matrix) - torch.eye(out.shape[0], device=sim_matrix.device)).bool()
            sim_matrix = sim_matrix.masked_select(mask).view(out.shape[0], -1)  # [2B, 2B-1]
            
            # 計算正樣本對的相似度（weak-strong pair）
            pos_sim = torch.exp(torch.sum(feat_weak_norm * feat_strong_norm, dim=-1) / args.tt)  # [B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2B]
            
            # 計算 contrastive loss（InfoNCE，所有樣本都計算）
            contrast_loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1)).mean()  # 所有樣本的平均
            # losses += contrast_loss
            
            # Accumulate individual losses for epoch logging
            epoch_entropy_loss += im_loss.item()
            epoch_contrast_loss += contrast_loss.item()
            epoch_text_loss += text_loss.item()

            # Scale loss by accumulation steps and backward
            scaled_loss = losses / args.accumulation_steps
            scaled_loss.backward()
            
            epoch_total_loss += scaled_loss.item()
            epoch_batches += 1

        # Update parameters after accumulating gradients
        optimizer.step()
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Total': f'{scaled_loss.item():.4f}',
            'Ent': f'{im_loss.item():.4f}',
            'Text': f'{text_loss.item():.4f}',
            'Con': f'{contrast_loss.item():.4f}'
        })
        
        # Check if epoch is complete
        if (iter_num - epoch_start_iter + 1) % batches_per_epoch == 0:
            # Close current progress bar
            pbar.close()
            
            # Calculate average losses for this epoch
            avg_total_loss = epoch_total_loss / epoch_batches if epoch_batches > 0 else 0
            avg_entropy_loss = epoch_entropy_loss / epoch_batches if epoch_batches > 0 else 0
            avg_contrast_loss = epoch_contrast_loss / epoch_batches if epoch_batches > 0 else 0
            avg_text_loss = epoch_text_loss / epoch_batches if epoch_batches > 0 else 0
            
            # Evaluate model at the end of each epoch
            netP.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netP, text_features, True)
                log_str = 'Epoch: {}/{}, Accuracy = {:.2f}%'.format(current_epoch+1, args.max_epoch, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netP, text_features, False)
                log_str = 'Epoch: {}/{}, Accuracy = {:.2f}%'.format(current_epoch+1, args.max_epoch, acc_s_te)
                print("Epoch {}/{} - Avg Loss: {:.4f}, Entropy: {:.4f}, Text: {:.4f}, Contrast: {:.4f}, Accuracy: {:.2f}%".format(
                    current_epoch+1, args.max_epoch, avg_total_loss, avg_entropy_loss, avg_text_loss, avg_contrast_loss, acc_s_te))
            
            # Log to wandb once per epoch
            wandb.log({
                "epoch": current_epoch + 1,
                "avg_total_loss": avg_total_loss,
                "avg_im_loss": avg_entropy_loss,
                "avg_text_loss": avg_text_loss,
                "avg_contrast_loss": avg_contrast_loss,
                "test_accuracy": acc_s_te,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })
            
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            
            # Save current stage models (overwrite each epoch)
            torch.save(netP.state_dict(), osp.join(args.output_dir, "target_P_current.pt"))
            
            # Save best model if current accuracy is better
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                torch.save(netP.state_dict(), osp.join(args.output_dir, "target_P_best.pt"))
                print(f"New best accuracy: {acc_s_te:.2f}%, saving best model...")
            
            netP.train()
            
            # Reset epoch variables for next epoch
            current_epoch += 1
            epoch_start_iter = iter_num + 1
            epoch_total_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_contrast_loss = 0.0
            epoch_text_loss = 0.0
            epoch_batches = 0
            
            # Create new progress bar for next epoch if not finished
            if current_epoch < args.max_epoch:
                pbar = tqdm(total=batches_per_epoch, desc=f'Epoch {current_epoch+1}/{args.max_epoch}', leave=False)

        iter_num += 1

    # Close final progress bar
    if 'pbar' in locals():
        pbar.close()
    
    # Finish wandb run
    wandb.finish()
    
    return netF, netP, text_features


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netP, text_features, args):
    """
    [已棄用] 使用 Projector + Text Similarity 獲取偽標籤
    注意：新架構中不再需要此函數
    """
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            # CLIP visual encoder → Projector
            feat_512 = netF(inputs.type(netF.conv1.weight.dtype))
            feat_projected = netP(feat_512)
            
            # Text similarity
            feat_norm = feat_projected / feat_projected.norm(dim=-1, keepdim=True)
            text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            outputs = feat_norm @ text_norm.t()
            
            if start_test:
                all_fea = feat_projected.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feat_projected.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)
    entropy = loss.Entropy(all_output)

    weight = 1.0 - torch.exp(-entropy)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    # 簡化版本：不需要複雜的 clustering
    softpred = all_output
    
    acc = accuracy
    log_str = 'Pseudo-label Accuracy = {:.2f}%'.format(acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return softpred, weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=5, help="max iterations")
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'M58'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    # parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--net', type=str, default='RN101', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--plabel', action='store_true')
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0) # im_loss weight
    parser.add_argument('--bn_par', type=float, default=1.0) # bn_loss weight
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--tt', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', type=int, default=1, help="gradient accumulation steps")
    
    # Text alignment loss parameters
    parser.add_argument('--text_par', type=float, default=0.5, help="weight for text alignment loss") # text_loss weight
    parser.add_argument('--text_logitScale', type=float, default=50, help="temperature for text similarity (tau)")
    parser.add_argument('--text_threshold', type=float, default=0.7, help="confidence threshold for text loss mask")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'M58':
        names = ['CAD_ratioFilter', 'Real_all_nobg_augmented']
        args.class_num = 79
        # args.class_num = 30

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = 'data/'
    if args.dset == 'M58':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_79_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_79_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_79_list.txt'
        # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_hard_30_list.txt'
        # args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_hard_30_list.txt'
        # args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_hard_30_list.txt'
    else:
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    print("output_dir_src:", args.output_dir_src)
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][
        0].upper() + datetime.datetime.now().strftime("%m-%d_%H:%M"))
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'cls_par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)