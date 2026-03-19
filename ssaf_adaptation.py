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

class CLIP_Backbone_Wrapper(nn.Module):
    """
    保留完整 CLIP visual encoder，同時提供兩條路徑的特徵輸出：
    - Path 1: attnpool 之前 → GAP → [B, 2048] (用於 Bottleneck → Classifier)
    - Path 2: 經過 attnpool → [B, 512] (用於 CLIP similarity)
    
    attnpool 會隨著訓練更新（不凍結）
    """
    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model
        # 保留完整的 attnpool，不替換成 Identity
        self.in_features = 2048  # Bottleneck 輸入維度
        
    def forward(self, x, return_both=False):
        """
        Args:
            x: 輸入影像 [B, 3, 224, 224]
            return_both: 是否返回雙路徑特徵
                - False: 只返回 feat_2048 (向後兼容)
                - True: 返回 (feat_2048, feat_512)
        
        Returns:
            如果 return_both=False: feat_2048 [B, 2048]
            如果 return_both=True: (feat_2048 [B, 2048], feat_512 [B, 512])
        """
        # 1. 精度對齊
        x = x.type(self.visual.conv1.weight.dtype)
        
        # 2. 手動執行 ResNet backbone（到 layer4 為止）
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        x = self.visual.layer4(x)  # [B, 2048, 7, 7]
        
        # 3. Path 1: GAP → 2048維（用於 Bottleneck → Classifier）
        feat_2048 = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 2048, 1, 1]
        feat_2048 = feat_2048.flatten(1)  # [B, 2048]
        
        # 如果只需要 Path 1，直接返回（向後兼容現有程式碼）
        if not return_both:
            return feat_2048
        
        # 4. Path 2: attnpool → 512維（用於 CLIP similarity）
        feat_512 = self.visual.attnpool(x)  # [B, 512]
        
        return feat_2048, feat_512
    
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


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
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


def load_feature_extractor_weights(model, checkpoint_path):
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
    ## set base network
    clip_model = load_clip_to_cpu(backbone=args.net)
    clip_model.float()
    netF = CLIP_Backbone_Wrapper(clip_model.visual).cuda()
    netF_orignal = CLIP_Backbone_Wrapper(clip_model.visual).cuda()
    
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

    # if args.net[0:3] == 'res':
    #     netF = network.ResBase(res_name=args.net).cuda()
    #     netF_orignal = network.ResBase(res_name=args.net).cuda()
    # elif args.net[0:3] == 'vgg':
    #     netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netB_orignal = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                           bottleneck_dim=args.bottleneck).cuda()

    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # 載入權重 - 修正版
    modelpath = args.output_dir_src + '/source_F.pt'
    pretrained_dict = torch.load(modelpath)
    
    # 檢查權重的 key 結構並正確載入
    print(f"\n{'='*60}")
    print(f"Loading Feature Extractor weights from: {modelpath}")
    print(f"Sample keys from checkpoint: {list(pretrained_dict.keys())[:3]}")
    
    # 為 CLIP_Backbone_Wrapper 正確映射權重
    # CLIP_Backbone_Wrapper 的結構是 wrapper.visual.layer1.xxx
    # source_F.pt 儲存的可能是 layer1.xxx（沒有 visual 前綴）或 visual.layer1.xxx
    sample_key = list(pretrained_dict.keys())[0]
    if not sample_key.startswith('visual.'):
        # 如果 checkpoint 沒有 visual. 前綴，需要加上
        new_state_dict = {f"visual.{k}": v for k, v in pretrained_dict.items()}
        print("Mapping: layer.xxx -> visual.layer.xxx")
    else:
        # 如果已經有 visual. 前綴，直接使用
        new_state_dict = pretrained_dict
        print("Using original keys (already have visual. prefix)")
    
    # 使用 strict=True 來確保權重正確載入
    missing_keys, unexpected_keys = netF.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:5]}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}")
    if not missing_keys and not unexpected_keys:
        print("✅ All weights loaded successfully!")
    print(f"{'='*60}\n")
    
    netF_orignal.load_state_dict(new_state_dict, strict=False)
    
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath)) # Bottleneck Layer
    netB_orignal.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath)) # Classifier
    netC.eval()
    
    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netF_orignal.named_parameters():
        v.requires_grad = False
    for k, v in netB_orignal.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    
    # Initialize wandb
    wandb.init(project="SFDA", 
               name=f"ATSSL_{args.dset}_{args.s}_to_{args.t}_test",
               config=vars(args))
    
    # Calculate micro batch size for memory efficiency
    micro_batch_size = args.batch_size // args.accumulation_steps
    print(f"Batch size: {args.batch_size}, Micro batch size: {micro_batch_size}, Accumulation steps: {args.accumulation_steps}")

    # Initialize best accuracy for saving
    acc_init = 0

    # Training loop variables for epoch tracking
    batches_per_epoch = len(dset_loaders["target"])
    current_epoch = 0
    epoch_start_iter = 0
    epoch_total_loss = 0.0
    epoch_bn_loss = 0.0
    # epoch_classifier_loss = 0.0
    epoch_entropy_loss = 0.0
    epoch_contrast_loss = 0.0
    epoch_text_loss = 0.0
    epoch_batches = 0
    
    # 獲取第一個 epoch 的 loss 權重
    current_weights = get_epoch_loss_weights(current_epoch, args)
    print(f"\n{'='*60}")
    print(f"Epoch {current_epoch+1} Loss Weights:")
    for key, value in current_weights.items():
        print(f"  {key}: {value}")
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

        if iter_num % interval_iter == 0 and current_weights['cls_par'] > 0:
            netF.eval()
            netB.eval()
            softpred, weight = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            softpred = softpred.cuda()
            netF.train()
            netB.train()

        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Move to GPU
        inputs_test = inputs_test.cuda()
        inputs_s = inputs_s.cuda()
        inputs_original = inputs_original.cuda()  # 原始圖片用於 text loss
        true_labels = true_labels.cuda()  # 真實標籤用於評估偽標籤準確率
        mas = weight[tar_idx].cuda() # weight for contrastive loss
        
        # Accumulate gradients over micro batches
        total_loss = 0
        total_bn_loss = 0
        # total_classifier_loss = 0
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
            micro_tar_idx = tar_idx[start_idx:end_idx]
            micro_mas = mas[start_idx:end_idx]
            micro_true_labels = true_labels[start_idx:end_idx]  # 真實標籤
            micro_paths = path[start_idx:end_idx]  # 樣本路徑
            
            # Clear cache periodically
            if micro_step == 0 and iter_num % 100 == 0:
                torch.cuda.empty_cache()
            
            # ===== 獲得雙路徑特徵 =====
            # Process strong augmentation - 獲得 2048 和 512 維特徵
            feat_2048_s= netF(micro_inputs_s)
            features_s = netB(feat_2048_s)  # [B, 2048] → [B, 256]
            out_2 = F.normalize(features_s, dim=-1)

            # Setup BN hooks for micro batch
            loss_bn_layers = []
            if args.bn:
                i, j = 0, 0
                # module1 from target model, module2 from source model
                for module1 in netF.modules():
                    i += 1
                    for module2 in netF_orignal.modules():
                        j += 1
                        if isinstance(module1, nn.BatchNorm2d) and i == j:
                            loss_bn_layers.append(ShotHook(module1, module2)) # ShotHook fetch source/target model's BN statistics
                    j = 0
                for module1 in netB.modules():
                    for module2 in netB_orignal.modules():
                        if isinstance(module1, nn.BatchNorm1d) and isinstance(module2, nn.BatchNorm1d):
                            loss_bn_layers.append(ShotHook(module1, module2))

            # Process weak augmentation - 獲得 2048 和 512 維特徵
            feat_2048_test= netF(micro_inputs_test)
            features_test = netB(feat_2048_test)  # [B, 2048] → [B, 256]
            out_1 = F.normalize(features_test, dim=-1)
            outputs_test = netC(features_test) # weakly augmented Classifier outputs 

            # Calculate losses for micro batch
            losses = torch.tensor(0.0).cuda()
            
            # BN loss
            if args.bn and loss_bn_layers:
                bn_loss = sum([mod.r_feature for mod in loss_bn_layers]) / len(loss_bn_layers) # r_feature from ShotHook KL Divergence
                bn_loss *= current_weights['bn_par']
                losses += bn_loss
            
            # Close hooks
            for x in loss_bn_layers:
                x.close()

            # Classifier loss
            # classifier_loss = torch.tensor(0.0).cuda()
            # if args.cls_par > 0 and args.plabel:
            #     pred = softpred[micro_tar_idx] # output logits from target model
            #     x = F.log_softmax(outputs_test, 1) # Log softmax of weakly augmented outputs
            #     y = F.softmax(pred, 1) # Soft pseudo labels
            #     classifier_loss = nn.KLDivLoss()(x, y) # Equation 5 Loss CLU
            #     classifier_loss *= args.cls_par
            #     if iter_num < interval_iter and args.dset == "VISDA-C":
            #         classifier_loss *= 0
            #     losses += classifier_loss

            # Entropy loss
            im_loss = torch.tensor(0.0).cuda()
            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out)) # Equation 1 Loss ent
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon)) # Equation 2 Loss div
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * current_weights['ent_par']
                losses += im_loss

            # Contrastive loss
            # out = torch.cat([out_1, out_2], dim=0) # out_1 : weakly augmented features, out_2 : strongly augmented features
            # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.tt) # (2N x D) ▪ (D x 2N) -> 2N x 2N similarity matrix
            # # ones_like: matrix of all ones with same shape as sim_matrix
            # # torch.eye: identity matrix (對角線為1，其餘為0)
            # # 目的: 建立一個對角線為0，其餘為1的mask矩陣，以去除自我相似度的影響
            # mask = (torch.ones_like(sim_matrix) - torch.eye(out.shape[0], device=sim_matrix.device)).bool() # Mask to remove self-similarity
            # sim_matrix = sim_matrix.masked_select(mask).view(out.shape[0], -1) # Lower Equation 10, Reshape from 2N to (2N-1) without self-similarity

            # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.tt)  # Upper Equation 10
            # pos_sim = torch.cat([pos_sim, pos_sim], dim=0) # To match 2N size since simCLR calculates Week -> Strong and Strong -> Week
            # micro_mas_expanded = torch.cat([micro_mas, micro_mas]) # Weighting for each sample

            # contrast_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1)) * micro_mas_expanded).mean() # Equation 12 Loss con
            # losses += contrast_loss
            
            # # ===== Text Alignment Loss =====
            # ===== Text Alignment Loss（先計算以取得 confidence_mask）=====
            # 使用「原始圖片」的 512 維特徵計算與 text embeddings 的對齊
            text_loss = torch.tensor(0.0).cuda()
            confidence_mask = torch.zeros(micro_inputs_test.size(0)).cuda()  # 預設全為低信心 [B]

            if current_weights['text_par'] > 0:
                # 0. 提取原始圖片的特徵（不使用 augmented 圖片）
                micro_inputs_original = inputs_original[start_idx:end_idx]
                _, feat_512_original = netF(micro_inputs_original, return_both=True)
                
                # 1. 正規化圖片特徵 z_i
                feat_512_norm = feat_512_original / feat_512_original.norm(dim=-1, keepdim=True)  # [B, 512]
                
                # 2. 計算與所有文本原型的相似度 sim(z_i, t_j)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)  # [K, 512]
                similarities = feat_512_norm @ text_features_norm.t()  # [B, K]
                logits_text = similarities * args.text_logitScale  # 放大相似度以增強區分度
                # 3. 獲取偽標籤 ŷ_i（使用 cosine similarity 最高分數的類別）
                with torch.no_grad():
                    # ================= DEBUG START =================
                    # print(f"\n[DEBUG] Iter {iter_num} Check:")
                    # print(f"1. Raw Similarities: min={similarities.min().item():.4f}, max={similarities.max().item():.4f}, mean={similarities.mean().item():.4f}")
                    # print(f"2. Logit Scale Used: {args.text_logitScale}")
                    # print(f"3. Final Logits (before softmax): min={logits_text.min().item():.4f}, max={logits_text.max().item():.4f}")
                    # # 檢查是否有 NaN
                    # if torch.isnan(logits_text).any():
                    #     print("⚠️ ALERT: Logits contain NaN!")
                    # ================= DEBUG END =================
                    # 使用 cosine similarity 來決定偽標籤（而非 classifier 輸出）
                    sim_probs = F.softmax(logits_text, dim=1)  # [B, K]
                    current_class_prob = sim_probs.mean(dim=0)  # [K]
    
                    # C. 計算懲罰項：頻率越高，扣分越重
                    # 加上 1e-8 避免 log(0)
                    # alpha 係數控制平衡強度，通常 1.0 即可
                    balance_penalty = torch.log(current_class_prob + 1e-8) 
                    
                    # D. 修正 Logits
                    # Logit_new = Logit_old - log(P(y))
                    # 這樣做等同於在最大化互信息 (Mutual Information)
                    balanced_logits = logits_text - balance_penalty.unsqueeze(0)
                    # E. 使用修正後的 Logits 產生偽標籤
                    balanced_probs = F.softmax(balanced_logits, dim=1)

                    max_sim_probs, pseudo_labels = torch.max(balanced_probs, dim=1)  # [B]
                    
                    # Debug: 監控平衡後的效果 (可選)
                    if iter_num % 100 == 0 and micro_step == 0:
                        pred_counts = torch.bincount(pseudo_labels, minlength=args.class_num)
                        log_str = (f"[Balance Check] Top 5 predicted classes count:")
                        top5_val, top5_idx = torch.topk(pred_counts, 5)
                        for val, idx in zip(top5_val, top5_idx):
                            log_str += (f"  - Class {idx.item()}: {val.item()} samples")
                        print(log_str)
                        args.out_file.write(log_str + '\n')
                        args.out_file.flush()

                    # 4. 計算置信度 mask M_i（只對高置信度樣本計算 loss）
                    confidence_mask = (max_sim_probs > current_weights['text_threshold']).float()  # [B]
                    
                    # Debug: 每 100 個 iteration 打印一次統計資訊
                    if iter_num % 100 == 0 and micro_step == 0:
                        log_str = f"[Text Loss Debug] Iter {iter_num}:\n"
                        log_str += f"  - 通過樣本: {int(confidence_mask.sum().item())}/{len(confidence_mask)};\n"
                        log_str += f"  - 平均最高機率: {max_sim_probs.mean().item():.4f};\n"
                        log_str += f"  - Max probs 範圍: [{max_sim_probs.min():.4f}, {max_sim_probs.max():.4f}];\n"
                        log_str += f"  - Threshold: {args.text_threshold}\n"
                        
                        # 計算通過樣本的偽標籤準確率
                        passed_indices = confidence_mask.bool()
                        if passed_indices.any():
                            pseudo_label_accuracy = (pseudo_labels[passed_indices] == micro_true_labels[passed_indices]).float().mean()
                            log_str += f"  - 通過樣本的偽標籤準確率: {pseudo_label_accuracy.item()*100:.2f}%\n"
                        else:
                            log_str += f"  - ⚠️ 沒有樣本通過 threshold\n"
                        
                        # 打印所有樣本的詳細信息
                        log_str += f"\n  樣本詳細資訊:\n"
                        log_str += f"  {'Idx':<4} {'Pass':<5} {'Prob':<8} {'True':<5} {'Pred':<5} {'PredClass':<30} {'Correct':<8} Path\n"
                        log_str += f"  {'-'*130}\n"
                        for i in range(len(micro_true_labels)):
                            passed = "✓" if confidence_mask[i] > 0 else "✗"
                            prob = max_sim_probs[i].item()
                            true_id = micro_true_labels[i].item()
                            pred_id = pseudo_labels[i].item()
                            pred_classname = loaded_classnames[pred_id]  # 獲取預測類別名稱
                            is_correct = "✓" if true_id == pred_id else "✗"
                            sample_path = '/'.join(micro_paths[i].split('/')[-2:]) if '/' in micro_paths[i] else micro_paths[i]  # 顯示上一層資料夾/檔名
                            log_str += f"  {i:<4} {passed:<5} {prob:<8.4f} {true_id:<5} {pred_id:<5} {pred_classname:<30} {is_correct:<8} {sample_path}\n"
                        
                        print(log_str)
                        args.out_file.write(log_str + '\n')
                        args.out_file.flush()
                
                # 5. 計算帶溫度係數的 logits
                # logits_text = similarities  # [B, K]
                
                # 6-7. 使用 F.cross_entropy 計算 loss
                per_sample_loss = F.cross_entropy(logits_text, pseudo_labels, reduction='none')  # [B]
                
                # 手動應用 mask 並平均化
                text_loss = (per_sample_loss * confidence_mask).sum() / (confidence_mask.sum() + 1e-8)
                text_loss *= current_weights['text_par']
                
                losses += text_loss
            
            # ===== Contrastive Loss（只對低信心樣本計算，與 Text Loss 互補）=====
            # 策略：高信心樣本用 Text Alignment Loss，低信心樣本用 Contrastive Loss
            contrast_loss = torch.tensor(0.0).cuda()
            
            # 建立反向 mask（低信心樣本 = 1）
            contrastive_mask = 1.0 - confidence_mask  # [B]
            
            # 檢查是否有低信心樣本需要計算 Contrastive Loss
            if contrastive_mask.sum() > 0:
                # out_1: weakly augmented features [B, 256]
                # out_2: strongly augmented features [B, 256]
                out = torch.cat([out_1, out_2], dim=0)  # [2B, 256] out_1 : weakly augmented features, out_2 : strongly augmented features
                
                # 計算相似度矩陣 (SimCLR)
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.tt)  # [2B, 2B] (2N x D) ▪ (D x 2N) -> 2N x 2N similarity matrix
                
                # 建立 mask 去除自我相似度（對角線）
                # ones_like: 全 1 矩陣，torch.eye: 對角線為 1 的單位矩陣 matrix of all ones with same shape as sim_matrix
                # torch.eye: identity matrix (對角線為1，其餘為0)
                # 目的: 建立一個對角線為0，其餘為1的mask矩陣，以去除自我相似度的影響
                mask = (torch.ones_like(sim_matrix) - torch.eye(out.shape[0], device=sim_matrix.device)).bool() # Mask to remove self-similarity
                sim_matrix = sim_matrix.masked_select(mask).view(out.shape[0], -1)  # [2B, 2B-1] Lower Equation 10, Reshape from 2N to (2N-1) without self-similarity
                
                # 計算正樣本對的相似度（weak-strong pair）
                pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.tt)  # [B] Upper equation 10
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2B] (對稱：weak→strong 和 strong→weak) To match 2N size since simCLR calculates Week -> Strong and Strong -> Week
                
                # 擴展 mask 和權重到 2B
                contrastive_mask_expanded = torch.cat([contrastive_mask, contrastive_mask])  # [2B] Mask for low-confidence samples
                micro_mas_expanded = torch.cat([micro_mas, micro_mas])  # [2B] Weighting for each sample
                
                # 計算 per-sample contrastive loss (InfoNCE)
                per_sample_contrast_loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))  # [2B]
                
                # 只對低信心樣本計算 loss（結合 contrastive_mask 和 entropy weight）
                # per_sample_contrast_loss: equation 12 lcon(x_i)
                # micro_mas_expanded: equation 12 w_i
                # contrastive_mask_expanded: filter for low-confidence samples
                weighted_loss = per_sample_contrast_loss * contrastive_mask_expanded * micro_mas_expanded
                contrast_loss = weighted_loss.sum() / (contrastive_mask_expanded.sum() + 1e-8)
            else:
                # 所有樣本都是高信心，不需要 Contrastive Loss
                contrast_loss = torch.tensor(0.0).cuda()
            
            losses += contrast_loss
            
            # Accumulate individual losses for epoch logging
            epoch_bn_loss += bn_loss.item()
            # epoch_classifier_loss += classifier_loss.item()
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
        batch_in_epoch = (iter_num - epoch_start_iter) % batches_per_epoch
        pbar.update(1)
        pbar.set_postfix({
            'Total': f'{scaled_loss.item():.4f}',
            'BN': f'{bn_loss.item():.4f}',
            # 'Cls': f'{classifier_loss.item():.4f}',
            'Ent': f'{im_loss.item():.4f}',
            'Con': f'{contrast_loss.item():.4f}',
            'Text': f'{text_loss.item():.4f}'
        })
        
        # Check if epoch is complete
        if (iter_num - epoch_start_iter + 1) % batches_per_epoch == 0:
            # Close current progress bar
            pbar.close()
            
            # Calculate average losses for this epoch
            avg_total_loss = epoch_total_loss / epoch_batches if epoch_batches > 0 else 0
            avg_bn_loss = epoch_bn_loss / epoch_batches if epoch_batches > 0 else 0
            # avg_classifier_loss = epoch_classifier_loss / epoch_batches if epoch_batches > 0 else 0
            avg_entropy_loss = epoch_entropy_loss / epoch_batches if epoch_batches > 0 else 0
            avg_contrast_loss = epoch_contrast_loss / epoch_batches if epoch_batches > 0 else 0
            avg_text_loss = epoch_text_loss / epoch_batches if epoch_batches > 0 else 0
            
            # Evaluate model at the end of each epoch
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Epoch: {}/{}, Accuracy = {:.2f}%'.format(current_epoch+1, args.max_epoch, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Epoch: {}/{}, Accuracy = {:.2f}%'.format(current_epoch+1, args.max_epoch, acc_s_te)
                print("Epoch {}/{} - Avg Loss: {:.4f}, BN: {:.4f}, Entropy: {:.4f}, Contrast: {:.4f}, Text: {:.4f}, Accuracy: {:.2f}%".format(
                    current_epoch+1, args.max_epoch, avg_total_loss, avg_bn_loss, avg_entropy_loss, avg_contrast_loss, avg_text_loss, acc_s_te))
            
            # Log to wandb once per epoch
            wandb.log({
                "epoch": current_epoch + 1,
                "avg_total_loss": avg_total_loss,
                "avg_bn_loss": avg_bn_loss,
                # "avg_clu_loss": avg_classifier_loss,
                "avg_im_loss": avg_entropy_loss,
                "avg_contrast_loss": avg_contrast_loss,
                "avg_text_loss": avg_text_loss,
                "test_accuracy": acc_s_te,
                "learning_rate": optimizer.param_groups[0]['lr'],
                # 記錄當前使用的權重
                "weight_ent_par": current_weights['ent_par'],
                "weight_bn_par": current_weights['bn_par'],
                "weight_text_par": current_weights['text_par'],
                "weight_text_threshold": current_weights['text_threshold'],
                "weight_cls_par": current_weights['cls_par'],
            })
            
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            
            # Save current stage models (overwrite each epoch)
            # 如果不想要有visual.layer3..在前面的話
            #torch.save(netF.visual.state_dict(), osp.join(args.output_dir, "target_F_current.pt"))
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_current.pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_current.pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_current.pt"))
            
            # Save best model if current accuracy is better
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_best.pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_best.pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_best.pt"))
                print(f"New best accuracy: {acc_s_te:.2f}%, saving best model...")
            
            netF.train()
            netB.train()
            
            # Reset epoch variables for next epoch
            current_epoch += 1
            epoch_start_iter = iter_num + 1
            epoch_total_loss = 0.0
            epoch_bn_loss = 0.0
            # 更新下一個 epoch 的 loss 權重
            if current_epoch < args.max_epoch:
                current_weights = get_epoch_loss_weights(current_epoch, args)
                print(f"\n{'='*60}")
                print(f"Epoch {current_epoch+1} Loss Weights:")
                for key, value in current_weights.items():
                    print(f"  {key}: {value}")
                print(f"{'='*60}\n")
            
            # epoch_classifier_loss = 0.0
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
    
    return netF, netB, netC


def get_epoch_loss_weights(epoch, args):
    """
    根據當前 epoch 動態調整 loss 權重
    
    Args:
        epoch: 當前 epoch (0-indexed)
        args: 參數物件
    
    Returns:
        dict: 包含所有 loss 權重的字典
    """
    # 定義不同 epoch 區間的參數配置
    # 格式: (start_epoch, end_epoch): {參數名: 值}
    epoch_configs = {
        (0, 10): {
            'ent_par': 1.0,
            'bn_par': 1.0,
            'text_par': 0.5,
            'text_threshold': 0.7,  # 保持預設值
            'cls_par': 0.3,
        },
        (10, 50): {
            'ent_par': 1.0,
            'bn_par': 1.0,
            'text_par': 0.5,
            'text_threshold': 0.7,  # 降低閾值讓更多樣本參與訓練
            'cls_par': 0.3,
        },
        (50, 999): {  # 20 epoch 之後使用原始設定
            'ent_par': args.ent_par,
            'bn_par': args.bn_par,
            'text_par': args.text_par,
            'text_threshold': args.text_threshold,
            'cls_par': args.cls_par,
        }
    }
    
    # 找到當前 epoch 對應的配置
    for (start, end), config in epoch_configs.items():
        if start <= epoch < end:
            return config
    
    # 如果沒有匹配的區間，返回原始參數
    return {
        'ent_par': args.ent_par,
        'bn_par': args.bn_par,
        'text_par': args.text_par,
        'text_threshold': args.text_threshold,
        'cls_par': args.cls_par,
    }


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)
    entropy = loss.Entropy(all_output)

    weight = 1.0 - torch.exp(-entropy) # Equation 11

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)

    dd = cdist(all_fea, initc, args.distance)
    dd = 1 / dd
    dd = torch.Tensor(dd)
    softpred = nn.Softmax(dim=1)(dd / args.T)
    _, pred_label = torch.max(softpred, dim=-1)

    acc = torch.sum(pred_label == all_label) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

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