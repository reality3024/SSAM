"""
CLIP + MLPProjector EMA Teacher SFDA Training Script
Architecture:
  CLIP RN101 Visual Encoder (fully frozen)
  + Student MLP ProjectorC (trainable, used for classification)
  + Student MLP ProjectorH (trainable, chained after ProjectorC, for contrastive learning only)
  + Teacher MLP ProjectorC_ema (EMA, passively updated with Student ProjectorC)

Loss Functions (four types):
  [A] CE Loss          - Cross-entropy for high-confidence Pseudo-label samples (reliable)
  [B] Propagation Loss - MSE alignment to Teacher logits for low-confidence samples (unreliable)
  [C] Information Loss - Entropy minimization + diversity for all samples (im_loss)
  [D] Contrastive Loss - SimCLR contrastive learning on weak/strong augmented features

Feature Usage Principles:
  - Student: Strong augmented image → ProjectorC → used for CE, Propagation Loss
  - Student: Original image → ProjectorC → used for IM Loss
  - Teacher: Original image → ProjectorC_ema → used for Pseudo Label generation
  - Contrastive Loss: Weak + Strong → ProjectorC → ProjectorH (chained)

Training Strategy (Phase 2 only):
  - CE=1.0, Con=1.0, Prop starts from 0 and increases to 1.0 (knowledge distillation)
  - logitScale=5, Threshold=dynamic average confidence (reference C-SFDA)
  - Pseudo Label calculation: computed in real-time for each batch (using latest Teacher EMA)
  - Each Epoch end: use obtain_label_clip to monitor overall Pseudo Label quality

Reference Sources:
  - contrast_feature_micro.py (main backbone, Contrastive Loss, IM Loss)
  - C-SFDA/target_csfda.py (EMA Teacher, CE Loss, Propagation Loss, dynamic Threshold)
  - clip_adaptation_projector.py (Projector architecture, Text Feature generation)
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
import wandb
from data_list import ImageList, ImageList_idx
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import clip
import math


# ==================================================
# Model Definition
# ==================================================

class MLPProjector(nn.Module):
    """MLP Projector (identical architecture to Source Model training)"""
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
    Return three versions of images:
    1. weak augmentation (Aug 1): light geometric transformation → Teacher input
    2. strong augmentation (Aug 2): SimCLR style (color/blur/geometry) → Student input
    3. original: only Resize + CenterCrop, used for Pseudo Label monitoring (not used during training)
    """
    def __init__(self, resize_size=256, crop_size=224):
        # OpenAI CLIP official standard Mean and Std, must not use ImageNet values
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # Original: fully aligned with CLIP official preprocess (consistent with centroid extraction)
        # CLIP official uses Resize(224) instead of Resize((224, 224)), which preserves aspect ratio
        self.original = transforms.Compose([
            transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),  # short side scaling
            transforms.CenterCrop(crop_size)
        ])

        # Weak: light geometric transformation (translation, rotation, flip), no color changes
        self.weak = transforms.Compose([
            transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])

        # Strong: SimCLR style (strong color jitter and blur, forcing model to learn shape contours)
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
        # Apply CLIP's Normalization uniformly at the end
        return self.normalize(weak), self.normalize(strong), self.normalize(original)


# ==================================================
# Utility Functions
# ==================================================

def smoothed_cross_entropy(logits, labels, num_classes=37, epsilon=0.1):
    """
    epsilon=0.1 means 90% confidence, remaining 10% probability distributed uniformly to other classes.
    This greatly prevents overfitting to incorrect Pseudo-labels.
    """
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        # Create one-hot targets
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        # Mix in uniform distribution
        targets = (1 - epsilon) * targets + epsilon / num_classes

    # Compute Soft Cross Entropy
    loss = (-targets * log_probs).sum(dim=1).mean()
    return loss

def load_clip_to_cpu(backbone):
    """Load CLIP model (CPU version, will be moved to GPU later)"""
    model, _ = clip.load(backbone, device="cpu",
                         download_root=os.path.expanduser("~/.cache/clip"))
    return model


def op_copy(optimizer):
    """Record initial learning rates for lr_scheduler"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """Inverse decay learning rate scheduler"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr']           = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum']     = 0.9
        param_group['nesterov']     = True
    return optimizer


def get_clip_preprocess_transforms(crop_size=224):
    """
    Return CLIP official preprocess's Resize + CenterCrop part (without ToTensor and Normalize)
    Used for Augmentation's original branch
    """
    return transforms.Compose([
        transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size)
    ])


def generate_text_features(classnames_path, clip_model, device, template="a photo of a {}"):
    """
    Generate CLIP Text Features from classname file

    Args:
        classnames_path: Path to classname file (one per line)
        clip_model:      Complete CLIP model already moved to device
        device:          Computation device string
        template:        Prompt template, default "a photo of a {}"

    Returns:
        text_features: [C, 512] L2 normalized Text Features (float32, on device)
        classnames:    List of class names
    """
    with open(classnames_path, 'r', encoding='utf-8') as f:
        classnames = [line.strip() for line in f if line.strip()]

    print(f'Generating Text Features')
    print(f'  - Number of classes: {len(classnames)}')
    print(f'  - Prompt template: "{template}"')

    clip_model.eval()
    with torch.no_grad():
        texts        = [template.format(name) for name in classnames]
        tokens       = clip.tokenize(texts).to(device)
        text_features = clip_model.encode_text(tokens)        # [C, D]
        text_features = F.normalize(text_features.float(), dim=-1)

    print(f'Text Features generation complete: shape = {text_features.shape}')

    return text_features, classnames


def data_load(args, preprocess):
    """Load dataset (Target training + evaluation)

    Args:
        preprocess: CLIP official preprocess (used for test loader)
    """
    dsets       = {}
    dset_loaders = {}
    train_bs    = args.batch_size

    txt_tar  = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    root_path = f'data/{args.dset}/'

    # Training Loader: each sample returns (weak, strong, original) three augmented images
    dsets["target"] = ImageList_idx(txt_tar, transform=Augmentation(), root=root_path)
    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False
    )

    # Test Loader: use CLIP official preprocess (fully aligned with extract_class_centroid.py)
    dsets["test"] = ImageList_idx(txt_test, transform=preprocess, root=root_path)
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=train_bs * 3, shuffle=False,
        num_workers=args.worker, drop_last=False
    )

    return dset_loaders


def cal_acc(loader, netF, netP, class_centroids, target_global_mean, source_global_mean, args, flag=False):
    """
    Evaluate accuracy using Student Projector

    Args:
        class_centroids: [C, D] Class center features extracted from Source dataset (normalized)
        target_global_mean: [1, D] Target global mean
        source_global_mean: [1, D] Source global mean
    """
    netP.eval()
    clip_dtype = next(netF.parameters()).dtype

    # Pre-compute centered source centroids
    centered_source_centroids = F.normalize(class_centroids - source_global_mean, dim=-1)

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data   = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]

            feat_512  = netF(inputs.type(clip_dtype))       # [B, 512], frozen
            feat_proj = netP(feat_512)                       # [B, 512]
            feat_norm = F.normalize(feat_proj, dim=-1)

            # Use global centering to compute logits
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
    Generate pseudo labels for the entire dataset using Teacher Projector (EMA, eval mode)
    Note: This function is only used for monitoring overall Pseudo Label quality, called once per epoch
    Training Pseudo Labels are computed in real-time for each batch (in main training loop)

    Modified: Use stable Teacher instead of oscillating Student
    Modified: Use class_centroids (class centers extracted from Source) instead of text features
    Modified: Use global centering + logit_scale (consistent with plot_threshold_distribution.py)
    Modified: Support dynamic threshold (reference C-SFDA)

    Args:
        class_centroids: [C, D] Class center features extracted from Source (normalized)
        current_epoch: Current epoch number

    Returns:
        pseudo_labels : [N] (CPU LongTensor) Pseudo labels for each sample
        max_probs     : [N] (CPU FloatTensor) Maximum softmax confidence values
    """
    netP_ema.eval()
    clip_dtype = next(netF.parameters()).dtype

    # First pass: extract all Target features (for global centering)
    all_features = []
    all_labels = []

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data   = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]

            feat_512  = netF(inputs.type(clip_dtype))
            feat_proj = netP_ema(feat_512)  # Use Teacher EMA
            feat_norm = F.normalize(feat_proj, dim=-1)

            all_features.append(feat_norm.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)  # [N, D]
    all_labels = torch.cat(all_labels, dim=0)      # [N]

    # Use global centering (identical to plot_threshold_distribution.py)
    target_mean = all_features.mean(dim=0, keepdim=True)
    source_mean = class_centroids.mean(dim=0, keepdim=True).cpu()

    centered_target = F.normalize(all_features - target_mean, dim=-1)
    centered_source = F.normalize(class_centroids.cpu() - source_mean, dim=-1)

    # Use logitScale (consistent with training)
    use_logit_scale = args.centroid_logitScale

    # Calculate logits
    logits = use_logit_scale * (centered_target @ centered_source.T)  # [N, C]

    # Convert logits to probability
    all_probs = F.softmax(logits, dim=1)           # [N, C]
    max_probs, pseudo_labels = torch.max(all_probs, dim=1)  # [N]

    # Record statistics
    accuracy       = torch.sum(pseudo_labels.float() == all_labels.float()).item() / float(all_labels.size()[0])
    reliable_ratio = (max_probs > args.conf_thres).float().mean().item()

    # Compute purity for reliable samples
    reliable_mask = max_probs > args.conf_thres
    if reliable_mask.sum() > 0:
        reliable_correct = torch.sum((pseudo_labels[reliable_mask].float() == all_labels[reliable_mask].float())).item()
        reliable_total = reliable_mask.sum().item()
        reliable_purity = reliable_correct / reliable_total
    else:
        reliable_purity = 0.0

    log_str = (
        f'[Pseudo Label via Teacher Centroids + Global Centering] Accuracy = {accuracy * 100:.2f}%  |  '
        f'logitScale={use_logit_scale}  |  '
        f'  Reliable (conf > {args.conf_thres:.4f}) = {reliable_ratio * 100:.2f}%  |  '
        f'Reliable count = {int((max_probs > args.conf_thres).sum())}/{len(pseudo_labels)}  |  '
        f'Reliable Purity = {reliable_purity * 100:.2f}%'
    )
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)

    # Return statistics for wandb logging
    stats = {
        'pseudo_label_accuracy': accuracy,
        'reliable_ratio': reliable_ratio,
        'reliable_count': int((max_probs > args.conf_thres).sum()),
        'total_samples': len(pseudo_labels),
        'reliable_purity': reliable_purity,
    }

    return pseudo_labels, max_probs, stats


def compute_target_global_mean(netF, netP_C_ema, data_loader, clip_dtype):
    """
    Compute global mean of Target dataset (using current Teacher ProjectorC_ema)

    Args:
        netF: CLIP Visual Encoder (frozen)
        netP_C_ema: Teacher ProjectorC (EMA)
        data_loader: Target data loader
        clip_dtype: CLIP model dtype

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

            # Use weak augmentation
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


# ==================================================
# Main Training Function
# ==================================================

def train_target(args):
    # Load and freeze CLIP Visual Encoder
    print(f'Loading CLIP Model: {args.net}')
    clip_model, preprocess = clip.load(args.net, device="cpu",
                                        download_root=os.path.expanduser("~/.cache/clip"))
    clip_model.float()
    clip_model = clip_model.cuda()
    print(f'CLIP official preprocess loaded (consistent with extract_class_centroid.py)')

    # Load dataset (using CLIP official preprocess)
    dset_loaders = data_load(args, preprocess)

    netF = clip_model.visual       # Extract Visual Encoder only
    netF.eval()
    for param in netF.parameters():
        param.requires_grad = False

    clip_dtype = next(netF.parameters()).dtype
    print(f'CLIP Visual Encoder loaded and frozen (eval mode)')
    print(f'  - dtype: {clip_dtype}')

    # Generate fixed CLIP Text Features
    # text_features, classnames = generate_text_features(
    #     classnames_path=args.classnames_path,
    #     clip_model=clip_model,
    #     device='cuda',
    #     template="a photo of a {}"
    # )
    # text_features = text_features.cuda()   # [C, 512], L2 normalized
    # text_norm     = F.normalize(text_features, dim=-1)

    # if len(classnames) != args.class_num:
    #     raise ValueError(
    #         f'class_num mismatch: classname file has {len(classnames)} classes, '
    #         f'but args.class_num = {args.class_num}, please verify.'
    #     )

    # Load Class Centroids (for Pseudo Label generation)
    print(f'\nLoading Class Centroids from: {args.centroid_path}')
    if not os.path.exists(args.centroid_path):
        raise FileNotFoundError(
            f'Class Centroids file not found: {args.centroid_path}\n'
            f'Please run extract_class_centroid.py first to extract class centers from Source dataset.'
        )

    centroid_data = torch.load(args.centroid_path)
    class_centroids = centroid_data['centroids'].cuda()  # [C, 512]
    print(f'Class Centroids loaded: {class_centroids.shape}')
    print(f'   From {centroid_data.get("num_classes", "unknown")} classes')

    if class_centroids.size(0) != args.class_num:
        raise ValueError(
            f'class_num mismatch: Centroid file has {class_centroids.size(0)} classes, '
            f'but args.class_num = {args.class_num}, please verify.'
        )

    # Initialize mixed chained Projector architecture
    print(f'Initializing mixed chained Projector architecture:')
    print(f'  - ProjectorC: for classification tasks (CE/Prop/IM Loss)')
    print(f'  - ProjectorH: chained after ProjectorC, only for contrastive learning (Contrastive Loss)')
    print(f'  - Teacher only has ProjectorC_ema, no need for ProjectorH_ema')

    netP_C = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).cuda()
    netP_H = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).cuda()

    # Prefer directly specified path, otherwise derive from output_dir_src
    if args.projector_path is not None:
        projector_path = args.projector_path
    else:
        projector_path = osp.join(args.output_dir_src, 'source_P.pt')

    if not osp.exists(projector_path):
        raise FileNotFoundError(
            f'Projector checkpoint not found: {projector_path}\n'
            f'Please verify the path or specify directly via --projector_path.'
        )

    # Load pre-trained weights for both Projectors
    state_dict = torch.load(projector_path)
    netP_C.load_state_dict(state_dict)
    netP_H.load_state_dict(state_dict)
    netP_C.train()
    netP_H.train()
    print(f'ProjectorC and ProjectorH loaded: {projector_path}')

    # Create Teacher Projector (EMA, only for ProjectorC)
    netP_C_ema = copy.deepcopy(netP_C)
    for param in netP_C_ema.parameters():
        param.requires_grad = False   # Teacher fully frozen, updated only through EMA
    netP_C_ema.eval()
    print(f'Teacher ProjectorC (EMA) created (no need for ProjectorH_ema)')

    # Set up optimizer (optimize ProjectorC + ProjectorH simultaneously)
    param_group = []
    for k, v in netP_C.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netP_H.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    print(f'Optimizer configured')
    print(f'  - Trainable modules: ProjectorC + ProjectorH (chained)')
    print(f'  - Initial LR: {args.lr}')
    print(f'  - EMA momentum: {args.ema_m} (only updates Teacher ProjectorC)')
    print(f'  - conf_thres: {args.conf_thres}')
    print(f'  - centroid_logitScale: {args.centroid_logitScale}  (for monitoring with Centroids)')
    print(f'\n  Loss configuration:')
    print(f'  [A] CE Loss: 1.0  (high confidence Pseudo-label CE)')
    print(f'  [B] Propagation Loss: 0→1 (low confidence MSE to Teacher)')
    print(f'  [C] IM Loss: {args.ent_par}  (entropy minimization + diversity)')
    print(f'  [D] Contrastive Loss: 1.0  (SimCLR Weak+Strong, dynamic entropy weight)')
    print(f'\n  Training strategy:')
    print(f'  - Phase 2 only: CE=1.0, Con=1.0, Prop=0→1, logitScale=5, Threshold=dynamic average')
    print(f'  - Pseudo Label calculation: in real-time for each batch (using latest Teacher EMA)')
    print(f'\n  Image allocation:')
    print(f'  - CE / Propagation Loss: Strong augmented (Student)')
    print(f'  - IM Loss: Original image (Student)')
    print(f'  - Contrastive Loss: Weak + Strong (Student, chained to ProjectorH)')
    print(f'  - Teacher (Pseudo Label): Original image (all epochs, reference C-SFDA)')

    # Resume training (if resume_dir specified)
    start_epoch_offset = 0
    if args.resume_dir is not None:
        print(f'\nResume training mode activated')
        print(f'   Loading models from: {args.resume_dir}')

        # Check if files exist (mixed architecture: 3 models)
        projectorC_path = osp.join(args.resume_dir, "target_ProjectorC_best.pt")
        projectorH_path = osp.join(args.resume_dir, "target_ProjectorH_best.pt")
        projectorC_ema_path = osp.join(args.resume_dir, "target_ProjectorC_ema_current.pt")

        if not osp.exists(projectorC_path):
            raise FileNotFoundError(f'ProjectorC checkpoint not found: {projectorC_path}')
        if not osp.exists(projectorH_path):
            raise FileNotFoundError(f'ProjectorH checkpoint not found: {projectorH_path}')
        if not osp.exists(projectorC_ema_path):
            raise FileNotFoundError(f'ProjectorC_ema checkpoint not found: {projectorC_ema_path}')

        # Load model weights (no need for netP_H_ema)
        netP_C.load_state_dict(torch.load(projectorC_path))
        netP_H.load_state_dict(torch.load(projectorH_path))
        netP_C_ema.load_state_dict(torch.load(projectorC_ema_path))

        print(f'   ProjectorC loaded: {projectorC_path}')
        print(f'   ProjectorH loaded: {projectorH_path}')
        print(f'   ProjectorC_ema loaded: {projectorC_ema_path}')
        print(f'   (mixed architecture: no need for ProjectorH_ema)')
        print(f'   Starting Epoch: {args.start_epoch}')

        start_epoch_offset = args.start_epoch

        if args.con_par is not None:
            print(f'   Fixed Contrastive Loss weight: {args.con_par}')

    max_iter         = (args.max_epoch - args.start_epoch) * len(dset_loaders["target"])
    interval_iter    = max_iter // args.interval
    iter_num         = 0

    print(f'\nBatch size: {args.batch_size}')
    print(f'Total iterations: {max_iter}, Pseudo-label update interval: {interval_iter} iters')
    print(f'Seed: {args.seed}')
    print(f'Data path (Target): {args.t_dset_path}')
    print(f'Data path (Test): {args.test_dset_path}\n')

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="CLIP-SFDA",
            name=f"{args.dset}_s{args.s}t{args.t}_epoch{args.max_epoch}",
            config={
                "dataset": args.dset,
                "source": args.s,
                "target": args.t,
                "max_epoch": args.max_epoch,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "ema_momentum": args.ema_m,
                "conf_threshold": args.conf_thres,
                "centroid_logitScale": args.centroid_logitScale,
                "ent_par": args.ent_par,
                "seed": args.seed,
            }
        )
        print(f'wandb initialized: {wandb.run.name}\n')
    else:
        print(f'wandb disabled (use_wandb=False)\n')

    # Pre-compute Source global mean (fixed) and initial Target global mean
    print(f'Pre-computing global means for global centering...')

    # Source global mean (fixed, unchanged during entire training)
    source_global_mean = class_centroids.mean(dim=0, keepdim=True)  # [1, D]

    # Initial Target global mean (using initial Teacher)
    target_global_mean = compute_target_global_mean(
        netF, netP_C_ema, dset_loaders["target"], clip_dtype
    )

    # Pre-compute centered source centroids (will be updated after each batch)
    centered_source_centroids = F.normalize(class_centroids - source_global_mean, dim=-1)  # [C, D]

    print(f'Global means computation complete')
    print(f'   Source global mean: {source_global_mean.shape} (fixed)')
    print(f'   Target global mean: {target_global_mean.shape} (initial, updated after each batch)')
    print(f'   Centered Source Centroids: {centered_source_centroids.shape}')

    # Initialize best accuracy
    acc_init = 0

    # Epoch tracking variables
    batches_per_epoch   = len(dset_loaders["target"])
    current_epoch       = start_epoch_offset
    epoch_start_iter    = 0
    epoch_total_loss    = 0.0
    epoch_ce_loss       = 0.0
    epoch_prop_loss     = 0.0
    epoch_entropy_loss  = 0.0
    epoch_contrast_loss = 0.0
    epoch_batches       = 0

    # Create initial progress bar
    iter_test = iter(dset_loaders["target"])
    pbar      = tqdm(total=batches_per_epoch,
                     desc=f'Epoch {current_epoch+1}/{args.max_epoch}',
                     leave=False)

    while iter_num < max_iter:

        # Get batch (three augmentations)
        try:
            imgs, true_labels, tar_idx, path = next(iter_test)
        except StopIteration:
            iter_test = iter(dset_loaders["target"])
            imgs, true_labels, tar_idx, path = next(iter_test)

        inputs_weak, inputs_strong, inputs_original = imgs

        if inputs_weak.size(0) == 1:
            iter_num += 1
            continue

        # Learning rate scheduling + gradient zeroing
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        optimizer.zero_grad()

        # GPU transfer
        inputs_weak     = inputs_weak.cuda()
        inputs_strong   = inputs_strong.cuda()
        inputs_original = inputs_original.cuda()

        if iter_num % 100 == 0:
            torch.cuda.empty_cache()

        # Feature extraction (CLIP Encoder frozen, no gradients)
        with torch.no_grad():
            # Student uses Strong, Teacher uses Weak, IM Loss uses Original
            feat_512_weak     = netF(inputs_weak.type(clip_dtype))      # [B, 512]
            feat_512_strong   = netF(inputs_strong.type(clip_dtype))    # [B, 512]
            feat_512_original = netF(inputs_original.type(clip_dtype))  # [B, 512]

        # Student: ProjectorC Forward (Strong aug) -> CE / Propagation Loss
        feat_C_strong = netP_C(feat_512_strong)                           # [B, 512], has gradient
        feat_stu_norm = F.normalize(feat_C_strong, dim=-1)

        # Use global centering to compute Student logits (aligned with Teacher)
        centered_student_features = F.normalize(feat_stu_norm - target_global_mean, dim=-1)  # [B, D]
        logits_student = args.centroid_logitScale * (centered_student_features @ centered_source_centroids.t())  # [B, C]

        # Student: ProjectorC Forward (Original aug) -> IM Loss
        feat_C_original = netP_C(feat_512_original)                       # [B, 512], has gradient
        feat_orig_norm  = F.normalize(feat_C_original, dim=-1)

        # Use global centering to compute Original logits
        centered_original_features = F.normalize(feat_orig_norm - target_global_mean, dim=-1)  # [B, D]
        logits_original = args.centroid_logitScale * (centered_original_features @ centered_source_centroids.t())  # [B, C]

        # Student: ProjectorC -> ProjectorH (Weak + Strong) -> Contrastive Loss
        feat_C_weak   = netP_C(feat_512_weak)                             # [B, 512]
        feat_H_weak   = netP_H(feat_C_weak)                               # [B, 512], chained
        feat_H_strong = netP_H(feat_C_strong)                             # [B, 512], chained (reused)

        # Teacher: ProjectorC_ema Forward (Original) -> compute Pseudo Labels in real-time
        # Use Original (no augmentation) for all epochs
        with torch.no_grad():
            feat_tea_input = netP_C_ema(feat_512_original)                # [B, 512]
            feat_tea_norm = F.normalize(feat_tea_input, dim=-1)

            # Update global mean with batch EMA (avoids scanning entire dataset each batch)
            batch_mean = feat_tea_norm.mean(dim=0, keepdim=True)
            target_global_mean = F.normalize(target_global_mean * args.ema_m + batch_mean * (1 - args.ema_m), dim=-1)

            # Use updated target_global_mean for global centering
            centered_batch_features = F.normalize(feat_tea_norm - target_global_mean, dim=-1)  # [B, D]

            # Compute similarity using centered source centroids
            logits_teacher = args.centroid_logitScale * (centered_batch_features @ centered_source_centroids.t())  # [B, C]

            # Compute pseudo labels for current batch in real-time
            probs_teacher = F.softmax(logits_teacher, dim=1)              # [B, C]
            batch_max_probs, batch_pseudo_labels = torch.max(probs_teacher, dim=1)  # [B]

            # Compute entropy weight dynamically (based on latest Teacher predictions)
            entropy_batch = loss.Entropy(probs_teacher)                   # [B]
            batch_mas     = 1.0 - torch.exp(-entropy_batch)               # [B], high entropy->high weight

        # Confidence mask (based on real-time computed pseudo labels)
        # High confidence (Reliable): direct CE Loss
        reliable_mask = batch_max_probs > args.conf_thres   # [B] bool

        # Low confidence (Unreliable): all non-high-confidence samples
        unreliable_mask = ~reliable_mask                   # [B] bool

        losses = torch.tensor(0.0).cuda()

        # [A] CE Loss: high-confidence pseudo-label samples
        ce_loss = torch.tensor(0.0).cuda()
        reliable_count = reliable_mask.sum().item()

        if reliable_count > 0:
            ce_loss = smoothed_cross_entropy(
                logits_student[reliable_mask],
                batch_pseudo_labels[reliable_mask],
                num_classes=args.class_num,
                epsilon=0.1
            ) * args.cls_par
        losses += ce_loss

        # [B] Propagation Loss: low-confidence samples approach Teacher logit distribution (MSE)
        prop_loss = torch.tensor(0.0).cuda()
        if unreliable_mask.sum() > 0 and args.prop_par > 0:
            prop_loss = F.mse_loss(
                logits_student[unreliable_mask],
                logits_teacher[unreliable_mask].detach()
            ) * args.prop_par

        losses += prop_loss

        # [C] Information Loss: entropy minimization (all samples, using Original image)
        im_loss = torch.tensor(0.0).cuda()
        if args.ent:
            softmax_out  = F.softmax(logits_original, dim=1)  # Use Original image logits
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax      = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss  -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
        losses += im_loss

        # [D] Contrastive Loss: SimCLR on Weak + Strong features (with entropy weight)
        out_1 = F.normalize(feat_H_weak,   dim=-1)   # [B, 512], through netP_C -> netP_H
        out_2 = F.normalize(feat_H_strong, dim=-1)   # [B, 512], through netP_C -> netP_H
        out   = torch.cat([out_1, out_2], dim=0)      # [2B, 512]

        # Similarity matrix (remove diagonal self-similarity)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.tt)  # [2B, 2B]
        diag_mask  = (torch.ones_like(sim_matrix) -
                      torch.eye(out.shape[0], device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(diag_mask).view(out.shape[0], -1)  # [2B, 2B-1]

        # Positive pairs (Weak -> Strong and Strong -> Weak)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.tt)  # [B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)                    # [2B]

        # Entropy weighting (high entropy samples -> need more contrastive supervision)
        micro_mas = torch.cat([batch_mas, batch_mas])                     # [2B]
        contrast_loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1)) * micro_mas).mean()
        contrast_loss = contrast_loss * args.con_par
        losses += contrast_loss

        # Backward pass
        losses.backward()

        # Accumulate epoch loss (for end-of-epoch statistics)
        epoch_ce_loss       += ce_loss.item()
        epoch_prop_loss     += prop_loss.item()
        epoch_entropy_loss  += im_loss.item()
        epoch_contrast_loss += contrast_loss.item()
        epoch_total_loss    += losses.item()
        epoch_batches       += 1

        # Update Student parameters
        optimizer.step()

        # EMA update Teacher ProjectorC (only update classification projector)
        with torch.no_grad():
            for param_q, param_k in zip(netP_C.parameters(), netP_C_ema.parameters()):
                param_k.data = param_k.data * args.ema_m + param_q.data * (1.0 - args.ema_m)

        # Progress bar update
        pbar.update(1)
        pbar.set_postfix({
            'Total': f'{losses.item():.4f}',
            'CE':    f'{ce_loss.item():.4f}',
            'Prop':  f'{prop_loss.item():.4f}',
            'Ent':   f'{im_loss.item():.4f}',
            'Con':   f'{contrast_loss.item():.4f}'
        })

        # wandb logging (per batch)
        if args.use_wandb:
            wandb.log({
                'batch/total_loss': losses.item(),
                'batch/ce_loss': ce_loss.item(),
                'batch/prop_loss': prop_loss.item(),
                'batch/entropy_loss': im_loss.item(),
                'batch/contrast_loss': contrast_loss.item(),
                'batch/reliable_count': reliable_count,
                'batch/reliable_ratio': reliable_count / inputs_weak.size(0),
                'batch/conf_threshold': args.conf_thres,
                'batch/lr': optimizer.param_groups[0]['lr'],
                'iter': iter_num,
            })

        # Epoch end: evaluation + save
        if (iter_num - epoch_start_iter + 1) % batches_per_epoch == 0:
            pbar.close()

            avg_total    = epoch_total_loss    / epoch_batches if epoch_batches > 0 else 0
            avg_ce       = epoch_ce_loss       / epoch_batches if epoch_batches > 0 else 0
            avg_prop     = epoch_prop_loss     / epoch_batches if epoch_batches > 0 else 0
            avg_entropy  = epoch_entropy_loss  / epoch_batches if epoch_batches > 0 else 0
            avg_contrast = epoch_contrast_loss / epoch_batches if epoch_batches > 0 else 0

            # Evaluate accuracy using ProjectorC (eval)
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netP_C,
                                  class_centroids, target_global_mean, source_global_mean, args,
                                  flag=False)

            # Monitor pseudo-label quality per epoch (using Centroids)
            pseudo_label_stats = None
            if current_epoch % 1 == 0:
                _, _, pseudo_label_stats = obtain_label_clip(
                    dset_loaders['test'], netF, netP_C_ema, class_centroids, args, current_epoch
                )

            teacher_logit_scale  = args.centroid_logitScale

            log_str = (
                f'Epoch: {current_epoch+1}/{args.max_epoch}  |  Accuracy = {acc_s_te:.2f}%  |\n'
                f'  Avg Loss: {avg_total:.4f}  '
                f'CE: {avg_ce:.4f}  '
                f'Prop: {avg_prop:.4f}  '
                f'Ent: {avg_entropy:.4f}  '
                f'Con: {avg_contrast:.4f}\n'
                f'  Loss Weights: cls={args.cls_par:.4f}, ent={args.ent_par:.4f}, '
                f'con={args.con_par:.4f}, prop={args.prop_par:.4f}\n'
                f'  Teacher Strategy: Aug=Original, logitScale={teacher_logit_scale}, Threshold={args.conf_thres}'
            )
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()

            # wandb logging (per epoch)
            epoch_log = {
                'epoch/total_loss': avg_total,
                'epoch/CE_loss': avg_ce,
                'epoch/prop_loss': avg_prop,
                'epoch/IM_loss': avg_entropy,
                'epoch/contrast_loss': avg_contrast,
                'epoch/accuracy': acc_s_te,
                'epoch/best_accuracy': acc_init,
                'epoch': current_epoch + 1,
            }

            # Add pseudo label statistics if available
            if pseudo_label_stats is not None:
                epoch_log.update({
                    'epoch/pseudo_label_accuracy': pseudo_label_stats['pseudo_label_accuracy'] * 100,
                    'epoch/reliable_count': pseudo_label_stats['reliable_count'],
                    'epoch/reliable_ratio': pseudo_label_stats['reliable_ratio'] * 100,
                    'epoch/reliable_purity': pseudo_label_stats['reliable_purity'] * 100,
                    'epoch/total_samples': pseudo_label_stats['total_samples'],
                })

            if args.use_wandb:
                wandb.log(epoch_log)

            # Save current model (overwrite each epoch)
            torch.save(netP_C.state_dict(),
                       osp.join(args.output_dir, "target_ProjectorC_current.pt"))
            torch.save(netP_H.state_dict(),
                       osp.join(args.output_dir, "target_ProjectorH_current.pt"))
            torch.save(netP_C_ema.state_dict(),
                       osp.join(args.output_dir, "target_ProjectorC_ema_current.pt"))

            # Save best model (when accuracy reaches new high)
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                torch.save(netP_C.state_dict(),
                           osp.join(args.output_dir, "target_ProjectorC_best.pt"))
                torch.save(netP_H.state_dict(),
                           osp.join(args.output_dir, "target_ProjectorH_best.pt"))
                torch.save(netP_C_ema.state_dict(),
                           osp.join(args.output_dir, "target_ProjectorC_ema_best.pt"))
                print(f'New best accuracy: {acc_s_te:.2f}%, Best Model Saved')

            # Reset epoch tracking variables
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

    # Close last progress bar
    if 'pbar' in locals():
        pbar.close()

    # Close wandb
    if args.use_wandb:
        wandb.finish()
        print('wandb run finished')

    return netP_C, netP_H, netP_C_ema


# Main program entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP Projector EMA SFDA')

    # Basic configuration
    parser.add_argument('--gpu_id',   type=str, nargs='?', default='0',  help='device id to run')
    parser.add_argument('--s',        type=int,  default=0,   help='source domain index')
    parser.add_argument('--t',        type=int,  default=1,   help='target domain index')
    parser.add_argument('--max_epoch',type=int,  default=5,   help='max training epochs')
    parser.add_argument('--interval', type=int,  default=20,  help='pseudo-label update interval')
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--worker',   type=int,  default=4)
    parser.add_argument('--seed',     type=int,  default=2022)

    # Dataset
    parser.add_argument('--dset', type=str, default='M58',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'M58'])

    # CLIP configuration
    parser.add_argument('--net',           type=str,   default='RN101',
                        help='CLIP backbone (RN50, RN101, ViT-B/32, etc.)')
    parser.add_argument('--centroid_logitScale',type=float, default=8.1,
                        help='logit scale for centroid similarity (for pseudo label generation)')
    parser.add_argument('--classnames_path',type=str,
                        default='/mnt/backups/andycw/M58/classname37.txt',
                        help='Path to class names file (one per line)')
    parser.add_argument('--centroid_path',type=str,
                        default='/mnt/backups/andycw/UDA-AI/output/m58/PureCLIP_Source_Model/RN101_projector_originCLIP_ep50_LR0.01_37classes_Standalone/source_class_centroids.pth',
                        help='Path to Class Centroids file (class centers extracted from Source)')

    # Projector path (direct path overrides output_dir_src/source_P.pt)
    parser.add_argument('--projector_path', type=str,
                        default='/mnt/backups/andycw/UDA-AI/output/m58/PureCLIP_Source_Model/RN101_projector_originCLIP_ep50_LR0.01_37classes_Standalone/source_projector.pt',
                        help='Direct path to Projector pre-trained weights')

    # EMA
    parser.add_argument('--ema_m', type=float, default=0.99,
                        help='Teacher EMA momentum (larger->more stable)')

    # Loss weights
    parser.add_argument('--gent',     type=bool,  default=True,
                        help='Whether to use diversity loss (gentropy term of IM Loss)')
    parser.add_argument('--ent',      type=bool,  default=True,
                        help='Whether to use Information Loss (entropy minimization)')
    parser.add_argument('--cls_par',  type=float, default=0.3,
                        help='[A] CE Loss weight')
    parser.add_argument('--prop_par', type=float, default=0.5,
                        help='[B] Propagation Loss weight')
    parser.add_argument('--ent_par',  type=float, default=1.0,
                        help='[C] Information Loss weight')

    # Contrastive Loss temperature
    parser.add_argument('--tt',       type=float, default=0.05,
                        help='[D] Contrastive Loss temperature (SimCLR)')

    # Pseudo-label confidence threshold
    parser.add_argument('--conf_thres',type=float, default=0.5,
                        help='Pseudo-label confidence threshold (> conf_thres = reliable)')

    # Output paths
    parser.add_argument('--output',     type=str, default='ckps/target/')
    parser.add_argument('--output_src', type=str, default='ckps/source/')
    parser.add_argument('--da',         type=str, default='uda',
                        choices=['uda', 'pda'])

    # Others
    parser.add_argument('--epsilon',            type=float, default=1e-5,
                        help='Epsilon for entropy calculation numerical stability')
    parser.add_argument('--lr',                 type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--issave',             type=bool,  default=True)

    # Resume parameters
    parser.add_argument('--resume_dir',         type=str,   default=None,
                        help='Model directory for resume training (containing _current.pt files)')
    parser.add_argument('--start_epoch',        type=int,   default=0,
                        help='Starting epoch for resume (check log to find)')

    # Dynamic parameter adjustment
    parser.add_argument('--con_par',            type=float, default=None,
                        help='Contrastive Loss weight (if specified, overrides calculation)')

    # wandb configuration
    parser.add_argument('--use_wandb',          type=bool,  default=False,
                        help='Whether to use wandb for logging')

    args = parser.parse_args()

    # Dataset mapping configuration
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

    # Environment and random seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Data paths
    folder = 'data/'
    if args.dset == 'M58':
        args.s_dset_path    = folder + args.dset + '/' + names[args.s] + '_37_list.txt'
        args.t_dset_path    = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
    else:
        args.s_dset_path    = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path    = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    # Output directory
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

    # Log file
    args.savename = (
        f'conf_{args.conf_thres}_cls_{args.cls_par}_prop_{args.prop_par}_ema_{args.ema_m}'
    )
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    train_target(args)