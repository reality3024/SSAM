import argparse
import os
import os.path as osp
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import clip
from data_list import ImageList_idx


class MLPProjector(nn.Module):
    """Projector architecture aligned with projector_ema / contrast_clip_projector_ema."""

    def __init__(self, in_dim=512, out_dim=512, hidden_dim=1024):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_default_target_list(dset, t):
    if dset == "M58":
        names = ["CAD_ratioFilter", "Real_all_nobg"]
        return osp.join(PROJECT_ROOT, "data", dset, f"{names[t]}_37_list.txt")

    mapping = {
        "office-home": ["Art", "Clipart", "Product", "RealWorld"],
        "office": ["amazon", "dslr", "webcam"],
        "VISDA-C": ["train", "validation"],
        "office-caltech": ["amazon", "caltech", "dslr", "webcam"],
    }
    if dset not in mapping:
        raise ValueError(f"Unsupported dset: {dset}")
    return osp.join(PROJECT_ROOT, "data", dset, f"{mapping[dset][t]}_list.txt")


def build_target_loader(target_list_path, dset, preprocess, batch_size, workers):
    with open(target_list_path, "r") as f:
        txt_tar = f.readlines()

    root_path = osp.join(PROJECT_ROOT, "data", dset)
    dataset = ImageList_idx(txt_tar, transform=preprocess, root=root_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    return loader


def extract_original_features_with_labels(netF, netP, loader, device, max_samples=None):
    netF.eval()
    netP.eval()
    clip_dtype = next(netF.parameters()).dtype

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            labels = batch[1]  # Get ground truth labels
            feat_512 = netF(inputs.type(clip_dtype))
            feat_projected = netP(feat_512)
            feat_norm = F.normalize(feat_projected, dim=-1)
            all_features.append(feat_norm.cpu())
            all_labels.append(labels)

            if max_samples is not None:
                current_count = sum(x.size(0) for x in all_features)
                if current_count >= max_samples:
                    break

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if max_samples is not None and features.size(0) > max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
    return features, labels


def compute_feature_sets(class_centroids, original_features, use_global_centering):
    class_centroids = class_centroids.float()
    original_features = original_features.float()

    if use_global_centering:
        source_global_mean = class_centroids.mean(dim=0, keepdim=True)
        target_global_mean = original_features.mean(dim=0, keepdim=True)

        source_features = F.normalize(class_centroids - source_global_mean, dim=-1)
        target_features = F.normalize(original_features - target_global_mean, dim=-1)
    else:
        source_features = F.normalize(class_centroids, dim=-1)
        target_features = F.normalize(original_features, dim=-1)

    return source_features, target_features


def run_pca(source_features, target_features, seed):
    n_source = source_features.size(0)
    n_target = target_features.size(0)
    all_features = torch.cat([source_features, target_features], dim=0).cpu().numpy()
    
    # 換成 PCA
    pca = PCA(n_components=2, random_state=seed)
    embedding = pca.fit_transform(all_features)
    
    source_emb = embedding[:n_source]
    target_emb = embedding[n_source:n_source + n_target]
    return source_emb, target_emb

def run_tsne(source_features, target_features, seed, perplexity):
    n_source = source_features.size(0)
    n_target = target_features.size(0)

    all_features = torch.cat([source_features, target_features], dim=0).cpu().numpy()

    max_perplexity = max(5, min(perplexity, all_features.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=max_perplexity,
    )
    embedding = tsne.fit_transform(all_features)

    source_emb = embedding[:n_source]
    target_emb = embedding[n_source:n_source + n_target]
    return source_emb, target_emb


def plot_tsne(source_emb, target_emb, title, out_path):
    plt.figure(figsize=(10, 8), dpi=220)

    plt.scatter(
        target_emb[:, 0],
        target_emb[:, 1],
        s=10,
        alpha=0.45,
        c="#2a9d8f",
        label="target original/centered_original",
        linewidths=0,
    )
    plt.scatter(
        source_emb[:, 0],
        source_emb[:, 1],
        s=80,
        alpha=0.95,
        c="#e76f51",
        marker="^",
        edgecolors="white",
        linewidths=0.7,
        label="source centroids/centered_source_centroids",
    )

    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compute_prediction_accuracy(class_centroids, original_features, target_true_labels=None, logit_scale=8.1):
    """
    Compare prediction accuracy with and without Global Centering.
    
    Mimics the logic from contrast_clip_projector_ema.py:
    - logits = centroid_logitScale * (centered_features @ centered_source_centroids.t())
    - probs = softmax(logits)
    - pred_labels = argmax(probs)
    
    Args:
        class_centroids: [C, 512] Source class centroids
        original_features: [N, 512] Target features
        target_true_labels: [N] True labels (optional, for accuracy computation)
        logit_scale: Scaling factor for logits (default 8.1, from contrast_clip_projector_ema.py)
    
    Returns:
        dict containing:
        - 'pred_no_gc': [N] Predicted labels without GC
        - 'pred_with_gc': [N] Predicted labels with GC
        - 'probs_no_gc': [N, C] Softmax probabilities without GC
        - 'probs_with_gc': [N, C] Softmax probabilities with GC
        - 'max_probs_no_gc': [N] Max softmax scores without GC
        - 'max_probs_with_gc': [N] Max softmax scores with GC
        - 'agreement_ratio': How many predictions match between GC and no-GC
        - 'accuracy_no_gc': (optional) Accuracy without GC if true_labels provided
        - 'accuracy_with_gc': (optional) Accuracy with GC if true_labels provided
        - 'accuracy_diff': (optional) Accuracy difference (with_gc - no_gc)
    """
    class_centroids = class_centroids.float()
    original_features = original_features.float()
    
    # ========== WITHOUT Global Centering ==========
    source_features_no_gc = F.normalize(class_centroids, dim=-1)  # [C, 512]
    target_features_no_gc = F.normalize(original_features, dim=-1)  # [N, 512]
    
    # Compute logits (from ema.py style)
    logits_no_gc = logit_scale * (target_features_no_gc @ source_features_no_gc.t())  # [N, C]
    probs_no_gc = F.softmax(logits_no_gc, dim=1)  # [N, C]
    max_probs_no_gc, pred_labels_no_gc = torch.max(probs_no_gc, dim=1)  # [N]
    
    # ========== WITH Global Centering ==========
    source_global_mean = class_centroids.mean(dim=0, keepdim=True)  # [1, 512]
    target_global_mean = original_features.mean(dim=0, keepdim=True)  # [1, 512]
    
    source_features_with_gc = F.normalize(class_centroids - source_global_mean, dim=-1)  # [C, 512]
    target_features_with_gc = F.normalize(original_features - target_global_mean, dim=-1)  # [N, 512]
    
    logits_with_gc = logit_scale * (target_features_with_gc @ source_features_with_gc.t())  # [N, C]
    probs_with_gc = F.softmax(logits_with_gc, dim=1)  # [N, C]
    max_probs_with_gc, pred_labels_with_gc = torch.max(probs_with_gc, dim=1)  # [N]
    
    # ========== Compute Statistics ==========
    results = {
        'pred_no_gc': pred_labels_no_gc.cpu(),
        'pred_with_gc': pred_labels_with_gc.cpu(),
        'probs_no_gc': probs_no_gc.cpu(),
        'probs_with_gc': probs_with_gc.cpu(),
        'max_probs_no_gc': max_probs_no_gc.cpu(),
        'max_probs_with_gc': max_probs_with_gc.cpu(),
    }
    
    # Agreement ratio between GC and no-GC predictions
    agreement = (pred_labels_no_gc == pred_labels_with_gc).float().mean().item()
    results['agreement_ratio'] = agreement
    
    # If true labels provided, compute accuracy
    if target_true_labels is not None:
        target_true_labels = target_true_labels.cpu()
        
        acc_no_gc = (pred_labels_no_gc.cpu() == target_true_labels).float().mean().item()
        acc_with_gc = (pred_labels_with_gc.cpu() == target_true_labels).float().mean().item()
        
        results['accuracy_no_gc'] = acc_no_gc
        results['accuracy_with_gc'] = acc_with_gc
        results['accuracy_diff'] = acc_with_gc - acc_no_gc  # positive = GC improves accuracy
    
    return results


def plot_similarity_histogram(src_no, tar_no, src_yes, tar_yes, out_dir):
    # 1. 計算 Without GC 的最大相似度
    sim_no = tar_no @ src_no.t() # [N, 37]
    max_sim_no, _ = torch.max(sim_no, dim=1)
    
    # 2. 計算 With GC 的最大相似度
    sim_yes = tar_yes @ src_yes.t() # [N, 37]
    max_sim_yes, _ = torch.max(sim_yes, dim=1)
    
    # 3. 畫分佈直方圖
    plt.figure(figsize=(10, 6), dpi=200)
    
    # 沒有 GC 的分佈 (紅色)
    plt.hist(max_sim_no.numpy(), bins=50, range=(0, 1.0), alpha=0.6, 
             color='#e76f51', label='Without Global Centering', edgecolor='black', linewidth=0.5)
    
    # 有 GC 的分佈 (綠色)
    plt.hist(max_sim_yes.numpy(), bins=50, range=(0, 1.0), alpha=0.7, 
             color='#2a9d8f', label='With Global Centering', edgecolor='black', linewidth=0.5)
    
    # 畫一條你設定的 Threshold 線 (例如 0.9)
    plt.axvline(x=0.9, color='gray', linestyle='--', linewidth=2, label='Threshold = 0.9')
    
    plt.title('Max Cosine Similarity Distribution (Target to Source Centroids)', fontsize=14)
    plt.xlabel('Cosine Similarity Score (Unscaled)', fontsize=12)
    plt.ylabel('Frequency (Number of Target Samples)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out_path = osp.join(out_dir, "similarity_histogram_compare.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Done] Saved Similarity Histogram to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare t-SNE distributions with and without Global Centering."
    )
    parser.add_argument(
        "--projector-path",
        type=str,
        # default="/mnt/backups/andycw/UDA-AI/output/m58/PureCLIP_Source_Model/RN101_projector_originCLIP_ep50_LR0.01_37classes_Standalone/source_projector.pt",
        default="/mnt/backups/andycw/UDA-AI/output/visda/PureCLIP_Source_Model/RN101_ep50_lr0.01_New/source_projector.pt",
        help="Path to source_projector.pt (optional)",
    )
    parser.add_argument(
        "--centroid-path",
        type=str,
        default=None,
        help="Path to source class centroids (.pth). If omitted, infer from projector path.",
    )

    args = parser.parse_args()
    
    # Set fixed defaults for all non-explicit parameters
    # args.dset = "M58"
    args.dset = "VISDA-C"
    args.target_domain_index = 1
    args.clip_backbone = "RN101"
    args.batch_size = 256
    args.workers = 4
    args.max_target_samples = 3000
    args.perplexity = 40
    args.seed = 2022
    args.out_dir = osp.join(PROJECT_ROOT, "visual", "tsne_global_centering_compare")
    
    seed_everything(args.seed)

    if args.centroid_path is None:
        args.centroid_path = osp.join(osp.dirname(args.projector_path), "source_class_centroids.pth")

    args.target_list_path = get_default_target_list(args.dset, args.target_domain_index)

    # Resolve user-provided relative paths against project root so the script works
    # no matter where it is launched from.
    if not osp.isabs(args.target_list_path):
        args.target_list_path = osp.join(PROJECT_ROOT, args.target_list_path)
    if not osp.isabs(args.out_dir):
        args.out_dir = osp.join(PROJECT_ROOT, args.out_dir)

    if not osp.exists(args.projector_path):
        raise FileNotFoundError(f"Projector not found: {args.projector_path}")
    if not osp.exists(args.centroid_path):
        raise FileNotFoundError(f"Centroids not found: {args.centroid_path}")
    if not osp.exists(args.target_list_path):
        raise FileNotFoundError(f"Target list not found: {args.target_list_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")
    print(f"[Info] projector = {args.projector_path}")
    print(f"[Info] centroids = {args.centroid_path}")
    print(f"[Info] target list = {args.target_list_path}")

    clip_model, preprocess = clip.load(
        args.clip_backbone,
        device="cpu",
        download_root=os.path.expanduser("~/.cache/clip"),
    )
    clip_model.float()
    clip_model = clip_model.to(device)
    netF = clip_model.visual
    netF.eval()
    for p in netF.parameters():
        p.requires_grad = False

    netP = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).to(device)
    projector_state = torch.load(args.projector_path, map_location="cpu")
    netP.load_state_dict(projector_state, strict=True)
    netP.eval()

    centroid_data = torch.load(args.centroid_path, map_location="cpu")
    if "centroids" not in centroid_data:
        raise KeyError(f"'centroids' key not found in {args.centroid_path}")
    class_centroids = centroid_data["centroids"].float()

    target_loader = build_target_loader(
        target_list_path=args.target_list_path,
        dset=args.dset,
        preprocess=preprocess,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    original_features = extract_original_features_with_labels(
        netF=netF,
        netP=netP,
        loader=target_loader,
        device=device,
        max_samples=args.max_target_samples,
    )
    original_features, true_labels = original_features

    print(f"[Info] class_centroids shape: {tuple(class_centroids.shape)}")
    print(f"[Info] original_features shape: {tuple(original_features.shape)}")

    # 1) Without Global Centering
    src_no, tar_no = compute_feature_sets(
        class_centroids=class_centroids,
        original_features=original_features,
        use_global_centering=False,
    )
    src_no_emb, tar_no_emb = run_pca(
        source_features=src_no,
        target_features=tar_no,
        seed=args.seed,
    )
    out_no = osp.join(args.out_dir, "tsne_no_global_centering.png")
    plot_tsne(
        source_emb=src_no_emb,
        target_emb=tar_no_emb,
        title="t-SNE without Global Centering",
        out_path=out_no,
    )

    # 2) With Global Centering
    src_yes, tar_yes = compute_feature_sets(
        class_centroids=class_centroids,
        original_features=original_features,
        use_global_centering=True,
    )
    src_yes_emb, tar_yes_emb = run_pca(
        source_features=src_yes,
        target_features=tar_yes,
        seed=args.seed,
    )
    out_yes = osp.join(args.out_dir, "tsne_with_global_centering.png")
    plot_tsne(
        source_emb=src_yes_emb,
        target_emb=tar_yes_emb,
        title="t-SNE with Global Centering",
        out_path=out_yes,
    )

    print("[Done] Saved figures:")
    print(f"  - {out_no}")
    print(f"  - {out_yes}")

    plot_similarity_histogram(src_no, tar_no, src_yes, tar_yes, args.out_dir)

    # ========== Compute Prediction Accuracy Comparison ==========
    print("\n" + "="*80)
    print("Prediction Accuracy Comparison (Logit Scale = 10.0)")
    print("="*80)
    
    pred_results = compute_prediction_accuracy(
        class_centroids=class_centroids,
        original_features=original_features,
        target_true_labels=true_labels,
        logit_scale=10.0
    )
    
    print(f"\n[Accuracy against Ground Truth]")
    print(f"  Without Global Centering: {pred_results['accuracy_no_gc']*100:.2f}%")
    print(f"  With Global Centering:    {pred_results['accuracy_with_gc']*100:.2f}%")
    print(f"  Improvement (GC - No-GC): {pred_results['accuracy_diff']*100:+.2f}%")
    
    # Save accuracy to file
    out_stats_path = osp.join(args.out_dir, "accuracy_comparison.txt")
    with open(out_stats_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Prediction Accuracy Comparison (Global Centering vs No-GC)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Config:\n")
        f.write(f"  Logit Scale: 8.1 (from contrast_clip_projector_ema.py)\n")
        f.write(f"  Target samples: {original_features.size(0)}\n")
        f.write(f"  Source classes: {class_centroids.size(0)}\n\n")
        
        f.write(f"Accuracy against Ground Truth:\n")
        f.write(f"  Without Global Centering: {pred_results['accuracy_no_gc']*100:.2f}%\n")
        f.write(f"  With Global Centering:    {pred_results['accuracy_with_gc']*100:.2f}%\n")
        f.write(f"  Improvement (GC - No-GC): {pred_results['accuracy_diff']*100:+.2f}%\n")
    
    print(f"\n[Accuracy saved to]: {out_stats_path}")


if __name__ == "__main__":
    main()
