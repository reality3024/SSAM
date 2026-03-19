"""
诊断 Image Features 和 Text Features 的相似度
检查训练是否真的拉近了两者
"""
import os
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

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


def image_transform(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def load_clip_to_cpu(backbone):
    model, _ = clip.load(backbone, device="cpu", download_root=os.path.expanduser("~/.cache/clip"))
    return model


def load_classnames(classname_file):
    with open(classname_file, 'r', encoding='utf-8') as f:
        classnames = [line.strip() for line in f.readlines()]
    return classnames


def load_projector(projector_path, device='cuda'):
    projector = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).to(device)
    state_dict = torch.load(projector_path, map_location=device)
    projector.load_state_dict(state_dict)
    projector.eval()
    return projector


def create_data_loader(list_path, batch_size=64):
    txt = open(list_path).readlines()
    root_path = '/mnt/backups/andycw/'
    transform = image_transform()
    dataset = ImageList_idx(txt, transform=transform, root=root_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader


def diagnose_similarity(clip_model, projector, data_loader, classnames):
    """
    診斷 image features 和 text features 的相似度
    """
    print(f"\n{'='*80}")
    print("開始診斷特徵相似度")
    print(f"{'='*80}\n")
    
    clip_model.eval()
    projector.eval()
    
    # 1. 生成 Text Features（測試兩種 Prompt）
    print("1. 生成 Text Features...")
    
    # 方法 1: 原始方法 "A photo of a {classname}"
    print("\n   【方法 1】使用 'A photo of a {classname}'")
    prompts_v1 = [f"A photo of a {classname}" for classname in classnames]
    prompts_tokens_v1 = clip.tokenize(prompts_v1).cuda()
    
    with torch.no_grad():
        text_features_v1 = clip_model.encode_text(prompts_tokens_v1)
        text_features_v1 = text_features_v1 / text_features_v1.norm(dim=-1, keepdim=True)
    
    # 計算 v1 的 text-text 相似度
    text_similarity_v1 = text_features_v1 @ text_features_v1.t()
    mask = torch.eye(len(text_features_v1), device=text_similarity_v1.device).bool()
    text_sim_v1_off_diag = text_similarity_v1[~mask]
    
    print(f"   ✓ Text features shape: {text_features_v1.shape}")
    print(f"   ✓ Text-Text 平均相似度: {text_sim_v1_off_diag.mean().item():.4f}")
    print(f"   ✓ Text-Text 最大相似度: {text_sim_v1_off_diag.max().item():.4f}")
    
    # 方法 2: 直接用 classname（不加前綴）
    print("\n   【方法 2】直接使用 classname（不加前綴）")
    prompts_v2 = classnames  # 直接用類別名
    prompts_tokens_v2 = clip.tokenize(prompts_v2).cuda()
    
    with torch.no_grad():
        text_features_v2 = clip_model.encode_text(prompts_tokens_v2)
        text_features_v2 = text_features_v2 / text_features_v2.norm(dim=-1, keepdim=True)
    
    # 計算 v2 的 text-text 相似度
    text_similarity_v2 = text_features_v2 @ text_features_v2.t()
    text_sim_v2_off_diag = text_similarity_v2[~mask]
    
    print(f"   ✓ Text features shape: {text_features_v2.shape}")
    print(f"   ✓ Text-Text 平均相似度: {text_sim_v2_off_diag.mean().item():.4f}")
    print(f"   ✓ Text-Text 最大相似度: {text_sim_v2_off_diag.max().item():.4f}")
    
    # 比較
    diff = text_sim_v1_off_diag.mean().item() - text_sim_v2_off_diag.mean().item()
    print(f"\n   📊 Text-Text 相似度差異: {diff:.4f} ({'方法2更分散' if diff > 0 else '方法1更分散'})")
    
    # 使用方法 2 繼續分析（因為我們想測試這個）
    text_features = text_features_v2
    print(f"\n   ➡️  後續分析使用【方法 2】")
    print(f"   ✓ Text features 已歸一化: {text_features.norm(dim=-1).mean().item():.4f}")
    
    # 2. 收集 Image Features
    print("\n2. 收集 Image Features (前 10 個 batch)...")
    all_img_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, (imgs, lbls, _, _) in enumerate(data_loader):
            if i >= 10:  # 只取前 10 個 batch
                break
            
            imgs = imgs.cuda()
            
            # 提取特徵 (與訓練時完全一致)
            feat_clip = clip_model.visual(imgs.type(clip_model.dtype))
            feat_projected = projector(feat_clip)
            feat_projected = feat_projected / feat_projected.norm(dim=-1, keepdim=True)
            
            all_img_features.append(feat_projected)
            all_labels.append(lbls)
    
    all_img_features = torch.cat(all_img_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"   ✓ 收集了 {len(all_img_features)} 個 image features")
    print(f"   ✓ Image features 已歸一化: {all_img_features.norm(dim=-1).mean().item():.4f}")
    
    # 3. 計算相似度矩陣
    print("\n3. 計算 Image-Text 相似度...")
    similarity_matrix = all_img_features @ text_features.t()  # [N, 79]
    
    # 4. 統計結果
    print(f"\n{'='*80}")
    print("統計結果")
    print(f"{'='*80}")
    
    # 4.1 整體統計
    print(f"\n【整體相似度】")
    print(f"  最大相似度: {similarity_matrix.max().item():.4f}")
    print(f"  最小相似度: {similarity_matrix.min().item():.4f}")
    print(f"  平均相似度: {similarity_matrix.mean().item():.4f}")
    print(f"  標準差: {similarity_matrix.std().item():.4f}")
    
    # 4.2 正確類別的相似度
    correct_similarity = similarity_matrix[torch.arange(len(all_labels)), all_labels]
    print(f"\n【正確類別的相似度】(應該較高)")
    print(f"  平均相似度: {correct_similarity.mean().item():.4f}")
    print(f"  最小相似度: {correct_similarity.min().item():.4f}")
    print(f"  最大相似度: {correct_similarity.max().item():.4f}")
    
    # 4.3 最高預測類別的相似度
    predicted_labels = similarity_matrix.argmax(dim=1)
    max_similarity = similarity_matrix.max(dim=1)[0]
    accuracy = (predicted_labels == all_labels.cuda()).float().mean().item()
    
    print(f"\n【預測結果】")
    print(f"  準確率: {accuracy * 100:.2f}%")
    print(f"  預測類別的平均相似度: {max_similarity.mean().item():.4f}")
    
    # 4.4 查看是否拉近了
    # 對於每個樣本，正確類別的相似度排名
    ranks = (similarity_matrix >= correct_similarity.unsqueeze(1)).sum(dim=1)
    print(f"\n【正確類別的排名】")
    print(f"  排名第1的樣本數: {(ranks == 1).sum().item()} / {len(ranks)} ({(ranks == 1).float().mean().item()*100:.1f}%)")
    print(f"  排名前3的樣本數: {(ranks <= 3).sum().item()} / {len(ranks)} ({(ranks <= 3).float().mean().item()*100:.1f}%)")
    print(f"  排名前5的樣本數: {(ranks <= 5).sum().item()} / {len(ranks)} ({(ranks <= 5).float().mean().item()*100:.1f}%)")
    print(f"  平均排名: {ranks.float().mean().item():.2f}")
    
    # 4.5 分析距離
    print(f"\n【L2 距離分析】(歸一化後，L2 距離 = sqrt(2 - 2*cosine_sim))")
    distances = torch.sqrt(2 - 2 * correct_similarity)
    print(f"  正確類別的平均 L2 距離: {distances.mean().item():.4f}")
    print(f"  最小 L2 距離: {distances.min().item():.4f}")
    print(f"  最大 L2 距離: {distances.max().item():.4f}")
    
    print(f"\n💡 解讀：")
    print(f"  - 如果準確率高(>80%)但相似度低(<0.3)，說明特徵空間可能有問題")
    print(f"  - 如果準確率和相似度都低，說明需要更多訓練")
    print(f"  - 理想情況：準確率>90%，正確類別相似度>0.5")
    print(f"  - Cosine similarity 範圍: [-1, 1]，0.3 對應約 72° 角度")
    
    # 4.6 檢查 image features 之間的相似度（診斷是否真的是 projector 在做 cluster）
    print(f"\n【Image Features 之間的相似度】")
    # 對於每個類別，計算類內相似度
    unique_labels = torch.unique(all_labels)
    intra_class_sims = []
    inter_class_sims = []
    
    for label in unique_labels:
        mask_label = (all_labels == label)
        if mask_label.sum() > 1:  # 至少要有 2 個樣本
            class_features = all_img_features[mask_label.cuda()]
            # 計算類內相似度
            class_sim = class_features @ class_features.t()
            # 去除對角線
            class_mask = ~torch.eye(len(class_features), device=class_sim.device).bool()
            intra_class_sims.append(class_sim[class_mask].mean().item())
    
    # 計算類間相似度（隨機取樣）
    for _ in range(100):  # 取 100 個樣本對
        idx1, idx2 = torch.randint(0, len(all_img_features), (2,))
        if all_labels[idx1] != all_labels[idx2]:
            sim = (all_img_features[idx1] @ all_img_features[idx2]).item()
            inter_class_sims.append(sim)
    
    intra_class_mean = sum(intra_class_sims) / len(intra_class_sims) if intra_class_sims else 0
    inter_class_mean = sum(inter_class_sims) / len(inter_class_sims) if inter_class_sims else 0
    
    print(f"  類內平均相似度 (同類別): {intra_class_mean:.4f}")
    print(f"  類間平均相似度 (不同類別): {inter_class_mean:.4f}")
    print(f"  類內/類間比值: {intra_class_mean / inter_class_mean if inter_class_mean > 0 else 0:.2f}")
    print(f"  💡 比值越大，說明 Projector 的 clustering 效果越好")
    
    print(f"\n{'='*80}\n")


def main():
    print("\n" + "="*80)
    print("特徵相似度診斷工具")
    print("="*80 + "\n")
    
    # 設定路徑
    source_list = '/mnt/backups/andycw/M58/CAD_ratioFilter_StableDiffusion_random_list.txt'
    classname_file = '/mnt/backups/andycw/M58/classname79.txt'
    projector_path = '/mnt/backups/andycw/CLIP/output/m58/PureCLIP_Source_Model/rn101_projector_originCLIP_ep50_LR0.01_random/source_projector.pt'
    
    # 載入類別名稱
    print("載入類別名稱...")
    classnames = load_classnames(classname_file)
    print(f"  ✓ {len(classnames)} 個類別\n")
    
    # 載入 CLIP 模型
    print("載入 CLIP 模型...")
    clip_model = load_clip_to_cpu(backbone='RN101')
    clip_model.float()
    clip_model.cuda()
    clip_model.eval()
    print(f"  ✓ CLIP RN101 已載入\n")
    
    # 載入 Projector
    print("載入 Projector...")
    projector = load_projector(projector_path, device='cuda')
    print(f"  ✓ Projector 已載入\n")
    
    # 創建 DataLoader
    print("創建 DataLoader...")
    data_loader = create_data_loader(source_list, batch_size=64)
    print(f"  ✓ DataLoader 已創建\n")
    
    # 診斷相似度
    diagnose_similarity(clip_model, projector, data_loader, classnames)
    
    print("="*80)
    print("✅ 診斷完成")
    print("="*80)


if __name__ == "__main__":
    main()
