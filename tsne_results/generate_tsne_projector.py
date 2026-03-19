"""
生成 t-SNE 可視化圖
使用原始 CLIP Image Encoder + 訓練好的 Projector
Text Features 通過 "A photo of a [classname]" 模板動態生成
"""
import os
import os.path as osp
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 導入必要模組
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
    """標準的圖片預處理"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def load_clip_to_cpu(backbone):
    """載入 CLIP 模型到 CPU"""
    model, _ = clip.load(backbone, device="cpu", download_root=os.path.expanduser("~/.cache/clip"))
    return model


def load_classnames(classname_file):
    """從文件載入類別名稱"""
    with open(classname_file, 'r', encoding='utf-8') as f:
        classnames = [line.strip() for line in f.readlines()]
    return classnames


def create_data_loaders(source_list_path, target_list_path, batch_size=32):
    """創建 Source 和 Target 的 DataLoader"""
    txt_source = open(source_list_path).readlines()
    txt_target = open(target_list_path).readlines()
    
    root_path = '/mnt/backups/andycw/'
    transform = image_transform()
    
    source_dataset = ImageList_idx(txt_source, transform=transform, root=root_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0, drop_last=False)
    
    target_dataset = ImageList_idx(txt_target, transform=transform, root=root_path)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0, drop_last=False)
    
    print(f"✓ Source 數據: {len(source_dataset)} 張")
    print(f"✓ Target 數據: {len(target_dataset)} 張")
    
    return source_loader, target_loader


def load_projector(projector_path, device='cuda'):
    """載入訓練好的 Projector"""
    projector = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).to(device)
    state_dict = torch.load(projector_path, map_location=device)
    projector.load_state_dict(state_dict)
    projector.eval()
    return projector


def visualize_tsne(clip_model, projector, source_loader, target_loader, classnames, output_dir, filename='tsne.png'):
    """
    使用 CLIP image encoder + Projector 進行 t-SNE 視覺化
    
    Args:
        clip_model: 原始 CLIP 模型 (用於 image encoder 和 text encoder)
        projector: 訓練好的 MLP Projector
        source_loader: Source domain 的 dataloader (返回 img, label, idx, path)
        target_loader: Target domain 的 dataloader (返回 img, label, idx, path)
        classnames: 類別名稱列表
        output_dir: 輸出目錄
        filename: 輸出檔名
    """
    print(f"\n{'='*60}")
    print(f"開始繪製 t-SNE: {filename}")
    print(f"{'='*60}")
    
    # 設定為評估模式
    clip_model.eval()
    projector.eval()
    
    # 1. 生成 Text Features (使用 "A photo of a [classname]" 模板)
    print(f"  - 生成 Text Features (模板: 'A photo of a [classname]')...")
    prompts = [f"A photo of a {classname}" for classname in classnames]
    prompts_tokens = clip.tokenize(prompts).cuda()
    
    with torch.no_grad():
        text_features = clip_model.encode_text(prompts_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print(f"    ✓ Text Features shape: {text_features.shape}")
    
    features = []
    labels = []
    domains = []  # 'Source', 'Target', 'Text'
    
    # 2. 收集 Source Features (取部分樣本避免太慢)
    # print(f"  - 收集 Source Features...")
    # with torch.no_grad():
    #     for i, (imgs, lbls, _, _) in enumerate(source_loader):
    #         # if i > 5:  # 取前 6 個 batch
    #         #     break
    #         imgs = imgs.cuda()
            
    #         # 使用 CLIP image encoder + Projector
    #         feat_clip = clip_model.visual(imgs.type(clip_model.dtype))
    #         feat_projected = projector(feat_clip)
    #         feat_projected = feat_projected / feat_projected.norm(dim=-1, keepdim=True)
            
    #         features.append(feat_projected.cpu())
    #         labels.append(lbls)
    #         domains.extend(['Source'] * len(lbls))
    
    # print(f"    ✓ 收集了 {len([d for d in domains if d == 'Source'])} 個 Source 樣本")
    
    # 3. 收集 Target Features
    print(f"  - 收集 Target Features...")
    with torch.no_grad():
        for i, (imgs, lbls, _, _) in enumerate(target_loader):
            # if i > 5:  # 取前 6 個 batch
            #     break
            imgs = imgs.cuda()
            
            # 使用 CLIP image encoder + Projector
            feat_clip = clip_model.visual(imgs.type(clip_model.dtype))
            feat_projected = projector(feat_clip)
            feat_projected = feat_projected / feat_projected.norm(dim=-1, keepdim=True)
            
            features.append(feat_projected.cpu())
            labels.append(lbls)
            domains.extend(['Target'] * len(lbls))
    
    print(f"    ✓ 收集了 {len([d for d in domains if d == 'Target'])} 個 Target 樣本")
    
    # 4. 加入 Text Anchors
    print(f"  - 加入 Text Anchors ({len(text_features)} 個類別)...")
    # text_features 已经在生成时 normalize 过了，不需要重复 normalize
    features.append(text_features.cpu())
    labels.append(torch.arange(len(text_features)))
    domains.extend(['Text'] * len(text_features))
    
    # 5. 合併與降維
    print(f"  - 合併特徵並執行 t-SNE 降維...")
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    print(f"    ✓ Total samples: {len(features)} (Source + Target + Text)")
    
    # t-SNE 運算
    print(f"    ⏳ 執行 t-SNE (這可能需要一些時間)...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_embedded = tsne.fit_transform(features)
    print(f"    ✓ t-SNE 降維完成")
    
    # 6. 繪圖
    print(f"  - 繪製 t-SNE 視覺化圖...")
    df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Domain': domains,
        'Class': labels
    })
    
    plt.figure(figsize=(14, 12))
    
    # 畫 Source (小點，半透明)
    # source_data = df[df['Domain'] == 'Source']
    # if len(source_data) > 0:
    #     sns.scatterplot(
    #         data=source_data, x='x', y='y', hue='Class',
    #         palette='tab20', alpha=0.6, s=30, legend=False, marker='o'
    #     )
    
    # 畫 Target (叉，較明顯)
    target_data = df[df['Domain'] == 'Target']
    if len(target_data) > 0:
        sns.scatterplot(
            data=target_data, x='x', y='y', hue='Class',
            palette='tab20', alpha=0.6, s=50, legend=False, marker='X'
        )
    
    # 畫 Text (黑色星號，最明顯)
    # text_data = df[df['Domain'] == 'Text']
    # if len(text_data) > 0:
    #     sns.scatterplot(
    #         data=text_data, x='x', y='y',
    #         color='black', alpha=1.0, s=200, marker='*', label='Text Anchors'
    #     )
    
    plt.title("t-SNE Visualization: Source vs Target vs Text Anchors", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"    ✓ t-SNE 圖已保存: {save_path}")
    print(f"{'='*60}\n")


def main():
    """主函數：生成 t-SNE 圖"""
    print("\n" + "="*60)
    print("開始生成 t-SNE 可視化圖")
    print("使用: CLIP Image Encoder + Trained Projector")
    print("="*60 + "\n")
    
    # ===== 設定路徑 =====
    source_list = '/mnt/backups/andycw/M58/CAD_ratioFilter_StableDiffusion_random_list.txt'
    target_list = '/mnt/backups/andycw/M58/Real_all_nobg_list.txt'
    classname_file = '/mnt/backups/andycw/M58/classname79.txt'
    
    # Projector 權重路徑（使用 CLIP 訓練的 source_projector.pt）
    projector_path = '/mnt/backups/andycw/UDA-AI/ckps/source/uda/M58/C/source_projector.pt'
    
    output_dir = '/mnt/backups/andycw/UDA-AI/tsne_results'
    
    # ===== 檢查文件是否存在 =====
    print("檢查文件...")
    for path, name in [(source_list, 'Source list'), 
                       (target_list, 'Target list'),
                       (classname_file, 'Classname file'),
                       (projector_path, 'Projector weights')]:
        if not osp.exists(path):
            raise FileNotFoundError(f"{name} 不存在: {path}")
        print(f"  ✓ {name}")
    print()
    
    # ===== 載入類別名稱 =====
    print("載入類別名稱...")
    classnames = load_classnames(classname_file)
    print(f"  ✓ 載入 {len(classnames)} 個類別")
    print(f"    前 5 個: {classnames[:5]}")
    print()
    
    # ===== 載入 CLIP 模型 =====
    print("載入 CLIP 模型...")
    clip_model = load_clip_to_cpu(backbone='RN101')
    clip_model.float()
    clip_model.cuda()
    clip_model.eval()
    print(f"  ✓ CLIP 模型已載入 (backbone: RN101)")
    print()
    
    # ===== 載入 Projector =====
    print("載入 Projector...")
    projector = load_projector(projector_path, device='cuda')
    print(f"  ✓ Projector 已載入: {projector_path}")
    print()
    
    # ===== 創建 DataLoader =====
    print("創建 DataLoader...")
    source_loader, target_loader = create_data_loaders(
        source_list_path=source_list,
        target_list_path=target_list,
        batch_size=64
    )
    print()
    
    # ===== 生成 t-SNE 視覺化圖 =====
    visualize_tsne(
        clip_model=clip_model,
        projector=projector,
        source_loader=source_loader,
        target_loader=target_loader,
        classnames=classnames,
        output_dir=output_dir,
        filename='Before_IM_Loss_train.png'
    )
    
    print("="*60)
    print("✅ t-SNE 可視化圖生成完成！")
    print(f"輸出目錄: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
