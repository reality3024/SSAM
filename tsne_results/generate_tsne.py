"""
生成 t-SNE 可視化圖
顯示 Source (CAD)、Target (Real) 和 Text Embeddings 的特徵分布
"""
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
import sys

# 導入必要模組
sys.path.append('/mnt/backups/andycw/UDA-AI')
from data_list import ImageList_idx
from tsne_RN import visualize_tsne, visualize_target_tsne
import network


class CLIP_Backbone_Wrapper(nn.Module):
    """
    CLIP Visual Encoder Wrapper
    提供兩條路徑：
    - Path 1: GAP → [B, 2048] (用於 Bottleneck → Classifier)
    - Path 2: attnpool → [B, 512] (用於 CLIP similarity 和 t-SNE)
    """
    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model
        self.in_features = 2048
        
    def forward(self, x, return_both=False):
        """
        Args:
            x: 輸入影像 [B, 3, 224, 224]
            return_both: 是否返回雙路徑特徵
        
        Returns:
            如果 return_both=False: feat_2048 [B, 2048]
            如果 return_both=True: (feat_2048 [B, 2048], feat_512 [B, 512])
        """
        # 1. 精度對齊
        x = x.type(self.visual.conv1.weight.dtype)
        
        # 2. ResNet backbone（到 layer4）
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        x = self.visual.layer4(x)  # [B, 2048, 7, 7]
        
        # 3. Path 1: GAP → 2048維
        feat_2048 = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 2048, 1, 1]
        feat_2048 = feat_2048.flatten(1)  # [B, 2048]
        
        if not return_both:
            return feat_2048
        
        # 4. Path 2: attnpool → 512維
        feat_512 = self.visual.attnpool(x)  # [B, 512]
        
        return feat_2048, feat_512


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


def create_data_loaders(source_list_path, target_list_path, batch_size=32):
    """
    創建 Source 和 Target 的 DataLoader
    
    Args:
        source_list_path: Source 數據列表文件路徑
        target_list_path: Target 數據列表文件路徑
        batch_size: Batch size
    
    Returns:
        source_loader, target_loader
    """
    # 讀取數據列表
    txt_source = open(source_list_path).readlines()
    txt_target = open(target_list_path).readlines()
    
    # M58 數據根目錄
    root_path = '/mnt/backups/andycw/'
    
    # 創建 Dataset 和 DataLoader
    transform = image_transform()
    
    # Source Loader - 注意 ImageList_idx 返回 (img, label, idx, path)
    source_dataset = ImageList_idx(txt_source, transform=transform, root=root_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4, drop_last=False)
    
    # Target Loader
    target_dataset = ImageList_idx(txt_target, transform=transform, root=root_path)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4, drop_last=False)
    
    print(f"✓ Source 數據: {len(source_dataset)} 張")
    print(f"✓ Target 數據: {len(target_dataset)} 張")
    
    return source_loader, target_loader


def load_models(model_dir, backbone='RN101', bottleneck_dim=256, num_classes=79):
    """
    載入訓練好的模型
    
    Args:
        model_dir: 模型權重目錄
        backbone: CLIP backbone 名稱
        bottleneck_dim: Bottleneck 維度
        num_classes: 類別數量
    
    Returns:
        netF, netB, text_features
    """
    print(f"\n{'='*60}")
    print(f"載入模型權重...")
    print(f"{'='*60}")
    
    # 1. 載入 CLIP 模型並創建 Wrapper
    clip_model = load_clip_to_cpu(backbone=backbone)
    clip_model.float()
    netF = CLIP_Backbone_Wrapper(clip_model.visual).cuda()
    
    # 2. 創建 Bottleneck
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,
                                   bottleneck_dim=bottleneck_dim).cuda()
    
    # 3. 載入權重
    netF_path = osp.join(model_dir, 'target_F_best.pt')
    netB_path = osp.join(model_dir, 'target_B_best.pt')
    text_emb_path = osp.join('/mnt/backups/andycw/UDA-AI/ckps/source/uda/M58/C', 'M58_text_embeddings.pt')
    
    # 載入 Feature Extractor
    pretrained_dict = torch.load(netF_path)
    sample_key = list(pretrained_dict.keys())[0]
    if not sample_key.startswith('visual.'):
        new_state_dict = {f"visual.{k}": v for k, v in pretrained_dict.items()}
    else:
        new_state_dict = pretrained_dict
    
    netF.load_state_dict(new_state_dict, strict=False)
    print(f"✓ Feature Extractor 載入成功: {netF_path}")
    
    # 載入 Bottleneck
    netB.load_state_dict(torch.load(netB_path))
    print(f"✓ Bottleneck 載入成功: {netB_path}")
    
    # 載入 Text Embeddings
    embeddings_dict = torch.load(text_emb_path, map_location='cuda')
    text_features = embeddings_dict['text_features'].cuda()
    classnames = embeddings_dict['classnames']
    print(f"✓ Text Embeddings 載入成功: {text_emb_path}")
    print(f"  - Shape: {text_features.shape}")
    print(f"  - Classes: {len(classnames)}")
    
    # 設為評估模式
    netF.eval()
    netB.eval()
    
    print(f"{'='*60}\n")
    
    return netF, netB, text_features


def main():
    """主函數：生成 t-SNE 圖"""
    print("\n" + "="*60)
    print("開始生成 t-SNE 可視化圖")
    print("="*60 + "\n")
    
    test_only = False
    # 設定路徑
    source_list = '/mnt/backups/andycw/M58/CAD_79_ratioFilter_list.txt'
    target_list = '/mnt/backups/andycw/M58/Real_79_nobg_augmented_list.txt'
    model_dir = '/mnt/backups/andycw/UDA-AI/ckps/target/uda/M58/79_class_ssaf'
    output_dir = '/mnt/backups/andycw/UDA-AI/tsne_results'
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 檢查文件是否存在
    print("檢查文件...")
    for path, name in [(source_list, 'Source list'), 
                       (target_list, 'Target list'),
                       (model_dir, 'Model directory')]:
        if not osp.exists(path):
            raise FileNotFoundError(f"{name} 不存在: {path}")
        print(f"✓ {name}: {path}")
    print()
    
    # 1. 載入模型
    netF, netB, text_features = load_models(
        model_dir=model_dir,
        backbone='RN101',
        bottleneck_dim=256,
        num_classes=79
    )
    
    # 2. 創建 DataLoader
    print("創建 DataLoader...")
    source_loader, target_loader = create_data_loaders(
        source_list_path=source_list,
        target_list_path=target_list,
        batch_size=64  # 可調整
    )

    if test_only:
        print("\n" + "=" * 60)
        print("生成 Target Domain 單獨圖")
        print("=" * 60)
        visualize_target_tsne(
            netF=netF,
            netB=netB,
            target_loader=target_loader,
            text_features=text_features,
            output_dir=output_dir,
            filename='tsne_M58_CAD_to_Real_target_ssaf.png'
        )
    
    else:
        visualize_tsne(
            netF=netF,
            netB=netB,
            source_loader=source_loader,
            target_loader=target_loader,
            text_features=text_features,
            output_dir=output_dir,
            filename='tsne_M58_CAD_to_Real_comparison.png'
        )
if __name__ == "__main__":
    main()
