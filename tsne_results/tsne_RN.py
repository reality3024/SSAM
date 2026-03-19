from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib.patches import Rectangle

def visualize_tsne(netF, netB, source_loader, target_loader, text_features, output_dir, filename='tsne_start.png'):
    """
    生成 t-SNE 可視化圖，顯示 Source、Target 和 Text embeddings 的分布
    
    Args:
        netF: Feature Extractor (CLIP_Backbone_Wrapper)
        netB: Bottleneck (不用於此函數，但保留接口一致性)
        source_loader: Source 數據的 DataLoader
        target_loader: Target 數據的 DataLoader
        text_features: Text embeddings [num_classes, 512]
        output_dir: 輸出目錄
        filename: 輸出文件名
    """
    print(f"\n{'='*60}")
    print(f"開始繪製 t-SNE: {filename}")
    print(f"{'='*60}\n")
    
    netF.eval()
    if netB is not None:
        netB.eval()
    
    features = []
    labels = []
    domains = []  # 'Source', 'Target', 'Text'
    
    # 1. 收集 Source Features - 每個類別取約 20 個樣本
    print("收集 Source 特徵 (每個類別約 20 個樣本)...")
    samples_per_class = 20
    source_class_samples = {}  # {class_id: [(feature, label), ...]}
    
    with torch.no_grad():
        for i, data in enumerate(source_loader):
            # ImageList_idx 返回 (img, label, idx, path)
            if len(data) == 4:
                imgs, lbls, idx, path = data
            else:
                imgs, lbls, idx = data
            
            imgs = imgs.cuda()
            
            # 使用 return_both=True 獲取 512 維特徵（用於對齊 text embeddings）
            _, feat_512 = netF(imgs, return_both=True)
            feat_512 = F.normalize(feat_512, dim=-1)
            
            # 按類別分組
            for j, lbl in enumerate(lbls):
                class_id = lbl.item()
                if class_id not in source_class_samples:
                    source_class_samples[class_id] = []
                
                # 每個類別最多收集 samples_per_class 個樣本
                if len(source_class_samples[class_id]) < samples_per_class:
                    source_class_samples[class_id].append((feat_512[j].cpu(), lbl))
    
    # 整理 Source 樣本
    for class_id in sorted(source_class_samples.keys()):
        for feat, lbl in source_class_samples[class_id]:
            features.append(feat.unsqueeze(0))
            labels.append(lbl.unsqueeze(0))
            domains.append('Source')
    
    print(f"  ✓ 收集了 {len([d for d in domains if d == 'Source'])} 個 Source 樣本")
    print(f"  ✓ 涵蓋 {len(source_class_samples)} 個類別")
    if len(source_class_samples) < 79:
        missing_classes = set(range(79)) - set(source_class_samples.keys())
        print(f"  ⚠ 警告：缺少 {len(missing_classes)} 個類別: {sorted(missing_classes)}")
    
    # 2. 收集 Target Features - 每個類別取約 20 個樣本
    print("收集 Target 特徵 (每個類別約 20 個樣本)...")
    target_class_samples = {}  # {class_id: [(feature, label), ...]}
    
    with torch.no_grad():
        for i, data in enumerate(target_loader):
            # ImageList_idx 返回 (img, label, idx, path)
            if len(data) == 4:
                imgs, lbls, idx, path = data
            else:
                imgs, lbls, idx = data
            
            # 如果 imgs 是 tuple（來自 Augmentation），取第一個（weak augmentation）
            if isinstance(imgs, (tuple, list)):
                imgs = imgs[0]
            
            imgs = imgs.cuda()
            
            # 使用 return_both=True 獲取 512 維特徵
            _, feat_512 = netF(imgs, return_both=True)
            feat_512 = F.normalize(feat_512, dim=-1)
            
            # 按類別分組
            for j, lbl in enumerate(lbls):
                class_id = lbl.item()
                if class_id not in target_class_samples:
                    target_class_samples[class_id] = []
                
                # 每個類別最多收集 samples_per_class 個樣本
                if len(target_class_samples[class_id]) < samples_per_class:
                    target_class_samples[class_id].append((feat_512[j].cpu(), lbl))
    
    # 整理 Target 樣本
    for class_id in sorted(target_class_samples.keys()):
        for feat, lbl in target_class_samples[class_id]:
            features.append(feat.unsqueeze(0))
            labels.append(lbl.unsqueeze(0))
            domains.append('Target')
    
    print(f"  ✓ 收集了 {len([d for d in domains if d == 'Target'])} 個 Target 樣本")
    print(f"  ✓ 涵蓋 {len(target_class_samples)} 個類別")
    if len(target_class_samples) < 79:
        missing_classes = set(range(79)) - set(target_class_samples.keys())
        print(f"  ⚠ 警告：缺少 {len(missing_classes)} 個類別: {sorted(missing_classes)}")
    
    # 3. 加入 Text Anchors
    print("加入 Text Embeddings...")
    text_features_norm = F.normalize(text_features, dim=-1)
    features.append(text_features_norm.cpu())
    labels.append(torch.arange(len(text_features)))  # 0 ~ num_classes-1
    domains.extend(['Text'] * len(text_features))
    print(f"  ✓ 加入 {len(text_features)} 個 Text Anchors")
    
    # 4. 合併所有特徵
    print("\n執行 t-SNE 降維...")
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    print(f"  總樣本數: {len(features)}")
    print(f"  特徵維度: {features.shape[1]}")
    
    # t-SNE 運算
    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        random_state=42,
        perplexity=min(30, len(features) - 1)  # 確保 perplexity < n_samples
    )
    X_embedded = tsne.fit_transform(features)
    print("  ✓ t-SNE 降維完成")
    
    # 5. 繪圖
    print("\n繪製圖表...")
    df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Domain': domains,
        'Class': labels
    })
    
    # 分離數據
    source_df = df[df['Domain'] == 'Source']
    target_df = df[df['Domain'] == 'Target']
    text_df = df[df['Domain'] == 'Text']
    
    # 1. 取得所有類別（不重複）
    unique_classes = sorted(df['Class'].unique())
    num_classes = len(unique_classes)
    print(f"  總類別數: {num_classes}")

    # 2. 生成高對比度顏色序列
    if num_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    elif num_classes <= 32:
        colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_set3 = plt.cm.Set3(np.linspace(0, 1, num_classes - 20))
        colors = np.vstack([colors_tab20, colors_set3])
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes))

    # 3. 建立 Class 與 顏色的對應字典
    class_palette = dict(zip(unique_classes, colors))

    # 創建左右對比圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # ===== 左圖：Source Domain =====
    print("\n繪製 Source Domain...")
    for i, cls in enumerate(unique_classes):
        cls_df = source_df[source_df['Class'] == cls]
        if len(cls_df) > 0:
            ax1.scatter(cls_df['x'], cls_df['y'], 
                       c=[colors[i]], label=f"Class {int(cls)}", 
                       alpha=0.7, s=50, marker='o', edgecolors='none')
            
            # 為每個類別標註一個點
            random_idx = np.random.randint(0, len(cls_df))
            random_point = cls_df.iloc[random_idx]
            ax1.annotate(str(int(cls)), 
                        (random_point['x'], random_point['y']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                                 edgecolor='black', alpha=0.8, linewidth=1),
                        color='white' if np.sum(colors[i][:3]) < 1.5 else 'black')
    
    # 在 Source 圖上也畫 Text Features (黑色星號)
    if len(text_df) > 0:
        ax1.scatter(text_df['x'], text_df['y'],
                   c='black', alpha=1.0, s=250, marker='*',
                   edgecolors='yellow', linewidths=1.5, label='Text Anchors', zorder=1000)
    
    ax1.set_title('Source Domain (CAD)\n+ Text Features', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ===== 右圖：Target Domain =====
    print("繪製 Target Domain...")
    for i, cls in enumerate(unique_classes):
        cls_df = target_df[target_df['Class'] == cls]
        if len(cls_df) > 0:
            ax2.scatter(cls_df['x'], cls_df['y'], 
                       c=[colors[i]], label=f"Class {int(cls)}", 
                       alpha=0.7, s=50, marker='X', 
                       edgecolors='black', linewidths=0.5)
            
            # 為每個類別標註一個點
            random_idx = np.random.randint(0, len(cls_df))
            random_point = cls_df.iloc[random_idx]
            ax2.annotate(str(int(cls)), 
                        (random_point['x'], random_point['y']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                                 edgecolor='black', alpha=0.8, linewidth=1),
                        color='white' if np.sum(colors[i][:3]) < 1.5 else 'black')
    
    # 在 Target 圖上也畫 Text Features (黑色星號)
    if len(text_df) > 0:
        ax2.scatter(text_df['x'], text_df['y'],
                   c='black', alpha=1.0, s=250, marker='*',
                   edgecolors='yellow', linewidths=1.5, label='Text Anchors', zorder=1000)
    
    ax2.set_title('Target Domain (Real)\n+ Text Features', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 總標題
    # fig.suptitle('t-SNE Visualization: Source vs Target (with Text Features)\nM58 Dataset - 79 Classes', 
    #             fontsize=16, fontweight='bold', y=0.98)
    
    # 添加圖例（顯示在右側）
    print("添加圖例...")
    if num_classes <= 15:
        fig.legend(*ax1.get_legend_handles_labels(), 
                  bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=10)
    else:
        fig.legend(*ax1.get_legend_handles_labels(), 
                  bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, top=0.95)
    
    # 保存圖片
    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"✅ t-SNE 圖繪製完成！")
    print(f"📁 輸出路徑: {output_path}")
    print(f"{'='*60}\n")
    
    # 統計資訊
    print("\n統計資訊:")
    print(f"  Source 樣本: {len(source_df)}")
    print(f"  Target 樣本: {len(target_df)}")
    print(f"  Text Anchors: {len(text_df)}")
    print(f"  總計: {len(df)}")
    print()
def visualize_target_tsne(netF, netB, target_loader, text_features, output_dir, filename='tsne_target.png'):
    """
    生成 t-SNE 可視化圖，顯示 Source、Target 和 Text embeddings 的分布
    
    Args:
        netF: Feature Extractor (CLIP_Backbone_Wrapper)
        netB: Bottleneck (不用於此函數，但保留接口一致性)
        target_loader: Target 數據的 DataLoader
        text_features: Text embeddings [num_classes, 512]
        output_dir: 輸出目錄
        filename: 輸出文件名
    """
    print(f"\n{'='*60}")
    print(f"開始繪製 t-SNE: {filename}")
    print(f"{'='*60}\n")
    
    netF.eval()
    if netB is not None:
        netB.eval()
    
    features = []
    labels = []
    domains = []  # 'Target', 'Text'
    samples_per_class = 20
   
    # 收集 Target Features - 每個類別取約 20 個樣本
    print("收集 Target 特徵 (每個類別約 20 個樣本)...")
    target_class_samples = {}  # {class_id: [(feature, label), ...]}
    
    with torch.no_grad():
        for i, data in enumerate(target_loader):
            # ImageList_idx 返回 (img, label, idx, path)
            if len(data) == 4:
                imgs, lbls, idx, path = data
            else:
                imgs, lbls, idx = data
            
            # 如果 imgs 是 tuple（來自 Augmentation），取第一個（weak augmentation）
            if isinstance(imgs, (tuple, list)):
                imgs = imgs[0]
            
            imgs = imgs.cuda()
            
            # 使用 return_both=True 獲取 512 維特徵
            _, feat_512 = netF(imgs, return_both=True)
            feat_512 = F.normalize(feat_512, dim=-1)
            
            # 按類別分組
            for j, lbl in enumerate(lbls):
                class_id = lbl.item()
                if class_id not in target_class_samples:
                    target_class_samples[class_id] = []
                
                # 每個類別最多收集 samples_per_class 個樣本
                if len(target_class_samples[class_id]) < samples_per_class:
                    target_class_samples[class_id].append((feat_512[j].cpu(), lbl))
    
    # 整理 Target 樣本
    for class_id in sorted(target_class_samples.keys()):
        for feat, lbl in target_class_samples[class_id]:
            features.append(feat.unsqueeze(0))
            labels.append(lbl.unsqueeze(0))
            domains.append('Target')
    
    print(f"  ✓ 收集了 {len([d for d in domains if d == 'Target'])} 個 Target 樣本")
    print(f"  ✓ 涵蓋 {len(target_class_samples)} 個類別")
    if len(target_class_samples) < 79:
        missing_classes = set(range(79)) - set(target_class_samples.keys())
        print(f"  ⚠ 警告：缺少 {len(missing_classes)} 個類別: {sorted(missing_classes)}")
    
    # 3. 加入 Text Anchors
    print("加入 Text Embeddings...")
    text_features_norm = F.normalize(text_features, dim=-1)
    features.append(text_features_norm.cpu())
    labels.append(torch.arange(len(text_features)))  # 0 ~ num_classes-1
    domains.extend(['Text'] * len(text_features))
    print(f"  ✓ 加入 {len(text_features)} 個 Text Anchors")
    
    # 4. 合併所有特徵
    print("\n執行 t-SNE 降維...")
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    print(f"  總樣本數: {len(features)}")
    print(f"  特徵維度: {features.shape[1]}")
    
    # t-SNE 運算
    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        random_state=42,
        perplexity=min(30, len(features) - 1)  # 確保 perplexity < n_samples
    )
    X_embedded = tsne.fit_transform(features)
    print("  ✓ t-SNE 降維完成")
    
    # 5. 繪圖
    print("\n繪製圖表...")
    df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Domain': domains,
        'Class': labels
    })
    
    # 分離數據
    target_df = df[df['Domain'] == 'Target']
    text_df = df[df['Domain'] == 'Text']
    
    # 1. 取得所有類別（不重複）
    unique_classes = sorted(df['Class'].unique())
    num_classes = len(unique_classes)
    print(f"  總類別數: {num_classes}")

    # 2. 生成高對比度顏色序列
    if num_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    elif num_classes <= 32:
        colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_set3 = plt.cm.Set3(np.linspace(0, 1, num_classes - 20))
        colors = np.vstack([colors_tab20, colors_set3])
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes))

    # 3. 建立 Class 與 顏色的對應字典
    class_palette = dict(zip(unique_classes, colors))

    # 創建左右對比圖
    fig= plt.figure(figsize=(12, 10))
    ax2 = fig.add_subplot(111)
    # 建立Target Domain t-SNE圖
    print("繪製 Target Domain...")
    for i, cls in enumerate(unique_classes):
        cls_df = target_df[target_df['Class'] == cls]
        if len(cls_df) > 0:
            ax2.scatter(cls_df['x'], cls_df['y'], 
                       c=[colors[i]], label=f"Class {int(cls)}", 
                       alpha=0.7, s=50, marker='X', 
                       edgecolors='black', linewidths=0.5)
            
            # 為每個類別標註一個點
            random_idx = np.random.randint(0, len(cls_df))
            random_point = cls_df.iloc[random_idx]
            ax2.annotate(str(int(cls)), 
                        (random_point['x'], random_point['y']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                                 edgecolor='black', alpha=0.8, linewidth=1),
                        color='white' if np.sum(colors[i][:3]) < 1.5 else 'black')
    
    # 在 Target 圖上也畫 Text Features (黑色星號)
    if len(text_df) > 0:
        ax2.scatter(text_df['x'], text_df['y'],
                   c='black', alpha=1.0, s=250, marker='*',
                   edgecolors='yellow', linewidths=1.5, label='Text Anchors', zorder=1000)
    
    ax2.set_title('SSAF on Real dataset + Text Features', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 總標題
    # fig.suptitle('t-SNE Visualization: Target (with Text Features)', 
    #             fontsize=16, fontweight='bold', y=0.98)
    
    # 添加圖例（顯示在右側）
    print("添加圖例...")
    if num_classes <= 15:
        fig.legend(*ax2.get_legend_handles_labels(), 
                  bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=10)
    else:
        fig.legend(*ax2.get_legend_handles_labels(), 
                  bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, top=0.95)
    
    # 保存圖片
    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"✅ t-SNE 圖繪製完成！")
    print(f"📁 輸出路徑: {output_path}")
    print(f"{'='*60}\n")
    
    # 統計資訊
    print("\n統計資訊:")
    print(f"  Target 樣本: {len(target_df)}")
    print(f"  Text Anchors: {len(text_df)}")
    print(f"  總計: {len(df)}")
    print()
