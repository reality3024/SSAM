"""
生成 t-SNE 視覺化，比較三個類別在不同表示方式下的特徵分布：
1. Generic prompt: "A photo of a [class name]"
2. Descriptive text: 從 CSV 讀取的描述性文本（10句平均）
3. Image features: 實際圖像的特徵

作者: Andy
日期: 2026-01-30
"""

import torch
import clip
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import glob

# ==================== 配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "RN101"  # 與 clip_source_model_projectOnly.py 一致

# 三個目標類別
CLASSES = [
    "M58-從動側外殼",
    "M58-下方護蓋1(驅動側)",
    "M58-從動側走行機身"
]

# 資料路徑
IMAGE_DIR = "/mnt/backups/andycw/M58/Real_all_nobg_augmented"
CSV_PATH = "/mnt/backups/andycw/M58/M58_79classes_sourceImage_descriptions_summerized.csv"
OUTPUT_DIR = "/mnt/backups/andycw/UDA-AI/tsne_results"

# t-SNE 參數
TSNE_PERPLEXITY = 5  # 因為只有9個點（3類 × 3特徵）
TSNE_RANDOM_STATE = 42

# 視覺化參數
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 藍、橙、綠
MARKERS = ['o', 's', '^']  # 圓形、方形、三角形
MARKER_LABELS = ['Image', 'Generic', 'Descriptive']
MARKER_SIZE = 250


# ==================== 函數定義 ====================

def load_clip_model():
    """載入 CLIP 模型"""
    print(f"\n{'='*60}")
    print(f"載入 CLIP 模型: {CLIP_MODEL}")
    print(f"設備: {DEVICE}")
    model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    model.eval()
    print(f"✓ CLIP 模型載入成功")
    print(f"{'='*60}\n")
    return model, preprocess


def load_descriptive_texts(csv_path, target_classes):
    """
    從 CSV 檔案載入描述性文本
    
    Returns:
        dict: {class_name: [10 sentences]}
    """
    print(f"從 CSV 載入描述性文本...")
    df = pd.read_csv(csv_path)
    
    descriptions = {}
    for class_name in target_classes:
        row = df[df['class_name'] == class_name]
        if row.empty:
            raise ValueError(f"找不到類別: {class_name}")
        
        # 解析描述文本（格式：1. xxx\n2. xxx\n...）
        response = row['response'].values[0]
        sentences = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # 去除編號（例如 "1. " 或 "10. "）
                sentence = line.split('.', 1)[1].strip()
                sentences.append(sentence)
        
        descriptions[class_name] = sentences
        print(f"  ✓ {class_name}: {len(sentences)} 句描述")
    
    print(f"✓ 描述性文本載入完成\n")
    return descriptions


def get_image_paths(image_dir, class_name, max_images=None):
    """
    獲取類別的所有圖片路徑
    
    Args:
        image_dir: 圖片根目錄
        class_name: 類別名稱
        max_images: 最多讀取多少張圖片（None=全部）
    """
    class_dir = os.path.join(image_dir, class_name)
    
    if not os.path.exists(class_dir):
        raise ValueError(f"找不到類別目錄: {class_dir}")
    
    # 獲取所有 jpg 圖片
    img_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
    
    if max_images:
        img_paths = img_paths[:max_images]
    
    return img_paths


def extract_image_features(model, preprocess, img_paths):
    """
    提取圖像特徵（批量處理）
    
    Args:
        model: CLIP 模型
        preprocess: CLIP 預處理函數
        img_paths: 圖片路徑列表
        
    Returns:
        torch.Tensor: [N, 512] 歸一化的圖像特徵
    """
    print(f"  提取 {len(img_paths)} 張圖片的特徵...")
    
    batch_size = 32
    all_features = []
    
    for i in tqdm(range(0, len(img_paths), batch_size), desc="  Processing batches"):
        batch_paths = img_paths[i:i+batch_size]
        
        # 載入並預處理圖片
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = preprocess(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"  ⚠ 無法載入圖片 {path}: {e}")
                continue
        
        if not images:
            continue
        
        # 批量編碼
        images_tensor = torch.stack(images).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(images_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        all_features.append(features.cpu())
    
    # 合併所有批次
    all_features = torch.cat(all_features, dim=0)
    return all_features


def extract_text_features(model, texts):
    """
    提取文本特徵
    
    Args:
        model: CLIP 模型
        texts: 文本列表
        
    Returns:
        torch.Tensor: [N, 512] 歸一化的文本特徵
    """
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu()


def compute_class_features(model, preprocess, class_name, img_dir, descriptions):
    """
    計算單個類別的三種特徵表示
    
    Returns:
        dict: {
            'image_centroid': [1, 512],
            'generic_text': [1, 512],
            'descriptive_text': [1, 512]
        }
    """
    print(f"\n處理類別: {class_name}")
    print(f"{'-'*60}")
    
    # 1. 圖像特徵中心
    img_paths = get_image_paths(img_dir, class_name)
    img_features = extract_image_features(model, preprocess, img_paths)
    img_centroid = img_features.mean(dim=0, keepdim=True)
    img_centroid = img_centroid / img_centroid.norm(dim=-1, keepdim=True)
    print(f"  ✓ 圖像中心: {img_features.shape} → {img_centroid.shape}")
    
    # 2. Generic text 特徵
    generic_prompt = f"A photo of a {class_name}"
    generic_feature = extract_text_features(model, [generic_prompt])
    print(f"  ✓ Generic prompt: \"{generic_prompt}\"")
    
    # 3. Descriptive text 特徵（10句平均）
    desc_sentences = descriptions[class_name]
    desc_features = extract_text_features(model, desc_sentences)
    desc_centroid = desc_features.mean(dim=0, keepdim=True)
    desc_centroid = desc_centroid / desc_centroid.norm(dim=-1, keepdim=True)
    print(f"  ✓ Descriptive text: {len(desc_sentences)} 句 → 平均")
    
    return {
        'image_centroid': img_centroid,
        'generic_text': generic_feature,
        'descriptive_text': desc_centroid
    }


def run_tsne(features, perplexity=5):
    """
    執行 t-SNE 降維
    
    Args:
        features: [N, D] numpy array
        perplexity: t-SNE 參數
        
    Returns:
        [N, 2] numpy array
    """
    print(f"\n執行 t-SNE 降維...")
    print(f"  輸入維度: {features.shape}")
    print(f"  Perplexity: {perplexity}")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=TSNE_RANDOM_STATE,
        init='pca',
        learning_rate='auto'
    )
    
    embeddings = tsne.fit_transform(features)
    print(f"  ✓ t-SNE 完成: {embeddings.shape}")
    
    return embeddings


def visualize_tsne(embeddings, classes, output_dir, features_np):
    """
    繪製 t-SNE 視覺化圖
    
    Args:
        embeddings: [9, 2] t-SNE 座標
        classes: 類別名稱列表
        output_dir: 輸出目錄
        features_np: [9, 512] 原始特徵（用於計算 cosine distance）
    """
    print(f"\n繪製 t-SNE 視覺化...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # 為每個類別繪製三個點（Image, Generic, Descriptive）
    for class_idx, class_name in enumerate(classes):
        base_idx = class_idx * 3
        color = COLORS[class_idx]
        
        # 繪製三個特徵點
        for feat_idx, (marker, label) in enumerate(zip(MARKERS, MARKER_LABELS)):
            idx = base_idx + feat_idx
            x, y = embeddings[idx]
            
            # 完整的標籤（只在圖例中顯示一次）
            full_label = f"{class_name} - {label}"
            
            plt.scatter(
                x, y,
                c=[color],
                marker=marker,
                s=MARKER_SIZE,
                alpha=0.8,
                edgecolors='black',
                linewidths=2.0,
                label=full_label
            )
        
        # 在每個類別的中心位置標註類別名稱
        # center_x = np.mean([embeddings[base_idx + i, 0] for i in range(3)])
        # center_y = np.mean([embeddings[base_idx + i, 1] for i in range(3)])
        # plt.text(
        #     center_x, center_y, class_name,
        #     fontsize=11, fontweight='bold', ha='center', va='center',
        #     bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        # )
    
    # 計算 cosine distance 的輔助函數
    def cosine_distance(feat1, feat2):
        # feat1 和 feat2 已經是歸一化的，所以 cosine similarity = dot product
        cos_sim = np.dot(feat1, feat2)
        cos_dist = 1 - cos_sim
        return cos_dist
    
    # 繪製同類型特徵之間的連線（跨類別）
    # feat_type: 0=Image(○), 1=Generic(□), 2=Descriptive(△)
    line_colors = ['blue', 'green', 'red']  # Image用藍色, Generic用綠色, Descriptive用紅色
    line_styles = ['-', '--', ':']  # 不同線型
    
    for feat_type in range(3):  # 0=Image, 1=Generic, 2=Descriptive
        # 收集該類型特徵的所有索引
        indices = [class_idx * 3 + feat_type for class_idx in range(len(classes))]
        
        # 在這些點之間兩兩連線
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                
                # 計算 cosine distance
                cos_dist = cosine_distance(features_np[idx1], features_np[idx2])
                
                # 中點位置
                mid_x = (embeddings[idx1, 0] + embeddings[idx2, 0]) / 2
                mid_y = (embeddings[idx1, 1] + embeddings[idx2, 1]) / 2
                
                # 繪製連線
                plt.plot(
                    [embeddings[idx1, 0], embeddings[idx2, 0]], 
                    [embeddings[idx1, 1], embeddings[idx2, 1]],
                    color=line_colors[feat_type], 
                    linestyle=line_styles[feat_type], 
                    alpha=0.4, 
                    linewidth=1.5
                )
                
                # 標註 cosine distance
                plt.text(
                    mid_x, mid_y, f'{cos_dist:.3f}', 
                    fontsize=10, ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.8, edgecolor=line_colors[feat_type], linewidth=1.5)
                )
    
    # 設置圖表
    plt.title(
        "t-SNE Visualization: Image vs Generic vs Descriptive Feature Representations\n"
        f"Model: {CLIP_MODEL} | Dataset: M58 (3 Classes) | Line Values: Cosine Distance",
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=13)
    plt.ylabel("t-SNE Dimension 2", fontsize=13)
    
    # 圖例（放在左下角）
    plt.legend(
        loc='lower left',
        fontsize=10,
        frameon=True,
        shadow=True,
        framealpha=0.9
    )
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存圖片
    output_path = os.path.join(output_dir, "tsne_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 圖片已保存: {output_path}")
    
    plt.show()
    plt.close()


def main():
    """主函數"""
    print("\n" + "="*60)
    print("t-SNE 視覺化：比較 Image vs Generic vs Descriptive 表示")
    print("="*60)
    
    # 1. 載入 CLIP 模型
    model, preprocess = load_clip_model()
    
    # 2. 載入描述性文本
    descriptions = load_descriptive_texts(CSV_PATH, CLASSES)
    
    # 3. 對每個類別提取三種特徵
    all_features = []
    feature_labels = []
    
    for class_name in CLASSES:
        features_dict = compute_class_features(
            model, preprocess, class_name, IMAGE_DIR, descriptions
        )
        
        # 按順序添加：Image → Generic → Descriptive
        all_features.append(features_dict['image_centroid'])
        all_features.append(features_dict['generic_text'])
        all_features.append(features_dict['descriptive_text'])
        
        feature_labels.extend([
            f"{class_name}_Image",
            f"{class_name}_Generic",
            f"{class_name}_Descriptive"
        ])
    
    # 4. 合併所有特徵並轉換為 numpy
    features_tensor = torch.cat(all_features, dim=0)  # [9, 512]
    features_np = features_tensor.numpy()
    
    print(f"\n{'='*60}")
    print(f"特徵匯總:")
    print(f"  總計: {features_np.shape[0]} 個點（3 類 × 3 特徵）")
    print(f"  特徵維度: {features_np.shape[1]}")
    print(f"{'='*60}")
    
    # 5. t-SNE 降維
    embeddings = run_tsne(features_np, perplexity=TSNE_PERPLEXITY)
    
    # 6. 視覺化（傳入原始特徵用於計算 cosine distance）
    visualize_tsne(embeddings, CLASSES, OUTPUT_DIR, features_np)
    
    print(f"\n{'='*60}")
    print("✓ 所有任務完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
