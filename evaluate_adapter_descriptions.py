"""
評估 Adapter 在 Target Domain Descriptions 上的表現
使用 InternVL 生成的描述進行評估
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import sklearn.metrics as sm
import clip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ==================== Adapter Modules ====================
class Adapter(nn.Module):
    """Linear Adapter (MLP) - 簡單的前饋網路"""
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = torch.nn.functional.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, ratio=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.xavier_normal_(self.w_qs.weight, gain=1.0)
        nn.init.xavier_normal_(self.w_ks.weight, gain=1.0)
        nn.init.xavier_normal_(self.w_vs.weight, gain=0.67)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight, gain=0.67)

        self.dropout = nn.Dropout(dropout)
        self.ratio = ratio

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(2 * (self.ratio * output + (1 - self.ratio) * residual))

        return output

class SelfAttnAdapter(nn.Module):
    def __init__(self, c_in, reduction=4, ratio=0.5):
        super(SelfAttnAdapter, self).__init__()
        self.attn = MultiHeadAttention(
            1, c_in, c_in // reduction, c_in // reduction, dropout=0.5, ratio=ratio
        )

    def forward(self, x):
        x = self.attn(x, x, x)
        return x

# ==================== Text Embedding Evaluator ====================
class TextEmbeddingEvaluator:
    def __init__(self, model_path, ref_embedding_path, device='cuda', adapter_type='self_attn', load_adapter=True):
        self.device = device
        self.adapter_type = adapter_type
        self.load_adapter = load_adapter
        
        # 載入 CLIP model（轉為 float32，與訓練時一致）
        print(f"載入 CLIP model (RN101)...")
        self.clip_model, _ = clip.load('RN101', device=device)
        self.clip_model.float()  # 與訓練時一致：clip_model.float()
        self.clip_model.eval()
        
        # 載入 Adapter（根據類型和 load_adapter 標誌）
        if load_adapter:
            print(f"載入 Adapter weights 從 {model_path} (類型: {adapter_type})")
            if adapter_type == 'self_attn':
                self.adapter = SelfAttnAdapter(c_in=512, reduction=4, ratio=0.2).to(device)
            elif adapter_type == 'linear':
                self.adapter = Adapter(c_in=512, reduction=4).to(device)
            else:
                raise ValueError(f"不支援的 adapter_type: {adapter_type}")
            
            checkpoint = torch.load(model_path, map_location=device)
            self.adapter.load_state_dict(checkpoint['state_dict'], strict=False)
            self.adapter.eval()
        else:
            print(f"不載入 Adapter weights")
            self.adapter = None
        
        # 載入 Reference Embeddings
        print(f"載入 Reference Embeddings 從 {ref_embedding_path}")
        ref_data = torch.load(ref_embedding_path, map_location=device)
        self.text_features = ref_data['text_features'].to(device)  # [79, 512]
        self.classnames = ref_data['classnames']
        
        # 建立 class name 到 index 的映射
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classnames)}
        
        print(f"✓ 載入完成！")
        print(f"  - 類別數量: {len(self.classnames)}")
        print(f"  - Text features shape: {self.text_features.shape}")
        print(f"\n參考類別名稱 (前 5 個):")
        for i, name in enumerate(self.classnames[:5]):
            print(f"  [{i}] {name}")
        print(f"  ...")
        
    def process_response(self, response_text, use_adapter=True, adapter_mode=None):
        """
        處理 response 文本 - 完全比照訓練時的處理方式
        (clip_adapter_internVL.py 中 text_template_type == 'internvl' 的邏輯)
        
        Args:
            response_text: 輸入的描述文本
            use_adapter: 是否使用 Adapter（預設 True）
            adapter_mode: 'self_attn' 或 'linear'，如果為 None 則使用 self.adapter_type
        
        Returns:
            normalized embedding [1, 512]
        """
        if adapter_mode is None:
            adapter_mode = self.adapter_type
        
        # 分割成句子列表 (與訓練時 _parse_response 相同)
        # sentences = response_text.strip().split('\n')
        # sentences = [s.strip() for s in sentences if s.strip()]
        sentences = str(response_text)
        sentences = response_text.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = sentences[:10]  # 限制最多10個句子（與訓練時一致）
        
        # Tokenize 所有句子 (與訓練時相同)
        tokens = torch.cat([clip.tokenize(s) for s in sentences]).to(self.device)
        tokens = tokens.to('cuda')
        
        with torch.no_grad():
            # CLIP encode: [num_sentences, 512]
            text_features = self.clip_model.encode_text(tokens)
            
            if use_adapter and self.adapter is not None:
                if adapter_mode == 'self_attn':
                    # Self-Attention Adapter: 需要 batch 維度
                    text_features = text_features.unsqueeze(0)  # [1, num_sentences, 512]
                    text_features = self.adapter(text_features)  # [1, num_sentences, 512]
                    text_features = text_features.mean(dim=1)  # [1, 512]
                elif adapter_mode == 'linear':
                    # Linear Adapter: 先平均池化，再通過 Adapter
                    # text_features = text_features.mean(dim=1)
                    text_features = text_features.mean(dim=0, keepdim=True)  # [1, 512]
                    text_features = self.adapter(text_features)  # [1, 512]
            else:
                # 不使用 Adapter，直接平均池化
                # print("不使用 Adapter，直接平均池化文本特徵")
                text_features = text_features.mean(dim=0, keepdim=True)  # [1, 512]
            
            # 正規化（與訓練時 forward() 一致）
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def evaluate_csv(self, csv_path, batch_size=32):
        """
        評估整個 CSV 檔案
        """
        print(f"\n載入 CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"總樣本數: {len(df)}")
        
        # 初始化統計
        class_stats = {i: {'total': 0, 'correct': 0} for i in range(len(self.classnames))}
        detailed_results = []
        
        # 逐筆處理
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="評估進度"):
            class_name = row['class_name']
            response = row['response']
            img_name = row['image_file_name']
            
            # 取得 ground truth label
            true_label = self.class_to_idx[class_name]
            
            # 處理 response
            text_emb = self.process_response(response)  # [1, 512]
            
            # 計算 cosine similarity with temperature scaling (放大差異)
            # 參照 clip_adapter_internVL.py 的 forward() 邏輯
            temperature = 2.0  # 放大差異係數（原始為1.0）
            logits = temperature * (text_emb @ self.text_features.t())  # [1, 79]
            
            # 取得 top-1 prediction
            pred_idx = logits.argmax(dim=1).item()
            prob = torch.nn.functional.softmax(logits, dim=1)[0, pred_idx].item()
            
            # 統計
            class_stats[true_label]['total'] += 1
            if pred_idx == true_label:
                class_stats[true_label]['correct'] += 1
            

            # 記錄詳細結果
            detailed_results.append({
                'class_id': true_label,
                'class_name': class_name,
                'img_name': img_name,
                'pred_class_id': pred_idx,
                'pred_class_name': self.classnames[pred_idx],
                'prob': f"{prob:.4f}",
                'correct': int(pred_idx == true_label)
            })
        
        return class_stats, detailed_results
    
    def evaluate_source_target_without_adapter(self, source_csv_path, target_csv_path):
        """
        使用原始模型（不使用任何 Adapter）比較 source 和 target 的描述
        
        Args:
            source_csv_path: source domain 描述的 CSV 路徑
            target_csv_path: target domain 描述的 CSV 路徑
        
        Returns:
            class_stats: 每個類別的統計資訊
            detailed_results: 詳細的預測結果
        """
        print(f"\n{'='*60}")
        print(f"模式: 使用原始模型 (不使用任何 Adapter) 進行 Source-Target 比較")
        print(f"{'='*60}")
        
        # 載入 source CSV
        print(f"\n載入 Source CSV: {source_csv_path}")
        source_df = pd.read_csv(source_csv_path)
        print(f"Source 樣本數: {len(source_df)}")
        
        # 載入 target CSV
        print(f"載入 Target CSV: {target_csv_path}")
        target_df = pd.read_csv(target_csv_path)
        print(f"Target 樣本數: {len(target_df)}")
        
        # 為每個類別計算 source embedding（平均）
        print(f"\n計算 Source Domain 的類別 embeddings (使用原始模型)...")
        source_embeddings = {}  # class_name -> embedding [1, 512]
        
        for class_name in tqdm(self.classnames, desc="處理 Source"):
            class_data = source_df[source_df['class_name'] == class_name]
            if len(class_data) == 0:
                print(f"警告: 類別 '{class_name}' 在 source CSV 中找不到資料")
                continue
            
            # 收集該類別的所有 embeddings
            class_embeddings = []
            for _, row in class_data.iterrows():
                response = row['response']
                # 使用原始模型（不使用任何 Adapter）
                emb = self.process_response(response, use_adapter=False, adapter_mode='linear')
                class_embeddings.append(emb)
            
            # 平均所有 source images 的 embeddings
            class_emb = torch.cat(class_embeddings, dim=0).mean(dim=0, keepdim=True)  # [1, 512]
            class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)  # 重新正規化
            source_embeddings[class_name] = class_emb
        
        # 將 source embeddings 組成矩陣 [79, 512]
        source_features = torch.cat([source_embeddings[name] for name in self.classnames], dim=0)
        print(f"Source embeddings shape: {source_features.shape}")
        
        # 評估 target domain
        print(f"\n評估 Target Domain (使用原始模型)...")
        class_stats = {i: {'total': 0, 'correct': 0} for i in range(len(self.classnames))}
        detailed_results = []
        
        for idx, row in tqdm(target_df.iterrows(), total=len(target_df), desc="評估 Target"):
            class_name = row['class_name']
            response = row['response']
            img_name = row['image_file_name']
            
            # 取得 ground truth label
            true_label = self.class_to_idx[class_name]
            
            # 處理 target response（使用原始模型）
            text_emb = self.process_response(response, use_adapter=False, adapter_mode='linear')  # [1, 512]
            
            # 計算與 source embeddings 的 cosine similarity
            temperature = 1.0  # 保持一致的溫度係數
            logits = temperature * (text_emb @ source_features.t())  # [1, 79]
            
            # 取得 top-1 prediction
            pred_idx = logits.argmax(dim=1).item()
            prob = torch.nn.functional.softmax(logits, dim=1)[0, pred_idx].item()
            
            # 統計
            class_stats[true_label]['total'] += 1
            if pred_idx == true_label:
                class_stats[true_label]['correct'] += 1
            
            # 記錄詳細結果
            detailed_results.append({
                'class_id': true_label,
                'class_name': class_name,
                'img_name': img_name,
                'pred_class_id': pred_idx,
                'pred_class_name': self.classnames[pred_idx],
                'prob': f"{prob:.4f}",
                'correct': int(pred_idx == true_label)
            })
        
        return class_stats, detailed_results

def generate_confusion_matrix(detailed_results, class_names, output_dir, output_prefix):
    # ==================== 生成混淆矩陣 ====================
    print(f"\n生成混淆矩陣...")
    
    # 提取真實標籤和預測標籤
    y_true = [result['class_id'] for result in detailed_results]
    y_pred = [result['pred_class_id'] for result in detailed_results]
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    # 保存混淆矩陣為 CSV
    cm_df = pd.DataFrame(cm, 
                         index=class_names, 
                         columns=class_names)
    cm_csv_filename = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_filename, encoding='utf-8-sig')
    print(f"✓ 混淆矩陣已保存到: {cm_csv_filename}")
    
    # 繪製混淆矩陣熱圖
    plt.figure(figsize=(24, 20))
    
    # 使用對數刻度以更好地顯示差異
    cm_log = np.log1p(cm)  # log(1+x) 避免 log(0)
    
    sns.heatmap(cm_log, 
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='YlOrRd',
                fmt='d',
                cbar_kws={'label': 'log(Count + 1)'},
                linewidths=0.5,
                linecolor='gray')
    
    plt.title(f'Confusion Matrix - {output_prefix}\n(Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # 保存圖片
    cm_img_filename = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.png")
    plt.savefig(cm_img_filename, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩陣圖片已保存到: {cm_img_filename}")
    plt.close()
    
    # 繪製原始數值的混淆矩陣（帶標註）
    plt.figure(figsize=(24, 20))
    
    # 只在對角線和高頻錯誤處標註數字
    annot_array = np.array([[str(int(val)) if val > 0 else '' for val in row] for row in cm])
    
    sns.heatmap(cm, 
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues',
                annot=annot_array,
                fmt='',
                cbar_kws={'label': 'Count'},
                linewidths=0.5,
                linecolor='gray',
                annot_kws={'size': 6})
    
    plt.title(f'Confusion Matrix - {output_prefix}\n(Original Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # 保存原始數值圖片
    cm_img_raw_filename = os.path.join(output_dir, f"{output_prefix}_confusion_matrix_raw.png")
    plt.savefig(cm_img_raw_filename, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩陣圖片(原始)已保存到: {cm_img_raw_filename}")
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='評估 Adapter 在 Target Domain 上的表現')
    parser.add_argument('--mode', type=str, default='without_adapter', 
                       choices=['with_adapter', 'without_adapter', 'random_linear'],
                       help='評估模式: with_adapter (Self-Attention Adapter) / without_adapter (原始) / random_linear (隨機初始化 Linear Adapter)')
    args = parser.parse_args()
    
    # 路徑設定
    self_attn_model_path = '/mnt/backups/andycw/CLIP/output/m58/CLIP_Adapter_internVL/rn101_self_attn_ratio0.2_ep200_clipTemplate/best.pth'
    linear_model_path = '/mnt/backups/andycw/CLIP/output/m58/CLIP_Adapter_internVL/rn101_linear_ratio0.2_ep200_clipTemplate/best.pth'
    self_attn_ref_embedding_path = '/mnt/backups/andycw/CLIP/output/m58/CLIP_Adapter_internVL/rn101_self_attn_ratio0.2_ep200_clipTemplate/M58_text_embeddings.pt'
    linear_ref_embedding_path = '/mnt/backups/andycw/CLIP/output/m58/CLIP_Adapter_internVL/rn101_linear_ratio0.2_ep200_clipTemplate/M58_text_embeddings.pt'
    target_csv_path = '/mnt/backups/andycw/M58/M58_79classes_targetImage_descriptions.csv'
    source_csv_path = '/mnt/backups/andycw/M58/M58_79classes_sourceImage_descriptions_summerized.csv'
    output_dir = '/mnt/backups/andycw/UDA-AI/results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 根據模式執行評估
    if args.mode == 'with_adapter':
        # 使用 Self-Attention Adapter
        print(f"\n{'='*60}")
        print(f"模式: Self-Attention Adapter (包含 ScaledDotProductAttention)")
        print(f"{'='*60}")
        
        evaluator = TextEmbeddingEvaluator(
            model_path=self_attn_model_path,
            ref_embedding_path=self_attn_ref_embedding_path,
            device='cuda',
            adapter_type='self_attn',
            load_adapter=True
        )
        class_stats, detailed_results = evaluator.evaluate_csv(target_csv_path)
        output_prefix = "M58_SelfAttnAdapter_TargetDesc"
        
    elif args.mode == 'without_adapter':
        # 使用原始模型（不使用任何 Adapter）
        print(f"\n{'='*60}")
        print(f"模式: 原始模型 (不使用任何 Adapter)")
        print(f"{'='*60}")
        
        evaluator_linear = TextEmbeddingEvaluator(
            model_path=linear_model_path,
            ref_embedding_path=linear_ref_embedding_path,
            device='cuda',
            adapter_type='linear',
            load_adapter=False  # 不載入任何 Adapter 權重
        )
        class_stats, detailed_results = evaluator_linear.evaluate_source_target_without_adapter(
            source_csv_path=source_csv_path,
            target_csv_path=target_csv_path
        )
        output_prefix = "M58_LinearAdapter_SourceTarget"
        
    else:  # random_linear
        # 使用隨機初始化的 Linear Adapter（不載入任何預訓練權重）
        print(f"\n{'='*60}")
        print(f"模式: 隨機初始化 Linear Adapter (未訓練的簡單 MLP)")
        print(f"{'='*60}")
        
        evaluator_random = TextEmbeddingEvaluator(
            model_path=linear_model_path,  # 不會被使用
            ref_embedding_path=linear_ref_embedding_path,
            device='cuda',
            adapter_type='linear',
            load_adapter=False  # 不載入權重
        )
        # adapter 已經在 __init__ 中被設為 None，手動創建隨機初始化的
        evaluator_random.adapter = Adapter(c_in=512, reduction=4).to('cuda')
        evaluator_random.adapter.eval()
        print("✓ 已創建隨機初始化的 Linear Adapter (完全未訓練)\n")
        
        class_stats, detailed_results = evaluator_random.evaluate_source_target_without_adapter(
            source_csv_path=source_csv_path,
            target_csv_path=target_csv_path
        )
        output_prefix = "M58_RandomLinearAdapter_SourceTarget"
    
    # ==================== 保存詳細結果 ====================
    output_df = pd.DataFrame(detailed_results)
    output_df = output_df.sort_values('class_id').reset_index(drop=True)
    output_filename = os.path.join(output_dir, f"{output_prefix}_output.csv")
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ 詳細預測結果已保存到: {output_filename}")
    
    # ==================== 計算並保存類別統計 ====================
    result_data = []
    total_images_all = 0
    total_correct_all = 0
    
    # 獲取正確的 evaluator 實例
    if args.mode == 'with_adapter':
        current_evaluator = evaluator
    elif args.mode == 'without_adapter':
        current_evaluator = evaluator_linear
    else:  # random_linear
        current_evaluator = evaluator_random
    
    for class_id in range(len(current_evaluator.classnames)):
        class_name = current_evaluator.classnames[class_id]
        stats = class_stats[class_id]
        total_img = stats['total']
        top1_correct = stats['correct']
        top1_acc = (top1_correct / total_img * 100) if total_img > 0 else 0.0
        
        result_data.append({
            'class_id': class_id,
            'class_name': class_name,
            'total_imgNum': total_img,
            'top1_correct': top1_correct,
            'top1_acc': f"{top1_acc:.2f}%"
        })
        
        total_images_all += total_img
        total_correct_all += top1_correct
    
    # 添加平均準確率行
    avg_acc = (total_correct_all / total_images_all * 100) if total_images_all > 0 else 0.0
    result_data.append({
        'class_id': '',
        'class_name': 'Average',
        'total_imgNum': total_images_all,
        'top1_correct': total_correct_all,
        'top1_acc': f"{avg_acc:.2f}%"
    })
    
    # 保存類別統計結果
    result_df = pd.DataFrame(result_data)
    result_filename = os.path.join(output_dir, f"{output_prefix}_result.csv")
    result_df.to_csv(result_filename, index=False, encoding='utf-8-sig')
    print(f"✓ 類別統計結果已保存到: {result_filename}")
    
    # # ==================== 生成混淆矩陣 ====================
    generate_confusion_matrix(detailed_results, current_evaluator.classnames, output_dir, output_prefix)
    
    # ==================== 顯示結果預覽 ====================
    print(f"\n{'='*60}")
    print(f"評估結果")
    print(f"{'='*60}")
    print(f"\n類別統計 (前 10 個類別):")
    print(result_df.head(10).to_string(index=False))
    print(f"\n...")
    print(f"\n總體準確率: {avg_acc:.2f}%")
    print(f"  - 總樣本數: {total_images_all}")
    print(f"  - 正確數: {total_correct_all}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
