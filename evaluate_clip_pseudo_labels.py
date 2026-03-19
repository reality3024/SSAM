"""
評估 CLIP Text-Image Alignment 的 Pseudo-label 準確率
檢查使用 cosine similarity 生成的 pseudo-label 與真實標籤的對應情況
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data_list import ImageList_idx
from sklearn.metrics import confusion_matrix, classification_report
import clip
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_prediction_conflict(outputs_test, pseudo_labels, max_sim_probs, iter_num, 
                               verbose=True, log_file=None):
    """
    分析 Classifier 預測與 Text Matcher 預測之間的衝突率
    
    Args:
        outputs_test: Classifier 的輸出 logits [B, num_classes]
        pseudo_labels: Text Matcher 的預測標籤 [B]
        max_sim_probs: Text Matcher 的信心度 [B]
        iter_num: 當前迭代次數
        verbose: 是否打印詳細資訊
        log_file: 日誌文件對象（可選）
    
    Returns:
        dict: 包含分析結果的字典
    """
    with torch.no_grad():
        # 1. 取得 Classifier 的預測
        cls_probs = F.softmax(outputs_test, dim=1)
        max_cls_prob, cls_pred = torch.max(cls_probs, dim=1)
        
        # 2. 計算一致性
        agreement = (cls_pred == pseudo_labels).float()
        agreement_rate = agreement.mean().item()
        
        # 3. 統計雙方都預測特定類別的比例（例如 Class 0）
        both_class_0 = ((cls_pred == 0) & (pseudo_labels == 0)).float().mean().item()
        
        # 4. 統計不一致的樣本中，各自的信心度
        disagreement_mask = (cls_pred != pseudo_labels)
        if disagreement_mask.sum() > 0:
            cls_conf_on_disagree = max_cls_prob[disagreement_mask].mean().item()
            text_conf_on_disagree = max_sim_probs[disagreement_mask].mean().item()
        else:
            cls_conf_on_disagree = 0.0
            text_conf_on_disagree = 0.0
        
        # 5. 計算平均信心度
        avg_cls_conf = max_cls_prob.mean().item()
        avg_text_conf = max_sim_probs.mean().item()
        
        # 構建結果字典
        results = {
            'agreement_rate': agreement_rate,
            'both_class_0_rate': both_class_0,
            'avg_classifier_conf': avg_cls_conf,
            'avg_text_conf': avg_text_conf,
            'disagreement_cls_conf': cls_conf_on_disagree,
            'disagreement_text_conf': text_conf_on_disagree,
            'num_disagreements': disagreement_mask.sum().item(),
            'total_samples': len(cls_pred)
        }
        
        # 打印詳細資訊
        if verbose and iter_num % 100 == 0:
            log_str = f"\n[Analysis] Conflict Check (Iter {iter_num}):\n"
            log_str += f"  - Agreement Rate (兩者預測相同): {agreement_rate*100:.2f}%\n"
            log_str += f"  - Disagreements: {results['num_disagreements']}/{results['total_samples']}\n"
            log_str += f"  - Both Predict Class 0: {both_class_0*100:.2f}%\n"
            log_str += f"  - Classifier Avg Conf: {avg_cls_conf:.4f}\n"
            log_str += f"  - Text Matcher Avg Conf: {avg_text_conf:.4f}\n"
            
            if disagreement_mask.sum() > 0:
                log_str += f"  - Classifier Conf (不一致時): {cls_conf_on_disagree:.4f}\n"
                log_str += f"  - Text Matcher Conf (不一致時): {text_conf_on_disagree:.4f}\n"
            
            if agreement_rate > 0.9:
                log_str += "  🚨 結論：高度一致。VLM 可能只是在重複 Classifier 的預測，沒有提供新資訊。\n"
            elif agreement_rate < 0.5:
                log_str += "  ⚠️  結論：預測差異大。兩個模型對樣本有不同理解。\n"
            else:
                log_str += "  ✓ 結論：適度差異。VLM 提供了互補資訊。\n"
            
            print(log_str)
            if log_file is not None:
                log_file.write(log_str)
                log_file.flush()
        
        return results


def check_gradient_flow(netF, netB, iter_num, verbose=True, log_file=None):
    """
    檢查模型各部分的梯度流動情況
    
    Args:
        netF: Feature Extractor (Backbone)
        netB: Bottleneck
        iter_num: 當前迭代次數
        verbose: 是否打印詳細資訊
        log_file: 日誌文件對象（可選）
    
    Returns:
        dict: 包含梯度統計的字典
    """
    # 檢查 Encoder (Backbone)
    grad_norm_F = 0.0
    grad_count_F = 0
    max_grad_F = 0.0
    min_grad_F = float('inf')
    
    for name, p in netF.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            grad_norm_F += param_norm
            grad_count_F += 1
            max_grad_F = max(max_grad_F, param_norm)
            min_grad_F = min(min_grad_F, param_norm)
    
    # 檢查 Bottleneck
    grad_norm_B = 0.0
    grad_count_B = 0
    max_grad_B = 0.0
    min_grad_B = float('inf')
    
    for name, p in netB.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            grad_norm_B += param_norm
            grad_count_B += 1
            max_grad_B = max(max_grad_B, param_norm)
            min_grad_B = min(min_grad_B, param_norm)
    
    # 計算平均梯度
    avg_grad_F = grad_norm_F / grad_count_F if grad_count_F > 0 else 0.0
    avg_grad_B = grad_norm_B / grad_count_B if grad_count_B > 0 else 0.0
    
    # 構建結果字典
    results = {
        'encoder_grad_norm': grad_norm_F,
        'encoder_avg_grad': avg_grad_F,
        'encoder_max_grad': max_grad_F if max_grad_F != 0.0 else 0.0,
        'encoder_min_grad': min_grad_F if min_grad_F != float('inf') else 0.0,
        'bottleneck_grad_norm': grad_norm_B,
        'bottleneck_avg_grad': avg_grad_B,
        'bottleneck_max_grad': max_grad_B if max_grad_B != 0.0 else 0.0,
        'bottleneck_min_grad': min_grad_B if min_grad_B != float('inf') else 0.0,
        'encoder_param_count': grad_count_F,
        'bottleneck_param_count': grad_count_B
    }
    
    # 打印詳細資訊
    if verbose and iter_num % 100 == 0:
        log_str = f"\n[Analysis] Gradient Check (Iter {iter_num}):\n"
        log_str += f"  Encoder (Feature Extractor):\n"
        log_str += f"    - Total Gradient Norm: {grad_norm_F:.6f}\n"
        log_str += f"    - Avg Gradient Norm: {avg_grad_F:.6f}\n"
        log_str += f"    - Max Gradient Norm: {max_grad_F:.6f}\n"
        log_str += f"    - Min Gradient Norm: {min_grad_F:.6f}\n"
        log_str += f"    - Param Count: {grad_count_F}\n"
        
        log_str += f"  Bottleneck:\n"
        log_str += f"    - Total Gradient Norm: {grad_norm_B:.6f}\n"
        log_str += f"    - Avg Gradient Norm: {avg_grad_B:.6f}\n"
        log_str += f"    - Max Gradient Norm: {max_grad_B:.6f}\n"
        log_str += f"    - Min Gradient Norm: {min_grad_B:.6f}\n"
        log_str += f"    - Param Count: {grad_count_B}\n"
        
        # 診斷
        if grad_norm_F < 1e-4:
            log_str += "  🚨 結論：Encoder 幾乎沒有收到更新訊號（梯度過小）。\n"
        elif grad_norm_F > 1e3:
            log_str += "  ⚠️  結論：Encoder 梯度過大，可能有梯度爆炸風險。\n"
        else:
            log_str += "  ✓ 結論：Encoder 梯度正常。\n"
        
        if grad_norm_B < 1e-4:
            log_str += "  🚨 結論：Bottleneck 幾乎沒有收到更新訊號（梯度過小）。\n"
        elif grad_norm_B > 1e3:
            log_str += "  ⚠️  結論：Bottleneck 梯度過大，可能有梯度爆炸風險。\n"
        else:
            log_str += "  ✓ 結論：Bottleneck 梯度正常。\n"
        
        print(log_str)
        if log_file is not None:
            log_file.write(log_str)
            log_file.flush()
    
    return results


def check_text_similarity(args):
    text_emb_path = osp.join(args.output_dir_src, 'M58_text_embeddings.pt')
    if not osp.exists(text_emb_path):
        raise FileNotFoundError(f"找不到 text embeddings: {text_emb_path}")
    
    embeddings_dict = torch.load(text_emb_path, map_location='cuda')
    text_features = embeddings_dict['text_features'].cuda()
    classnames = embeddings_dict['classnames']

    # 計算相似度矩陣 [79, 79]
    sim_matrix = text_features @ text_features.t()

    # 1. 畫出熱力圖
    plt.figure(figsize=(20, 18))
    sns.heatmap(sim_matrix.cpu().numpy(), cmap='viridis', vmin=0.0, vmax=1.0)
    plt.title("Text Embedding Similarity Matrix")
    plt.savefig(os.path.join(args.output_dir_src, "text_similarity_matrix.png"))
    plt.close()

    mask = torch.eye(len(text_features)).cuda()
    sim_matrix_masked = sim_matrix - mask
    
    max_sim, max_idx = torch.max(sim_matrix_masked, dim=1)
    
    print("\n[Diagnosis 1] Text Prompt Confusion Analysis:")
    print("Top 10 Most Similar Class Pairs (Risk of Confusion):")
    
    # 找出全域相似度最高的前 10 對
    pairs = []
    for i in range(len(text_features)):
        for j in range(i + 1, len(text_features)):
            pairs.append((sim_matrix[i, j].item(), i, j))
    
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    for i in range(10):
        score, idx1, idx2 = pairs[i]
        print(f"  - Pair: {classnames[idx1]} <--> {classnames[idx2]}")
        print(f"    Similarity: {score:.4f}")

    # 特別檢查 Class 74 (高信心錯誤) 跟誰最像
    target_cls = 0
    most_sim_idx = max_idx[target_cls].item()
    print(f"\n[Spot Check] Class {target_cls} ({classnames[target_cls]}) most similar text is:")
    print(f"  -> Class {most_sim_idx} ({classnames[most_sim_idx]}) with score {max_sim[target_cls].item():.4f}")

class CLIP_Backbone_Wrapper(nn.Module):
    """CLIP Backbone Wrapper - 與 ssaf_adaptation.py 相同"""
    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model
        self.in_features = 2048
        
    def forward(self, x, return_both=False):
        x = x.type(self.visual.conv1.weight.dtype)
        
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        x = self.visual.layer4(x)
        
        feat_2048 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        
        if not return_both:
            return feat_2048
        
        feat_512 = self.visual.attnpool(x)
        return feat_2048, feat_512

def evaluate_pseudo_labels(args):
    """評估 CLIP pseudo-label 的準確率"""
    
    print(f"\n{'='*80}")
    print(f"評估 CLIP Pseudo-label 準確率")
    print(f"{'='*80}\n")
    
    # ===== 1. 載入模型 =====
    print("步驟 1: 載入 CLIP 模型和權重")
    print("-" * 80)
    
    clip_model, _ = clip.load(args.net, device='cuda')
    clip_model.float()
    netF = CLIP_Backbone_Wrapper(clip_model.visual).cuda()
    netF.eval()
    
    # 載入訓練好的權重
    modelpath = osp.join(args.output_dir_src, 'source_F.pt')
    if osp.exists(modelpath):
        pretrained_dict = torch.load(modelpath)
        sample_key = list(pretrained_dict.keys())[0]
        if not sample_key.startswith('visual.'):
            new_state_dict = {f"visual.{k}": v for k, v in pretrained_dict.items()}
        else:
            new_state_dict = pretrained_dict
        netF.load_state_dict(new_state_dict, strict=False)
        print(f"✓ 已載入模型權重: {modelpath}")
    else:
        print(f"⚠️  找不到 {modelpath}，使用 CLIP 預訓練權重")
    print()
    
    # ===== 2. 載入 Text Embeddings =====
    print("步驟 2: 載入 Text Embeddings")
    print("-" * 80)
    
    text_emb_path = osp.join(args.output_dir_src, 'M58_text_embeddings.pt')
    if not osp.exists(text_emb_path):
        raise FileNotFoundError(f"找不到 text embeddings: {text_emb_path}")
    
    embeddings_dict = torch.load(text_emb_path, map_location='cuda')
    text_features = embeddings_dict['text_features'].cuda()
    classnames = embeddings_dict['classnames']
    
    print(f"✓ Text embeddings shape: {text_features.shape}")
    print(f"✓ 類別數量: {len(classnames)}")
    print(f"✓ 已正規化: {(text_features.norm(dim=-1) - 1.0).abs().max().item() < 0.01}")
    print()
    
    # ===== 3. 準備測試資料 =====
    print("步驟 3: 載入測試資料")
    print("-" * 80)
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    txt_test = open(args.test_dset_path).readlines()
    test_dataset = ImageList_idx(txt_test, transform=test_transform, root=f'data/{args.dset}/')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.worker, drop_last=False)
    
    print(f"✓ 測試資料集大小: {len(test_dataset)}")
    print(f"✓ Batch size: {args.batch_size}")
    print(f"✓ 總 batch 數: {len(test_loader)}")
    print()
    
    # ===== 4. 生成 Pseudo-labels 並評估 =====
    print("步驟 4: 生成 Pseudo-labels 並評估準確率")
    print("-" * 80)
    
    all_pseudo_labels = []
    all_true_labels = []
    all_confidences = []
    all_similarities = []
    all_paths = []
    
    # 正規化 text features
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    
    with torch.no_grad():
        for batch_idx, (images, labels, idx, paths) in enumerate(tqdm(test_loader, desc="處理測試資料")):
            images = images.cuda()
            labels = labels.cuda()
            
            # 提取 512 維 CLIP 特徵
            _, feat_512 = netF(images, return_both=True)
            
            # 正規化圖片特徵
            feat_512_norm = feat_512 / feat_512.norm(dim=-1, keepdim=True)
            
            # 計算 cosine similarity
            similarities = feat_512_norm @ text_features_norm.t()  # [B, num_classes]
            
            # 使用 logit scale 放大相似度
            logits = similarities * args.logit_scale
            
            # 獲取 pseudo-label（最高相似度的類別）
            sim_probs = F.softmax(logits, dim=1)
            confidences, pseudo_labels = torch.max(sim_probs, dim=1)
            
            # 儲存結果
            all_pseudo_labels.extend(pseudo_labels.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_similarities.append(similarities.cpu().numpy())
            all_paths.extend(paths)
    
    all_pseudo_labels = np.array(all_pseudo_labels)
    all_true_labels = np.array(all_true_labels)
    all_confidences = np.array(all_confidences)
    all_similarities = np.concatenate(all_similarities, axis=0)
    
    print(f"\n✓ 完成！處理了 {len(all_true_labels)} 個樣本\n")
    
    # ===== 5. 計算整體準確率 =====
    print(f"{'='*80}")
    print(f"整體評估結果")
    print(f"{'='*80}\n")
    
    correct = (all_pseudo_labels == all_true_labels).sum()
    total = len(all_true_labels)
    accuracy = correct / total * 100
    
    print(f"整體準確率: {accuracy:.2f}% ({correct}/{total})")
    print(f"平均信心度: {all_confidences.mean():.4f}")
    print(f"信心度範圍: [{all_confidences.min():.4f}, {all_confidences.max():.4f}]")
    print()
    
    # 按信心度分段統計
    confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"按信心度分段統計:")
    print(f"{'信心度範圍':<20} {'樣本數':<10} {'準確率':<10}")
    print("-" * 50)
    for i in range(len(confidence_bins)-1):
        low, high = confidence_bins[i], confidence_bins[i+1]
        mask = (all_confidences >= low) & (all_confidences < high) if i < len(confidence_bins)-2 else (all_confidences >= low)
        if mask.sum() > 0:
            bin_acc = (all_pseudo_labels[mask] == all_true_labels[mask]).mean() * 100
            print(f"[{low:.1f}, {high:.1f})        {mask.sum():<10} {bin_acc:.2f}%")
    print()
    
    # ===== 6. 每個類別的準確率 =====
    print(f"{'='*80}")
    print(f"每個類別的詳細評估")
    print(f"{'='*80}\n")
    
    per_class_correct = []
    per_class_total = []
    per_class_acc = []
    
    print(f"{'類別ID':<8} {'類別名稱':<35} {'準確率':<12} {'正確/總數':<15} {'平均信心度':<12}")
    print("-" * 100)
    
    for class_id in range(len(classnames)):
        mask = (all_true_labels == class_id)
        if mask.sum() > 0:
            class_correct = (all_pseudo_labels[mask] == all_true_labels[mask]).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total * 100
            class_conf = all_confidences[mask].mean()
            
            per_class_correct.append(class_correct)
            per_class_total.append(class_total)
            per_class_acc.append(class_acc)
            
            classname_short = classnames[class_id][:33]
            print(f"{class_id:<8} {classname_short:<35} {class_acc:>6.2f}%      {class_correct:>4}/{class_total:<8} {class_conf:.4f}")
    
    print()
    print(f"平均每類準確率 (Average Per-Class Accuracy): {np.mean(per_class_acc):.2f}%")
    print()
    
    # ===== 7. 混淆矩陣（只顯示前10個類別）=====
    if len(classnames) <= 20:
        print(f"{'='*80}")
        print(f"混淆矩陣")
        print(f"{'='*80}\n")
        
        cm = confusion_matrix(all_true_labels, all_pseudo_labels)
        
        print(f"混淆矩陣 (行=True Label, 列=Predicted Label):")
        print(f"對角線數值代表正確預測的數量\n")
        
        # 顯示類別名稱（縮短）
        header = "True\\Pred  "
        for i in range(min(10, len(classnames))):
            header += f"{i:>6} "
        print(header)
        print("-" * (12 + 7 * min(10, len(classnames))))
        
        for i in range(min(10, len(classnames))):
            row = f"Class {i:<3}  "
            for j in range(min(10, len(classnames))):
                row += f"{cm[i,j]:>6} "
            print(row)
        print()
    
    # ===== 8. 錯誤分析：找出最容易混淆的類別對 =====
    print(f"{'='*80}")
    print(f"錯誤分析：最容易混淆的類別對 (Top 20)")
    print(f"{'='*80}\n")
    
    # 統計每對類別的混淆次數
    confusion_pairs = {}
    for true_label, pred_label in zip(all_true_labels, all_pseudo_labels):
        if true_label != pred_label:
            pair = (int(true_label), int(pred_label))
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # 排序並顯示 top 20
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print(f"{'排名':<5} {'真實類別':<35} {'預測類別':<35} {'混淆次數':<10}")
    print("-" * 100)
    for rank, ((true_id, pred_id), count) in enumerate(sorted_pairs, 1):
        true_name = classnames[true_id][:33]
        pred_name = classnames[pred_id][:33]
        print(f"{rank:<5} {true_name:<35} {pred_name:<35} {count:<10}")
    print()
    
    # ===== 9. 儲存錯誤樣本的詳細資訊 =====
    print(f"{'='*80}")
    print(f"儲存錯誤樣本詳細資訊")
    print(f"{'='*80}\n")
    
    error_mask = (all_pseudo_labels != all_true_labels)
    error_indices = np.where(error_mask)[0]
    
    output_dir = args.output_dir if hasattr(args, 'output_dir') else 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    error_file = osp.join(output_dir, 'pseudo_label_errors.txt')
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write(f"CLIP Pseudo-label 錯誤樣本分析\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"整體準確率: {accuracy:.2f}% ({correct}/{total})\n")
        f.write(f"錯誤樣本數: {len(error_indices)}/{total}\n\n")
        f.write(f"{'='*100}\n\n")
        
        f.write(f"錯誤樣本詳細資訊:\n")
        f.write(f"{'-'*100}\n")
        f.write(f"{'Idx':<6} {'信心度':<10} {'真實類別':<35} {'預測類別':<35} 路徑\n")
        f.write(f"{'-'*100}\n")
        
        for idx in error_indices[:500]:  # 只儲存前 500 個錯誤
            true_id = all_true_labels[idx]
            pred_id = all_pseudo_labels[idx]
            conf = all_confidences[idx]
            path = all_paths[idx]
            
            true_name = classnames[true_id][:33]
            pred_name = classnames[pred_id][:33]
            
            f.write(f"{idx:<6} {conf:<10.4f} {true_name:<35} {pred_name:<35} {path}\n")
        
        if len(error_indices) > 500:
            f.write(f"\n... 還有 {len(error_indices) - 500} 個錯誤樣本未顯示\n")
    
    print(f"✓ 錯誤樣本詳細資訊已儲存至: {error_file}")
    print()
    
    # ===== 10. 儲存所有樣本的預測結果 =====
    result_file = osp.join(output_dir, 'all_predictions.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"所有樣本的預測結果\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"{'Idx':<6} {'正確':<6} {'信心度':<10} {'真實ID':<8} {'預測ID':<8} {'真實類別':<30} {'預測類別':<30} 路徑\n")
        f.write(f"{'-'*150}\n")
        
        for idx in range(len(all_true_labels)):
            is_correct = "✓" if all_true_labels[idx] == all_pseudo_labels[idx] else "✗"
            true_id = all_true_labels[idx]
            pred_id = all_pseudo_labels[idx]
            conf = all_confidences[idx]
            path = all_paths[idx].split('/')[-1] if '/' in all_paths[idx] else all_paths[idx]
            
            true_name = classnames[true_id][:28]
            pred_name = classnames[pred_id][:28]
            
            f.write(f"{idx:<6} {is_correct:<6} {conf:<10.4f} {true_id:<8} {pred_id:<8} {true_name:<30} {pred_name:<30} {path}\n")
    
    print(f"✓ 所有預測結果已儲存至: {result_file}")
    print()
    
    # ===== 11. 總結 =====
    print(f"{'='*80}")
    print(f"評估總結")
    print(f"{'='*80}\n")
    print(f"✓ 整體準確率: {accuracy:.2f}%")
    print(f"✓ 平均每類準確率: {np.mean(per_class_acc):.2f}%")
    print(f"✓ 平均信心度: {all_confidences.mean():.4f}")
    print(f"✓ 結果已儲存至: {output_dir}/")
    print()
    
    # 返回評估結果
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'mean_confidence': all_confidences.mean(),
        'all_pseudo_labels': all_pseudo_labels,
        'all_true_labels': all_true_labels,
        'all_confidences': all_confidences,
        'confusion_matrix': confusion_matrix(all_true_labels, all_pseudo_labels)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='評估 CLIP Pseudo-label 準確率')
    
    # 基本參數
    parser.add_argument('--dset', type=str, default='M58', help='資料集名稱')
    parser.add_argument('--net', type=str, default='RN101', help='CLIP 模型架構')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--worker', type=int, default=4, help='num workers')
    parser.add_argument('--logit_scale', type=float, default=20.0, help='logit scale for similarity')
    
    # 路徑參數
    parser.add_argument('--output_dir_src', type=str, default='ckps/source/uda/M58/C',
                       help='source model 目錄')
    parser.add_argument('--test_dset_path', type=str, 
                       default='data/M58/CAD_ratioFilter_79_list.txt',
                       help='測試資料集路徑')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='輸出目錄')
    
    # 分析功能參數
    parser.add_argument('--mode', type=str, default='test_analysis', 
                       choices=['evaluate', 'text_similarity', 'test_analysis'],
                       help='執行模式: evaluate=完整評估, text_similarity=文本相似度分析, test_analysis=測試分析函數')
    parser.add_argument('--analyze_conflict', action='store_true',
                       help='在評估時執行預測衝突分析')
    parser.add_argument('--analyze_gradients', action='store_true',
                       help='在評估時檢查梯度流動（需要訓練模式）')
    
    args = parser.parse_args()
    
    # 根據模式執行不同功能
    if args.mode == 'text_similarity':
        # 執行文本相似度分析
        check_text_similarity(args)
    
    elif args.mode == 'test_analysis':
        # 測試分析函數（創建假數據）
        print("\n" + "="*80)
        print("測試分析函數")
        print("="*80 + "\n")
        
        # 創建假數據來測試分析函數
        batch_size = 32
        num_classes = 79
        
        # 模擬 Classifier 輸出和 Text Matcher 預測
        outputs_test = torch.randn(batch_size, num_classes).cuda()
        pseudo_labels = torch.randint(0, num_classes, (batch_size,)).cuda()
        max_sim_probs = torch.rand(batch_size).cuda() * 0.5 + 0.5  # 0.5-1.0 範圍
        
        print("1. 測試 analyze_prediction_conflict():")
        print("-" * 80)
        conflict_results = analyze_prediction_conflict(
            outputs_test=outputs_test,
            pseudo_labels=pseudo_labels,
            max_sim_probs=max_sim_probs,
            iter_num=100,
            verbose=True,
            log_file=None
        )
        print("\n返回結果:")
        for key, value in conflict_results.items():
            print(f"  {key}: {value}")
        
        print("\n\n2. 測試 check_gradient_flow():")
        print("-" * 80)
        
        # 載入模型來測試梯度檢查
        clip_model, _ = clip.load(args.net, device='cuda')
        clip_model.float()  # 轉換為全精度
        netF = CLIP_Backbone_Wrapper(clip_model.visual).cuda()
        netF.train()  # 設為訓練模式以啟用梯度
        
        # 創建假的 Bottleneck
        import network
        netB = network.feat_bootleneck(type='bn', feature_dim=2048, bottleneck_dim=256).cuda()
        netB.train()  # 設為訓練模式
        
        # 模擬一個 forward-backward pass
        x = torch.randn(4, 3, 224, 224).cuda()
        feat = netF(x)
        bottleneck_feat = netB(feat)
        loss = bottleneck_feat.sum()
        loss.backward()
        
        grad_results = check_gradient_flow(
            netF=netF,
            netB=netB,
            iter_num=100,
            verbose=True,
            log_file=None
        )
        print("\n返回結果:")
        for key, value in grad_results.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print("測試完成！這些函數已準備好在 ssaf_adaptation.py 中使用")
        print("="*80 + "\n")
    
    else:
        # 執行完整評估
        print("\n開始執行 CLIP Pseudo-label 評估...")
        results = evaluate_pseudo_labels(args)
        
        # 如果啟用衝突分析，創建一個簡單的示例
        if args.analyze_conflict:
            print("\n" + "="*80)
            print("執行預測衝突分析示例")
            print("="*80 + "\n")
            print("注意：完整的衝突分析應在訓練循環中執行（ssaf_adaptation.py）")
            print("這裡僅展示如何使用這個函數。\n")
            
            # 使用評估結果創建示例
            sample_size = min(64, len(results['all_pseudo_labels']))
            sample_pseudo_labels = torch.tensor(results['all_pseudo_labels'][:sample_size]).cuda()
            sample_confidences = torch.tensor(results['all_confidences'][:sample_size]).cuda()
            
            # 創建假的 classifier 輸出（實際使用時這會是真實的 classifier 輸出）
            fake_classifier_outputs = torch.randn(sample_size, 79).cuda()
            
            conflict_results = analyze_prediction_conflict(
                outputs_test=fake_classifier_outputs,
                pseudo_labels=sample_pseudo_labels,
                max_sim_probs=sample_confidences,
                iter_num=0,
                verbose=True,
                log_file=None
            )
        
        if args.analyze_gradients:
            print("\n⚠️  梯度分析需要在訓練模式下執行")
            print("請在 ssaf_adaptation.py 的訓練循環中使用 check_gradient_flow()")
            print("參考用法:")
            print("  grad_results = check_gradient_flow(netF, netB, iter_num, verbose=True, log_file=args.out_file)")