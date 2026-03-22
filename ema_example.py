class SATSelector:
    def __init__(self, num_classes=37, momentum=0.99):
        """
        純淨版 FreeMatch 動態閾值選擇器 (無手動超參數)
        """
        self.num_classes = num_classes
        self.momentum = momentum
        
        # 1. p_model: 類別期望預測機率 (估計全域邊際分佈)
        self.p_model = (torch.ones(num_classes) / num_classes).cuda()
        
        # 2. p_t: 全域動態閾值 (Global Mean Threshold)
        # 初始值設定為隨機猜測機率 (1/37)
        self.p_t = torch.tensor(1.0 / num_classes).cuda()
        
    def get_reliable_mask(self, probs_teacher, pseudo_labels):
        """
        Args:
            probs_teacher: [B, C] Teacher 輸出的預測機率
            pseudo_labels: [B] argmax 後的預測類別
        """
        with torch.no_grad():
            # 取得每個樣本的最大預測機率
            max_probs = probs_teacher.max(dim=1)[0]
            
            # ---------------------------------------------------------
            # [A] 更新全域動態閾值 (p_t)
            # ---------------------------------------------------------
            batch_max_mean = max_probs.mean()
            self.p_t = self.momentum * self.p_t + (1 - self.momentum) * batch_max_mean
            
            # ---------------------------------------------------------
            # [B] 更新類別期望分佈並計算 MaxNorm (p_norm)
            # ---------------------------------------------------------
            batch_mean_prob = probs_teacher.mean(dim=0)
            self.p_model = self.momentum * self.p_model + (1 - self.momentum) * batch_mean_prob
            
            max_p = self.p_model.max()
            p_norm = self.p_model / max_p
            
            # ---------------------------------------------------------
            # [C] 計算最終 37 類的專屬閾值
            # ---------------------------------------------------------
            # 最終閾值 = 模型總體信心 (p_t) * 類別公平性權重 (p_norm)
            class_thresholds = self.p_t * p_norm
            
            # ---------------------------------------------------------
            # [D] 分配閾值並產生 Mask
            # ---------------------------------------------------------
            sample_thresholds = class_thresholds[pseudo_labels]
            reliable_mask = max_probs > sample_thresholds
            
        return reliable_mask, class_thresholds
    
def compute_unsupervised_metrics(max_probs, pseudo_labels, all_probs, conf_threshold, num_classes=37):
    """
    計算無監督健康指標 (不涉及真實標籤，完全基於模型內部狀態)
    
    Args:
        max_probs: [B] 每個樣本的最大預測機率
        pseudo_labels: [B] 每個樣本的預測類別 (argmax)
        all_probs: [B, C] 所有樣本的完整機率分佈
        conf_threshold: [1] 或 [C] 目前的過濾閾值
        num_classes: 類別總數
        
    Returns:
        coverage_ratio: 覆蓋率 (跨越閾值的樣本比例)
        active_classes: 活躍類別數 (產生至少一個 Reliable 標籤的類別數量)
        mean_entropy: 預測熵均值 (所有樣本預測機率的資訊熵平均)
    """
    with torch.no_grad():
        # 1. 覆蓋率 (Coverage Ratio)
        if isinstance(conf_threshold, float):
            reliable_mask = max_probs > conf_threshold
        else:
            # 處理 Class-Aware Threshold 的情況
            reliable_mask = max_probs > conf_threshold[pseudo_labels]
        
        coverage_ratio = reliable_mask.float().mean().item() * 100.0
        
        # 2. 活躍類別數 (Active Class Count)
        reliable_labels = pseudo_labels[reliable_mask]
        active_classes = len(torch.unique(reliable_labels))
        
        # 3. 預測熵均值 (Mean Prediction Entropy)
        # 加上極小值防止 log(0)
        entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-8), dim=1)
        mean_entropy = entropy.mean().item()
        
    return coverage_ratio, active_classes, mean_entropy


# ... (前面的全域均值計算等程式碼保持不變) ...

# =====================================================
# 【新增】初始化全域歷史預測期望 (用於計算 L_saf)
# 假設初始時模型對 37 類的預測是完全均勻的
# =====================================================
global_expected_prob = (torch.ones(args.class_num) / args.class_num).cuda()
ema_saf = 0.99  # SAF 期望值的更新動量

# 初始化最佳準確率
acc_init = 0

# Phase 2 將 tau_base 設定為 0.9，搭配較低的 Logit Scale (如 5.0)
sat_selector = SATSelector(num_classes=args.class_num, momentum=0.99, tau_base=args.conf_thres)

# *** 以下在跑iteration的回圈內，替換原本的閾值選擇邏輯 ***
# --- 替換原本的 conf_threshold = torch.clamp(...) 以及 reliable_mask = ... ---
reliable_mask, current_class_thresholds = sat_selector.get_reliable_mask(probs_teacher, batch_pseudo_labels)

# --------------------------------------------------
# [C] IM Loss 升級：Entropy Minimization - Global SAF
# --------------------------------------------------
im_loss = torch.tensor(0.0).cuda()
if args.ent:
    softmax_out = F.softmax(logits_original, dim=1)  # [B, C]
    
    # 1. 局部銳利化 (使用你原本的寫法)
    entropy_loss = torch.mean(loss.Entropy(softmax_out))
    
    # 2. 全域公平性 (SAF)
    current_batch_prob = softmax_out.mean(dim=0)  # [C]
    combined_global_prob = ema_saf * global_expected_prob.detach() + (1 - ema_saf) * current_batch_prob
    
    # 將 [37] 擴充為 [1, 37] 餵入你的函數，算出後再 squeeze 壓回純量
    saf_loss = loss.Entropy(combined_global_prob.unsqueeze(0)).squeeze()
    
    # 3. 結合 (一正一負)
    total_im_loss = entropy_loss - saf_loss
    
    # 4. 更新歷史期望 (無梯度)
    with torch.no_grad():
        global_expected_prob = ema_saf * global_expected_prob + (1 - ema_saf) * current_batch_prob.detach()
        
    im_loss = total_im_loss * args.ent_par
    
losses += im_loss

# ... (Epoch 追蹤變數等程式碼) ...
# =================================================
# 計算並監控無監督健康指標
# =================================================
cov_ratio, act_classes, mean_ent = compute_unsupervised_metrics(
    batch_max_probs, batch_pseudo_labels, probs_teacher, conf_threshold, args.class_num
)

# =================================================
# 進度條更新 (加入新指標)
# =================================================
pbar.update(1)
pbar.set_postfix({
    'Loss': f'{losses.item():.3f}',
    'CE': f'{ce_loss.item():.3f}',
    'IM': f'{im_loss.item():.3f}',
    'Cov%': f'{cov_ratio:.1f}',
    'ActCls': f'{act_classes}',
    'H': f'{mean_ent:.2f}'
})