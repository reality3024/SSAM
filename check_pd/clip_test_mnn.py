"""
CLIP Adaptation with Projector - K-Means Pseudo Labeling
只使用 im_loss + loss_ce (K-Means pseudo labels)
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
import loss
from torch.utils.data import DataLoader
from data_list import ImageList_idx
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from clip import clip


class MLPProjector(nn.Module):
    """MLP Projector - 與 clip_source_model_projectOnly.py 完全相同"""
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


def load_clip_to_cpu(backbone):
    """載入 CLIP 模型"""
    model, _ = clip.load(backbone, device="cpu", download_root=os.path.expanduser("~/.cache/clip"))
    return model


def op_copy(optimizer):
    """複製優化器的學習率"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """學習率調度器"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224):
    """訓練時的數據增強（與 image_target.py 一致）"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224):
    """測試時的 transform（與 image_target.py 一致）"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    """載入數據集"""
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    root_path = f'data/{args.dset}/'

    # 使用 image_train() 和 image_test()（不是 Augmentation）
    dsets["target"] = ImageList_idx(txt_tar, transform=image_test(), root=root_path)
    dset_loaders["target"] = DataLoader(
        dsets["target"], 
        batch_size=train_bs, 
        shuffle=False, 
        num_workers=args.worker, 
        drop_last=False
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), root=root_path)
    dset_loaders["test"] = DataLoader(
        dsets["test"], 
        batch_size=train_bs * 3, 
        shuffle=False, 
        num_workers=args.worker, 
        drop_last=False
    )

    return dset_loaders


def load_source_prototypes(path='/mnt/backups/andycw/CLIP/output/m58/class_centroids.pth'):
    """載入預先計算好的 Source 類別特徵中心"""
    print(f'\n{"="*60}')
    print(f'載入 Source Visual Prototypes: {path}')
    
    data = torch.load(path)
    source_prototypes = data['centroids']  # [79, 512]
    classnames = data['classnames']
    
    # L2 正規化
    source_prototypes = F.normalize(source_prototypes, dim=-1)
    
    print(f'✓ Source Visual Prototypes 載入成功')
    print(f'  - shape: {source_prototypes.shape}')
    print(f'  - num_classes: {len(classnames)}')
    print(f'  - 已正規化: {(source_prototypes.norm(dim=-1) - 1.0).abs().max().item() < 0.01}')
    print(f'{"="*60}\n')
    
    return source_prototypes.cuda(), classnames

def get_cross_domain_mnn(target_features, source_prototypes, k_t2p=1, k_p2t=10):
    """
    計算 Target 樣本與 Source Prototypes 之間的 MNN 關係
    """
    N = target_features.shape[0]
    C = source_prototypes.shape[0]
    
    # 這裡假設傳入的 features 已經做過 L2 Normalize
    # 計算 Cosine Similarity Matrix (N, C)
    sim_matrix = torch.mm(target_features, source_prototypes.t())
    
    # 條件一 (T2P): 每個 Target 樣本找最近的 k_t2p 個 Prototypes
    _, topk_proto_indices = torch.topk(sim_matrix, k=k_t2p, dim=1)
    t2p_graph = torch.zeros((N, C), device=target_features.device, dtype=torch.bool)
    t2p_graph.scatter_(1, topk_proto_indices, True)
    
    # 條件二 (P2T): 每個 Prototype 找最近的 k_p2t 個 Target 樣本
    _, topk_target_indices = torch.topk(sim_matrix, k=k_p2t, dim=0)
    
    p2t_graph = torch.zeros((N, C), device=target_features.device, dtype=torch.bool)
    p2t_graph_t = p2t_graph.t()              
    topk_target_indices_t = topk_target_indices.t() 
    p2t_graph_t.scatter_(1, topk_target_indices_t, True)
    p2t_graph = p2t_graph_t.t()              
    
    # MNN 雙向交集
    mnn_mask = t2p_graph & p2t_graph
    
    return mnn_mask

def get_centered_shot_pseudo_labels(features, source_prototypes):
    """
    結合 Global Centering 與 SHOT 的偽標籤演算法
    """
    # ==========================================
    # 1. 關鍵防護：消除全域偏移 (回到乾淨的相對空間)
    # ==========================================
    t_mean = features.mean(dim=0, keepdim=True)
    s_mean = source_prototypes.mean(dim=0, keepdim=True)
    
    feat_c = features - t_mean
    proto_c = source_prototypes - s_mean
    
    # 正規化到單位球面上
    feat_c = F.normalize(feat_c, dim=1)
    proto_c = F.normalize(proto_c, dim=1)

    # ==========================================
    # 2. SHOT 演算法 (在乾淨空間中執行)
    # ==========================================
    # 計算初始機率 (使用 scale=15.0 讓機率分佈銳利化)
    sim_matrix = feat_c @ proto_c.T  # [2214, 79]
    probs = F.softmax(sim_matrix * 15.0, dim=1) 
    
    # 類別平衡權重 (防止某些零件特徵太強，把大家都吸走)
    weight = probs ** 2 / probs.sum(dim=0)
    probs_balanced = (weight.T / weight.sum(dim=1)).T 
    
    # 捏出 Target 專屬的 79 顆質心
    target_centroids = probs_balanced.T @ feat_c
    target_centroids = F.normalize(target_centroids, dim=1)
    
    # ==========================================
    # 3. 最終配對
    # ==========================================
    # 讓 Target 樣本去跟「Target 自己的質心」算距離
    final_sim = feat_c @ target_centroids.T
    final_pseudo_labels = final_sim.argmax(dim=1)
    
    return final_pseudo_labels

def update_pseudo_labels(netF, netP, target_loader, source_visual_prototypes, num_classes=79, threshold=0.9, all_features=None, all_true_labels=None):
    dataset_size = all_features.shape[0]
    # 1. 獲取數據集大小並預先分配空間
    # dataset_size = len(target_loader.dataset)
    # all_features = torch.zeros(dataset_size, 512)  # 預先分配固定大小
    # all_true_labels = torch.zeros(dataset_size, dtype=torch.long)  # 收集真實標籤
    
    # netP.eval()  # 設為 eval 模式
    
    # with torch.no_grad():
    #     for data in tqdm(target_loader, desc="提取特徵", leave=False):
    #         imgs, labels, indices, _ = data  # 取出 imgs, labels, indices
    #         imgs = imgs.cuda()
            
    #         # CLIP Encoder → Projector → Normalize
    #         feat_512 = netF(imgs.type(netF.conv1.weight.dtype))
    #         feat_projected = netP(feat_512)
    #         feat_norm = F.normalize(feat_projected, dim=-1)
            
    #         # 使用 indices 明確指定位置，確保對應關係
    #         all_features[indices] = feat_norm.cpu()
    #         all_true_labels[indices] = labels.cpu()  # 收集真實標籤
    
    # all_features_np = all_features.numpy()  # [N, 512]
    # print(f'✓ 特徵提取完成: {all_features_np.shape}')
    # print(f'  使用明確索引 (indices) 確保樣本對應正確')
    
    # 直接使用 SHOT 方法生成 pseudo labels（不需要置中或 MNN 篩選）
    pseudo_labels = get_centered_shot_pseudo_labels(all_features.cuda(), source_visual_prototypes.cuda())
    
    # 接受所有樣本（不使用 confidence mask 篩選）
    confidence_mask = torch.ones(dataset_size, dtype=torch.bool, device='cuda')
    
    # 計算整體純度
    correct_count = (pseudo_labels == all_true_labels.cuda()).sum().item()
    purity = (correct_count / dataset_size) * 100
    print(f'>>> Pseudo Label 純度: {purity:.2f}% ({correct_count}/{dataset_size})')
    
    # 動態更新 Source Prototypes（使用所有樣本）
    momentum = 0.99
    updated_prototypes = source_visual_prototypes.clone()
    
    for c in range(num_classes):
        class_mask = (pseudo_labels == c)
        
        if class_mask.sum() > 0:
            # 取出被預測為類別 c 的 Target 樣本特徵並計算平均
            target_class_feat = all_features[class_mask.cpu()].mean(dim=0)
            target_class_feat = F.normalize(target_class_feat.cuda(), dim=-1)
            
            # EMA 更新 Prototype
            updated_prototypes[c] = momentum * updated_prototypes[c] + (1.0 - momentum) * target_class_feat
    
    # 確保 Prototypes 依然是單位向量
    updated_prototypes = F.normalize(updated_prototypes, dim=-1)

    return pseudo_labels, confidence_mask, updated_prototypes


def cal_acc(loader, netF, netP, source_visual_prototypes, flag=False, logit_scale=100.0):
    """使用 CLIP + Projector + Source Prototypes 計算準確率"""
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            
            # CLIP Encoder → Projector
            feat_512 = netF(inputs.type(netF.conv1.weight.dtype))
            feat_projected = netP(feat_512)
            feat_norm = F.normalize(feat_projected, dim=-1)
            
            # 計算與 source prototypes 的相似度
            source_prototypes_norm = F.normalize(source_visual_prototypes, dim=-1)
            outputs = logit_scale * (feat_norm @ source_prototypes_norm.T)  # [B, 79]
            
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_target(args):
    """訓練主函數"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dset_loaders = data_load(args)
    
    # ===== 載入 CLIP Model =====
    print(f'\n{"="*60}')
    print(f'載入 CLIP Model: {args.net}')
    clip_model = load_clip_to_cpu(backbone=args.net)
    clip_model.float()
    
    # 只使用 visual encoder
    netF = clip_model.visual.cuda()
    netF.eval()  # 🔒 凍結 CLIP Encoder
    for param in netF.parameters():
        param.requires_grad = False
    print(f'✅ CLIP Visual Encoder: 已載入並凍結 (eval mode)')
    print(f'{"="*60}\n')
    
    # ===== 載入 Source Visual Prototypes =====
    source_visual_prototypes, loaded_classnames = load_source_prototypes(
        path='/mnt/backups/andycw/CLIP/output/m58/class_centroids.pth'
    )
    
    # 驗證類別數量一致性
    if len(loaded_classnames) != args.class_num:
        raise ValueError(
            f'Source prototypes 的類別數 ({len(loaded_classnames)}) '
            f'與 args.class_num ({args.class_num}) 不一致！'
        )
    
    # ===== 初始化並載入 Projector =====
    print(f'\n{"="*60}')
    print(f'初始化 MLP Projector')
    netP = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).cuda()
    
    # 載入預訓練的 Projector 權重
    projector_path = osp.join(args.output_dir_src, 'source_projector.pt')
    if not osp.exists(projector_path):
        raise FileNotFoundError(
            f'找不到 Projector 權重: {projector_path}\n'
            f'請先執行 clip_source_model_projectOnly.py 訓練'
        )
    
    netP.load_state_dict(torch.load(projector_path))
    netP.train()  # ✅ Projector 設為訓練模式
    print(f'✅ Projector 已載入: {projector_path}')
    print(f'✅ Projector: train() 模式（可訓練）')
    print(f'{"="*60}\n')
    
    # ===== 設定優化器（只優化 Projector）=====
    param_group = []
    for k, v in netP.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)
    
    print(f'\n{"="*60}')
    print(f'優化器設定:')
    print(f'  - 只優化 Projector')
    print(f'  - 初始 Learning Rate: {args.lr}')
    print(f'  - Momentum: 0.9')
    print(f'  - Weight Decay: 1e-3')
    print(f'  - Nesterov: True')
    print(f'  - LR Scheduler: Inverse Decay (gamma=10, power=0.75)')
    print(f'      公式: lr = lr0 / (1 + gamma * iter / max_iter)^power')
    print(f'{"="*60}\n')

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    
    # 初始化最佳準確率
    best_acc = 0.0
    
    # 訓練循環變數
    batches_per_epoch = len(dset_loaders["target"])
    current_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"開始訓練")
    print(f"{'='*60}")
    print(f"Total epochs: {args.max_epoch}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total iterations: {max_iter}")
    print(f"Evaluation interval: {interval_iter} iterations")
    print(f"\nLoss 設定:")
    print(f"  - loss_ce (K-Means pseudo labels): weight = {args.cls_par}")
    print(f"  - im_loss (entropy minimization): weight = {args.ent_par}")
    if args.gent:
        print(f"  - diversity regularization: enabled")
    print(f"\nLogit Scale 設定:")
    print(f"  - logit_scale (temperature): {args.logit_scale} (固定值，不可學習)")
    print(f"{'='*60}\n")
    
    # ===== 訓練前先更新一次 pseudo labels（Epoch 1 開始前）=====
    print(f"\n{'='*60}")
    print(f"Epoch 1/{args.max_epoch} 開始")
    print(f"{'='*60}")
    pseudo_labels_tensor = update_pseudo_labels(
        netF, netP, dset_loaders["target"],
        source_visual_prototypes, args.class_num
    )
    
    # 初始化 iterator
    iter_test = iter(dset_loaders["target"])

    while iter_num < max_iter:
        try:
            inputs_test, true_labels, tar_idx, path = next(iter_test)
        except StopIteration:
            # ===== Epoch 結束，在開始新 epoch 前更新 pseudo labels =====
            current_epoch += 1
            print(f"\n{'='*60}")
            print(f"Epoch {current_epoch}/{args.max_epoch} 完成")
            print(f"當前最佳準確率: {best_acc:.2f}%")
            print(f"{'='*60}\n")
            
            if current_epoch < args.max_epoch:
                print(f"\n{'='*60}")
                print(f"Epoch {current_epoch + 1}/{args.max_epoch} 開始")
                print(f"{'='*60}")
                pseudo_labels_tensor = update_pseudo_labels(
                    netF, netP, dset_loaders["target"],
                    source_visual_prototypes, args.class_num
                )
            
            # 重新創建 iterator 並取下一個 batch
            iter_test = iter(dset_loaders["target"])
            inputs_test, true_labels, tar_idx, path = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        # Learning rate scheduling
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        optimizer.zero_grad()
        
        # 移動數據到 GPU
        inputs_test = inputs_test.cuda()
        true_labels = true_labels.cuda()
        
        # ===== 特徵提取：CLIP Encoder + Projector =====
        with torch.no_grad():
            feat_512 = netF(inputs_test.type(netF.conv1.weight.dtype))  # [B, 512]
        feat_projected = netP(feat_512)  # [B, 512]
        feat_norm = F.normalize(feat_projected, dim=-1)
        
        # ===== 計算 Logits =====
        logit_scale = args.logit_scale
        source_prototypes_norm = F.normalize(source_visual_prototypes, dim=-1)
        logits = logit_scale * (feat_norm @ source_prototypes_norm.T)  # [B, 79]
        
        # ===== Loss 1: Cross-Entropy Loss（使用 K-Means pseudo labels）=====
        if args.cls_par > 0:
            batch_pseudo_labels = pseudo_labels_tensor[tar_idx]
            loss_ce = F.cross_entropy(logits, batch_pseudo_labels)
            loss_ce *= args.cls_par
        else:
            loss_ce = torch.tensor(0.0).cuda()
        
        # ===== Loss 2: Entropy Minimization (im_loss) =====
        if args.ent:
            softmax_out = F.softmax(logits, dim=1)
            entropy_loss = torch.mean(-torch.sum(softmax_out * torch.log(softmax_out + args.epsilon), dim=1))
            
            # Diversity regularization (optional)
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            
            im_loss = entropy_loss * args.ent_par
        else:
            im_loss = torch.tensor(0.0).cuda()
        
        # ===== Total Loss =====
        total_loss = loss_ce + im_loss
        
        # Backward and optimize
        total_loss.backward()
        optimizer.step()
        
        # ===== 每 50 個 iteration 顯示詳細 loss 和 learning rate =====
        if iter_num % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Iter [{iter_num}/{max_iter}] | LR: {current_lr:.6f} | '
                  f'loss_ce: {loss_ce.item():.4f} | im_loss: {im_loss.item():.4f} | '
                  f'total: {total_loss.item():.4f}')
        
        # ===== 評估和保存 =====
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netP.eval()
            
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netP, source_visual_prototypes, True, args.logit_scale)
                log_str = 'Iter: {}/{}, Accuracy = {:.2f}%'.format(iter_num, max_iter, acc_s_te)
                log_str += '\n' + acc_list
            else:
                acc_s_te, mean_ent = cal_acc(dset_loaders['test'], netF, netP, source_visual_prototypes, False, args.logit_scale)
                log_str = 'Iter: {}/{}, Accuracy = {:.2f}%, Mean Ent = {:.4f}'.format(
                    iter_num, max_iter, acc_s_te, mean_ent
                )
            
            print(f"\n{log_str}")
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            
            # ===== 保存模型 =====
            # 保存當前模型（每次評估都覆蓋）
            torch.save(netP.state_dict(), osp.join(args.output_dir, "target_P_current.pt"))
            print(f"✓ 當前模型已保存: target_P_current.pt")
            
            # 保存最佳模型
            if acc_s_te >= best_acc:
                best_acc = acc_s_te
                torch.save(netP.state_dict(), osp.join(args.output_dir, "target_P_best.pt"))
                print(f"✓ 新的最佳準確率: {best_acc:.2f}%, 最佳模型已保存: target_P_best.pt")
            
            netP.train()

    print(f"\n{'='*60}")
    print(f"訓練完成！")
    print(f"{'='*60}")
    print(f"最佳準確率: {best_acc:.2f}%")
    print(f"模型已保存到: {args.output_dir}")
    print(f"  - target_P_best.pt (最佳模型)")
    print(f"  - target_P_current.pt (最終模型)")
    print(f"{'='*60}\n")

    return netF, netP


def print_args(args):
    """打印參數"""
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP Adaptation with K-Means Pseudo Labeling')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max epochs")
    parser.add_argument('--interval', type=int, default=15, help="evaluation interval")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='M58',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'M58'])
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate")
    parser.add_argument('--net', type=str, default='RN101', help="CLIP backbone")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    
    # Loss parameters
    parser.add_argument('--gent', type=bool, default=True, help="use diversity regularization")
    parser.add_argument('--ent', type=bool, default=True, help="use entropy loss")
    parser.add_argument('--cls_par', type=float, default=0.3, help="weight for pseudo-label CE loss")
    parser.add_argument('--ent_par', type=float, default=1.0, help="weight for entropy minimization loss")
    parser.add_argument('--logit_scale', type=float, default=100.0, help="temperature for logits")
    
    # Other parameters
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--output', type=str, default='ckps/adapt')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    args = parser.parse_args()

    # 設定類別數量和數據集名稱
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    elif args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    elif args.dset == 'M58':
        names = ['CAD_ratioFilter_StableDiffusion_random', 'Real_all_nobg']
        args.class_num = 79
    
    # 設定隨機種子
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 設定數據路徑
    folder = 'data/'
    if args.dset == 'M58':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_79_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dset_loaders = data_load(args)
    
    # ===== 載入 CLIP Model =====
    print(f'\n{"="*60}')
    print(f'載入 CLIP Model: {args.net}')
    clip_model = load_clip_to_cpu(backbone=args.net)
    clip_model.float()
    
    # 只使用 visual encoder
    netF = clip_model.visual.cuda()
    netF.eval()  # 🔒 凍結 CLIP Encoder
    for param in netF.parameters():
        param.requires_grad = False
    print(f'✅ CLIP Visual Encoder: 已載入並凍結 (eval mode)')
    print(f'{"="*60}\n')
    
    # ===== 載入 Source Visual Prototypes =====
    source_visual_prototypes, loaded_classnames = load_source_prototypes(
        path='/mnt/backups/andycw/CLIP/output/m58/class_centroids.pth'
    )
    
    # 驗證類別數量一致性
    if len(loaded_classnames) != args.class_num:
        raise ValueError(
            f'Source prototypes 的類別數 ({len(loaded_classnames)}) '
            f'與 args.class_num ({args.class_num}) 不一致！'
        )
    
    netF = clip_model.visual.cuda()
    netF.eval()  # 🔒 凍結 CLIP Encoder

    # ===== 初始化並載入 Projector =====
    print(f'\n{"="*60}')
    print(f'初始化 MLP Projector')
    netP = MLPProjector(in_dim=512, out_dim=512, hidden_dim=1024).cuda()
    
    # 載入預訓練的 Projector 權重
    projector_path = "/mnt/backups/andycw/UDA-AI/ckps/source/uda/M58/C/source_projector.pt"
    if not osp.exists(projector_path):
        raise FileNotFoundError(
            f'找不到 Projector 權重: {projector_path}\n'
            f'請先執行 clip_source_model_projectOnly.py 訓練'
        )
    
    netP.load_state_dict(torch.load(projector_path))
    netP.eval()

    dataset_size = len(dset_loaders["target"].dataset)
    all_features = torch.zeros(dataset_size, 512)  # 預先分配固定大小
    all_true_labels = torch.zeros(dataset_size, dtype=torch.long)  # 收集真實標籤
    
    netP.eval()  # 設為 eval 模式
    
    with torch.no_grad():
        for data in tqdm(dset_loaders["target"], desc="提取特徵", leave=False):
            imgs, labels, indices, _ = data  # 取出 imgs, labels, indices
            imgs = imgs.cuda()
            
            # CLIP Encoder → Projector → Normalize
            feat_512 = netF(imgs.type(netF.conv1.weight.dtype))
            feat_projected = netP(feat_512)
            feat_norm = F.normalize(feat_projected, dim=-1)
            
            # 使用 indices 明確指定位置，確保對應關係
            all_features[indices] = feat_norm.cpu()
            all_true_labels[indices] = labels.cpu()  # 收集真實標籤

    for i in range(10):  # 進行 5 次迭代更新
        _, _, source_visual_prototypes_new = update_pseudo_labels(
            netF, netP, dset_loaders["target"],
            source_visual_prototypes, args.class_num,
            all_features=all_features, all_true_labels=all_true_labels
        )
        source_visual_prototypes = source_visual_prototypes_new
    # for data in dset_loaders["target"]:
    #     imgs, _, indices, _ = data
    #     imgs = imgs.cuda()

    #     feats = F.normalize(netP(netF(imgs)), dim=-1)
    #     logits = 100.0 * (feats @ source_visual_prototypes.T)

