import wandb
# wandb.login()

import argparse
import os, sys
import datetime
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from randaugment import RandAugmentMC
from gaussian_blur import GaussianBlur
import loss


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


class Augmentation(object):
    def __init__(self, resize_size=256, crop_size=224):
        self.weak = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip()
        ])
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.strong = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def data_load(args):
    ## prepare data
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

    # Determine root path based on dataset
    root_path = f'data/{args.dset}/'

    dsets["target"] = ImageList_idx(txt_tar, transform=Augmentation(), root=root_path)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), root=root_path)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
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


def calculate_gaussian_kl_divergence(m1, m2, v1, v2):
    return torch.log(v2 / v1) * 0.5 + torch.div(torch.add(v1, torch.square(m1 - m2)), 2 * v2) - 0.5

class ShotHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 or KL 散度 as a loss
    '''

    def __init__(self, module1, module2):
        self.hook = module1.register_forward_hook(self.hook_fn)
        self.mean_orignal = module2.running_mean
        self.var_orignal = module2.running_var

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        if isinstance(module, nn.BatchNorm2d):
            mean = input[0].mean([0, 2, 3])
            # Memory efficient variance calculation - process in smaller chunks
            x = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1])
            if x.size(1) > 10000:  # If too large, compute variance in chunks
                chunk_size = 5000
                var_chunks = []
                for i in range(0, x.size(1), chunk_size):
                    chunk = x[:, i:i+chunk_size]
                    var_chunks.append(chunk.var(1, unbiased=False, keepdim=True))
                var = torch.cat(var_chunks, dim=1).mean(dim=1)
            else:
                var = x.var(1, unbiased=False)

        if isinstance(module, nn.BatchNorm1d):
            mean = input[0].mean([0])
            x = input[0].permute(1, 0).contiguous().view([nch, -1])
            if x.size(1) > 10000:  # If too large, compute variance in chunks
                chunk_size = 5000
                var_chunks = []
                for i in range(0, x.size(1), chunk_size):
                    chunk = x[:, i:i+chunk_size]
                    var_chunks.append(chunk.var(1, unbiased=False, keepdim=True))
                var = torch.cat(var_chunks, dim=1).mean(dim=1)
            else:
                var = x.var(1, unbiased=False)

        klc = 0.0
        for i in range(mean.size()[0]):
            klc += calculate_gaussian_kl_divergence(self.mean_orignal[i], mean[i], self.var_orignal[i], var[i]) # Equation 9
        r_feature = klc / mean.size()[0]

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        netF_orignal = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netB_orignal = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                           bottleneck_dim=args.bottleneck).cuda()

    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # 智能載入 CLIP Source Model 權重（處理架構差異）
    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath)) # Feature Extractor
    netF_orignal.load_state_dict(torch.load(modelpath))

    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath)) # Bottleneck Layer
    netB_orignal.load_state_dict(torch.load(modelpath))

    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath)) # Classifier
    netC.eval()

    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netF_orignal.named_parameters():
        v.requires_grad = False
    for k, v in netB_orignal.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    
    # Initialize wandb
    # wandb.init(project="SFDA", 
    #            name=f"ATSSL_{args.dset}_{args.s}_to_{args.t}",
    #            config=vars(args))
    
    # Calculate micro batch size for memory efficiency
    micro_batch_size = args.batch_size // args.accumulation_steps
    print(f"Batch size: {args.batch_size}, Micro batch size: {micro_batch_size}, Accumulation steps: {args.accumulation_steps}")

    # Initialize best accuracy for saving
    acc_init = 0

    # Training loop variables for epoch tracking
    batches_per_epoch = len(dset_loaders["target"])
    current_epoch = 0
    epoch_start_iter = 0
    epoch_total_loss = 0.0
    epoch_bn_loss = 0.0
    epoch_classifier_loss = 0.0
    epoch_entropy_loss = 0.0
    epoch_contrast_loss = 0.0
    epoch_batches = 0
    
    # Create initial progress bar
    pbar = tqdm(total=batches_per_epoch, desc=f'Epoch {current_epoch+1}/{args.max_epoch}', leave=False)

    while iter_num < max_iter:
        try:
            (inputs_test, inputs_s), _, tar_idx, path = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_s), _, tar_idx, path = next(iter_test) # inputs_test: weakly augmented target image, inputs_s: strongly augmented target image

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            softpred, weight = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            softpred = softpred.cuda()
            netF.train()
            netB.train()

        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Move to GPU
        inputs_test = inputs_test.cuda()
        inputs_s = inputs_s.cuda()
        mas = weight[tar_idx].cuda() # weight for contrastive loss
        
        # Accumulate gradients over micro batches
        total_loss = 0
        total_bn_loss = 0
        total_classifier_loss = 0
        total_entropy_loss = 0
        total_contrast_loss = 0
        current_batch_size = inputs_test.size(0)
        
        for micro_step in range(args.accumulation_steps):
            start_idx = (micro_step * current_batch_size) // args.accumulation_steps
            end_idx = ((micro_step + 1) * current_batch_size) // args.accumulation_steps
            
            if start_idx >= end_idx:
                continue
                
            # Get micro batch
            micro_inputs_test = inputs_test[start_idx:end_idx]
            micro_inputs_s = inputs_s[start_idx:end_idx]
            micro_tar_idx = tar_idx[start_idx:end_idx]
            micro_mas = mas[start_idx:end_idx]
            
            # Clear cache periodically
            if micro_step == 0 and iter_num % 100 == 0:
                torch.cuda.empty_cache()
            
            # Process strong augmentation
            features_s = netB(netF(micro_inputs_s)) # strongly augmented features (without softmax logits)
            out_2 = F.normalize(features_s, dim=-1)

            # Setup BN hooks for micro batch
            loss_bn_layers = []
            if args.bn:
                i, j = 0, 0
                # module1 from target model, module2 from source model
                for module1 in netF.modules():
                    i += 1
                    for module2 in netF_orignal.modules():
                        j += 1
                        if isinstance(module1, nn.BatchNorm2d) and i == j:
                            loss_bn_layers.append(ShotHook(module1, module2)) # ShotHook fetch source/target model's BN statistics
                    j = 0
                for module1 in netB.modules():
                    for module2 in netB_orignal.modules():
                        if isinstance(module1, nn.BatchNorm1d) and isinstance(module2, nn.BatchNorm1d):
                            loss_bn_layers.append(ShotHook(module1, module2))

            # Process weak augmentation
            features_test = netB(netF(micro_inputs_test))
            out_1 = F.normalize(features_test, dim=-1)
            outputs_test = netC(features_test) # weakly augmented Classifier outputs 

            # Calculate losses for micro batch
            losses = torch.tensor(0.0).cuda()
            
            # BN loss
            if args.bn and loss_bn_layers:
                bn_loss = sum([mod.r_feature for mod in loss_bn_layers]) / len(loss_bn_layers) # r_feature from ShotHook KL Divergence
                bn_loss *= args.bn_par
                losses += bn_loss
            
            # Close hooks
            for x in loss_bn_layers:
                x.close()

            # Classifier loss
            classifier_loss = torch.tensor(0.0).cuda()
            if args.cls_par > 0 and args.plabel:
                pred = softpred[micro_tar_idx] # output logits from target model
                x = F.log_softmax(outputs_test, 1) # Log softmax of weakly augmented outputs
                y = F.softmax(pred, 1) # Soft pseudo labels
                classifier_loss = nn.KLDivLoss()(x, y) # Equation 5 Loss CLU
                classifier_loss *= args.cls_par
                if iter_num < interval_iter and args.dset == "VISDA-C":
                    classifier_loss *= 0
                losses += classifier_loss

            # Entropy loss
            im_loss = torch.tensor(0.0).cuda()
            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out)) # Equation 1 Loss ent
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon)) # Equation 2 Loss div
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * args.ent_par
                losses += im_loss

            # Contrastive loss
            out = torch.cat([out_1, out_2], dim=0) # out_1 : weakly augmented features, out_2 : strongly augmented features
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.tt) # (2N x D) ▪ (D x 2N) -> 2N x 2N similarity matrix
            # ones_like: matrix of all ones with same shape as sim_matrix
            # torch.eye: identity matrix (對角線為1，其餘為0)
            # 目的: 建立一個對角線為0，其餘為1的mask矩陣，以去除自我相似度的影響
            mask = (torch.ones_like(sim_matrix) - torch.eye(out.shape[0], device=sim_matrix.device)).bool() # Mask to remove self-similarity
            sim_matrix = sim_matrix.masked_select(mask).view(out.shape[0], -1) # Lower Equation 10, Reshape from 2N to (2N-1) without self-similarity

            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.tt)  # Upper Equation 10
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0) # To match 2N size since simCLR calculates Week -> Strong and Strong -> Week
            micro_mas_expanded = torch.cat([micro_mas, micro_mas]) # Weighting for each sample

            contrast_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1)) * micro_mas_expanded).mean() # Equation 12 Loss con
            losses += contrast_loss
            
            # Accumulate individual losses for epoch logging
            epoch_bn_loss += bn_loss.item()
            epoch_classifier_loss += classifier_loss.item()
            epoch_entropy_loss += im_loss.item()
            epoch_contrast_loss += contrast_loss.item()

            # Scale loss by accumulation steps and backward
            scaled_loss = losses / args.accumulation_steps
            scaled_loss.backward()
            
            epoch_total_loss += scaled_loss.item()
            epoch_batches += 1

        # Update parameters after accumulating gradients
        optimizer.step()
        
        # Update progress bar
        batch_in_epoch = (iter_num - epoch_start_iter) % batches_per_epoch
        pbar.update(1)
        pbar.set_postfix({
            'Total': f'{scaled_loss.item():.4f}',
            'BN': f'{bn_loss.item():.4f}',
            'Cls': f'{classifier_loss.item():.4f}',
            'Ent': f'{im_loss.item():.4f}',
            'Con': f'{contrast_loss.item():.4f}'
        })
        
        # Check if epoch is complete
        if (iter_num - epoch_start_iter + 1) % batches_per_epoch == 0:
            # Close current progress bar
            pbar.close()
            
            # Calculate average losses for this epoch
            avg_total_loss = epoch_total_loss / epoch_batches if epoch_batches > 0 else 0
            avg_bn_loss = epoch_bn_loss / epoch_batches if epoch_batches > 0 else 0
            avg_classifier_loss = epoch_classifier_loss / epoch_batches if epoch_batches > 0 else 0
            avg_entropy_loss = epoch_entropy_loss / epoch_batches if epoch_batches > 0 else 0
            avg_contrast_loss = epoch_contrast_loss / epoch_batches if epoch_batches > 0 else 0
            
            # Evaluate model at the end of each epoch
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Epoch: {}/{}, Accuracy = {:.2f}%'.format(current_epoch+1, args.max_epoch, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Epoch: {}/{}, Accuracy = {:.2f}%'.format(current_epoch+1, args.max_epoch, acc_s_te)
                print("Epoch {}/{} - Avg Loss: {:.4f}, BN: {:.4f}, Classifier: {:.4f}, Entropy: {:.4f}, Contrast: {:.4f}, Accuracy: {:.2f}%".format(
                    current_epoch+1, args.max_epoch, avg_total_loss, avg_bn_loss, avg_classifier_loss, avg_entropy_loss, avg_contrast_loss, acc_s_te))
            
            # Log to wandb once per epoch
            # wandb.log({
            #     "epoch": current_epoch + 1,
            #     "avg_total_loss": avg_total_loss,
            #     "avg_bn_loss": avg_bn_loss,
            #     "avg_clu_loss": avg_classifier_loss,
            #     "avg_im_loss": avg_entropy_loss,
            #     "avg_contrast_loss": avg_contrast_loss,
            #     "test_accuracy": acc_s_te,
            #     "learning_rate": optimizer.param_groups[0]['lr']
            # })
            
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            
            # Save current stage models (overwrite each epoch)
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_current.pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_current.pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_current.pt"))
            
            # Save best model if current accuracy is better
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_best.pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_best.pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_best.pt"))
                print(f"New best accuracy: {acc_s_te:.2f}%, saving best model...")
            
            netF.train()
            netB.train()
            
            # Reset epoch variables for next epoch
            current_epoch += 1
            epoch_start_iter = iter_num + 1
            epoch_total_loss = 0.0
            epoch_bn_loss = 0.0
            epoch_classifier_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_contrast_loss = 0.0
            epoch_batches = 0
            
            # Create new progress bar for next epoch if not finished
            if current_epoch < args.max_epoch:
                pbar = tqdm(total=batches_per_epoch, desc=f'Epoch {current_epoch+1}/{args.max_epoch}', leave=False)

        iter_num += 1

    # Close final progress bar
    if 'pbar' in locals():
        pbar.close()
    
    # Finish wandb run
    # wandb.finish()
    
    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)
    entropy = loss.Entropy(all_output)

    weight = 1.0 - torch.exp(-entropy) # Equation 11

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)

    dd = cdist(all_fea, initc, args.distance)
    dd = 1 / dd
    dd = torch.Tensor(dd)
    softpred = nn.Softmax(dim=1)(dd / args.T)
    _, pred_label = torch.max(softpred, dim=-1)

    acc = torch.sum(pred_label == all_label) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return softpred, weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'M58'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--plabel', action='store_true')
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--bn_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--tt', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', type=int, default=1, help="gradient accumulation steps")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'M58':
        names = ['CAD_ratioFilter', 'Real_all_nobg']
        args.class_num = 37
        # args.class_num = 30

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = 'data/'
    if args.dset == 'M58':
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_37_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_37_list.txt'
        # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_hard_30_list.txt'
        # args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_hard_30_list.txt'
        # args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_hard_30_list.txt'
    else:
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    print("output_dir_src:", args.output_dir_src)
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][
        0].upper() + datetime.datetime.now().strftime("%m-%d_%H:%M"))
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'cls_par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)