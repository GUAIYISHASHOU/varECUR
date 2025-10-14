# -*- coding: utf-8 -*-
import argparse, os, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from vis.datasets.macro_frames import MacroFrames
# === 修改：从模型文件导入新的模型和损失函数 ===
from models.macro_transformer_sa import MacroTransformerSA, CombinedUncertaintyLoss

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def spearman_torch(x, y):
    """
    正确的 Spearman 相关系数实现 (PyTorch 版本)
    先中心化，再用中心化后的值计算归一化系数
    """
    # x,y: (N,) torch
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx -= rx.mean()  # 先中心化
    ry -= ry.mean()  # 先中心化
    # 用中心化后的值计算归一化系数
    denom = (rx.square().sum().sqrt() * ry.square().sum().sqrt()) + 1e-12
    return (rx * ry).sum() / denom

def count_trainable_params(model):
    """统计可训练参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", type=str, required=True)
    p.add_argument("--val_npz",   type=str, required=True)
    # === 新增：特征归一化统计量文件 ===
    p.add_argument("--geom_stats_npz", type=str, default=None,
                   help="预计算的geoms统计量文件路径 (mean/std)，用于Z-score归一化")
    p.add_argument("--save_dir",  type=str, default="runs/vis_macro_sa")
    p.add_argument("--epochs",    type=int, default=30)
    p.add_argument("--batch_size",type=int, default=32)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--wd",        type=float, default=1e-4)
    p.add_argument("--device",    type=str, default="cuda")
    p.add_argument("--nu",        type=float, default=-1.0, help=">0 则用Student-t残差监督；<=0则回归logvar")
    p.add_argument("--a_max",     type=float, default=3.0)
    p.add_argument("--drop_token_p", type=float, default=0.1)
    p.add_argument("--heads",     type=int, default=4)
    p.add_argument("--layers",    type=int, default=1)
    p.add_argument("--d_model",   type=int, default=128)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--logv_min",  type=float, default=-10.0)
    p.add_argument("--logv_max",  type=float, default=6.0)
    # === 新增：用于两阶段训练的参数 ===
    p.add_argument("--stage1_epochs", type=int, default=0, help="Epochs for stage 1 (q-only training)")
    p.add_argument("--nll_weight",    type=float, default=1.0, help="Weight for NLL loss")
    p.add_argument("--bce_weight",    type=float, default=1.0, help="Weight for BCE loss")
    p.add_argument("--rank_weight",   type=float, default=0.0, help="Weight for pairwise ranking loss")
    p.add_argument("--rank_margin",   type=float, default=0.0, help="Margin for ranking loss")
    # === 早停参数 ===
    p.add_argument("--patience",      type=int, default=0, help="Early stopping patience (0=disabled)")
    # === OOF 训练：子集索引文件 ===
    p.add_argument("--subset_idx_file", type=str, default=None,
                   help="Optional .npy of 1D indices to select a subset for training (for K-fold OOF)")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(Path(args.save_dir)/"hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # === OOF 训练：加载子集索引 ===
    subset_idx = None
    if args.subset_idx_file is not None and os.path.isfile(args.subset_idx_file):
        subset_idx = np.load(args.subset_idx_file)
        print(f"[OOF] 加载训练子集索引: {args.subset_idx_file}")
        print(f"[OOF] 子集大小: {len(subset_idx)}")

    # === 修改：传递geom_stats_path和subset_idx到数据集 ===
    train_ds = MacroFrames(args.train_npz, geom_stats_path=args.geom_stats_npz, subset_idx=subset_idx)
    val_ds   = MacroFrames(args.val_npz, geom_stats_path=args.geom_stats_npz)

    geom_dim = train_ds.geoms.shape[2]
    mdl = MacroTransformerSA(
        geom_dim=geom_dim, d_model=args.d_model, n_heads=args.heads, n_layers=args.layers,
        a_max=args.a_max, drop_token_p=args.drop_token_p,
        logv_min=args.logv_min, logv_max=args.logv_max
    ).to(args.device)

    opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = CosineAnnealingLR(opt, T_max=args.epochs)

    # Windows兼容：使用num_workers=0避免多进程序列化问题
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # === 新增：计算训练集内点率，用于BCE的pos_weight（类别平衡）===
    train_inlier_rate = float(train_ds.y_inlier.mean())
    print(f"\n训练集内点率: {train_inlier_rate*100:.1f}%")
    if train_inlier_rate > 0 and train_inlier_rate < 1.0:
        # pos_weight = (1-p) / p，给正样本（内点）更高权重以平衡类别
        pos_weight_value = (1.0 - train_inlier_rate) / max(train_inlier_rate, 1e-6)
        pos_weight = torch.tensor([pos_weight_value], device=args.device)
        print(f"BCE pos_weight: {pos_weight_value:.3f} (平衡类别不平衡)\n")
    else:
        pos_weight = None
        print("内点率为0或1，不使用pos_weight\n")

    best_spear = -1.0
    patience_counter = 0  # 早停计数器
    for ep in range(1, args.epochs+1):
        mdl.train()
        lossv = []
        
        # === 实现两阶段训练 ===
        is_stage1 = ep <= args.stage1_epochs
        
        # === 关键修改：冻结/解冻参数 ===
        if is_stage1:
            # 第一阶段：只训练 q_head 和共享特征层
            for n, p in mdl.named_parameters():
                # 只训练 q_head 和 head_shared，冻结其他所有参数
                if 'q_head' in n or 'head_shared' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            trainable, total = count_trainable_params(mdl)
            print(f"[ep {ep:03d}] Stage 1: Training q-head + shared layers only ({trainable}/{total} params, {100*trainable/total:.1f}%)")
        else:
            # 第二阶段：解冻所有参数
            if ep == args.stage1_epochs + 1:
                for p in mdl.parameters():
                    p.requires_grad = True
                trainable, total = count_trainable_params(mdl)
                print(f"[ep {ep:03d}] Stage 2: Unfrozen all parameters ({trainable}/{total} params)")
        
        crit = CombinedUncertaintyLoss(
            nll_weight=args.nll_weight, 
            bce_weight=args.bce_weight,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            stage1_mode=is_stage1,
            pos_weight=pos_weight  # 类别平衡权重
        )
        
        for b in train_dl:
            patches = b["patches"].to(args.device, non_blocking=True)
            geoms   = b["geoms"].to(args.device, non_blocking=True)
            y_true  = b["y_true"].to(args.device, non_blocking=True)
            num_tok = b["num_tok"].to(args.device, non_blocking=True)
            # === 新增：获取内点标签 ===
            y_inlier = b["y_inlier"].to(args.device, non_blocking=True)

            # === 修改：模型现在返回两个输出 ===
            pred_logvar, pred_q_logit = mdl(patches, geoms, num_tok)
            
            # === 修改：使用新的损失函数 ===
            loss = crit(pred_logvar, pred_q_logit, y_true, y_inlier)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
            opt.step()
            lossv.append(loss.detach().item())

        sch.step()

        # ----- 验证 -----
        mdl.eval()
        with torch.no_grad():
            pv_logvar, pv_q, gv_logvar, gv_inlier = [], [], [], []
            for b in val_dl:
                pred_logvar, pred_q_logit = mdl(b["patches"].to(args.device),
                                                b["geoms"].to(args.device),
                                                b["num_tok"].to(args.device))
                pv_logvar.append(pred_logvar.cpu())
                pv_q.append(torch.sigmoid(pred_q_logit).cpu())  # 转换为概率
                gv_logvar.append(b["y_true"])
                gv_inlier.append(b["y_inlier"])
            
            pv_logvar = torch.cat(pv_logvar, 0)
            gv_logvar = torch.cat(gv_logvar, 0)
            pv_q = torch.cat(pv_q, 0)
            gv_inlier = torch.cat(gv_inlier, 0)
            
            # 计算 spearman (只在内点上计算更有意义)
            inlier_mask_val = (gv_inlier > 0.5).squeeze()
            if inlier_mask_val.any():
                sp_x = float(spearman_torch(pv_logvar[inlier_mask_val, 0], gv_logvar[inlier_mask_val, 0]))
                sp_y = float(spearman_torch(pv_logvar[inlier_mask_val, 1], gv_logvar[inlier_mask_val, 1]))
            else:
                sp_x = sp_y = 0.0
            sp_m = (sp_x + sp_y) * 0.5
            
            # 计算分类准确率 (Accuracy)
            preds_q = (pv_q > 0.5).float()
            accuracy = (preds_q == gv_inlier).float().mean()

        mean_loss = sum(lossv)/max(1,len(lossv))
        print(f"[ep {ep:03d}] train_loss={mean_loss:.4f}  spear_mean={sp_m:.3f}  q_acc={accuracy:.3f}")
        
        # ----- 保存 & 早停检查 -----
        if sp_m > best_spear:
            best_spear = sp_m
            patience_counter = 0  # 重置计数器
            ckpt = {
                "model": mdl.state_dict(),
                "args": vars(args),
                "val_spear": {"x": sp_x, "y": sp_y, "mean": sp_m},
                "val_q_acc": float(accuracy)
            }
            torch.save(ckpt, Path(args.save_dir)/"best_macro_sa.pt")
            print(f"  ↳ saved best (mean spear={sp_m:.3f}, q_acc={accuracy:.3f})")
        else:
            patience_counter += 1
            if args.patience > 0:
                print(f"  ↳ no improvement ({patience_counter}/{args.patience})")
        
        # 早停触发
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n[Early Stop] No improvement for {args.patience} epochs. Best spear={best_spear:.3f}")
            print(f"[Early Stop] Stopped at epoch {ep}/{args.epochs}")
            break

if __name__ == "__main__":
    main()

