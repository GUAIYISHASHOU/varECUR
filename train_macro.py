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
    # x,y: (N,) torch
    x_rank = torch.argsort(torch.argsort(x))
    y_rank = torch.argsort(torch.argsort(y))
    xr = (x_rank.float() - x_rank.float().mean()) / (x_rank.numel()**0.5)
    yr = (y_rank.float() - y_rank.float().mean()) / (y_rank.numel()**0.5)
    return (xr * yr).sum() / (xr.square().sum().sqrt() * yr.square().sum().sqrt() + 1e-12)

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
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(Path(args.save_dir)/"hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # === 修改：传递geom_stats_path到数据集 ===
    train_ds = MacroFrames(args.train_npz, geom_stats_path=args.geom_stats_npz)
    val_ds   = MacroFrames(args.val_npz, geom_stats_path=args.geom_stats_npz)

    geom_dim = train_ds.geoms.shape[2]
    mdl = MacroTransformerSA(
        geom_dim=geom_dim, d_model=args.d_model, n_heads=args.heads, n_layers=args.layers,
        a_max=args.a_max, drop_token_p=args.drop_token_p,
        logv_min=args.logv_min, logv_max=args.logv_max
    ).to(args.device)

    opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = CosineAnnealingLR(opt, T_max=args.epochs)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_spear = -1.0
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
            stage1_mode=is_stage1
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
        
        # ----- 保存 -----
        if sp_m > best_spear:
            best_spear = sp_m
            ckpt = {
                "model": mdl.state_dict(),
                "args": vars(args),
                "val_spear": {"x": sp_x, "y": sp_y, "mean": sp_m},
                "val_q_acc": float(accuracy)
            }
            torch.save(ckpt, Path(args.save_dir)/"best_macro_sa.pt")
            print(f"  ↳ saved best (mean spear={sp_m:.3f}, q_acc={accuracy:.3f})")

if __name__ == "__main__":
    main()

