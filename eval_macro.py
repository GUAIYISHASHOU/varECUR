# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from vis.datasets.macro_frames import MacroFrames
from models.macro_transformer_sa import MacroTransformerSA

def spearman_np(x, y):
    xr = x.argsort().argsort().astype(np.float32)
    yr = y.argsort().argsort().astype(np.float32)
    xr = (xr - xr.mean()) / (np.sqrt((xr**2).sum()) + 1e-9)
    yr = (yr - yr.mean()) / (np.sqrt((yr**2).sum()) + 1e-9)
    return float((xr * yr).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    # === 新增：特征归一化统计量文件 ===
    ap.add_argument("--geom_stats_npz", type=str, default=None,
                   help="预计算的geoms统计量文件路径 (mean/std)，用于Z-score归一化")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--plots_dir", type=str, default=None)
    ap.add_argument("--temp_global", type=float, default=1.0, help="可选的全局温标倍率（>1 增大方差）")
    args = ap.parse_args()

    # === 修改：传递geom_stats_path到数据集 ===
    ds = MacroFrames(args.npz, geom_stats_path=args.geom_stats_npz)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    a = ckpt["args"]
    mdl = MacroTransformerSA(
        geom_dim=ds.geoms.shape[2],
        d_model=a["d_model"], n_heads=a["heads"], n_layers=a["layers"],
        a_max=a["a_max"], drop_token_p=0.0,
        logv_min=a["logv_min"], logv_max=a["logv_max"]
    ).to(args.device)
    mdl.load_state_dict(ckpt["model"], strict=True)
    mdl.eval()

    pred, gt = [], []
    pred_q = []  # 存储内点概率
    with torch.no_grad():
        for b in dl:
            # === 修改：解包两个返回值 ===
            pred_logvar, pred_q_logit = mdl(b["patches"].to(args.device),
                                             b["geoms"].to(args.device),
                                             b["num_tok"].to(args.device))
            pred.append(pred_logvar.detach().cpu())
            pred_q.append(torch.sigmoid(pred_q_logit).detach().cpu())
            gt.append(b["y_true"])
    pred = torch.cat(pred, 0).numpy()      # logvar
    gt   = torch.cat(gt, 0).numpy()
    pred_q = torch.cat(pred_q, 0).numpy()  # 内点概率

    # 可选：全局温标
    pred = pred + np.log(np.array([args.temp_global, args.temp_global], dtype=np.float32))

    sx = spearman_np(pred[:,0], gt[:,0])
    sy = spearman_np(pred[:,1], gt[:,1])
    
    # === 新增：评估内点概率（如果数据中有y_inlier标签）===
    metrics = {"spearman_x": sx, "spearman_y": sy, "spearman_mean": 0.5*(sx+sy)}
    
    try:
        # 尝试获取内点标签
        gt_inlier = []
        for b in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4):
            gt_inlier.append(b["y_inlier"])
        gt_inlier = torch.cat(gt_inlier, 0).numpy().ravel()
        
        # 计算分类指标
        q_pred_binary = (pred_q.ravel() > 0.5).astype(np.float32)
        q_acc = (q_pred_binary == gt_inlier).mean()
        metrics["q_accuracy"] = float(q_acc)
        
        # 如果可用，计算 AUC
        try:
            from sklearn.metrics import roc_auc_score
            q_auc = roc_auc_score(gt_inlier, pred_q.ravel())
            metrics["q_auc"] = float(q_auc)
        except ImportError:
            pass
    except (KeyError, AttributeError):
        # 旧数据没有y_inlier标签
        pass
    
    print(json.dumps(metrics, indent=2))

    if args.plots_dir:
        p = Path(args.plots_dir); p.mkdir(parents=True, exist_ok=True)
        for i,axis in enumerate(["x","y"]):
            plt.figure()
            plt.scatter(gt[:,i], pred[:,i], s=6, alpha=0.5)
            plt.xlabel("GT logvar"); plt.ylabel("Pred logvar"); plt.title(f"axis={axis}")
            plt.grid(True, ls="--", alpha=0.3)
            plt.savefig(p/f"scatter_logvar_{axis}.png", dpi=180)
            plt.close()

if __name__ == "__main__":
    main()

