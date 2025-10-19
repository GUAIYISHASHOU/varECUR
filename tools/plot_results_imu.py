# -*- coding: utf-8 -*-
"""
IMU 评测结果可视化（VIS 风格）
- 兼容 1/2/3 维 logvar（各向同性/各轴）
- 复刻 VIS 的颜色与版式：
  * 散点：GT外点=灰，GT内点=蓝；对角线=红色虚线 "Perfect"
  * q 分布直方图：外点=红，内点=绿；阈值竖线=黑色虚线
  * 残差直方图：steelblue
  * 阈值扫描：红色高亮“Best”点（⭐）
  * 若 D==2：提供各向异性图（与 VIS 一致）
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def robust_std(x):
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.349


def spearman_np(x, y):
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()) + 1e-12
    return float((rx * ry).sum() / denom)


def _axis_names(D):
    if D == 1: return ["σ"]
    if D == 2: return ["x", "y"]
    if D == 3: return ["x", "y", "z"]
    return [f"dim{i}" for i in range(D)]


def _ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 1: a = a[:, None]
    return a


def vis_plot_all(pred_logvar, gt_logvar, pred_q=None, gt_inlier=None,
                 out_dir="runs/imu_plots", scan_q_threshold=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pred = _ensure_2d(pred_logvar).astype(np.float64)
    gt   = _ensure_2d(gt_logvar).astype(np.float64)
    assert pred.shape == gt.shape, "pred 与 gt 形状不一致"
    N, D = pred.shape
    names = _axis_names(D)

    # ===== 1) LogVar 散点图（逐轴，颜色与 VIS 相同）=====
    metrics = {}
    for i, name in enumerate(names):
        if pred.shape[0] == 0:
            break
        fig, ax = plt.subplots(figsize=(8, 6))
        if gt_inlier is not None:
            mask_out = gt_inlier.ravel() < 0.5
            mask_in  = gt_inlier.ravel() > 0.5
            if mask_out.any():
                ax.scatter(gt[mask_out, i], pred[mask_out, i],
                           s=10, alpha=0.3, c='gray', label='Outlier (GT)')
            if mask_in.any():
                ax.scatter(gt[mask_in, i], pred[mask_in, i],
                           s=10, alpha=0.6, c='blue', label='Inlier (GT)')
        else:
            ax.scatter(gt[:, i], pred[:, i], s=10, alpha=0.5, c='blue')

        if gt.shape[0] > 0:
            lo = min(gt[:, i].min(), pred[:, i].min())
            hi = max(gt[:, i].max(), pred[:, i].max())
        else:
            lo, hi = -1.0, 1.0
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, linewidth=2, label='Perfect')

        spear_all = spearman_np(pred[:, i], gt[:, i]) if gt.shape[0] > 0 else 0.0
        metrics[f"spearman_{i}_all"] = float(spear_all)

        ax.set_xlabel(f"GT log(σ{names[i]}²)" if D>1 else "GT log(σ²)", fontsize=12)
        ax.set_ylabel(f"Pred log(σ{names[i]}²)" if D>1 else "Pred log(σ²)", fontsize=12)
        ttl = f"LogVar Prediction - Axis {names[i].upper() if D>1 else ''}".strip()
        ax.set_title(f"{ttl}\nSpearman = {spear_all:.3f}",
                     fontsize=14, fontweight='bold')
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"scatter_logvar_{names[i]}.png",
                    dpi=200, bbox_inches='tight')
        plt.close()

    # ===== 1b) 全局口径：Σσ² (Pred) vs Σe² (GT) 散点与 Spearman =====
    if pred.shape[0] > 0:
        s2_pred_sum = np.exp(pred).sum(axis=1)
        e2_sum = np.exp(gt).sum(axis=1)
        spear_global = spearman_np(s2_pred_sum, e2_sum)
        metrics["spearman_global_varsum"] = float(spear_global)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(e2_sum, s2_pred_sum, s=10, alpha=0.5, c='blue')
        lo = min(e2_sum.min(), s2_pred_sum.min())
        hi = max(e2_sum.max(), s2_pred_sum.max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, linewidth=2, label='Perfect')
        ax.set_xlabel("GT Σ e² (axes)", fontsize=12)
        ax.set_ylabel("Pred Σ σ² (axes)", fontsize=12)
        ax.set_title(f"Global Variance vs Error (axes)\nSpearman = {spear_global:.3f}", fontsize=14, fontweight='bold')
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "global_scatter_varsum.png", dpi=200, bbox_inches='tight')
        plt.close()

        # —— log-log 变体（更适合长尾分布的可视化）——
        eps = 1e-12
        x_ll = np.log10(np.clip(e2_sum, eps, None))
        y_ll = np.log10(np.clip(s2_pred_sum, eps, None))
        pearson_ll = float(np.corrcoef(x_ll, y_ll)[0, 1]) if x_ll.size > 1 else 0.0
        metrics["pearson_global_varsum_loglog"] = pearson_ll
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_ll, y_ll, s=10, alpha=0.5, c='blue')
        lo = min(x_ll.min(), y_ll.min())
        hi = max(x_ll.max(), y_ll.max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, linewidth=2, label='Perfect')
        ax.set_xlabel("GT log10(Σ e²)", fontsize=12)
        ax.set_ylabel("Pred log10(Σ σ²)", fontsize=12)
        ax.set_title(f"Global Variance vs Error (log-log)\nPearson = {pearson_ll:.3f} | Spearman = {spear_global:.3f}", fontsize=14, fontweight='bold')
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "global_scatter_varsum_loglog.png", dpi=200, bbox_inches='tight')
        plt.close()

    # ===== 2) 内点概率分布（若提供了 pred_q 与 gt_inlier）=====
    if (pred_q is not None) and (gt_inlier is not None):
        fig, ax = plt.subplots(figsize=(10, 5))
        q_in = pred_q[gt_inlier > 0.5].ravel()
        q_out = pred_q[gt_inlier < 0.5].ravel()
        if len(q_out): ax.hist(q_out, bins=50, alpha=0.6, color='red',
                               label=f'Outlier (n={len(q_out)})', density=True)
        if len(q_in):  ax.hist(q_in,  bins=50, alpha=0.6, color='green',
                               label=f'Inlier (n={len(q_in)})', density=True)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2,
                   label='Threshold=0.5')
        qb = (pred_q.ravel() > 0.5).astype(np.float32)
        q_acc = float((qb == gt_inlier.ravel()).mean())
        ax.set_xlabel("Predicted Inlier Probability (q)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Inlier Probability Distribution\nAccuracy = {q_acc:.3f}",
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper center', fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "q_distribution.png", dpi=200, bbox_inches='tight')
        plt.close()
        metrics["q_accuracy"] = q_acc

    # ===== 3) 残差直方图 =====
    if pred.shape[0] > 0:
        fig, axes = plt.subplots(1, D, figsize=(7*D, 5))
        if D == 1: axes = [axes]
        for i, name in enumerate(_axis_names(D)):
            ax = axes[i]
            residual = (pred[:, i] - gt[:, i]).astype(np.float64)
            ax.hist(residual, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            mean_res = residual.mean(); std_res = residual.std()
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax.axvline(mean_res, color='green', linestyle='--', linewidth=2,
                       label=f'Mean={mean_res:.3f}')
            ax.set_xlabel(f"Residual (Pred - GT) for {name}", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"Residual Distribution - Axis {name.upper() if D>1 else ''}\n"
                         f"Mean={mean_res:.3f}, Std={std_res:.3f}",
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "residual_analysis.png", dpi=200, bbox_inches='tight')
        plt.close()

        # 全局直方图（所有轴合并）
        residual_all = (pred - gt).astype(np.float64).reshape(-1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residual_all, bins=80, alpha=0.7, color='steelblue', edgecolor='black')
        mean_all = residual_all.mean(); std_all = residual_all.std()
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(mean_all, color='green', linestyle='--', linewidth=2, label=f'Mean={mean_all:.3f}')
        ax.set_xlabel("Residual (Pred - GT) over all axes", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Residual Distribution - Global\nMean={mean_all:.3f}, Std={std_all:.3f}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "residual_analysis_global.png", dpi=200, bbox_inches='tight')
        plt.close()
        metrics["residual_global_mean"] = float(mean_all)
        metrics["residual_global_std"] = float(std_all)

    # ===== 4) q 阈值扫描（可选）=====
    if scan_q_threshold and (pred_q is not None) and (pred is not None) and pred.shape[0] > 0:
        thresholds = np.linspace(0.3, 0.8, 11)
        spear_mean_list, n_samples_list = [], []
        for thr in thresholds:
            mask = (pred_q.ravel() > thr)
            if mask.sum() < 10:
                spear_mean_list.append(np.nan); n_samples_list.append(0); continue
            sm = 0.0
            for i in range(D):
                sm += spearman_np(pred[mask, i], gt[mask, i])
            sm /= D
            spear_mean_list.append(sm); n_samples_list.append(int(mask.sum()))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.plot(thresholds, spear_mean_list, '^-', linewidth=2.5, markersize=7)
        best_idx = int(np.nanargmax(spear_mean_list))
        ax.scatter(thresholds[best_idx], spear_mean_list[best_idx],
                   s=200, c='gold', marker='*', edgecolors='black',
                   linewidths=2, zorder=10, label='Best')
        ax.set_xlabel("q Threshold", fontsize=12)
        ax.set_ylabel("Spearman (mean over dims)", fontsize=12)
        ax.set_title(f"Spearman vs q Threshold\nBest: q>{thresholds[best_idx]:.2f}, "
                     f"Spear={spear_mean_list[best_idx]:.4f}",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, ls="--", alpha=0.3)
        ax = axes[1]
        ax.plot(thresholds, n_samples_list, 'o-', linewidth=2, markersize=6)
        ax.axvline(thresholds[best_idx], color='red', linestyle='--', linewidth=2,
                   label=f'Best Threshold={thresholds[best_idx]:.2f}')
        ax.set_xlabel("q Threshold", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title("Sample Count vs q Threshold", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "threshold_scan.png", dpi=200, bbox_inches='tight')
        plt.close()

    # ===== 5) 各向异性（仅 D==2 与 VIS 对齐时绘制）=====
    if pred.shape[0] > 0 and pred.shape[1] == 2:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            aniso_pred = (pred[:, 0] - pred[:, 1]) / 2.0
            aniso_gt   = (gt[:, 0]   - gt[:, 1])   / 2.0
            ax = axes[0]
            if gt_inlier is not None:
                mask_in  = gt_inlier.ravel() > 0.5
                mask_out = gt_inlier.ravel() < 0.5
                if mask_out.any():
                    ax.scatter(aniso_gt[mask_out], aniso_pred[mask_out],
                               s=10, alpha=0.3, c='gray', label='Outlier')
                if mask_in.any():
                    ax.scatter(aniso_gt[mask_in], aniso_pred[mask_in],
                               s=10, alpha=0.6, c='blue', label='Inlier')
            else:
                ax.scatter(aniso_gt, aniso_pred, s=10, alpha=0.5, c='blue')
            lo = min(aniso_gt.min(), aniso_pred.min())
            hi = max(aniso_gt.max(), aniso_pred.max())
            ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, linewidth=2, label='Perfect')
            ax.set_xlabel("GT Anisotropy a = (lvx - lvy)/2", fontsize=11)
            ax.set_ylabel("Pred Anisotropy a", fontsize=11)
            ax.set_title("Anisotropy Prediction", fontsize=12, fontweight='bold')
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend(fontsize=9); ax.set_aspect('equal', adjustable='box')

            ax = axes[1]
            ax.hist(aniso_gt,   bins=50, alpha=0.5, color='green',
                    label=f'GT (mean={aniso_gt.mean():.3f})', density=True)
            ax.hist(aniso_pred, bins=50, alpha=0.5, color='blue',
                    label=f'Pred (mean={aniso_pred.mean():.3f})', density=True)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel("Anisotropy a", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title("Anisotropy Distribution", fontsize=12, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "anisotropy_analysis.png",
                        dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"警告: 各向异性图绘制失败 - {e}")

    try:
        with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    except Exception as _:
        pass


def _load_npz(path):
    d = np.load(path)
    if "pred_logvar" in d:
        pred = d["pred_logvar"]
    else:
        xs = [k for k in d.keys() if k.startswith("pred_logvar")]
        if not xs:
            raise KeyError("找不到 pred_logvar*")
        parts = []
        for axis in ["x", "y", "z"]:
            k = f"pred_logvar_{axis}"
            if k in d:
                parts.append(d[k].reshape(-1, 1))
        pred = np.concatenate(parts, 1) if parts else d[xs[0]]

    if "gt_logvar" in d:
        gt = d["gt_logvar"]
    else:
        xs = [k for k in d.keys() if k.startswith("gt_logvar")]
        if not xs:
            raise KeyError("找不到 gt_logvar*")
        parts = []
        for axis in ["x", "y", "z"]:
            k = f"gt_logvar_{axis}"
            if k in d:
                parts.append(d[k].reshape(-1, 1))
        gt = np.concatenate(parts, 1) if parts else d[xs[0]]

    pred_q = d["pred_q"] if "pred_q" in d else None
    gt_in  = d["gt_inlier"] if "gt_inlier" in d else None
    return pred, gt, pred_q, gt_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_npz", required=True, help="eval 导出的 npz 路径")
    ap.add_argument("--out_dir",   required=True, help="输出图目录")
    ap.add_argument("--scan_q_threshold", action="store_true")
    args = ap.parse_args()
    pred, gt, pred_q, gt_in = _load_npz(args.preds_npz)
    vis_plot_all(pred, gt, pred_q, gt_in, args.out_dir, args.scan_q_threshold)


if __name__ == "__main__":
    main()
