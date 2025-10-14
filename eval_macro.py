# -*- coding: utf-8 -*-
import argparse, json
import os
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from vis.datasets.macro_frames import MacroFrames
from models.macro_transformer_sa import MacroTransformerSA
from vis.calibration.affine import AffineCalibrator

def spearman_np(x, y):
    """
    正确的 Spearman 相关系数实现
    先中心化，再用中心化后的值计算归一化系数
    """
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()  # 先中心化
    ry -= ry.mean()  # 先中心化
    # 用中心化后的值计算归一化系数
    denom = np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()) + 1e-12
    return float((rx * ry).sum() / denom)

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
    ap.add_argument("--scan_q_threshold", action="store_true", help="扫描q阈值以找到最优Spearman")
    # === OOF 相关参数 ===
    ap.add_argument("--subset_idx_file", type=str, default=None,
                   help="Optional .npy of 1D indices to select a subset for evaluation (for K-fold OOF)")
    ap.add_argument("--dump_preds_npz", type=str, default=None,
                   help="导出预测结果到 npz 文件（用于 OOF 聚合）")
    ap.add_argument("--calibrator_json", type=str, default=None,
                   help="加载 OOF 校准器并应用到预测结果")
    args = ap.parse_args()

    # === OOF：加载子集索引 ===
    subset_idx = None
    if args.subset_idx_file is not None and os.path.isfile(args.subset_idx_file):
        subset_idx = np.load(args.subset_idx_file)
        print(f"[OOF] 加载评测子集索引: {args.subset_idx_file}")
        print(f"[OOF] 子集大小: {len(subset_idx)}")

    # === 修改：传递geom_stats_path和subset_idx到数据集 ===
    ds = MacroFrames(args.npz, geom_stats_path=args.geom_stats_npz, subset_idx=subset_idx)
    # Windows兼容：使用num_workers=0避免多进程序列化问题
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

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
    
    # === OOF：应用校准器 ===
    if args.calibrator_json is not None and os.path.isfile(args.calibrator_json):
        print(f"\n[OOF] 加载校准器: {args.calibrator_json}")
        with open(args.calibrator_json, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        
        cal_x = AffineCalibrator.from_dict(calib_data["x"])
        cal_y = AffineCalibrator.from_dict(calib_data["y"])
        
        print(f"  LogVar-X: α={cal_x.alpha:.4f}, β={cal_x.beta:.4f}")
        print(f"  LogVar-Y: α={cal_y.alpha:.4f}, β={cal_y.beta:.4f}")
        
        pred[:, 0] = cal_x.apply(pred[:, 0])
        pred[:, 1] = cal_y.apply(pred[:, 1])
        print("  ✅ 校准已应用")
    
    # === OOF：导出预测结果 ===
    if args.dump_preds_npz is not None:
        os.makedirs(os.path.dirname(args.dump_preds_npz) or ".", exist_ok=True)
        np.savez(args.dump_preds_npz,
                 pred_logvar_x=pred[:, 0],
                 pred_logvar_y=pred[:, 1],
                 gt_logvar_x=gt[:, 0],
                 gt_logvar_y=gt[:, 1],
                 q=pred_q.ravel(),
                 idx=np.arange(len(pred)))
        print(f"\n✅ 预测结果已导出到: {args.dump_preds_npz}")
        return  # 导出后直接返回，不进行后续评测
    
    # === 全样本spearman（参考值，可能被外点拉低）===
    sx = spearman_np(pred[:,0], gt[:,0])
    sy = spearman_np(pred[:,1], gt[:,1])
    
    metrics = {
        "spearman_x_all": sx, 
        "spearman_y_all": sy, 
        "spearman_mean_all": 0.5*(sx+sy)
    }
    
    # === 获取内点标签（用于评测指标计算）===
    try:
        gt_inlier = []
        for b in DataLoader(ds, batch_size=64, shuffle=False, num_workers=0):
            gt_inlier.append(b["y_inlier"])
        gt_inlier = torch.cat(gt_inlier, 0).numpy().ravel()
    except (KeyError, AttributeError):
        gt_inlier = None
    
    if gt_inlier is not None:
        
        # === 1) 只在GT内点上计算（与训练口径一致）===
        mask_gt = gt_inlier > 0.5
        if mask_gt.any():
            sx_in = spearman_np(pred[mask_gt, 0], gt[mask_gt, 0])
            sy_in = spearman_np(pred[mask_gt, 1], gt[mask_gt, 1])
            metrics["spearman_x_inlier_gt"] = float(sx_in)
            metrics["spearman_y_inlier_gt"] = float(sy_in)
            metrics["spearman_mean_inlier_gt"] = float(0.5*(sx_in+sy_in))
            metrics["n_inliers_gt"] = int(mask_gt.sum())
        
        # === 2) 用预测的内点(q>0.5)做掩码（推理时更贴近实战）===
        mask_q = (pred_q.ravel() > 0.5)
        if mask_q.any():
            sx_q = spearman_np(pred[mask_q, 0], gt[mask_q, 0])
            sy_q = spearman_np(pred[mask_q, 1], gt[mask_q, 1])
            metrics["spearman_x_predq"] = float(sx_q)
            metrics["spearman_y_predq"] = float(sy_q)
            metrics["spearman_mean_predq"] = float(0.5*(sx_q+sy_q))
            metrics["n_inliers_pred"] = int(mask_q.sum())
        
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
    else:
        # 旧数据没有y_inlier标签
        print("警告: 数据集没有y_inlier标签，无法计算内点掩码的spearman")
    
    print(json.dumps(metrics, indent=2))
    
    # === 可选：扫描q阈值找到最优Spearman ===
    if args.scan_q_threshold and 'y_inlier' in ds.__dict__ or hasattr(ds, 'y_inlier'):
        print("\n" + "="*60)
        print("扫描 q 阈值以优化 Spearman 相关系数")
        print("="*60)
        thresholds = np.linspace(0.3, 0.8, 11)
        best_thr = 0.5
        best_spear = -1.0
        
        print(f"{'Threshold':>10} {'n_samples':>10} {'Spear_x':>10} {'Spear_y':>10} {'Spear_mean':>12}")
        print("-" * 60)
        
        for thr in thresholds:
            mask_thr = (pred_q.ravel() > thr)
            if mask_thr.sum() < 10:  # 至少需要10个样本
                continue
            
            sx_thr = spearman_np(pred[mask_thr, 0], gt[mask_thr, 0])
            sy_thr = spearman_np(pred[mask_thr, 1], gt[mask_thr, 1])
            sm_thr = 0.5 * (sx_thr + sy_thr)
            
            print(f"{thr:>10.2f} {mask_thr.sum():>10d} {sx_thr:>10.4f} {sy_thr:>10.4f} {sm_thr:>12.4f}", end="")
            
            if sm_thr > best_spear:
                best_spear = sm_thr
                best_thr = thr
                print("  ← Best")
            else:
                print()
        
        print("-" * 60)
        print(f"最优阈值: q > {best_thr:.2f}, Spearman Mean = {best_spear:.4f}")
        print()

    if args.plots_dir:
        p = Path(args.plots_dir); p.mkdir(parents=True, exist_ok=True)
        
        # ========== 1. LogVar 散点图（分轴，带内点标记）==========
        try:
            for i, axis in enumerate(["x", "y"]):
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # 绘制外点（灰色）
                if gt_inlier is not None:
                    mask_out = gt_inlier < 0.5
                    if mask_out.any():
                        ax.scatter(gt[mask_out, i], pred[mask_out, i], 
                                 s=10, alpha=0.3, c='gray', label='Outlier (GT)')
                    
                    # 绘制内点（彩色）
                    mask_in = gt_inlier > 0.5
                    if mask_in.any():
                        ax.scatter(gt[mask_in, i], pred[mask_in, i], 
                                 s=10, alpha=0.6, c='blue', label='Inlier (GT)')
                else:
                    ax.scatter(gt[:, i], pred[:, i], s=10, alpha=0.5, c='blue')
                
                # 对角线（完美预测）
                lims = [min(gt[:, i].min(), pred[:, i].min()), 
                        max(gt[:, i].max(), pred[:, i].max())]
                ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect')
                
                ax.set_xlabel(f"GT log(σ{axis}²)", fontsize=12)
                ax.set_ylabel(f"Pred log(σ{axis}²)", fontsize=12)
                ax.set_title(f"LogVar Prediction - Axis {axis.upper()}\nSpearman = {metrics.get(f'spearman_{axis}_all', 0):.3f}", 
                           fontsize=14, fontweight='bold')
                ax.grid(True, ls="--", alpha=0.3)
                ax.legend(loc='upper left', fontsize=10)
                ax.set_aspect('equal', adjustable='box')
                plt.tight_layout()
                plt.savefig(p / f"scatter_logvar_{axis}.png", dpi=200, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"警告: 散点图绘制失败 - {e}")
        
        # ========== 2. 内点概率分布（直方图）==========
        try:
            if gt_inlier is not None:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # GT内点的q分布
                q_inlier = pred_q[gt_inlier > 0.5].ravel()
                # GT外点的q分布
                q_outlier = pred_q[gt_inlier < 0.5].ravel()
                
                ax.hist(q_outlier, bins=50, alpha=0.6, color='red', 
                       label=f'Outlier (n={len(q_outlier)})', density=True)
                ax.hist(q_inlier, bins=50, alpha=0.6, color='green', 
                       label=f'Inlier (n={len(q_inlier)})', density=True)
                
                ax.axvline(0.5, color='black', linestyle='--', linewidth=2, 
                          label='Threshold=0.5')
                ax.set_xlabel("Predicted Inlier Probability (q)", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.set_title(f"Inlier Probability Distribution\nAccuracy = {metrics.get('q_accuracy', 0):.3f}, AUC = {metrics.get('q_auc', 0):.3f}", 
                           fontsize=14, fontweight='bold')
                ax.legend(loc='upper center', fontsize=10)
                ax.grid(True, ls="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(p / "q_distribution.png", dpi=200, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"警告: q分布图绘制失败 - {e}")
        
        
        # ========== 3. 各向异性分析（椭圆轴比）==========
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 计算各向异性 a = (lvx - lvy) / 2
            aniso_pred = (pred[:, 0] - pred[:, 1]) / 2.0
            aniso_gt = (gt[:, 0] - gt[:, 1]) / 2.0
            
            # 左图：各向异性散点图
            ax = axes[0]
            if gt_inlier is not None:
                mask_in = gt_inlier > 0.5
                mask_out = gt_inlier < 0.5
                if mask_out.any():
                    ax.scatter(aniso_gt[mask_out], aniso_pred[mask_out], 
                             s=10, alpha=0.3, c='gray', label='Outlier')
                if mask_in.any():
                    ax.scatter(aniso_gt[mask_in], aniso_pred[mask_in], 
                             s=10, alpha=0.6, c='blue', label='Inlier')
            else:
                ax.scatter(aniso_gt, aniso_pred, s=10, alpha=0.5, c='blue')
            
            lims = [min(aniso_gt.min(), aniso_pred.min()), 
                    max(aniso_gt.max(), aniso_pred.max())]
            ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect')
            ax.set_xlabel("GT Anisotropy a = (lvx - lvy)/2", fontsize=11)
            ax.set_ylabel("Pred Anisotropy a", fontsize=11)
            ax.set_title("Anisotropy Prediction", fontsize=12, fontweight='bold')
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            
            # 右图：各向异性分布对比
            ax = axes[1]
            ax.hist(aniso_gt, bins=50, alpha=0.5, color='green', 
                   label=f'GT (mean={aniso_gt.mean():.3f})', density=True)
            ax.hist(aniso_pred, bins=50, alpha=0.5, color='blue', 
                   label=f'Pred (mean={aniso_pred.mean():.3f})', density=True)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel("Anisotropy a", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title("Anisotropy Distribution", fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, ls="--", alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(p / "anisotropy_analysis.png", dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"警告: 各向异性图绘制失败 - {e}")
        
        # ========== 4. 阈值扫描曲线（如果执行了扫描）==========
        if args.scan_q_threshold and 'gt_inlier' in locals():
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                thresholds = np.linspace(0.3, 0.8, 11)
                spear_x_list, spear_y_list, spear_mean_list = [], [], []
                n_samples_list = []
                
                for thr in thresholds:
                    mask_thr = (pred_q.ravel() > thr)
                    if mask_thr.sum() < 10:
                        spear_x_list.append(np.nan)
                        spear_y_list.append(np.nan)
                        spear_mean_list.append(np.nan)
                        n_samples_list.append(0)
                        continue
                    
                    sx_thr = spearman_np(pred[mask_thr, 0], gt[mask_thr, 0])
                    sy_thr = spearman_np(pred[mask_thr, 1], gt[mask_thr, 1])
                    sm_thr = 0.5 * (sx_thr + sy_thr)
                    
                    spear_x_list.append(sx_thr)
                    spear_y_list.append(sy_thr)
                    spear_mean_list.append(sm_thr)
                    n_samples_list.append(mask_thr.sum())
                
                # 左图：Spearman vs 阈值
                ax = axes[0]
                ax.plot(thresholds, spear_x_list, 'o-', label='Spearman X', linewidth=2, markersize=6)
                ax.plot(thresholds, spear_y_list, 's-', label='Spearman Y', linewidth=2, markersize=6)
                ax.plot(thresholds, spear_mean_list, '^-', label='Spearman Mean', 
                       linewidth=2.5, markersize=7, color='red')
                
                # 标记最优点
                best_idx = np.nanargmax(spear_mean_list)
                ax.scatter(thresholds[best_idx], spear_mean_list[best_idx], 
                          s=200, c='gold', marker='*', edgecolors='black', 
                          linewidths=2, zorder=10, label='Best')
                
                ax.set_xlabel("q Threshold", fontsize=12)
                ax.set_ylabel("Spearman Correlation", fontsize=12)
                ax.set_title(f"Spearman vs q Threshold\nBest: q>{thresholds[best_idx]:.2f}, Spear={spear_mean_list[best_idx]:.4f}", 
                           fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, ls="--", alpha=0.3)
                
                # 右图：样本数 vs 阈值
                ax = axes[1]
                ax.plot(thresholds, n_samples_list, 'o-', linewidth=2, 
                       markersize=6, color='purple')
                ax.axvline(thresholds[best_idx], color='red', linestyle='--', 
                          linewidth=2, label=f'Best Threshold={thresholds[best_idx]:.2f}')
                ax.set_xlabel("q Threshold", fontsize=12)
                ax.set_ylabel("Number of Samples", fontsize=12)
                ax.set_title("Sample Count vs q Threshold", fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, ls="--", alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(p / "threshold_scan.png", dpi=200, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"警告: 阈值扫描图绘制失败 - {e}")
        
        # ========== 5. 误差分析（残差分布）==========
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for i, axis in enumerate(["x", "y"]):
                ax = axes[i]
                residual = pred[:, i] - gt[:, i]
                
                # 绘制残差直方图
                ax.hist(residual, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                
                # 统计信息
                mean_res = residual.mean()
                std_res = residual.std()
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                ax.axvline(mean_res, color='green', linestyle='--', linewidth=2, 
                          label=f'Mean={mean_res:.3f}')
                
                ax.set_xlabel(f"Residual (Pred - GT) for {axis}", fontsize=11)
                ax.set_ylabel("Frequency", fontsize=11)
                ax.set_title(f"Residual Distribution - Axis {axis.upper()}\nMean={mean_res:.3f}, Std={std_res:.3f}", 
                           fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, ls="--", alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(p / "residual_analysis.png", dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"警告: 残差分析图绘制失败 - {e}")
        
        print(f"\n✅ 可视化图表已保存到: {p}/")
        print(f"   - scatter_logvar_x.png / scatter_logvar_y.png (LogVar散点图)")
        print(f"   - q_distribution.png (内点概率分布)")
        print(f"   - anisotropy_analysis.png (各向异性分析)")
        if args.scan_q_threshold:
            print(f"   - threshold_scan.png (阈值扫描曲线)")
        print(f"   - residual_analysis.png (残差分布)")
        print()

if __name__ == "__main__":
    main()

