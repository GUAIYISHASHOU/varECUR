# -*- coding: utf-8 -*-
# tools/train_oof.py
"""
OOF (Out-of-Fold) 训练脚本
对训练集进行 K 折交叉训练，收集每个样本的 OOF 预测，然后拟合校准器
"""
import argparse
import os
import json
import subprocess
import tempfile
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from vis.calibration.affine import AffineCalibrator

# ===== 新增：稳健统计工具函数 =====
def winsorize(x, p=2.5):
    """Winsorize: 将极端值截断到指定百分位"""
    lo, hi = np.percentile(x, [p, 100-p])
    return np.clip(x, lo, hi)

def robust_std(x):
    """稳健标准差估计：基于IQR（四分位距）"""
    q1, q3 = np.percentile(x, [25, 75])
    return (q3 - q1) / 1.349  # 正态分布下 IQR -> sigma

def run_cmd(cmd):
    """运行命令并打印"""
    print("\n" + "="*80)
    print(">>", " ".join(cmd))
    print("="*80)
    subprocess.check_call(cmd)

def dump_fold_indices(tmp_dir, arr, name):
    """将索引数组保存为 npy 文件"""
    p = Path(tmp_dir) / f"{name}.npy"
    np.save(p.as_posix(), arr.astype(np.int64))
    return p.as_posix()

def main():
    ap = argparse.ArgumentParser(description="OOF 训练并拟合校准器")
    ap.add_argument("--train_npz", required=True, help="训练集 npz 文件")
    ap.add_argument("--geom_stats_npz", required=True, help="几何特征统计量文件")
    ap.add_argument("--kfold_json", required=True, help="K 折切分 JSON 文件")
    ap.add_argument("--save_root", required=True, help="保存根目录")
    
    # 训练超参数
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stage1_epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--a_max", type=float, default=3.0)
    ap.add_argument("--drop_token_p", type=float, default=0.1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nll_weight", type=float, default=1.5)
    ap.add_argument("--bce_weight", type=float, default=0.6)
    ap.add_argument("--rank_weight", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12)
    
    # === s/a 校准参数 ===
    ap.add_argument("--sa_mode", choices=["std", "robust"], default="robust",
                   help="s/a 校准方式: std=标准方差比, robust=稳健估计(默认)")
    ap.add_argument("--winsor_p", type=float, default=2.5,
                   help="Winsorize 百分位(robust模式用, 默认2.5)")
    
    args = ap.parse_args()

    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    
    # 加载 K 折配置
    with open(args.kfold_json, "r", encoding="utf-8") as f:
        kconf = json.load(f)
    folds = kconf["folds"]
    print(f"\n加载 K 折配置: {len(folds)} 折")

    # 载入训练集真值（用于最终拟合校准）
    tr_data = np.load(args.train_npz)
    y_true  = tr_data["y_true"]  # (M,2) logvar_x, logvar_y
    M = y_true.shape[0]
    print(f"训练集样本数: {M}")

    # 累积 OOF 预测容器
    oof_pred = np.full((M, 2), np.nan, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tdir:
        for k, f in enumerate(folds):
            print(f"\n{'='*80}")
            print(f"开始训练 Fold {k}/{len(folds)}")
            print(f"{'='*80}")
            
            tr_idx = np.array(f["train_idx"], dtype=np.int64)
            va_idx = np.array(f["val_idx"], dtype=np.int64)
            
            print(f"  训练样本: {len(tr_idx)}")
            print(f"  验证样本: {len(va_idx)}")

            # 保存训练索引到临时文件
            tr_idx_npy = dump_fold_indices(tdir, tr_idx, f"tr_idx_fold{k}")
            
            # 训练：只喂 train_idx 子集
            save_dir = os.path.join(args.save_root, f"fold{k}")
            train_cmd = [
                "python", "train_macro.py",
                "--train_npz", args.train_npz,
                "--val_npz", args.train_npz,  # 用同一文件，但 val 不使用 subset
                "--geom_stats_npz", args.geom_stats_npz,
                "--save_dir", save_dir,
                "--epochs", str(args.epochs),
                "--stage1_epochs", str(args.stage1_epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--a_max", str(args.a_max),
                "--drop_token_p", str(args.drop_token_p),
                "--heads", str(args.heads),
                "--layers", str(args.layers),
                "--d_model", str(args.d_model),
                "--nll_weight", str(args.nll_weight),
                "--bce_weight", str(args.bce_weight),
                "--rank_weight", str(args.rank_weight),
                "--patience", str(args.patience),
                "--subset_idx_file", tr_idx_npy  # 关键：训练只见本折的训练子集
            ]
            run_cmd(train_cmd)

            # 用该折模型在"未见过"的 va_idx 上做预测
            va_idx_npy = dump_fold_indices(tdir, va_idx, f"va_idx_fold{k}")
            pred_npz = os.path.join(save_dir, f"oof_fold{k}.npz")
            
            eval_cmd = [
                "python", "eval_macro.py",
                "--npz", args.train_npz,
                "--ckpt", os.path.join(save_dir, "best_macro_sa.pt"),
                "--geom_stats_npz", args.geom_stats_npz,
                "--subset_idx_file", va_idx_npy,  # 只评测本折的 holdout
                "--dump_preds_npz", pred_npz  # 导出预测
            ]
            run_cmd(eval_cmd)

            # 读取该折 OOF 预测结果
            ck = np.load(pred_npz)
            oof_pred[va_idx, 0] = ck["pred_logvar_x"]
            oof_pred[va_idx, 1] = ck["pred_logvar_y"]
            
            print(f"\n✅ Fold {k} 完成，OOF 预测已收集")

    # 检查 OOF 覆盖率
    coverage_x = (~np.isnan(oof_pred[:, 0])).sum()
    coverage_y = (~np.isnan(oof_pred[:, 1])).sum()
    print(f"\n{'='*80}")
    print(f"OOF 预测覆盖率:")
    print(f"  LogVar-X: {coverage_x}/{M} ({coverage_x/M*100:.1f}%)")
    print(f"  LogVar-Y: {coverage_y}/{M} ({coverage_y/M*100:.1f}%)")
    print(f"{'='*80}")

    # 拟合全训练集 OOF 的仿射校准器（x、y 各一套）
    print("\n开始拟合 OOF 校准器...")
    
    cal_x = AffineCalibrator("logvar_x_oof")
    cal_y = AffineCalibrator("logvar_y_oof")

    m_x = ~np.isnan(oof_pred[:, 0])
    m_y = ~np.isnan(oof_pred[:, 1])

    cal_x.fit(oof_pred[m_x, 0], y_true[m_x, 0])
    cal_y.fit(oof_pred[m_y, 1], y_true[m_y, 1])
    
    print(f"  LogVar-X 校准: α={cal_x.alpha:.4f}, β={cal_x.beta:.4f}")
    print(f"  LogVar-Y 校准: α={cal_y.alpha:.4f}, β={cal_y.beta:.4f}")

    # ===== New: s/a 域幅度校准 (在轴向仿射之后) =====
    m_xy = (~np.isnan(oof_pred[:, 0])) & (~np.isnan(oof_pred[:, 1]))
    # 先把 OOF 预测用 x/y 仿射拉正，再在 s/a 域拟合
    px = cal_x.apply(oof_pred[m_xy, 0])
    py = cal_y.apply(oof_pred[m_xy, 1])
    s_pred = 0.5 * (px + py)
    a_pred = 0.5 * (px - py)
    s_gt   = 0.5 * (y_true[m_xy, 0] + y_true[m_xy, 1])
    a_gt   = 0.5 * (y_true[m_xy, 0] - y_true[m_xy, 1])

    def _safe_var(x, eps=1e-12): return float(np.var(x)) + eps
    # s: 协方差比/方差比形式的仿射（多为平移修正）
    alpha_s = float(np.cov(s_pred, s_gt, bias=True)[0,1] / _safe_var(s_pred))
    beta_s  = float(s_gt.mean() - alpha_s * s_pred.mean())
    
    # a: 根据模式选择方差匹配方法
    mode = args.sa_mode
    if mode == "std":
        # 标准方法：直接用标准差比
        alpha_a = float(a_gt.std() / (a_pred.std() + 1e-8))
        beta_a  = float(a_gt.mean() - alpha_a * a_pred.mean())
    elif mode == "robust":
        # 稳健方法：先 winsorize 再用 robust_std
        p = args.winsor_p
        a_pred_w = winsorize(a_pred, p)
        a_gt_w = winsorize(a_gt, p)
        alpha_a = float(robust_std(a_gt_w) / (robust_std(a_pred_w) + 1e-8))
        beta_a  = float(a_gt.mean() - alpha_a * a_pred.mean())  # 均值对齐用原始数据
    
    sa_calib = {
        "version": 2,
        "mode": mode,
        "alpha_s": alpha_s, 
        "beta_s": beta_s, 
        "alpha_a": alpha_a, 
        "beta_a": beta_a
    }
    if mode == "robust":
        sa_calib["winsor_p"] = args.winsor_p
    
    print(f"  s/a calib [{mode}]: αs={alpha_s:.4f}, βs={beta_s:.4f}, αa={alpha_a:.4f}, βa={beta_a:.4f}")

    # 保存校准器（向后兼容）
    calib_path = os.path.join(args.save_root, "calibrator_oof.json")
    payload = {"x": cal_x.to_dict(), "y": cal_y.to_dict(), "sa_calib": sa_calib}
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ OOF 校准器已保存到: {calib_path}")
    print(f"\n{'='*80}")
    print("OOF 训练完成！")
    print(f"{'='*80}")
    print("\n下一步：使用以下命令在 val/test 上评测（应用校准）:")
    print(f"\npython eval_macro.py \\")
    print(f"  --npz <val_or_test.npz> \\")
    print(f"  --ckpt <your_model.pt> \\")
    print(f"  --geom_stats_npz {args.geom_stats_npz} \\")
    print(f"  --calibrator_json {calib_path} \\")
    print(f"  --scan_q_threshold \\")
    print(f"  --plots_dir <output_dir>")

if __name__ == "__main__":
    main()
