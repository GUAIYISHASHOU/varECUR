#!/usr/bin/env python3
"""
找出"问题折"：检测训练曲线不降的fold
用于快速定位OOF训练中的异常折
"""
import numpy as np
import json
import argparse
from pathlib import Path


def fold_curve_summary(curve_dir: str, min_epochs: int = 6):
    """
    读取每折训练日志，找到"不降"的折
    
    Args:
        curve_dir: 包含curve_fold*.json文件的目录
        min_epochs: 最小epoch数（少于此数的折会被跳过）
    
    Returns:
        bad_folds: 问题折的列表
    """
    curve_dir = Path(curve_dir)
    curve_files = sorted(curve_dir.glob("curve_fold*.json"))
    
    if not curve_files:
        print(f"No curve files found in {curve_dir}")
        return []
    
    print(f"\n{'='*80}")
    print(f"Analyzing {len(curve_files)} fold curves")
    print(f"{'='*80}\n")
    
    bad = []
    
    for curve_file in curve_files:
        # 提取fold_id
        fold_id = int(curve_file.stem.replace("curve_fold", ""))
        
        with open(curve_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_losses = np.array(data.get('train', []), dtype=float)
        val_losses = np.array(data.get('val', []), dtype=float)
        
        if len(val_losses) < min_epochs:
            print(f"Fold {fold_id}: Skipped (only {len(val_losses)} epochs)")
            continue
        
        # 检测1: 验证损失幅度极小（最后6个epoch）
        tail = val_losses[-6:]
        amplitude = tail.max() - tail.min()
        
        # 检测2: 线性回归斜率接近0
        x = np.arange(len(val_losses))
        slope = np.polyfit(x, val_losses, 1)[0]
        
        # 检测3: 验证损失始终很高（相对于其他折）
        final_val = val_losses[-1]
        
        is_bad = False
        reasons = []
        
        if amplitude < 1e-3:
            is_bad = True
            reasons.append(f"幅度极小 ({amplitude:.2e})")
        
        if abs(slope) < 5e-4:
            is_bad = True
            reasons.append(f"斜率接近0 ({slope:.2e})")
        
        if is_bad:
            bad.append(fold_id)
            print(f"⚠️  Fold {fold_id}: POTENTIALLY BAD")
            print(f"    Reasons: {', '.join(reasons)}")
            print(f"    Final val loss: {final_val:.4f}")
            print(f"    Val loss range: [{val_losses.min():.4f}, {val_losses.max():.4f}]")
        else:
            print(f"✓  Fold {fold_id}: OK")
            print(f"    Final val loss: {final_val:.4f}")
            print(f"    Slope: {slope:.2e}, Amplitude: {amplitude:.4f}")
        print()
    
    return sorted(bad)


def main():
    ap = argparse.ArgumentParser(description="检测问题折")
    ap.add_argument("--curve_dir", required=True, help="包含curve_fold*.json的目录")
    ap.add_argument("--min_epochs", type=int, default=6, help="最小epoch数")
    args = ap.parse_args()
    
    bad_folds = fold_curve_summary(args.curve_dir, args.min_epochs)
    
    print(f"{'='*80}")
    if bad_folds:
        print(f"⚠️  Found {len(bad_folds)} potentially bad fold(s): {bad_folds}")
        print(f"\n建议：")
        print(f"  1. 检查这些fold的数据分布（使用 debug_folds_imu.py）")
        print(f"  2. 确认scaler是否按折隔离")
        print(f"  3. 检查是否存在数据泄漏")
    else:
        print(f"✓ All folds look good!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
