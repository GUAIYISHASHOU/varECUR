#!/usr/bin/env python3
"""
对比多个NPZ文件的数据质量和分布差异
验证物理原理的正确使用
"""
import argparse
import numpy as np
from pathlib import Path
import sys

def analyze_npz(npz_path):
    """分析单个NPZ文件，返回统计信息"""
    data = np.load(npz_path)
    
    stats = {
        'path': npz_path,
        'name': Path(npz_path).stem,
        'n_windows': data['X_IMU_ACC'].shape[0],
    }
    
    # 加速度计统计
    x_acc = data['X_IMU_ACC']
    e2_acc = data['E2_IMU_ACC']
    
    stats['acc_input_mean'] = np.mean(x_acc)
    stats['acc_input_std'] = np.std(x_acc)
    stats['acc_input_max'] = np.max(np.abs(x_acc))
    
    stats['acc_e2_mean'] = np.mean(e2_acc)
    stats['acc_e2_std'] = np.std(e2_acc)
    stats['acc_e2_max'] = np.max(e2_acc)
    stats['acc_e2_median'] = np.median(e2_acc)
    stats['acc_e2_p95'] = np.percentile(e2_acc, 95)
    stats['acc_e2_p99'] = np.percentile(e2_acc, 99)
    
    # 加速度计信噪比
    signal_power_acc = np.mean(x_acc ** 2)
    noise_power_acc = np.mean(e2_acc)
    stats['acc_snr_db'] = 10 * np.log10(signal_power_acc / noise_power_acc) if noise_power_acc > 0 else float('inf')
    
    # 陀螺仪统计
    x_gyr = data['X_IMU_GYR']
    e2_gyr = data['E2_IMU_GYR']
    
    stats['gyr_input_mean'] = np.mean(x_gyr)
    stats['gyr_input_std'] = np.std(x_gyr)
    stats['gyr_input_max'] = np.max(np.abs(x_gyr))
    
    stats['gyr_e2_mean'] = np.mean(e2_gyr)
    stats['gyr_e2_std'] = np.std(e2_gyr)
    stats['gyr_e2_max'] = np.max(e2_gyr)
    stats['gyr_e2_median'] = np.median(e2_gyr)
    stats['gyr_e2_p95'] = np.percentile(e2_gyr, 95)
    stats['gyr_e2_p99'] = np.percentile(e2_gyr, 99)
    
    # 陀螺仪信噪比
    signal_power_gyr = np.mean(x_gyr ** 2)
    noise_power_gyr = np.mean(e2_gyr)
    stats['gyr_snr_db'] = 10 * np.log10(signal_power_gyr / noise_power_gyr) if noise_power_gyr > 0 else float('inf')
    
    # 掩码覆盖率
    mask = data['MASK_IMU']
    stats['mask_coverage'] = (mask > 0).sum() / mask.size * 100
    
    # 物理合理性检查
    stats['acc_has_negative'] = np.any(e2_acc < 0)
    stats['gyr_has_negative'] = np.any(e2_gyr < 0)
    stats['acc_has_nan'] = np.any(np.isnan(e2_acc))
    stats['gyr_has_nan'] = np.any(np.isnan(e2_gyr))
    stats['acc_has_inf'] = np.any(np.isinf(e2_acc))
    stats['gyr_has_inf'] = np.any(np.isinf(e2_gyr))
    
    return stats

def print_comparison_table(all_stats):
    """打印对比表格"""
    print("\n" + "="*100)
    print("数据质量对比表")
    print("="*100)
    
    # 基本信息
    print("\n## 基本信息 ##\n")
    print(f"{'序列':<20} {'窗口数':>10} {'掩码覆盖率':>12}")
    print("-"*50)
    for s in all_stats:
        print(f"{s['name']:<20} {s['n_windows']:>10} {s['mask_coverage']:>11.2f}%")
    
    # 加速度计
    print("\n## 加速度计 ##\n")
    print(f"{'序列':<20} {'输入均值':>10} {'输入标准差':>12} {'误差²均值':>12} {'误差²中位数':>14} {'SNR(dB)':>10}")
    print("-"*90)
    for s in all_stats:
        print(f"{s['name']:<20} {s['acc_input_mean']:>10.4f} {s['acc_input_std']:>12.4f} "
              f"{s['acc_e2_mean']:>12.6f} {s['acc_e2_median']:>14.6f} {s['acc_snr_db']:>10.2f}")
    
    # 陀螺仪
    print("\n## 陀螺仪 ##\n")
    print(f"{'序列':<20} {'输入均值':>10} {'输入标准差':>12} {'误差²均值':>12} {'误差²中位数':>14} {'SNR(dB)':>10}")
    print("-"*90)
    for s in all_stats:
        print(f"{s['name']:<20} {s['gyr_input_mean']:>10.4f} {s['gyr_input_std']:>12.4f} "
              f"{s['gyr_e2_mean']:>12.6f} {s['gyr_e2_median']:>14.6f} {s['gyr_snr_db']:>10.2f}")
    
    # 误差分布对比
    print("\n## 误差分布 (P95 / P99) ##\n")
    print(f"{'序列':<20} {'加速度计 P95':>15} {'加速度计 P99':>15} {'陀螺仪 P95':>15} {'陀螺仪 P99':>15}")
    print("-"*85)
    for s in all_stats:
        print(f"{s['name']:<20} {s['acc_e2_p95']:>15.6f} {s['acc_e2_p99']:>15.6f} "
              f"{s['gyr_e2_p95']:>15.6f} {s['gyr_e2_p99']:>15.6f}")

def check_physics_validity(all_stats):
    """检查物理原理的正确使用"""
    print("\n" + "="*100)
    print("物理原理验证")
    print("="*100 + "\n")
    
    all_valid = True
    
    for s in all_stats:
        print(f"\n【{s['name']}】")
        issues = []
        
        # 检查误差平方非负
        if s['acc_has_negative']:
            issues.append("  ❌ 加速度计误差平方包含负值（违反物理定义）")
            all_valid = False
        else:
            print("  ✓ 加速度计误差平方非负")
        
        if s['gyr_has_negative']:
            issues.append("  ❌ 陀螺仪误差平方包含负值（违反物理定义）")
            all_valid = False
        else:
            print("  ✓ 陀螺仪误差平方非负")
        
        # 检查NaN/Inf
        if s['acc_has_nan'] or s['acc_has_inf']:
            issues.append("  ⚠️  加速度计误差包含 NaN 或 Inf")
            all_valid = False
        else:
            print("  ✓ 加速度计误差无异常值")
        
        if s['gyr_has_nan'] or s['gyr_has_inf']:
            issues.append("  ⚠️  陀螺仪误差包含 NaN 或 Inf")
            all_valid = False
        else:
            print("  ✓ 陀螺仪误差无异常值")
        
        # 检查信噪比合理性
        if s['acc_snr_db'] < 10 or s['acc_snr_db'] > 40:
            print(f"  ⚠️  加速度计信噪比异常: {s['acc_snr_db']:.2f} dB (正常范围: 10-40 dB)")
        else:
            print(f"  ✓ 加速度计信噪比正常: {s['acc_snr_db']:.2f} dB")
        
        if s['gyr_snr_db'] < 5 or s['gyr_snr_db'] > 30:
            print(f"  ⚠️  陀螺仪信噪比异常: {s['gyr_snr_db']:.2f} dB (正常范围: 5-30 dB)")
        else:
            print(f"  ✓ 陀螺仪信噪比正常: {s['gyr_snr_db']:.2f} dB")
        
        # 检查掩码覆盖率
        if s['mask_coverage'] < 95:
            print(f"  ⚠️  掩码覆盖率较低: {s['mask_coverage']:.2f}%")
        else:
            print(f"  ✓ 掩码覆盖率良好: {s['mask_coverage']:.2f}%")
        
        for issue in issues:
            print(issue)
    
    print("\n" + "="*100)
    if all_valid:
        print("✅ 所有序列都符合物理原理")
        print("✅ 误差平方 = (测量值 - Ground Truth)²")
    else:
        print("❌ 发现物理原理违规")
    print("="*100)

def print_summary(all_stats):
    """打印总结"""
    print("\n" + "="*100)
    print("数据质量总结")
    print("="*100 + "\n")
    
    # 总窗口数
    total_windows = sum(s['n_windows'] for s in all_stats)
    print(f"总序列数: {len(all_stats)}")
    print(f"总窗口数: {total_windows}")
    
    # 平均信噪比
    avg_acc_snr = np.mean([s['acc_snr_db'] for s in all_stats])
    avg_gyr_snr = np.mean([s['gyr_snr_db'] for s in all_stats])
    print(f"\n平均信噪比:")
    print(f"  加速度计: {avg_acc_snr:.2f} dB")
    print(f"  陀螺仪:   {avg_gyr_snr:.2f} dB")
    
    # 误差水平
    avg_acc_e2 = np.mean([s['acc_e2_mean'] for s in all_stats])
    avg_gyr_e2 = np.mean([s['gyr_e2_mean'] for s in all_stats])
    print(f"\n平均误差平方:")
    print(f"  加速度计: {avg_acc_e2:.6f} (m/s²)²")
    print(f"  陀螺仪:   {avg_gyr_e2:.6f} (rad/s)²")
    
    # 数据质量评级
    print("\n数据质量评级:")
    if avg_acc_snr > 15 and avg_gyr_snr > 10:
        print("  🌟🌟🌟 优秀 (SNR充足，标签质量高)")
    elif avg_acc_snr > 12 and avg_gyr_snr > 8:
        print("  🌟🌟 良好 (SNR适中，可用于训练)")
    else:
        print("  🌟 一般 (SNR偏低，可能需要调整参数)")
    
    print("\n" + "="*100)

def main():
    ap = argparse.ArgumentParser(description="对比多个NPZ文件的数据质量")
    ap.add_argument('--root', required=True, help='NPZ文件所在目录')
    ap.add_argument('--pattern', default='*_T50_S25.npz', help='文件名匹配模式')
    args = ap.parse_args()
    
    root = Path(args.root)
    npz_files = sorted(root.glob(args.pattern))
    
    if not npz_files:
        print(f"❌ 未找到匹配的NPZ文件: {root / args.pattern}")
        sys.exit(1)
    
    print(f"\n找到 {len(npz_files)} 个NPZ文件")
    
    # 分析所有文件
    all_stats = []
    for npz_file in npz_files:
        print(f"  分析: {npz_file.name}")
        stats = analyze_npz(npz_file)
        all_stats.append(stats)
    
    # 打印对比
    print_comparison_table(all_stats)
    
    # 验证物理原理
    check_physics_validity(all_stats)
    
    # 总结
    print_summary(all_stats)

if __name__ == '__main__':
    main()
