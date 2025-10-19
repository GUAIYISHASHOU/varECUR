from __future__ import annotations
import argparse
import numpy as np
import sys

"""
详细检查 npz 文件的数据分布情况
"""

def print_separator(char='=', length=80):
    print(char * length)

def print_stats(name: str, data: np.ndarray):
    """打印数组的统计信息"""
    print(f"\n【{name}】")
    print(f"  形状: {data.shape}")
    print(f"  类型: {data.dtype}")
    print(f"  均值: {np.mean(data):.6f}")
    print(f"  标准差: {np.std(data):.6f}")
    print(f"  最小值: {np.min(data):.6f}")
    print(f"  最大值: {np.max(data):.6f}")
    print(f"  中位数: {np.median(data):.6f}")
    
    # 检查异常值
    if data.dtype in [np.float32, np.float64]:
        n_nan = np.isnan(data).sum()
        n_inf = np.isinf(data).sum()
        if n_nan > 0:
            print(f"  ⚠️  NaN 数量: {n_nan}")
        if n_inf > 0:
            print(f"  ⚠️  Inf 数量: {n_inf}")
        
        # 检查零值
        n_zeros = (data == 0).sum()
        zero_ratio = n_zeros / data.size * 100
        print(f"  零值数量: {n_zeros} ({zero_ratio:.2f}%)")
        
        # 百分位数
        p1, p5, p25, p75, p95, p99 = np.percentile(data, [1, 5, 25, 75, 95, 99])
        print(f"  分位数:")
        print(f"    P1:  {p1:.6f}  |  P5:  {p5:.6f}")
        print(f"    P25: {p25:.6f}  |  P75: {p75:.6f}")
        print(f"    P95: {p95:.6f}  |  P99: {p99:.6f}")

def check_temporal_consistency(data: np.ndarray, name: str):
    """检查时间序列的连续性"""
    print(f"\n【{name} - 时间连续性】")
    if len(data.shape) >= 2:
        # 计算相邻时间步的差异
        N, T = data.shape[0], data.shape[1]
        if T > 1:
            diffs = np.diff(data, axis=1)  # (N, T-1, ...)
            diff_mean = np.mean(np.abs(diffs))
            diff_std = np.std(np.abs(diffs))
            diff_max = np.max(np.abs(diffs))
            
            print(f"  相邻步差异统计:")
            print(f"    均值: {diff_mean:.6f}")
            print(f"    标准差: {diff_std:.6f}")
            print(f"    最大值: {diff_max:.6f}")
            
            # 检查是否有突变
            threshold = diff_mean + 5 * diff_std
            n_jumps = (np.abs(diffs) > threshold).sum()
            if n_jumps > 0:
                print(f"  ⚠️  检测到 {n_jumps} 个异常跳变 (> μ+5σ)")

def check_error_distribution(e2: np.ndarray, x: np.ndarray, name: str):
    """检查误差平方的分布合理性"""
    print(f"\n【{name} - 误差合理性】")
    
    # 误差平方应该是非负的
    if np.any(e2 < 0):
        print(f"  ❌ 错误: 发现负值误差平方！")
        return
    else:
        print(f"  ✓ 所有误差平方非负")
    
    # 计算信噪比 (SNR)
    signal_power = np.mean(x ** 2)
    noise_power = np.mean(e2)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
        print(f"  信噪比 (SNR): {snr_db:.2f} dB")
    
    # 误差平方与信号强度的关系
    x_abs_mean = np.mean(np.abs(x), axis=(1,2))  # (N,)
    e2_mean = np.mean(e2, axis=(1,2))  # (N,)
    correlation = np.corrcoef(x_abs_mean, e2_mean)[0, 1]
    print(f"  误差与信号强度相关性: {correlation:.4f}")
    if abs(correlation) < 0.1:
        print(f"    → 低相关，可能使用了全局误差模型")
    elif correlation > 0.5:
        print(f"    → 高正相关，误差随信号增大")

def check_mask_coverage(mask: np.ndarray):
    """检查掩码覆盖率"""
    print(f"\n【MASK_IMU - 有效性】")
    total = mask.size
    valid = (mask > 0).sum()
    coverage = valid / total * 100
    print(f"  总样本数: {mask.shape[0]}")
    print(f"  时间步数: {mask.shape[1]}")
    print(f"  有效率: {valid}/{total} ({coverage:.2f}%)")
    
    # 检查每个样本的有效率
    per_sample_coverage = (mask > 0).sum(axis=1) / mask.shape[1]
    print(f"  样本有效率分布:")
    print(f"    最小: {per_sample_coverage.min():.2%}")
    print(f"    最大: {per_sample_coverage.max():.2%}")
    print(f"    均值: {per_sample_coverage.mean():.2%}")
    
    # 完全无效的样本
    n_empty = (per_sample_coverage == 0).sum()
    if n_empty > 0:
        print(f"  ⚠️  {n_empty} 个样本完全无效")

def main():
    ap = argparse.ArgumentParser(description="详细检查 npz 文件的数据分布")
    ap.add_argument('--npz', required=True, help='npz 文件路径')
    ap.add_argument('--verbose', action='store_true', help='显示更详细的信息')
    args = ap.parse_args()
    
    print(f"\n检查文件: {args.npz}")
    print_separator()
    
    try:
        d = np.load(args.npz)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        sys.exit(1)
    
    # 1. 基本信息
    print(f"\n✓ 文件加载成功")
    print(f"✓ 包含 {len(d.files)} 个键")
    print(f"\n键列表: {', '.join(d.files)}")
    
    # 2. 必需键检查
    required_keys = ['X_IMU_ACC', 'X_IMU_GYR', 'E2_IMU_ACC', 'E2_IMU_GYR', 'MASK_IMU']
    print(f"\n必需键检查:")
    for key in required_keys:
        if key in d.files:
            print(f"  ✓ {key}")
        else:
            print(f"  ❌ {key} (缺失)")
    
    print_separator()
    
    # 3. 详细统计
    print("\n## 数据统计信息 ##")
    
    # 加速度计
    if 'X_IMU_ACC' in d.files:
        print_stats("X_IMU_ACC (加速度计输入)", d['X_IMU_ACC'])
        check_temporal_consistency(d['X_IMU_ACC'], "X_IMU_ACC")
    
    # 陀螺仪
    if 'X_IMU_GYR' in d.files:
        print_stats("X_IMU_GYR (陀螺仪输入)", d['X_IMU_GYR'])
        check_temporal_consistency(d['X_IMU_GYR'], "X_IMU_GYR")
    
    # 误差平方 - 加速度计
    if 'E2_IMU_ACC' in d.files:
        print_stats("E2_IMU_ACC (加速度计误差平方)", d['E2_IMU_ACC'])
        if 'X_IMU_ACC' in d.files:
            check_error_distribution(d['E2_IMU_ACC'], d['X_IMU_ACC'], "E2_IMU_ACC")
    
    # 误差平方 - 陀螺仪
    if 'E2_IMU_GYR' in d.files:
        print_stats("E2_IMU_GYR (陀螺仪误差平方)", d['E2_IMU_GYR'])
        if 'X_IMU_GYR' in d.files:
            check_error_distribution(d['E2_IMU_GYR'], d['X_IMU_GYR'], "E2_IMU_GYR")
    
    # 掩码
    if 'MASK_IMU' in d.files:
        check_mask_coverage(d['MASK_IMU'])
    
    # 时间戳
    if 'TS_IMU' in d.files:
        ts = d['TS_IMU']
        print(f"\n【TS_IMU (时间戳)】")
        print(f"  形状: {ts.shape}")
        print(f"  类型: {ts.dtype}")
        if ts.size > 0:
            print(f"  起始时间: {ts.min()}")
            print(f"  结束时间: {ts.max()}")
            duration_ns = ts.max() - ts.min()
            duration_s = duration_ns / 1e9
            print(f"  持续时间: {duration_s:.2f} 秒")
            
            # 检查时间戳单调性
            if len(ts.shape) >= 2 and ts.shape[1] > 1:
                dts = np.diff(ts, axis=1)
                if np.all(dts > 0):
                    print(f"  ✓ 时间戳单调递增")
                    avg_dt = np.mean(dts) / 1e9
                    print(f"  平均采样间隔: {avg_dt*1000:.2f} ms ({1/avg_dt:.1f} Hz)")
                else:
                    print(f"  ⚠️  时间戳非单调")
    
    print_separator()
    
    # 4. 数据质量评分
    print("\n## 数据质量评估 ##\n")
    
    issues = []
    
    # 检查 NaN/Inf
    for key in d.files:
        if d[key].dtype in [np.float32, np.float64]:
            if np.isnan(d[key]).any():
                issues.append(f"  ⚠️  {key} 包含 NaN")
            if np.isinf(d[key]).any():
                issues.append(f"  ⚠️  {key} 包含 Inf")
    
    # 检查负值误差
    for key in ['E2_IMU_ACC', 'E2_IMU_GYR']:
        if key in d.files and np.any(d[key] < 0):
            issues.append(f"  ❌ {key} 包含负值")
    
    # 检查数据量
    if 'X_IMU_ACC' in d.files:
        n_samples = d['X_IMU_ACC'].shape[0]
        if n_samples < 100:
            issues.append(f"  ⚠️  样本数过少: {n_samples}")
    
    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(issue)
    else:
        print("✓ 未发现明显问题，数据质量良好")
    
    print_separator()
    print()

if __name__ == '__main__':
    main()
