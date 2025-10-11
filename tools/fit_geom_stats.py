#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算训练NPZ文件中 'geoms' 特征的均值和标准差
这些统计量用于训练和推理时的Z-score归一化
"""
import argparse
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="计算几何特征的归一化统计量"
    )
    parser.add_argument("--train_npz", type=str, required=True,
                        help="训练NPZ文件路径 (例如: train_frame.npz)")
    parser.add_argument("--out_npz", type=str, required=True,
                        help="输出统计量NPZ文件路径 (例如: geoms_stats.npz)")
    args = parser.parse_args()

    print(f"正在加载数据: {args.train_npz}")
    data = np.load(args.train_npz)
    geoms = data['geoms']           # (M, K, D)
    num_tokens = data['num_tokens'] # (M,)
    
    M, K, D = geoms.shape
    print(f"找到 {M} 个样本, K_max={K}, geom_dim={D}")

    # 创建mask选择有效token（非padding）
    token_indices = np.arange(K)
    mask = token_indices[None, :] < num_tokens[:, None]  # (M, K)
    
    # 应用mask并flatten得到所有有效token
    valid_geoms = geoms[mask]  # (N_valid_tokens, D)
    
    print(f"有效token总数: {valid_geoms.shape[0]:,}")
    
    # 计算均值和标准差
    print("正在计算均值和标准差...")
    mean = np.mean(valid_geoms, axis=0, dtype=np.float32)
    std = np.std(valid_geoms, axis=0, dtype=np.float32)
    
    # 安全检查：防止除以零（常量特征）
    std[std < 1e-6] = 1.0
    
    print("\n统计量计算完成:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    
    # 打印每个特征的统计信息
    if 'meta' in data and 'geoms_desc' in data['meta'].item():
        geoms_desc = data['meta'].item()['geoms_desc']
        print("\n各特征统计信息:")
        print(f"{'Feature':<25} {'Mean':>12} {'Std':>12}")
        print("-" * 52)
        for i, desc in enumerate(geoms_desc):
            print(f"{desc:<25} {mean[i]:>12.6f} {std[i]:>12.6f}")
    else:
        print("\nMean:", mean)
        print("Std:", std)
    
    # 保存到输出文件
    np.savez(args.out_npz, mean=mean, std=std)
    print(f"\n✅ 统计量已保存到: {args.out_npz}")

if __name__ == "__main__":
    main()

