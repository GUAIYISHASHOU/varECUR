# -*- coding: utf-8 -*-
"""
从训练集拟合几何特征的 mean/std（只用有效 token）
保存为独立的统计量 NPZ，用于训练时 geoms 标准化
"""
import numpy as np
import os

# 训练集路径
train_npz_path = r"F:/SLAMdata/_cache/macro/train_frame.npz"

print(f"加载训练集: {train_npz_path}")
d = np.load(train_npz_path)

G = d["geoms"]            # (N, T, 24) 例如 (6553, 256, 24)
nt = d["num_tokens"]      # (N,)

print(f"  样本数: {G.shape[0]}")
print(f"  Token容量: {G.shape[1]}")
print(f"  几何特征维度: {G.shape[2]}")

# 只统计每样本前 nt[i] 个 token
T = G.shape[1]
mask = (np.arange(T)[None, :] < nt[:, None])  # (N, T) - boolean mask
# 用mask选出有效token
valid_geoms_list = []
for i in range(len(G)):
    n_tok = nt[i]
    valid_geoms_list.append(G[i, :n_tok, :])  # (n_tok, 24)
Gs = np.concatenate(valid_geoms_list, axis=0)  # (M, 24)

print(f"  有效token总数: {Gs.shape[0]}")

# 计算均值和标准差
mean = Gs.mean(0)
std  = Gs.std(0) + 1e-8

print(f"\n特征统计量:")
print(f"  Mean 范围: [{mean.min():.4f}, {mean.max():.4f}]")
print(f"  Std  范围: [{std.min():.4f}, {std.max():.4f}]")
print(f"  Std  最小值: {std.min():.6f} (应 > 1e-8)")

# 保存
output_path = r"F:/SLAMdata/_cache/macro/geom_stats_24d.npz"
np.savez(output_path, mean=mean, std=std)

print(f"\n✓ 已保存: {output_path}")
print(f"  shapes: mean={mean.shape}, std={std.shape}")
