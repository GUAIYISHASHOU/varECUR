#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量可视化工具
用于对比训练集和验证集的误差分布、图像质量等
帮助诊断train降val不降的问题
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_distributions(ax, data, name):
    """绘制误差分布直方图"""
    e2_total = data['e2x'] + data['e2y']
    err_total = np.sqrt(e2_total)
    
    # 统计信息
    mean_err = np.mean(err_total)
    median_err = np.median(err_total)
    p95_err = np.percentile(err_total, 95)
    
    ax.hist(err_total, bins=100, range=(0, 20), density=True, alpha=0.7, 
            label=f'{name.upper()} (mean={mean_err:.2f}, p95={p95_err:.2f})')
    ax.set_title(f'Reprojection Error Distribution - {name.capitalize()} Set')
    ax.set_xlabel('Reprojection Error (pixels)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本
    stats_text = f'Mean: {mean_err:.2f}px\nMedian: {median_err:.2f}px\n95th: {p95_err:.2f}px'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_patches(ax, data, name, indices):
    """绘制样本图像块"""
    patches = data['I0'][indices, 0]  # 显示第一帧的patch
    n = len(indices)
    for i in range(n):
        ax[i].imshow(patches[i], cmap='gray', vmin=0, vmax=1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if i == 0:
            ax[i].set_ylabel(f'{name.upper()}', fontsize=10, fontweight='bold')

def plot_geom_features(ax, data, name):
    """绘制几何特征分布（前4维：归一化坐标）"""
    geom = data['geom']
    
    # 只显示前4维（归一化坐标）
    for i in range(min(4, geom.shape[1])):
        ax.hist(geom[:, i], bins=50, alpha=0.5, label=f'dim {i}')
    
    ax.set_title(f'Geometric Features (coords) - {name.capitalize()}')
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_gradient_features(ax, data, name):
    """绘制梯度特征分布（第5-6维）"""
    geom = data['geom']
    
    if geom.shape[1] >= 6:
        g0 = geom[:, 4]  # gradient at frame 0
        g2 = geom[:, 5]  # gradient at frame 2
        
        ax.hist(g0, bins=50, alpha=0.6, label='g0 (frame 0)', range=(0, 100))
        ax.hist(g2, bins=50, alpha=0.6, label='g2 (frame 2)', range=(0, 100))
        
        ax.set_title(f'Gradient Magnitude - {name.capitalize()}')
        ax.set_xlabel('Gradient Value')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 统计低梯度样本比例
        low_grad_ratio = np.mean((g0 < 10) | (g2 < 10))
        ax.text(0.98, 0.98, f'Low gradient (<10): {low_grad_ratio*100:.1f}%', 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No gradient features\n(geom_dim < 6)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Gradient Features - {name.capitalize()}')

def main():
    parser = argparse.ArgumentParser(description="Compare train and validation NPZ data quality.")
    parser.add_argument('--train_npz', type=str, required=True, help='Path to train NPZ file')
    parser.add_argument('--val_npz', type=str, required=True, help='Path to validation NPZ file')
    parser.add_argument('--out', type=str, default='data_quality_comparison.png', 
                       help='Output figure path')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("📊 数据质量可视化工具")
    print("="*60)
    
    print(f"\n[1/3] 加载数据...")
    train_data = np.load(args.train_npz, allow_pickle=True)
    val_data = np.load(args.val_npz, allow_pickle=True)
    
    print(f"  训练集: {len(train_data['I0'])} 样本")
    print(f"  验证集: {len(val_data['I0'])} 样本")
    print(f"  几何特征维度: {train_data['geom'].shape[1]}")
    
    # 打印元数据
    if 'meta' in train_data:
        train_meta = train_data['meta'].item() if isinstance(train_data['meta'], np.ndarray) else train_data['meta']
        print(f"\n  训练集序列: {train_meta.get('seqs', 'N/A')}")
    if 'meta' in val_data:
        val_meta = val_data['meta'].item() if isinstance(val_data['meta'], np.ndarray) else val_data['meta']
        print(f"  验证集序列: {val_meta.get('seqs', 'N/A')}")
    
    print(f"\n[2/3] 生成可视化...")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 第一行：误差分布对比
    ax1 = fig.add_subplot(gs[0, :2])
    plot_distributions(ax1, train_data, 'train')
    ax2 = fig.add_subplot(gs[0, 2:])
    plot_distributions(ax2, val_data, 'val')
    
    # 第二行：几何特征分布
    ax3 = fig.add_subplot(gs[1, :2])
    plot_geom_features(ax3, train_data, 'train')
    ax4 = fig.add_subplot(gs[1, 2:])
    plot_geom_features(ax4, val_data, 'val')
    
    # 第三行：梯度特征分布
    ax5 = fig.add_subplot(gs[2, :2])
    plot_gradient_features(ax5, train_data, 'train')
    ax6 = fig.add_subplot(gs[2, 2:])
    plot_gradient_features(ax6, val_data, 'val')
    
    # 第四行：样本图像块
    print("  随机选择样本进行可视化...")
    rng = np.random.default_rng(0)
    train_indices = rng.choice(len(train_data['I0']), 4, replace=False)
    val_indices = rng.choice(len(val_data['I0']), 4, replace=False)
    
    train_patch_axes = [fig.add_subplot(gs[3, i]) for i in range(4)]
    plot_patches(train_patch_axes, train_data, 'train', train_indices)
    
    val_patch_axes = [fig.add_subplot(gs[3, i]) for i in range(4)]
    plot_patches(val_patch_axes, val_data, 'val', val_indices)
    
    # 添加总标题
    fig.suptitle('Training vs Validation Data Quality Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图像
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    
    print(f"\n[3/3] 保存完成")
    print(f"  ✅ 对比图已保存到: {out_path.absolute()}")
    
    # 计算并打印关键差异
    print(f"\n" + "="*60)
    print("📈 关键差异分析")
    print("="*60)
    
    train_err = np.sqrt(train_data['e2x'] + train_data['e2y'])
    val_err = np.sqrt(val_data['e2x'] + val_data['e2y'])
    
    print(f"\n误差统计:")
    print(f"  训练集 - 均值: {np.mean(train_err):.2f}px, 中位数: {np.median(train_err):.2f}px")
    print(f"  验证集 - 均值: {np.mean(val_err):.2f}px, 中位数: {np.median(val_err):.2f}px")
    print(f"  差异率: {(np.mean(val_err)/np.mean(train_err)-1)*100:+.1f}%")
    
    if train_data['geom'].shape[1] >= 6:
        train_grad = train_data['geom'][:, 4:6].mean()
        val_grad = val_data['geom'][:, 4:6].mean()
        print(f"\n梯度统计:")
        print(f"  训练集平均梯度: {train_grad:.2f}")
        print(f"  验证集平均梯度: {val_grad:.2f}")
        print(f"  差异率: {(val_grad/train_grad-1)*100:+.1f}%")
    
    if np.mean(val_err) > np.mean(train_err) * 1.2:
        print(f"\n⚠️  警告: 验证集误差比训练集高20%以上！")
        print(f"   这可能导致train降val不降的问题。")
        print(f"   建议:")
        print(f"   1. 检查数据划分是否合理（easy/medium/difficult混合）")
        print(f"   2. 增加训练集中的困难样本比例")
        print(f"   3. 使用质量感知训练（--photometric clahe）")
    
    print(f"\n" + "="*60 + "\n")
    
    # 显示图像（可选）
    try:
        plt.show()
    except:
        print("注意: 无法显示图形窗口（可能是无GUI环境），但图像已保存。")

if __name__ == "__main__":
    main()

