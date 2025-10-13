# -*- coding: utf-8 -*-
"""
从终端日志或文件解析训练历史，绘制训练曲线
"""
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_from_text(log_text):
    """从日志文本解析训练历史"""
    pattern = r'\[ep (\d+)\] (?:Stage [12].*?\n)?\[ep \d+\] train_loss=([\d.]+)\s+spear_mean=([-\d.]+)\s+q_acc=([\d.]+)'
    
    epochs, losses, spears, q_accs = [], [], [], []
    
    for line in log_text.split('\n'):
        # 匹配格式: [ep 001] train_loss=0.3465  spear_mean=-0.080  q_acc=0.804
        match = re.search(r'\[ep (\d+)\] train_loss=([\d.]+)\s+spear_mean=([-\d.]+)\s+q_acc=([\d.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            spears.append(float(match.group(3)))
            q_accs.append(float(match.group(4)))
    
    return epochs, losses, spears, q_accs

# 从用户日志复制（或从文件读取）
log_text = """
[ep 001] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 001] train_loss=0.3465  spear_mean=-0.080  q_acc=0.804
[ep 002] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 002] train_loss=0.2759  spear_mean=-0.079  q_acc=0.817
[ep 003] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 003] train_loss=0.2591  spear_mean=-0.082  q_acc=0.878
[ep 004] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 004] train_loss=0.2533  spear_mean=-0.095  q_acc=0.872
[ep 005] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 005] train_loss=0.2485  spear_mean=-0.098  q_acc=0.881
[ep 006] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 006] train_loss=0.2460  spear_mean=-0.105  q_acc=0.882
[ep 007] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 007] train_loss=0.2442  spear_mean=-0.097  q_acc=0.878
[ep 008] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 008] train_loss=0.2420  spear_mean=-0.098  q_acc=0.877
[ep 009] Stage 2: Unfrozen all parameters (335075/335075 params)
[ep 009] train_loss=0.3851  spear_mean=0.501  q_acc=0.905
[ep 010] train_loss=0.3064  spear_mean=0.585  q_acc=0.905
[ep 011] train_loss=0.2847  spear_mean=0.590  q_acc=0.922
[ep 012] train_loss=0.2666  spear_mean=0.584  q_acc=0.922
[ep 013] train_loss=0.2496  spear_mean=0.617  q_acc=0.925
[ep 014] train_loss=0.2360  spear_mean=0.633  q_acc=0.922
[ep 015] train_loss=0.2295  spear_mean=0.587  q_acc=0.922
[ep 016] train_loss=0.2174  spear_mean=0.618  q_acc=0.920
[ep 017] train_loss=0.2147  spear_mean=0.631  q_acc=0.921
[ep 018] train_loss=0.2038  spear_mean=0.611  q_acc=0.922
[ep 019] train_loss=0.2023  spear_mean=0.625  q_acc=0.909
[ep 020] train_loss=0.1943  spear_mean=0.626  q_acc=0.884
[ep 021] train_loss=0.1847  spear_mean=0.613  q_acc=0.884
[ep 022] train_loss=0.1854  spear_mean=0.634  q_acc=0.919
[ep 023] train_loss=0.1744  spear_mean=0.643  q_acc=0.912
[ep 024] train_loss=0.1707  spear_mean=0.615  q_acc=0.901
[ep 025] train_loss=0.1624  spear_mean=0.612  q_acc=0.915
[ep 026] train_loss=0.1608  spear_mean=0.621  q_acc=0.900
[ep 027] train_loss=0.1570  spear_mean=0.630  q_acc=0.896
[ep 028] train_loss=0.1522  spear_mean=0.620  q_acc=0.905
[ep 029] train_loss=0.1532  spear_mean=0.610  q_acc=0.915
[ep 030] train_loss=0.1456  spear_mean=0.642  q_acc=0.905
[ep 031] train_loss=0.1405  spear_mean=0.619  q_acc=0.912
[ep 032] train_loss=0.1367  spear_mean=0.633  q_acc=0.909
[ep 033] train_loss=0.1353  spear_mean=0.631  q_acc=0.897
"""

epochs, losses, spears, q_accs = parse_log_from_text(log_text)

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss曲线
axes[0].plot(epochs, losses, 'o-', linewidth=2, markersize=4)
axes[0].axvline(8.5, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Train Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Spearman曲线
axes[1].plot(epochs, spears, 'o-', linewidth=2, markersize=4, color='green')
axes[1].axvline(8.5, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
axes[1].axhline(max(spears), color='orange', linestyle=':', alpha=0.7, label=f'Best={max(spears):.3f}')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Val Spearman', fontsize=12)
axes[1].set_title('Validation Spearman', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Q Accuracy曲线
axes[2].plot(epochs, q_accs, 'o-', linewidth=2, markersize=4, color='purple')
axes[2].axvline(8.5, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Val Q Accuracy', fontsize=12)
axes[2].set_title('Quality Prediction Accuracy', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig('runs/macro_icgvins_v1_normalized222/training_curves.png', dpi=200)
print("✓ 训练曲线已保存: runs/macro_icgvins_v1_normalized222/training_curves.png")
plt.show()

# 打印关键统计
print(f"\n训练总结:")
print(f"  总轮次: {len(epochs)}")
print(f"  最佳Spearman: {max(spears):.3f} (Epoch {epochs[spears.index(max(spears))]})")
print(f"  最终Loss: {losses[-1]:.4f}")
print(f"  最终Q Acc: {q_accs[-1]:.3f}")
