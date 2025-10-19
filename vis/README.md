# Visual Uncertainty Estimation

独立的视觉不确定性估计模块，与IMU代码完全分离。

## 概述

从EuRoC立体相机对中学习2D重投影误差的不确定性（σx², σy²）。

### 特点
- 基于patch pairs + 几何特征的轻量级CNN+MLP
- Kendall异方差损失（per-axis log-variance）
- 可选Huber鲁棒性
- 温度校正支持

## 快速开始

### 1. 生成训练数据

```bash
python tools/gen_vis_pairs_euroc.py \
  --euroc_root F:/SLAMdata \
  --seq MH_01_easy \
  --camchain F:/SLAMdata/camchain-imucam.yaml \
  --delta 1 \
  --patch 32 \
  --max_pairs 20000 \
  --out data_vis/MH_01_easy_pairs.npz
```

**参数说明：**
- `--delta`: 时间间隔（帧数），推荐1-3
- `--patch`: patch大小，推荐32
- `--max_pairs`: 最大样本数

### 2. 训练模型

```bash
python train_vis.py \
  --train_npz data_vis/train_pairs.npz \
  --val_npz data_vis/val_pairs.npz \
  --epochs 20 \
  --batch 256 \
  --lr 1e-3 \
  --huber 1.0 \
  --save_dir runs/vis_uncert
```

### 3. 评估

```bash
python eval_vis.py \
  --npz data_vis/test_pairs.npz \
  --model runs/vis_uncert/best_vis_kendall.pt \
  --auto_temp global
```

**输出指标：**
- `z2_mean`: 归一化误差均值（理想=1）
- `cov68/cov95`: 68%/95%覆盖率
- `spearman`: 误差-方差相关性

## 数据格式

NPZ文件包含：
- `I0, I1`: [N,1,H,W] patch对
- `geom`: [N,4] 归一化坐标
- `e2x, e2y`: [N] 每轴误差平方
- `mask`: [N] 有效性掩码

## 依赖

```bash
pip install torch opencv-python scikit-learn pyyaml
```

## 架构

```
vis/
├── datasets/       # 数据加载
├── models/         # 模型定义
├── losses/         # 损失函数
└── README.md       # 本文档
```

## 与IMU代码的区别

| 特性 | IMU | Visual |
|------|-----|--------|
| 输入 | 时序IMU + 窗口DR | Patch pairs + 几何 |
| 输出 | 1D或3D log-variance | 2D log-variance (x,y) |
| 损失 | Huber-NLL on 3D | Kendall-NLL on 2D |
| 自由度 | df=3 | df=2 |

## 注意事项

1. **近似投影**: 当前版本使用恒等变换近似（适用于小Δ）。如需精确，请修改`gen_vis_pairs_euroc.py`中的投影部分，注入GT相机位姿。

2. **匹配质量**: ORB特征匹配可能在纹理弱区域失败。可调整`--max_pairs`或使用更鲁棒的特征（如SuperPoint）。

3. **校准**: 推荐训练后使用`--auto_temp global`做温度缩放。

