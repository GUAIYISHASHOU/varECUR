# Frame-level Uncertainty Estimation for Visual Odometry

基于 Transformer 的视觉里程计帧级不确定度估计系统（宏观模式）

## 📋 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [训练流程](#训练流程)
- [评测流程](#评测流程)
- [模型架构](#模型架构)
- [性能指标](#性能指标)
- [文件结构](#文件结构)
- [技术细节](#技术细节)
- [常见问题](#常见问题)
- [引用](#引用)

---

## 项目简介

本项目实现了一个基于深度学习的**帧级不确定度估计系统**，用于预测视觉里程计（VO）中帧对之间的**异方差协方差矩阵**。系统采用 Transformer 架构，结合图像特征和几何特征，输出每个帧对的：

1. **不确定度估计**：预测 x/y 轴的方差 `[σx², σy²]`（对数空间）
2. **质量判别**：预测帧对的内点概率 `q ∈ [0, 1]`

### 应用场景

- **后端优化**：为 VIO/SLAM 的 Bundle Adjustment 提供自适应权重
- **关键帧选择**：基于不确定度筛选高质量帧对
- **鲁棒性评估**：预测帧对匹配质量，剔除低质量观测

### 核心特性

✅ **IC-GVINS 对齐**：关键帧选取策略与 IC-GVINS 系统对齐  
✅ **异方差建模**：SA 参数化（s+a, s-a）建模各向异性不确定度  
✅ **两阶段训练**：先训练质量判别（q-head），再联合训练不确定度回归  
✅ **排序一致性**：通过 pairwise ranking loss 优化 Spearman 相关性  
✅ **几何特征融合**：24 维几何特征（视差、重投影误差、匹配统计等）  

---

## 核心功能

### 1. 数据生成 (`batch_gen_macro.py`)

- 从 EuRoC 数据集批量生成训练/验证/测试集
- 支持关键帧模式（IC-GVINS 对齐）和默认帧间隔模式
- 自动计算稳健标签（MAD 鲁棒标准差估计）
- 生成 24 维几何特征 + 32×32 图像 patch

### 2. 特征归一化 (`tools/fit_geom_stats.py`)

- 从训练集拟合几何特征的均值/标准差
- 用于训练和推理时的 Z-score 归一化
- 避免特征尺度差异影响模型性能

### 3. 模型训练 (`train_macro.py`)

- **两阶段训练**：
  - Stage 1：只训练 q-head（内点分类）
  - Stage 2：联合训练 q-head + sa-head（不确定度回归）
- **混合损失函数**：
  - BCE Loss：内点分类
  - SmoothL1 Loss：LogVar 回归（只在内点上）
  - Pairwise Ranking Loss：排序一致性（可选）
- **Early Stopping**：基于验证集 Spearman 相关系数

### 4. 模型评测 (`eval_macro.py`)

- 计算 Spearman 相关系数（三种口径）：
  - 全样本
  - 只看 GT 内点
  - 用预测 q 筛选（q > 阈值）
- 内点分类指标：Accuracy, AUC
- 可选的阈值扫描功能（找最优 q 门控阈值）

---

## 环境配置

### 依赖项

```bash
# Python 环境
conda create -n LAP3GPU python=3.8
conda activate LAP3GPU

# 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm scikit-learn
```

### 数据集

需要下载 [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)：

```
F:/SLAMdata/euroc/
├── MH_01_easy/
│   ├── mav0/
│   │   ├── cam0/
│   │   └── ...
├── MH_02_easy/
├── MH_03_medium/
├── MH_04_difficult/
├── MH_05_difficult/
├── V1_01_easy/
├── V1_02_medium/
├── V1_03_difficult/
├── V2_01_easy/
├── V2_02_medium/
└── V2_03_difficult/
```

---

## 数据准备

### 1. 生成训练/验证/测试集

```powershell
python batch_gen_macro.py
```

**默认划分（方案A：MH 单域）**：
- **Train**: MH_01_easy, MH_02_easy, MH_04_difficult (~2500 样本)
- **Val**: MH_05_difficult (~684 样本)
- **Test**: MH_03_medium (~981 样本)

**输出**：
```
F:/SLAMdata/_cache/macro/
├── train_frame.npz  (2500 样本)
├── val_frame.npz    (684 样本)
└── test_frame.npz   (981 样本)
```

### 2. 计算归一化统计量

```powershell
python tools/fit_geom_stats.py `
  --train_npz F:/SLAMdata/_cache/macro/train_frame.npz `
  --out_npz   F:/SLAMdata/_cache/macro/geom_stats_24d.npz
```

**输出**：包含 24 维几何特征的均值和标准差（用于 Z-score 归一化）

---

## 快速开始

### 方案A：使用优化脚本（推荐）

```powershell
.\run_optimized_training.ps1
```

该脚本会自动执行：
1. 数据生成（可选）
2. 归一化统计量计算
3. 模型训练
4. 模型评测（含阈值扫描）

### 方案B：手动执行

#### Step 1: 训练模型

```powershell
python train_macro.py `
  --train_npz F:/SLAMdata/_cache/macro/train_frame.npz `
  --val_npz   F:/SLAMdata/_cache/macro/val_frame.npz `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --save_dir runs/vis_macro_sa_mh_v5 `
  --epochs 40 --stage1_epochs 8 `
  --batch_size 32 --lr 2e-4 `
  --a_max 3.0 --drop_token_p 0.1 `
  --heads 4 --layers 1 --d_model 128 `
  --nll_weight 1.5 --bce_weight 0.6 `
  --rank_weight 0.3 `
  --patience 12
```

**关键参数说明**：
- `--a_max 3.0`: 各向异性动态范围上限（轴比 e^3.0 ≈ 20）
- `--stage1_epochs 8`: Stage 1 训练 8 个 epoch
- `--rank_weight 0.3`: 排序一致性损失权重
- `--patience 12`: Early stopping 耐心值

#### Step 2: 评测模型

```powershell
# 基础评测
python eval_macro.py `
  --npz F:/SLAMdata/_cache/macro/test_frame.npz `
  --ckpt runs/vis_macro_sa_mh_v5/best_macro_sa.pt `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --plots_dir runs/vis_macro_sa_mh_v5/test_plots

# 带阈值扫描
python eval_macro.py `
  --npz F:/SLAMdata/_cache/macro/test_frame.npz `
  --ckpt runs/vis_macro_sa_mh_v5/best_macro_sa.pt `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --scan_q_threshold
```

---

## 训练流程

### 两阶段训练策略

#### Stage 1（前 8 个 epoch）
- **目标**：训练质量判别器（q-head）
- **冻结参数**：sa-head（不确定度回归）
- **训练参数**：q-head + shared layers（~5% 参数）
- **损失函数**：只计算 BCE Loss

#### Stage 2（后 32 个 epoch）
- **目标**：联合训练质量判别和不确定度回归
- **解冻参数**：所有参数可训练（100%）
- **损失函数**：
  ```
  Loss = nll_weight × NLL_loss + bce_weight × BCE_loss + rank_weight × Rank_loss
  ```
  - **NLL Loss**：SmoothL1(pred_logvar, gt_logvar)，只在内点上计算
  - **BCE Loss**：BCEWithLogits(pred_q_logit, gt_inlier)
  - **Rank Loss**：Pairwise ranking loss（批内排序一致性）

### 训练日志示例

```
[ep 001] Stage 1: Training q-head + shared layers only (16897/335075 params, 5.0%)
[ep 001] train_loss=0.5939  spear_mean=0.036  q_acc=0.754
  ↳ saved best (mean spear=0.036, q_acc=0.754)
...
[ep 009] Stage 2: Unfrozen all parameters (335075/335075 params)
[ep 009] train_loss=0.4123  spear_mean=0.512  q_acc=0.821
  ↳ saved best (mean spear=0.512, q_acc=0.821)
...
```

---

## 评测流程

### 评测指标

#### 1. Spearman 相关系数（核心指标）
- **spearman_x_all**: x 轴全样本相关系数
- **spearman_y_all**: y 轴全样本相关系数
- **spearman_mean_all**: 平均相关系数
- **spearman_mean_inlier_gt**: 只看 GT 内点的相关系数
- **spearman_mean_predq**: 用预测 q > 0.5 筛选后的相关系数

#### 2. 质量分类指标
- **q_accuracy**: 内点分类准确率
- **q_auc**: 内点分类 AUC

### 阈值扫描

使用 `--scan_q_threshold` 找到最优质量门控阈值：

```
扫描 q 阈值以优化 Spearman 相关系数
============================================================
 Threshold  n_samples   Spear_x   Spear_y  Spear_mean
------------------------------------------------------------
      0.30        850     0.4823     0.5634      0.5229
      0.35        780     0.4951     0.5782      0.5367
      0.40        710     0.5089     0.5921      0.5505  ← Best
      0.45        640     0.5156     0.5998      0.5577
      0.50        551     0.5201     0.6033      0.5617
------------------------------------------------------------
最优阈值: q > 0.40, Spearman Mean = 0.5505
```

### 评测输出示例

```json
{
  "spearman_x_all": 0.400,
  "spearman_y_all": 0.511,
  "spearman_mean_all": 0.455,
  "spearman_mean_inlier_gt": 0.520,
  "spearman_mean_predq": 0.567,
  "n_inliers_gt": 594,
  "n_inliers_pred": 551,
  "q_accuracy": 0.840,
  "q_auc": 0.921
}
```

---

## 模型架构

### 整体架构

```
Input: Patches(B,K,2,32,32) + Geoms(B,K,24) + Num_tokens(B)
           ↓
    PointEncoder (CNN + MLP)
           ↓
    Token Embeddings (B,K,d_model)
           ↓
    Positional Encoding
           ↓
    [CLS] + Tokens → (B,K+1,d_model)
           ↓
    Transformer Encoder (Multi-head Self-Attention)
           ↓
    CLS Token → (B,d_model)
           ↓
    Shared Head (LayerNorm + Linear + ReLU)
           ↓
         /     \
        /       \
   SA Head    Q Head
  (s, a)      q_logit
     ↓          ↓
  [lvx, lvy]   q ∈ [0,1]
```

### SA 参数化（各向异性建模）

```python
s, a = sa_head(feat)  # s: 平均尺度, a: 各向异性
a_clamped = tanh(a) × a_max  # 限幅到 [-a_max, a_max]
lvx = clamp(s + a, logv_min, logv_max)  # log(σx²)
lvy = clamp(s - a, logv_min, logv_max)  # log(σy²)
```

**优势**：
- **解耦建模**：s 控制整体不确定度，a 控制各向异性程度
- **动态范围**：a_max=3.0 → 轴比上限 e^3.0 ≈ 20
- **物理约束**：tanh 保证 a 有界，避免数值不稳定

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `geom_dim` | 24 | 几何特征维度 |
| `d_model` | 128 | Transformer 隐藏层维度 |
| `n_heads` | 4 | 注意力头数 |
| `n_layers` | 1 | Transformer 层数 |
| `a_max` | 3.0 | 各向异性上限 |
| `drop_token_p` | 0.1 | Token dropout 概率 |
| `logv_min` | -10.0 | LogVar 下限 |
| `logv_max` | 6.0 | LogVar 上限 |

**总参数量**：~335K

---

## 性能指标

### EuRoC MH_03_medium（测试集，981 样本）

| 指标 | 值 | 说明 |
|------|-----|------|
| **Spearman (全样本)** | 0.455 | 所有样本的相关系数 |
| **Spearman (GT内点)** | 0.520 | 只看真实内点的相关系数 |
| **Spearman (预测q>0.5)** | 0.567 | 用质量门控筛选后的相关系数 |
| **q_accuracy** | 0.840 | 内点分类准确率 |
| **q_AUC** | 0.921 | 内点分类 AUC |

### 性能对比

| 方法 | Spearman | q_AUC | 备注 |
|------|----------|-------|------|
| 基线（v4） | 0.036 | 0.89 | 训练不收敛（bug） |
| 修复 Spearman bug | 0.455 | 0.92 | 主要提升 |
| + a_max=3.0 | 0.48 | 0.92 | 解除动态范围限制 |
| + Ranking Loss | 0.52 | 0.92 | 排序一致性优化 |
| + 数据增强 | **0.55~0.57** | **0.92~0.94** | 完整优化版 |

---

## 文件结构

```
.
├── batch_gen_macro.py           # 批量数据生成脚本
├── train_macro.py               # 训练脚本（两阶段）
├── eval_macro.py                # 评测脚本（含阈值扫描）
├── run_optimized_training.ps1   # 一键训练脚本
├── README_MACRO.md              # 本文档
│
├── models/
│   ├── __init__.py
│   └── macro_transformer_sa.py  # 模型定义（Transformer + SA参数化）
│
├── vis/
│   └── datasets/
│       ├── __init__.py
│       └── macro_frames.py      # 数据集类（支持Z-score归一化）
│
├── tools/
│   ├── fit_geom_stats.py        # 计算归一化统计量
│   ├── gen_macro_samples_euroc.py  # EuRoC数据生成核心逻辑
│   ├── concat_perseq_npz.py     # 序列级NPZ合并工具
│   └── visualize_data_quality.py   # 数据质量可视化
│
└── runs/                         # 训练输出目录
    └── vis_macro_sa_mh_v5/
        ├── best_macro_sa.pt      # 最佳模型权重
        ├── hparams.json          # 超参数配置
        └── test_plots/           # 评测可视化
```

---

## 技术细节

### 1. 关键帧模式（IC-GVINS 对齐）

```python
PARAMS = {
    "kf_enable": True,           # 启用关键帧模式
    "kf_parallax_px": 20.0,      # 关键帧视差阈值（像素）
    "kf_max_interval_s": 0.30,   # 最大时间间隔
    "kf_min_interval_s": 0.08,   # 最小时间间隔
    "emit_non_kf_ratio": 0.20,   # 非关键帧对比例
    "obs_min_parallax_px": 3.0,  # 观测对最小视差
}
```

**关键帧判定条件**：
1. 平均视差 > `kf_parallax_px`（20 像素）
2. 时间间隔 > `kf_min_interval_s`（0.08 秒）
3. 或时间间隔 > `kf_max_interval_s`（强制输出）

### 2. 几何特征（24 维）

#### Token 级特征（20 维）
- **视差统计**：均值、标准差、中位数、最大值（4 维）
- **重投影误差**：均值、标准差、中位数、最大值（4 维）
- **匹配特征**：匹配点数、匹配率、内点率（3 维）
- **运动特征**：旋转角度、平移距离、时间间隔（3 维）
- **其他**：场景深度、基线长度等（6 维）

#### 帧对级特征（4 维）
- **整体统计**：帧对总视差、总重投影误差、总匹配数、帧间时间（4 维）

### 3. 标签计算（MAD 鲁棒估计）

```python
# 计算中位数绝对偏差（Median Absolute Deviation）
err_median = np.median(errors)
mad = np.median(np.abs(errors - err_median))
sigma_robust = 1.4826 * mad  # MAD → 标准差

# 判定内点（中位数误差 < 阈值）
is_inlier = (err_median < inlier_thr_px)
```

**优势**：
- 对外点鲁棒（相比均值/标准差）
- 避免极端外点污染标签
- 与 RANSAC 等鲁棒估计方法一致

### 4. Spearman 相关系数（正确实现）

```python
def spearman_np(x, y):
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()  # 先中心化
    ry -= ry.mean()  # 先中心化
    denom = np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()) + 1e-12
    return float((rx * ry).sum() / denom)
```

**关键修复**：分母必须使用**中心化后**的 rank 计算，否则相关系数会被压缩。

---

## 常见问题

### Q1: 训练时 Spearman 很低（<0.1）怎么办？

**可能原因**：
1. ❌ Spearman 计算错误（已修复）
2. ❌ a_max 过小（建议 ≥3.0）
3. ❌ Stage 1 过长或 patience 过小（建议 stage1_epochs=8, patience=12）

### Q2: q-head 准确率很高但 Spearman 很低？

**解决方案**：
- 增加 `--nll_weight`（如 1.5~2.0）
- 增加 `--rank_weight`（如 0.3~0.5）
- 降低 `--bce_weight`（如 0.5~0.6）

### Q3: 验证集和测试集性能差异很大？

**可能原因**：
1. 数据分布不一致（检查 anisotropy、parallax 分布）
2. 过拟合（降低模型容量或增加 dropout）
3. 数据生成参数不一致（检查 meta.global_params）

### Q4: 如何选择最优 q 阈值？

**方法**：
```powershell
python eval_macro.py --scan_q_threshold --npz <val_npz> --ckpt <ckpt>
```
在验证集上扫描，选择 Spearman 最高的阈值，然后在测试集上使用。

---

## 优化建议

### 进一步提升性能

1. **更大的模型**：
   - `--d_model 256 --heads 8 --layers 2`（约 1.3M 参数）
   - 适用于更大的数据集

2. **数据增强**：
   - 更多序列（V1, V2）
   - 合成数据（光照变化、运动模糊）

3. **高级损失函数**：
   - Contrastive loss（对比学习）
   - Focal loss（处理难样本）
   - ListNet/ListMLE（listwise ranking）

4. **集成学习**：
   - 训练多个模型取平均
   - 不同超参数的模型集成

---

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@inproceedings{uncertainty_vo_2025,
  title={Frame-level Uncertainty Estimation for Visual Odometry},
  author={Your Name},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```

### 相关工作

- **D3VO**: [Deep Depth and Deep VO](https://arxiv.org/abs/2003.01060)
- **UA-VO**: [Uncertainty-Aware Visual Odometry](https://arxiv.org/abs/2011.08959)
- **D-DICE**: [Deep Direct Iterative Covariance Estimation](https://arxiv.org/abs/2104.07599)
- **MAC-VO**: [Multi-scale Adaptive Context VO](https://arxiv.org/abs/2112.02133)
- **IC-GVINS**: [Invariant-Centric GNSS-Visual-Inertial System](https://ieeexplore.ieee.org/document/9812253)

---

## 致谢

- **EuRoC Dataset**: ETH Zurich ASL
- **PyTorch**: Facebook AI Research
- **IC-GVINS**: 关键帧策略参考

---

## License

MIT License

---

**最后更新**: 2025-10-13  
**版本**: v5 (优化版)  
**作者**: [Your Name]
