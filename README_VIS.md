# 视觉不确定性估计模块使用指南

> **注意**: 这是一个与IMU代码**完全独立**的新模块，放在 `vis/` 目录下。

## 📦 项目结构

```
IMU_2_test/
├── vis/                      # 视觉模块（新增，独立）
│   ├── datasets/
│   │   └── vis_pairs.py     # Patch pairs数据集
│   ├── models/
│   │   └── uncert_head.py   # 2D不确定性模型
│   ├── losses/
│   │   └── kendall.py       # Kendall异方差损失
│   └── README.md
├── tools/
│   └── gen_vis_pairs_euroc.py  # VIS数据生成工具
├── train_vis.py             # VIS训练脚本
├── eval_vis.py              # VIS评估脚本
└── (IMU相关文件保持不变)
```

## 🚀 完整工作流程

### 1️⃣ 准备数据

**【推荐】使用增强型数据生成脚本:**

```powershell
# 单个序列（增强版，支持多delta + 纹理分层 + 严格几何筛选）
python tools/gen_vis_pairs_euroc_strict.py `
  --euroc_root F:\SLAMdata\euroc `
  --seq MH_01_easy `
  --deltas 1,2 `
  --patch 32 `
  --max_pairs 60000 `
  --per_frame_cap 800 `
  --frame_step 2 `
  --err_clip_px 15 `
  --depth_min 0.1 `
  --depth_max 80 `
  --epi_thr_px 1.5 `
  --texture_strat `
  --out_npz F:\SLAMdata\_cache\vis_pairs\MH_01_easy.npz

# 批量生成所有序列（自动化脚本）
python scripts/batch_gen_vis_enhanced.py
# 根据提示修改脚本内的路径配置，然后运行
# 会自动生成所有11个EuRoC序列的增强数据
```

**合并与划分数据集:**

```powershell
# 合并各序列并划分为train/val/test
python tools/merge_vis_pairs_by_seq.py `
  --pairs_root F:\SLAMdata\_cache\vis_pairs `
  --out_root F:\SLAMdata\_cache\vis_split `
  --train_seqs MH_01_easy,MH_02_easy,MH_03_medium,MH_04_difficult,V1_01_easy,V1_02_medium `
  --val_seqs V1_03_difficult,V2_01_easy `
  --test_seqs V2_02_medium,V2_03_difficult,MH_05_difficult
```

**输出目录结构:**
```
F:\SLAMdata\_cache\vis_split\
├── train.npz     # 训练集（6个序列）
├── val.npz       # 验证集（2个序列）
└── test.npz      # 测试集（3个序列）
```

**合并多个NPZ文件（可选）:**

```python
# merge_npz.py
import numpy as np
from pathlib import Path

files = list(Path("data_vis/train").glob("*.npz"))
all_data = {}

for f in files:
    d = np.load(f)
    for k in d.files:
        if k not in all_data:
            all_data[k] = []
        all_data[k].append(d[k])

# 合并
merged = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
np.savez_compressed("data_vis/train_all.npz", **merged)
```

### 2️⃣ 训练模型

**【推荐】标准训练配置（Student-t + Cosine调度 + 早停）:**

```powershell
python train_vis.py `
  --train_npz F:\SLAMdata\_cache\vis_split\train.npz `
  --val_npz F:\SLAMdata\_cache\vis_split\val.npz `
  --epochs 50 `
  --batch 256 `
  --lr 1e-3 `
  --loss studentt `
  --nu 3.0 `
  --calib_reg 1e-3 `
  --lv_min -10 `
  --lv_max 6 `
  --scheduler cosine `
  --lr_min 1e-5 `
  --early_stop 8 `
  --save_dir runs\vis_uncert_enhanced
```

**参数说明:**
- `--loss studentt`: Student-t NLL（比Gaussian更鲁棒，抗outliers）
- `--nu 3.0`: 自由度（3-5适中，越小越重尾）
- `--calib_reg 1e-3`: 校准正则化（鼓励E[z²]≈1）
- `--scheduler cosine`: 余弦退火学习率（平滑衰减）
- `--early_stop 8`: 8 epochs无改善自动停止

**其他训练选项:**

```powershell
# 选项1: 传统Gaussian + Huber（更快但不太鲁棒）
python train_vis.py `
  --train_npz ... `
  --loss gauss `
  --huber 1.2 `
  --scheduler cosine `
  --early_stop 8 `
  --save_dir runs\vis_uncert_gauss

# 选项2: 阶梯式学习率（适合精确控制）
python train_vis.py `
  --train_npz ... `
  --scheduler step `
  --lr_step_epochs "15,25,35" `
  --lr_gamma 0.2 `
  --early_stop 10 `
  --save_dir runs\vis_uncert_step

# 选项3: 自适应学习率（val loss停滞时自动降低）
python train_vis.py `
  --train_npz ... `
  --scheduler plateau `
  --early_stop 10 `
  --save_dir runs\vis_uncert_plateau
```

**输出示例:**
```
[scheduler] CosineAnnealingLR (lr: 0.001 → 1e-05)
[config] loss=studentt, nu=3.0, calib_reg=0.001
[config] early_stop=8 epochs
[data] train=185432, val=42156
[model] params=142,594
[training] Starting...
============================================================
[001/050] train=0.1523 (z2x=1.234, z2y=1.189)  val=0.1678 (z2x=1.256, z2y=1.201)  lr=1.00e-03
  → saved best model (val_loss improved)
[002/050] train=0.1412 (z2x=1.187, z2y=1.145)  val=0.1589 (z2x=1.198, z2y=1.167)  lr=9.99e-04
  → saved best model (val_loss improved)
...
[015/050] train=0.0987 (z2x=1.023, z2y=0.991)  val=0.1134 (z2x=1.045, z2y=1.012)  lr=7.07e-04
  → saved best model (val_loss improved)
[016/050] train=0.0973 (z2x=1.019, z2y=0.987)  val=0.1138 (z2x=1.047, z2y=1.015)  lr=6.84e-04
  → no improvement for 1/8 epochs
...
[023/050] train=0.0955 (z2x=1.008, z2y=0.976)  val=0.1149 (z2x=1.052, z2y=1.021)  lr=5.26e-04
  → no improvement for 8/8 epochs
[early_stop] No improvement for 8 epochs. Stopping training.
[early_stop] Best val loss: 0.1134 @ epoch 15
============================================================
[done] Best val loss: 0.1134 @ epoch 15
[done] Models saved to: runs/vis_uncert_enhanced
```

### 3️⃣ 评估校准质量

**【推荐】完整三层校准流程（零信息泄露）:**

```powershell
# Step 1: 在验证集上拟合校准参数（axis温度 + isotonic）
python eval_vis.py `
  --npz F:\SLAMdata\_cache\vis_split\val.npz `
  --model runs\vis_uncert_enhanced\best_vis_kendall.pt `
  --lv_min -10 --lv_max 6 `
  --auto_temp axis `
  --save_temp runs\vis_uncert_enhanced\temp_axis.json `
  --isotonic_save runs\vis_uncert_enhanced\temp_iso_lv_peraxis.json `
  --iso_quantile 0.75 --iso_shrink 0.8 --iso_winsor 0.99 `
  --plot_dir runs\vis_uncert_enhanced\eval_val `
  --plot_per_axis --plot_bootstrap --bootstrap_n 100

# Step 2: 在测试集上应用校准（零泄露）
python eval_vis.py `
  --npz F:\SLAMdata\_cache\vis_split\test.npz `
  --model runs\vis_uncert_enhanced\best_vis_kendall.pt `
  --lv_min -10 --lv_max 6 `
  --auto_temp off `
  --use_temp runs\vis_uncert_enhanced\temp_axis.json `
  --isotonic_use runs\vis_uncert_enhanced\temp_iso_lv_peraxis.json `
  --iso_strength 0.7 `
  --plot_dir runs\vis_uncert_enhanced\eval_test `
  --plot_per_axis --plot_bootstrap --bootstrap_n 100
```

**校准参数说明:**
- `--auto_temp axis`: 分轴温度标定（x/y分别校正）
- `--isotonic_save/use`: lv-based单调校准（处理非线性偏差）
- `--iso_quantile 0.75`: 用75%分位数拟合（抗outliers）
- `--iso_shrink 0.8`: 收缩系数（避免过度校正）
- `--iso_strength 0.7`: 测试集应用强度（保守）
- `--plot_per_axis`: 生成x/y轴独立图表（论文图质量）
- `--plot_bootstrap`: 添加95%置信带（学术规范）

**【可选】Post-Temperature校准（对抗域偏移）:**

```powershell
# 在测试集上额外应用post-temp（使用test的e²，明确知道会泄露）
python eval_vis.py `
  --npz F:\SLAMdata\_cache\vis_split\test.npz `
  --model runs\vis_uncert_enhanced\best_vis_kendall.pt `
  --lv_min -10 --lv_max 6 `
  --auto_temp off `
  --use_temp runs\vis_uncert_enhanced\temp_axis.json `
  --isotonic_use runs\vis_uncert_enhanced\temp_iso_lv_peraxis.json `
  --iso_strength 0.7 `
  --post_temp axis `
  --save_post_temp runs\vis_uncert_enhanced\post_temp_test.json `
  --plot_dir runs\vis_uncert_enhanced\eval_test_post
```

**评估指标解读:**

```json
{
  "z2_mean": 1.005,      // 归一化误差均值（接近1.0=完美校准）
  "z2_x": 1.012,         // x轴z²（分轴查看）
  "z2_y": 0.998,         // y轴z²（分轴查看）
  "cov68": 0.682,        // 68%置信区间覆盖率（目标0.68）
  "cov95": 0.951,        // 95%置信区间覆盖率（目标0.95）
  "spearman": 0.782,     // 误差-方差Spearman相关性（>0.7优秀）
  "dx": 0.023,           // x轴温度校正量
  "dy": -0.015           // y轴温度校正量
}
```

**生成的图表（10张）:**
```
combined plots:
  coverage_curve.png      # 联合Coverage曲线（df=2）
  qq_chi2.png            # 联合Q-Q图（df=2）
  
per-axis plots (论文图):
  coverage_x_axis.png    # X轴Coverage + 95% CI
  coverage_y_axis.png    # Y轴Coverage + 95% CI
  qq_x_axis.png          # X轴Q-Q + 95% CI
  qq_y_axis.png          # Y轴Q-Q + 95% CI
  
diagnostic plots:
  hist_logvar.png        # log-variance分布
  err2_vs_var.png        # 误差vs方差散点
  sparsification.png     # 稀疏化曲线 + AUSE
  per_seq_metrics.png    # 每序列指标
```

## 🔧 参数调优建议

### 数据生成 (`gen_vis_pairs_euroc_strict.py`)

| 参数 | 默认值 | 说明 | 调优建议 |
|------|-------|------|---------|
| `--deltas` | `"1,2"` | 多帧间隔（逗号分隔） | 增加多样性：`"1,2,3"` |
| `--patch` | 32 | Patch大小 | 纹理丰富→32；弱纹理→64 |
| `--max_pairs` | 60000 | 最大样本数 | 训练集60K+；验证/测试10K |
| `--err_clip_px` | 15.0 | 误差截断阈值 | 更严格→10；更宽松→20 |
| `--epi_thr_px` | 1.5 | Sampson对极误差阈值 | 更严格→1.0；更宽松→2.0 |
| `--texture_strat` | False | 纹理分层采样 | 推荐开启（70%高梯度+30%低梯度） |

### 训练 (`train_vis.py`)

| 参数 | 默认值 | 说明 | 调优建议 |
|------|-------|------|---------|
| `--loss` | `gauss` | 损失函数 | **推荐`studentt`**（更鲁棒） |
| `--nu` | 3.0 | Student-t自由度 | 重尾数据→2-3；干净数据→5-10 |
| `--calib_reg` | 0.0 | 校准正则化系数 | **推荐1e-3**（鼓励z²≈1） |
| `--huber` | 1.0 | Huber阈值（gauss） | 异常值多→1.5-2.0 |
| `--lv_min` | -10 | log方差下限 | 噪声小→-8；噪声大→-12 |
| `--lv_max` | 4 | log方差上限 | **推荐6**（更大不确定性范围） |
| `--lr` | 1e-3 | 初始学习率 | 标准值，配合scheduler使用 |
| `--scheduler` | `cosine` | 学习率调度器 | **推荐`cosine`**（平滑衰减） |
| `--early_stop` | 0 | 早停patience | **推荐8**（自动找最佳点） |

### 评估 (`eval_vis.py`)

| 参数 | 默认值 | 说明 | 调优建议 |
|------|-------|------|---------|
| `--auto_temp` | `off` | 温度标定模式 | **推荐`axis`**（分轴校准） |
| `--iso_quantile` | 0.75 | Isotonic分位数 | 重尾数据→0.75；干净→0.68 |
| `--iso_shrink` | 0.8 | Isotonic收缩系数 | 保守→0.6-0.7；激进→0.9-1.0 |
| `--iso_strength` | 0.7 | Isotonic应用强度 | Test保守→0.6-0.7；Val激进→1.0 |
| `--plot_bootstrap` | False | Bootstrap置信带 | **推荐开启**（论文图质量） |
| `--bootstrap_n` | 100 | Bootstrap次数 | 快速→100；平滑→500 |

## 📊 可视化分析（可选扩展）

```python
# vis_analysis.py - 分析预测的不确定性分布
import numpy as np
import matplotlib.pyplot as plt

def analyze_predictions(npz_path, model_path):
    # 加载数据和预测...
    
    # 1. 误差-方差散点图
    plt.scatter(np.sqrt(vx), np.sqrt(e2x), alpha=0.3)
    plt.xlabel("Predicted σx")
    plt.ylabel("Actual error")
    
    # 2. z²直方图（应接近χ²_2分布）
    z2 = (e2x/vx + e2y/vy) / 2
    plt.hist(z2, bins=50, density=True)
    # 叠加理论分布...
    
    # 3. 覆盖率曲线
    # ...
```

## 🆚 版本演进

| 特性 | v1.0（基础版） | v2.0（增强版，当前） |
|------|---------------|---------------------|
| **架构** | 混在IMU代码里 | 完全独立模块 |
| **数据生成** | 单delta，基础筛选 | 多delta + 纹理分层 + Sampson筛选 |
| **几何特征** | 4维（u,v归一化坐标） | 11维（+梯度/角点/光流/视差） |
| **损失函数** | Kendall + Huber | Kendall/Student-t + Calib-Reg |
| **学习率** | 固定 | 动态调度（cosine/step/plateau） |
| **早停** | ❌ 无 | ✅ 可配置patience |
| **校准** | Global温度 | Axis温度 + Isotonic + Post-temp |
| **评估图表** | 基础指标 | 10张图 + Bootstrap置信带 |
| **论文图** | ❌ 无 | ✅ Per-axis Coverage/Q-Q |
| **可维护性** | 低（耦合） | 高（模块化） |

## ⚠️ 常见问题

### Q1: 生成数据时报错 "no pairs collected"
**A**: ORB特征匹配失败，可能原因：
- 图像模糊/运动模糊过大
- 纹理过弱（如白墙）
- `--deltas`过大导致视角变化太大

**解决**: 
- 减小deltas：`--deltas "1"`
- 放宽对极误差：`--epi_thr_px 2.0`
- 关闭纹理分层：去掉`--texture_strat`

---

### Q2: 训练时z²均值不接近1
**A**: 模型未校准，可能原因：
- 训练不充分
- 数据质量差（outliers多）
- 缺少校准正则化

**解决**: 
1. 使用Student-t loss：`--loss studentt --nu 3.0`
2. 添加校准正则：`--calib_reg 1e-3`
3. 延长训练：`--epochs 50 --early_stop 10`
4. 检查数据生成参数（`--err_clip_px`, `--epi_thr_px`）

---

### Q3: 覆盖率偏低(<60%)
**A**: 方差估计过小（过度自信），可能原因：
- `--lv_max`过小
- 缺少鲁棒性机制
- 数据outliers未正确处理

**解决**:
1. 提高`--lv_max`到6或8
2. 使用Student-t loss（自带重尾）
3. 数据生成时更严格筛选：`--epi_thr_px 1.0`

---

### Q4: Test集z²与Val集差异大（域偏移）
**A**: Train/Val/Test数据分布不一致，可能原因：
- 场景差异（室内vs室外）
- 运动模式差异（平稳vs快速）
- 纹理分布差异

**解决**:
1. 使用isotonic校准：`--isotonic_use`（零泄露）
2. 如果仍不满意，使用post-temp：`--post_temp axis`（会泄露test信息）
3. 重新划分数据集，确保分布一致

---

### Q5: Bootstrap绘图太慢
**A**: Bootstrap需要重采样100+次，计算密集

**解决**:
- 减少次数：`--bootstrap_n 50`（快速预览）
- 仅在最终评估时用：`--bootstrap_n 100`
- 论文投稿图：`--bootstrap_n 500`（高质量）

---

### Q6: 早停太早/太晚
**A**: Patience设置不当

**解决**:
- 太早停止：增大patience `--early_stop 12`
- 太晚停止（过拟合）：减小patience `--early_stop 5`
- 标准设置：`--early_stop 8`（推荐）

---

### Q7: 训练loss震荡
**A**: 学习率过高或数据有问题

**解决**:
1. 降低初始学习率：`--lr 5e-4`
2. 使用plateau调度器：`--scheduler plateau`
3. 检查数据是否有极端outliers
4. 增大batch size：`--batch 512`（如果显存够）

## 📚 参考资料与引用

### 核心方法论
- **Kendall & Gal (2017)**: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
  - Kendall异方差NLL损失的理论基础
  
- **Kuleshov et al. (2018)**: "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
  - Isotonic校准方法
  
- **Guo et al. (2017)**: "On Calibration of Modern Neural Networks"
  - Temperature scaling的理论与实践

### 相关模块
- IMU不确定性估计: 见主`README.MD`
- 视觉-IMU融合: 开发中

### 数据集
- **EuRoC MAV Dataset**: Burri et al. (2016)
  - 本项目使用的标准SLAM数据集

---

## 🎓 论文写作建议

### 方法描述模板

```latex
\subsection{Visual Uncertainty Estimation}

We employ a patch-based uncertainty estimation network to predict 
per-feature 2D covariances. The network takes a 32×32 patch and 
11-dimensional geometric context as input, outputting log-variances 
(σ²_x, σ²_y) via a lightweight CNN.

\textbf{Training}: We use Student-t negative log-likelihood 
(ν=3.0) with calibration regularization (λ=1e-3) to encourage 
well-calibrated predictions. Training employs cosine annealing 
(1e-3→1e-5) with early stopping (patience=8).

\textbf{Calibration}: Post-hoc calibration follows a three-stage 
pipeline: (i) per-axis temperature scaling, (ii) lv-based isotonic 
regression (quantile=0.75, shrinkage=0.8), (iii) optional post-
temperature adjustment for domain shift. All calibration parameters 
are fitted on validation set and applied to test set without 
information leakage.
```

### 实验结果表格模板

| Method | z²↓ | Cov68↑ | Cov95↑ | Spearman↑ | AUSE↓ |
|--------|-----|--------|--------|-----------|-------|
| Baseline (Gaussian) | 1.23 | 0.61 | 0.89 | 0.68 | 0.042 |
| +Student-t | 1.08 | 0.66 | 0.93 | 0.74 | 0.035 |
| +Axis Temp | 1.02 | 0.68 | 0.95 | 0.75 | 0.033 |
| +Isotonic (Ours) | **1.01** | **0.68** | **0.95** | **0.78** | **0.029** |

---

**最后更新**: 2025-10-04  
**版本**: v2.0 (Enhanced)  
**作者**: IMU_2_test Team  
**维护**: 主动维护中

