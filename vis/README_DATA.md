# VIS数据生成指南

## 📋 概述

本模块提供**两种**VIS训练数据生成方式：

| 版本 | 文件 | 几何精度 | 适用场景 |
|------|------|---------|---------|
| **基础版** | `gen_vis_pairs_euroc.py` | 近似（小运动假设） | 快速原型 |
| **严格版** | `gen_vis_pairs_euroc_strict.py` | 精确GT位姿 | 最终训练 |

**推荐使用严格版**以获得最佳性能。

---

## 🎯 严格版数据生成（推荐）

### 特点
- ✅ 使用真实GT位姿（world←IMU）
- ✅ 精确的相机-IMU外参
- ✅ SLERP四元数插值
- ✅ 严格几何监督

### 数据集划分

```python
Train (8序列): 
  V1_01_easy, V1_02_medium, V2_01_easy, V2_02_medium,
  MH_01_easy, MH_02_easy, MH_03_medium, V1_03_difficult

Val (1序列): 
  MH_04_difficult

Test (2序列): 
  V2_03_difficult, MH_05_difficult
```

### 快速开始

#### 方式1: 一键批量生成（推荐）

**Windows (PowerShell):**
```powershell
# 生成所有序列
.\scripts\generate_vis_data_all.ps1

# 合并为train/val/test
python tools\merge_vis_pairs_by_seq.py `
  --pairs_root F:\SLAMdata\_cache\vis_pairs `
  --out_root F:\SLAMdata\_cache\vis_split `
  --cap_per_seq 60000
```

**Linux/Mac (Bash):**
```bash
# 生成所有序列
bash scripts/generate_vis_data_all.sh

# 合并
python tools/merge_vis_pairs_by_seq.py \
  --pairs_root /path/to/vis_pairs \
  --out_root /path/to/vis_split \
  --cap_per_seq 60000
```

#### 方式2: 单序列生成

```powershell
# 生成单个序列
python tools\gen_vis_pairs_euroc_strict.py `
  --euroc_root F:\SLAMdata `
  --seq MH_01_easy `
  --delta 1 `
  --patch 32 `
  --max_pairs 60000 `
  --out_npz F:\SLAMdata\_cache\vis_pairs\MH_01_easy.npz
```

### 参数说明

| 参数 | 默认值 | 说明 | 调优建议 |
|------|-------|------|---------|
| `--delta` | 1 | 时间间隔（帧） | 运动慢→1；运动快→2~3 |
| `--patch` | 32 | Patch尺寸 | 纹理丰富→32；弱纹理→64 |
| `--max_pairs` | 60000 | 最大样本数 | 训练集60K；验证/测试各30K |

### 合并参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--cap_per_seq` | 60000 | 每序列最大样本（均衡） |
| `--seed` | 0 | 随机采样种子 |

---

## 📊 预期输出

### 单序列NPZ

```
F:/SLAMdata/_cache/vis_pairs/
├── V1_01_easy.npz        (~25MB, 15K-60K pairs)
├── V1_02_medium.npz
├── ...
└── MH_05_difficult.npz
```

**NPZ内容：**
- `I0`: [N,1,H,W] 第一帧patches
- `I1`: [N,1,H,W] 第二帧patches
- `geom`: [N,4] 归一化坐标 [u0,v0,u1,v1]
- `e2x`, `e2y`: [N] 每轴误差平方
- `mask`: [N] 有效性掩码
- `meta`: dict(seq, patch, delta)

### 合并后数据集

```
F:/SLAMdata/_cache/vis_split/
├── train.npz    (~200MB, ≤480K pairs = 8×60K)
├── val.npz      (~25MB,  ≤60K pairs)
└── test.npz     (~50MB,  ≤120K pairs = 2×60K)
```

---

## 🔧 故障排查

### Q1: "No pairs collected"
**原因：** ORB特征匹配失败
**解决：**
```bash
# 方案1: 减小delta（降低运动幅度）
--delta 1

# 方案2: 增加特征数量（修改代码）
# 在 orb_det_desc() 中: n=1200 → n=2000

# 方案3: 检查图像质量
# 某些序列可能有模糊/曝光问题
```

### Q2: CSV列名错误 "KeyError"
**原因：** EuRoC不同版本列名有差异
**解决：** 脚本已支持多种变体，如仍报错请提供CSV前几行

### Q3: 样本数不均衡
**原因：** 不同序列运动模式差异大
**解决：**
```bash
# 调整cap_per_seq参数
--cap_per_seq 40000  # 降低上限提高均衡性

# 或在训练时使用WeightedRandomSampler
```

### Q4: 内存不足
**原因：** 一次性加载过多patches
**解决：**
```bash
# 降低max_pairs
--max_pairs 30000

# 或分批生成后合并
```

---

## 🎨 可视化检查（可选）

```python
# vis_check_npz.py
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
D = np.load("F:/SLAMdata/_cache/vis_pairs/MH_01_easy.npz")

# 查看一个patch pair
i = 100
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(D["I0"][i, 0], cmap='gray')
axes[0].set_title(f"Frame t")
axes[1].imshow(D["I1"][i, 0], cmap='gray')
axes[1].set_title(f"Frame t+Δ")

# 显示误差
print(f"Error: ex²={D['e2x'][i]:.2f}, ey²={D['e2y'][i]:.2f}")
plt.show()

# 误差分布
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.hist(D["e2x"], bins=50, alpha=0.7)
plt.xlabel("e_x²"); plt.ylabel("Count")
plt.subplot(132)
plt.hist(D["e2y"], bins=50, alpha=0.7)
plt.xlabel("e_y²")
plt.subplot(133)
plt.hist(D["e2x"] + D["e2y"], bins=50, alpha=0.7)
plt.xlabel("e² total")
plt.tight_layout()
plt.show()
```

---

## 📈 数据质量指标

### 良好数据特征
- ✅ e² 分布：大部分在 0~10 px²
- ✅ 匹配数：每帧 >40 matches
- ✅ 有效样本率：>70%
- ✅ Patch清晰度：无严重模糊

### 问题数据特征
- ❌ e² 异常大（>100 px²）→ GT位姿错误
- ❌ 匹配数过少（<20）→ 纹理太弱
- ❌ 大量NaN → 三角化失败

---

## 🔄 与基础版对比

| 特性 | 基础版 | 严格版 |
|------|--------|--------|
| **位姿来源** | 恒等近似 | GT插值 |
| **精度** | ~5px误差 | <1px误差 |
| **速度** | 快（无需插值） | 稍慢（+GT处理） |
| **依赖** | 无 | 需要GT CSV |
| **适用** | 原型测试 | 正式训练 |

---

## 📚 相关文档

- 训练指南: `vis/README.md`
- 完整工作流: `README_VIS.md`
- EuRoC数据集: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

---

**最后更新**: 2025-10-02  
**维护者**: IMU_2_test Team

