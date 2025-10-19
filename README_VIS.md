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

**批量生成多个序列的patch pairs:**

```powershell
# 训练集 (MH sequences)
python tools/gen_vis_pairs_euroc.py `
  --euroc_root F:\SLAMdata `
  --seq MH_01_easy `
  --camchain F:\SLAMdata\camchain-imucam.yaml `
  --delta 1 `
  --patch 32 `
  --max_pairs 20000 `
  --out data_vis\train\MH_01_easy.npz

# 重复其他序列...

# 验证集
python tools/gen_vis_pairs_euroc.py `
  --euroc_root F:\SLAMdata `
  --seq V1_01_easy `
  --camchain F:\SLAMdata\camchain-imucam.yaml `
  --delta 1 `
  --patch 32 `
  --max_pairs 5000 `
  --out data_vis\val\V1_01_easy.npz

# 测试集
python tools/gen_vis_pairs_euroc.py `
  --euroc_root F:\SLAMdata `
  --seq V2_01_easy `
  --camchain F:\SLAMdata\camchain-imucam.yaml `
  --delta 1 `
  --patch 32 `
  --max_pairs 5000 `
  --out data_vis\test\V2_01_easy.npz
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

```powershell
python train_vis.py `
  --train_npz data_vis\train_all.npz `
  --val_npz data_vis\val_all.npz `
  --epochs 30 `
  --batch 256 `
  --lr 1e-3 `
  --huber 1.2 `
  --lv_min -10 `
  --lv_max 4 `
  --save_dir runs\vis_uncert_exp1
```

**输出示例:**
```
[data] train=18543, val=4521
[model] params=142,594
[training] Starting...
[001/030] train=2.3451 (z2x=1.234, z2y=1.156)  val=2.1234 (z2x=1.123, z2y=1.089)
  → saved best model
[002/030] train=2.1023 (z2x=1.098, z2y=1.045)  val=1.9876 (z2x=1.056, z2y=1.012)
  → saved best model
...
```

### 3️⃣ 评估校准质量

```powershell
# 不带温度校正
python eval_vis.py `
  --npz data_vis\test_all.npz `
  --model runs\vis_uncert_exp1\best_vis_kendall.pt `
  --lv_min -10 `
  --lv_max 4

# 带全局温度校正（推荐）
python eval_vis.py `
  --npz data_vis\test_all.npz `
  --model runs\vis_uncert_exp1\best_vis_kendall.pt `
  --auto_temp global `
  --lv_min -10 `
  --lv_max 4
```

**评估指标解读:**

```json
{
  "z2_mean": 1.023,      // 归一化误差均值（接近1=校准良好）
  "cov68": 0.682,        // 68%置信区间覆盖率（目标0.68）
  "cov95": 0.948,        // 95%置信区间覆盖率（目标0.95）
  "spearman": 0.456,     // 误差-方差相关性（>0=有效）
  "delta_logvar": 0.023  // 温度校正量
}
```

## 🔧 参数调优建议

### 数据生成 (`gen_vis_pairs_euroc.py`)

| 参数 | 默认值 | 说明 | 调优建议 |
|------|-------|------|---------|
| `--delta` | 1 | 时间间隔（帧） | 运动快→增大(2-3)；静止→保持1 |
| `--patch` | 32 | Patch大小 | 纹理丰富→32；弱纹理→64 |
| `--max_pairs` | 20000 | 最大样本数 | 训练集20K+；验证/测试5K |

### 训练 (`train_vis.py`)

| 参数 | 默认值 | 说明 | 调优建议 |
|------|-------|------|---------|
| `--huber` | 1.0 | Huber阈值 | 异常值多→增大(1.5-2.0) |
| `--lv_min` | -10 | log方差下限 | 噪声小→提高(-8) |
| `--lv_max` | 4 | log方差上限 | 噪声大→提高(6) |
| `--lr` | 1e-3 | 学习率 | 不收敛→降低；震荡→降低 |

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

## 🆚 与旧VIS代码的对比

| 特性 | 旧实现（已删除） | 新实现（vis/） |
|------|---------------|--------------|
| **架构** | 混在IMU代码里 | 完全独立模块 |
| **数据** | 窗口级聚合 | 逐patch精细 |
| **损失** | 复杂混合 | 简洁Kendall-NLL |
| **输出** | 2D对角 | 2D独立轴 |
| **可维护性** | 低（耦合） | 高（解耦） |

## ⚠️ 常见问题

### Q1: 生成数据时报错 "no pairs collected"
**A**: ORB特征匹配失败，可能原因：
- 图像模糊/运动模糊过大
- 纹理过弱（如白墙）
- `--delta`过大导致视角变化太大

**解决**: 减小`--delta`，或检查图像质量

### Q2: z²均值远离1
**A**: 模型未校准，可能原因：
- 训练不充分
- 过拟合
- 数据分布偏差

**解决**: 
1. 延长训练（更多epochs）
2. 使用`--auto_temp global`校正
3. 检查训练/测试数据一致性

### Q3: 覆盖率偏低(<60%)
**A**: 方差估计过小（过度自信），可能原因：
- `--lv_max`过小
- 缺少鲁棒性机制

**解决**:
1. 提高`--lv_max`到6
2. 增大`--huber`阈值

## 📚 参考资料

- Kendall & Gal (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
- IMU不确定性模块: 见主`README.MD`

---

**最后更新**: 2025-10-02  
**作者**: IMU_2_test Team

