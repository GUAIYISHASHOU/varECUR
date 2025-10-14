# OOF 校准方案

## 快速开始

```powershell
# 一键运行 OOF 工作流
.\run_oof_workflow.ps1
```

## 手动运行

### 1. 生成 K 折切分

```powershell
python tools/make_kfold_splits.py `
  --train_npz F:/SLAMdata/_cache/macro/train_frame.npz `
  --out_json runs/oof/k5_splits.json `
  --k 5
```

### 2. OOF 训练并拟合校准器

```powershell
python tools/train_oof.py `
  --train_npz F:/SLAMdata/_cache/macro/train_frame.npz `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --kfold_json runs/oof/k5_splits.json `
  --save_root runs/oof_macro_v1
```

### 3. 在 val/test 上评测（应用校准）

```powershell
# 验证集
python eval_macro.py `
  --npz F:/SLAMdata/_cache/macro/val_frame.npz `
  --ckpt runs/oof_macro_v1/fold0/best_macro_sa.pt `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --calibrator_json runs/oof_macro_v1/calibrator_oof.json `
  --kappa 0.8 `
  --scan_q_threshold `
  --plots_dir runs/oof_macro_v1/val_plots

# 测试集
python eval_macro.py `
  --npz F:/SLAMdata/_cache/macro/test_frame.npz `
  --ckpt runs/oof_macro_v1/fold0/best_macro_sa.pt `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --calibrator_json runs/oof_macro_v1/calibrator_oof.json `
  --kappa 0.8 `
  --scan_q_threshold `
  --plots_dir runs/oof_macro_v1/test_plots
```

## s/a 域"幅度收缩"修正

OOF 的 x/y 轴向仿射校准后，仍可能出现各向异性幅度收缩（a 的方差偏小、散点主轴斜率<1）。
从 `calibrator_oof.json` 读取的 `sa_calib` 将在评测阶段自动应用：

```json
{
  "x": {...}, "y": {...},
  "sa_calib": {
    "version": 2,
    "mode": "robust",
    "alpha_s": <cov/var slope>, "beta_s": <offset>,
    "alpha_a": <std ratio>,     "beta_a": <offset>,
    "winsor_p": 2.5
  }
}
```

### 校准模式

- **`robust`（默认推荐）**：使用 Winsorize + 稳健标准差（IQR），对异常值不敏感
- **`std`**：标准方差比方法（旧版本）

### 参数说明

- `--kappa` (0.6~1.0, 默认1.0)：各向异性强度缩放系数
- `--sa_mode` (robust/std, 默认robust)：s/a 校准方式
- `--winsor_p` (默认2.5)：Winsorize 百分位（robust 模式用）

### 诊断输出

评测时会输出以下诊断指标，帮助判断校准质量：
- **OLS slope**: 各向异性主轴斜率（目标接近 1.0）
- **Robust std ratio**: 稳健标准差比（目标接近 1.0）
- **Quantile diff**: 分位数差异（5%, 50%, 95%）

评测时会生成 `anisotropy_prepost.png` 对比图，可视化 s/a 校准前后的各向异性分布变化。

## 核心优势

- ✅ **零数据泄露**: 校准参数完全来自训练集 OOF 预测
- ✅ **更稳健**: 使用训练分布，不易被 val 带偏
- ✅ **样本量大**: 使用全训练集，而非小的 val 集
- ✅ **保序校准**: s/a 映射单调，不破坏 Spearman 排序

## 详细文档

参考 `OOF_CALIBRATION_GUIDE.md` 获取完整说明。
