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
  --kappa 1.0 `
  --sa_recenter `
  --scan_q_threshold `
  --plots_dir runs/oof_macro_v1/val_plots

# 测试集
python eval_macro.py `
  --npz F:/SLAMdata/_cache/macro/test_frame.npz `
  --ckpt runs/oof_macro_v1/fold0/best_macro_sa.pt `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --calibrator_json runs/oof_macro_v1/calibrator_oof.json `
  --kappa 1.0 `
  --sa_recenter `
  --scan_q_threshold `
  --plots_dir runs/oof_macro_v1/test_plots
```

**注意**：`--sa_recenter` 会自动将各向异性均值校正到 OOF 训练时的目标均值，解决 κ 缩放带来的均值偏移问题。

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

- **`deming`（推荐）**：Deming 回归，考虑 x 和 y 都有误差，修正"斜率<1"的偏置
- **`robust`**：使用 Winsorize + 稳健标准差（IQR），对异常值不敏感
- **`by_q`**：按质量分段校准，高质量样本用更激进的参数
- **`std`**：标准方差比方法（基线）

### 参数说明

**训练端（tools/train_oof.py）：**
- `--sa_mode` (std/robust/deming/by_q, 默认deming)：s/a 校准方式
- `--winsor_p` (默认2.5)：Winsorize 百分位（robust/by_q 模式用）
- `--deming_lambda` (默认1.0)：Deming 回归的误差比 λ
- `--byq_bins` (默认"0.00,0.55,1.00")：by_q 分段边界

**评估端（eval_macro.py）：**
- `--kappa` (0.6~1.0, 默认1.0)：各向异性强度缩放系数
- `--sa_recenter`：应用 κ 后对 a 再中心到 OOF 均值（修正均值偏移）

### 诊断输出

评测时会输出以下诊断指标，帮助判断校准质量：
- **OLS slope**: 各向异性主轴斜率（目标接近 1.0）
- **Robust std ratio**: 稳健标准差比（目标接近 1.0）
- **Quantile diff**: 分位数差异（5%, 50%, 95%）

评测时会生成 `anisotropy_prepost.png` 对比图，可视化 s/a 校准前后的各向异性分布变化。

## 推荐工作流

1. **先用 Deming 模式**（默认）：修正"斜率<1"的回归偏置
2. **观察诊断指标**：OLS slope、Robust std ratio、分位数差异
3. **如果尾部仍窄**：改用 `--sa_mode by_q` 进行分段校准
4. **始终启用 `--sa_recenter`**：修正均值偏移

## 核心优势

- ✅ **零数据泄露**: 校准参数完全来自训练集 OOF 预测
- ✅ **更稳健**: 使用训练分布，不易被 val 带偏
- ✅ **样本量大**: 使用全训练集，而非小的 val 集
- ✅ **保序校准**: s/a 映射单调，不破坏 Spearman 排序
- ✅ **Deming 回归**: 考虑双向误差，修正斜率偏置
- ✅ **均值校正**: 自动修正 κ 缩放带来的均值偏移

## 详细文档



主要改动
1. 
tools/train_oof.py

✅ 新增 
deming_fit()
 函数（Deming 回归）
✅ 扩展 --sa_mode 支持 4 种模式：std, robust, deming, by_q
✅ 新增 --byq_bins 和 --deming_lambda 参数
✅ OOF 聚合时收集 q_oof_pred（用于 by_q 分段）
✅ JSON 中存储 a_mean_pred 和 a_mean_gt（用于 recenter）
2. 
eval_macro.py

✅ 新增 --sa_recenter 参数（解决均值偏移）
✅ 支持 by_q 分段应用
✅ 导出时包含 pred_q（改名为 pred_q）
3. 
run_oof_workflow.ps1

✅ 默认使用 --sa_mode deming（推荐从这个开始）
✅ 评估时添加 --sa_recenter 参数
使用方式
方式1：使用 Deming 回归（推荐先试这个）

powershell
conda activate LAP3GPU
.\run_oof_workflow.ps1

方式2：使用 by_q 分段（如果 Deming 效果不够） 修改 
run_oof_workflow.ps1
 第 45 行：

powershell
--sa_mode by_q --byq_bins 0.00,0.55,1.00