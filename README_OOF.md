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
  --scan_q_threshold `
  --plots_dir runs/oof_macro_v1/val_plots

# 测试集
python eval_macro.py `
  --npz F:/SLAMdata/_cache/macro/test_frame.npz `
  --ckpt runs/oof_macro_v1/fold0/best_macro_sa.pt `
  --geom_stats_npz F:/SLAMdata/_cache/macro/geom_stats_24d.npz `
  --calibrator_json runs/oof_macro_v1/calibrator_oof.json `
  --scan_q_threshold `
  --plots_dir runs/oof_macro_v1/test_plots
```

## 核心优势

- ✅ **零数据泄露**: 校准参数完全来自训练集 OOF 预测
- ✅ **更稳健**: 使用训练分布，不易被 val 带偏
- ✅ **样本量大**: 使用全训练集，而非小的 val 集

## 详细文档

参考 `OOF_CALIBRATION_GUIDE.md` 获取完整说明。
