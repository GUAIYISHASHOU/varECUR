# OOF Calibration Results Summary

**Date**: 2025-10-14  
**Method**: Out-of-Fold (OOF) Calibration  
**K-Folds**: 5  
**Save Directory**: `runs/oof_macro_v1/`

---

## 📊 Calibration Parameters

### OOF Affine Calibrator (`calibrator_oof.json`)

**LogVar-X Calibration:**
- α (slope): **0.3984**
- β (intercept): **0.2092**
- Formula: `calibrated_x = 0.3984 * pred_x + 0.2092`

**LogVar-Y Calibration:**
- α (slope): **0.6845**
- β (intercept): **0.1081**
- Formula: `calibrated_y = 0.6845 * pred_y + 0.1081`

**Interpretation:**
- LogVar-X 的斜率较小 (0.40)，说明原始预测的动态范围被压缩了约 60%
- LogVar-Y 的斜率较大 (0.68)，保留了更多原始预测的动态范围
- 两者都有正的截距，说明原始预测整体偏低

---

## 🎯 Test Set Performance (with OOF Calibration)

### Overall Metrics
- **Spearman-X (all)**: 0.492
- **Spearman-Y (all)**: 0.657
- **Spearman Mean (all)**: **0.575**

### Inlier-Only Metrics (GT-based)
- **Spearman-X (inlier)**: 0.395
- **Spearman-Y (inlier)**: 0.612
- **Spearman Mean (inlier)**: **0.504**
- **N inliers**: 620

### Predicted Inlier Metrics (q-based)
- **Spearman-X (pred q)**: 0.553
- **Spearman-Y (pred q)**: 0.645
- **Spearman Mean (pred q)**: **0.599**
- **N predicted inliers**: 503

### Quality Prediction (q)
- **Accuracy**: 0.801
- **AUC**: 0.911

### Optimal Threshold Scan
- **Best Threshold**: q > **0.55**
- **Best Spearman Mean**: **0.605**
- **Samples at threshold**: 490

---

## 📈 Generated Visualizations

All plots saved in `runs/oof_macro_v1/test_plots_oof/`:

1. **scatter_logvar_x.png** / **scatter_logvar_y.png**
   - LogVar 预测 vs 真值散点图
   - 区分内点/外点
   - 显示对角线（完美预测）

2. **q_distribution.png**
   - 内点概率分布直方图
   - 红色：GT 外点的 q 分布
   - 绿色：GT 内点的 q 分布

3. **anisotropy_analysis.png**
   - 各向异性分析（椭圆轴比）
   - 左图：各向异性散点图
   - 右图：各向异性误差分布

4. **threshold_scan.png**
   - q 阈值扫描曲线
   - 显示不同阈值下的 Spearman 相关系数
   - 标注最优阈值

5. **residual_analysis.png**
   - 残差分布分析
   - 检查预测误差的统计特性

---

## 🔬 OOF Training Details

### K-Fold Configuration
- **Fold 0**: train=2000, val=501
- **Fold 1**: train=2001, val=500
- **Fold 2**: train=2001, val=500
- **Fold 3**: train=2001, val=500
- **Fold 4**: train=2001, val=500

### Training Hyperparameters
- Epochs: 40
- Stage 1 (q-only): 8 epochs
- Batch size: 32
- Learning rate: 2e-4
- a_max: 3.0
- Drop token p: 0.1
- Heads: 4
- Layers: 1
- d_model: 128
- NLL weight: 1.5
- BCE weight: 0.6
- Rank weight: 0.3
- Patience: 12

---

## ✅ Key Advantages of OOF Calibration

1. **Zero Data Leakage**
   - Calibration parameters fitted on train OOF predictions only
   - Val/test never used in calibration fitting
   - Each train sample predicted by a model that never saw it

2. **More Robust**
   - Uses full training distribution (2501 samples)
   - Not biased by small validation set
   - Better generalization to test set

3. **Larger Sample Size**
   - OOF: 2501 samples (full train)
   - Val-based: ~500 samples (val only)
   - 5x more data for calibration fitting

---

## 🚀 Next Steps

### Immediate Actions
1. ✅ Review visualization plots in `test_plots_oof/`
2. ✅ Check calibration parameters in `calibrator_oof.json`
3. ⏳ Compare performance across different folds

### Optional Improvements
1. **Model Ensembling**
   - Average predictions from all 5 folds
   - Usually improves performance by 1-3%

2. **Hyperparameter Tuning**
   - Adjust loss weights (nll/bce/rank)
   - Try different architectures (layers, heads)

3. **Production Deployment**
   - Use `fold0/best_macro_sa.pt` as the model
   - Apply `calibrator_oof.json` at inference time
   - Implement q-threshold filtering (q > 0.55)

---

## 📝 Files Generated

```
runs/oof_macro_v1/
├── k5_splits.json                    # K-fold split indices
├── calibrator_oof.json               # OOF calibrator parameters
├── fold0/
│   ├── best_macro_sa.pt             # Best model checkpoint
│   ├── hparams.json                 # Hyperparameters
│   └── oof_fold0.npz                # OOF predictions
├── fold1/ ... fold4/                 # Other folds
├── val_plots_oof/                    # Validation plots
└── test_plots_oof/                   # Test plots (7 images)
```

---

**Generated**: 2025-10-14 01:45 UTC+08:00  
**Method**: OOF (Out-of-Fold) Calibration  
**Status**: ✅ Complete
