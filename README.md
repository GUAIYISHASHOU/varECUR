# IMU方差建模与OOF训练（Student-t/SA3）

本项目用于对 IMU（ACC/GYR）误差方差进行建模与评估，支持 Student-t 对角分布与三轴“尺度+各向异性”参数化（SA3）。包含 OOF（Out-of-Fold）交叉验证训练、OOF 预测聚合与校准，以及单模型评估与可视化。

---

## 快速开始

- **环境**
  - 建议使用 conda 环境：`LAP3GPU`（Windows）
  - Python 依赖（训练/评估）：`torch`、`numpy`、`tqdm`、`pyyaml`、`matplotlib`
  - 数据预处理/工具脚本可能还需：`pandas`、`opencv-python`

- **配置文件**
  - 单路由 OOF 训练：`config_oof.yaml`
  - 陀螺仪专用 OOF 训练：`config_gyr_oof.yaml`

- **数据（示例）**
  - OOF 序列目录示例：`F:/SLAMdata/_cache/imu_euroc_MH4/`
  - 分折文件：`F:/SLAMdata/_cache/imu_euroc_MH4/splits_oof.json`

---

## 数据准备（EuRoC → NPZ）

- 推荐使用 `tools/gen_euroc_step_npz.py`（新版，带回退机制）或 `tools/legacy/gen_euroc_step_npz_physics.py`（增强物理版）
- 示例（生成窗口 T=512、stride=256 的步级标签）：
```powershell
# 生成多个序列的 IMU NPZ（示意）
python tools/gen_euroc_step_npz.py --euroc_root F:/SLAMdata/EuRoC --seqs MH_01_easy,MH_02_easy --out F:/SLAMdata/_cache/imu_euroc_MH4 --T 512 --stride 256
```
- 如需生成 OOF 分折：
```powershell
# 1) 扫描序列目录生成 manifest.json
python tools/generate_manifest.py --seq_dir F:/SLAMdata/_cache/imu_euroc_MH4 --out F:/SLAMdata/_cache/imu_euroc_MH4/manifest.json

# 2) 按序列分层构建 splits_oof.json（k 折）
python tools/build_group_stratified_folds.py --manifest F:/SLAMdata/_cache/imu_euroc_MH4/manifest.json --out F:/SLAMdata/_cache/imu_euroc_MH4/splits_oof.json --k 4 --route gyr
```

---

## OOF 训练

- 入口：`train_oof.py`
- 示例（GYR 路由）：
```powershell
python train_oof.py --route gyr `
  --config config_gyr_oof.yaml `
  --splits F:/SLAMdata/_cache/imu_euroc_MH4/splits_oof.json `
  --seq_dir F:/SLAMdata/_cache/imu_euroc_MH4
```
- 产物：`runs/oof_gyr/`（或 `runs/oof_acc/`）目录下的 `best_fold*.pt`、`oof_summary.json`、`curve_fold*.json`、`scaler_fold*.json`

> 说明：配置中的 `train.dist=studentt` 且 `variance_param=sa3` 时，将使用三轴 Student-t NLL 与 SA3 参数化。`nu>2` 才有有限方差。

---

## OOF 后处理（预测/聚合/校准）

1) 生成各折验证集预测并展平保存：
```powershell
python oof/predict_oof.py --runs_dir runs/oof_gyr --out_dir runs/oof_gyr/oof --splits F:/SLAMdata/_cache/imu_euroc_MH4/splits_oof.json --route gyr --device cuda
```
2) 聚合所有折并计算整体指标：
```powershell
python oof/aggregate_oof.py --in_dir runs/oof_gyr/oof `
  --out_json runs/oof_gyr/oof/report_precalib.json `
  --out_npz  runs/oof_gyr/oof/oof_predictions.npz
```
3) 校准（基于 SA3 的 γ-Δs 两阶段）：
```powershell
python oof/calibrate_oof.py --oof_npz runs/oof_gyr/oof/oof_predictions.npz `
  --out_cal runs/oof_gyr/oof/calibrator_oof.json `
  --out_report runs/oof_gyr/oof/report_postcalib.json `
  --sa_mode deming `
  --logv_min <train_logv_min> `
  --logv_max <train_logv_max>
```
4) 进一步调节各向异性 γ 以匹配 C68：
```powershell
python oof/tune_gamma_cov68.py --oof_npz runs/oof_gyr/oof/oof_predictions.npz `
  --base_cal runs/oof_gyr/oof/calibrator_oof.json `
  --out_cal  runs/oof_gyr/oof/calibrator_final.json `
  --target_cov68 0.68 --gamma_min 0.7 --gamma_max 1.3
```

---

## 单模型评估与可视化

- 入口：`eval.py`
```powershell
python eval.py --route gyr `
  --config config_gyr_oof.yaml `
  --npz F:/SLAMdata/_cache/imu_euroc_MH4/MH_04_difficult_T512_S256.npz `
  --model runs/oof_gyr/best_fold0.pt `
  --plots_dir runs/oof_gyr/plots `
  --dump_preds_npz runs/oof_gyr/preds_fold0.npz
```
- 指标：`z2_mean`、`cov68`、`cov95`、`spearman`、边界饱和度等
- 可选：提供 `--calibrator_json`（仅三轴输出时生效）

> 说明：若未在校准阶段显式传入 `--logv_min/--logv_max`，校准器会默认裁剪到 `[-8, 6]`，可能与训练区间不一致，导致出图“截平”。推荐在校准时传入与训练一致的上下界。

---

## 目录结构（核心）

- **根目录**
  - `train_oof.py`：OOF 训练主脚本
  - `eval.py`：单模型评估与可视化
  - `dataset.py`：数据加载与 OOF 序列数据集构建
  - `models.py`：`IMURouteModel`，含 `BoundedLogVar` 与 SA3 方差头接入
  - `var_heads.py`：`Sa3VarHead`（三轴尺度+各向异性参数化）
  - `losses.py`：Gaussian/Student-t NLL 与训练期正则
  - `metrics.py`：z²覆盖率与阈值（含 Student-t 动态阈值）
  - `utils.py`、`utils_sa3.py`：工具函数/SA3 工具
  - `config_oof.yaml`、`config_gyr_oof.yaml`：配置示例
  - `oof/`：`predict_oof.py`、`aggregate_oof.py`、`calibrate_oof.py`、`tune_gamma_cov68.py`
  - `imu_oof/`：`calibrator.py`、`apply.py`（评估阶段可选校准）
  - `tools/`：数据生成/分折工具；`tools/legacy/` 为旧版工具
  - `datasets/`：数据集 I/O（EuRoC 辅助函数）
  - `scripts/`：环境/可视化脚本（可选）

---

## 常见配置要点

- **Student-t**：`train.dist=studentt`、`train.nu>2` 才有有限方差与正确阈值
- **SA3 参数化**：`train.variance_param=sa3` 且 `d_out=3`（步级标签）
- **边界与饱和**：`use_bounded=true` 使用 `tanh` 平滑有界，避免硬裁剪

- **上下界一致性（强烈推荐）**：训练区间会写入 checkpoint；评估从 ckpt 读取。若使用校准器，务必在 `oof/calibrate_oof.py` 传入相同的 `--logv_min/--logv_max`，避免评估/出图口径不一致。
- **GYR 分位数建议**：基于原始数据的分位数包络，GYR 的合理区间可设为 `[-11.18, -0.64]`（P05–P95）或更宽松的 `[-14.5, 0.1]`（P01–P99）。
- **Windows num_workers**：在 `config_*` 中建议设为 `0` 以避免 DataLoader 问题

---

## 已清理项

以下不再被训练/评估主流程依赖，已从仓库移除以避免混淆：

- **分析/一次性脚本**
  - `analyze_gyr_range.py`
  - `check_gyro_correlation.py`
  - `analysis_plots/`
- **旧/不常用工具**
  - `oof/tune_temperature.py`
  - `tools/gen_vis_pairs_euroc.py`
  - `tools/gen_vis_pairs_euroc_strict.py`
  - `tools/merge_vis_pairs_by_seq.py`

> 说明：`common/` 目录仍被 `tools/gen_euroc_step_npz.py` 等工具使用，请保留。

---

## 注意事项

- Windows CMD 批处理 `train_gyr_oof.cmd` 中的聚合命令参数应为 `--in_dir`（非 `--oof_dir`）；建议直接按上文 CLI 示例运行，或我可修正该批处理脚本。
- `.gitignore` 已忽略 `runs/`、`.pt`、`.npz` 等训练/数据产物。

---

## 许可证

仅供研究使用；如需商用请另行确认。
