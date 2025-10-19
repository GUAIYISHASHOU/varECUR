## IMU_EUROC_OOF — IMU不确定性建模与OOF训练

基于 EuRoC IMU 序列的数据驱动不确定性建模与评估框架，支持：
- OOF(Out-of-Fold) 分折训练，避免数据泄漏
- 学习对角各向同性/各向异性 log-variance（支持 Student-t）
- 训练后 OOF 预测汇总、温度/各向异性后校准与评估可视化
 - Student-t 覆盖率阈值按 F 分布分位自动对齐（dist=studentt 时）

### 环境依赖
- Python 3.10+
- PyTorch 2.x（GPU 可选）
- Numpy, Matplotlib, Tqdm, PyYAML

示例（Windows PowerShell）安装依赖：
```bash
pip install torch numpy matplotlib tqdm pyyaml
```

### 目录结构概览
- `train_oof.py`：主 OOF 训练脚本
- `oof/`：OOF 推断、聚合与后校准脚本
  - `predict_oof.py`、`aggregate_oof.py`、`calibrate_oof.py`、`tune_temperature.py`
- `tools/`：数据清单/分折与诊断等工具（不常用脚本已移至 `tools/legacy/`）
  - `generate_manifest.py`、`build_group_stratified_folds.py`、`inspect_npz_detailed.py`、`check_oof_consistency.py`、`infer_ensemble.py`
- `dataset.py`：数据读取与构建
- `models.py`：TCN/可选Transformer 主干与方差头（direct/sa3）
- `metrics.py`、`losses.py`：训练指标与损失
- `imu_oof/`：在线应用与各向异性校准器
- `config_oof.yaml`：示例配置

### 数据准备
期望的 NPZ 键：
- 输入 X：
  - 加速度路由 `acc`：`X_IMU_ACC` 或 `X_acc` 或 `X`，形状 `(N, T, D)`
  - 陀螺路由 `gyr`：`X_IMU_GYR` 或 `X_gyr` 或 `X`
- 标签（优先步级三轴误差平方 e²）：
  - `acc`：`E2_IMU_ACC`，形状 `(N, T, 3)`；若缺失则回退 `Y_IMU_ACC`（会扩为3轴）
  - `gyr`：`E2_IMU_GYR`，形状 `(N, T, 3)`；若缺失则回退 `Y_IMU_GYR`
- 可选：
  - `MASK_IMU`：有效掩码 `(N, T)`；缺失时视为全1
  - `SEQ_NAME`：序列名 `(N,)`，用于 OOF 分组

序列组织：`oof.seq_npz_dir` 目录中每个序列一个 `*.npz`，如 `MH_01_easy.npz`。

### 快速开始
1) 生成 `manifest.json`
```powershell
python tools/generate_manifest.py --seq_dir F:/SLAMdata/_cache/imu_euroc_MH4 --out manifest.json --routes acc gyr
```
2) 基于清单构建或复用 `splits_oof.json`
```powershell
python tools/build_group_stratified_folds.py --manifest manifest.json --out splits_oof.json --k 4 --seed 42 --route acc
```
3) 执行 OOF 训练（输出到 `runs/oof_acc/`）
```powershell
python train_oof.py --route acc --config config_oof.yaml --splits splits_oof.json --seq_dir F:/SLAMdata/_cache/imu_euroc_MH4 --folds all
```

### 生成 Manifest 与分折
1) 生成清单（统计每序列窗口数）：
```bash
python tools/generate_manifest.py --seq_dir F:/SLAMdata/_cache/imu_euroc_MH4 --out manifest.json --routes acc gyr
```
2) 基于清单构建分层 Group KFold：
```bash
python tools/build_group_stratified_folds.py --manifest manifest.json --out splits_oof.json --k 4 --seed 42 --route acc
```
（可选）可使用 `tools/` 下脚本检查分折一致性与数据质量。

### OOF 训练
配置文件见 `config_oof.yaml`，关键项：
- `common.device`、`common.num_workers`
- `train.dist` = `gauss` 或 `studentt`（兼容 `use_studentt`）
- `train.variance_param` = `direct` 或 `sa3`（三轴各向异性）
- `train.nu`、`train.logv_min`、`train.logv_max`、`train.lambda_*`
- `oof.seq_npz_dir`：序列NPZ目录；`oof.splits_json`：分折文件

运行训练：
```bash
python train_oof.py --route acc --config config_oof.yaml --splits splits_oof.json --seq_dir F:/SLAMdata/_cache/imu_euroc_MH4 --folds all
```
输出：`runs/oof_acc/`
- `best_fold{K}.pt` 每折最佳权重（含 `cfg/d_in/d_out/scaler/nu/...`）
- `curve_fold{K}.json` 训练/验证曲线
- `oof_summary.json` 汇总

### OOF 推断、聚合与后校准
1) 按每折最佳权重在其验证集上推断（默认输出到 `runs/oof_acc/oof/`）：
```bash
python oof/predict_oof.py --runs_dir runs/oof_acc --splits splits_oof.json --route acc --device cpu
```
2) 聚合各折预测并统计覆盖率、z² 等，同时导出合并数组：
```bash
python oof/aggregate_oof.py --in_dir runs/oof_acc/oof --out_json runs/oof_acc/oof/report_precalib.json --out_npz runs/oof_acc/oof/oof_predictions.npz
```
3A) 各向同性温度校准（logv' = a*logv + b）：
```bash
python oof/tune_temperature.py --oof_npz runs/oof_acc/oof/oof_predictions.npz \
  --out_cal runs/oof_acc/oof/calibrator_temp.json --out_report runs/oof_acc/oof/report_postcalib_temp.json \
  --target_cov68 0.68 --a_min 0.7 --a_max 1.3 --a_steps 121
```
3B) 三轴各向异性 SA 域校准（γ-Δs 稳定方案）：
```bash
python oof/calibrate_oof.py --oof_npz runs/oof_acc/oof/oof_predictions.npz \
  --out_cal runs/oof_acc/oof/calibrator_oof.json --out_report runs/oof_acc/oof/report_postcalib_sa.json
```

在线应用（仅各向异性三轴时生效）：
- 评估时可传入 `--calibrator_json`，内部将使用 `imu_oof/SA3AffineCalibrator` 应用于 `(N,T,3)` 的 log-variance。

### 单模型评估与可视化
```bash
python eval.py --route acc --config config_oof.yaml \
  --npz F:/SLAMdata/_cache/imu_euroc_MH4/MH_03_medium.npz \
  --model runs/oof_acc/best_fold0.pt \
  --calibrator_json runs/oof_acc/oof/calibrator_oof.json \
  --plots_dir runs/test_eval/MH_03_acc_fold0
```
将输出 z² 直方图/时间序列与覆盖率柱状图。

### 关键设计
- 模型：TCN 堆叠，可选 Transformer 编码器；方差参数化支持 `direct` 与 `sa3`
- 损失：Gaussian 或 Student-t，对三轴时支持对角各向异性 NLL 与正则项
- 指标：`z²_mean` 采用 χ²(df)/df 口径；当 `dist=studentt` 时，覆盖率阈值使用 F 分位映射（非高斯常数）
- 训练期结构控制：各向异性预热冻结、`s/a` 学习率拆分、动态 `logv_max` 护栏、逐轴覆盖率正则（可选）
- 校准：
  - 温度：全局仿射 `a·logv + b`（各向同性）
  - SA 校准：仅缩放各向异性 γ 并用 Δs 调整尺度使 z²≈1

### 常见路径与文件
- 配置：`config_oof.yaml`
- 分折：`splits_oof.json`，或工具生成
- 训练产物：`runs/oof_{route}/`
- OOF 合并数组：`runs/oof_{route}/oof/oof_predictions.npz`

### 备注
- Windows 上建议将 `common.num_workers` 设为 0
- 若自定义数据根目录，请同步更新 `config_oof.yaml` 中的 `oof.seq_npz_dir` 与 `oof.splits_json`


