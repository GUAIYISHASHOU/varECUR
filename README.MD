# 环境要求
Python 3.10+`PyTorch ≥ 2.x
```bash
pip install torch torchvision torchaudio  # 选择与CUDA匹配的版本
pip install numpy matplotlib pyyaml
```

# 快速开始

## 1) 从 EuRoC 生成步级标签 NPZ
```bash
# 单序列示例
python tools/gen_euroc_step_npz.py `
  --euroc_root F:\SLAMdata `
  --seqs V2_03_difficult `
  --out F:\SLAMdata\_cache\euroc_step_v2 `
  --T 512 --stride 256 --use_gt_bias `
  --acc_smooth 7 --mask_edge 3 `
  --gyr_smooth 7 --gyr_mask_edge 3 `
  --est_tau 2 --w_logmap

# 批量生成六个序列
python tools/gen_euroc_step_npz.py `
  --euroc_root F:\SLAMdata `
  --seqs "V1_01_easy,V1_02_medium,V1_03_difficult,V2_01_easy,V2_02_medium,V2_03_difficult" `
  --out F:\SLAMdata\_cache\euroc_step_v2 `
  --T 512 --stride 256 --use_gt_bias `
  --acc_smooth 7 --mask_edge 3 `
  --gyr_smooth 7 --gyr_mask_edge 3 `
  --est_tau 2 --w_logmap
```

## 2) NPZ 数据质量检查
```bash
# 详细体检（包含图像输出）
python tools/npz_sanity_check.py --glob "F:\SLAMdata\_cache\euroc_step_v2\*.npz" --plots

# 快速检查
python tools/check_npz.py --root "F:\SLAMdata\_cache\euroc_step_v2"
```

## 3) 数据集切分与合并
```bash
# 按时间顺序切分`防止数据泄露
python tools/split_eachseq_merge_npz.py `
  --root F:\SLAMdata\_cache\euroc_step_v2 `
  --out_dir F:\SLAMdata\_cache\euroc_split_eachseq `
  --pattern "*.npz" --split 0.70 0.15 0.15 --mode chronological
```

# 训练模型

## 4.1) Gaussian+Huber 损失（推荐）
```bash
# 使用配置文件训练
python train.py --config config.yaml

# 或者手动指定参数
python train.py --config config.yaml `
  --route gyr `
  --imu_loss gauss_huber `
  --huber_delta 1.5 `
  --cal_reg 0.05 `
  --run_dir runs/gyr_euroc_step_huber
```

## 4.2) 传统高斯损失
```bash
python train.py --config config.yaml `
  --route acc `
  --imu_loss iso `
  --run_dir runs/acc_euroc_step_iso
```

## 4.3) Student-t 对角损失（异常值稳健）
```bash
python train.py --config config.yaml `
  --route gyr `
  --imu_loss studentt_diag `
  --student_nu 6.0 `
  --run_dir runs/gyr_euroc_step_studentt
```

# 模型评估

## 5.1) 基础评估
```bash
python eval.py --route gyr `
  --npz "F:/SLAMdata/_cache/euroc_split_eachseq/test_all.npz" `
  --model runs/gyr_euroc_step_huber/best.pt
```

## 5.2) 带全局温度校正的评估
```bash
python eval.py --route gyr `
  --npz "F:/SLAMdata/_cache/euroc_split_eachseq/test_all.npz" `
  --model runs/gyr_euroc_step_huber/best.pt `
  --auto_temp global
```

## 5.3) 按序列分析与绘图
```bash
python eval.py --route gyr `
  --npz "F:/SLAMdata/_cache/euroc_split_eachseq/test_all.npz" `
  --model runs/gyr_euroc_step_huber/best.pt `
  --manifest "F:/SLAMdata/_cache/euroc_split_eachseq/manifest.json" `
  --plots_dir runs/gyr_euroc_step_huber/perseq_plots `
  --auto_temp global
```

# 损失函数对比

| 损失函数 | 特点 | 适用场景 |
|---------|------|----------|
| `iso` | 传统高斯NLL | 基线对比 |
| `gauss_huber` | Gaussian+Huber`稳健性强 | **推荐使用** |
| `studentt_diag` | Student-t对角`异常值稳健 | 数据噪声较大时 |

# 重要说明

⚠️ **配置文件语法**：使用 `--config config.yaml` 或 `@config.yaml`
⚠️ **数据一致性**：生成`训练`评测需保持相同的预处理参数
⚠️ **温度校正**：Huber损失配合全局温度校正效果最佳
⚠️ **路径格式**：Windows下使用正斜杠 `/` 或转义反斜杠 `\\`
