# 环境（Python 3.10+，PyTorch ≥ 2.x）
pip install torch torchvision torchaudio  # 选择与你 CUDA 匹配的命令
pip install numpy matplotlib

# 1) 从 EuRoC 生成步级标签 NPZ
# 单序列示例（输出到 _cache\euroc_step_v2）
python tools/gen_euroc_step_npz.py `
  --euroc_root F:\SLAMdata `
  --seqs V2_03_difficult `
  --out F:\SLAMdata\_cache\euroc_step_v2 `
  --T 512 --stride 256 --use_gt_bias `
  --acc_smooth 7 --mask_edge 3 `
  --gyr_smooth 7 --gyr_mask_edge 3 `
  --est_tau 2 --w_logmap

# 批量六序列
python tools/gen_euroc_step_npz.py `
  --euroc_root F:\SLAMdata `
  --seqs "V1_01_easy,V1_02_medium,V1_03_difficult,V2_01_easy,V2_02_medium,V2_03_difficult" `
  --out F:\SLAMdata\_cache\euroc_step_v2 `
  --T 512 --stride 256 --use_gt_bias `
  --acc_smooth 7 --mask_edge 3 `
  --gyr_smooth 7 --gyr_mask_edge 3 `
  --est_tau 2 --w_logmap

# 2) NPZ 质量体检（含出图）
python tools/npz_sanity_check.py --glob "F:\SLAMdata\_cache\euroc_step_v2\*.npz" --plots
# 轻量体检
python tools/check_npz.py --root "F:\SLAMdata\_cache\euroc_step_v2"

# 3) 每序列切分并合并（防泄露：按时间切块）
python tools/split_eachseq_merge_npz.py `
  --root F:\SLAMdata\_cache\euroc_step_v2 `
  --out_dir F:\SLAMdata\_cache\euroc_split_eachseq `
  --pattern "*.npz" --split 0.70 0.15 0.15 --mode chronological

# 4) 训练（acc / gyr）
# 4.1 一维高斯（默认）
python train.py --config config.yaml --route acc `
  --train_npz "F:/SLAMdata/_cache/euroc_split_eachseq/train_all.npz" `
  --val_npz   "F:/SLAMdata/_cache/euroc_split_eachseq/val_all.npz" `
  --test_npz  "F:/SLAMdata/_cache/euroc_split_eachseq/test_all.npz"

# 4.2 三轴 Student‑t 对角 NLL（步级标签）
python train.py --config config.yaml --route gyr `
  --train_npz "F:/SLAMdata/_cache/euroc_split_eachseq/train_all.npz" `
  --val_npz   "F:/SLAMdata/_cache/euroc_split_eachseq/val_all.npz" `
  --test_npz  "F:/SLAMdata/_cache/euroc_split_eachseq/test_all.npz" `
  --imu_loss studentt_diag --student_nu 6.0

# 5) 评测与按序列出图
# 总体评测（示例 gyr）
python eval.py --route gyr --npz "F:/SLAMdata/_cache/euroc_split_eachseq/test_all.npz" --model runs/gyr_euroc_step/best.pt

# 按序列图（需要 split 的 manifest.json）
python eval.py --route acc `
  --npz "F:/SLAMdata/_cache/euroc_split_eachseq/test_all.npz" `
  --model runs/acc_euroc_step/best.pt `
  --manifest "F:/SLAMdata/_cache/euroc_split_eachseq/manifest.json" `
  --plots_dir runs/acc_euroc_step/perseq_plots

# 备注
# - 生成阶段可加 --use_gt_bias，但训练/评测需保持一致口径。
# - 可用 train.yaml 的 z2_center 做轻度居中；GNSS 路由支持 post_scale 与 t 口径评测。
