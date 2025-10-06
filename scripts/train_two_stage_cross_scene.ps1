# -*- coding: utf-8 -*-
# 跨场景两阶段训练脚本：使用旧的按序列切分数据（对比基线）
# 阶段A：冻结backbone，只训练头部（warmup）
# 阶段B：解冻全网，加载阶段A权重，小学习率微调

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "跨场景两阶段训练：按序列切分数据" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# 确保目录存在
New-Item -ItemType Directory -Path "runs\vis_cross_warmup_head" -Force | Out-Null
New-Item -ItemType Directory -Path "runs\vis_cross_finetune_all" -Force | Out-Null

# ============================================
# 阶段 A：头部热身（Warmup - Head Only）
# ============================================
Write-Host "[阶段 A/2] 冻结backbone，只训练头部（20 epochs）" -ForegroundColor Yellow
Write-Host "-----------------------------------------------" -ForegroundColor Yellow

python train_vis.py `
  --train_npz F:/SLAMdata/_cache/vis_split/train.npz `
  --val_npz   F:/SLAMdata/_cache/vis_split/val.npz `
  --save_dir  runs/vis_cross_warmup_head `
  --epochs 20 --batch 256 `
  --loss studentt --nu 3 `
  --lv_min -10 --lv_max 10 `
  --lr 5e-4 --scheduler plateau `
  --photometric clahe `
  --grad_thr 0.03 --grad_scale 8 --w_gamma 3 `
  --skip_on_blur --err_clip_px 10 --calib_reg 1e-3 `
  --early_stop 6 `
  --freeze_backbone

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ 阶段A训练失败！" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ 阶段A完成！" -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 2

# ============================================
# 阶段 B：全网微调（Fine-tune All）
# ============================================
Write-Host "[阶段 B/2] 解冻全网，小学习率微调（60 epochs）" -ForegroundColor Yellow
Write-Host "-----------------------------------------------" -ForegroundColor Yellow

python train_vis.py `
  --train_npz F:/SLAMdata/_cache/vis_split/train.npz `
  --val_npz   F:/SLAMdata/_cache/vis_split/val.npz `
  --save_dir  runs/vis_cross_finetune_all `
  --load_model runs/vis_cross_warmup_head/best_vis_kendall.pt `
  --epochs 60 --batch 256 `
  --loss studentt --nu 3 `
  --lv_min -10 --lv_max 10 `
  --lr 5e-5 --scheduler plateau `
  --photometric clahe `
  --grad_thr 0.03 --grad_scale 8 --w_gamma 3 `
  --skip_on_blur --err_clip_px 10 --calib_reg 1e-3 `
  --early_stop 12

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ 阶段B训练失败！" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "✅ 跨场景两阶段训练完成！" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "模型已保存到:" -ForegroundColor Cyan
Write-Host "  阶段A: runs/vis_cross_warmup_head/best_vis_kendall.pt"
Write-Host "  阶段B: runs/vis_cross_finetune_all/best_vis_kendall.pt"
Write-Host ""
Write-Host "接下来可以进行评估和校准:" -ForegroundColor Cyan
Write-Host "  python eval_vis.py \"
Write-Host "    --npz F:/SLAMdata/_cache/vis_split/val.npz \"
Write-Host "    --model runs/vis_cross_finetune_all/best_vis_kendall.pt \"
Write-Host "    --auto_temp axis --isotonic_save iso_axis.json \"
Write-Host "    --plot_dir runs/vis_cross_finetune_all/eval_val"
Write-Host ""
Write-Host "最终测试集评估:" -ForegroundColor Cyan
Write-Host "  python eval_vis.py \"
Write-Host "    --npz F:/SLAMdata/_cache/vis_split/test.npz \"
Write-Host "    --model runs/vis_cross_finetune_all/best_vis_kendall.pt \"
Write-Host "    --use_temp temp_axis.json --isotonic_use iso_axis.json \"
Write-Host "    --plot_dir runs/vis_cross_finetune_all/eval_test"
Write-Host ""

