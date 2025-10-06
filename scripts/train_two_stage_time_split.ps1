# -*- coding: utf-8 -*-
# 两阶段训练脚本：使用按时间切分的数据
# 阶段A：冻结backbone，只训练头部（warmup）
# 阶段B：解冻全网，加载阶段A权重，小学习率微调

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "两阶段训练：按时间切分数据" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# ============================================
# 阶段 A：头部热身（Warmup - Head Only）
# ============================================
Write-Host "[阶段 A/2] 冻结backbone，只训练头部（20 epochs）" -ForegroundColor Yellow
Write-Host "-----------------------------------------------" -ForegroundColor Yellow

python train_vis.py `
  --train_npz F:/SLAMdata/_cache/vis_split_time/train.npz `
  --val_npz   F:/SLAMdata/_cache/vis_split_time/val.npz `
  --save_dir  runs/vis_time_warmup_head `
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
  --train_npz F:/SLAMdata/_cache/vis_split_time/train.npz `
  --val_npz   F:/SLAMdata/_cache/vis_split_time/val.npz `
  --save_dir  runs/vis_time_finetune_all `
  --load_model runs/vis_time_warmup_head/best_vis_kendall.pt `
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
Write-Host "✅ 两阶段训练完成！" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "模型已保存到:" -ForegroundColor Cyan
Write-Host "  阶段A: runs/vis_time_warmup_head/best_vis_kendall.pt"
Write-Host "  阶段B: runs/vis_time_finetune_all/best_vis_kendall.pt"
Write-Host ""
Write-Host "接下来可以进行评估和校准:" -ForegroundColor Cyan
Write-Host "  python eval_vis.py \"
Write-Host "    --npz F:/SLAMdata/_cache/vis_split_time/val.npz \"
Write-Host "    --model runs/vis_time_finetune_all/best_vis_kendall.pt \"
Write-Host "    --auto_temp axis --isotonic_use \"
Write-Host "    --iso_quantile 0.90 --iso_shrink 0.7 \"
Write-Host "    --save_temp     runs/vis_time_finetune_all/temp_axis.json \"
Write-Host "    --isotonic_save runs/vis_time_finetune_all/iso_axis.json"
Write-Host ""

