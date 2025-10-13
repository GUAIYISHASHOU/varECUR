# ============================================================
# Optimized Training Pipeline (PowerShell Script)
# Usage: .\run_training.ps1
# ============================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Optimized Model Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# ---- Configuration ----
$TRAIN_NPZ = "F:/SLAMdata/_cache/macro/train_frame.npz"
$VAL_NPZ = "F:/SLAMdata/_cache/macro/val_frame.npz"
$TEST_NPZ = "F:/SLAMdata/_cache/macro/test_frame.npz"
$GEOM_STATS = "F:/SLAMdata/_cache/macro/geom_stats_24d.npz"
$SAVE_DIR = "runs/vis_macro_sa_mh_v5_a3_rank"

# ---- Step 1: Regenerate Data (Optional) ----
Write-Host "[Step 1] Regenerate training data (optional, for more anisotropic samples)" -ForegroundColor Yellow
$response = Read-Host "Regenerate data? (y/N)"
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "Generating data..." -ForegroundColor Green
    python batch_gen_macro.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Data generation failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "`nCalculating normalization statistics..." -ForegroundColor Green
    python tools/fit_geom_stats.py `
        --train_npz $TRAIN_NPZ `
        --out_npz $GEOM_STATS
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Statistics calculation failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Skip data generation, using existing data.`n" -ForegroundColor Gray
}

# ---- Step 2: Train Model ----
Write-Host "`n[Step 2] Start training optimized model" -ForegroundColor Yellow
Write-Host "Save directory: $SAVE_DIR" -ForegroundColor Gray
Write-Host "Key parameters: a_max=3.0, nll_weight=1.5, bce_weight=0.6, rank_weight=0.3`n" -ForegroundColor Gray

python train_macro.py `
    --train_npz $TRAIN_NPZ `
    --val_npz $VAL_NPZ `
    --geom_stats_npz $GEOM_STATS `
    --save_dir $SAVE_DIR `
    --epochs 40 --stage1_epochs 8 `
    --batch_size 32 --lr 2e-4 `
    --a_max 3.0 --drop_token_p 0.1 `
    --heads 4 --layers 1 --d_model 128 `
    --nll_weight 1.5 --bce_weight 0.6 `
    --rank_weight 0.3 `
    --patience 12

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nTraining failed!" -ForegroundColor Red
    exit 1
}

# ---- Step 3: Evaluate Model ----
Write-Host "`n[Step 3] Evaluate model performance" -ForegroundColor Yellow

# Validation set evaluation
Write-Host "`nEvaluating validation set..." -ForegroundColor Green
python eval_macro.py `
    --npz $VAL_NPZ `
    --ckpt "$SAVE_DIR/best_macro_sa.pt" `
    --geom_stats_npz $GEOM_STATS `
    --plots_dir "$SAVE_DIR/val_plots"

# Test set evaluation (with threshold scanning)
Write-Host "`nEvaluating test set (with threshold scanning)..." -ForegroundColor Green
python eval_macro.py `
    --npz $TEST_NPZ `
    --ckpt "$SAVE_DIR/best_macro_sa.pt" `
    --geom_stats_npz $GEOM_STATS `
    --scan_q_threshold `
    --plots_dir "$SAVE_DIR/test_plots"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nEvaluation failed!" -ForegroundColor Red
    exit 1
}

# ---- Complete ----
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "All steps completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nModel saved at: $SAVE_DIR/best_macro_sa.pt" -ForegroundColor Cyan
Write-Host "Evaluation results and plots saved at: $SAVE_DIR/test_plots/`n" -ForegroundColor Cyan
