# OOF (Out-of-Fold) Calibration Workflow
# Strict no-leakage approach: calibration from train OOF predictions only

$ErrorActionPreference = "Stop"

# ============================================================
# Configuration (modify as needed)
# ============================================================
$TRAIN_NPZ = "F:/SLAMdata/_cache/macro/train_frame.npz"
$VAL_NPZ = "F:/SLAMdata/_cache/macro/val_frame.npz"
$TEST_NPZ = "F:/SLAMdata/_cache/macro/test_frame.npz"
$GEOM_STATS = "F:/SLAMdata/_cache/macro/geom_stats_24d.npz"
$SAVE_ROOT = "runs/oof_macro_v1"
$K_FOLDS = 5

# ============================================================
# Step 1: Generate K-fold splits (run once)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 1: Generate K-fold splits" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$KFOLD_JSON = "$SAVE_ROOT/k${K_FOLDS}_splits.json"

if (Test-Path $KFOLD_JSON) {
    Write-Host "K-fold split file exists: $KFOLD_JSON" -ForegroundColor Yellow
    Write-Host "Skipping generation..." -ForegroundColor Yellow
} else {
    python tools/make_kfold_splits.py --train_npz $TRAIN_NPZ --out_json $KFOLD_JSON --k $K_FOLDS --seed 42
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: K-fold split failed" -ForegroundColor Red
        exit 1
    }
}

# ============================================================
# Step 2: OOF training and calibrator fitting (train only)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 2: OOF training and calibrator fitting" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Note: This will train $K_FOLDS models, may take a while..." -ForegroundColor Yellow

python tools/train_oof.py --train_npz $TRAIN_NPZ --geom_stats_npz $GEOM_STATS --kfold_json $KFOLD_JSON --save_root $SAVE_ROOT --epochs 40 --stage1_epochs 8 --batch_size 32 --lr 2e-4 --a_max 3.0 --drop_token_p 0.1 --heads 4 --layers 1 --d_model 128 --nll_weight 1.5 --bce_weight 0.6 --rank_weight 0.3 --patience 12 --sa_mode deming --deming_lambda 1.0

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: OOF training failed" -ForegroundColor Red
    exit 1
}

$CALIBRATOR_JSON = "$SAVE_ROOT/calibrator_oof.json"
Write-Host "`nOOF calibrator generated: $CALIBRATOR_JSON" -ForegroundColor Green

# ============================================================
# Step 3: Evaluate on validation set (apply OOF calibration)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 3: Evaluate on validation set (apply OOF calibration)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$BEST_CKPT = "$SAVE_ROOT/fold0/best_macro_sa.pt"

python eval_macro.py --npz $VAL_NPZ --ckpt $BEST_CKPT --geom_stats_npz $GEOM_STATS --calibrator_json $CALIBRATOR_JSON --kappa 1.0 --sa_recenter --scan_q_threshold --plots_dir "$SAVE_ROOT/val_plots_oof"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Validation evaluation failed" -ForegroundColor Red
    exit 1
}

# ============================================================
# Step 4: Evaluate on test set (apply OOF calibration)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 4: Evaluate on test set (apply OOF calibration)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python eval_macro.py --npz $TEST_NPZ --ckpt $BEST_CKPT --geom_stats_npz $GEOM_STATS --calibrator_json $CALIBRATOR_JSON --kappa 1.0 --sa_recenter --scan_q_threshold --plots_dir "$SAVE_ROOT/test_plots_oof"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Test evaluation failed" -ForegroundColor Red
    exit 1
}

# ============================================================
# Done
# ============================================================
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "OOF workflow completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nResults:" -ForegroundColor Yellow
Write-Host "  - OOF calibrator: $CALIBRATOR_JSON" -ForegroundColor White
Write-Host "  - Validation results: $SAVE_ROOT/val_plots_oof/" -ForegroundColor White
Write-Host "  - Test results: $SAVE_ROOT/test_plots_oof/" -ForegroundColor White

Write-Host "`nOOF advantages:" -ForegroundColor Yellow
Write-Host "  - Calibration from train OOF predictions only" -ForegroundColor White
Write-Host "  - Val/test never used in fitting, zero leakage" -ForegroundColor White
Write-Host "  - Uses train distribution, more robust" -ForegroundColor White
