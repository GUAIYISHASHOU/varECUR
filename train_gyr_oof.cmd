@echo off
REM 陀螺仪OOF训练脚本 - Windows批处理版本

echo ==========================================
echo 陀螺仪 OOF 训练流程
echo ==========================================
echo.
echo [INFO] 激活conda环境: LAP3GPU
call conda activate LAP3GPU
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 无法激活LAP3GPU环境！
    pause
    exit /b 1
)
echo.

REM 配置路径
set DATA_DIR=F:/SLAMdata/_cache/IMU_GYR
set CONFIG=config_gyr_oof.yaml
set SPLITS=splits_oof_gyr.json
set SAVE_DIR=runs/oof_gyr

echo [Step 1/1] 训练所有折（4折交叉验证）...
python train_oof.py --route gyr --config %CONFIG% --splits %SPLITS% --seq_dir %DATA_DIR%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 训练失败！
    pause
    exit /b 1
)

echo.
echo ==========================================
echo ✓ 训练完成！
echo 最佳模型: %SAVE_DIR%\best_fold*.pt
echo 摘要: %SAVE_DIR%\oof_summary.json
echo ==========================================
echo.

REM 询问是否继续后处理
set /p continue="是否继续OOF预测和校准? (y/n): "
if /i "%continue%"=="y" (
    echo.
    echo [Step 2/5] 生成OOF预测...
    python oof/predict_oof.py --runs_dir %SAVE_DIR% --splits %SPLITS% --route gyr --device cuda
    
    echo [Step 3/5] 聚合OOF预测...
    python oof/aggregate_oof.py --in_dir %SAVE_DIR%/oof --out_json %SAVE_DIR%/oof/report_precalib.json --out_npz %SAVE_DIR%/oof/oof_predictions.npz
    
    echo [Step 4/5] 校准预测...
    python oof/calibrate_oof.py --oof_npz %SAVE_DIR%/oof/oof_predictions.npz --out_cal %SAVE_DIR%/oof/calibrator_oof.json --out_report %SAVE_DIR%/oof/report_postcalib.json --sa_mode deming
    
    echo [Step 5/5] 微调各向异性gamma...
    python oof/tune_gamma_cov68.py --oof_npz %SAVE_DIR%/oof/oof_predictions.npz --base_cal %SAVE_DIR%/oof/calibrator_oof.json --out_cal %SAVE_DIR%/oof/calibrator_final.json --target_cov68 0.68 --gamma_min 0.7 --gamma_max 1.3
    
    echo.
    echo ==========================================
    echo ✓ 完整流程完成！
    echo 最终校准器: %SAVE_DIR%/oof/calibrator_final.json
    echo ==========================================
)

pause

