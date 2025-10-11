#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成宏观模式数据 (Frame-level Uncertainty)
按照新的数据集划分生成 train/val/test NPZ
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可显示进度条 (pip install tqdm)\n")

# 配置路径
EUROC_ROOT = "F:/SLAMdata/EUROC"
OUT_DIR = "F:/SLAMdata/_cache/macro"

# 数据集划分
TRAIN_SEQS = [
    "V1_01_easy",
    "V2_01_easy",
    "MH_01_easy",
    "MH_02_easy",
    "V1_02_medium",
    "V2_02_medium",
    "V1_03_difficult",
    "MH_05_difficult",
]

VAL_SEQS = [
    "MH_03_medium",
    "MH_04_difficult",
]

TEST_SEQS = [
    "V2_03_difficult",
]

# 参数配置
PARAMS = {
    "patch": 32,
    "K_tokens": 256,
    "deltas": [1, 2],
    "frame_step": 2,
    "min_matches": 24,
    "pos_thr_px": 3.0,
    "pos_thr_px_v1": 3.0,
    "pos_thr_px_v2": 3.0,
    "pos_thr_px_mh": 3.0,
    "err_clip_px": 20.0,
    "inlier_thr_px": 2.0,  # 新增：内点判定阈值
    
    # ========== 关键帧模式参数（IC-GVINS对齐）==========
    # 推荐：如果要与 IC-GVINS 行为对齐，将 kf_enable 设为 True
    # 关键帧模式会生成24维几何特征（20维token级 + 4维帧对级）
    # 默认模式保留20维特征（向后兼容）
    "kf_enable": False,           # ← 改为 True 启用关键帧模式（推荐）
    "kf_parallax_px": 20.0,       # 关键帧视差阈值（像素，与IC-GVINS对齐）
    "kf_max_interval_s": 0.5,     # 最大时间间隔（秒，超过则强制输出）
    "kf_min_interval_s": 0.08,    # 最小时间间隔（秒，避免过密采样）
    "emit_non_kf_ratio": 0.2,     # 非关键帧对比例（0.1-0.3为宜，增加数据多样性）
}

def print_header(msg, color="cyan"):
    """打印带颜色的标题"""
    colors = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "end": "\033[0m"
    }
    c = colors.get(color, "")
    end = colors["end"]
    print(f"\n{c}{'='*50}")
    print(msg)
    print(f"{'='*50}{end}\n")

def generate_macro_data(set_name, sequences, output_npz):
    """生成宏观模式数据"""
    print_header(f"生成 {set_name} 集", "green")
    print(f"序列: {', '.join(sequences)}")
    print(f"输出: {output_npz}\n")
    
    # 构建命令
    deltas_str = " ".join(map(str, PARAMS["deltas"]))
    seqs_str = " ".join(sequences)
    
    cmd = [
        "python", "tools/gen_macro_samples_euroc.py",
        "--euroc_root", EUROC_ROOT,
        "--seqs"] + sequences + [
        "--cam_id", "0",
        "--out_npz", output_npz,
        "--patch", str(PARAMS["patch"]),
        "--K_tokens", str(PARAMS["K_tokens"]),
        "--min_matches", str(PARAMS["min_matches"]),
        "--pos_thr_px", f"{PARAMS['pos_thr_px']}",
        "--pos_thr_px_v1", f"{PARAMS['pos_thr_px_v1']}",
        "--pos_thr_px_v2", f"{PARAMS['pos_thr_px_v2']}",
        "--pos_thr_px_mh", f"{PARAMS['pos_thr_px_mh']}",
        "--err_clip_px", f"{PARAMS['err_clip_px']}",
        "--inlier_thr_px", f"{PARAMS['inlier_thr_px']}",
    ]
    
    # 根据模式添加不同的参数
    if PARAMS["kf_enable"]:
        # 关键帧模式
        cmd += [
            "--kf_enable",
            "--kf_parallax_px", str(PARAMS["kf_parallax_px"]),
            "--kf_max_interval_s", str(PARAMS["kf_max_interval_s"]),
            "--kf_min_interval_s", str(PARAMS["kf_min_interval_s"]),
            "--emit_non_kf_ratio", str(PARAMS["emit_non_kf_ratio"]),
        ]
    else:
        # 默认间隔帧模式
        cmd += [
            "--deltas"] + [str(d) for d in PARAMS["deltas"]] + [
            "--frame_step", str(PARAMS["frame_step"]),
        ]
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"\n\033[91m✗ {set_name}集生成失败!\033[0m\n")
        return False
    
    print(f"\n\033[92m✓ {set_name}集生成成功!\033[0m\n")
    
    # 显示数据统计
    if os.path.exists(output_npz):
        print("\033[96m数据统计:\033[0m")
        try:
            import numpy as np
            z = np.load(output_npz)
            print(f"  样本数: {z['patches'].shape[0]}")
            print(f"  Patches shape: {z['patches'].shape}")
            print(f"  Geoms shape: {z['geoms'].shape}")
            print(f"  Y_true shape: {z['y_true'].shape}")
            print()
        except Exception as e:
            print(f"  (无法读取统计信息: {e})")
    
    return True

def main():
    print_header("批量生成宏观模式数据 (Macro Mode)", "cyan")
    
    print("数据集划分:")
    print(f"  训练集 ({len(TRAIN_SEQS)} seqs): {', '.join(TRAIN_SEQS)}")
    print(f"  验证集 ({len(VAL_SEQS)} seqs): {', '.join(VAL_SEQS)}")
    print(f"  测试集 ({len(TEST_SEQS)} seq ): {', '.join(TEST_SEQS)}")
    print()
    print("参数配置:")
    print(f"  K_tokens: {PARAMS['K_tokens']} (每帧最多{PARAMS['K_tokens']}个点)")
    print(f"  patch: {PARAMS['patch']}x{PARAMS['patch']}")
    
    if PARAMS["kf_enable"]:
        print(f"  \033[93m模式: 关键帧选取（IC-GVINS对齐）\033[0m")
        print(f"  kf_parallax_px: {PARAMS['kf_parallax_px']} (关键帧视差阈值)")
        print(f"  kf_max_interval_s: {PARAMS['kf_max_interval_s']} (最大时间间隔)")
        print(f"  kf_min_interval_s: {PARAMS['kf_min_interval_s']} (最小时间间隔)")
        print(f"  emit_non_kf_ratio: {PARAMS['emit_non_kf_ratio']} (非关键帧对比例)")
    else:
        print(f"  \033[93m模式: 默认间隔帧\033[0m")
        print(f"  deltas: {PARAMS['deltas']} (时间间隔)")
        print(f"  frame_step: {PARAMS['frame_step']} (每隔{PARAMS['frame_step']-1}帧采样)")
    
    print(f"  min_matches: {PARAMS['min_matches']} (最少匹配点)")
    print(f"  pos_thr_px: {PARAMS['pos_thr_px']} (正负样本判定阈值, 所有序列统一)")
    print(f"  err_clip_px: {PARAMS['err_clip_px']} (尾部裁剪阈值)")
    print(f"  inlier_thr_px: {PARAMS['inlier_thr_px']} (内点判定阈值)")
    print()
    
    # 检查EuRoC数据集
    if not os.path.exists(EUROC_ROOT):
        print(f"\033[91m错误: 找不到EuRoC数据集: {EUROC_ROOT}\033[0m")
        print("请修改脚本中的 EUROC_ROOT 路径")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 定义输出文件
    train_npz = os.path.join(OUT_DIR, "train_frame.npz")
    val_npz = os.path.join(OUT_DIR, "val_frame.npz")
    test_npz = os.path.join(OUT_DIR, "test_frame.npz")
    
    # 1. 生成训练集
    if not generate_macro_data("训练", TRAIN_SEQS, train_npz):
        sys.exit(1)
    
    # 2. 生成验证集
    if not generate_macro_data("验证", VAL_SEQS, val_npz):
        sys.exit(1)
    
    # 3. 生成测试集
    if not generate_macro_data("测试", TEST_SEQS, test_npz):
        sys.exit(1)
    
    # 完成
    print_header("全部完成!", "green")
    print("生成的数据文件:")
    print(f"  训练集: {train_npz}")
    print(f"  验证集: {val_npz}")
    print(f"  测试集: {test_npz}")
    print()
    print("\033[93m下一步: 开始训练\033[0m")
    print()
    print(f"python train_macro.py \\")
    print(f"  --train_npz {train_npz} \\")
    print(f"  --val_npz {val_npz} \\")
    print(f"  --save_dir runs/vis_macro_sa \\")
    print(f"  --epochs 40 --batch_size 32 --lr 2e-4 \\")
    print(f"  --a_max 3.0 --drop_token_p 0.1 \\")
    print(f"  --heads 4 --layers 1 --d_model 128")
    print()

if __name__ == "__main__":
    main()

