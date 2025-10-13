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
import numpy as np
import random
import time

# 固定随机种子，保证可复现
np.random.seed(42)
random.seed(42)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可显示进度条 (pip install tqdm)\n")

# 配置路径
EUROC_ROOT = "F:/SLAMdata/euroc"
OUT_DIR = "F:/SLAMdata/_cache/macro"

# 数据集划分（优化版：更均衡的域分布）
# 目标：让Val/Test都包含多域+多难度，提升泛化评估可靠性
# ========== 方案A: 单域策略（MH only，最保险）==========
# 优点: 无分布偏移、样本多(4165)、LogVar一致、训练包含difficult
# 缺点: 不测试跨域泛化
TRAIN_SEQS = [
    "MH_01_easy", "MH_02_easy", "MH_04_difficult",  # 2500样本 (752+919+361)
]

VAL_SEQS = [
    "MH_05_difficult",  # 684样本
]

TEST_SEQS = [
    "MH_03_medium",  # 981样本（测试medium泛化）
]

# # ========== 方案B: 多域策略（接受偏移）==========
# # 优点: 多域覆盖、泛化性强
# # 缺点: Test偏移+170%（Test全是MH高LogVar序列）
# TRAIN_SEQS = [
#     "V1_01_easy", "V1_02_medium",
#     "V2_01_easy", "V2_02_medium",
#     "MH_01_easy", "MH_03_medium",  # 5183样本
# ]
# 
# VAL_SEQS = [
#     "V1_03_difficult", "MH_02_easy",  # 1667样本
# ]
# 
# TEST_SEQS = [
#     "MH_04_difficult", "MH_05_difficult",  # 1045样本
# ]

# 参数配置
PARAMS = {
    "patch": 32,
    "K_tokens": 256,
    "deltas": [1, 2],
    "frame_step": 2,
    "min_matches": 32,
    # ===== 重投影误差阈值（筛选用于标签计算的点）=====
    "pos_thr_px": 5.0,       # 基础值：中等宽松
    "pos_thr_px_v1": 4.0,    # V1专用：稍严格（光照好）
    "pos_thr_px_v2": 5.0,    # V2专用：标准
    "pos_thr_px_mh": 6.0,    # MH专用：最宽松（纹理弱）
    "err_clip_px": 20.0,     # 尾部裁剪：极端外点截断
    "inlier_thr_px": 3.0,    # 内点判定：中位数误差<3.0px为内点（平衡质量与数量）
    
    # ========== 关键帧模式参数（IC-GVINS对齐 + 优化版）==========
    # 推荐：如果要与 IC-GVINS 行为对齐，将 kf_enable 设为 True
    # 关键帧模式会生成24维几何特征（20维token级 + 4维帧对级）
    # 默认模式也生成24维特征（统一格式）
    "kf_enable": True,             # ← 启用关键帧模式（推荐，IC-GVINS对齐）✅
    "kf_parallax_px": 20.0,       # 关键帧视差阈值（像素，提高以获得更多大视差/强各向异性样本）
    "kf_max_interval_s": 0.30,    # 最大时间间隔（秒，超过则强制输出）
    "kf_min_interval_s": 0.08,    # 最小时间间隔（秒，避免过密采样）
    "emit_non_kf_ratio": 0.20,    # 非关键帧对比例（提高以增加各向异性尾部覆盖）
    "obs_min_parallax_px": 3.0,   # 观测对最小视差阈值（像素，过滤极小运动）
    "kf_only": False,             # 是否只保留关键帧对（False=包含观测对）
}

# ========== 序列级参数覆盖（根据难度调整采样策略）==========
# 针对不同序列类型使用差异化参数，以优化数据分布
SEQ_OVERRIDES = {
    # ========== 统一核心参数策略（优化版）==========
    # 目标：减少Train/Val/Test分布差异，提升泛化性，增加各向异性尾部覆盖
    # 核心统一：kf_parallax_px=20, kf_max_interval_s=0.30, min_matches=32, emit_non_kf_ratio=0.20
    
    # Easy序列：优化参数（增加各向异性样本）
    "V1_01_easy":      {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "V2_01_easy":      {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "MH_01_easy":      {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "MH_02_easy":      {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    
    # Medium序列：优化参数
    "V1_02_medium":    {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "V2_02_medium":    {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "MH_03_medium":    {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    
    # Difficult序列：优化参数（统一，增加各向异性覆盖）
    "V1_03_difficult": {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "V2_03_difficult": {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "MH_04_difficult": {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
    "MH_05_difficult": {"emit_non_kf_ratio": 0.20, "obs_min_parallax_px": 3.0},
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

def generate_one_sequence(seq_name, temp_npz_path):
    """为单个序列生成数据（应用序列级参数覆盖）"""
    # 复制全局参数
    params = PARAMS.copy()
    
    # 应用序列级覆盖
    if seq_name in SEQ_OVERRIDES:
        params.update(SEQ_OVERRIDES[seq_name])
        print(f"  [{seq_name}] 应用序列级参数: {SEQ_OVERRIDES[seq_name]}")
    
    # 构建命令
    cmd = [
        "python", "tools/gen_macro_samples_euroc.py",
        "--euroc_root", EUROC_ROOT,
        "--seqs", seq_name,
        "--cam_id", "0",
        "--out_npz", temp_npz_path,
        "--patch", str(params["patch"]),
        "--K_tokens", str(params["K_tokens"]),
        "--min_matches", str(params["min_matches"]),
        "--pos_thr_px", f"{params['pos_thr_px']}",
        "--pos_thr_px_v1", f"{params['pos_thr_px_v1']}",
        "--pos_thr_px_v2", f"{params['pos_thr_px_v2']}",
        "--pos_thr_px_mh", f"{params['pos_thr_px_mh']}",
        "--err_clip_px", f"{params['err_clip_px']}",
        "--inlier_thr_px", f"{params['inlier_thr_px']}",
    ]
    
    # 根据模式添加不同的参数
    if params["kf_enable"]:
        cmd += [
            "--kf_enable",
            "--kf_parallax_px", str(params["kf_parallax_px"]),
            "--kf_max_interval_s", str(params["kf_max_interval_s"]),
            "--kf_min_interval_s", str(params["kf_min_interval_s"]),
            "--emit_non_kf_ratio", str(params["emit_non_kf_ratio"]),
            "--obs_min_parallax_px", str(params["obs_min_parallax_px"]),
        ]
        if params.get("kf_only", False):
            cmd += ["--kf_only"]
    else:
        cmd += [
            "--deltas"] + [str(d) for d in params["deltas"]] + [
            "--frame_step", str(params["frame_step"]),
        ]
    
    # 执行命令
    # 执行命令，捕获输出以便调试
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"\n\033[91m✗ {seq_name} 生成失败!\033[0m")
        print("---- STDOUT ----")
        print(res.stdout)
        print("---- STDERR ----")
        print(res.stderr)
        res.check_returncode()  # 抛出异常
    return temp_npz_path

def merge_npz_files(npz_paths, output_path, sequences):
    """合并多个npz文件，并保存完整的元数据"""
    all_patches, all_geoms, all_ytrue, all_numtok, all_yinlier = [], [], [], [], []
    
    print(f"  [DEBUG] 准备合并 {len(npz_paths)} 个文件:")
    for npz_path in npz_paths:
        if not Path(npz_path).exists():
            print(f"    ⚠️  文件不存在: {npz_path}")
            continue
        
        with np.load(npz_path, allow_pickle=True) as data:
            n_samples = data['patches'].shape[0]
            print(f"    ✓ {Path(npz_path).name}: {n_samples} 样本")
            
            all_patches.append(data['patches'].copy())
            all_geoms.append(data['geoms'].copy())
            all_ytrue.append(data['y_true'].copy())
            all_numtok.append(data['num_tokens'].copy())
            if 'y_inlier' in data:
                all_yinlier.append(data['y_inlier'].copy())
    
    # 合并数组
    if not all_patches:
        print(f"  ❌ 错误: 没有数据可合并!")
        return
    
    merged_patches = np.concatenate(all_patches, axis=0)
    merged_geoms = np.concatenate(all_geoms, axis=0)
    merged_ytrue = np.concatenate(all_ytrue, axis=0)
    merged_numtok = np.concatenate(all_numtok, axis=0)
    
    print(f"  [DEBUG] 合并后总样本数: {merged_patches.shape[0]}")
    
    # 构建完整的元数据（可追溯性++）
    meta = {
        "euroc_root": EUROC_ROOT,
        "sequences": sequences,
        "global_params": PARAMS.copy(),
        "seq_overrides": {s: SEQ_OVERRIDES.get(s, {}) for s in sequences},
        "generated_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_version": "v4_unified",
        "sources": [str(p) for p in npz_paths],
    }
    
    save_dict = {
        'patches': merged_patches,
        'geoms': merged_geoms,
        'y_true': merged_ytrue,
        'num_tokens': merged_numtok,
        'meta': meta,
    }
    
    if all_yinlier:
        merged_yinlier = np.concatenate(all_yinlier, axis=0)
        save_dict['y_inlier'] = merged_yinlier
    
    np.savez_compressed(output_path, **save_dict)

def generate_macro_data(set_name, sequences, output_npz):
    """生成宏观模式数据（支持序列级参数覆盖）"""
    print_header(f"生成 {set_name} 集", "green")
    print(f"序列: {', '.join(sequences)}")
    print(f"输出: {output_npz}\n")
    
    # 逐序列生成临时文件
    temp_dir = Path(OUT_DIR) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_npzs = []
    for seq in sequences:
        temp_npz = temp_dir / f"{seq}.npz"
        try:
            print(f"  正在生成 {seq}...")
            generate_one_sequence(seq, str(temp_npz))
            
            # 验证生成结果
            if Path(temp_npz).exists():
                with np.load(temp_npz, allow_pickle=True) as data:
                    n = data['patches'].shape[0]
                    print(f"    ✓ 生成成功: {n} 样本")
                temp_npzs.append(str(temp_npz))
            else:
                print(f"    ⚠️  文件未生成: {temp_npz}")
        except subprocess.CalledProcessError as e:
            print(f"\n\033[91m✗ {seq} 生成失败!\033[0m")
            return False
    
    # 合并所有临时文件
    print(f"\n  合并 {len(temp_npzs)} 个序列...")
    merge_npz_files(temp_npzs, output_npz, sequences)
    
    # 清理临时文件
    print(f"\n  清理 {len(temp_npzs)} 个临时文件...")
    for temp_npz in temp_npzs:
        if Path(temp_npz).exists():
            Path(temp_npz).unlink()
            print(f"    已删除: {Path(temp_npz).name}")
        else:
            print(f"    跳过(不存在): {Path(temp_npz).name}")
    
    print(f"\n\033[92m✓ {set_name}集生成成功!\033[0m\n")
    
    # 显示数据统计
    if os.path.exists(output_npz):
        print("\033[96m数据统计:\033[0m")
        try:
            with np.load(output_npz) as z:
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

