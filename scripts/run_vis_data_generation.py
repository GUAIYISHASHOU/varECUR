#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIS数据生成完整流程 - Python版
自动生成所有序列 + 合并为train/val/test
"""

import os
import sys
import time
import subprocess
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可显示进度条 (pip install tqdm)\n")

# 配置
EUROC_ROOT = "F:/SLAMdata/euroc"
PAIRS_ROOT = "F:/SLAMdata/_cache/vis_pairs"
SPLIT_ROOT = "F:/SLAMdata/_cache/vis_split"

# 所有序列
SEQUENCES = [
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult"
]

def print_header(msg):
    print("\n" + "=" * 50)
    print(msg)
    print("=" * 50)

def run_command(cmd):
    """运行命令并返回是否成功"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print_header("VIS数据生成完整流程")
    print("步骤1: 生成11个序列的VIS pairs")
    print("步骤2: 合并为train/val/test")
    print()
    
    # 检查EuRoC数据集
    if not os.path.exists(EUROC_ROOT):
        print(f"错误: 找不到EuRoC数据集: {EUROC_ROOT}")
        print("请修改脚本中的 EUROC_ROOT 路径")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(PAIRS_ROOT, exist_ok=True)
    os.makedirs(SPLIT_ROOT, exist_ok=True)
    
    # 确认继续
    response = input("\n是否继续生成数据? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        sys.exit(0)
    
    # ========== 步骤1: 生成各序列 ==========
    print_header("步骤1: 生成各序列VIS pairs")
    
    completed = 0
    failed = []
    
    # 创建序列级进度条
    if HAS_TQDM:
        seq_iter = tqdm(enumerate(SEQUENCES, 1), total=len(SEQUENCES), 
                       desc="Overall progress", unit="seq")
    else:
        seq_iter = enumerate(SEQUENCES, 1)
    
    for i, seq in seq_iter:
        if not HAS_TQDM:
            print(f"\n[{i}/11] 处理 {seq} ...")
        
        # 检查序列是否存在
        seq_path = os.path.join(EUROC_ROOT, seq)
        if not os.path.exists(seq_path):
            print(f"  ⚠ 跳过: 序列不存在 {seq_path}")
            failed.append(seq)
            continue
        
        out_npz = os.path.join(PAIRS_ROOT, f"{seq}.npz")
        
        # 如果已存在，询问是否跳过
        if os.path.exists(out_npz):
            skip = input(f"  文件已存在: {out_npz}\n  是否跳过? (y/n): ")
            if skip.lower() == 'y':
                print(f"  → 跳过")
                completed += 1
                continue
        
        cmd = (
            f'python tools/gen_vis_pairs_euroc_strict.py '
            f'--euroc_root "{EUROC_ROOT}" '
            f'--seq {seq} '
            f'--delta 1 '
            f'--patch 32 '
            f'--max_pairs 60000 '
            f'--per_frame_cap 200 '
            f'--frame_step 2 '
            f'--min_frames 300 '
            f'--out_npz "{out_npz}"'
        )
        
        start_time = time.time()
        
        # 更新进度条描述
        if HAS_TQDM:
            seq_iter.set_description(f"Processing {seq}")
        
        success = run_command(cmd)
        elapsed = time.time() - start_time
        
        if success:
            if HAS_TQDM:
                tqdm.write(f"  ✓ {seq} 完成 (耗时: {elapsed:.1f}s)")
            else:
                print(f"  ✓ 完成 (耗时: {elapsed:.1f}s)")
            completed += 1
        else:
            if HAS_TQDM:
                tqdm.write(f"  ✗ {seq} 失败")
            else:
                print(f"  ✗ 失败")
            failed.append(seq)
    
    # 步骤1总结
    print_header(f"步骤1完成: {completed}/11 成功")
    if failed:
        print(f"失败序列: {', '.join(failed)}")
    print()
    
    if completed == 0:
        print("错误: 没有成功生成任何序列")
        sys.exit(1)
    
    # ========== 步骤2: 合并 ==========
    print_header("步骤2: 合并为train/val/test")
    
    cmd = (
        f'python tools/merge_vis_pairs_by_seq.py '
        f'--pairs_root "{PAIRS_ROOT}" '
        f'--out_root "{SPLIT_ROOT}" '
        f'--cap_per_seq 60000 '
        f'--seed 0'
    )
    
    success = run_command(cmd)
    
    if success:
        print_header("✓ 全部完成！")
        print("\n生成的数据:")
        print(f"  训练集: {SPLIT_ROOT}/train.npz")
        print(f"  验证集: {SPLIT_ROOT}/val.npz")
        print(f"  测试集: {SPLIT_ROOT}/test.npz")
        print("\n下一步: 训练模型")
        print(f"  python train_vis.py --train_npz {SPLIT_ROOT}/train.npz --val_npz {SPLIT_ROOT}/val.npz")
    else:
        print("\n错误: 合并失败")
        sys.exit(1)

if __name__ == "__main__":
    main()

