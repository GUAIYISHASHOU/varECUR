#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 - 只生成1个序列验证流程
用于测试环境是否正确配置
"""

import os
import sys
import subprocess
from pathlib import Path

EUROC_ROOT = "F:/SLAMdata/euroc"
TEST_SEQ = "MH_01_easy"
OUT_FILE = "F:/SLAMdata/_cache/vis_test/MH_01_easy.npz"

def main():
    print("=" * 50)
    print("VIS数据生成 - 快速测试")
    print("=" * 50)
    print(f"只生成1个序列: {TEST_SEQ}")
    print("用于验证环境配置")
    print("=" * 50)
    print()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    
    # 检查EuRoC数据
    seq_path = os.path.join(EUROC_ROOT, TEST_SEQ)
    if not os.path.exists(seq_path):
        print(f"错误: 找不到序列: {seq_path}")
        print(f"请确保EuRoC数据集已下载到 {EUROC_ROOT}")
        sys.exit(1)
    
    print(f"✓ 找到序列: {seq_path}")
    
    # 运行生成
    print("\n开始生成...")
    cmd = (
        f'python tools/gen_vis_pairs_euroc_strict.py '
        f'--euroc_root "{EUROC_ROOT}" '
        f'--seq {TEST_SEQ} '
        f'--delta 1 '
        f'--patch 32 '
        f'--max_pairs 5000 '
        f'--out_npz "{OUT_FILE}"'
    )
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        
        print("\n" + "=" * 50)
        print("✓ 测试成功！")
        print("=" * 50)
        print()
        print(f"生成文件: {OUT_FILE}")
        
        # 检查文件大小
        file_size = os.path.getsize(OUT_FILE) / (1024 * 1024)
        print(f"文件大小: {file_size:.1f} MB")
        
        print("\n环境配置正确！可以运行完整数据生成:")
        print("  python scripts/run_vis_data_generation.py")
        
    except subprocess.CalledProcessError:
        print("\n错误: 生成失败")
        print("请检查:")
        print("  1. Python包是否完整安装 (numpy, opencv-python, pyyaml, scikit-learn)")
        print("  2. EuRoC数据集是否完整")
        print("  3. 错误信息（见上方）")
        sys.exit(1)

if __name__ == "__main__":
    main()

