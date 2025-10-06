#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成增强版VIS数据（所有EuRoC序列）
"""
import os
import subprocess
from pathlib import Path

# 配置
EUROC_ROOT = "F:/SLAMdata/euroc"
PAIRS_ROOT = "F:/SLAMdata/_cache/vis_pairs"

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

# 生成参数（增强版）
PARAMS = {
    "deltas": "1,2",
    "patch": 32,
    "max_pairs": 60000,
    "per_frame_cap": 800,
    "frame_step": 2,
    "min_frames": 300,
    "err_clip_px": 15,
    "depth_min": 0.1,
    "depth_max": 80,
    "epi_thr_px": 1.5,
    "texture_strat": True,
}

def main():
    print("=" * 60)
    print("批量生成增强版VIS数据")
    print("=" * 60)
    print(f"序列数量: {len(SEQUENCES)}")
    print(f"增强特性: 多delta({PARAMS['deltas']}), 纹理分层, Sampson过滤, 11维特征")
    print()
    
    # 检查已存在的文件
    existing = []
    for seq in SEQUENCES:
        out_file = Path(PAIRS_ROOT) / f"{seq}.npz"
        if out_file.exists():
            existing.append(seq)
    
    if existing:
        print(f"已存在 {len(existing)} 个文件:")
        for seq in existing:
            print(f"  - {seq}.npz")
        response = input("\n是否跳过已存在文件? (y/n): ")
        skip_existing = (response.lower() == 'y')
    else:
        skip_existing = False
    
    print("\n开始生成...")
    print("-" * 60)
    
    success_count = 0
    failed = []
    skipped = []
    
    for i, seq in enumerate(SEQUENCES, 1):
        print(f"\n[{i}/{len(SEQUENCES)}] 处理 {seq}...")
        
        out_npz = Path(PAIRS_ROOT) / f"{seq}.npz"
        
        # 跳过已存在
        if skip_existing and out_npz.exists():
            print(f"  ✓ 跳过（文件已存在）")
            skipped.append(seq)
            continue
        
        # 构建命令
        cmd = [
            "python", "tools/gen_vis_pairs_euroc_strict.py",
            "--euroc_root", EUROC_ROOT,
            "--seq", seq,
            "--deltas", PARAMS["deltas"],
            "--patch", str(PARAMS["patch"]),
            "--max_pairs", str(PARAMS["max_pairs"]),
            "--per_frame_cap", str(PARAMS["per_frame_cap"]),
            "--frame_step", str(PARAMS["frame_step"]),
            "--min_frames", str(PARAMS["min_frames"]),
            "--err_clip_px", str(PARAMS["err_clip_px"]),
            "--depth_min", str(PARAMS["depth_min"]),
            "--depth_max", str(PARAMS["depth_max"]),
            "--epi_thr_px", str(PARAMS["epi_thr_px"]),
            "--out_npz", str(out_npz)
        ]
        
        if PARAMS.get("texture_strat"):
            cmd.append("--texture_strat")
        
        # 执行
        try:
            result = subprocess.run(cmd, check=True, text=True)
            print(f"  ✓ {seq} 成功")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {seq} 失败")
            failed.append(seq)
        except KeyboardInterrupt:
            print(f"\n\n⚠ 用户中断")
            print(f"已完成: {success_count}/{len(SEQUENCES)}")
            return
    
    # 总结
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"跳过: {len(skipped)}")
    print(f"失败: {len(failed)}")
    
    if failed:
        print(f"\n失败序列: {', '.join(failed)}")
    
    if success_count + len(skipped) == len(SEQUENCES):
        print("\n✓ 所有序列已准备就绪！")
        print(f"\n下一步: 合并数据")
        print(f"  python tools/merge_vis_pairs_by_seq.py \\")
        print(f"    --pairs_root {PAIRS_ROOT} \\")
        print(f"    --out_root F:/SLAMdata/_cache/vis_split")

if __name__ == "__main__":
    main()

