#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查V2_03为什么样本这么少"""
import subprocess

cmd = [
    "python", "tools/gen_macro_samples_euroc.py",
    "--euroc_root", "F:/SLAMdata/euroc",
    "--seqs", "V2_03_difficult",
    "--cam_id", "0",
    "--out_npz", "F:/SLAMdata/_cache/macro/temp/debug_V2_03.npz",
    "--patch", "32",
    "--K_tokens", "256",
    "--min_matches", "32",
    "--pos_thr_px", "5.0",
    "--pos_thr_px_v2", "5.0",
    "--err_clip_px", "20.0",
    "--inlier_thr_px", "3.0",
    "--kf_enable",
    "--kf_parallax_px", "18.0",
    "--kf_max_interval_s", "0.30",
    "--kf_min_interval_s", "0.08",
    "--emit_non_kf_ratio", "0.10",
    "--obs_min_parallax_px", "3.0",
]

print("检查V2_03_difficult生成过程...")
print("="*70)

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print("\n错误:")
    print(res.stderr)
