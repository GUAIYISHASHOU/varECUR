#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速检查宏观模式 NPZ 数据质量"""
import numpy as np
import argparse

def check(npz_path):
    print("="*70)
    print(f"检查宏观模式数据: {npz_path}")
    print("="*70)
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Meta 信息
    meta = data.get("meta", None)
    if meta is not None and isinstance(meta, np.ndarray):
        meta = meta.item()
    
    if meta:
        print(f"\n[Meta 信息]")
        print(f"  Pipeline:       {meta.get('pipeline', 'N/A')}")
        print(f"  Sequences:      {', '.join(meta.get('seqs', []))}")
        print(f"  Deltas:         {meta.get('deltas', 'N/A')}")
        print(f"  Patch:          {meta.get('patch', 'N/A')}x{meta.get('patch', 'N/A')}")
        print(f"  K_tokens:       {meta.get('K_tokens', 'N/A')}")
        print(f"  Frame step:     {meta.get('frame_step', 'N/A')}")
        print(f"  Min matches:    {meta.get('min_matches', 'N/A')}")
        print(f"  Pos thr (base): {meta.get('pos_thr_px', 'N/A')} px")
        print(f"    V1 upper:     {meta.get('pos_thr_px_v1', 'N/A')} px")
        print(f"    V2 upper:     {meta.get('pos_thr_px_v2', 'N/A')} px")
        print(f"    MH lower:     {meta.get('pos_thr_px_mh', 'N/A')} px")
        print(f"  Err clip:       {meta.get('err_clip_px', 'N/A')} px")
        print(f"  Geom dim:       {meta.get('geom_dim', 'N/A')}")
        print(f"  Y format:       {meta.get('y_format', 'N/A')}")
    else:
        print(f"\n⚠️  [WARN] No metadata found! (旧版本数据)")
    
    # 数据形状
    patches = data["patches"]
    geoms = data["geoms"]
    y_true = data["y_true"]
    num_tokens = data["num_tokens"]
    
    print(f"\n[数据形状]")
    print(f"  Samples:        {patches.shape[0]:,}")
    print(f"  Patches shape:  {patches.shape}  (M, K, 2, H, W)")
    print(f"  Geoms shape:    {geoms.shape}  (M, K, geom_dim)")
    print(f"  Y_true shape:   {y_true.shape}  (M, 2)")
    print(f"  Num tokens:     {num_tokens.shape}")
    
    # Y 标签统计 (log方差)
    print(f"\n[Y 标签统计] (log方差)")
    print(f"  LogVar X:")
    print(f"    Mean:   {np.mean(y_true[:, 0]):.3f}")
    print(f"    Median: {np.median(y_true[:, 0]):.3f}")
    print(f"    Std:    {np.std(y_true[:, 0]):.3f}")
    print(f"    Range:  [{np.min(y_true[:, 0]):.3f}, {np.max(y_true[:, 0]):.3f}]")
    
    print(f"  LogVar Y:")
    print(f"    Mean:   {np.mean(y_true[:, 1]):.3f}")
    print(f"    Median: {np.median(y_true[:, 1]):.3f}")
    print(f"    Std:    {np.std(y_true[:, 1]):.3f}")
    print(f"    Range:  [{np.min(y_true[:, 1]):.3f}, {np.max(y_true[:, 1]):.3f}]")
    
    # 转换为实际方差 (px²)
    var_x = np.exp(y_true[:, 0])
    var_y = np.exp(y_true[:, 1])
    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)
    
    print(f"\n[实际不确定度] (σ, 单位px)")
    print(f"  Sigma X:")
    print(f"    Mean:   {np.mean(sigma_x):.3f} px")
    print(f"    Median: {np.median(sigma_x):.3f} px")
    print(f"    P25:    {np.percentile(sigma_x, 25):.3f} px")
    print(f"    P75:    {np.percentile(sigma_x, 75):.3f} px")
    print(f"    P95:    {np.percentile(sigma_x, 95):.3f} px")
    
    print(f"  Sigma Y:")
    print(f"    Mean:   {np.mean(sigma_y):.3f} px")
    print(f"    Median: {np.median(sigma_y):.3f} px")
    print(f"    P25:    {np.percentile(sigma_y, 25):.3f} px")
    print(f"    P75:    {np.percentile(sigma_y, 75):.3f} px")
    print(f"    P95:    {np.percentile(sigma_y, 95):.3f} px")
    
    # Token 数量统计
    print(f"\n[Token 数量统计]")
    print(f"  Mean tokens:    {np.mean(num_tokens):.1f}")
    print(f"  Median:         {np.median(num_tokens):.0f}")
    print(f"  Min:            {np.min(num_tokens)}")
    print(f"  Max:            {np.max(num_tokens)}")
    print(f"  K (max):        {patches.shape[1]}")
    full_pct = np.sum(num_tokens == patches.shape[1]) / len(num_tokens) * 100
    print(f"  Full tokens:    {full_pct:.1f}%")
    
    # 各向异性分析
    aniso = y_true[:, 0] - y_true[:, 1]  # log(σx²) - log(σy²) = log(σx²/σy²)
    print(f"\n[各向异性分析] (log(σx²/σy²))")
    print(f"  Mean:           {np.mean(aniso):.3f}")
    print(f"  Std:            {np.std(aniso):.3f}")
    print(f"  Range:          [{np.min(aniso):.3f}, {np.max(aniso):.3f}]")
    iso_pct = np.sum(np.abs(aniso) < 0.5) / len(aniso) * 100
    print(f"  Near-isotropic: {iso_pct:.1f}% (|log ratio| < 0.5)")
    
    # 诊断
    print(f"\n[诊断]")
    if meta is None:
        print(f"  ⚠️  无 meta 信息，建议重新生成数据")
    
    mean_sigma = (np.mean(sigma_x) + np.mean(sigma_y)) / 2
    if mean_sigma < 0.5:
        print(f"  ✓ 平均不确定度 < 0.5px - 质量极高（可能过拟合）")
    elif mean_sigma < 1.0:
        print(f"  ✓ 平均不确定度 < 1.0px - 质量优秀")
    elif mean_sigma < 2.0:
        print(f"  ✓ 平均不确定度 < 2.0px - 质量良好")
    elif mean_sigma < 3.0:
        print(f"  ⚠️ 平均不确定度 2-3px - 质量尚可")
    else:
        print(f"  ❌ 平均不确定度 > 3px - 数据质量差")
        print(f"    -> 建议检查：pos_thr_px 是否太宽，err_clip_px 是否合理")
    
    if np.mean(num_tokens) < patches.shape[1] * 0.3:
        print(f"  ⚠️ 平均token数 < K*0.3，可能 min_matches 太低")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to macro NPZ file")
    args = ap.parse_args()
    check(args.npz)

