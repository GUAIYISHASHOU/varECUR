#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并同一序列的多个 delta NPZ 文件（如 _d1.npz, _d2.npz）为单个文件
"""
import os
import glob
import numpy as np
import argparse
from pathlib import Path

def concat_one(root, seq):
    """
    合并同一序列的所有 _d*.npz 文件
    
    Args:
        root: 目录路径
        seq: 序列名（如 "MH_01_easy"）
    """
    fs = sorted(glob.glob(os.path.join(root, f"{seq}_d*.npz")))
    if not fs:
        print(f"[skip] {seq}: no _d*.npz files found")
        return
    
    print(f"[{seq}] Found {len(fs)} delta files: {[Path(f).name for f in fs]}")
    
    packs = [np.load(f, allow_pickle=True) for f in fs]
    
    # 找所有文件共有的键（排除 meta）
    keys = set(packs[0].files)
    for p in packs[1:]:
        keys &= set(p.files)
    
    # meta 单独处理
    data_keys = sorted(k for k in keys if k not in {"meta"})
    
    # 拼接数据
    out = {}
    for k in data_keys:
        arrays = [p[k] for p in packs]
        try:
            out[k] = np.concatenate(arrays, axis=0)
        except Exception as e:
            print(f"  [error] Failed to concatenate '{k}': {e}")
            shapes = [a.shape for a in arrays]
            print(f"          Shapes: {shapes}")
            raise
    
    # 合并 meta（优先使用第一个文件的 meta，更新关键字段）
    meta = {}
    m0 = packs[0].get("meta", None)
    if m0 is not None:
        meta = m0.item() if isinstance(m0, np.ndarray) else dict(m0)
    
    # 确保包含 pipeline 标识，检查是否有 RANSAC
    if "pipeline" not in meta:
        # 检查所有源文件是否包含 RANSAC
        has_ransac = any(
            (p.get("meta").item() if isinstance(p.get("meta"), np.ndarray) else p.get("meta") or {}).get("ransac")
            for p in packs if p.get("meta") is not None
        )
        meta["pipeline"] = "strict_euroc_v2_ransac" if has_ransac else "strict_euroc_v1"
    
    # 收集所有 deltas
    all_deltas = []
    for p in packs:
        m = p.get("meta", None)
        if m is not None:
            m = m.item() if isinstance(m, np.ndarray) else m
            d = m.get("deltas", m.get("delta", None))
            if d is not None:
                if isinstance(d, list):
                    all_deltas.extend(d)
                else:
                    all_deltas.append(d)
    
    if all_deltas:
        meta["deltas"] = sorted(set(all_deltas))
    
    meta["note"] = f"concatenated from {len(fs)} delta files"
    meta["source_files"] = [Path(f).name for f in fs]
    
    # 保存合并后的文件
    out_path = os.path.join(root, f"{seq}.npz")
    np.savez_compressed(out_path, **out, meta=meta)
    
    total_samples = out[data_keys[0]].shape[0] if data_keys else 0
    print(f"  [OK] → {seq}.npz ({total_samples:,} total pairs)")
    print(f"       Pipeline: {meta.get('pipeline', 'unknown')}")
    print(f"       Deltas: {meta.get('deltas', 'N/A')}")

def main():
    ap = argparse.ArgumentParser("Concatenate per-sequence delta NPZ files")
    ap.add_argument("--root", required=True, help="Directory containing _d*.npz files")
    ap.add_argument("--seqs", nargs="+", required=True, help="Sequence names to process")
    args = ap.parse_args()
    
    print("="*60)
    print("合并同序列 delta 文件")
    print("="*60)
    print(f"Root: {args.root}")
    print(f"Sequences: {len(args.seqs)}")
    print()
    
    success = 0
    for s in args.seqs:
        try:
            concat_one(args.root, s)
            success += 1
        except Exception as e:
            print(f"[ERROR] {s}: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("="*60)
    print(f"完成: {success}/{len(args.seqs)} 序列")
    print("="*60)

if __name__ == "__main__":
    main()
