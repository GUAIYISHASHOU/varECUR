#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-sequence VIS pair NPZ files into train/val/test splits.
按序列分割，避免数据泄露。
"""
import argparse
import numpy as np
from pathlib import Path

# EuRoC sequence split (按建议划分)
SPLIT = {
    "train": [
        "V1_01_easy", "V1_02_medium", "V2_01_easy", "V2_02_medium",
        "MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_03_difficult"
    ],
    "val": ["MH_04_difficult"],
    "test": ["V2_03_difficult", "MH_05_difficult"],
}

def concat_npz(npzs, cap_per_seq=None, seed=0):
    """
    Concatenate multiple NPZ files.
    
    Args:
        npzs: List of NPZ file paths
        cap_per_seq: Maximum samples per sequence (None = no cap)
        seed: Random seed for sampling
    
    Returns:
        dict with concatenated arrays
    """
    I0 = I1 = geom = e2x = e2y = mask = None
    rng = np.random.default_rng(seed)
    
    for f in npzs:
        D = np.load(f)
        idx = np.arange(D["e2x"].shape[0])
        
        # Sample if exceeds cap
        if cap_per_seq and len(idx) > cap_per_seq:
            idx = rng.choice(idx, size=cap_per_seq, replace=False)
        
        def take(X):
            return X[idx]
        
        # Concatenate arrays
        I0 = take(D["I0"]) if I0 is None else np.concatenate([I0, take(D["I0"])], 0)
        I1 = take(D["I1"]) if I1 is None else np.concatenate([I1, take(D["I1"])], 0)
        geom = take(D["geom"]) if geom is None else np.concatenate([geom, take(D["geom"])], 0)
        e2x = take(D["e2x"]) if e2x is None else np.concatenate([e2x, take(D["e2x"])], 0)
        e2y = take(D["e2y"]) if e2y is None else np.concatenate([e2y, take(D["e2y"])], 0)
        mask = take(D["mask"]) if mask is None else np.concatenate([mask, take(D["mask"])], 0)
    
    return dict(I0=I0, I1=I1, geom=geom, e2x=e2x, e2y=e2y, mask=mask)

def main():
    ap = argparse.ArgumentParser("Merge VIS pairs by sequence split")
    ap.add_argument("--pairs_root", required=True, 
                   help="Directory containing per-sequence NPZ files")
    ap.add_argument("--out_root", required=True, 
                   help="Output directory for train/val/test NPZ files")
    ap.add_argument("--cap_per_seq", type=int, default=60000,
                   help="Max samples per sequence (for balancing)")
    ap.add_argument("--seed", type=int, default=0,
                   help="Random seed for sampling")
    args = ap.parse_args()
    
    pairs_root = Path(args.pairs_root)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"[merge] Processing splits...")
    print(f"  Pairs root: {pairs_root}")
    print(f"  Output root: {out}")
    print(f"  Cap per seq: {args.cap_per_seq}")
    print()
    
    for split, seqs in SPLIT.items():
        print(f"[{split}] Merging sequences: {seqs}")
        
        # Find existing NPZ files
        files = [
            pairs_root / f"{s}.npz" 
            for s in seqs 
            if (pairs_root / f"{s}.npz").exists()
        ]
        
        if not files:
            print(f"  [WARN] No files found for {split}")
            continue
        
        print(f"  Found {len(files)}/{len(seqs)} sequences")
        
        # Concatenate
        D = concat_npz(files, cap_per_seq=args.cap_per_seq, seed=args.seed)
        
        # Save
        out_file = out / f"{split}.npz"
        np.savez_compressed(
            out_file, 
            **D, 
            meta=dict(split=split, seqs=seqs, cap_per_seq=args.cap_per_seq)
        )
        
        print(f"  [OK] Wrote {out_file}")
        print(f"       Total pairs: {D['e2x'].shape[0]:,}")
        print(f"       Shape: I0={D['I0'].shape}, geom={D['geom'].shape}")
        print()
    
    print("[done] All splits merged successfully!")

if __name__ == "__main__":
    main()

