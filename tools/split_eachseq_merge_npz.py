#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, hashlib
from pathlib import Path
import numpy as np

CORE_KEYS = ["TS_IMU","X_IMU_ACC","X_IMU_GYR","E2_IMU_ACC","E2_IMU_GYR","MASK_IMU"]

def load_npz(path, keys):
    d = np.load(path, allow_pickle=False)
    if not all(k in d for k in keys): return None
    return {k: d[k] for k in keys}

def concat_blocks(blocks):
    out = {}
    for k in blocks[0].keys():
        out[k] = np.concatenate([b[k] for b in blocks], axis=0)
    return out

def per_seq_split_idx(N, r_train, r_val, mode, seed):
    n_tr = int(np.floor(N*r_train))
    n_va = int(np.floor(N*r_val))
    n_te = N - n_tr - n_va
    if mode == "chronological":
        idx_tr = np.arange(0, n_tr)
        idx_va = np.arange(n_tr, n_tr+n_va)
        idx_te = np.arange(n_tr+n_va, N)
    else:  # shuffle
        rng = np.random.RandomState(seed)
        perm = rng.permutation(N)
        idx_tr = perm[:n_tr]
        idx_va = perm[n_tr:n_tr+n_va]
        idx_te = perm[n_tr+n_va:]
    return idx_tr, idx_va, idx_te, (n_tr, n_va, n_te)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="单序列 npz 所在目录（由 gen_euroc_step_npz.py 产出）")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--split", nargs=3, type=float, default=[0.70,0.15,0.15])
    ap.add_argument("--mode", choices=["chronological","shuffle"], default="chronological",
                    help="chronological 更能避免重叠窗口泄露；shuffle 随机打散窗口")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keys", nargs="*", default=CORE_KEYS)
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob(args.pattern))
    assert files, f"No npz in {root} with pattern {args.pattern}"

    r_tr, r_va, r_te = args.split
    assert abs(r_tr+r_va+r_te-1.0) < 1e-6, "split ratios must sum to 1"

    merged = {"train": [], "val": [], "test": []}
    manifest = {"per_sequence": {}, "split": args.split, "mode": args.mode}

    for f in files:
        pack = load_npz(f, args.keys)
        if pack is None:
            print(f"[skip] {f.name} missing keys {args.keys}")
            continue
        # 取该文件第一键的窗口数
        any_key = args.keys[0]
        N = pack[any_key].shape[0]

        # 为每个序列给个稳定 seed（避免不同序列同一洗牌）
        h = int(hashlib.md5(f.name.encode()).hexdigest(), 16) & 0xffffffff
        idx_tr, idx_va, idx_te, counts = per_seq_split_idx(N, r_tr, r_va, args.mode, (args.seed ^ h) & 0xffffffff)

        # 切片
        def slice_block(idxs):
            out = {}
            for k,arr in pack.items():
                out[k] = arr[idxs]
            return out

        tr_blk = slice_block(idx_tr)
        va_blk = slice_block(idx_va)
        te_blk = slice_block(idx_te)

        merged["train"].append(tr_blk)
        merged["val"].append(va_blk)
        merged["test"].append(te_blk)

        manifest["per_sequence"][f.name] = {
            "N_total": int(N),
            "counts": {"train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])}
        }
        print(f"[{f.name}] N={N} -> train={counts[0]} val={counts[1]} test={counts[2]}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train","val","test"]:
        assert merged[split], f"{split}: nothing to merge"
        big = concat_blocks(merged[split])
        np.savez(out_dir / f"{split}_all.npz", **big)
        manifest[f"{split}_windows"] = int(next(iter(big.values())).shape[0])
        print(f"[write] {split}_all.npz  windows={manifest[f'{split}_windows']}")

    with open(out_dir/"manifest.json","w",encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("manifest:", out_dir/"manifest.json")

if __name__ == "__main__":
    main()
