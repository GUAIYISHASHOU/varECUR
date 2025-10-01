#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-sequence NPZ chronological split and merge.
- Works for VIS-only, IMU-only, or BOTH. (auto-detectable)
- Accepts key aliases (E2≈E2_VIS, MASK≈MASK_VIS, TS≈TS_VIS, X≈X_VIS).
- For VIS, DF_VIS is optional (will be filled as 2.0 if missing).
- Produces train_all.npz / val_all.npz / test_all.npz and a manifest.json.

Usage example (your case, VIS only):
  python tools/split_eachseq_merge_npz.py \
    --root F:/SLAMdata/_cache/vis_eachseq \
    --out_dir F:/SLAMdata/_cache/vis_split \
    --pattern "*_vis.npz" --mode chronological --split 0.70 0.15 0.15 --route vis

"""

import argparse, glob, json, os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

VIS_KEYS = ["TS_VIS", "X_VIS", "E2_VIS", "MASK_VIS", "DF_VIS"]
IMU_KEYS = ["TS_IMU","X_IMU_ACC","X_IMU_GYR","E2_IMU_ACC","E2_IMU_GYR","MASK_IMU"]

ALIASES = {
    # VIS aliases
    "TS_VIS":   ["TS_VIS","TS"],
    "X_VIS":    ["X_VIS","X"],
    "E2_VIS":   ["E2_VIS","E2"],
    "MASK_VIS": ["MASK_VIS","MASK"],
    "DF_VIS":   ["DF_VIS"],
    # IMU aliases (keep strict)
    "TS_IMU":      ["TS_IMU"],
    "X_IMU_ACC":   ["X_IMU_ACC"],
    "X_IMU_GYR":   ["X_IMU_GYR"],
    "E2_IMU_ACC":  ["E2_IMU_ACC"],
    "E2_IMU_GYR":  ["E2_IMU_GYR"],
    "MASK_IMU":    ["MASK_IMU"],
}

def find_key(d: np.lib.npyio.NpzFile, canonical: str):
    for k in ALIASES.get(canonical, [canonical]):
        if k in d.files:
            return k
    return None

def detect_route(d: np.lib.npyio.NpzFile) -> str:
    has_vis = (find_key(d, "E2_VIS") is not None) or (find_key(d, "MASK_VIS") is not None)
    has_imu = (find_key(d, "X_IMU_ACC") is not None) or (find_key(d, "E2_IMU_ACC") is not None)
    if has_vis and has_imu: return "both"
    if has_vis: return "vis"
    if has_imu: return "imu"
    return "none"

def get_len_from_any(arrs: List[np.ndarray]) -> int:
    for a in arrs:
        if a is None: continue
        # accept 1D (T,) or 2D (T,D) or 3D (T,...) -> take first dim as T
        if a.ndim >= 1:
            return int(a.shape[0])
    return 0

def slice_by_mode(T: int, mode: str, split: Tuple[float,float,float]) -> Tuple[slice,slice,slice]:
    p_train, p_val, p_test = split
    n_tr = int(np.floor(T * p_train))
    n_va = int(np.floor(T * p_val))
    n_te = T - n_tr - n_va
    if mode == "chronological":
        i_tr = slice(0, n_tr)
        i_va = slice(n_tr, n_tr+n_va)
        i_te = slice(n_tr+n_va, T)
    else:
        # for reproducibility, random split should be done outside with perm index;
        # here we keep chronological for stability
        i_tr = slice(0, n_tr)
        i_va = slice(n_tr, n_tr+n_va)
        i_te = slice(n_tr+n_va, T)
    return i_tr, i_va, i_te

def ensure_2d(a: np.ndarray) -> np.ndarray:
    if a is None: return None
    if a.ndim == 1:
        return a.copy()
    return a

def normalize_vis_dict(d: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    key_e2   = find_key(d, "E2_VIS")
    key_m    = find_key(d, "MASK_VIS")
    key_ts   = find_key(d, "TS_VIS")
    key_x    = find_key(d, "X_VIS")
    key_df   = find_key(d, "DF_VIS")

    out = {}
    # minimal required for VIS: E2 or MASK (prefer both)
    if key_e2 is None and key_m is None:
        return {}  # not a VIS file

    if key_e2 is not None:
        out["E2_VIS"] = d[key_e2]
    if key_m is not None:
        out["MASK_VIS"] = d[key_m]
    if key_ts is not None:
        out["TS_VIS"] = d[key_ts]
    if key_x is not None:
        out["X_VIS"] = d[key_x]
    if key_df is not None:
        out["DF_VIS"] = d[key_df]

    # fill defaults
    T = get_len_from_any([out.get("E2_VIS"), out.get("MASK_VIS"), out.get("TS_VIS")])
    if T == 0:
        return {}

    if "MASK_VIS" not in out:
        out["MASK_VIS"] = np.ones((T,), dtype=np.float32)
    # DF_VIS default to 2.0 for (u,v)
    if "DF_VIS" not in out:
        out["DF_VIS"] = np.full((T,), 2.0, dtype=np.float32)

    # enforce shapes to be at least 1D with first dim T
    for k in list(out.keys()):
        out[k] = ensure_2d(out[k])
        # keep as-is (T,) or (T,D)
        if out[k].shape[0] != T:
            raise ValueError(f"[shape-mismatch] {k} shape {out[k].shape}, T={T}")
    return out

def normalize_imu_dict(d: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    # Optional: keep as-is; only include keys that exist
    out = {}
    for k in IMU_KEYS:
        kk = find_key(d, k)
        if kk is not None:
            out[k] = d[kk]
    # determine T from any IMU time series
    T = get_len_from_any([out.get("TS_IMU"), out.get("X_IMU_ACC"), out.get("X_IMU_GYR")])
    if T == 0:
        return {}
    # basic sanity
    for k in list(out.keys()):
        if out[k].shape[0] != T:
            raise ValueError(f"[shape-mismatch] {k} shape {out[k].shape}, T={T}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir of per-seq npz")
    ap.add_argument("--out_dir", required=True, help="output dir")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--mode", choices=["chronological","random"], default="chronological")
    ap.add_argument("--split", nargs=3, type=float, default=[0.70,0.15,0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route", choices=["auto","vis","imu","both"], default="auto",
                    help="which route keys are required. 'auto' detects per file.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not paths:
        print(f"[warn] no files matched: {args.root}/{args.pattern}")

    merged = {"train": {}, "val": {}, "test": {}}
    manifest = {"splits": {"train": [], "val": [], "test": []}}

    for p in paths:
        try:
            d = np.load(p, allow_pickle=False)
        except Exception as e:
            print(f"[skip] {Path(p).name} cannot open: {e}")
            continue

        want = args.route
        if want == "auto":
            want = detect_route(d)  # vis / imu / both / none

        vis_dict = normalize_vis_dict(d) if want in ("vis","both") else {}
        imu_dict = normalize_imu_dict(d) if want in ("imu","both") else {}

        if want == "vis" and not vis_dict:
            print(f"[skip] {Path(p).name} missing VIS keys (E2/MASK/TS).")
            continue
        if want == "imu" and not imu_dict:
            print(f"[skip] {Path(p).name} missing IMU keys.")
            continue
        if want == "both" and (not vis_dict or not imu_dict):
            print(f"[skip] {Path(p).name} needs both VIS and IMU keys, got VIS={bool(vis_dict)} IMU={bool(imu_dict)}.")
            continue
        if want == "none":
            print(f"[skip] {Path(p).name} none route detected.")
            continue

        # decide T based on available dict(s)
        T_vis = get_len_from_any([vis_dict.get("E2_VIS"), vis_dict.get("MASK_VIS"), vis_dict.get("TS_VIS")]) if vis_dict else 0
        T_imu = get_len_from_any([imu_dict.get("TS_IMU"), imu_dict.get("X_IMU_ACC"), imu_dict.get("X_IMU_GYR")]) if imu_dict else 0
        T = max(T_vis, T_imu)
        if T <= 1:
            print(f"[skip] {Path(p).name} T={T} too short.")
            continue

        i_tr, i_va, i_te = slice_by_mode(T, args.mode, tuple(args.split))

        def add_split(name: str, sub: Dict[str,np.ndarray], sl: slice):
            if not sub: return
            bucket = merged[name]
            for k, v in sub.items():
                vv = v[sl]
                if k not in bucket:
                    bucket[k] = [vv]
                else:
                    bucket[k].append(vv)

        add_split("train", vis_dict, i_tr)
        add_split("val",   vis_dict, i_va)
        add_split("test",  vis_dict, i_te)
        add_split("train", imu_dict, i_tr)
        add_split("val",   imu_dict, i_va)
        add_split("test",  imu_dict, i_te)

        manifest["splits"]["train"].append({"file": Path(p).name, "idx": [i_tr.start, i_tr.stop]})
        manifest["splits"]["val"].append({"file": Path(p).name, "idx": [i_va.start, i_va.stop]})
        manifest["splits"]["test"].append({"file": Path(p).name, "idx": [i_te.start, i_te.stop]})

    # save
    for split in ["train","val","test"]:
        if not merged[split]:
            print(f"[warn] {split}: nothing to merge")
            continue
        out = {}
        for k, arrs in merged[split].items():
            try:
                out[k] = np.concatenate(arrs, axis=0)
            except Exception:
                # fallback to list-save if shapes differ slightly
                out[k] = np.array(arrs, dtype=object)
        np.savez(os.path.join(args.out_dir, f"{split}_all.npz"), **out)
        print(f"[ok] wrote {split}_all.npz with keys: {list(out.keys())}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[ok] wrote manifest.json")
    print("[done]")
    
if __name__ == "__main__":
    main()
