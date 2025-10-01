#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, numpy as np
from pathlib import Path

def fix_file(path):
    d = np.load(path)
    if "E2_VIS" not in d:
        print("[skip] no E2_VIS:", path); return
    e2 = d["E2_VIS"].astype(np.float32)
    bad = (~np.isfinite(e2)) | (e2 >= 999.0)
    if "MASK_VIS" in d:
        mask = (d["MASK_VIS"] > 0.5).astype(np.float32)
    else:
        mask = np.ones_like(e2, dtype=np.float32)
    if bad.any() or "MASK_VIS" not in d:
        mask[bad] = 0.0
        e2[bad] = np.nan
        out = {k: d[k] for k in d.files if k not in {"E2_VIS","MASK_VIS"}}
        out["E2_VIS"] = e2
        out["MASK_VIS"] = mask.astype(np.float32)
        np.savez(path, **out)
        print(f"[fix] {Path(path).name}: bad={int(bad.sum())}/{e2.size}")
    else:
        print(f"[ok ] {Path(path).name}: no change")

def main(root):
    for p in glob.glob(os.path.join(root, "*.npz")):
        fix_file(p)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dir of per-seq VIS npz")
    args = ap.parse_args()
    main(args.root)


