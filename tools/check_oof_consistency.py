#!/usr/bin/env python3
from __future__ import annotations
import os
import glob
import json
import argparse
import torch


def main():
    ap = argparse.ArgumentParser(description='Check consistency of OOF fold checkpoints')
    ap.add_argument('--runs_dir', type=str, default='runs/oof_acc')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.runs_dir, 'best_fold*.pt')))
    assert files, f'No best_fold*.pt found in {args.runs_dir}'

    metas = []
    for p in files:
        blob = torch.load(p, map_location='cpu')
        meta = {k: blob.get(k) for k in [
            'd_out','dist','nu','logv_min','logv_max','variance_param','label_version'
        ]}
        metas.append((p, meta))

    print(json.dumps({p: m for p, m in metas}, indent=2))

    keys = ['d_out','dist','nu','logv_min','logv_max','variance_param','label_version']
    base = metas[0][1]
    for p, m in metas[1:]:
        for k in keys:
            assert m.get(k) == base.get(k), f'Inconsistent {k} in {p}: {m.get(k)} vs {base.get(k)}'

    print('All folds consistent.')


if __name__ == '__main__':
    main()
