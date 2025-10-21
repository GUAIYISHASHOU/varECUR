#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import glob
import json
import argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from metrics import studentt_z2_threshold, C68_GAUSS, C95_GAUSS


def spearmanr_np(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description='Aggregate OOF fold predictions and compute metrics')
    ap.add_argument('--in_dir', type=str, default='runs/oof_acc/oof')
    ap.add_argument('--out_json', type=str, default='runs/oof_acc/oof/report_precalib.json')
    ap.add_argument('--out_npz', type=str, default='runs/oof_acc/oof/oof_predictions.npz')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, 'fold*_pred.npz')))
    assert files, f'No fold*_pred.npz found in {args.in_dir}'

    logv_list, e2_list, logv_axes_list = [], [], []
    dist, nu = None, None
    for f in files:
        npz = np.load(f)
        logv_list.append(npz['logv'])          # (N,)
        e2_list.append(npz['e2'])              # (N,3)
        if 'logv_axes' in npz.files:
            logv_axes_list.append(npz['logv_axes'])
        meta = json.loads(str(npz['meta']))
        dist = dist or meta.get('dist', 'gauss')
        nu = nu or meta.get('nu', None)

    logv = np.concatenate(logv_list)
    e2 = np.concatenate(e2_list, axis=0)
    has_axes = len(logv_axes_list) > 0
    if has_axes:
        logv_axes = np.concatenate(logv_axes_list, axis=0)
        assert logv_axes.shape[0] == logv.shape[0], "logv_axes length mismatch with logv"
    # sanitize invalid entries (NaN/Inf) before computing metrics
    valid = np.isfinite(logv) & np.all(np.isfinite(e2), axis=-1)
    if has_axes:
        valid = valid & np.all(np.isfinite(logv_axes), axis=-1)
    if not np.all(valid):
        logv = logv[valid]
        e2 = e2[valid]
        if has_axes:
            logv_axes = logv_axes[valid]
    # ensure non-negative errors
    e2 = np.clip(e2, 0.0, None)
    e2sum = e2.sum(axis=-1)
    v = np.exp(logv)

    if dist == 'studentt':
        nu = float(nu if nu is not None else 5.0)
        nll = 0.5*(3*logv) + 0.5*(nu+3.0)*np.log1p(e2sum/(nu*v))
        v_eff = v * (nu/(nu-2.0))
        z2 = e2sum / (3.0 * v_eff)
    else:
        nll = 0.5*(3*logv) + 0.5*(e2sum / v)
        z2 = e2sum / (3.0 * v)

    if dist == 'studentt':
        thr68 = float(studentt_z2_threshold(0.68, nu, 3))
        thr95 = float(studentt_z2_threshold(0.95, nu, 3))
    else:
        thr68, thr95 = C68_GAUSS, C95_GAUSS
    cov68 = float((z2 <= thr68).mean())
    cov95 = float((z2 <= thr95).mean())
    sp = spearmanr_np(np.exp(logv), e2sum)

    out = dict(
        nll=float(nll.mean()),
        z2_mean=float(z2.mean()),
        cov68=cov68,
        cov95=cov95,
        spearman=sp,
        count=int(logv.shape[0]),
        dist=dist,
        nu=(float(nu) if dist == 'studentt' else None),
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    # also dump merged arrays for calibration (optionally with logv_axes)
    save_dict = {'logv': logv, 'e2': e2, 'dist': dist}
    if dist == 'studentt':
        save_dict['nu'] = float(nu)
    if has_axes:
        save_dict['logv_axes'] = logv_axes
    np.savez(args.out_npz, **save_dict)


if __name__ == '__main__':
    main()
