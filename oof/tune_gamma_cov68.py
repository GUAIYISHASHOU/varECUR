#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

# Add project root to path for utils_sa3
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils_sa3 import decompose_sa3_np, reconstruct_sa3_np
from metrics import studentt_z2_threshold, C68_GAUSS, C95_GAUSS


def compute_metrics_diag_axes(logv_axes: np.ndarray, e2: np.ndarray, dist: str, nu: float | None) -> dict:
    s2 = np.exp(logv_axes)  # (N,3)
    if dist == 'studentt' and (nu is not None) and (nu > 2.0):
        v_eff = s2 * (nu / (nu - 2.0))
    else:
        v_eff = s2
    z2 = (e2 / (v_eff + 1e-12)).sum(axis=-1) / 3.0
    if dist == 'studentt' and (nu is not None) and (nu > 2.0):
        thr68 = float(studentt_z2_threshold(0.68, float(nu), 3))
        thr95 = float(studentt_z2_threshold(0.95, float(nu), 3))
    else:
        thr68, thr95 = C68_GAUSS, C95_GAUSS
    cov68 = float((z2 <= thr68).mean())
    cov95 = float((z2 <= thr95).mean())
    z2_mean = float(z2.mean())
    return {"z2_mean": z2_mean, "cov68": cov68, "cov95": cov95}


def load_beta_s(base_cal_path: str) -> float:
    p = Path(base_cal_path)
    if p.exists():
        cal = json.load(open(p, 'r', encoding='utf-8'))
        sa = cal.get('sa_affine') or {}
        return float(sa.get('beta_s', 0.0))
    return 0.0


def main():
    ap = argparse.ArgumentParser(description='Sweep gamma (anisotropy) to target cov68 while keeping z2_mean close to 1')
    ap.add_argument('--oof_npz', type=str, default='runs/oof_acc/oof/oof_predictions.npz')
    ap.add_argument('--base_cal', type=str, default='runs/oof_acc/oof/calibrator_oof.json', help='Read beta_s from here if present')
    ap.add_argument('--out_cal', type=str, default='runs/oof_acc/oof/calibrator_oof_gamma.json')
    ap.add_argument('--out_report', type=str, default='runs/oof_acc/oof/gamma_sweep_report.json')
    ap.add_argument('--gamma_min', type=float, default=0.7)
    ap.add_argument('--gamma_max', type=float, default=1.1)
    ap.add_argument('--gamma_step', type=float, default=0.02)
    ap.add_argument('--target_cov68', type=float, default=0.68)
    ap.add_argument('--z2_lo', type=float, default=0.95)
    ap.add_argument('--z2_hi', type=float, default=1.05)
    args = ap.parse_args()

    npz = np.load(args.oof_npz, allow_pickle=True)
    if 'logv_axes' not in npz.files:
        raise RuntimeError('oof_npz has no logv_axes; this tool requires d_out==3 OOF predictions')
    logv_axes = npz['logv_axes']  # (N,3)
    e2 = npz['e2']                # (N,3)
    dist = str(npz['dist']) if ('dist' in npz.files) else 'gauss'
    nu = float(npz['nu']) if (dist == 'studentt' and 'nu' in npz.files) else None

    beta_s = load_beta_s(args.base_cal)
    print(f'[info] Using beta_s={beta_s:.6f} from {args.base_cal if Path(args.base_cal).exists() else "<none> (0.0)"}')

    s_pred, a_xy_pred, a_z_pred = decompose_sa3_np(logv_axes)

    gammas = np.arange(args.gamma_min, args.gamma_max + 1e-9, args.gamma_step)
    rows = []
    best_idx = None
    best_score = float('inf')

    for i, g in enumerate(gammas):
        lv_g = reconstruct_sa3_np(s_pred + beta_s, g * a_xy_pred, g * a_z_pred)
        met = compute_metrics_diag_axes(lv_g, e2, dist, nu)
        z2_ok = (args.z2_lo <= met['z2_mean'] <= args.z2_hi)
        score = abs(met['cov68'] - args.target_cov68) + (0.0 if z2_ok else 1e3)
        rows.append({
            'gamma': float(g),
            'z2_mean': met['z2_mean'],
            'cov68': met['cov68'],
            'cov95': met['cov95'],
            'z2_ok': bool(z2_ok),
            'score': float(score),
        })
        print(f"gamma: {g:.4f} | z2_mean: {met['z2_mean']:.4f} | cov68: {met['cov68']:.4f} | cov95: {met['cov95']:.4f} | z2_ok={z2_ok}")
        if score < best_score:
            best_score = score
            best_idx = i

    assert best_idx is not None
    best = rows[best_idx]
    g_star = float(best['gamma'])

    print('\n[best]')
    print(json.dumps(best, indent=2))

    # Save a calibrator JSON using the chosen gamma and fixed beta_s
    calibrator = {
        'axis_affine': None,
        'sa_affine': {
            'alpha_s': 1.0, 'beta_s': float(beta_s),
            'alpha_a_xy': g_star, 'beta_a_xy': 0.0,
            'alpha_a_z': g_star, 'beta_a_z': 0.0,
        },
        'mode': 'gamma_fixed_delta_s',
        'dist': 'studentt_diag_axes' if dist == 'studentt' else 'gauss_iso',
        'nu': (float(nu) if (dist == 'studentt' and nu is not None) else None),
    }

    Path(os.path.dirname(args.out_cal) or '.').mkdir(parents=True, exist_ok=True)
    with open(args.out_cal, 'w', encoding='utf-8') as f:
        json.dump(calibrator, f, indent=2)

    with open(args.out_report, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'best': best}, f, indent=2)


if __name__ == '__main__':
    main()
