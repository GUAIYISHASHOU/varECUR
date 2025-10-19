#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from metrics import studentt_z2_threshold, C68_GAUSS, C95_GAUSS


def winsorize(x: np.ndarray, p: float = 2.5) -> np.ndarray:
    lo, hi = np.percentile(x, [p, 100 - p])
    return np.clip(x, lo, hi)


def robust_std(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return float((q3 - q1) / 1.349)


def deming_fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xbar, ybar = x.mean(), y.mean()
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    disc = (sy2 - lam * sx2) ** 2 + 4.0 * lam * (sxy ** 2)
    slope = (sy2 - lam * sx2 + np.sqrt(disc)) / (2.0 * sxy + 1e-12)
    intercept = ybar - slope * xbar
    return float(slope), float(intercept)


def fit_linear(x: np.ndarray, y: np.ndarray, mode: str = "deming", lam: float = 1.0) -> tuple[float, float]:
    if mode == "std":
        sx = np.std(x)
        sy = np.std(y)
        a = float(sy / (sx + 1e-12))
        b = float(y.mean() - a * x.mean())
        return a, b
    elif mode == "robust":
        xw, yw = winsorize(x, 2.5), winsorize(y, 2.5)
        sx = robust_std(xw)
        sy = robust_std(yw)
        a = float(sy / (sx + 1e-12))
        b = float(y.mean() - a * x.mean())
        return a, b
    else:
        return deming_fit(x, y, lam)


def apply_axis_affine(logv_axes: np.ndarray, axis_params: list[dict]) -> np.ndarray:
    y = logv_axes.copy()
    for i in range(3):
        p = axis_params[i]
        y[:, i] = p['alpha'] * y[:, i] + p['beta']
    return y


def apply_sa_affine(logv_axes: np.ndarray, sa_params: dict) -> np.ndarray:
    from utils_sa3 import decompose_sa3_np, reconstruct_sa3_np
    s, a_xy, a_z = decompose_sa3_np(logv_axes)
    s_cal = sa_params['alpha_s'] * s + sa_params['beta_s']
    a_xy_cal = sa_params['alpha_a_xy'] * a_xy + sa_params['beta_a_xy']
    a_z_cal = sa_params['alpha_a_z'] * a_z + sa_params['beta_a_z']
    return reconstruct_sa3_np(s_cal, a_xy_cal, a_z_cal)


def compute_metrics_diag_axes(logv_axes: np.ndarray, e2: np.ndarray, dist: str, nu: float | None,
                              logv_min: float | None = None, logv_max: float | None = None) -> dict:
    if logv_min is not None and logv_max is not None:
        logv_axes = np.clip(logv_axes, logv_min, logv_max)
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
    v_iso = s2.mean(axis=-1)
    e2sum = e2.sum(axis=-1)
    ra = np.argsort(np.argsort(v_iso))
    rb = np.argsort(np.argsort(e2sum))
    sp = float(np.corrcoef(ra, rb)[0, 1]) if v_iso.size >= 3 else 0.0
    return {"z2_mean": z2_mean, "cov68": cov68, "cov95": cov95, "spearman": sp}


def main():
    ap = argparse.ArgumentParser(description='Two-stage calibration: axis-wise affine + SA-domain affine (OOF)')
    ap.add_argument('--oof_npz', type=str, default='runs/oof_acc/oof/oof_predictions.npz')
    ap.add_argument('--out_cal', type=str, default='runs/oof_acc/oof/calibrator_oof.json')
    ap.add_argument('--out_report', type=str, default='runs/oof_acc/oof/report_postcalib.json')
    ap.add_argument('--sa_mode', choices=['std','robust','deming'], default='deming')
    ap.add_argument('--deming_lambda', type=float, default=1.0)
    ap.add_argument('--logv_min', type=float, default=None)
    ap.add_argument('--logv_max', type=float, default=None)
    args = ap.parse_args()

    # Ensure project root on sys.path so utils_sa3 can be imported
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    # Late import after sys.path adjustment
    from utils_sa3 import decompose_sa3_np

    npz = np.load(args.oof_npz, allow_pickle=True)
    logv = npz['logv']                 # (N,)
    e2 = npz['e2']                     # (N,3)
    has_axes = ('logv_axes' in npz.files)
    logv_axes_pred = npz['logv_axes'] if has_axes else np.repeat(logv[:, None], 3, axis=1)
    dist = str(npz['dist']) if ('dist' in npz.files) else 'gauss'
    nu = float(npz['nu']) if (dist == 'studentt' and 'nu' in npz.files) else None

    # Stable two-parameter SA calibration: gamma for anisotropy (NLL grid via utils_sa3), Delta_s for scale (center z2 to 1)
    from utils_sa3 import tune_gamma_on_oof, reconstruct_sa3_np, decompose_sa3_np
    e_axes = np.sqrt(np.clip(e2, 0.0, None))
    gamma = tune_gamma_on_oof(logv_axes_pred, e_axes, gamma_grid=None, nu=(nu if (nu is not None) else 5.0))

    # Apply gamma on anisotropy, keep s unchanged
    s_pred, a_xy_pred, a_z_pred = decompose_sa3_np(logv_axes_pred)
    logv_axes_gamma = reconstruct_sa3_np(s_pred, gamma * a_xy_pred, gamma * a_z_pred)

    # Compute Delta_s to make mean z2 ≈ 1
    met_gamma = compute_metrics_diag_axes(logv_axes_gamma, e2, dist, nu, args.logv_min, args.logv_max)
    delta_s = float(np.log(max(met_gamma['z2_mean'], 1e-12)))

    # Final calibrated axes: add Delta_s uniformly on all axes via s-shift
    logv_axes_final = logv_axes_gamma + delta_s
    met = compute_metrics_diag_axes(logv_axes_final, e2, dist, nu, args.logv_min, args.logv_max)

    # Save calibrator JSON (sa_affine only)
    dist_for_apply = 'studentt_diag_axes' if dist == 'studentt' else 'gauss_iso'
    calibrator = {
        'axis_affine': None,
        'sa_affine': {
            'alpha_s': 1.0, 'beta_s': delta_s,
            'alpha_a_xy': float(gamma), 'beta_a_xy': 0.0,
            'alpha_a_z': float(gamma), 'beta_a_z': 0.0,
        },
        'mode': 'gamma_delta_s',
        'dist': dist_for_apply,
        'nu': (float(nu) if (dist == 'studentt' and nu is not None) else None),
    }
    if args.logv_min is not None:
        calibrator['logv_min'] = float(args.logv_min)
    if args.logv_max is not None:
        calibrator['logv_max'] = float(args.logv_max)

    Path(os.path.dirname(args.out_cal) or '.').mkdir(parents=True, exist_ok=True)
    with open(args.out_cal, 'w', encoding='utf-8') as f:
        json.dump(calibrator, f, indent=2)

    # Save report
    report = {
        'calibrator': calibrator,
        'z2_mean': met['z2_mean'],
        'cov68': met['cov68'],
        'cov95': met['cov95'],
        'spearman': met['spearman'],
        'count': int(logv.shape[0]),
    }
    with open(args.out_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
