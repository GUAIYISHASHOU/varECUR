#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import argparse
import numpy as np
from metrics import studentt_z2_threshold, C68_GAUSS, C95_GAUSS


def get_metrics(logv_cal: np.ndarray, e2: np.ndarray, dist: str, nu: float | None,
                logv_min: float | None = None, logv_max: float | None = None) -> dict:
    """
    Compute z^2 mean and coverage after optional clamp, under Gaussian or Student-t.
    logv_cal: (N,) calibrated log-variance (isotropic)
    e2:       (N,3) squared error per axis
    dist:     'gauss' or 'studentt'
    nu:       dof if studentt else None
    """
    if logv_min is not None and logv_max is not None:
        logv_cal = np.clip(logv_cal, logv_min, logv_max)

    v = np.exp(logv_cal)
    e2sum = e2.sum(axis=-1)

    if dist == 'studentt' and (nu is not None) and (nu > 2.0):
        v_eff = v * (nu / (nu - 2.0))
        z2 = e2sum / (3.0 * v_eff)
        thr68 = float(studentt_z2_threshold(0.68, float(nu), 3))
        thr95 = float(studentt_z2_threshold(0.95, float(nu), 3))
    else:
        z2 = e2sum / (3.0 * v)
        thr68, thr95 = C68_GAUSS, C95_GAUSS

    cov68 = float((z2 <= thr68).mean())
    cov95 = float((z2 <= thr95).mean())
    z2_mean = float(z2.mean())

    # Rank correlation between variance magnitude and error magnitude
    ra = np.argsort(np.argsort(v))
    rb = np.argsort(np.argsort(e2sum))
    sp = float(np.corrcoef(ra, rb)[0, 1]) if (v.size >= 3) else 0.0

    return {"z2_mean": z2_mean, "cov68": cov68, "cov95": cov95, "spearman": sp}


def main():
    ap = argparse.ArgumentParser(description="Two-parameter affine calibration on isotropic log-variance: logv' = a*logv + b, search a and solve b(a) s.t. z2_mean≈1, then match cov68 target")
    ap.add_argument('--oof_npz', type=str, required=True, help='Path to OOF predictions (oof_predictions.npz)')
    ap.add_argument('--out_cal', type=str, required=True, help='Where to save affine calibrator (compatible with imu_oof/calibrator.py)')
    ap.add_argument('--out_report', type=str, required=True, help='Where to save post-calibration report JSON')
    ap.add_argument('--target_cov68', type=float, default=0.68, help='Target coverage at 68%%')
    ap.add_argument('--a_min', type=float, default=0.7, help='Min a in grid')
    ap.add_argument('--a_max', type=float, default=1.3, help='Max a in grid')
    ap.add_argument('--a_steps', type=int, default=121, help='Number of grid steps (inclusive)')
    ap.add_argument('--penalty_z2', type=float, default=0.0, help='Optional penalty weight for (z2_mean-1)^2')
    ap.add_argument('--logv_min', type=float, default=None, help='Optional clamp min used both offline and online')
    ap.add_argument('--logv_max', type=float, default=None, help='Optional clamp max used both offline and online')
    args = ap.parse_args()

    npz = np.load(args.oof_npz, allow_pickle=True)
    logv = npz['logv']
    e2 = npz['e2']
    dist_raw = str(npz['dist']) if ('dist' in npz.files) else 'gauss'
    is_studentt = ('studentt' in dist_raw)
    dist = 'studentt' if is_studentt else 'gauss'
    nu = float(npz['nu']) if (is_studentt and 'nu' in npz.files) else None

    e2sum = e2.sum(axis=-1)
    a_grid = np.linspace(args.a_min, args.a_max, args.a_steps)

    print("Searching a to match target cov68 with b(a) s.t. z2_mean≈1 ...")
    print(f"target={args.target_cov68:.4f}  grid=[{args.a_min:.3f},{args.a_max:.3f}] steps={args.a_steps}")
    print(f"dist={dist} nu={nu if nu is not None else 'N/A'}  clamp=({args.logv_min},{args.logv_max})")
    print(f"{'a':>8} | {'cov68':>8} | {'z2_mean':>8} | {'loss':>10}")
    print("-" * 44)

    best = (1e18, 1.0, 0.0, None)  # (loss, a, b, metrics)
    for i, a in enumerate(a_grid):
        mean_term = float((e2sum * np.exp(-a * logv)).mean())
        b = np.log(max(mean_term, 1e-12) / 3.0)
        if (dist == 'studentt') and (nu is not None) and (nu > 2.0):
            b += np.log((nu - 2.0) / nu)
        logv_cal = a * logv + b
        met = get_metrics(logv_cal, e2, dist, nu, args.logv_min, args.logv_max)
        loss = (met['cov68'] - args.target_cov68) ** 2 + args.penalty_z2 * (met['z2_mean'] - 1.0) ** 2
        if loss < best[0]:
            best = (loss, float(a), float(b), met)
        if (i == 0) or (i == len(a_grid) - 1) or (abs((a - args.a_min) / max(args.a_max - args.a_min, 1e-9) * 10 - round((a - args.a_min) / max(args.a_max - args.a_min, 1e-9) * 10)) < 1e-6):
            print(f"{a:8.3f} | {met['cov68']:8.4f} | {met['z2_mean']:8.4f} | {loss:10.6f}")

    best_a = best[1]
    best_b = best[2]
    best_met = best[3]
    print("-" * 44)
    print(f"Best a = {best_a:.6f}, b = {best_b:.6f}  -> cov68={best_met['cov68']:.4f} z2_mean={best_met['z2_mean']:.4f}")

    final_logv = best_a * logv + best_b
    final_met = get_metrics(final_logv, e2, dist, nu, args.logv_min, args.logv_max)

    # Save calibrator in affine form compatible with imu_oof/calibrator.py: logv' = a*logv + b
    dist_save = 'studentt_diag_axes' if dist == 'studentt' else 'gauss_iso'
    cal = {
        "a": float(best_a),
        "b": float(best_b),
        "dist": dist_save,
    }
    if dist == 'studentt' and (nu is not None):
        cal["nu"] = float(nu)
    if args.logv_min is not None:
        cal["logv_min"] = float(args.logv_min)
    if args.logv_max is not None:
        cal["logv_max"] = float(args.logv_max)

    os.makedirs(os.path.dirname(args.out_cal), exist_ok=True)
    with open(args.out_cal, 'w', encoding='utf-8') as f:
        json.dump(cal, f, indent=2)

    report = {
        "calibrator": {"a": float(best_a), "b": float(best_b)},
        "note": "Two-parameter affine calibration on isotropic log-variance",
        "target_cov68": float(args.target_cov68),
        "z2_mean": final_met['z2_mean'],
        "cov68": final_met['cov68'],
        "cov95": final_met['cov95'],
        "spearman": final_met['spearman'],
        "dist": dist,
        "nu": (float(nu) if (dist == 'studentt' and nu is not None) else None)
    }

    with open(args.out_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\nPost-calibration report:")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
