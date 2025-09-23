#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ Sanity Checker (IMU per-timestep labels)
--------------------------------------------
- Lists keys, shapes, dtypes, NaN/Inf counts
- Validates alignment among X/E2/MASK/TS (per-window & per-timestep)
- Checks timestamp monotonicity; estimates dt (Hz) and window stride (samples)
- Detects near-constant labels across time (broadcast-style)
- Optional plots: histograms & short time-series previews for E2

Usage:
  python tools/npz_sanity_check.py --file F:\path\to\V1_01_easy_T512_S256.npz --plots
  python tools/npz_sanity_check.py --glob "F:\SLAMdata\_cache\euroc_step\*.npz" --plots
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ------------------------- small utils -------------------------

def safe_stats(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr)
    finite = np.isfinite(a)
    n = a.size
    stats = {
        "size": int(n),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "nan_count": int(np.isnan(a).sum()),
        "inf_count": int(np.isinf(a).sum()),
        "finite_ratio": float(finite.sum() / max(1, n)),
    }
    if finite.any():
        fa = a[finite]
        mn, mx = float(fa.min()), float(fa.max())
        mean, std = float(fa.mean()), float(fa.std())
        cv = float(std / (abs(mean) + 1e-12))
        stats.update({"min": mn, "max": mx, "mean": mean, "std": std, "cv": cv})
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "cv": None})
    return stats


def time_monotonic(ts_win: np.ndarray) -> Dict[str, Any]:
    """
    ts_win: (T,) or (N,T). If (N,T), check each window and aggregate.
    """
    t = np.asarray(ts_win)
    out = {}
    if t.ndim == 1:
        diffs = np.diff(t.astype(np.float64))
        out["is_strictly_increasing"] = bool(np.all(diffs > 0))
        out["min_diff"] = float(diffs.min()) if diffs.size else None
        out["max_diff"] = float(diffs.max()) if diffs.size else None
        out["mean_diff"] = float(diffs.mean()) if diffs.size else None
        out["num_nonpos_steps"] = int(np.sum(diffs <= 0))
    elif t.ndim == 2:
        N = t.shape[0]
        bad = 0
        dts = []
        for i in range(N):
            diffs = np.diff(t[i].astype(np.float64))
            if not np.all(diffs > 0):
                bad += 1
            if diffs.size:
                dts.append([diffs.min(), diffs.max(), diffs.mean()])
        dts = np.array(dts) if len(dts) else np.zeros((0,3))
        out["windows"] = int(N)
        out["strict_increasing_frac"] = float((N - bad) / max(1, N))
        if dts.size:
            out["min_diff_median"] = float(np.median(dts[:,0]))
            out["max_diff_median"] = float(np.median(dts[:,1]))
            out["mean_diff_median"] = float(np.median(dts[:,2]))
        else:
            out["min_diff_median"] = out["max_diff_median"] = out["mean_diff_median"] = None
    else:
        out["error"] = "TS must be 1D or 2D"
    return out


def infer_stride_samples(TS_IMU: np.ndarray) -> int:
    """
    Estimate stride in samples:
      stride_samples ≈ median(diff(window_starts)) / median(in-window dt)
    """
    ts = np.asarray(TS_IMU)
    if ts.ndim != 2 or ts.shape[0] < 2:
        return ts.shape[1] if ts.ndim == 2 else 0
    starts = ts[:, 0].astype(np.float64)
    dt_start = np.median(np.diff(starts)) if len(starts) > 1 else 0.0
    dt_in = np.median(np.diff(ts[0].astype(np.float64))) if ts.shape[1] > 1 else 0.0
    if dt_in <= 0:
        return 0
    stride = int(round(dt_start / dt_in))
    stride = max(1, min(stride, ts.shape[1]))
    return stride


def summarize_e2(E2: np.ndarray) -> Dict[str, Any]:
    """
    E2: (N,T,3) or (N,T) or (T,3).
    We compute:
      - global stats over all finite values
      - per-window variance across time (using sum over axes if 3)
      - fraction of windows with near-zero time variance (broadcast-like)
      - quantiles of flattened E2_sum
    """
    A = np.asarray(E2)
    if A.ndim == 3 and A.shape[-1] == 3:
        e2_sum = A.sum(axis=-1)  # (N,T)
    elif A.ndim == 2:
        e2_sum = A
    else:
        # fallback: flatten last dim
        e2_sum = A.reshape(A.shape[0], -1) if A.ndim >= 2 else A[None, :]

    finite = np.isfinite(e2_sum)
    vals = e2_sum[finite]
    stats = {
        "size": int(e2_sum.size),
        "nan_count": int(np.isnan(e2_sum).sum()),
        "inf_count": int(np.isinf(e2_sum).sum()),
    }
    if vals.size:
        stats.update({
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "cv": float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)),
            "q50": float(np.quantile(vals, 0.50)),
            "q68": float(np.quantile(vals, 0.68)),
            "q95": float(np.quantile(vals, 0.95)),
            "q99": float(np.quantile(vals, 0.99)),
        })
    else:
        stats.update({k: None for k in ["min","max","mean","std","cv","q50","q68","q95","q99"]})

    # time-variance per window
    if e2_sum.ndim == 2:
        var_t = np.var(e2_sum, axis=1)
        frac_const = float(np.mean(var_t < 1e-8))
        stats.update({
            "time_var_median": float(np.median(var_t)),
            "frac_windows_near_constant": frac_const
        })
    return stats


# ------------------------- analyzer -------------------------

def analyze_file(path: Path, make_plots: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"file": str(path), "exists": path.exists()}
    if not path.exists():
        return out

    with np.load(path, allow_pickle=False) as d:
        keys = list(d.keys())
        out["keys"] = keys

        # Key stats
        key_stats = {k: safe_stats(d[k]) for k in keys}
        out["key_stats"] = key_stats

        warns: List[str] = []

        # Expected IMU keys (per-timestep schema)
        req = ["TS_IMU","X_IMU_ACC","X_IMU_GYR","E2_IMU_ACC","E2_IMU_GYR","MASK_IMU"]
        missing = [k for k in req if k not in d]
        out["missing_keys"] = missing
        if missing:
            warns.append(f"missing keys: {missing}")

        # Shape alignment
        shapes = {k: tuple(d[k].shape) for k in keys}
        out["shapes"] = {k: list(v) for k,v in shapes.items()}

        def shape_of(k): return tuple(d[k].shape) if k in d else None

        Xacc, Xgyr = shape_of("X_IMU_ACC"), shape_of("X_IMU_GYR")
        E2a, E2g = shape_of("E2_IMU_ACC"), shape_of("E2_IMU_GYR")
        TS = shape_of("TS_IMU")
        MSK = shape_of("MASK_IMU")

        # Basic alignment checks
        if Xacc and E2a and (Xacc != E2a):
            warns.append(f"shape mismatch: X_IMU_ACC {Xacc} vs E2_IMU_ACC {E2a}")
        if Xgyr and E2g and (Xgyr != E2g):
            warns.append(f"shape mismatch: X_IMU_GYR {Xgyr} vs E2_IMU_GYR {E2g}")
        if TS and Xacc and (TS[0] != Xacc[0] or TS[1] != Xacc[1]):
            warns.append(f"shape mismatch: TS_IMU {TS} vs X_IMU_ACC {Xacc} (N,T should match)")
        if MSK and Xacc and (MSK[0] != Xacc[0] or MSK[1] != Xacc[1]):
            warns.append(f"shape mismatch: MASK_IMU {MSK} vs X_IMU_ACC {Xacc} (N,T should match)")

        # Mask values
        if "MASK_IMU" in d:
            m = d["MASK_IMU"]
            uniq = np.unique(m)
            if not set(uniq.tolist()).issubset({0,1}):
                warns.append(f"MASK_IMU contains values other than 0/1: {uniq.tolist()}")
            if m.mean() == 0.0:
                warns.append("MASK_IMU is all zeros")

        # Timestamp checks & rates
        time_info = {}
        if "TS_IMU" in d:
            time_info["per_window"] = time_monotonic(d["TS_IMU"])
            if d["TS_IMU"].ndim == 2 and d["TS_IMU"].shape[1] >= 2:
                dt = float(np.median(np.diff(d["TS_IMU"][0].astype(np.float64))))
                hz = float(1.0/dt) if dt > 0 else None
                T = int(d["TS_IMU"].shape[1])
                stride = infer_stride_samples(d["TS_IMU"])
                time_info["dt_median"] = dt
                time_info["Hz_median"] = hz
                time_info["T"] = T
                time_info["stride_samples_est"] = stride
        out["time_info"] = time_info

        # E2 summaries
        e2_info = {}
        if "E2_IMU_ACC" in d:
            e2_info["E2_IMU_ACC"] = summarize_e2(d["E2_IMU_ACC"])
        if "E2_IMU_GYR" in d:
            e2_info["E2_IMU_GYR"] = summarize_e2(d["E2_IMU_GYR"])
        out["e2_info"] = e2_info

        # Broadcast-like warning
        for k,info in e2_info.items():
            frac_const = info.get("frac_windows_near_constant", 0.0)
            if frac_const is not None and frac_const > 0.5:
                warns.append(f"{k}: {frac_const*100:.1f}% windows near-constant over time (possible broadcast)")

        # NaN/Inf quick warn
        bad_fin = [k for k,s in key_stats.items() if s["nan_count"] or s["inf_count"]]
        if bad_fin:
            warns.append(f"NaN/Inf present in: {bad_fin}")

        out["warnings"] = warns

        # Optional plots
        if make_plots and plt is not None:
            out_dir = path.parent
            stem = path.stem

            # Histogram of E2 sum (ACC)
            if "E2_IMU_ACC" in d:
                e2a = d["E2_IMU_ACC"]
                vals = e2a.sum(axis=-1).reshape(-1)  # (N*T,)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    plt.figure()
                    plt.hist(vals, bins=80)
                    plt.title(f"E2_IMU_ACC sum-over-axes (flattened)")
                    plt.xlabel("E2_acc_sum")
                    plt.ylabel("count")
                    png1 = out_dir / f"{stem}_E2acc_hist.png"
                    plt.savefig(png1, dpi=150, bbox_inches="tight")
                    plt.close()
                    out["plot_E2acc_hist"] = str(png1)

            # Tiny time-series (first window) for ACC and GYR
            if "E2_IMU_ACC" in d:
                e2a = d["E2_IMU_ACC"]
                plt.figure()
                plt.plot(e2a[0,:,0], label="acc_x")
                plt.plot(e2a[0,:,1], label="acc_y")
                plt.plot(e2a[0,:,2], label="acc_z")
                plt.legend(); plt.title("E2_IMU_ACC (window 0)")
                plt.xlabel("t index"); plt.ylabel("E2")
                png2 = out_dir / f"{stem}_E2acc_ts.png"
                plt.savefig(png2, dpi=150, bbox_inches="tight")
                plt.close()
                out["plot_E2acc_ts"] = str(png2)

            if "E2_IMU_GYR" in d:
                e2g = d["E2_IMU_GYR"]
                plt.figure()
                plt.plot(e2g[0,:,0], label="gyr_x")
                plt.plot(e2g[0,:,1], label="gyr_y")
                plt.plot(e2g[0,:,2], label="gyr_z")
                plt.legend(); plt.title("E2_IMU_GYR (window 0)")
                plt.xlabel("t index"); plt.ylabel("E2")
                png3 = out_dir / f"{stem}_E2gyr_ts.png"
                plt.savefig(png3, dpi=150, bbox_inches="tight")
                plt.close()
                out["plot_E2gyr_ts"] = str(png3)

    # Sidecar files
    rep_lines = []
    rep_lines.append(f"File: {path}")
    rep_lines.append("")
    rep_lines.append("== Keys ==")
    rep_lines.append(", ".join(out.get("keys", [])))
    rep_lines.append("")
    rep_lines.append("== Shapes ==")
    for k,sh in out.get("shapes", {}).items():
        rep_lines.append(f"{k}: {sh}")
    rep_lines.append("")
    rep_lines.append("== Time info ==")
    rep_lines.append(json.dumps(out.get("time_info", {}), indent=2))
    rep_lines.append("")
    rep_lines.append("== E2 summaries ==")
    rep_lines.append(json.dumps(out.get("e2_info", {}), indent=2))
    rep_lines.append("")
    rep_lines.append("== Warnings ==")
    for w in out.get("warnings", []):
        rep_lines.append(f"- {w}")

    try:
        report_path = path.with_name(path.stem + "_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rep_lines))
        out["report"] = str(report_path)
    except Exception as e:
        out["report_error"] = repr(e)

    try:
        summary_path = path.with_name(path.stem + "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        out["summary"] = str(summary_path)
    except Exception as e:
        out["summary_error"] = repr(e)

    return out


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, help="Path to a single NPZ file")
    ap.add_argument("--glob", type=str, help="Glob to multiple NPZ files")
    ap.add_argument("--plots", action="store_true", help="Save simple plots if available")
    args = ap.parse_args()

    paths: List[Path] = []
    if args.file:
        paths.append(Path(args.file))
    if args.glob:
        import glob
        for p in glob.glob(args.glob):
            paths.append(Path(p))
    if not paths:
        ap.error("Please provide --file <path.npz> or --glob \"pattern\"")

    for p in paths:
        print(f"[NPZ CHECK] {p}")
        res = analyze_file(p, make_plots=args.plots)
        keys = res.get("keys", [])
        print("  keys:", keys)
        for k, s in (res.get("key_stats") or {}).items():
            cv = s.get("cv", None)
            flag_const = (cv is not None and cv < 1e-3)
            mean = s.get("mean", None); std = s.get("std", None)
            if mean is None or std is None:
                continue
            print(f"  - {k}: shape={s['shape']} mean={mean:.4g} std={std:.4g} cv={None if cv is None else f'{cv:.3e}'} const_like={flag_const}")
        ti = res.get("time_info", {})
        if ti:
            print("  time:", ti)
        if res.get("warnings"):
            for w in res["warnings"]:
                print("  WARN:", w)
        if res.get("report"):
            print("  report:", res["report"])
        if res.get("summary"):
            print("  summary:", res["summary"])
        if res.get("plot_E2acc_hist"): print("  plot:", res["plot_E2acc_hist"])
        if res.get("plot_E2acc_ts"):   print("  plot:", res["plot_E2acc_ts"])
        if res.get("plot_E2gyr_ts"):   print("  plot:", res["plot_E2gyr_ts"])

if __name__ == "__main__":
    main()
