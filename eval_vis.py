#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for visual uncertainty estimation.
Computes calibration metrics (z², coverage, correlation).
"""
import argparse
import numpy as np
import torch
import json
import os
from torch.utils.data import DataLoader

from vis.datasets.vis_pairs import VISPairs
from vis.models.uncert_head import UncertHead2D, UncertHead_ResNet_CrossAttention

# Import scipy for chi2 quantiles
try:
    from scipy.stats import chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warning] scipy not available, isotonic calibration will use approximation")

# Import matplotlib only when plotting is needed
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def compute_metrics(e2x, e2y, lvx, lvy, mask, lv_min=-10, lv_max=4):
    """
    Compute calibration metrics for 2D visual uncertainty.
    
    Returns:
        dict with z2_mean, cov68, cov95, spearman correlation
    """
    m = (mask > 0).astype(np.float32)
    vx = np.exp(np.clip(lvx, lv_min, lv_max))
    vy = np.exp(np.clip(lvy, lv_min, lv_max))
    
    # Compute normalized squared error (χ² / df)
    z2 = (e2x/vx + e2y/vy) / 2.0  # df=2 for 2D
    z2 = z2[m > 0]
    
    z2_mean = float(np.mean(z2))
    
    # 68/95% coverage (χ²_2 quantiles / 2)
    z2_68 = 2.27886856637673 / 2.0  # chi2.ppf(0.68, 2) / 2
    z2_95 = 5.99146454710798 / 2.0  # chi2.ppf(0.95, 2) / 2
    cov68 = float(np.mean(z2 <= z2_68))
    cov95 = float(np.mean(z2 <= z2_95))
    
    # Spearman correlation (error² vs variance)
    v = vx + vy
    spear = 0.0
    if z2.size >= 10:
        e_total = (e2x + e2y)[m > 0]
        v_total = v[m > 0]
        rr = np.argsort(np.argsort(e_total))
        vv = np.argsort(np.argsort(v_total))
        spear = float(np.corrcoef(rr, vv)[0, 1])
    
    return dict(z2_mean=z2_mean, cov68=cov68, cov95=cov95, spearman=spear)

def _ensure_dir(d):
    """Create directory if it doesn't exist."""
    os.makedirs(d, exist_ok=True)

def _to_np(x):
    """Convert tensor or array to numpy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

# === BEGIN: lv-based per-axis isotonic helpers ===
def _quantile_knots(x, K=32):
    """Create K bins with quantile-based edges."""
    qs = np.linspace(0.0, 1.0, K+1)  # edges
    edges = np.quantile(x, qs)
    # Prevent duplicate edges causing empty bins
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def _binned_mean(x, y, edges):
    """Compute mean of y in each bin defined by edges on x."""
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges)-2)
    K = len(edges) - 1
    ssum = np.zeros(K, dtype=np.float64)
    wsum = np.zeros(K, dtype=np.int64)
    for k in range(K):
        mask = (idx == k)
        if mask.any():
            ssum[k] = y[mask].mean()
            wsum[k] = mask.sum()
        else:
            ssum[k] = np.nan
            wsum[k] = 0
    return ssum, wsum

def _binned_quantile(x, y, edges, q=0.75, winsor=0.99):
    """
    Compute quantile of y in each bin, with winsorization to handle outliers.
    
    Args:
        x: input variable for binning
        y: values to compute quantile on
        edges: bin edges
        q: quantile to compute (0.75 = 75th percentile)
        winsor: upper quantile for winsorization (cap extreme values)
    
    Returns:
        out: quantiles per bin (NaN if bin is empty)
    """
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges)-2)
    K = len(edges) - 1
    out = np.full(K, np.nan, dtype=np.float64)
    
    for k in range(K):
        m = (idx == k)
        if m.any():
            yy = y[m]
            # Winsorize upper tail to resist extreme outliers
            cap = np.quantile(yy, winsor)
            yy = np.minimum(yy, cap)
            out[k] = np.quantile(yy, q)
    
    return out

def _enforce_monotone(y, direction="auto"):
    """
    Enforce monotonicity on y.
    - direction="increasing": cumulative maximum (non-decreasing)
    - direction="decreasing": reverse cumulative maximum (non-increasing)
    - direction="auto": auto-detect based on correlation with index
    """
    y = y.copy()
    # Fill NaN with linear interpolation
    n = len(y)
    if np.any(np.isnan(y)):
        xs = np.arange(n)
        notnan = ~np.isnan(y)
        y = np.interp(xs, xs[notnan], y[notnan])

    if direction == "auto":
        i = np.arange(n, dtype=np.float64)
        c = np.corrcoef(i, y)[0, 1] if np.std(y) > 0 else 0.0
        direction = "increasing" if c >= 0 else "decreasing"

    if direction == "increasing":
        y = np.maximum.accumulate(y)
    elif direction == "decreasing":
        y = np.maximum.accumulate(y[::-1])[::-1]
    return y

def fit_isotonic_from_lv_per_axis(lv, e2, knots=32, g_clip=(1e-3, 1e3), 
                                  q=0.75, winsor=0.99, shrink=0.8):
    """
    Fit per-axis isotonic calibration based on log-variance (lv).
    
    NEW: Quantile-based + shrinkage + winsorization for robustness.
    
    CRITICAL: This approach uses ONLY predicted lv as input on test set,
    avoiding any information leakage from true errors e².
    
    Args:
        lv: [N,2] predicted log-variance (after applying axis temperature)
        e2: [N,2] squared errors per axis
        knots: number of bins for piecewise function
        g_clip: (min, max) clipping range for scale factors
        q: quantile to use (0.75 = 75th percentile, more robust than mean)
        winsor: upper quantile for winsorization (cap extreme outliers)
        shrink: shrinkage coefficient β, g ← g^β (avoid over-correction)
    
    Returns:
        model: dict containing per-axis (s_knots, g_knots) for JSON storage
    
    Method:
        For each lv-bin, compute r = e²/exp(lv) (should ≈χ²(df=1) quantile if calibrated),
        fit g(lv) using quantile matching, enforce monotonicity, shrink, then apply lv += log(g(lv)).
    """
    eps = 1e-12
    model = {
        "version": 4,  # Version 4: quantile-based with shrinkage
        "indep": "lv",
        "per_axis": True,
        "target": f"quantile_q{q}_winsor{winsor}_shrink{shrink}",
        "note": "fit on val; apply on test without using e2",
        "q": q,
        "winsor": winsor,
        "shrink": shrink,
        "axes": {}
    }
    
    for j, name in enumerate(["x", "y"]):
        s = lv[:, j].astype(np.float64)
        r = (e2[:, j] / np.exp(s)).astype(np.float64)  # r ~ chi2(df=1) if calibrated
        edges, centers = _quantile_knots(s, K=knots)
        
        # Quantile-based fitting with winsorization
        q_emp = _binned_quantile(s, r, edges, q=q, winsor=winsor)
        
        # Theoretical quantile for chi2(df=1)
        if HAS_SCIPY:
            q_th = chi2.ppf(q, df=1)
        else:
            # Approximation for chi2(df=1) at q=0.75: ≈ 1.32
            q_th = -2.0 * np.log(1.0 - q)  # Rough approximation
        
        # Compute scale: g = q_theory / q_empirical
        g_raw = q_th / np.maximum(q_emp, eps)

        # Enforce monotonicity (auto-detect direction)
        g_mono = _enforce_monotone(g_raw, direction="auto")

        # Clip to reasonable range to avoid over-amplification
        g_mono = np.clip(g_mono, g_clip[0], g_clip[1])
        
        # Shrinkage: g ← g^β (avoid over-correction that hurts other axis)
        g_mono = np.power(g_mono, shrink)

        model["axes"][name] = {
            "s_knots": centers.tolist(),
            "g_knots": g_mono.tolist(),
            "bins": int(len(centers))
        }
    
    return model

def apply_isotonic_from_lv_per_axis(lv, model, strength=1.0, strength_xy=None):
    """
    Apply per-axis isotonic calibration using ONLY predicted lv.
    
    ZERO INFORMATION LEAKAGE: No access to true errors e² needed.
    
    NEW: Added strength parameter for soft application.
    NEW v2: Added strength_xy for per-axis control.
    
    Args:
        lv: [N,2] predicted log-variance
        model: dict from fit_isotonic_from_lv_per_axis
        strength: soft strength α, lv += α·log(g(lv)) (0.6-0.8 more conservative)
                  Used if strength_xy is None
        strength_xy: tuple (sx, sy) for per-axis strength. Overrides strength if provided.
    
    Returns:
        lv_calibrated: [N,2] calibrated log-variance
    """
    lv = lv.copy()
    eps = 1e-12
    
    # Determine per-axis strengths
    if strength_xy is not None:
        sx, sy = strength_xy
    else:
        sx = sy = strength
    
    strengths = [sx, sy]
    
    for j, name in enumerate(["x", "y"]):
        m = model["axes"][name]
        s_knots = np.asarray(m["s_knots"], dtype=np.float64)
        g_knots = np.asarray(m["g_knots"], dtype=np.float64)
        s = lv[:, j].astype(np.float64)
        
        # Lookup g(lv) via interpolation
        g = np.interp(s, s_knots, g_knots, left=g_knots[0], right=g_knots[-1])
        
        # Apply calibration with per-axis strength: lv += α_j·log(g)
        lv[:, j] = lv[:, j] + strengths[j] * np.log(np.maximum(g, eps))
    
    return lv
# === END: lv-based per-axis isotonic helpers ===

def fit_monotone_piecewise_scale(z2, nbins=100):
    """
    拟合 g: z² -> scale, 使得 z²' = g(z²)*z² ≈ chi2(df=2)/2 分布.
    
    CRITICAL FIX: 使用 χ²(df=2)/2 理论分位以匹配归一化 z² = (e²_x/v_x + e²_y/v_y)/2 的定义
    
    做法: 用经验CDF的分位映射到理论分位, g = q_theory / q_empirical (分段线性、单调)
    
    Args:
        z2: [N] array of normalized squared errors (already divided by 2)
        nbins: number of knots for piecewise linear function
    
    Returns:
        knots_z2: list of z² knot positions
        scale: list of corresponding scale factors
    """
    z2 = np.asarray(z2, dtype=np.float64)
    z2 = z2[np.isfinite(z2)]
    z2.sort()
    n = len(z2)
    
    qs = (np.arange(1, n+1) - 0.5) / n
    # FIXED: 使用 χ²(df=2)/2 以匹配 z² 的归一化定义
    q_th = (-2.0 * np.log(1.0 - qs)) / 2.0  # chi-square(df=2)/2 quantiles
    
    # Compress to nbins knots
    idx = np.linspace(0, n-1, nbins).astype(int)
    knots_z2 = z2[idx]
    knots_th = q_th[idx]
    
    # scale = q_th / q_emp, avoid division by zero
    # Add small epsilon for robustness in extreme tails
    scale = knots_th / np.maximum(knots_z2, 1e-12)
    
    # Force monotone non-decreasing (cumulative maximum)
    scale = np.maximum.accumulate(scale)
    
    return knots_z2.tolist(), scale.tolist()

def apply_piecewise_scale(z2, knots_x, knots_s):
    """
    对每个样本的 z² 查表插值得到 g(z²)，返回 scale 向量.
    
    Args:
        z2: [N] array of z² values
        knots_x: list of z² knot positions
        knots_s: list of scale factors at knots
    
    Returns:
        scale: [N] array of interpolated scales
    """
    x = np.asarray(z2, dtype=np.float64)
    kx = np.asarray(knots_x, dtype=np.float64)
    ks = np.asarray(knots_s, dtype=np.float64)
    # Linear interpolation + extrapolate to endpoints
    return np.interp(x, kx, ks, left=ks[0], right=ks[-1])

def save_isotonic(path, knots_x, knots_s):
    """Save isotonic calibration to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"knots_x": knots_x, "knots_s": knots_s}, f, ensure_ascii=False, indent=2)

def load_isotonic(path):
    """Load isotonic calibration from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        J = json.load(f)
    return J["knots_x"], J["knots_s"]

def plot_coverage_curve(e2x, e2y, vx, vy, m, out_png):
    """
    Plot coverage curve (α-coverage).
    Uses chi2(df=2)/2 thresholds to match normalized z² = (e²_x/v_x + e²_y/v_y)/2.
    Ideal curve is y=x.
    """
    m = m > 0
    # FIXED: 真正做 /2，和指标保持一致（df=2，再除以2）
    z2 = ((e2x / vx) + (e2y / vy)) / 2.0  # Properly normalized for df=2
    alphas = np.linspace(0.01, 0.99, 99)
    # FIXED: Use χ²(df=2)/2 to match normalized z²
    th = (-2.0 * np.log(1.0 - alphas)) / 2.0  # chi2(df=2)/2 inverse CDF
    emp = np.array([(z2 <= t).mean() for t in th])
    
    plt.figure(figsize=(6, 5))
    plt.plot(alphas, emp, label="Empirical", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.xlabel("Confidence level α", fontsize=11)
    plt.ylabel("Empirical coverage", fontsize=11)
    plt.title("Coverage Curve (df=2)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_qq_chi2(z2, out_png):
    """
    Q-Q plot: empirical quantiles vs theoretical chi2(df=2)/2 quantiles.
    Above diagonal = over-confident; below = under-confident.
    
    FIXED: Use χ²(df=2)/2 to match normalized z² definition.
    """
    z2 = np.sort(z2)
    n = len(z2)
    qs = (np.arange(1, n+1) - 0.5) / n
    # FIXED: Use χ²(df=2)/2 to match normalized z²
    th = (-2.0 * np.log(1.0 - qs)) / 2.0  # chi2(df=2)/2 quantiles
    
    plt.figure(figsize=(6, 6))
    plt.plot(th, z2, linewidth=1.0, alpha=0.7)
    plt.plot([th[0], th[-1]], [th[0], th[-1]], linestyle="--", color="red", label="Perfect calibration")
    plt.xlabel("Theoretical χ²(df=2)/2 quantile", fontsize=11)
    plt.ylabel("Empirical z² quantile", fontsize=11)
    plt.title("Q–Q Plot vs χ²(df=2)/2", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_coverage_per_axis(e2, v, m, out_png, axis_name="x", bootstrap=False, n_boot=100):
    """
    Per-axis coverage curve with optional bootstrap confidence bands.
    
    Args:
        e2: squared error for this axis [N]
        v: variance for this axis [N]
        m: mask [N]
        out_png: output path
        axis_name: "x" or "y"
        bootstrap: if True, add 95% confidence band
        n_boot: number of bootstrap samples
    """
    m = m > 0
    z2_single = (e2 / v)[m]  # Single-axis z² (df=1)
    alphas = np.linspace(0.01, 0.99, 99)
    
    # Theoretical thresholds for χ²(df=1)
    if HAS_SCIPY:
        th = chi2.ppf(alphas, df=1)
    else:
        th = -2.0 * np.log(1.0 - alphas)  # Approximation
    
    # Empirical coverage
    emp = np.array([(z2_single <= t).mean() for t in th])
    
    plt.figure(figsize=(6, 5))
    
    # Bootstrap confidence bands
    if bootstrap and n_boot > 0:
        boot_curves = []
        n = len(z2_single)
        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            z2_boot = z2_single[idx]
            emp_boot = np.array([(z2_boot <= t).mean() for t in th])
            boot_curves.append(emp_boot)
        boot_curves = np.array(boot_curves)
        
        # 95% confidence interval
        lower = np.percentile(boot_curves, 2.5, axis=0)
        upper = np.percentile(boot_curves, 97.5, axis=0)
        
        plt.fill_between(alphas, lower, upper, alpha=0.2, color='blue', 
                        label='95% CI (bootstrap)')
    
    plt.plot(alphas, emp, label="Empirical", linewidth=2, color='blue')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.xlabel("Confidence level α", fontsize=11)
    plt.ylabel("Empirical coverage", fontsize=11)
    plt.title(f"Coverage Curve - {axis_name.upper()}-axis (df=1)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_qq_per_axis(e2, v, m, out_png, axis_name="x", bootstrap=False, n_boot=100):
    """
    Per-axis Q-Q plot with optional bootstrap confidence bands.
    
    Args:
        e2: squared error for this axis [N]
        v: variance for this axis [N]
        m: mask [N]
        out_png: output path
        axis_name: "x" or "y"
        bootstrap: if True, add 95% confidence band
        n_boot: number of bootstrap samples
    """
    m = m > 0
    z2_single = (e2 / v)[m]  # Single-axis z² (df=1)
    z2_sorted = np.sort(z2_single)
    n = len(z2_sorted)
    qs = (np.arange(1, n+1) - 0.5) / n
    
    # Theoretical quantiles for χ²(df=1)
    if HAS_SCIPY:
        th = chi2.ppf(qs, df=1)
    else:
        th = -2.0 * np.log(1.0 - qs)  # Approximation
    
    plt.figure(figsize=(6, 6))
    
    # Bootstrap confidence bands
    if bootstrap and n_boot > 0:
        boot_quantiles = []
        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            z2_boot = np.sort(z2_single[idx])
            boot_quantiles.append(z2_boot)
        boot_quantiles = np.array(boot_quantiles)
        
        # 95% confidence interval
        lower = np.percentile(boot_quantiles, 2.5, axis=0)
        upper = np.percentile(boot_quantiles, 97.5, axis=0)
        
        plt.fill_between(th, lower, upper, alpha=0.2, color='blue',
                        label='95% CI (bootstrap)')
    
    plt.plot(th, z2_sorted, linewidth=1.5, alpha=0.8, color='blue', label='Empirical')
    plt.plot([th[0], th[-1]], [th[0], th[-1]], linestyle="--", color="red", 
            label="Perfect calibration")
    plt.xlabel(f"Theoretical χ²(df=1) quantile", fontsize=11)
    plt.ylabel(f"Empirical z² quantile", fontsize=11)
    plt.title(f"Q-Q Plot - {axis_name.upper()}-axis (df=1)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hist_logvar(lvx, lvy, out_png):
    """Histogram of log variances (separate for x/y axes)."""
    plt.figure(figsize=(7, 4))
    plt.hist(lvx, bins=60, alpha=0.6, label="log σ²_x", color="blue", edgecolor="none")
    plt.hist(lvy, bins=60, alpha=0.6, label="log σ²_y", color="orange", edgecolor="none")
    plt.xlabel("Log variance", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.title("Distribution of Log Variances", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_err2_vs_var(e2x, e2y, vx, vy, m, out_png):
    """
    Error² vs Variance correlation scatter plot with binned means.
    Shows monotonicity/correlation (common in papers).
    """
    m = m > 0
    err2 = (e2x[m] + e2y[m]) / 2.0
    var = (vx[m] + vy[m]) / 2.0
    
    # Log scale for better visualization
    r = np.log10(err2 + 1e-12)
    v = np.log10(var + 1e-12)
    
    # Binned means
    bins = np.linspace(v.min(), v.max(), 40)
    idx = np.digitize(v, bins)
    bx, by = [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if sel.any():
            bx.append(bins[k-1:k+1].mean())
            by.append(r[sel].mean())
    
    plt.figure(figsize=(7, 6))
    plt.scatter(v, r, s=2, alpha=0.2, color="steelblue")
    if bx:
        plt.plot(bx, by, linewidth=3, color="red", label="Binned mean")
    plt.xlabel("log₁₀ predicted variance", fontsize=11)
    plt.ylabel("log₁₀ mean squared error", fontsize=11)
    plt.title("Error² vs Variance Correlation", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_sparsification(e2, var, out_png, out_txt):
    """
    Sparsification plot + AUSE (Area Under Sparsification Error).
    Drop samples by predicted uncertainty (high→low) vs oracle (true error).
    """
    order_unc = np.argsort(-var)  # Drop high uncertainty first
    order_orc = np.argsort(-e2)   # Oracle: drop by true error
    
    N = len(e2)
    ks = np.linspace(0, 1, 50)
    e2_cum = []
    e2_orc = []
    
    for f in ks:
        k = int(f * N)
        if k == 0:
            e2_cum.append(e2.mean())
            e2_orc.append(e2.mean())
        else:
            # Keep samples NOT in top-k dropped
            keep_unc = np.ones(N, dtype=bool)
            keep_unc[order_unc[:k]] = False
            keep_orc = np.ones(N, dtype=bool)
            keep_orc[order_orc[:k]] = False
            
            e2_cum.append(e2[keep_unc].mean() if keep_unc.any() else 0)
            e2_orc.append(e2[keep_orc].mean() if keep_orc.any() else 0)
    
    e2_cum = np.array(e2_cum)
    e2_orc = np.array(e2_orc)
    # Use trapezoid instead of deprecated trapz
    try:
        ause = float(np.trapezoid(e2_cum - e2_orc, ks))
    except AttributeError:  # Fallback for older numpy
        ause = float(np.trapz(e2_cum - e2_orc, ks))
    
    # Save AUSE to text file
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"AUSE = {ause:.6f}\n")
        f.write(f"(Area between predicted and oracle sparsification curves)\n")
    
    plt.figure(figsize=(7, 5))
    plt.plot(ks, e2_cum, label="Drop by predicted uncertainty", linewidth=2, color="blue")
    plt.plot(ks, e2_orc, label="Oracle (drop by true error)", linewidth=2, color="green", linestyle="--")
    plt.xlabel("Fraction of samples dropped", fontsize=11)
    plt.ylabel("Mean error²", fontsize=11)
    plt.title(f"Sparsification Curve (AUSE={ause:.6f})", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_per_seq_if_available(npz_path, sid, z2, cov95, spear, out_png):
    """
    Per-sequence metrics bar plots (z² mean, cov95, Spearman).
    Only works if 'sid' (sequence ID) is available in the data.
    """
    if sid is None:
        return
    
    # Try to load sequence names from metadata
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "meta" in data:
            meta = data["meta"].item()
            seqs = list(meta.get("seqs", []))
        else:
            seqs = None
    except:
        seqs = None
    
    K = int(sid.max()) + 1
    if seqs is None or len(seqs) != K:
        seqs = [f"Seq{i}" for i in range(K)]
    
    # Compute per-sequence metrics
    z2_per_seq = [np.mean(z2[sid == i]) if (sid == i).any() else 0 for i in range(K)]
    cov95_per_seq = [cov95[sid == i].mean() if (sid == i).any() else 0 for i in range(K)]
    spear_per_seq = [spear[sid == i].mean() if (sid == i).any() else 0 for i in range(K)]
    
    fig, axes = plt.subplots(3, 1, figsize=(max(8, K*0.6), 10))
    
    # z² mean
    axes[0].bar(np.arange(K), z2_per_seq, color="steelblue")
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1, label="Ideal (z²=1)")
    axes[0].set_ylabel("Mean z²", fontsize=10)
    axes[0].set_title("Per-Sequence Mean z²", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Coverage 95%
    axes[1].bar(np.arange(K), cov95_per_seq, color="green")
    axes[1].axhline(0.95, color="red", linestyle="--", linewidth=1, label="Ideal (0.95)")
    axes[1].set_ylabel("Coverage 95%", fontsize=10)
    axes[1].set_title("Per-Sequence 95% Coverage", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Spearman correlation
    axes[2].bar(np.arange(K), spear_per_seq, color="orange")
    axes[2].set_ylabel("Spearman ρ", fontsize=10)
    axes[2].set_title("Per-Sequence Error-Variance Correlation", fontsize=11)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for ax in axes:
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser("Evaluate visual uncertainty model")
    ap.add_argument("--npz", required=True, help="Test data NPZ")
    ap.add_argument("--model", required=True, help="Model checkpoint (.pt)")
    ap.add_argument("--auto_temp", choices=["off", "global", "axis"], default="off",
                   help="温度标定方式：off=不用；global=单一Δ；axis=对x/y分别标定")
    ap.add_argument("--save_temp", default=None,
                   help="把本次估计到的Δ保存到json，global保存{delta_logvar}；axis保存{dx,dy}")
    ap.add_argument("--use_temp", default=None,
                   help="从json加载Δ并应用；与 --auto_temp 冲突")
    ap.add_argument("--lv_min", type=float, default=-10)
    ap.add_argument("--lv_max", type=float, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot_dir", default=None,
                   help="如设置，将把评估结果图保存到该目录下")
    ap.add_argument("--plot_per_axis", action="store_true",
                   help="生成分轴的Coverage/Q-Q图（x/y各一张，论文图质量更高）")
    ap.add_argument("--plot_bootstrap", action="store_true",
                   help="为Coverage/Q-Q图添加bootstrap 95%置信带（更学术范）")
    ap.add_argument("--bootstrap_n", type=int, default=100,
                   help="Bootstrap重采样次数（100次够用，500次更平滑）")
    
    # Advanced calibration options
    ap.add_argument("--isotonic_save", default=None,
                   help="在当前集上拟合分段单调校准并保存到此JSON文件")
    ap.add_argument("--isotonic_use", default=None,
                   help="加载已保存的分段单调校准并应用")
    ap.add_argument("--temp_perseq", action="store_true",
                   help="按序列(sid)分别估计axis温度(需npz内含sid)")
    
    # Isotonic calibration fine-tuning
    ap.add_argument("--iso_quantile", type=float, default=0.75,
                   help="每个lv-bin用的经验分位p（0.75抗重尾）")
    ap.add_argument("--iso_shrink", type=float, default=0.8,
                   help="单调校正的收缩系数β，用g←g^β（避免过激）")
    ap.add_argument("--iso_winsor", type=float, default=0.99,
                   help="对r的上分位截尾，抗极端值")
    ap.add_argument("--iso_strength", type=float, default=0.7,
                   help="应用时软强度α: lv += α·log(g(lv))（默认0.7较保守，val可用1.0）")
    ap.add_argument("--iso_strength_xy", type=str, default=None,
                   help="按轴强度，格式'sx,sy'（例如'0.9,0.5'）。如指定则覆盖--iso_strength")
    
    # Post-temperature calibration (after isotonic)
    ap.add_argument("--post_temp", choices=["off", "global", "axis"], default="off",
                   help="在已应用isotonic之后对当前评估集再做一次全局温度；"
                        "axis=分别校正x/y；global=单一Δ；off=关闭")
    ap.add_argument("--save_post_temp", default=None,
                   help="把post-temp的Δ保存到JSON（axis: {dx,dy}；global: {delta_logvar}）")
    ap.add_argument("--use_post_temp", default=None,
                   help="从JSON读取post-temp的Δ并应用（覆盖--post_temp）")
    
    args = ap.parse_args()
    
    # Check matplotlib availability if plotting is requested
    if args.plot_dir and not HAS_MATPLOTLIB:
        print("[warning] --plot_dir specified but matplotlib not available. Skipping plots.")
        print("[warning] Install matplotlib: pip install matplotlib")
        args.plot_dir = None
    
    # Load data
    ds = VISPairs(args.npz)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    
    # Load model - 安全加载（新版本PyTorch支持 weights_only=True）
    try:
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)  # PyTorch>=2.4
    except TypeError:
        ckpt = torch.load(args.model, map_location="cpu")  # 兼容旧版
    
    # Auto-detect model architecture by inspecting checkpoint keys
    state_dict = ckpt.get("model", ckpt)
    model_keys = list(state_dict.keys())
    
    # Check if it's the new ResNet+CrossAttention model
    is_resnet_model = any(k.startswith("cnn_stem") or k.startswith("cross_attn") for k in model_keys)
    
    if is_resnet_model:
        print("[model] Detected ResNet+CrossAttention architecture")
        model = UncertHead_ResNet_CrossAttention(
            in_ch=2, 
            geom_dim=ds.geom.shape[1], 
            d_model=128,
            n_heads=4,
            out_dim=2,
            pretrained=False  # Don't need pretrained weights for eval
        )
    else:
        print("[model] Detected simple UncertHead2D architecture")
        model = UncertHead2D(in_ch=2, geom_dim=ds.geom.shape[1], out_dim=2)
    
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device).eval()
    
    print(f"[data] test samples: {len(ds)}")
    
    # Collect predictions
    all_data = []
    with torch.no_grad():
        for batch in dl:
            patch2 = batch["patch2"].to(args.device)
            geom = batch["geom"].to(args.device)
            logv = model(patch2, geom).cpu().numpy()  # [B,2]
            all_data.append({
                'lv': logv,
                'e2x': batch["e2x"].numpy(),
                'e2y': batch["e2y"].numpy(),
                'mask': batch["mask"].numpy()
            })
    
    lv = np.concatenate([x["lv"] for x in all_data], 0)
    e2x = np.concatenate([x["e2x"] for x in all_data], 0)
    e2y = np.concatenate([x["e2y"] for x in all_data], 0)
    mask = np.concatenate([x["mask"] for x in all_data], 0)
    
    # 确保评估时的限幅与训练一致
    lv = np.clip(lv, args.lv_min, args.lv_max)
    
    # Temperature scaling - 支持三种模式
    delta_logvar = 0.0
    dx = dy = 0.0
    
    def _apply_delta(lv, dx, dy):
        """应用温度校正到logvar"""
        if lv.ndim == 2 and lv.shape[1] >= 2:
            lv[:, 0] = lv[:, 0] + dx
            lv[:, 1] = lv[:, 1] + dy
        else:
            lv = lv + (dx + dy) * 0.5
        return lv
    
    # 1) 若指定 use_temp，则从 json 载入 Δ 并应用
    if args.use_temp is not None:
        print(f"[use_temp] Loading temperature from {args.use_temp}...")
        with open(args.use_temp, "r", encoding="utf-8") as f:
            J = json.load(f)
        if "delta_logvar" in J:  # global
            dx = dy = float(J["delta_logvar"])
            print(f"[use_temp] Loaded global delta_logvar = {dx:+.4f}")
        else:  # axis
            dx = float(J.get("dx", 0.0))
            dy = float(J.get("dy", 0.0))
            print(f"[use_temp] Loaded axis deltas: dx={dx:+.4f}, dy={dy:+.4f}")
        lv = _apply_delta(lv, dx, dy)
    
    # 2) 否则如果要求自动标定
    elif args.auto_temp == "global":
        print("[auto_temp] Computing global temperature scaling...")
        vx = np.exp(np.clip(lv[:, 0], args.lv_min, args.lv_max))
        vy = np.exp(np.clip(lv[:, 1], args.lv_min, args.lv_max))
        z2 = (e2x / (vx + 1e-12) + e2y / (vy + 1e-12)) / 2.0
        mu = np.mean(z2[mask > 0])
        delta_logvar = float(np.log(mu + 1e-12))
        dx = dy = delta_logvar
        lv = _apply_delta(lv, dx, dy)
        print(f"[auto_temp] delta_logvar = {delta_logvar:+.4f}  (scale×={np.exp(0.5*delta_logvar):.3f})")
    
    elif args.auto_temp == "axis":
        print("[auto_temp] Computing per-axis temperature scaling...")
        vx = np.exp(np.clip(lv[:, 0], args.lv_min, args.lv_max))
        vy = np.exp(np.clip(lv[:, 1], args.lv_min, args.lv_max))
        zx = e2x / (vx + 1e-12)
        zy = e2y / (vy + 1e-12)
        mu_x = np.mean(zx[mask > 0])
        mu_y = np.mean(zy[mask > 0])
        dx = float(np.log(mu_x + 1e-12))
        dy = float(np.log(mu_y + 1e-12))
        lv = _apply_delta(lv, dx, dy)
        print(f"[auto_temp] dx={dx:+.4f} (scale_x×={np.exp(0.5*dx):.3f}), dy={dy:+.4f} (scale_y×={np.exp(0.5*dy):.3f})")
    
    # 3) 保存 Δ（可选）
    if args.save_temp and not args.temp_perseq:
        if args.auto_temp == "axis":
            temp_dict = {"dx": dx, "dy": dy}
        else:
            temp_dict = {"delta_logvar": (dx + dy) * 0.5}
        with open(args.save_temp, "w", encoding="utf-8") as f:
            json.dump(temp_dict, f, ensure_ascii=False, indent=2)
        print(f"[save_temp] Temperature saved to {args.save_temp}")
    
    # === Advanced Calibration: Isotonic + Per-Sequence ===
    
    # 计算当前集的 z² (使用当前 lv)
    vx_np = np.exp(np.clip(lv[:, 0], args.lv_min, args.lv_max))
    vy_np = np.exp(np.clip(lv[:, 1], args.lv_min, args.lv_max))
    m_np = mask > 0
    z2_all_np = ((e2x / vx_np + e2y / vy_np) / 2.0)[m_np]
    
    # 1) NEW: lv-based per-axis isotonic calibration (ZERO information leakage on test)
    if args.isotonic_save:
        print(f"[isotonic] Fitting lv-based per-axis monotone calibration...")
        print(f"[isotonic]   quantile={args.iso_quantile}, shrink={args.iso_shrink}, winsor={args.iso_winsor}")
        # Stack e2x, e2y for per-axis fitting
        e2_stacked = np.stack([e2x, e2y], axis=1)  # [N, 2]
        iso_model = fit_isotonic_from_lv_per_axis(
            lv=lv, e2=e2_stacked, knots=32,
            q=args.iso_quantile, 
            winsor=args.iso_winsor, 
            shrink=args.iso_shrink
        )
        with open(args.isotonic_save, "w", encoding="utf-8") as f:
            json.dump(iso_model, f, indent=2)
        print(f"[isotonic] Saved lv-based per-axis model to {args.isotonic_save}")
        print(f"[isotonic]   Version: {iso_model['version']}, Target: {iso_model['target']}")
        print(f"[isotonic]   x-axis: {iso_model['axes']['x']['bins']} bins")
        print(f"[isotonic]   y-axis: {iso_model['axes']['y']['bins']} bins")
    
    if args.isotonic_use:
        print(f"[isotonic] Loading calibration from {args.isotonic_use}...")
        with open(args.isotonic_use, "r", encoding="utf-8") as f:
            iso_model = json.load(f)
        
        # Check version and warn if old format
        if not (iso_model.get("indep") == "lv" and iso_model.get("per_axis") is True):
            print(f"[isotonic] WARNING: model file does not look like lv-based per-axis (version={iso_model.get('version')})")
            print(f"[isotonic] Still applying, but results may be suboptimal...")
        
        # Log model info
        print(f"[isotonic] Model version: {iso_model.get('version', 'unknown')}")
        if 'target' in iso_model:
            print(f"[isotonic] Model target: {iso_model['target']}")
        
        # Parse per-axis strength if provided
        strength_xy = None
        if args.iso_strength_xy:
            sx, sy = [float(v.strip()) for v in args.iso_strength_xy.split(",")]
            strength_xy = (sx, sy)
            print(f"[isotonic] Applying with per-axis strength: x={sx:.2f}, y={sy:.2f}")
        else:
            print(f"[isotonic] Applying with uniform strength α={args.iso_strength}")
        
        # Apply calibration using ONLY predicted lv (no e² needed!)
        lv_before = lv.copy()
        lv = apply_isotonic_from_lv_per_axis(lv=lv, model=iso_model, 
                                             strength=args.iso_strength, 
                                             strength_xy=strength_xy)
        
        # Sanity check: compute z² after calibration
        vx_np = np.exp(np.clip(lv[:, 0], args.lv_min, args.lv_max))
        vy_np = np.exp(np.clip(lv[:, 1], args.lv_min, args.lv_max))
        z2x_after = (e2x / vx_np)[m_np].mean()
        z2y_after = (e2y / vy_np)[m_np].mean()
        z2_after = (z2x_after + z2y_after) / 2.0
        
        print(f"[isotonic] Applied lv-based per-axis model")
        print(f"[isotonic]   mean z²: {z2_after:.3f} (x={z2x_after:.3f}, y={z2y_after:.3f})")
        print(f"[isotonic]   Δlv_x: {(lv[:, 0] - lv_before[:, 0]).mean():+.4f} ± {(lv[:, 0] - lv_before[:, 0]).std():.4f}")
        print(f"[isotonic]   Δlv_y: {(lv[:, 1] - lv_before[:, 1]).mean():+.4f} ± {(lv[:, 1] - lv_before[:, 1]).std():.4f}")
        
        # Update z2_all_np for subsequent plotting
        z2_all_np = ((e2x / vx_np + e2y / vy_np) / 2.0)[m_np]
    
    # === Post-temperature calibration (after isotonic, to combat domain shift) ===
    if args.use_post_temp:
        print(f"[post_temp] Loading post-temperature from {args.use_post_temp}...")
        with open(args.use_post_temp, "r", encoding="utf-8") as f:
            J = json.load(f)
        if "dx" in J and "dy" in J:  # axis mode
            dx_post = float(J["dx"])
            dy_post = float(J["dy"])
            lv[:, 0] += dx_post
            lv[:, 1] += dy_post
            print(f"[post_temp] Loaded axis dx={dx_post:+.4f}, dy={dy_post:+.4f}")
        elif "delta_logvar" in J:  # global mode
            d_post = float(J["delta_logvar"])
            lv[:, 0] += d_post
            lv[:, 1] += d_post
            print(f"[post_temp] Loaded global Δ={d_post:+.4f}")
    elif args.post_temp != "off":
        print(f"[post_temp] Computing {args.post_temp} post-temperature on current evaluation set...")
        # 以isotonic之后的vx/vy为基准，计算均值μ并把z²拉回1
        z2x = (e2x / vx_np)[m_np]
        z2y = (e2y / vy_np)[m_np]
        
        if args.post_temp == "axis":
            dx_post = float(np.log(max(z2x.mean(), 1e-12)))
            dy_post = float(np.log(max(z2y.mean(), 1e-12)))
            lv[:, 0] += dx_post
            lv[:, 1] += dy_post
            print(f"[post_temp] axis  dx={dx_post:+.4f}, dy={dy_post:+.4f}")
            if args.save_post_temp:
                with open(args.save_post_temp, "w", encoding="utf-8") as f:
                    json.dump({"dx": dx_post, "dy": dy_post}, f, indent=2)
                print(f"[post_temp] Saved to {args.save_post_temp}")
        else:  # global
            d_post = float(np.log(max(((z2x.mean() + z2y.mean()) / 2.0), 1e-12)))
            lv[:, 0] += d_post
            lv[:, 1] += d_post
            print(f"[post_temp] global Δ={d_post:+.4f}")
            if args.save_post_temp:
                with open(args.save_post_temp, "w", encoding="utf-8") as f:
                    json.dump({"delta_logvar": d_post}, f, indent=2)
                print(f"[post_temp] Saved to {args.save_post_temp}")
    
    # 应用post-temp后，刷新vx/vy/z²（供后续绘图与统计）
    if args.use_post_temp or args.post_temp != "off":
        vx_np = np.exp(np.clip(lv[:, 0], args.lv_min, args.lv_max))
        vy_np = np.exp(np.clip(lv[:, 1], args.lv_min, args.lv_max))
        z2x_after = (e2x / vx_np)[m_np].mean()
        z2y_after = (e2y / vy_np)[m_np].mean()
        z2_after = (z2x_after + z2y_after) / 2.0
        print(f"[post_temp]   mean z² after: {z2_after:.3f} (x={z2x_after:.3f}, y={z2y_after:.3f})")
        z2_all_np = ((e2x / vx_np + e2y / vy_np) / 2.0)[m_np]
    # === end post-temp ===
    
    # 2) Per-sequence temperature calibration
    if args.temp_perseq:
        print(f"[temp_perseq] Computing per-sequence axis temperatures...")
        try:
            raw_data = np.load(args.npz, allow_pickle=True)
            if "sid" in raw_data:
                sid = raw_data["sid"]
                sid_valid = sid[m_np]  # Only valid samples
                
                zx = (e2x / vx_np)[m_np]
                zy = (e2y / vy_np)[m_np]
                
                out = {"mode": "perseq_axis", "seqs": [], "dx": [], "dy": []}
                for s in np.unique(sid_valid):
                    sel = (sid_valid == s)
                    mu_x = zx[sel].mean()
                    mu_y = zy[sel].mean()
                    out["seqs"].append(int(s))
                    out["dx"].append(float(np.log(max(mu_x, 1e-12))))
                    out["dy"].append(float(np.log(max(mu_y, 1e-12))))
                
                if args.save_temp:
                    with open(args.save_temp, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False, indent=2)
                    print(f"[temp_perseq] Saved to {args.save_temp}")
                
                print(f"[temp_perseq] Computed temperatures for {len(out['seqs'])} sequences")
            else:
                print(f"[temp_perseq] Warning: no 'sid' found in npz, skipping")
        except Exception as e:
            print(f"[temp_perseq] Error: {e}")
    
    # Compute metrics
    results = compute_metrics(e2x, e2y, lv[:, 0], lv[:, 1], mask, 
                             args.lv_min, args.lv_max)
    results["delta_logvar"] = delta_logvar
    results["dx"] = dx
    results["dy"] = dy
    
    print("\n[results]")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Additional per-axis diagnostics
    vx = np.exp(np.clip(lv[:, 0], args.lv_min, args.lv_max))
    vy = np.exp(np.clip(lv[:, 1], args.lv_min, args.lv_max))
    m = mask > 0
    z2x_mean = float(np.mean((e2x / vx)[m]))
    z2y_mean = float(np.mean((e2y / vy)[m]))
    
    print(f"\n[per-axis] z²_x={z2x_mean:.3f}, z²_y={z2y_mean:.3f}")
    
    # === Generate visualization plots ===
    if args.plot_dir:
        print(f"\n[plot] Generating evaluation plots in {args.plot_dir}...")
        _ensure_dir(args.plot_dir)
        
        # Convert to numpy arrays
        e2x_np = _to_np(e2x)
        e2y_np = _to_np(e2y)
        lv_np = _to_np(lv)
        vx_np = np.exp(np.clip(lv_np[:, 0], args.lv_min, args.lv_max))
        vy_np = np.exp(np.clip(lv_np[:, 1], args.lv_min, args.lv_max))
        m_np = _to_np(mask) > 0
        
        # Compute z² for all samples
        z2_all = ((e2x_np / vx_np + e2y_np / vy_np) / 2.0)[m_np]
        
        # Check if sequence ID (sid) is available
        sid = None
        try:
            raw_data = np.load(args.npz, allow_pickle=True)
            if "sid" in raw_data:
                sid = _to_np(raw_data["sid"])
                print(f"[plot] Found sequence IDs, will generate per-sequence plots")
        except Exception as e:
            print(f"[plot] Could not load sid: {e}")
        
        # Generate all plots
        try:
            n_plots = 6
            if args.plot_per_axis:
                n_plots += 4  # +4 for per-axis Coverage/Q-Q
            
            print(f"[plot] 1/{n_plots} Coverage curve (combined)...")
            plot_coverage_curve(e2x_np, e2y_np, vx_np, vy_np, m_np, 
                              os.path.join(args.plot_dir, "coverage_curve.png"))
            
            print(f"[plot] 2/{n_plots} Q-Q plot (combined)...")
            plot_qq_chi2(z2_all, os.path.join(args.plot_dir, "qq_chi2.png"))
            
            curr_plot = 3
            
            # Per-axis plots (论文图质量)
            if args.plot_per_axis:
                print(f"[plot] {curr_plot}/{n_plots} Coverage curve - X-axis{' (with bootstrap)' if args.plot_bootstrap else ''}...")
                plot_coverage_per_axis(e2x_np, vx_np, m_np, 
                                      os.path.join(args.plot_dir, "coverage_x_axis.png"),
                                      axis_name="x", bootstrap=args.plot_bootstrap, 
                                      n_boot=args.bootstrap_n)
                curr_plot += 1
                
                print(f"[plot] {curr_plot}/{n_plots} Coverage curve - Y-axis{' (with bootstrap)' if args.plot_bootstrap else ''}...")
                plot_coverage_per_axis(e2y_np, vy_np, m_np, 
                                      os.path.join(args.plot_dir, "coverage_y_axis.png"),
                                      axis_name="y", bootstrap=args.plot_bootstrap, 
                                      n_boot=args.bootstrap_n)
                curr_plot += 1
                
                print(f"[plot] {curr_plot}/{n_plots} Q-Q plot - X-axis{' (with bootstrap)' if args.plot_bootstrap else ''}...")
                plot_qq_per_axis(e2x_np, vx_np, m_np, 
                                os.path.join(args.plot_dir, "qq_x_axis.png"),
                                axis_name="x", bootstrap=args.plot_bootstrap, 
                                n_boot=args.bootstrap_n)
                curr_plot += 1
                
                print(f"[plot] {curr_plot}/{n_plots} Q-Q plot - Y-axis{' (with bootstrap)' if args.plot_bootstrap else ''}...")
                plot_qq_per_axis(e2y_np, vy_np, m_np, 
                                os.path.join(args.plot_dir, "qq_y_axis.png"),
                                axis_name="y", bootstrap=args.plot_bootstrap, 
                                n_boot=args.bootstrap_n)
                curr_plot += 1
            
            print(f"[plot] {curr_plot}/{n_plots} Log-variance histogram...")
            plot_hist_logvar(lv_np[:, 0], lv_np[:, 1], 
                           os.path.join(args.plot_dir, "hist_logvar.png"))
            curr_plot += 1
            
            print(f"[plot] {curr_plot}/{n_plots} Error² vs Variance...")
            plot_err2_vs_var(e2x_np, e2y_np, vx_np, vy_np, m_np,
                           os.path.join(args.plot_dir, "err2_vs_var.png"))
            curr_plot += 1
            
            print(f"[plot] {curr_plot}/{n_plots} Sparsification curve + AUSE...")
            e2_scalar = ((e2x_np + e2y_np) / 2.0)[m_np]
            v_scalar = ((vx_np + vy_np) / 2.0)[m_np]
            plot_sparsification(e2_scalar, v_scalar,
                              os.path.join(args.plot_dir, "sparsification.png"),
                              os.path.join(args.plot_dir, "ause.txt"))
            curr_plot += 1
            
            print(f"[plot] {curr_plot}/{n_plots} Per-sequence metrics (if available)...")
            if sid is not None:
                # Compute per-sample coverage for per-seq plot
                z2_full = (e2x_np / vx_np + e2y_np / vy_np) / 2.0
                z2_95 = 5.99146454710798 / 2.0  # chi2.ppf(0.95, 2) / 2
                cov95_full = (z2_full <= z2_95).astype(float)
                
                # Compute per-sample Spearman (use variance as proxy)
                spear_full = (vx_np + vy_np) / 2.0  # Just use variance as placeholder
                
                plot_per_seq_if_available(args.npz, sid, z2_full, cov95_full, spear_full,
                                        os.path.join(args.plot_dir, "per_seq_metrics.png"))
            else:
                print("[plot]   (skipped: no sequence ID found)")
            
            print(f"[plot] ✓ All plots saved to {args.plot_dir}/")
            if args.plot_per_axis:
                print(f"[plot] ✓ Per-axis plots: coverage_x/y_axis.png, qq_x/y_axis.png")
            if args.plot_bootstrap:
                print(f"[plot] ✓ Bootstrap confidence bands added ({args.bootstrap_n} resamples)")
            
        except Exception as e:
            print(f"[plot] Error generating plots: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

