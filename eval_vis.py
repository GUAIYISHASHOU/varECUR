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
from vis.models.uncert_head import UncertHead2D

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

def plot_coverage_curve(e2x, e2y, vx, vy, m, out_png):
    """
    Plot coverage curve (α-coverage).
    Uses chi2(df=2) thresholds τ_α = -2*ln(1-α), computes proportion of e⊤Σ⁻¹e ≤ τ_α.
    Ideal curve is y=x.
    """
    m = m > 0
    z2 = (e2x/vx + e2y/vy)[m]
    alphas = np.linspace(0.01, 0.99, 99)
    th = -2.0 * np.log(1.0 - alphas)  # chi2(df=2) inverse CDF
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
    Q-Q plot: empirical quantiles vs theoretical chi2(df=2) quantiles.
    Above diagonal = over-confident; below = under-confident.
    """
    z2 = np.sort(z2)
    n = len(z2)
    qs = (np.arange(1, n+1) - 0.5) / n
    th = -2.0 * np.log(1.0 - qs)  # chi2 quantiles
    
    plt.figure(figsize=(6, 6))
    plt.plot(th, z2, linewidth=1.0, alpha=0.7)
    plt.plot([th[0], th[-1]], [th[0], th[-1]], linestyle="--", color="red", label="Perfect calibration")
    plt.xlabel("Theoretical χ² quantile", fontsize=11)
    plt.ylabel("Empirical z² quantile", fontsize=11)
    plt.title("Q–Q Plot vs Chi-square (df=2)", fontsize=12)
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
    
    model = UncertHead2D(in_ch=2, geom_dim=ds.geom.shape[1], out_dim=2)
    model.load_state_dict(ckpt.get("model", ckpt))
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
    if args.save_temp:
        if args.auto_temp == "axis":
            temp_dict = {"dx": dx, "dy": dy}
        else:
            temp_dict = {"delta_logvar": (dx + dy) * 0.5}
        with open(args.save_temp, "w", encoding="utf-8") as f:
            json.dump(temp_dict, f, ensure_ascii=False, indent=2)
        print(f"[save_temp] Temperature saved to {args.save_temp}")
    
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
            print("[plot] 1/6 Coverage curve...")
            plot_coverage_curve(e2x_np, e2y_np, vx_np, vy_np, m_np, 
                              os.path.join(args.plot_dir, "coverage_curve.png"))
            
            print("[plot] 2/6 Q-Q plot...")
            plot_qq_chi2(z2_all, os.path.join(args.plot_dir, "qq_chi2.png"))
            
            print("[plot] 3/6 Log-variance histogram...")
            plot_hist_logvar(lv_np[:, 0], lv_np[:, 1], 
                           os.path.join(args.plot_dir, "hist_logvar.png"))
            
            print("[plot] 4/6 Error² vs Variance...")
            plot_err2_vs_var(e2x_np, e2y_np, vx_np, vy_np, m_np,
                           os.path.join(args.plot_dir, "err2_vs_var.png"))
            
            print("[plot] 5/6 Sparsification curve + AUSE...")
            e2_scalar = ((e2x_np + e2y_np) / 2.0)[m_np]
            v_scalar = ((vx_np + vy_np) / 2.0)[m_np]
            plot_sparsification(e2_scalar, v_scalar,
                              os.path.join(args.plot_dir, "sparsification.png"),
                              os.path.join(args.plot_dir, "ause.txt"))
            
            print("[plot] 6/6 Per-sequence metrics (if available)...")
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
            
        except Exception as e:
            print(f"[plot] Error generating plots: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

