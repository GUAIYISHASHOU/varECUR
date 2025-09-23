from __future__ import annotations
import math
import torch
import numpy as np


def _prepare_inputs(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor):
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if logv.dim() != 2 or e2sum.dim() != 2 or mask.dim() != 2:
        raise ValueError("Expected (B,T) tensors after squeeze")
    return e2sum, logv, mask


@torch.no_grad()
def _route_metrics(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                  logv_min: float, logv_max: float, df: float,
                  yvar: torch.Tensor | None = None) -> dict:
    e2sum, logv, mask = _prepare_inputs(e2sum, logv, mask)
    logv = torch.clamp(logv, min=logv_min, max=logv_max)
    var = torch.clamp(torch.exp(logv), min=1e-12)
    m = mask.float()

    # NaN-safe for e2sum and mask-out invalids
    e2sum = torch.nan_to_num(e2sum, nan=0.0, posinf=0.0, neginf=0.0)
    e2sum = torch.where(m > 0.5, e2sum, torch.zeros_like(e2sum))
    z2 = (e2sum / var) / float(df)
    z2 = torch.clamp(z2, min=0.0)
    msum = torch.clamp(m.sum(), min=1.0)
    z2_mean = float((z2 * m).sum() / msum)
    
    # 直接在z²空间做覆盖率（不取sqrt）
    if abs(df - 2.0) < 1e-6:
        z2_68, z2_95 = 2.27886856637673/2.0, 5.99146454710798/2.0  # χ²₂(0.68)/2, χ²₂(0.95)/2
    elif abs(df - 3.0) < 1e-6:
        z2_68, z2_95 = 3.505882355768183/3.0, 7.814727903251178/3.0  # χ²₃(0.68)/3, χ²₃(0.95)/3
    else:
        z2_68, z2_95 = 1.0, 4.0  # fallback for other df values
    
    cov68 = float((((z2 <= z2_68).float() * m).sum()) / msum)
    cov95 = float((((z2 <= z2_95).float() * m).sum()) / msum)

    # 排序相关性（err² vs var）
    v = torch.exp(torch.clamp(logv, min=logv_min, max=logv_max))
    mask_flat = (m.reshape(-1) > 0).cpu().numpy()
    v_np = v.reshape(-1).detach().cpu().numpy()[mask_flat]
    e_np = e2sum.reshape(-1).detach().cpu().numpy()[mask_flat]
    if v_np.size >= 3:
        rr = np.argsort(np.argsort(e_np))
        vv = np.argsort(np.argsort(v_np))
        spear = float(np.corrcoef(rr, vv)[0, 1])
    else:
        spear = 0.0

    # 饱和分解，便于判断是打上限还是打下限
    lv = torch.clamp(logv, min=logv_min, max=logv_max)
    sat_min = float((((lv <= logv_min).float() * m).sum()) / msum)
    sat_max = float((((lv >= logv_max).float() * m).sum()) / msum)
    sat = sat_min + sat_max

    out = {
        "z2_mean": z2_mean,
        "cov68": cov68,
        "cov95": cov95,
        "spear": spear,
        "sat": sat,
        "sat_min": sat_min,
        "sat_max": sat_max,
        "ez2": z2_mean,
    }

    if yvar is not None:
        if yvar.dim() == 3 and yvar.size(-1) == 1:
            yv = yvar.squeeze(-1)
        else:
            yv = yvar
        yv = torch.clamp(yv, min=1e-12)
        log_bias = float(((logv - yv.log()) * m).sum() / msum)
        log_rmse = float(torch.sqrt(((logv - yv.log()) ** 2 * m).sum() / msum))
        y_np = (yv * m).detach().cpu().numpy().reshape(-1)[mask_flat]
        if y_np.size >= 3:
            ry2 = np.argsort(np.argsort(y_np))
            spear_vy = float(np.corrcoef(np.argsort(np.argsort(vv)), ry2)[0, 1])
        else:
            spear_vy = 0.0
        ez2_true = float((((e2sum / yv) / float(df)) * m).sum() / msum)
        out.update({
            "log_bias": log_bias,
            "log_rmse": log_rmse,
            "spear_v_y": spear_vy,
            "ez2_true": ez2_true,
        })

    return out


def _collapse_axes_diag(e2_axes: torch.Tensor, logv_axes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse (B,T,3) per-axis e2/logv to ISO-1 (B,T) for metrics.
    e2_sum = sum_i e2_i;  v_avg = mean_i exp(logv_i); logv_iso = log(v_avg).
    """
    e2_sum = e2_axes.sum(dim=-1)  # (B,T)
    logv_iso = torch.logsumexp(logv_axes, dim=-1) - math.log(float(e2_axes.size(-1)))  # (B,T)
    return e2_sum, logv_iso

@torch.no_grad()
def route_metrics_imu(e2: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                     logv_min: float, logv_max: float,
                     yvar: torch.Tensor | None = None) -> dict:
    """
    兼容两种输入：
      1) ISO-1: (B,T) 或 (B,T,1)
      2) DIAG-3: (B,T,3) —— 自动折叠为 ISO-1 再计算指标（df=3）
    """
    if (e2.dim() == 3 and e2.size(-1) == 3) or (logv.dim() == 3 and logv.size(-1) == 3):
        # 折叠到 (B,T)
        if e2.dim() == 3 and e2.size(-1) == 3 and logv.dim() == 3 and logv.size(-1) == 3:
            e2sum, logv_iso = _collapse_axes_diag(e2, logv)
        elif e2.dim() == 3 and e2.size(-1) == 3:
            # 只有 e2 是三轴；logv 应该是 (B,T,1) 或 (B,T)
            e2sum = e2.sum(dim=-1)
            logv_iso = logv.squeeze(-1) if logv.dim() == 3 and logv.size(-1) == 1 else logv
        else:
            # 只有 logv 是三轴；e2 应该是 (B,T) 或 (B,T,1)
            e2sum = e2.squeeze(-1) if e2.dim() == 3 and e2.size(-1) == 1 else e2
            logv_iso = torch.logsumexp(logv, dim=-1) - math.log(float(logv.size(-1)))
        e2sum, logv_iso, mask2 = _prepare_inputs(e2sum, logv_iso, mask)
        return _route_metrics(e2sum, logv_iso, mask2, logv_min, logv_max, df=3.0, yvar=yvar)
    else:
        # 原 ISO 路径
        e2sum, logv1, mask1 = _prepare_inputs(e2, logv, mask)
        return _route_metrics(e2sum, logv1, mask1, logv_min, logv_max, df=3.0, yvar=yvar)


@torch.no_grad()
def route_metrics_vis(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                     logv_min: float, logv_max: float,
                     yvar: torch.Tensor | None = None) -> dict:
    return _route_metrics(e2sum, logv, mask, logv_min, logv_max, df=2.0, yvar=yvar)

# ======= New tools and improved GNSS metrics =======
from typing import Dict, Tuple, List

def _spearman_no_scipy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    c = np.corrcoef(rx, ry)
    return float(c[0, 1])

def _student_t_z2_thresholds(nu: float, coverages=(0.68, 0.95)) -> Dict[str, float]:
    """
    双侧覆盖率阈值：给定 Student-t(ν)，返回 z^2=e^2/var 的阈值（不取 sqrt）。
    例如 p=0.95 -> |t|<=t_{(1+p)/2}，z2_thresh = t^2
    """
    try:
        from scipy.stats import t as scipy_t
        out = {}
        for p in coverages:
            q = scipy_t.ppf((1.0 + p) / 2.0, df=nu)  # 正分位
            out[f"{int(round(p*100))}"] = q * q
        return out
    except ImportError:
        # 如果没有scipy，使用近似值
        import math
        out = {}
        for p in coverages:
            # 简单近似：对于常见的nu值使用预计算的值
            if abs(nu - 3.0) < 0.1:
                if abs(p - 0.68) < 0.01:
                    q_squared = 1.32  # t_3(0.84)^2 ≈ 1.32
                elif abs(p - 0.95) < 0.01:
                    q_squared = 9.22  # t_3(0.975)^2 ≈ 9.22
                else:
                    q_squared = 1.0  # fallback
            else:
                # 其他nu值的粗略近似
                if abs(p - 0.68) < 0.01:
                    q_squared = 1.0 + 0.5 / nu
                elif abs(p - 0.95) < 0.01:
                    q_squared = 4.0 + 2.0 / nu
                else:
                    q_squared = 1.0
            out[f"{int(round(p*100))}"] = q_squared
        return out

def _reliability_by_var(e2: np.ndarray, v: np.ndarray, m: np.ndarray, nbuckets: int = 10) -> dict:
    mask = (m.reshape(-1) > 0.5)
    if mask.sum() == 0:
        return {"bucket_edges": [], "bucket_ez2": [], "bucket_var": [], "bucket_err2": [],
                "slope": 0.0, "spearman": 0.0}
    e2 = e2.reshape(-1)[mask]
    v  = v.reshape(-1)[mask]
    v  = np.clip(v, 1e-12, None)

    # 分位桶
    edges = np.quantile(v, np.linspace(0.0, 1.0, nbuckets + 1))
    idx = np.digitize(v, edges[1:-1], right=True)
    bucket_ez2, bucket_var, bucket_err2 = [], [], []
    for b in range(nbuckets):
        sel = (idx == b)
        if sel.sum() == 0:
            bucket_ez2.append(float("nan"))
            bucket_var.append(float("nan"))
            bucket_err2.append(float("nan"))
        else:
            bucket_ez2.append(float(np.mean(e2[sel] / v[sel])))
            bucket_var.append(float(np.mean(v[sel])))
            bucket_err2.append(float(np.mean(e2[sel])))

    # 相关性与斜率：log(err²) ~ a*log(var)+b
    X = np.log(v)
    Y = np.log(np.clip(e2, 1e-18, None))
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]  # slope, intercept
    spearman = _spearman_no_scipy(X, Y)

    return {
        "bucket_edges": [float(e) for e in edges],
        "bucket_ez2": bucket_ez2,
        "bucket_var": bucket_var,
        "bucket_err2": bucket_err2,
        "slope": float(a),
        "spearman": float(spearman),
    }

@torch.no_grad()
def route_metrics_gns_axes(e2_axes: torch.Tensor, logv_axes: torch.Tensor, mask_axes: torch.Tensor,
                           logv_min: float, logv_max: float, nu: float = 0.0) -> dict:
    """
    GNSS 各向异性评测（逐轴）：统一 t 口径 + 高斯口径对照，并输出可靠性曲线。
    - e2_axes: (B,T,3) 逐轴误差平方（ENU）
    - logv_axes: (B,T,3) 逐轴 log(var)
    - mask_axes: (B,T,3)
    """
    lv = torch.clamp(logv_axes, min=logv_min, max=logv_max)
    v  = torch.clamp(torch.exp(lv), min=1e-12)
    m  = mask_axes.float()
    z2 = (e2_axes / v)  # 1D z²
    den = m.sum(dim=(0,1)).clamp_min(1.0)  # (3,)

    # —— t 口径（若 nu>2）——
    out = {}
    if nu and nu > 2.0:
        target = float(nu / (nu - 2.0))  # E[t^2]
        thr_t = _student_t_z2_thresholds(nu, coverages=(0.68, 0.95))
        z2_mean_raw = (z2 * m).sum(dim=(0,1)) / den                       # (3,)
        z2_mean_norm = z2_mean_raw / target                                # (3,)
        cov68_t = ((z2 <= thr_t["68"]).float() * m).sum(dim=(0,1)) / den
        cov95_t = ((z2 <= thr_t["95"]).float() * m).sum(dim=(0,1)) / den
        out.update({
            "t_nu": nu,
            "t_target": target,
            "z2_mean_raw": z2_mean_raw.detach().cpu().tolist(),
            "z2_mean_norm": z2_mean_norm.detach().cpu().tolist(),
            "cov68_t": cov68_t.detach().cpu().tolist(),
            "cov95_t": cov95_t.detach().cpu().tolist(),
            "t_z2_thr68": thr_t["68"],
            "t_z2_thr95": thr_t["95"],
        })
    else:
        target = 1.0  # 回落到高斯口径

    # —— 高斯口径对照（χ² df=1）——
    cov68_g = ((z2 <= 1.0).float() * m).sum(dim=(0,1)) / den
    cov95_g = ((z2 <= 3.841).float() * m).sum(dim=(0,1)) / den
    z2_mean = (z2 * m).sum(dim=(0,1)) / den
    out.update({
        "z2_mean_gauss": z2_mean.detach().cpu().tolist(),
        "cov68_g": cov68_g.detach().cpu().tolist(),
        "cov95_g": cov95_g.detach().cpu().tolist(),
    })

    # —— 可靠性曲线（分桶） —— 
    e2_np = e2_axes.detach().cpu().numpy()
    v_np  = v.detach().cpu().numpy()
    m_np  = m.detach().cpu().numpy()
    rel = [_reliability_by_var(e2_np[...,i], v_np[...,i], m_np[...,i], nbuckets=10) for i in range(e2_np.shape[-1])]
    out["reliability"] = rel  # list of dicts per axis

    return out
