from __future__ import annotations
import numpy as np
import torch
import math

DF_AXES = 3
# ===== 高斯分布阈值 =====
# 正确的覆盖率阈值：χ²(df=3) 分位数除以 df
# χ²₀.₆₈(df=3) ≈ 3.50588 → C68 = 3.50588/3 ≈ 1.1686
# χ²₀.₉₅(df=3) ≈ 7.81473 → C95 = 7.81473/3 ≈ 2.6049
C68_GAUSS = 1.1686
C95_GAUSS = 2.6049

# 向后兼容
C68 = C68_GAUSS
C95 = C95_GAUSS

def z2_cov_studentt_diag_axes(logv_axes, e_axes, mask_axes, nu=5.0, logv_min=-8.0, logv_max=6.0):
    """
    logv_axes: (..,3)  每轴 log(σ^2)
    e_axes:    (..,3)  每轴误差
    mask_axes: (..,3) 或 (..,)  每轴掩码
    有效方差 v_eff = σ^2 * ν/(ν-2)
    
    ✅ 正确口径：逐轴归一化 → 求和 → 除 df=3
    z² = [(e_x²/v_x + e_y²/v_y + e_z²/v_z) / 3]
    
    ⚠️ 使用 Student-t 专用阈值，不是高斯阈值！
    """
    # 支持 torch/np 混用
    if "torch" in str(type(logv_axes)):
        import torch
        logv_axes = logv_axes.detach().cpu().numpy()
    if "torch" in str(type(e_axes)):
        import torch
        e_axes = e_axes.detach().cpu().numpy()
    if "torch" in str(type(mask_axes)):
        import torch
        mask_axes = mask_axes.detach().cpu().numpy()
    
    nu = max(float(nu), 2.1)
    logv_axes = np.clip(logv_axes, logv_min, logv_max)
    s2 = np.exp(logv_axes)                      # (..,3)
    v_eff = s2 * (nu / (nu - 2.0))              # (..,3)
    
    # ✅ 逐轴归一化 → 求和 → 除 df
    z2_full = ((e_axes**2) / np.clip(v_eff, 1e-12, None)).sum(axis=-1) / DF_AXES
    
    # 仅在有效掩码内统计
    if mask_axes.ndim == logv_axes.ndim:
        m_all = (mask_axes > 0.5).all(axis=-1)
    else:
        m_all = (mask_axes > 0.5)
    
    z = z2_full[m_all] if np.any(m_all) else np.array([])
    z2_mean = float(z.mean()) if z.size > 0 else 0.0
    
    # Use Student-t thresholds for diag-axes Student-t
    thr68 = studentt_z2_threshold(0.68, nu, DF_AXES)
    thr95 = studentt_z2_threshold(0.95, nu, DF_AXES)
    cov68 = float((z <= thr68).mean()) if z.size > 0 else 0.0
    cov95 = float((z <= thr95).mean()) if z.size > 0 else 0.0
    return z2_mean, cov68, cov95

def spearmanr_np(a, b):
    """Spearman 相关系数"""
    if a.size < 3 or b.size < 3:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

def route_metrics_imu(e2: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                      logv_min=-12.0, logv_max=6.0):
    """
    单通道σ² + 三轴e²之和/df
    e2: (B,T,3) or (B,T)
    logv: (B,T,1) or (B,T)
    mask: (B,T)
    """
    if isinstance(e2, torch.Tensor):  e2 = e2.detach().cpu().numpy()
    if isinstance(logv, torch.Tensor): logv = logv.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()

    # 处理e2: 如果是3轴，求和
    if e2.ndim == 3 and e2.shape[-1] == 3:
        e2sum = e2.sum(axis=-1)  # (B,T)
    elif e2.ndim == 3 and e2.shape[-1] == 1:
        e2sum = e2.squeeze(-1)  # (B,T)
    else:
        e2sum = e2  # (B,T)
    
    # 处理logv
    if logv.ndim == 3 and logv.shape[-1] == 1:
        logv = logv.squeeze(-1)  # (B,T)
    
    logv = np.clip(logv, logv_min, logv_max)
    v = np.exp(logv)
    m = mask > 0.5
    
    # z² 计算
    z2 = (e2sum[m] / (v[m] * DF_AXES)) if m.sum() > 0 else np.array([])
    z2_mean = float(z2.mean()) if z2.size > 0 else 0.0
    
    # 覆盖率（使用正确的阈值）
    cov68 = float((z2 <= C68).mean()) if z2.size > 0 else 0.0
    cov95 = float((z2 <= C95).mean()) if z2.size > 0 else 0.0
    
    # Spearman 相关性
    sp = spearmanr_np(e2sum[m].reshape(-1), v[m].reshape(-1)) if m.sum() > 0 else 0.0
    
    # 饱和率：统计接近边界的比例（距离边界 < 5% 范围）
    # 对于 tanh 参数化，接近边界意味着 logv 接近 logv_min 或 logv_max
    boundary_margin = 0.05 * (logv_max - logv_min)
    sat_min = float(((logv <= logv_min + boundary_margin)[m]).mean()) if m.sum() > 0 else 0.0
    sat_max = float(((logv >= logv_max - boundary_margin)[m]).mean()) if m.sum() > 0 else 0.0
    sat = sat_min + sat_max
    
    return {
        "z2_mean": z2_mean,
        "cov68": cov68,
        "cov95": cov95,
        "spearman": sp,
        "saturation": sat,
        "sat_min": sat_min,
        "sat_max": sat_max
    }

# ===== Student-t thresholds for z^2 =====
def studentt_z2_threshold(q: float, nu: float, df_axes: int) -> float:
    q = float(q)
    nu = max(float(nu), 2.1)
    d1 = float(int(df_axes))
    d2 = float(nu)
    a = 0.5 * d1
    b = 0.5 * d2
    q = float(np.clip(q, 1e-9, 1 - 1e-9))

    def betacf(a_: float, b_: float, x_: float) -> float:
        MAXIT = 200
        EPS = 1e-12
        FPMIN = 1e-300
        m2 = 0
        aa = 0.0
        c = 1.0
        d = 1.0 - (a_ + b_) * x_ / (a_ + 1.0)
        d = 1.0 / max(d, FPMIN)
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b_ - m) * x_ / ((a_ + m2 - 1.0) * (a_ + m2))
            d = 1.0 + aa * d
            d = 1.0 / max(d, FPMIN)
            c = 1.0 + aa / max(c, FPMIN)
            h *= d * c
            aa = -(a_ + m) * (a_ + b_ + m) * x_ / ((a_ + m2) * (a_ + m2 + 1.0))
            d = 1.0 + aa * d
            d = 1.0 / max(d, FPMIN)
            c = 1.0 + aa / max(c, FPMIN)
            delh = d * c
            h *= delh
            if abs(delh - 1.0) < EPS:
                break
        return h

    def betainc_reg(x_: float, a_: float, b_: float) -> float:
        if x_ <= 0.0:
            return 0.0
        if x_ >= 1.0:
            return 1.0
        bt = math.exp(math.lgamma(a_ + b_) - math.lgamma(a_) - math.lgamma(b_) + a_ * math.log(x_) + b_ * math.log(1.0 - x_))
        if x_ < (a_ + 1.0) / (a_ + b_ + 2.0):
            return bt * betacf(a_, b_, x_) / a_
        else:
            return 1.0 - bt * betacf(b_, a_, 1.0 - x_) / b_

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = betainc_reg(mid, a, b)
        if cdf < q:
            lo = mid
        else:
            hi = mid
    y_q = 0.5 * (lo + hi)
    denom = max(1.0 - y_q, 1e-12)
    x_q = (d2 * y_q) / (d1 * denom)
    c_q = ((nu - 2.0) / nu) * x_q
    return float(c_q)
