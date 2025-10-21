from __future__ import annotations
import torch
import torch.nn.functional as F
from metrics import C68_GAUSS
from typing import Optional, Dict

DF_AXES = 3  # 三轴

# ========== 高斯各向同性（备用，保持一致的 df 用法） ==========
def nll_gauss_iso(logv, e2sum, mask, logv_min=-8.0, logv_max=6.0):
    """
    logv: (B,T,1) or (B,T)  模型输出的 log(σ^2)（已经通过 tanh 有界）
    e2sum: (B,T)   误差平方三轴求和 e_x^2 + e_y^2 + e_z^2
    mask: (B,T)
    正确的 df=3 用法：只乘在 log 项，不要把 e^2 项除 3
    
    注意：如果模型使用 BoundedLogVar，logv 已经在 [logv_min, logv_max] 内
    """
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)  # (B,T)
    # 模型已经使用 tanh 参数化，无需再 clamp（但保留 clamp_min 防止数值问题）
    sig2 = torch.exp(logv).clamp_min(1e-12)       # (B,T)
    nll = 0.5 * (DF_AXES * logv + e2sum / sig2)   # (B,T)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp_min(1.0)

def nll_studentt_iso(logv, e2sum, mask, nu=5.0, logv_min=-8.0, logv_max=6.0):
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)  # (B,T)
    nu = max(float(nu), 2.1)
    sig2 = torch.exp(logv).clamp_min(1e-12)  # (B,T)
    term = torch.log1p(e2sum / (nu * sig2))  # (B,T)
    nll = 0.5 * (DF_AXES * logv + (nu + DF_AXES) * term)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp_min(1.0)

# ========== Student-t（对角/各轴） —— final 用法 ==========
@torch.no_grad()
def _safe_nu(nu: float) -> float:
    return max(nu, 2.1)  # ν>2 保证方差存在

def nll_studentt_diag_axes(logv_axes, e_axes, mask_axes, nu=5.0, logv_min=-8.0, logv_max=6.0,
                           lambda_center_axes=0.0, lambda_aniso_l2=0.0, lambda_tv=0.0,
                           lambda_cov68: float = 0.0, cov68_target: float = 0.68, cov68_tau: float = 0.1,
                           lambda_z2_low: float = 0.0, z2_low_margin: float = 0.0,
                           c_thresh_internal: float | None = None):
    """
    Student-t 对角三轴 NLL + 训练期轻约束（center / aniso-L2 / TV）
    
    logv_axes: (B,T,3)  每轴 log(σ^2)（各轴独立/对角，已经通过 tanh 有界）
    e_axes:    (B,T,3)  每轴误差 e_x, e_y, e_z
    mask_axes: (B,T,3) 或 (B,T)  掩码
    nu:        自由度，建议 3~8 之间；和 final 一致
    lambda_center_axes: z² 居中约束系数（默认0）
    lambda_aniso_l2:    各向异性 L2 约束系数（默认0）
    lambda_tv:          时间平滑（TV）约束系数（默认0）
    
    公式（省略常数）:
      nll_j = 0.5 * [ log σ_j^2 + (ν+1) * log(1 + e_j^2 / (ν σ_j^2)) ]
      对 j∈{x,y,z} 求和
    
    注意：如果模型使用 BoundedLogVar，logv_axes 已经在 [logv_min, logv_max] 内
    """
    nu = _safe_nu(float(nu))
    EPS = 1e-12
    
    # --- Base NLL（忽略常数项） ---
    s2 = torch.exp(logv_axes).clamp_min(EPS)                # (B,T,3)
    term = torch.log1p((e_axes**2) / (nu * s2))             # (B,T,3)
    nll_per_axis = 0.5 * (logv_axes + (nu + 1.0) * term)    # (B,T,3)
    
    # 处理掩码：支持 (B,T,3) 或 (B,T)
    if mask_axes.dim() == logv_axes.dim():
        m_bt = (mask_axes > 0.5).all(dim=-1)                # (B,T)
        m_axes = mask_axes.float()                          # (B,T,3)
    else:
        m_bt = (mask_axes > 0.5)                            # (B,T)
        m_axes = mask_axes.unsqueeze(-1).float().expand_as(e_axes)  # (B,T,3)
    # 至少一个轴有效的时间步
    valid_bt_any = (m_axes.sum(dim=-1) > 0.5)
    
    # 汇总到 (B,T)
    nll_bt = (nll_per_axis * m_axes).sum(dim=-1) / (m_axes.sum(dim=-1).clamp_min(1.0))
    if m_bt.any():
        nll = nll_bt[m_bt].mean()
    elif valid_bt_any.any():
        nll = nll_bt[valid_bt_any].mean()
    else:
        nll = nll_bt.new_tensor(0.0)
    
    # 预计算 v_eff 与 z2（用于各正则）
    v_eff = s2 * (nu / (nu - 2.0))
    z2_bt = ((e_axes**2) / v_eff.clamp_min(EPS)).sum(dim=-1) / 3.0

    # --- ① z² 居中约束（拉向 1）---
    if lambda_center_axes > 0.0:
        if m_bt.any():
            center_loss = (z2_bt[m_bt] - 1.0).pow(2).mean()
            nll = nll + lambda_center_axes * center_loss
            if not hasattr(nll_studentt_diag_axes, "_dbg_constraint"):
                nll_studentt_diag_axes._dbg_constraint = True
                print(f"[constraint] center_loss={center_loss.item():.6f}, weighted={lambda_center_axes * center_loss.item():.6f}")
    
    # --- ② 各向异性幅度 L2（抑制 a 爆涨）---
    if lambda_aniso_l2 > 0.0:
        from utils_sa3 import decompose_sa3_torch
        s, a_xy, a_z = decompose_sa3_torch(logv_axes)       # (B,T)
        if m_bt.any():
            aniso_l2 = ((a_xy.pow(2) + a_z.pow(2))[m_bt]).mean()
            nll = nll + lambda_aniso_l2 * aniso_l2
    
    # --- ③ 时间平滑（TV）---
    if lambda_tv > 0.0 and logv_axes.size(1) >= 2:
        from utils_sa3 import decompose_sa3_torch
        s, a_xy, a_z = decompose_sa3_torch(logv_axes)       # (B,T)
        # 只在 t 与 t-1 均有效的位置计算差分
        m = m_bt.float()                                    # (B,T)
        m_pair = (m[:, 1:] * m[:, :-1])                     # (B,T-1)
        denom = m_pair.sum().clamp_min(1.0)
        
        ds  = (s[:, 1:]   - s[:, :-1]).pow(2)
        da1 = (a_xy[:, 1:] - a_xy[:, :-1]).pow(2)
        da2 = (a_z[:, 1:]  - a_z[:, :-1]).pow(2)
        tv = ((ds + da1 + da2) * m_pair).sum() / denom
        nll = nll + lambda_tv * tv

    # --- ④ 覆盖率导向（平滑指示函数近似）---
    if lambda_cov68 > 0.0:
        tau = float(max(cov68_tau, 1e-4))
        thr = float(C68_GAUSS) if (c_thresh_internal is None) else float(c_thresh_internal)
        if m_bt.any():
            cover_prob = torch.sigmoid((thr - z2_bt[m_bt]) / tau)
            pred = cover_prob.mean()
            cov_loss = (pred - float(cov68_target))**2
            nll = nll + lambda_cov68 * cov_loss

    # --- ⑤ 软惩罚：抑制过小 z2 ---
    if lambda_z2_low > 0.0:
        margin = float(max(z2_low_margin, 0.0))
        if m_bt.any():
            low = F.relu((1.0 + margin) - z2_bt[m_bt])
            low_loss = low.pow(2).mean()
            nll = nll + lambda_z2_low * low_loss
    
    return nll

# ========== 兼容旧接口的包装函数 ==========
def loss_total(e2, logv, mask, logv_min=-12.0, logv_max=6.0, lambda_center=0.0):
    """
    兼容旧代码的统一接口
    e2: (B,T,3)  三轴误差平方
    logv: (B,T,1) or (B,T)  单通道 logσ²
    mask: (B,T)
    """
    e2sum = e2.sum(dim=-1)  # (B,T)
    nll = nll_gauss_iso(logv, e2sum, mask, logv_min, logv_max)
    
    # 可选的中心化正则项
    if lambda_center > 0:
        if logv.dim() == 3 and logv.size(-1) == 1:
            lv = logv.squeeze(-1)
        else:
            lv = logv
        # 模型已经使用 tanh 参数化，无需再 clamp
        v = torch.exp(lv).clamp_min(1e-12)
        z2 = e2sum / (v * DF_AXES)
        m = mask.float()
        z2_mean = (z2 * m).sum() / m.sum().clamp_min(1.0)
        center_loss = (z2_mean - 1.0).pow(2)
        return nll + lambda_center * center_loss, {"nll": nll.item(), "center": center_loss.item()}
    
    return nll, {"nll": nll.item(), "center": 0.0}

# ===== Plan-B auxiliary penalties =====
def aux_cov_z2_penalties(
    z2_bt: torch.Tensor,
    *,
    lambda_cov68: float,
    cov68_target: float,
    cov68_tau: float,
    lambda_z2_low: float,
    z2_low_margin: float,
    use_ema_z2_stats: bool,
    ema_state: Optional[Dict],
    lambda_z2_var: float,
    c_thresh: Optional[float] = None,
):
    loss = z2_bt.new_tensor(0.0)
    stats = {}
    if z2_bt.numel() == 0:
        return loss, stats
    if float(lambda_cov68) > 0.0:
        tau = max(float(cov68_tau), 1e-4)
        thr = float(c_thresh) if (c_thresh is not None) else float(C68_GAUSS)
        cover_prob = torch.sigmoid((thr - z2_bt) / tau)
        cov_pred = cover_prob.mean()
        cov_loss = (cov_pred - float(cov68_target)) ** 2
        loss = loss + float(lambda_cov68) * cov_loss
        stats["cov68_pred"] = cov_pred.detach()
    if float(lambda_z2_low) > 0.0:
        low = F.relu((1.0 + float(z2_low_margin)) - z2_bt)
        low_loss = low.pow(2).mean()
        loss = loss + float(lambda_z2_low) * low_loss
        stats["z2_low_loss"] = low_loss.detach()
    if float(lambda_z2_var) > 0.0:
        if use_ema_z2_stats and (ema_state is not None) and (ema_state.get("z2_var") is not None):
            z2_var = ema_state["z2_var"]
            if not isinstance(z2_var, torch.Tensor):
                z2_var = torch.tensor(float(z2_var), device=z2_bt.device, dtype=z2_bt.dtype)
        else:
            z2_var = z2_bt.var(unbiased=False) if z2_bt.numel() > 0 else z2_bt.new_tensor(0.0)
        loss = loss + float(lambda_z2_var) * z2_var
        stats["z2_var"] = z2_var.detach()
    return loss, stats

def aux_scale_on_s(
    logv_axes: torch.Tensor,
    e_axes: torch.Tensor,
    nu: float,
    lambda_s_scale: float,
    mask_axes: Optional[torch.Tensor] = None,
):
    if float(lambda_s_scale) <= 0.0:
        return logv_axes.new_tensor(0.0), {}
    if mask_axes is not None:
        if mask_axes.dim() == e_axes.dim():
            valid = (mask_axes > 0.5).all(dim=-1)
        else:
            valid = (mask_axes > 0.5)
    else:
        valid = torch.ones_like(e_axes[..., 0], dtype=torch.bool)
    e2 = e_axes.pow(2)
    z2_bt = e2.sum(dim=-1) / 3.0
    z2_flat = z2_bt[valid]
    if z2_flat.numel() == 0:
        return logv_axes.new_tensor(0.0), {}
    e2_mean = z2_flat.mean()
    v_eff_tgt = e2_mean.clamp_min(1e-12)
    if (nu is not None) and (float(nu) > 2.0):
        sigma2_tgt = v_eff_tgt * (float(nu) - 2.0) / float(nu)
    else:
        sigma2_tgt = v_eff_tgt
    s_tgt = torch.log(sigma2_tgt).detach()
    from utils_sa3 import decompose_sa3_torch
    s, a_xy, a_z = decompose_sa3_torch(logv_axes)
    s_mean = s[valid].mean() if valid.any() else s.mean()
    scale_loss = (s_mean - s_tgt).pow(2)
    return float(lambda_s_scale) * scale_loss, {"s_mean": s_mean.detach(), "s_tgt": s_tgt.detach()}
