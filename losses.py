from __future__ import annotations
import torch
import torch.nn.functional as F

def _ste_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Forward: clamp，Backward: identity（避免梯度被硬截断）"""
    y = torch.clamp(x, min=lo, max=hi)
    return x + (y - x).detach()

def nll_iso3_e2(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                logv_min: float=-16.0, logv_max: float=6.0) -> torch.Tensor:
    """
    Negative log-likelihood using pre-pooled squared error sum.
    e2sum: (B,T,1) or (B,T)
    logv : (B,T,1) or (B,T)
    mask : (B,T)
    """
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    lv = _ste_clamp(logv, logv_min, logv_max)
    # NaN-safe + 按掩码清理
    e2sum = torch.nan_to_num(e2sum, nan=0.0, posinf=0.0, neginf=0.0)
    e2sum = torch.where(mask.float() > 0.5, e2sum, torch.zeros_like(e2sum))
    v = torch.exp(lv).clamp_min(1e-12)
    nll = 0.5 * (3.0 * lv + e2sum / v)
    m = mask.float()
    return (nll * m).sum() / torch.clamp(m.sum(), min=1.0)

def nll_iso2_e2(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                logv_min: float = -16.0, logv_max: float = 6.0) -> torch.Tensor:
    """Isotropic 2D negative log-likelihood for vision route."""
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    lv = _ste_clamp(logv, logv_min, logv_max)
    e2sum = torch.nan_to_num(e2sum, nan=0.0, posinf=0.0, neginf=0.0)
    e2sum = torch.where(mask.float() > 0.5, e2sum, torch.zeros_like(e2sum))
    v = torch.exp(lv).clamp_min(1e-12)
    m = mask.float()
    nll = 0.5 * (2.0 * lv + e2sum / v)
    return (nll * m).sum() / torch.clamp(m.sum(), min=1.0)


def mse_anchor_1d(logv: torch.Tensor, y_var: torch.Tensor, mask: torch.Tensor, lam: float=1e-3) -> torch.Tensor:
    """
    Optional scale anchor on log-variance.
    y_var: (B,T) anchor variance (>=0), will be log() with clamp.
    """
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    y = torch.clamp(y_var, min=1e-12).log()
    m = mask.float()
    se = (logv - y)**2 * m
    return lam * se.sum() / torch.clamp(m.sum(), min=1.0)

def nll_diag_axes(e2_axes: torch.Tensor, logv_axes: torch.Tensor, mask_axes: torch.Tensor,
                  logv_min: float=-16.0, logv_max: float=6.0) -> torch.Tensor:
    """
    各向异性对角高斯 NLL（逐轴）。适用于 GNSS ENU 三轴。
    e2_axes  : (B,T,3)   每轴误差平方
    logv_axes: (B,T,3)   每轴 log(σ^2)
    mask_axes: (B,T,3)   每轴有效掩码
    """
    lv = _ste_clamp(logv_axes, logv_min, logv_max)
    inv_v = torch.exp(-lv)                 # (B,T,3)
    nll = 0.5 * (e2_axes * inv_v + lv)    # (B,T,3)
    m = mask_axes.float()
    num = (nll * m).sum()
    den = torch.clamp(m.sum(), min=1.0)
    return num / den

def nll_diag_axes_weighted(e2_axes: torch.Tensor, logv_axes: torch.Tensor, mask_axes: torch.Tensor,
                           axis_w: torch.Tensor=None,
                           logv_min: float=-16.0, logv_max: float=6.0):
    """
    各向异性对角高斯 NLL（逐轴）+ 按轴权重。
    e2_axes, logv_axes, mask_axes: (B,T,3)
    axis_w: (3,) 归一到均值=1 更稳（外部可先做归一化）
    """
    lv = _ste_clamp(logv_axes, logv_min, logv_max)
    e2_axes = torch.nan_to_num(e2_axes, nan=0.0, posinf=0.0, neginf=0.0)
    e2_axes = torch.where(mask_axes.float() > 0.5, e2_axes, torch.zeros_like(e2_axes))
    inv_v = torch.exp(-lv)                    # (B,T,3)
    nll_axes = 0.5 * (e2_axes * inv_v + lv)  # (B,T,3)
    m = mask_axes.float()
    num = nll_axes.mul(m).sum(dim=(0,1))      # (3,)
    den = m.sum(dim=(0,1)).clamp_min(1.0)     # (3,)
    per_axis = num / den                       # (3,)
    if axis_w is None:
        axis_w = torch.ones_like(per_axis)
    # 归一到均值=1，便于 lr 稳定
    axis_w = axis_w * (3.0 / axis_w.sum().clamp_min(1e-6))
    return (per_axis * axis_w).sum(), per_axis.detach()

def nll_studentt_diag_axes(e2_axes: torch.Tensor, logv_axes: torch.Tensor, mask_axes: torch.Tensor,
                           nu: float = 3.0, logv_min: float = -16.0, logv_max: float = 6.0):
    """
    各向异性对角 Student-t NLL（逐轴）。对异常值更稳健。
    e2_axes  : (B,T,3)   每轴误差平方
    logv_axes: (B,T,3)   每轴 log(σ^2)
    mask_axes: (B,T,3)   每轴有效掩码
    nu       : 自由度参数（越小越重尾，越稳健）
    """
    lv = _ste_clamp(logv_axes, logv_min, logv_max)
    e2_axes = torch.nan_to_num(e2_axes, nan=0.0, posinf=0.0, neginf=0.0)
    e2_axes = torch.where(mask_axes.float() > 0.5, e2_axes, torch.zeros_like(e2_axes))
    v  = torch.exp(lv).clamp_min(1e-12)
    m  = mask_axes.float()
    # Student-t NLL（省略常数项）：0.5*log(v) + 0.5*(nu+1)*log(1 + e2/(nu*v))
    nll = 0.5*lv + 0.5*(nu + 1.0) * torch.log1p(e2_axes / (v * nu))
    num = (nll * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def mse_anchor_axes(logv_axes: torch.Tensor, y_var_axes: torch.Tensor, mask_axes: torch.Tensor, lam: float=1e-4) -> torch.Tensor:
    """
    GNSS 逐轴 log-variance 的软锚：把预测 logv 轻微拉向 log(vendor^2)。
    logv_axes   : (B,T,3)
    y_var_axes  : (B,T,3)  —— 逐轴 vendor 报告的方差（不是标准差）
    mask_axes   : (B,T,3)
    """
    lv = logv_axes
    y  = torch.clamp(y_var_axes, min=1e-12).log()
    m  = mask_axes.float()
    se = (lv - y)**2 * m
    return lam * se.sum() / torch.clamp(m.sum(), min=1.0)

def adaptive_nll_loss(logv: torch.Tensor, e2: torch.Tensor, mask: torch.Tensor, 
                      use_step_labels, y_anchor: torch.Tensor = None,
                      logv_min: float = -16.0, logv_max: float = 6.0,
                      route: str = "acc") -> torch.Tensor:
    """
    自适应NLL损失函数：
    - 如果有步级标签，使用纯步级NLL
    - 否则使用窗口锚点+轻中心化
    
    Args:
        logv: (B,T,D) 模型输出的log方差
        e2: (B,T,D) 步级误差平方 或 从窗口标签扩展的误差平方
        mask: (B,T) 掩码
        use_step_labels: bool or tensor 是否使用步级标签
        y_anchor: (B,D) 窗口级标签（仅在非步级模式使用）
        route: str 路由类型 ("acc", "gyr", "vis")
    """
    # 处理use_step_labels可能是tensor的情况
    if isinstance(use_step_labels, torch.Tensor):
        use_step_labels = bool(use_step_labels.item())
    
    # --- 情况 A：有步级标签（推荐） ---
    if use_step_labels:
        if route in ("acc", "gyr"):
            # IMU: 使用3轴ISO损失
            if e2.size(-1) == 3:
                # 对于三轴数据，求和后除以3
                e2sum = e2.sum(dim=-1)  # (B,T)
                if logv.size(-1) == 3:
                    # 如果模型输出也是3轴，取平均
                    logv_avg = logv.mean(dim=-1)  # (B,T)
                else:
                    logv_avg = logv.squeeze(-1)  # (B,T)
                return nll_iso3_e2(e2sum, logv_avg, mask, logv_min, logv_max)
            else:
                # 单轴模式
                e2sum = e2.squeeze(-1)  # (B,T)
                logv_avg = logv.squeeze(-1)  # (B,T)
                return nll_iso3_e2(e2sum, logv_avg, mask, logv_min, logv_max)
        else:  # vis
            # 视觉：使用2轴ISO损失
            e2sum = e2.squeeze(-1)  # (B,T)
            logv_avg = logv.squeeze(-1)  # (B,T)
            return nll_iso2_e2(e2sum, logv_avg, mask, logv_min, logv_max)
    
    # --- 情况 B：只有窗口标签（旧 EUROC npz 的权宜之计） ---
    else:
        # 基础步级NLL（使用扩展的e2）
        e2sum = e2.squeeze(-1) if e2.size(-1) == 1 else e2.sum(dim=-1)
        logv_avg = logv.squeeze(-1) if logv.size(-1) == 1 else logv.mean(dim=-1)
        
        if route in ("acc", "gyr"):
            nll = nll_iso3_e2(e2sum, logv_avg, mask, logv_min, logv_max)
        else:  # vis
            nll = nll_iso2_e2(e2sum, logv_avg, mask, logv_min, logv_max)
        
        # "中心化"：轻微约束 E[z²] -> 1，防止收敛到常数
        sig2 = torch.exp(logv_avg).clamp_min(1e-12)
        z2 = e2sum / sig2  # 标准化误差
        m = mask.float()
        z2_mean = (z2 * m).sum() / torch.clamp(m.sum(), min=1.0)
        loss_center = (z2_mean - 1.0).pow(2)
        
        # "窗口尺度锚"：用时间平均的σ²与Y_anchor对齐
        if y_anchor is not None:
            sig2_win = sig2.mean(dim=1)  # (B,)
            if route in ("acc", "gyr") and y_anchor.size(-1) == 3:
                # 三轴标签求平均
                y_target = y_anchor.mean(dim=-1)  # (B,)
            else:
                y_target = y_anchor.squeeze(-1)  # (B,)
            anchor_loss = F.smooth_l1_loss(sig2_win, y_target)
        else:
            anchor_loss = torch.tensor(0.0, device=logv.device)
        
        return nll + 1e-3 * loss_center + 1e-4 * anchor_loss