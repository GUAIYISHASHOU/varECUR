from __future__ import annotations
import math
import torch
import numpy as np

# 统一的自由度定义 (VIS removed)
DF_BY_ROUTE = {"acc": 3, "gyr": 3}

def z2_from_residual(residual, logvar, df):
    """统一的z²计算函数
    Args:
        residual: 残差张量，最后一维是空间维度 
        logvar: 对数方差
        df: 自由度
    """
    sigma2 = torch.exp(logvar)
    e2 = (residual**2).sum(dim=-1) / sigma2        # 2D: (rx^2+ry^2)/σ^2
    return e2 / df                                  # 统一这里除以 df


def _prepare_inputs(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor):
    # VIS 2D diagonal mode removed
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if logv.dim() != 2 or e2sum.dim() != 2 or mask.dim() != 2:
        raise ValueError(f"Expected (B,T) tensors after squeeze, got logv:{logv.shape}, e2sum:{e2sum.shape}, mask:{mask.shape}")
    return e2sum, logv, mask


@torch.no_grad()
def _route_metrics(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                  logv_min: float, logv_max: float, df: float,
                  yvar: torch.Tensor | None = None) -> dict:
    # VIS 2D diagonal mode removed
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    
    # squeeze (B,T,1) -> (B,T)
    if e2sum.dim()==3 and e2sum.size(-1)==1: 
        e2sum = e2sum.squeeze(-1)
    if mask.dim()==3  and mask.size(-1)==1:  
        mask  = mask.squeeze(-1)

    # (B,1) -> (B,T) 以 e2sum 的时间维为准
    if e2sum.dim()==2 and mask.dim()==2 and mask.size(1)==1 and e2sum.size(1)>1:
        mask = mask.expand(-1, e2sum.size(1))
    
    e2sum, logv, mask = _prepare_inputs(e2sum, logv, mask)
    
    logv = torch.clamp(logv, min=logv_min, max=logv_max)
    var = torch.clamp(torch.exp(logv), min=1e-12)
    m = mask.float()

    # NaN-safe for e2sum and mask-out invalids
    e2sum = torch.nan_to_num(e2sum, nan=0.0, posinf=0.0, neginf=0.0)
    e2sum = torch.where(m > 0.5, e2sum, torch.zeros_like(e2sum))
    z2 = e2sum / (var * float(df))
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
    df = DF_BY_ROUTE["acc"]  # IMU默认使用acc的自由度
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
        return _route_metrics(e2sum, logv_iso, mask2, logv_min, logv_max, df=df, yvar=yvar)
    else:
        # 原 ISO 路径
        e2sum, logv1, mask1 = _prepare_inputs(e2, logv, mask)
        return _route_metrics(e2sum, logv1, mask1, logv_min, logv_max, df=df, yvar=yvar)


