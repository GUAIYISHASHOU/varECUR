"""
Utility functions for Sa3 (Scale + Anisotropy 3-axis) parametrization.
"""
from __future__ import annotations
import numpy as np
import torch


def decompose_sa3_np(logv_axes):
    """
    分解三轴 log-variance 为尺度和各向异性分量（NumPy 版本）
    
    输入 (N,3) 的 [Lx, Ly, Lz]，输出：
      s    = (Lx + Ly + Lz) / 3        # 整体尺度
      a_xy = (Lx - Ly) / 2              # x<->y 拉扯
      a_z  = ((Lx + Ly)/2 - Lz) / 2     # 平面<->z 拉扯
    
    Args:
        logv_axes: (..., 3) 三轴 log-variance
    
    Returns:
        s: (...,) 整体尺度
        a_xy: (...,) x-y 各向异性
        a_z: (...,) 平面-z 各向异性
    """
    Lx = logv_axes[..., 0]
    Ly = logv_axes[..., 1]
    Lz = logv_axes[..., 2]
    
    s = (Lx + Ly + Lz) / 3.0
    a_xy = (Lx - Ly) / 2.0
    a_z = ((Lx + Ly) / 2.0 - Lz) / 2.0
    
    return s, a_xy, a_z


def decompose_sa3_torch(logv_axes):
    """
    分解三轴 log-variance 为尺度和各向异性分量（PyTorch 版本）
    
    Args:
        logv_axes: (..., 3) 三轴 log-variance
    
    Returns:
        s: (...,) 整体尺度
        a_xy: (...,) x-y 各向异性
        a_z: (...,) 平面-z 各向异性
    """
    Lx = logv_axes[..., 0]
    Ly = logv_axes[..., 1]
    Lz = logv_axes[..., 2]
    
    s = (Lx + Ly + Lz) / 3.0
    a_xy = (Lx - Ly) / 2.0
    a_z = ((Lx + Ly) / 2.0 - Lz) / 2.0
    
    return s, a_xy, a_z


def reconstruct_sa3_np(s, a_xy, a_z):
    """
    从尺度和各向异性分量重构三轴 log-variance（NumPy 版本）
    
    重构公式：
      Lx = s + a_xy + (2/3)*a_z
      Ly = s - a_xy + (2/3)*a_z
      Lz = s - (4/3)*a_z
    
    Args:
        s: (...,) 整体尺度
        a_xy: (...,) x-y 各向异性
        a_z: (...,) 平面-z 各向异性
    
    Returns:
        logv_axes: (..., 3) 三轴 log-variance
    """
    Lx = s + a_xy + (2.0/3.0) * a_z
    Ly = s - a_xy + (2.0/3.0) * a_z
    Lz = s - (4.0/3.0) * a_z
    
    return np.stack([Lx, Ly, Lz], axis=-1)


def reconstruct_sa3_torch(s, a_xy, a_z):
    """
    从尺度和各向异性分量重构三轴 log-variance（PyTorch 版本）
    
    Args:
        s: (...,) 整体尺度
        a_xy: (...,) x-y 各向异性
        a_z: (...,) 平面-z 各向异性
    
    Returns:
        logv_axes: (..., 3) 三轴 log-variance
    """
    Lx = s + a_xy + (2.0/3.0) * a_z
    Ly = s - a_xy + (2.0/3.0) * a_z
    Lz = s - (4.0/3.0) * a_z
    
    return torch.stack([Lx, Ly, Lz], dim=-1)


def student_t_nll_np(e_axes, logv_axes, nu=5.0):
    """
    计算 Student-t 分布的负对数似然（样本级，忽略常数项）
    
    Args:
        e_axes: (N, 3) 三轴误差
        logv_axes: (N, 3) 三轴 log-variance
        nu: Student-t 自由度
    
    Returns:
        nll: (N,) 每个样本的 NLL
    """
    s2 = np.exp(logv_axes)
    
    if nu is not None and nu > 2.0:
        # Student-t NLL（忽略常数项）
        inv = 1.0 / np.clip(s2, 1e-12, None)
        quad = (e_axes**2) * inv
        # 0.5*(nu+1)*log(1 + quad/nu) + 0.5*log(s2)
        nll = 0.5 * (nu + 1.0) * np.log1p(quad / nu).sum(-1) + 0.5 * np.log(s2).sum(-1)
    else:
        # 高斯 NLL
        inv = 1.0 / np.clip(s2, 1e-12, None)
        nll = 0.5 * (np.log(s2).sum(-1) + (e_axes**2 * inv).sum(-1))
    
    return nll


def tune_gamma_on_oof(logv_axes, e_axes, gamma_grid=None, nu=5.0):
    """
    在 OOF 数据上优化各向异性强度 γ（保持尺度 s 不变）
    
    通过网格搜索找到最优的 γ，使得 NLL 最小：
      logv_calibrated = reconstruct_sa3(s, γ*a_xy, γ*a_z)
    
    Args:
        logv_axes: (N, 3) 预测的三轴 log-variance
        e_axes: (N, 3) 真实误差
        gamma_grid: γ 的搜索网格，默认 [0.5, 2.0] 步长 0.02
        nu: Student-t 自由度
    
    Returns:
        best_gamma: 最优的 γ 值
    """
    if gamma_grid is None:
        gamma_grid = np.linspace(0.5, 2.0, 76)  # 0.5~2.0 步长 0.02
    
    s, a_xy, a_z = decompose_sa3_np(logv_axes)
    
    best_gamma = 1.0
    best_nll = 1e18
    
    for gamma in gamma_grid:
        # 缩放各向异性分量
        logv_scaled = reconstruct_sa3_np(s, gamma * a_xy, gamma * a_z)
        
        # 计算 NLL
        nll = student_t_nll_np(e_axes, logv_scaled, nu).mean()
        
        if nll < best_nll:
            best_nll = nll
            best_gamma = gamma
    
    return float(best_gamma)


def compute_aniso_stats(logv_axes):
    """
    计算各向异性分量的统计量（用于训练监控）
    
    Args:
        logv_axes: (..., 3) 三轴 log-variance
    
    Returns:
        dict: 包含 s_mean, a_xy_std, a_z_std 等统计量
    """
    if isinstance(logv_axes, torch.Tensor):
        logv_axes = logv_axes.detach().cpu().numpy()
    
    # 展平为 (N, 3)
    shape = logv_axes.shape
    logv_flat = logv_axes.reshape(-1, 3)
    
    s, a_xy, a_z = decompose_sa3_np(logv_flat)
    
    return {
        "s_mean": float(s.mean()),
        "s_std": float(s.std()),
        "a_xy_mean": float(a_xy.mean()),
        "a_xy_std": float(a_xy.std()),
        "a_z_mean": float(a_z.mean()),
        "a_z_std": float(a_z.std()),
    }
