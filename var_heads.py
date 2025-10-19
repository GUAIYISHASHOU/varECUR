"""
Variance head implementations for IMU uncertainty estimation.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Sa3VarHead(nn.Module):
    """
    三轴拉扯参数化的方差头 (Scale + Anisotropy 3-axis)
    
    输出：
    - s:   整体尺度（log-variance 的平均）
    - a_xy: 控制 x<->y 轴的相对粗细（log-std 差）
    - a_z:  控制 {x,y} 平面 vs z 轴的相对粗细（log-std 差）
    
    重构公式（闭式可逆，保持 Lx+Ly+Lz=3s）:
      Lx = s + a_xy + (2/3)*a_z
      Ly = s - a_xy + (2/3)*a_z
      Lz = s - (4/3)*a_z
    
    优势：
    - 解耦尺度和形状
    - 使用 tanh 限制各向异性幅度，避免极端拉扯
    - 保持三轴 log-variance 之和为常数（3s）
    """
    def __init__(self, in_dim: int, logv_min: float = -8.0, logv_max: float = 6.0,
                 use_tanh: bool = True, kappa_xy: float = 2.0, kappa_z: float = 2.0):
        """
        Args:
            in_dim: 输入特征维度
            logv_min: log-variance 最小值
            logv_max: log-variance 最大值
            use_tanh: 是否对 a_xy, a_z 使用 tanh 限幅
            kappa_xy: a_xy 的最大幅度（log-std 单位）
            kappa_z: a_z 的最大幅度（log-std 单位）
        """
        super().__init__()
        self.head_s = nn.Linear(in_dim, 1)      # 整体尺度
        self.head_axi = nn.Linear(in_dim, 2)    # [a_xy, a_z] 各向异性
        
        self.logv_min = logv_min
        self.logv_max = logv_max
        self.use_tanh = use_tanh
        self.kappa_xy = float(kappa_xy)
        self.kappa_z = float(kappa_z)
        
        # 平滑有界参数化（避免硬 clamp 截断梯度）
        self.s_mid = 0.5 * (logv_min + logv_max)
        self.s_rad = 0.5 * (logv_max - logv_min)
    
    def forward(self, feat):
        """
        Args:
            feat: (B,T,H) 或 (B,H) 特征
        
        Returns:
            logv: (B,T,3) 三轴 log-variance
            s:    (B,T) 或 (B,) 尺度
            a:    (B,T,2) 或 (B,2) [a_xy, a_z]
        """
        squeeze_T = False
        if feat.dim() == 2:
            feat = feat.unsqueeze(1)  # (B,1,H)
            squeeze_T = True
        
        # 尺度 s 使用平滑有界映射（tanh），避免硬 clamp 截断梯度
        raw_s = self.head_s(feat).squeeze(-1)   # (B,T) 无界
        s = self.s_mid + self.s_rad * torch.tanh(raw_s)  # (B,T) 平滑有界
        
        # 各向异性分量使用 tanh 限幅
        raw_a_xy, raw_a_z = self.head_axi(feat).unbind(-1)  # (B,T), (B,T)
        if self.use_tanh:
            a_xy = torch.tanh(raw_a_xy) * self.kappa_xy
            a_z = torch.tanh(raw_a_z) * self.kappa_z
        else:
            a_xy = raw_a_xy
            a_z = raw_a_z
        
        # 重构三轴 log-variance（保持 Lx+Ly+Lz=3s）
        Lx = s + a_xy + (2.0/3.0) * a_z
        Ly = s - a_xy + (2.0/3.0) * a_z
        Lz = s - (4.0/3.0) * a_z
        
        logv = torch.stack([Lx, Ly, Lz], dim=-1)  # (B,T,3)
        
        # 不再使用硬 clamp！梯度现在可以在边界处流动
        
        # 调试打印：首次前向传播时检查边界饱和度
        if not hasattr(self, "_dbg_once"):
            self._dbg_once = True
            with torch.no_grad():
                clamp_min = (logv <= self.logv_min + 1e-6).float().mean().item()
                clamp_max = (logv >= self.logv_max - 1e-6).float().mean().item()
                print(f"[sa3] boundary saturation: min={clamp_min:.2%}, max={clamp_max:.2%}")
                print(f"[sa3] logv range: [{logv.min().item():.3f}, {logv.max().item():.3f}]")
        
        if squeeze_T:
            logv = logv.squeeze(1)  # (B,3)
            s = s.squeeze(1)        # (B,)
            a_xy = a_xy.squeeze(1)  # (B,)
            a_z = a_z.squeeze(1)    # (B,)
        
        a = torch.stack([a_xy, a_z], dim=-1)
        return logv, s, a

