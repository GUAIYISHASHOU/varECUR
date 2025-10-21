from __future__ import annotations
import torch
import torch.nn as nn


class BoundedLogVar(nn.Module):
    """
    Tanh-based bounded parametrization for log-variance
    避免 hard clamp 导致的梯度消失，允许网络从边界"拉回来"
    
    logv = logv_mid + logv_rad * tanh(raw)
    其中 logv_mid = (logv_min + logv_max) / 2
         logv_rad = (logv_max - logv_min) / 2
    """
    def __init__(self, logv_min: float = -8.0, logv_max: float = 6.0):
        super().__init__()
        self.logv_min = logv_min
        self.logv_max = logv_max
        self.logv_mid = 0.5 * (logv_min + logv_max)
        self.logv_rad = 0.5 * (logv_max - logv_min)
    
    def forward(self, raw_logv):
        """
        raw_logv: (B,T,d_out) 模型输出的原始值（无界）
        返回: (B,T,d_out) 有界的 logv ∈ [logv_min, logv_max]
        """
        return self.logv_mid + self.logv_rad * torch.tanh(raw_logv)
    
    def saturation_ratio(self, raw_logv, threshold=0.98):
        """
        计算接近边界的比例（|tanh(raw)| > threshold）
        threshold=0.98 对应约 tanh^{-1}(0.98) ≈ 2.3
        """
        tanh_val = torch.tanh(raw_logv)
        near_boundary = (tanh_val.abs() > threshold).float()
        return near_boundary.mean().item()

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.left_pad = pad
    def forward(self, x):  # (B,C,T)
        y = super().forward(x)
        if self.left_pad > 0:
            y = y[..., :-self.left_pad]
        return y

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, kernel_size, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(ch, ch, kernel_size, dilation=1),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(ch, ch, 1)
    def forward(self, x):  # (B,C,T)
        return self.proj(x) + self.net(x)

class IMURouteModel(nn.Module):
    """
    (B,T,D_in) -> (B,T,d_out)  逐时间步输出 logσ²
    d_out=1: 各向同性（单通道）
    d_out=3: 各向异性（每轴独立）
    
    variance_param 参数化方式：
    - "direct": 直接输出三轴 logv（旧方法）
    - "sa3": 三轴拉扯参数化（尺度+各向异性）
    """
    def __init__(self, d_in: int, d_model: int = 128, d_out: int = 1, n_tcn: int = 4,
                 kernel_size: int = 3, n_layers_tf: int = 0, n_heads: int = 4,
                 dilations=None, dropout: float = 0.1, 
                 logv_min: float = -8.0, logv_max: float = 6.0, use_bounded: bool = True,
                 variance_param: str = "direct", aniso: dict = None):
        super().__init__()
        self.d_out = d_out
        self.use_bounded = use_bounded
        self.variance_param = variance_param
        self.inp = nn.Linear(d_in, d_model)
        
        # TCN blocks with optional custom dilations
        if dilations is None:
            dilations = [2**i for i in range(n_tcn)]
        blocks = []
        for i in range(n_tcn):
            dil = dilations[i % len(dilations)]
            blocks.append(TCNBlock(d_model, kernel_size=kernel_size, dilation=dil, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        
        # Optional Transformer encoder layers
        if n_layers_tf > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.tf = nn.TransformerEncoder(enc_layer, num_layers=n_layers_tf)
        else:
            self.tf = None
        
        # 方差头：根据 variance_param 选择参数化方式
        if variance_param == "sa3" and d_out == 3:
            # Sa3 三轴拉扯参数化
            from var_heads import Sa3VarHead
            aniso = aniso or {}
            self.var_head = Sa3VarHead(
                in_dim=d_model,
                logv_min=logv_min,
                logv_max=logv_max,
                use_tanh=aniso.get("use_tanh", True),
                kappa_xy=aniso.get("kappa_xy", 2.0),
                kappa_z=aniso.get("kappa_z", 2.0),
            )
            self.head = None  # 不使用旧的 head
            self.bounded = None
        else:
            # 直接输出（旧方法）
            self.head = nn.Linear(d_model, d_out)
            # Bounded parametrization (tanh) to avoid gradient vanishing at boundaries
            if use_bounded:
                self.bounded = BoundedLogVar(logv_min, logv_max)
            else:
                self.bounded = None
            self.var_head = None

    def forward(self, x, return_raw=False, return_sa3=False):  # x: (B,T,D)
        h = self.inp(x)             # (B,T,C)
        h = h.transpose(1, 2)       # (B,C,T)
        h = self.tcn(h)             # (B,C,T)
        h = h.transpose(1, 2)       # (B,T,C)
        if self.tf is not None:
            h = self.tf(h)          # (B,T,C)
        
        # 根据参数化方式输出
        if self.var_head is not None:
            # Sa3 参数化
            logv, s, a = self.var_head(h)  # (B,T,3), (B,T), (B,T,2)
            if return_sa3:
                return {"logv": logv, "s": s, "a": a}
            return logv
        else:
            # 直接输出（旧方法）
            raw_logv = self.head(h)     # (B,T,d_out)
            
            if self.bounded is not None:
                logv = self.bounded(raw_logv)
                if return_raw:
                    return logv, raw_logv
                return logv
            else:
                # 兼容旧的 hard clamp（不推荐）
                return raw_logv
