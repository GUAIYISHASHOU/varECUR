from __future__ import annotations
import torch
import torch.nn as nn

# ----- Causal TCN block -----
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.left_pad = padding
    def forward(self, x):
        y = super().forward(x)
        if self.left_pad > 0:
            y = y[..., :-self.left_pad]
        return y

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(ch, ch, kernel_size, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(ch, ch, 1)
    def forward(self, x):  # (B,C,T)
        return self.proj(x) + self.net(x)

# ----- Model: (B,T,D_in) -> (B,T,D_out logvar) -----
class IMURouteModel(nn.Module):
    def __init__(self, d_in: int, d_model: int=128, d_out: int=1, n_tcn: int=4, kernel_size:int=3,
                 dilations=(1,2,4,8), n_layers_tf: int=2, n_heads:int=4, dropout: float=0.1):
        super().__init__()
        self.d_out = d_out
        self.inp = nn.Linear(d_in, d_model)
        self.tcn = nn.Sequential(*[TCNBlock(d_model, kernel_size=kernel_size, dilation=dilations[i%len(dilations)], dropout=dropout) for i in range(n_tcn)])
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=n_layers_tf)
        self.head = nn.Linear(d_model, d_out)

    def forward(self, x):  # x: (B,T,D_in)
        h = self.inp(x)           # (B,T,C)
        h = h.transpose(1,2)      # (B,C,T) for TCN
        h = self.tcn(h)           # (B,C,T)
        h = h.transpose(1,2)      # (B,T,C)
        h = self.tf(h)            # (B,T,C)
        logv = self.head(h)       # (B,T,D_out)
        return logv
