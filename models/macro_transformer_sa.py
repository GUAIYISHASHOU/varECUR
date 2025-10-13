# -*- coding: utf-8 -*-
# models/macro_transformer_sa.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 可选的 Student-t NLL（ν>2更稳） ----------
class StudentTLoss(nn.Module):
    def __init__(self, nu: float = 3.0, reduction: str = "mean"):
        super().__init__()
        assert nu > 2.0
        self.nu = nn.Parameter(torch.tensor(float(nu)), requires_grad=False)
        self.reduction = reduction

    def forward(self, pred_logvar: torch.Tensor, gt_resid: torch.Tensor):
        # 这里给一个通用版：你若要直接回归logvar做监督，请换成 SmoothL1
        # gt_resid: (..., 2) 真实残差(e_x, e_y)，pred_logvar: (..., 2)
        nu = self.nu
        v = torch.clamp(torch.exp(pred_logvar), 1e-12, 1e12)
        z2 = (gt_resid ** 2) / v
        # log-likelihood（省去常数项）
        nll = 0.5 * (nu + 1.0) * torch.log1p(z2 / nu) + 0.5 * torch.log(v)
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll

# ==================== 新增：统一的损失函数 ====================
class CombinedUncertaintyLoss(nn.Module):
    """
    结合异方差NLL损失和内点概率BCE损失，支持排序一致性损失。
    """
    def __init__(self, nll_weight=1.0, bce_weight=1.0, rank_weight=0.0, 
                 rank_margin=0.0, stage1_mode=False, pos_weight=None):
        super().__init__()
        self.nll_weight = nll_weight
        self.bce_weight = bce_weight
        self.rank_weight = rank_weight  # 排序损失权重
        self.rank_margin = rank_margin  # 排序边界
        # 使用BCEWithLogitsLoss以获得更好的数值稳定性
        # pos_weight: 给正样本(inlier=1)的权重，用于处理类别不平衡
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.stage1_mode = stage1_mode  # 用于两阶段训练

    def _pairwise_rank_loss(self, pred, target, margin=0.0):
        """
        批内两两排序损失（Pairwise Ranking Loss）
        让预测值的相对大小关系与真实值一致
        
        Args:
            pred: (B,) 预测的标量值
            target: (B,) 真实的标量值
            margin: 排序边界（可选）
        Returns:
            标量损失值
        """
        # 构造 GT 排序关系 (i 应该比 j 大/小/相等)
        with torch.no_grad():
            order = target[:, None] - target[None, :]  # (B, B), >0 表示 i 的GT应高于 j
            sign = torch.sign(order)  # +1/-1/0
        
        # 预测值的差异
        diff = pred[:, None] - pred[None, :]  # (B, B)
        
        # Logistic ranking loss: log(1 + exp(-(pred_i - pred_j) * sign))
        # 只对有序对 (sign != 0) 计算损失
        loss = torch.log1p(torch.exp(-(diff - margin * sign) * sign))
        
        # 去掉对角线和无序对
        mask = (sign != 0).float()
        n_pairs = mask.sum() + 1e-6
        
        return (loss * mask).sum() / n_pairs

    def forward(self, pred_logvar, pred_q_logit, gt_y_true, gt_y_inlier):
        # pred_logvar: (B, 2), 模型预测的 [logσx², logσy²]
        # pred_q_logit: (B, 1), 模型预测的内点概率的 logit
        # gt_y_true: (B, 2), 真实标签 [logσx², logσy²]
        # gt_y_inlier: (B, 1), 真实的内点标签 [0.0 或 1.0]

        # --- 第一部分：BCE损失 (用于内点概率q) ---
        loss_bce = self.bce_loss(pred_q_logit, gt_y_inlier)

        if self.stage1_mode:
            # 如果是第一阶段，只返回BCE损失
            return loss_bce

        # --- 第二部分：NLL损失 (用于方差σ) ---
        # 关键：NLL损失只应该在真实的内点上计算！
        inlier_mask = (gt_y_inlier > 0.5).squeeze(-1)
        if not inlier_mask.any():
            # 如果这个batch里一个内点都没有，NLL损失为0
            loss_nll = torch.tensor(0.0, device=pred_logvar.device)
        else:
            # 筛选出内点进行计算
            pred_logvar_inliers = pred_logvar[inlier_mask]
            gt_y_true_inliers = gt_y_true[inlier_mask]
            
            # 使用SmoothL1回归logvar
            loss_nll = F.smooth_l1_loss(pred_logvar_inliers, gt_y_true_inliers)

        # --- 第三部分：排序一致性损失 (可选) ---
        loss_rank = torch.tensor(0.0, device=pred_logvar.device)
        if self.rank_weight > 0:
            # 使用 s = (logσx² + logσy²) / 2 作为整体不确定性度量
            s_pred = pred_logvar.mean(dim=-1)  # (B,)
            s_gt = gt_y_true.mean(dim=-1)      # (B,)
            loss_rank = self._pairwise_rank_loss(s_pred, s_gt, margin=self.rank_margin)

        # --- 合并损失 ---
        total_loss = self.nll_weight * loss_nll + self.bce_weight * loss_bce + self.rank_weight * loss_rank
        return total_loss

# --------- Patch + Geom → Token 的轻量编码器 ----------
class PointEncoder(nn.Module):
    def __init__(self, geom_dim: int, d_model: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1), nn.ReLU(inplace=True),   # 32->16
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True), # 8->4
            nn.AdaptiveAvgPool2d(1)                             # -> (B,128,1,1)
        )
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.ReLU(inplace=True)
        )
        self.proj = nn.Linear(128 + 64, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches: torch.Tensor, geoms: torch.Tensor):
        # patches: (B,K,2,H,W) in [0,1]; geoms: (B,K,G)
        B, K, C, H, W = patches.shape
        f_img = self.cnn(patches.reshape(B * K, C, H, W)).flatten(1)   # (B*K,128)
        f_geo = self.geom_mlp(geoms.reshape(B * K, -1))                # (B*K,64)
        tok = self.proj(torch.cat([f_img, f_geo], dim=1)).reshape(B, K, -1)
        return self.norm(tok)                                          # (B,K,d)

# --------- 位置编码（可选） ----------
class SinPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1,max_len,d)

    def forward(self, x: torch.Tensor):
        # x: (B,K,d)
        return x + self.pe[:, :x.size(1), :]

# --------- 宏观模型主体（SA 参数化） ----------
class MacroTransformerSA(nn.Module):
    def __init__(self,
                 geom_dim: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 1,
                 a_max: float = 3.0,
                 drop_token_p: float = 0.0,
                 use_posenc: bool = True,
                 logv_min: float = -10.0,
                 logv_max: float = 6.0):
        super().__init__()
        self.point_encoder = PointEncoder(geom_dim=geom_dim, d_model=d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation="relu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        
        # === 修改：分离两个输出头以便独立训练 ===
        # 共享特征提取
        self.head_shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True)
        )
        # SA参数化头（用于logvar回归）
        self.sa_head = nn.Linear(128, 2)  # 输出 (s, a)
        # 内点概率头（用于分类）
        self.q_head = nn.Linear(128, 1)   # 输出 q_logit
        
        self.a_max = float(a_max)
        self.drop_token_p = float(drop_token_p)
        self.use_posenc = use_posenc
        self.posenc = SinPosEncoding(d_model) if use_posenc else nn.Identity()
        self.logv_min = float(logv_min)
        self.logv_max = float(logv_max)

    @torch.no_grad()
    def _make_mask(self, B: int, K: int, num_tok: torch.Tensor):
        # True=ignore；CLS 永不mask
        base = torch.arange(K, device=num_tok.device)[None, :] >= num_tok[:, None]
        return torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=base.device), base], dim=1)

    def forward(self, patches: torch.Tensor, geoms: torch.Tensor, num_tok: torch.Tensor):
        # patches:(B,K,2,H,W) geoms:(B,K,G) num_tok:(B,)
        B, K = patches.shape[:2]
        tok = self.point_encoder(patches, geoms)           # (B,K,d)
        if self.training and self.drop_token_p > 0:
            drop = (torch.rand(B, K, 1, device=tok.device) < self.drop_token_p)
            tok = tok.masked_fill(drop, 0.0)
        tok = self.posenc(tok)
        cls = self.cls.expand(B, -1, -1)                   # (B,1,d)
        x = torch.cat([cls, tok], dim=1)                   # (B,K+1,d)
        mask = self._make_mask(B, K, num_tok)              # (B,K+1)
        out = self.enc(x, src_key_padding_mask=mask)       # (B,K+1,d)
        h = out[:, 0]                                      # (B,d) 取CLS
        
        # === 共享特征 ===
        feat = self.head_shared(h)                         # (B, 128)
        
        # === SA参数化头 ===
        sa = self.sa_head(feat)                            # (B, 2)
        s, a_raw = sa[:, 0:1], sa[:, 1:2]                  # (B, 1), (B, 1)
        a = torch.tanh(a_raw) * self.a_max
        lvx = torch.clamp(s + a, self.logv_min, self.logv_max)
        lvy = torch.clamp(s - a, self.logv_min, self.logv_max)
        pred_logvar = torch.cat([lvx, lvy], dim=-1)        # (B, 2)
        
        # === 内点概率头 ===
        q_logit = self.q_head(feat)                        # (B, 1)
        
        return pred_logvar, q_logit                        # (B,2), (B,1)

