#!/usr/bin/env python3
"""
OOF (Out-of-Fold) 训练脚本
按序列分组的K折交叉验证，避免数据泄漏
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import json
import random

from utils import set_seed, load_config_file, ensure_dir, to_device
from dataset import build_dataset_from_seqs, IMUDataset
from models import IMURouteModel
from losses import loss_total, nll_gauss_iso, nll_studentt_diag_axes, nll_studentt_iso, aux_cov_z2_penalties, aux_scale_on_s
from metrics import route_metrics_imu, z2_cov_studentt_diag_axes, spearmanr_np, C68_GAUSS, C95_GAUSS, studentt_z2_threshold


def set_seeds(seed=3407):
    """设置所有随机种子以保证可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_fold(cfg, route: str, fold_id: int, train_seqs: list, val_seqs: list, 
                   save_dir: Path, seq_npz_dir: str):
    """
    训练单个fold
    
    Args:
        cfg: 配置字典
        route: "acc" | "gyr"
        fold_id: 折编号
        train_seqs: 训练序列列表
        val_seqs: 验证序列列表
        save_dir: 保存目录
        seq_npz_dir: 序列NPZ文件目录
    
    Returns:
        best_path: 最佳模型路径
        best_val: 最佳验证目标（val_obj）
        best_state: 附加最佳指标（epoch/val_obj/val_nll/cov68/z2_mean）
    """
    # 设置种子（每折不同但可复现）
    seed_base = cfg.get("common", {}).get("seed", 2024)
    set_seeds(seed_base + fold_id)
    
    device = cfg.get("common", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg.get("train", {})
    # 统一分布口径：'gauss' | 'studentt'（兼容旧 use_studentt 布尔）
    dist = tr.get("dist", "studentt" if tr.get("use_studentt", False) else "gauss")
    
    print(f"\n{'='*80}")
    print(f"Fold {fold_id}: Training on {len(train_seqs)} seqs, validating on {len(val_seqs)} seqs")
    print(f"{'='*80}")
    
    # 构建数据集（序列级，不使用scaler）
    print("Loading training data...")
    ds_train = build_dataset_from_seqs(seq_npz_dir, train_seqs, route, scaler=None)
    
    # 拟合scaler（只在训练集上）
    print("Fitting scaler on training data...")
    scaler = ds_train.fit_scaler()
    
    # 应用scaler到训练集
    ds_train.apply_scaler(scaler)
    
    # 保存scaler
    scaler_path = save_dir / f"scaler_fold{fold_id}.json"
    ensure_dir(scaler_path.parent)
    with open(scaler_path, 'w') as f:
        json.dump({
            "mean": scaler["mean"].tolist(),
            "std": scaler["std"].tolist()
        }, f, indent=2)
    print(f"Saved scaler to {scaler_path}")
    
    # 构建验证集（使用训练集的scaler）
    print("Loading validation data...")
    ds_val = build_dataset_from_seqs(seq_npz_dir, val_seqs, route, scaler=scaler)
    
    # 安全检查
    assert len(ds_train) > 0 and len(ds_val) > 0, f"Empty fold! train={len(ds_train)}, val={len(ds_val)}"
    
    bs = tr.get("batch_size", 16)
    assert len(ds_train) >= bs, f"Train samples too few ({len(ds_train)}), reduce batch_size"
    
    # DataLoader
    num_workers = cfg.get("common", {}).get("num_workers", 0)
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, 
                             drop_last=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False, 
                           drop_last=False, num_workers=num_workers, pin_memory=True)
    
    # 模型参数
    d_in = ds_train.D
    use_studentt = tr.get("use_studentt", False)
    # 与普通训练保持一致：Student-t 且步级标签时输出3轴
    d_out = 3 if (use_studentt and getattr(ds_train, "use_step", True)) else 1
    
    logv_min = tr.get("logv_min", -8.0)
    logv_max = tr.get("logv_max", 6.0)
    use_bounded = tr.get("use_bounded", True)
    variance_param = tr.get("variance_param", "direct")
    aniso = tr.get("aniso", None)
    
    # 创建模型
    model = IMURouteModel(
        d_in=d_in, d_out=d_out, 
        d_model=cfg.get("model", {}).get("d_model", 128),
        n_tcn=cfg.get("model", {}).get("n_tcn", 4), 
        kernel_size=cfg.get("model", {}).get("kernel_size", 3),
        n_layers_tf=cfg.get("model", {}).get("n_layers_tf", 0), 
        n_heads=cfg.get("model", {}).get("n_heads", 4),
        dropout=cfg.get("model", {}).get("dropout", 0.1),
        logv_min=logv_min, logv_max=logv_max, 
        use_bounded=use_bounded,
        variance_param=variance_param, 
        aniso=aniso
    ).to(device)
    
    # 模型初始化（支持 sa3 与 direct）
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_batch = to_device(sample_batch, device)
        e2 = sample_batch["e2"]
        m = sample_batch["mask"]
        m_bool = m > 0.5
        e2_clean = torch.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0)
        
        if variance_param == "sa3" and d_out == 3 and (model.var_head is not None):
            # 估算每轴初始方差 (sigma^2)
            e2_masked = torch.where(m_bool.unsqueeze(-1), e2_clean, torch.zeros_like(e2_clean))
            num = e2_masked.sum(dim=(0, 1))
            den = m_bool.sum().clamp_min(1.0)
            var_axes = (num / den).clamp_min(1e-12)
            nu_eff = max(float(tr.get("nu", 5.0)), 2.1)
            sigma2_init_axes = (var_axes * ((nu_eff - 2.0) / nu_eff)).clamp_min(1e-12)

            # 目标 s（有界空间中的尺度，等于 log(sigma^2) 的均值），映射回 raw 空间写入 bias
            s_target = float(torch.log(sigma2_init_axes).mean())
            denom = max(float(model.var_head.s_rad), 1e-6)
            x = (s_target - float(model.var_head.s_mid)) / denom
            x = max(min(x, 0.999), -0.999)
            raw_s_bias = 0.5 * math.log((1.0 + x) / (1.0 - x))
            model.var_head.head_s.bias.data.fill_(raw_s_bias)
            model.var_head.head_axi.bias.data.zero_()
            print(f"[init] Sa3 head initialized: s_target={s_target:.3f}, raw_s_bias={raw_s_bias:.3f}, a_bias=[0., 0.]")
        else:
            if d_out == 3 and (model.head is not None):
                e2_masked = torch.where(m_bool.unsqueeze(-1), e2_clean, torch.zeros_like(e2_clean))
                num = e2_masked.sum(dim=(0, 1))
                den = m_bool.sum().clamp_min(1.0)
                var_axes = (num / den).clamp_min(1e-12)
                nu_eff = max(float(tr.get("nu", 5.0)), 2.1)
                sigma2_init_axes = (var_axes * ((nu_eff - 2.0) / nu_eff)).clamp_min(1e-12)
                model.head.bias.data.copy_(sigma2_init_axes.log().to(model.head.bias))
                print(f"[init] head bias (3-axis): {sigma2_init_axes.cpu().numpy()}")
            elif model.head is not None:
                e2_sum = e2_clean.sum(dim=-1) / 3.0
                e2_masked = torch.where(m_bool, e2_sum, torch.zeros_like(e2_sum))
                var0 = (e2_masked.sum() / m_bool.sum()).clamp_min(1e-12)
                model.head.bias.data.fill_(float(var0.log()))
                print(f"[init] head bias (iso): {var0:.3e}")
    
    # 优化器和调度器
    base_lr = tr.get("lr", 1e-4)
    var_head_lr_mult = float(tr.get("var_head_lr_mult", 1.0))
    # 细分方差头LR：尺度(s) vs 各向异性(a)
    var_head_lr_mult_s = float(tr.get("var_head_lr_mult_s", var_head_lr_mult))
    var_head_lr_mult_a = float(tr.get("var_head_lr_mult_a", var_head_lr_mult))
    if getattr(model, "var_head", None) is not None:
        # 拆分参数组
        hs_params = list(model.var_head.head_s.parameters())
        ha_params = list(model.var_head.head_axi.parameters())
        var_params = hs_params + ha_params
        var_id = {id(p) for p in var_params}
        base_params = [p for p in model.parameters() if id(p) not in var_id]
        opt = optim.AdamW([
            {"params": base_params, "lr": base_lr},
            {"params": hs_params,  "lr": base_lr * var_head_lr_mult_s},
            {"params": ha_params,  "lr": base_lr * var_head_lr_mult_a},
        ], weight_decay=tr.get("weight_decay", 1.0e-4))
    else:
        opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=tr.get("weight_decay", 1.0e-4))
    
    epochs = tr.get("epochs", 30)
    warmup_epochs = tr.get("warmup_epochs", 3)
    scheduler_type = tr.get("scheduler", "warmup_cosine")
    
    if scheduler_type == "warmup_cosine":
        warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_epochs))
        cosine = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs), eta_min=base_lr * 0.05)
        sch = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        warnings.filterwarnings('ignore', message='.*scheduler.step.*', category=UserWarning)
    else:
        sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    
    # 额外正则与调度参数
    best_val = 1e9
    best_path = save_dir / f"best_fold{fold_id}.pt"
    best_state = None
    patience = tr.get("early_stop_patience", 5)
    bad = 0
    nu = tr.get("nu", 5.0)
    lambda_center = tr.get("lambda_center", 0.0)
    # Student-t 三轴训练期约束系数
    lambda_center_axes = tr.get("lambda_center_axes", 0.0)
    lambda_aniso_l2 = tr.get("lambda_aniso_l2", 0.0)
    lambda_tv = tr.get("lambda_tv", 0.0)
    lambda_cov68 = tr.get("lambda_cov68", 0.0)
    cov68_target = tr.get("cov68_target", 0.68)
    cov68_tau = tr.get("cov68_tau", 0.1)
    lambda_z2_low = tr.get("lambda_z2_low", 0.0)
    z2_low_margin = tr.get("z2_low_margin", 0.0)
    lambda_z2_var = tr.get("lambda_z2_var", 0.0)
    lambda_s_scale = tr.get("lambda_s_scale", 0.0)
    use_ema_z2_stats = tr.get("use_ema_z2_stats", False)
    ema_z2_decay = float(tr.get("ema_z2_decay", 0.98))
    var_warmup_epochs = int(tr.get("variance_warmup_epochs", 0))
    # 各向异性专属参数
    var_aniso_warmup_epochs = int(tr.get("variance_aniso_warmup_epochs", 0))
    lambda_cov68_axes = float(tr.get("lambda_cov68_axes", 0.0))
    cov68_tau_axes = float(tr.get("cov68_tau_axes", tr.get("cov68_tau", 0.1)))
    # 动态 logv_max 护栏
    logv_max_warmup = tr.get("logv_max_warmup", None)
    logv_max_warmup_frac = float(tr.get("logv_max_warmup_frac", 0.2))

    # 覆盖率阈值（Student-t 使用 F 分布分位转换）
    thr68_overall = C68_GAUSS
    thr95_overall = C95_GAUSS
    thr68_axis = C68_GAUSS
    thr95_axis = C95_GAUSS
    if dist == "studentt":
        thr68_overall = float(studentt_z2_threshold(0.68, float(nu), 3))
        thr95_overall = float(studentt_z2_threshold(0.95, float(nu), 3))
        if d_out == 3:
            thr68_axis = float(studentt_z2_threshold(0.68, float(nu), 1))
            thr95_axis = float(studentt_z2_threshold(0.95, float(nu), 1))

    reg_warmup_frac = float(tr.get("lambda_center_axes_warmup_frac", 0.2))
    reg_warmup_epochs = int(max(0, round(reg_warmup_frac * tr.get("epochs", 30))))
    use_ema = bool(use_ema_z2_stats)
    ema_state = {"z2_mean": None, "z2_var": None}

    def ema_update(z2_values: torch.Tensor):
        if z2_values.numel() == 0:
            return
        m_ = z2_values.mean().detach()
        v_ = z2_values.var(unbiased=False).detach()
        if ema_state["z2_mean"] is None:
            ema_state["z2_mean"] = m_
            ema_state["z2_var"] = v_
        else:
            d = float(ema_z2_decay)
            ema_state["z2_mean"] = d * ema_state["z2_mean"] + (1 - d) * m_
            ema_state["z2_var"] = d * ema_state["z2_var"] + (1 - d) * v_

    def set_var_head_requires_grad(flag: bool):
        if getattr(model, "var_head", None) is not None:
            for p in model.var_head.parameters():
                p.requires_grad = flag
    def set_aniso_requires_grad(flag: bool):
        if getattr(model, "var_head", None) is not None:
            for p in model.var_head.head_axi.parameters():
                p.requires_grad = flag
    def set_scale_requires_grad(flag: bool):
        if getattr(model, "var_head", None) is not None:
            for p in model.var_head.head_s.parameters():
                p.requires_grad = flag
    
    # 记录训练曲线
    train_losses = []
    val_losses = []
    
    for ep in range(1, epochs + 1):
        model.train()
        if var_warmup_epochs > 0:
            if ep <= var_warmup_epochs:
                set_var_head_requires_grad(False)
                if ep == 1:
                    print(f"[warmup] freeze var-head for first {var_warmup_epochs} epochs")
            elif ep == var_warmup_epochs + 1:
                set_var_head_requires_grad(True)
                print("[warmup] unfreeze var-head")
        # 仅冻结各向异性更久，允许 s 学习
        if var_aniso_warmup_epochs > 0:
            if ep <= var_aniso_warmup_epochs:
                set_aniso_requires_grad(False)
                if ep == 1:
                    print(f"[warmup] freeze anisotropy head for first {var_aniso_warmup_epochs} epochs")
            elif ep == var_aniso_warmup_epochs + 1:
                set_aniso_requires_grad(True)
                print("[warmup] unfreeze anisotropy head")
        tr_losses = []
        pbar = tqdm(train_loader, desc=f"Fold {fold_id} Epoch {ep}/{epochs} [Train]", ncols=100, leave=False)
        for batch in pbar:
            batch = to_device(batch, device)
            x = batch["x"]
            e2 = batch["e2"]
            m = batch["mask"]
            
            logv = model(x)
            # 训练期动态上限护栏：前 warmup_frac 的 epoch 临时收紧上限
            logv_max_eff = logv_max
            if logv_max_warmup is not None:
                boundary_epochs = int(max(1, round(tr.get("epochs", 30) * logv_max_warmup_frac)))
                if ep <= boundary_epochs:
                    logv_max_eff = float(min(logv_max, float(logv_max_warmup)))
            logv_eff = torch.clamp(logv, min=logv_min, max=logv_max_eff)
            
            if dist == "studentt" and d_out == 3:
                e2_clean = torch.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0)
                m_axes = ((m.unsqueeze(-1) > 0.5) & torch.isfinite(e2_clean)).float()
                e_axes = torch.sqrt(torch.clamp(e2_clean, min=0.0))
                if reg_warmup_epochs > 0:
                    w = min(1.0, ep / float(reg_warmup_epochs))
                else:
                    w = 1.0
                lam_center_eff = float(lambda_center_axes) * w
                lam_cov68_eff = float(lambda_cov68) * w
                lam_z2_low_eff = float(lambda_z2_low) * w
                loss = nll_studentt_diag_axes(
                    logv_eff, e_axes, m_axes, nu, logv_min, logv_max_eff,
                    lam_center_eff, lambda_aniso_l2, lambda_tv,
                    0.0, cov68_target, cov68_tau,
                    0.0, z2_low_margin
                )
                with torch.no_grad():
                    s2_axes = torch.exp(logv_eff).clamp_min(1e-12)
                    if float(nu) > 2.0:
                        v_eff_axes = s2_axes * (float(nu) / (float(nu) - 2.0))
                    else:
                        v_eff_axes = s2_axes
                    z2_full = (e_axes.pow(2) / (v_eff_axes + 1e-12)).sum(dim=-1) / 3.0
                    valid_bt = (m_axes > 0.5).all(dim=-1)
                    z2_bt = z2_full[valid_bt]
                cov_loss, _ = aux_cov_z2_penalties(
                    z2_bt,
                    lambda_cov68=lam_cov68_eff,
                    cov68_target=cov68_target,
                    cov68_tau=cov68_tau,
                    lambda_z2_low=lam_z2_low_eff,
                    z2_low_margin=z2_low_margin,
                    use_ema_z2_stats=use_ema,
                    ema_state=ema_state,
                    lambda_z2_var=float(lambda_z2_var) * w,
                    c_thresh=float(thr68_overall),
                )
                loss = loss + cov_loss
                # 轴级覆盖率约束（df=1 → 阈值≈1.0）
                if lambda_cov68_axes > 0.0:
                    z2_axes = (e_axes.pow(2) / (v_eff_axes + 1e-12))  # (B,T,3)
                    axis_loss = 0.0
                    for j in range(3):
                        vb = (m_axes[..., j] > 0.5)
                        zj = z2_axes[..., j][vb]
                        if zj.numel() > 0:
                            al, _ = aux_cov_z2_penalties(
                                zj,
                                lambda_cov68=lam_cov68_eff * float(lambda_cov68_axes / max(lambda_cov68, 1e-8)) if lambda_cov68 > 0 else lambda_cov68_axes * w,
                                cov68_target=cov68_target,
                                cov68_tau=cov68_tau_axes,
                                lambda_z2_low=0.0,
                                z2_low_margin=0.0,
                                use_ema_z2_stats=False,
                                ema_state=None,
                                lambda_z2_var=0.0,
                                c_thresh=float(thr68_axis),
                            )
                            axis_loss = axis_loss + al
                    loss = loss + axis_loss
                s_loss, _ = aux_scale_on_s(
                    logv, e_axes, float(nu), float(lambda_s_scale) * w, m_axes
                )
                loss = loss + s_loss
            elif dist == "studentt":
                e2sum = e2.sum(dim=-1)
                loss = nll_studentt_iso(logv, e2sum, m, nu, logv_min, logv_max)
                lv_use = logv.squeeze(-1) if (logv.dim() == 3 and logv.size(-1) == 1) else logv
                s2_b = torch.exp(lv_use).clamp_min(1e-12)
                v_eff_b = s2_b * (nu / (nu - 2.0))
                z2_b = e2sum / (3.0 * v_eff_b)
                mb = m > 0.5
                z2_mean_b = z2_b.masked_select(mb).mean() if mb.any() else z2_b.mean()
                loss = loss + lambda_center * (z2_mean_b - 1.0) ** 2
            else:
                loss, _ = loss_total(e2, logv, m, logv_min, logv_max, lambda_center)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if dist == "studentt" and d_out == 3 and use_ema:
                ema_update(z2_bt.detach())
            tr_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(tr_losses):.4f}'})
        
        # 验证
        model.eval()
        with torch.no_grad():
            vals = []
            all_e2, all_lv, all_m, all_raw = [], [], [], []
            pbar_val = tqdm(val_loader, desc=f"Fold {fold_id} Epoch {ep}/{epochs} [Val]  ", ncols=100, leave=False)
            for batch in pbar_val:
                batch = to_device(batch, device)
                x = batch["x"]
                e2 = batch["e2"]
                m = batch["mask"]
                # 获取 logv 和 raw_logv（用于饱和度统计）
                if use_bounded and getattr(model, "bounded", None) is not None:
                    logv, raw_logv = model(x, return_raw=True)
                    all_raw.append(raw_logv.cpu())
                else:
                    logv = model(x)
                
                if dist == "studentt" and d_out == 3:
                    e2_clean = torch.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0)
                    m_axes = ((m.unsqueeze(-1) > 0.5) & torch.isfinite(e2_clean)).float()
                    e_axes = torch.sqrt(torch.clamp(e2_clean, min=0.0))
                    l = nll_studentt_diag_axes(logv, e_axes, m_axes, nu, logv_min, logv_max, 0.0, 0.0, 0.0)
                elif dist == "studentt":
                    e2sum = e2.sum(dim=-1)
                    l = nll_studentt_iso(logv, e2sum, m, nu, logv_min, logv_max)
                else:
                    l, _ = loss_total(e2, logv, m, logv_min, logv_max, lambda_center)
                
                vals.append(l.item())
                all_e2.append(e2.cpu())
                all_lv.append(logv.cpu())
                all_m.append(m.cpu())
                pbar_val.set_postfix({'loss': f'{np.mean(vals):.4f}'})
            
            val_loss = float(np.mean(vals)) if vals else 1e9
        
        # Step scheduler
        if scheduler_type == "warmup_cosine":
            sch.step()
        else:
            sch.step(val_loss)
        
        # 记录损失
        train_losses.append(float(np.mean(tr_losses)))
        val_losses.append(val_loss)
        val_obj = val_loss
        
        # 计算指标
        if len(all_e2) > 0:
            e2_cat = torch.cat(all_e2, dim=0)
            lv_cat = torch.cat(all_lv, dim=0)
            m_cat = torch.cat(all_m, dim=0)
            
            if dist == "studentt" and d_out == 3:
                e2_clean_cat = torch.nan_to_num(e2_cat, nan=0.0, posinf=0.0, neginf=0.0)
                e_axes_cat = torch.sqrt(torch.clamp(e2_clean_cat, min=0.0))
                m_axes_cat = ((m_cat.unsqueeze(-1) > 0.5) & torch.isfinite(e2_clean_cat)).float()
                z2_mean, cov68, cov95 = z2_cov_studentt_diag_axes(lv_cat, e_axes_cat, m_axes_cat, nu, logv_min, logv_max)
                e2_mean = e2_clean_cat.mean(dim=-1)
                s2 = torch.exp(lv_cat).clamp_min(1e-12)
                v_eff = s2 * (nu / (nu - 2.0))
                var_mean = v_eff.mean(dim=-1)
                mask_bool = m_cat > 0.5
                sp = spearmanr_np(e2_mean[mask_bool].detach().cpu().numpy().ravel(),
                                  var_mean[mask_bool].detach().cpu().numpy().ravel()) if mask_bool.sum() > 0 else 0.0
                sat_str = ""
                if use_bounded and len(all_raw) > 0 and getattr(model, "bounded", None) is not None:
                    raw_cat = torch.cat(all_raw, dim=0)
                    sat_ratio = model.bounded.saturation_ratio(raw_cat, threshold=0.98)
                    sat_str = f" sat={sat_ratio:.2%}"
                aniso_str = ""
                if variance_param == "sa3" and d_out == 3:
                    try:
                        from utils_sa3 import compute_aniso_stats
                        aniso_stats = compute_aniso_stats(lv_cat.numpy())
                        aniso_str = f" | s={aniso_stats['s_mean']:.3f} a_xy={aniso_stats['a_xy_std']:.3f} a_z={aniso_stats['a_z_std']:.3f}"
                    except Exception:
                        pass
                # 轴向覆盖（df=1）的额外偏差项（防止各向异性兜覆盖率）
                # 计算每轴 z2 并统计覆盖
                cov_x = cov_y = cov_z = 0.0
                try:
                    z2_axes = (e_axes_cat.pow(2) / v_eff).clamp_min(0.0)  # (B,T,3)
                    for j, name in enumerate(["x", "y", "z"]):
                        vb = (m_axes_cat[..., j] > 0.5)
                        zj = z2_axes[..., j][vb]
                        if zj.numel() > 0:
                            cov_val = float((zj <= float(thr68_axis)).float().mean().item())
                        else:
                            cov_val = 0.0
                        if j == 0:
                            cov_x = cov_val
                        elif j == 1:
                            cov_y = cov_val
                        else:
                            cov_z = cov_val
                except Exception:
                    pass
                beta = 8.0
                beta_ax = 0.3 * beta
                cov_dev = (cov68 - float(cov68_target)) ** 2
                cov_dev_axes = (cov_x - float(cov68_target)) ** 2 + (cov_y - float(cov68_target)) ** 2 + (cov_z - float(cov68_target)) ** 2
                val_obj = float(val_loss + beta * cov_dev + beta_ax * cov_dev_axes)
                print(f"[Fold {fold_id} ep {ep:02d}] train={np.mean(tr_losses):.4f} val={val_loss:.4f} | z2={z2_mean:.3f} cov68={cov68:.2%} cov95={cov95:.2%} sp={sp:.3f}{sat_str}{aniso_str} | thr68={thr68_overall:.4f} thr95={thr95_overall:.4f} | val_obj={val_obj:.6f}")
            else:
                lv_use = lv_cat.squeeze(-1) if (lv_cat.dim() == 3 and lv_cat.size(-1) == 1) else lv_cat
                e2sum_cat = e2_cat.sum(dim=-1)
                s2 = torch.exp(lv_use).clamp_min(1e-12)
                v_eff = s2 * (nu / (nu - 2.0)) if dist == "studentt" else s2
                mask_bool = m_cat > 0.5
                z2_vals = (e2sum_cat / (3.0 * v_eff)).masked_select(mask_bool)
                z2_mean = float(z2_vals.mean().item()) if z2_vals.numel() > 0 else 0.0
                cov68 = float((z2_vals <= thr68_overall).float().mean().item()) if z2_vals.numel() > 0 else 0.0
                cov95 = float((z2_vals <= thr95_overall).float().mean().item()) if z2_vals.numel() > 0 else 0.0
                sp = spearmanr_np(
                    e2sum_cat[mask_bool].detach().cpu().numpy().reshape(-1),
                    v_eff[mask_bool].detach().cpu().numpy().reshape(-1)
                ) if mask_bool.sum() > 0 else 0.0
                met = route_metrics_imu(e2_cat, lv_cat, m_cat, logv_min=logv_min, logv_max=logv_max)
                clamp_hit = met.get("saturation", 0.0)
                z2_np = z2_vals.detach().cpu().numpy().reshape(-1) if z2_vals.numel() > 0 else np.array([])
                if z2_np.size > 0:
                    p90, p95, p99 = np.quantile(z2_np, [0.90, 0.95, 0.99])
                else:
                    p90 = p95 = p99 = 0.0
                hint = ""
                if (z2_mean <= 1.1) and (cov68 >= 0.80):
                    hint = " | hint=var_high"
                beta = 8.0
                cov_dev = (cov68 - float(cov68_target)) ** 2
                val_obj = float(val_loss + beta * cov_dev)
                print(f"[Fold {fold_id} ep {ep:02d}] train={np.mean(tr_losses):.4f} val={val_loss:.4f} | " +
                      f"z2={z2_mean:.3f} cov68={cov68:.2%} cov95={cov95:.2%} sp={sp:.3f} | " +
                      f"p90={p90:.2f} p95={p95:.2f} p99={p99:.2f} | clamp_hit={clamp_hit:.2%}{hint} | " +
                      f"thr68={thr68_overall:.4f} thr95={thr95_overall:.4f} | val_obj={val_obj:.6f}")
        else:
            print(f"[Fold {fold_id} ep {ep:02d}] train={np.mean(tr_losses):.4f} val={val_loss:.4f}")
        
        # 保存最佳模型
        if val_obj + 1e-6 < best_val:
            best_val = val_obj
            bad = 0
            ensure_dir(best_path.parent)
            torch.save({
                "state_dict": model.state_dict(),
                "model": model.state_dict(),  # 兼容旧加载代码
                "model_name": "IMURouteModel",
                "cfg": cfg,
                "route": route,
                "fold_id": fold_id,
                "d_in": d_in,
                "d_out": d_out,
                "dist": dist,
                "nu": (float(nu) if dist == "studentt" else None),
                "logv_min": logv_min,
                "logv_max": logv_max,
                "use_bounded": use_bounded,
                "variance_param": variance_param,
                "aniso": aniso,
                "label_version": cfg.get("io", {}).get("label_version", None),
                "scaler": scaler
            }, best_path)
            # 记录最佳指标
            try:
                best_state = {
                    "epoch": ep,
                    "val_obj": float(val_obj),
                    "val_nll": float(val_loss),
                    "cov68": float(cov68),
                    "z2_mean": float(z2_mean),
                }
            except Exception:
                best_state = {
                    "epoch": ep,
                    "val_obj": float(val_obj),
                    "val_nll": float(val_loss),
                }
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep}")
                break
    
    # 保存训练曲线
    curve_path = save_dir / f"curve_fold{fold_id}.json"
    with open(curve_path, 'w') as f:
        json.dump({
            "train": train_losses,
            "val": val_losses
        }, f, indent=2)
    
    return str(best_path), best_val, best_state


def main():
    ap = argparse.ArgumentParser(description="OOF训练脚本")
    ap.add_argument("--route", choices=["acc", "gyr"], required=True, help="路由类型")
    ap.add_argument("--config", type=str, required=True, help="配置文件路径")
    ap.add_argument("--splits", type=str, required=True, help="Splits JSON文件路径")
    ap.add_argument("--seq_dir", type=str, required=True, help="序列NPZ文件目录")
    ap.add_argument("--folds", type=str, default="all", help="要训练的折，逗号分隔（如'0,1,2'）或'all'")
    args = ap.parse_args()
    
    cfg = load_config_file(args.config)
    save_dir = Path(cfg.get("common", {}).get("log_dir", "runs")) / f"oof_{args.route}"
    ensure_dir(save_dir)
    
    # 加载splits（支持两种格式：{"0":{...}} 或 {"folds":[...], "test":[...]})
    with open(args.splits, 'r', encoding='utf-8') as f:
        raw_splits = json.load(f)

    if isinstance(raw_splits, dict) and "folds" in raw_splits:
        folds_list = raw_splits["folds"]
        splits = {str(i): {"train": folds_list[i]["train"], "val": folds_list[i]["val"]}
                  for i in range(len(folds_list))}
        if "test" in raw_splits:
            print(f"[info] Found held-out test sequences: {raw_splits['test']} (not used during OOF training)")
    else:
        splits = raw_splits

    # 确定要训练的折
    if args.folds == "all":
        fold_ids = sorted(map(int, splits.keys()))
    else:
        fold_ids = [int(x.strip()) for x in args.folds.split(',')]
    
    print(f"\n{'='*80}")
    print(f"OOF Training: route={args.route}, folds={fold_ids}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    # 训练每个fold
    results = {}
    for fold_id in fold_ids:
        if str(fold_id) not in splits:
            print(f"Warning: Fold {fold_id} not found in splits, skipping")
            continue
        
        train_seqs = splits[str(fold_id)]["train"]
        val_seqs = splits[str(fold_id)]["val"]
        
        best_path, best_val, best_state = train_one_fold(
            cfg, args.route, fold_id, train_seqs, val_seqs, save_dir, args.seq_dir
        )
        
        res = {
            "best_path": best_path,
            "best_val_obj": best_val,
        }
        if best_state is not None:
            res.update(best_state)
        results[fold_id] = res
        
        print(f"\nFold {fold_id} completed: best_val_obj={best_val:.4f}, saved to {best_path}\n")
    
    # 保存汇总结果
    summary_path = save_dir / "oof_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总
    print(f"\n{'='*80}")
    print("OOF Training Summary:")
    print(f"{'='*80}")
    avg_val = np.mean([r["best_val_obj"] for r in results.values()])
    std_val = np.std([r["best_val_obj"] for r in results.values()])
    print(f"Average best val obj: {avg_val:.4f} ± {std_val:.4f}")
    for fold_id, res in sorted(results.items()):
        msg = f"  Fold {fold_id}: val_obj={res['best_val_obj']:.4f}"
        if "epoch" in res:
            msg += f", epoch={res['epoch']}"
        if "val_nll" in res:
            msg += f", val_nll={res['val_nll']:.4f}"
        if "cov68" in res:
            msg += f", cov68={res['cov68']:.2%}"
        if "z2_mean" in res:
            msg += f", z2={res['z2_mean']:.3f}"
        print(msg)
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
