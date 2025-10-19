from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import warnings

from utils import set_seed, load_config_file, ensure_dir, to_device
from dataset import build_loader, IMUDataset
from models import IMURouteModel
from losses import loss_total, nll_gauss_iso, nll_studentt_diag_axes
from metrics import route_metrics_imu, z2_cov_studentt_diag_axes, spearmanr_np


def train_one_run(cfg, route: str, save_dir: Path):
    device = cfg.get("common", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.get("common", {}).get("seed", 42)
    set_seed(seed)

    tr = cfg.get("train", {})
    bs = tr.get("batch_size", 16)
    train_loader = build_loader(tr["train_npz"], route, bs, True, cfg.get("common", {}).get("num_workers", 4))
    val_loader   = build_loader(tr["val_npz"],   route, bs, False, cfg.get("common", {}).get("num_workers", 4))

    # 维度探测
    probe = IMUDataset(tr["train_npz"], route)
    d_in = probe.D
    
    # 确定输出维度：如果使用Student-t对角模式且有步级标签，输出3维
    use_studentt = tr.get("use_studentt", False)
    d_out = 3 if (use_studentt and probe.use_step) else 1

    # 损失参数（提前获取用于模型创建）
    logv_min = tr.get("logv_min", -8.0)
    logv_max = tr.get("logv_max", 6.0)
    use_bounded = tr.get("use_bounded", True)  # 默认使用 tanh 有界参数化
    variance_param = tr.get("variance_param", "direct")  # 方差参数化方式
    aniso = tr.get("aniso", None)  # 各向异性参数
    
    model = IMURouteModel(d_in=d_in, d_out=d_out, d_model=cfg.get("model",{}).get("d_model",128),
                          n_tcn=cfg.get("model",{}).get("n_tcn",4), kernel_size=cfg.get("model",{}).get("kernel_size",3),
                          n_layers_tf=cfg.get("model",{}).get("n_layers_tf",0), n_heads=cfg.get("model",{}).get("n_heads",4),
                          dropout=cfg.get("model",{}).get("dropout",0.1),
                          logv_min=logv_min, logv_max=logv_max, use_bounded=use_bounded,
                          variance_param=variance_param, aniso=aniso).to(device)
    
    # 模型初始化：用数据统计量初始化方差头（统一支持 sa3 与 direct）
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_batch = to_device(sample_batch, device)
        e2 = sample_batch["e2"]  # (B,T,3)
        m = sample_batch["mask"]  # (B,T)
        m_bool = m > 0.5
        e2_clean = torch.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0)

        # sa3 参数化：初始化尺度 s 与各向异性 a 的偏置
        if variance_param == "sa3":
            if model.var_head is not None:
                # 估算每轴初始方差 (sigma^2)
                e2_masked = torch.where(m_bool.unsqueeze(-1), e2_clean, torch.zeros_like(e2_clean))
                num = e2_masked.sum(dim=(0, 1))
                den = m_bool.sum().clamp_min(1.0)
                var_axes = (num / den).clamp_min(1e-12)
                nu_eff = max(float(tr.get("nu", 5.0)), 2.1)
                sigma2_init_axes = (var_axes * ((nu_eff - 2.0) / nu_eff)).clamp_min(1e-12)

                # 目标 s（有界空间中的尺度，等于 log(sigma^2) 的均值）
                s_target = float(torch.log(sigma2_init_axes).mean())
                # 将目标 s 映射回 raw 空间：raw = atanh((s - s_mid)/s_rad)
                denom = max(float(model.var_head.s_rad), 1e-6)
                x = (s_target - float(model.var_head.s_mid)) / denom
                x = max(min(x, 0.999), -0.999)
                raw_s_bias = 0.5 * math.log((1.0 + x) / (1.0 - x))
                model.var_head.head_s.bias.data.fill_(raw_s_bias)

                # 各向异性偏置初始化为 0（初始各向同性）
                model.var_head.head_axi.bias.data.zero_()

                print(f"[init] Sa3 head initialized: s_target={s_target:.3f}, raw_s_bias={raw_s_bias:.3f}, a_bias=[0., 0.]")
            else:
                print("[init] Warning: Sa3VarHead not found, skipping initialization.")
        else:
            # direct 参数化：保持原有初始化逻辑
            if model.head is not None:
                if d_out == 3:
                    e2_masked = torch.where(m_bool.unsqueeze(-1), e2_clean, torch.zeros_like(e2_clean))
                    num = e2_masked.sum(dim=(0,1))
                    den = m_bool.sum().clamp_min(1.0)
                    var_axes = (num / den).clamp_min(1e-12)
                    nu_eff = max(float(tr.get("nu", 5.0)), 2.1)
                    sigma2_init_axes = (var_axes * ((nu_eff - 2.0) / nu_eff)).clamp_min(1e-12)
                    model.head.bias.data.copy_(sigma2_init_axes.log().to(model.head.bias))
                    print(f"[init] head bias (3-axis): [{sigma2_init_axes[0]:.3e}, {sigma2_init_axes[1]:.3e}, {sigma2_init_axes[2]:.3e}]")
                else:
                    # 单轴初始化：三轴均值
                    e2_sum = e2_clean.sum(dim=-1) / 3.0  # (B,T)
                    e2_masked = torch.where(m_bool, e2_sum, torch.zeros_like(e2_sum))
                    var0 = (e2_masked.sum() / m_bool.sum()).clamp_min(1e-12)
                    model.head.bias.data.fill_(float(var0.log()))
                    print(f"[init] head bias (iso): {var0:.3e}")

    base_lr = tr.get("lr",1e-4)
    opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=tr.get("weight_decay",1e-4))
    
    # 调度器：优先使用Warmup+Cosine
    epochs = tr.get("epochs", 30)
    warmup_epochs = tr.get("warmup_epochs", 3)
    scheduler_type = tr.get("scheduler", "warmup_cosine")
    
    if scheduler_type == "warmup_cosine":
        warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_epochs))
        cosine = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs), eta_min=base_lr * 0.05)
        sch = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        # Suppress deprecation warning for SequentialLR
        warnings.filterwarnings('ignore', message='.*scheduler.step.*', category=UserWarning)
    else:
        sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_val = 1e9
    best_path = save_dir / "best.pt"
    patience = tr.get("early_stop_patience", 5)
    bad = 0
    
    # 其他损失参数
    nu = tr.get("nu", 5.0)  # Student-t自由度
    lambda_center = tr.get("lambda_center", 0.0)
    lambda_center_axes = tr.get("lambda_center_axes", 0.0)
    lambda_aniso_l2 = tr.get("lambda_aniso_l2", 0.0)
    lambda_tv = tr.get("lambda_tv", 0.0)
    
    # 调试打印：验证配置是否被正确读取
    print(f"[cfg] route={route} | use_studentt={use_studentt} nu={nu} " +
          f"logv∈[{logv_min}, {logv_max}] | variance_param={variance_param} use_bounded={use_bounded}")
    print(f"[cfg] λ(center_axes={lambda_center_axes}, aniso_l2={lambda_aniso_l2}, tv={lambda_tv}) | aniso={aniso}")

    for ep in range(1, epochs+1):
        model.train()
        tr_losses = []
        # Training with progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [Train]", ncols=100, leave=False)
        for batch in pbar:
            batch = to_device(batch, device)
            x = batch["x"]        # (B,T,D)
            e2 = batch["e2"]      # (B,T,3)
            m  = batch["mask"]    # (B,T)

            logv = model(x)  # (B,T,d_out)
            
            # 选择损失函数
            if use_studentt and d_out == 3:
                m_axes = m.unsqueeze(-1).expand_as(e2)
                e_axes = torch.sqrt(torch.clamp(e2, min=0.0))
                loss = nll_studentt_diag_axes(logv, e_axes, m_axes, nu, logv_min, logv_max,
                                              lambda_center_axes, lambda_aniso_l2, lambda_tv)
                parts = {"nll": loss.item(), "center": 0.0}
                
                # 调试：首次打印损失值
                if not hasattr(train_one_run, "_dbg_loss"):
                    train_one_run._dbg_loss = True
                    print(f"[train_loss] total={loss.item():.6f} (with constraints)")
            else:
                # 高斯iso NLL
                loss, parts = loss_total(e2, logv, m, logv_min, logv_max, lambda_center)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(tr_losses):.4f}'})

        # 验证
        model.eval()
        with torch.no_grad():
            vals = []
            all_e2, all_lv, all_m, all_raw = [], [], [], []
            pbar_val = tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [Val]  ", ncols=100, leave=False)
            for batch in pbar_val:
                batch = to_device(batch, device)
                x = batch["x"]
                e2 = batch["e2"]
                m  = batch["mask"]
                # 获取 logv 和 raw_logv（用于饱和度统计）
                if use_bounded and model.bounded is not None:
                    logv, raw_logv = model(x, return_raw=True)
                    all_raw.append(raw_logv.cpu())
                else:
                    logv = model(x)
                
                # 计算验证损失（验证时不使用约束，只看纯 NLL）
                if use_studentt and d_out == 3:
                    m_axes = m.unsqueeze(-1).expand_as(e2)
                    e_axes = torch.sqrt(torch.clamp(e2, min=0.0))
                    # 验证时约束系数设为 0，只计算纯 NLL
                    l = nll_studentt_diag_axes(logv, e_axes, m_axes, nu, logv_min, logv_max,
                                               0.0, 0.0, 0.0)
                else:
                    l, _ = loss_total(e2, logv, m, logv_min, logv_max, lambda_center)
                
                vals.append(l.item())
                # Collect for metrics
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
        
        # Compute and display metrics
        if len(all_e2) > 0:
            e2_cat = torch.cat(all_e2, dim=0)
            lv_cat = torch.cat(all_lv, dim=0)
            m_cat = torch.cat(all_m, dim=0)
            if use_studentt and d_out == 3:
                e_axes_cat = torch.sqrt(torch.clamp(e2_cat, min=0.0))
                m_axes_cat = m_cat.unsqueeze(-1).expand_as(e_axes_cat)
                z2_mean, cov68, cov95 = z2_cov_studentt_diag_axes(lv_cat, e_axes_cat, m_axes_cat, nu, logv_min, logv_max)
                
                # 计算 Spearman 相关性：e² vs v_eff（正确口径）
                s2 = torch.exp(lv_cat).clamp_min(1e-12)                 # (B,T,3)
                v_eff = s2 * (nu / (nu - 2.0))                          # (B,T,3)
                e2_mean = e2_cat.mean(dim=-1)                           # (B,T) 三轴误差平均
                var_mean = v_eff.mean(dim=-1)                           # (B,T) 三轴方差平均
                mask_bool = m_cat > 0.5
                sp = spearmanr_np(e2_mean[mask_bool].detach().cpu().numpy().ravel(), 
                                  var_mean[mask_bool].detach().cpu().numpy().ravel()) if mask_bool.sum() > 0 else 0.0
                
                # 计算饱和度（如果使用 bounded）
                sat_str = ""
                if use_bounded and len(all_raw) > 0:
                    raw_cat = torch.cat(all_raw, dim=0)
                    sat_ratio = model.bounded.saturation_ratio(raw_cat, threshold=0.98)
                    sat_str = f" sat={sat_ratio:.2%}"
                
                # 计算 sa3 统计（如果使用 sa3 参数化）
                aniso_str = ""
                if variance_param == "sa3" and d_out == 3:
                    from utils_sa3 import compute_aniso_stats
                    aniso_stats = compute_aniso_stats(lv_cat.numpy())
                    aniso_str = f" | s={aniso_stats['s_mean']:.3f} a_xy={aniso_stats['a_xy_std']:.3f} a_z={aniso_stats['a_z_std']:.3f}"
                
                print(f"[ep {ep:02d}] train={np.mean(tr_losses):.4f} val={val_loss:.4f} | z2={z2_mean:.3f} cov68={cov68:.2%} cov95={cov95:.2%} sp={sp:.3f}{sat_str}{aniso_str}")
            else:
                met = route_metrics_imu(e2_cat, lv_cat, m_cat, logv_min=logv_min, logv_max=logv_max)
                # 计算饱和度（如果使用 bounded）
                sat_str = ""
                if use_bounded and len(all_raw) > 0:
                    raw_cat = torch.cat(all_raw, dim=0)
                    sat_ratio = model.bounded.saturation_ratio(raw_cat, threshold=0.98)
                    sat_str = f" sat={sat_ratio:.2%}"
                
                # 计算 sa3 统计（如果使用 sa3 参数化）
                aniso_str = ""
                if variance_param == "sa3" and d_out == 3:
                    from utils_sa3 import compute_aniso_stats
                    aniso_stats = compute_aniso_stats(lv_cat.numpy())
                    aniso_str = f" | s={aniso_stats['s_mean']:.3f} a_xy={aniso_stats['a_xy_std']:.3f} a_z={aniso_stats['a_z_std']:.3f}"
                
                print(f"[ep {ep:02d}] train={np.mean(tr_losses):.4f} val={val_loss:.4f} | z2={met['z2_mean']:.3f} cov68={met['cov68']:.2%} cov95={met['cov95']:.2%} sp={met['spearman']:.3f}{sat_str}{aniso_str}")
        else:
            print(f"[ep {ep:02d}] train={np.mean(tr_losses):.4f} val={val_loss:.4f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss; bad = 0
            ensure_dir(best_path.parent)
            torch.save({
                "model": model.state_dict(), 
                "cfg": cfg, 
                "route": route,
                "d_in": d_in,
                "d_out": d_out,
                "logv_min": logv_min,
                "logv_max": logv_max,
                "use_bounded": use_bounded,
                "variance_param": variance_param,
                "aniso": aniso
            }, best_path)
        else:
            bad += 1
            if bad >= patience:
                print("early stop")
                break

    return str(best_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["acc","gyr"], required=True)
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config_file(args.config)
    save_dir = Path(cfg.get("common",{}).get("log_dir","runs")) / args.route
    best = train_one_run(cfg, args.route, save_dir)
    print("best:", best)

if __name__ == "__main__":
    main()
