from __future__ import annotations
import argparse, os, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, LinearLR, SequentialLR

from utils import seed_everything, set_all_seeds, create_worker_init_fn, to_device, count_params, load_config_file
from dataset import build_loader
from models import IMURouteModel
from losses import nll_iso3_e2, nll_iso2_e2, mse_anchor_1d, nll_diag_axes, nll_diag_axes_weighted, nll_studentt_diag_axes, mse_anchor_axes, adaptive_nll_loss
from metrics import route_metrics_imu, route_metrics_vis, route_metrics_gns_axes

def parse_args():
    # 先只解析 --config 和 --route（不加载其它参数）
    pre_cfg = argparse.ArgumentParser(add_help=False)
    pre_cfg.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    pre_route = argparse.ArgumentParser(add_help=False)
    pre_route.add_argument("--route", choices=["acc","gyr","vis","gns"], default=None)

    args_pre_cfg, _ = pre_cfg.parse_known_args()
    args_pre_route, _ = pre_route.parse_known_args()

    cfg = load_config_file(args_pre_cfg.config)

    # 根据"命令行的 --route（优先）"或"配置里是否存在 train_gns 段（兜底）"选择前缀
    if args_pre_route.route is not None:
        route_hint = args_pre_route.route
    else:
        # 配置文件没有 route 明示时，用是否存在 train_gns/model_gns 来猜测
        route_hint = "gns" if ("train_gns" in cfg or "model_gns" in cfg or "eval_gns" in cfg) else "acc"

    if route_hint == "gns":
        tr = cfg.get("train_gns", cfg.get("train", {}))
        md = cfg.get("model_gns", cfg.get("model", {}))
    else:
        tr = cfg.get("train", {})
        md = cfg.get("model", {})

    rt = cfg.get("runtime", {})

    # 真正的参数解析器
    ap = argparse.ArgumentParser("Train single-route IMU variance model", parents=[pre_cfg])
    ap.add_argument("--route", choices=["acc","gyr","vis","gns"], default=tr.get("route", route_hint),
                    help="Which route to train")
    ap.add_argument("--train_npz", required=(tr.get("train_npz") is None), default=tr.get("train_npz"))
    ap.add_argument("--val_npz", required=(tr.get("val_npz") is None), default=tr.get("val_npz"))
    ap.add_argument("--test_npz", required=(tr.get("test_npz") is None), default=tr.get("test_npz"))
    ap.add_argument("--x_mode", choices=["both","route_only"], default=tr.get("x_mode","both"))
    ap.add_argument("--run_dir", required=(tr.get("run_dir") is None), default=tr.get("run_dir"))
    ap.add_argument("--epochs", type=int, default=tr.get("epochs",20))
    ap.add_argument("--batch_size", type=int, default=tr.get("batch_size",32))
    ap.add_argument("--lr", type=float, default=tr.get("lr",1e-3))
    ap.add_argument("--seed", type=int, default=tr.get("seed",0))
    ap.add_argument("--d_model", type=int, default=md.get("d_model",128))
    ap.add_argument("--n_tcn", type=int, default=md.get("n_tcn",4))
    ap.add_argument("--kernel_size", type=int, default=md.get("kernel_size",3))
    ap.add_argument("--n_heads", type=int, default=md.get("n_heads",4))
    ap.add_argument("--n_layers_tf", type=int, default=md.get("n_layers_tf",2))
    ap.add_argument("--dropout", type=float, default=md.get("dropout",0.1))
    ap.add_argument("--num_workers", type=int, default=rt.get("num_workers",0))
    ap.add_argument("--logv_min", type=float, default=tr.get("logv_min",-12.0))
    ap.add_argument("--logv_max", type=float, default=tr.get("logv_max",6.0))
    ap.add_argument("--z2_center", type=float, default=tr.get("z2_center",0.0), help="z²居中正则化权重")
    ap.add_argument("--z2_center_target", type=str, default=tr.get("z2_center_target","auto"), help="z²目标值: 'auto' 或数字")
    ap.add_argument("--anchor_weight", type=float, default=tr.get("anchor_weight",0.0))
    ap.add_argument("--early_patience", type=int, default=tr.get("early_patience", 10))
    # 轴感知 & 自适应（只对 GNSS 有效，但参数照常解析）
    ap.add_argument("--early_axis", action="store_true", default=tr.get("early_axis", True),
                    help="使用'最差轴 |E[z²]-1|'做早停监控（GNSS）")
    ap.add_argument("--axis_auto_balance", action="store_true", default=tr.get("axis_auto_balance", True),
                    help="对 GNSS 逐轴 NLL 引入按轴权重，并按验证集 |E[z²]-1| 自适应更新")
    ap.add_argument("--axis_power", type=float, default=tr.get("axis_power", 1.0),
                    help="轴权重 ~ dev^p 的指数 p")
    ap.add_argument("--axis_clip", type=str, default=tr.get("axis_clip", "0.5,2.0"),
                    help="权重裁剪区间 lo,hi")
    ap.add_argument("--student_nu", type=float, default=tr.get("student_nu", 0.0),
                    help="Student-t 自由度参数（0=使用高斯NLL，>0=使用t-NLL，推荐3.0）")
    ap.add_argument("--anchor_axes_weight", type=float, default=tr.get("anchor_axes_weight", 0.0),
                    help="GNSS 逐轴 vendor 软锚权重（0 关闭）")
    ap.add_argument("--post_scale", action="store_true", default=tr.get("post_scale", False),
                    help="在验证集上做一次温度缩放，把 z² 拉回 1")
    ap.add_argument("--device", default=rt.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    # Learning rate scheduler parameters
    ap.add_argument('--lr_sched', type=str, default=tr.get('lr_sched', 'none'),
                    choices=['none', 'cosine', 'plateau', 'onecycle'],
                    help="Learning rate scheduler type")
    ap.add_argument('--warmup_epochs', type=int, default=tr.get('warmup_epochs', 3),
                    help="Number of warmup epochs for cosine scheduler")
    ap.add_argument('--min_lr', type=float, default=tr.get('min_lr', 1e-5),
                    help="Minimum learning rate")
    ap.add_argument('--plateau_patience', type=int, default=tr.get('plateau_patience', 3),
                    help="Patience for ReduceLROnPlateau scheduler")
    ap.add_argument('--plateau_factor', type=float, default=tr.get('plateau_factor', 0.5),
                    help="Factor for ReduceLROnPlateau scheduler")
    # IMU loss options
    ap.add_argument('--imu_loss', type=str, choices=['iso','studentt_diag'], default=tr.get('imu_loss','iso'),
                    help="IMU loss: iso (scalar var) or studentt_diag (per-axis t-NLL)")

    args = ap.parse_args()

    # 启动时打印边界，防止再踩到"没有用上配置段"的坑
    nll_type = f"Student-t(ν={args.student_nu})" if args.student_nu > 0 else "Gaussian"
    print(f"[args] route={args.route}  logv_min={args.logv_min}  logv_max={args.logv_max}  NLL={nll_type}")

    return args

def main():
    args = parse_args()
    # 设置全家桶种子，确保完全可重现性
    set_all_seeds(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    # 创建确定性Generator和worker_init_fn
    g = torch.Generator()
    g.manual_seed(args.seed)
    worker_init_fn = create_worker_init_fn(args.seed) if args.num_workers > 0 else None

    # Data
    train_ds, train_dl = build_loader(args.train_npz, route=args.route, x_mode=args.x_mode,
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      generator=g, worker_init_fn=worker_init_fn)
    val_ds,   val_dl   = build_loader(args.val_npz,   route=args.route, x_mode=args.x_mode,
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers,
                                      generator=g, worker_init_fn=worker_init_fn)
    test_ds,  test_dl  = build_loader(args.test_npz,  route=args.route, x_mode=args.x_mode,
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers,
                                      generator=g, worker_init_fn=worker_init_fn)

    # 动态确定输入/输出维度
    if args.route == "gns":
        # 对于GNSS，从数据集直接获取维度
        sample_batch = next(iter(train_dl))
        d_in = sample_batch["X"].shape[-1]
        d_out = 3                      # ← 各向异性：ENU 三通道 logvar
    elif args.route == "vis":
        d_in = train_ds.X_all.shape[-1]
        d_out = 1  # VIS: 1维聚合误差
    else:
        d_in = train_ds.X_all.shape[-1] if args.x_mode=="both" else 3
        # IMU: 若使用逐轴 Student-t 且数据为步级标签，则输出3维
        if getattr(train_ds, 'use_step_labels', False) and args.imu_loss == 'studentt_diag':
            d_out = 3
        else:
            d_out = 1  # 默认 1 维
    
    model = IMURouteModel(d_in=d_in, d_out=d_out, d_model=args.d_model, n_tcn=args.n_tcn, kernel_size=args.kernel_size,
                          n_layers_tf=args.n_layers_tf, n_heads=args.n_heads, dropout=args.dropout).to(args.device)
    print(f"[model] params={count_params(model):,}  d_in={d_in}  d_out={d_out}")

    # ---- Warm start the head bias with data statistics ----
    with torch.no_grad():
        # 使用固定的seed和generator确保每次取到相同的batch
        fixed_gen = torch.Generator()
        fixed_gen.manual_seed(args.seed + 12345)  # 固定偏移避免与主generator重复
        
        # 创建确定性的单batch loader
        temp_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=min(args.batch_size, len(train_ds)), 
            shuffle=True, generator=fixed_gen
        )
        b = next(iter(temp_dl))
        b = to_device(b, args.device)
        if args.route == "gns":
            e2_axes = b["E2_AXES"].float()                 # (B,T,3)
            m_axes  = b["MASK_AXES"].float()
            num = (e2_axes * m_axes).sum(dim=(0,1))
            den = m_axes.sum(dim=(0,1)).clamp_min(1.0)
            var0 = (num / den).clamp_min(1e-12)            # (3,)
            model.head.bias.data = var0.log().to(model.head.bias)
            print(f"[warm-start] GNSS head bias initialized: E={var0[0]:.3e}, N={var0[1]:.3e}, U={var0[2]:.3e}")
        elif args.route == "vis":
            e2 = b["E2"].float().squeeze(-1); m = b["MASK"].float()
            var0 = ((e2 * m).sum() / m.sum() / 2.0).clamp_min(1e-12)  # df=2
            model.head.bias.data.fill_(float(var0.log()))
            print(f"[warm-start] VIS head bias initialized: var={var0:.3e}")
        else:
            # 适配步级标签：E2 可能是 (B,T,3) 或 (B,T,1)/(B,T)
            e2_any = b["E2"].float()
            m_any  = b["MASK"].float()
            if m_any.dim() == 3 and m_any.size(-1) == 1:
                m_any = m_any.squeeze(-1)
            eps = 1e-12
            if e2_any.dim() == 3 and e2_any.size(-1) == 3 and getattr(model.head, 'out_features', None) == 3:
                # 逐轴初始化（NaN-safe + 真正按mask统计）
                m_bool = (m_any > 0.5)
                e2_clean = torch.nan_to_num(e2_any, nan=0.0, posinf=0.0, neginf=0.0)
                e2_masked = torch.where(m_bool.unsqueeze(-1), e2_clean, torch.zeros_like(e2_clean))
                num = e2_masked.sum(dim=(0,1))  # (3,)
                den = m_bool.sum().clamp_min(1.0)
                var_axes = (num / den).clamp_min(1e-12)
                model.head.bias.data.copy_(var_axes.log().to(model.head.bias))
                print(f"[warm-start] IMU head bias initialized (diag): var=({var_axes[0]:.3e},{var_axes[1]:.3e},{var_axes[2]:.3e})")
            else:
                # 单轴或模型头是1维：用等效均值
                if e2_any.dim() == 3 and e2_any.size(-1) == 3:
                    e2_iso = (e2_any.sum(dim=-1) / 3.0)  # (B,T)
                else:
                    e2_iso = e2_any.squeeze(-1) if e2_any.dim() == 3 else e2_any
                m_bool = (m_any > 0.5)
                e2_iso_clean = torch.nan_to_num(e2_iso, nan=0.0, posinf=0.0, neginf=0.0)
                e2_iso_masked = torch.where(m_bool, e2_iso_clean, torch.zeros_like(e2_iso_clean))
                denom = m_bool.sum().clamp_min(1.0)
                var0 = e2_iso_masked.sum() / denom
                var0 = var0.clamp_min(1e-12)
                model.head.bias.data.fill_(float(var0.log()))
                print(f"[warm-start] IMU head bias initialized: var={var0:.3e}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_sched == 'cosine':
        warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                          total_iters=max(1, args.warmup_epochs))
        cosine = CosineAnnealingLR(opt,
                                   T_max=max(1, args.epochs - args.warmup_epochs),
                                   eta_min=args.min_lr)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine],
                                 milestones=[args.warmup_epochs])
    elif args.lr_sched == 'plateau':
        scheduler = ReduceLROnPlateau(opt, mode='min',
                                      factor=args.plateau_factor,
                                      patience=args.plateau_patience,
                                      min_lr=args.min_lr, verbose=True)
    elif args.lr_sched == 'onecycle':
        scheduler = OneCycleLR(opt, max_lr=args.lr,
                               epochs=args.epochs,
                               steps_per_epoch=len(train_dl),
                               pct_start=0.1, final_div_factor=max(10, args.lr/args.min_lr))
    
    best_val = 1e9   # 兼容原有基于 val_loss 的逻辑
    best_worst = 1e9 # 轴感知用
    epochs_since_improve = 0
    best_path = str(Path(args.run_dir) / "best.pt")
    
    # 轴权重（仅 GNSS 生效）
    lo, hi = map(float, args.axis_clip.split(","))
    axis_w = torch.ones(3, device=args.device)

    def run_epoch(loader, training: bool):
        model.train(training)
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = to_device(batch, args.device)
            x, m, y = batch["X"], batch["MASK"], batch["Y"]
            logv = model(x)
            if args.route == "vis":
                loss = nll_iso2_e2(batch["E2"], logv, m,
                                   logv_min=args.logv_min, logv_max=args.logv_max)
            elif args.route == "gns":
                # GNSS：逐轴各向异性（可选按轴加权 + Student-t NLL）
                if args.student_nu > 0:
                    # 使用稳健的 Student-t NLL（对异常值更稳健）
                    loss = nll_studentt_diag_axes(batch["E2_AXES"], logv, batch["MASK_AXES"],
                                                  nu=args.student_nu,
                                                  logv_min=args.logv_min, logv_max=args.logv_max)
                elif args.axis_auto_balance:
                    # 使用加权高斯 NLL
                    loss, per_axis_nll = nll_diag_axes_weighted(batch["E2_AXES"], logv, batch["MASK_AXES"],
                                                                axis_w=axis_w,
                                                                logv_min=args.logv_min, logv_max=args.logv_max)
                else:
                    # 使用标准高斯 NLL
                    loss = nll_diag_axes(batch["E2_AXES"], logv, batch["MASK_AXES"],
                                         logv_min=args.logv_min, logv_max=args.logv_max)
                
                # —— GNSS 逐轴 vendor 软锚（可选）
                if args.anchor_axes_weight > 0 and ("VENDOR_VAR_AXES" in batch):
                    loss = loss + mse_anchor_axes(logv, batch["VENDOR_VAR_AXES"], batch["MASK_AXES"], 
                                                 lam=args.anchor_axes_weight)
            else:
                # IMU (acc/gyr)
                e2 = batch["E2"]
                use_step_labels = (e2.size(-1) == 3)
                if args.imu_loss == 'studentt_diag' and use_step_labels:
                    # 逐轴 Student-t（对异常值稳健）
                    if logv.dim() == 2:
                        # 头是1维，扩展到3维以匹配 e2
                        logv = logv.unsqueeze(-1).expand_as(e2)
                    elif logv.size(-1) == 1:
                        logv = logv.expand_as(e2)
                    loss = nll_studentt_diag_axes(e2, logv, m.unsqueeze(-1).expand_as(e2),
                                                  nu=max(args.student_nu, 6.0),
                                                  logv_min=args.logv_min, logv_max=args.logv_max)
                else:
                    # 兼容旧口径：自适应一维 NLL（含回退/锚点）
                    y_anchor = batch.get("Y_anchor", None)
                    loss = adaptive_nll_loss(logv, e2, m,
                                             use_step_labels=use_step_labels,
                                             y_anchor=y_anchor,
                                             logv_min=args.logv_min,
                                             logv_max=args.logv_max,
                                             route=args.route)
                    if args.anchor_weight > 0 and not use_step_labels:
                        loss = loss + mse_anchor_1d(logv, y, m, lam=args.anchor_weight)
            
            # z²居中正则化（通用于所有路由）
            if args.z2_center > 0:
                # 与 NLL 一致地 clamp，再求方差
                lv = torch.clamp(logv, min=args.logv_min, max=args.logv_max)
                v = torch.exp(lv).clamp_min(1e-12)
                
                # 居中目标：VIS/IMU 仍按聚合 df；GNSS（各向异性）按逐轴 z²
                if args.route == "gns" and logv.shape[-1] == 3:
                    e2_axes = batch["E2_AXES"]
                    m_axes  = batch["MASK_AXES"].float()
                    z2 = (e2_axes / v)                 # (B,T,3), 1D z²
                    m_float = m_axes
                else:
                    # VIS/IMU 路由的z²正则化
                    if args.route == "vis":
                        df = 2.0
                    else:
                        df = 3.0
                    
                    e2 = batch["E2"]  # (B,T,D)
                    m_step = m.float().unsqueeze(-1)  # (B,T,1)
                    
                    if e2.size(-1) == 3:  # 步级标签模式
                        # 三轴步级标签：先求和再除以df
                        e2_sum = e2.sum(dim=-1)  # (B,T)
                        v_avg = v.mean(dim=-1) if v.size(-1) > 1 else v.squeeze(-1)  # (B,T)
                        z2 = (e2_sum / v_avg) / df  # (B,T)
                        m_float = m.float()  # (B,T)
                    else:
                        # 回退模式：窗口标签扩展
                        e2 = e2.squeeze(-1)  # (B,T)
                        v = v.squeeze(-1)  # (B,T)
                        z2 = (e2 / v) / df  # (B,T)
                        m_float = m.float()  # (B,T)
                        
                mean_z2 = (z2 * m_float).sum() / m_float.clamp_min(1.0).sum()
                
                # 目标值：高斯=1；若使用 Student-t 且 ν>2，则 target=ν/(ν-2)
                if args.z2_center_target == "auto":
                    if args.student_nu and args.student_nu > 2.0:
                        target = args.student_nu / (args.student_nu - 2.0)
                    else:
                        target = 1.0
                else:
                    target = float(args.z2_center_target)
                
                loss = loss + args.z2_center * (mean_z2 - target).pow(2)
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = run_epoch(train_dl, True)
        val_loss = run_epoch(val_dl, False)

        # Validation metrics（抽一批看 ez2/coverage/Spearman/饱和率）
        with torch.no_grad():
            model.eval()
            val_batch = next(iter(val_dl))
            val_batch = to_device(val_batch, args.device)
            logv = model(val_batch["X"])
            if args.route == "vis":
                stats = route_metrics_vis(val_batch["E2"], logv, val_batch["MASK"], args.logv_min, args.logv_max)
            elif args.route == "gns":
                # GNSS：逐轴指标
                stats = route_metrics_gns_axes(val_batch["E2_AXES"], logv, val_batch["MASK_AXES"],
                                               args.logv_min, args.logv_max)
            else:
                stats = route_metrics_imu(val_batch["E2"], logv, val_batch["MASK"], args.logv_min, args.logv_max)

            # === 轴感知统计（GNSS）===
            worst_dev = None
            ez2_axes_print = ""
            if args.route == "gns":
                num = torch.zeros(3, device=args.device)
                den = torch.zeros(3, device=args.device)
                for val_batch in val_dl:
                    val_batch = to_device(val_batch, args.device)
                    logv = model(val_batch["X"])                     # (B,T,3)
                    lv = torch.clamp(logv, min=args.logv_min, max=args.logv_max)
                    v  = torch.exp(lv).clamp_min(1e-12)
                    e2 = val_batch["E2_AXES"]                        # (B,T,3)
                    m  = val_batch["MASK_AXES"].float()              # (B,T,3)
                    z2 = e2 / v
                    num += (z2 * m).sum(dim=(0,1))
                    den += m.sum(dim=(0,1)).clamp_min(1.0)
                ez2_axes = (num / den).detach()                      # (3,)
                worst_dev = torch.abs(ez2_axes - 1.0).max().item()
                ez2_axes_print = f" ez2[E,N,U]=[{ez2_axes[0]:.3f},{ez2_axes[1]:.3f},{ez2_axes[2]:.3f}] worst={worst_dev:.3f}"

                # 按轴自适应权重（B）：谁偏得远谁更重
                if args.axis_auto_balance:
                    dev = (ez2_axes - 1.0).abs().clamp_min(1e-3)     # (3,)
                    new_w = dev.pow(args.axis_power)
                    new_w = (new_w / new_w.mean()).clamp_(lo, hi)     # 归一 + 裁剪
                    axis_w = new_w.detach()

        # Get current learning rate for logging
        cur_lr = opt.param_groups[0]['lr']
        
        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  lr={cur_lr:.3e}"
              f" z2_mean={stats['z2_mean']:.3f} cov68={stats['cov68']:.3f} cov95={stats['cov95']:.3f} "
              f"spear={stats['spear']:.3f} sat={stats['sat']:.3f}(↓{stats.get('sat_min',0):.3f}↑{stats.get('sat_max',0):.3f})"
              f"{ez2_axes_print}  time={time.time()-t0:.1f}s")

        # === A：轴感知早停 ===
        improved = False
        if args.route == "gns" and args.early_axis and worst_dev is not None:
            if epoch == 1 or worst_dev < best_worst:
                best_worst = worst_dev
                improved = True
        else:
            if val_loss < best_val:
                best_val = val_loss
                improved = True

        if improved:
            epochs_since_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.early_patience:
                print(f"[early-stop] No improvement for {args.early_patience} epochs. Stopping at epoch {epoch}.")
                break

        # Step the learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    # Final test - iterate over all batches like eval.py
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    
    agg, n = None, 0
    with torch.no_grad():
        for batch in test_dl:
            batch = to_device(batch, args.device)
            logv = model(batch["X"])
            if args.route == "vis":
                st = route_metrics_vis(batch["E2"], logv, batch["MASK"], args.logv_min, args.logv_max)
            elif args.route == "gns":
                st = route_metrics_gns_axes(batch["E2_AXES"], logv, batch["MASK_AXES"], args.logv_min, args.logv_max)
            else:
                st = route_metrics_imu(batch["E2"], logv, batch["MASK"], args.logv_min, args.logv_max)
            if agg is None: 
                agg = {k: 0.0 for k in st}
            for k, v in st.items(): 
                agg[k] += float(v)
            n += 1
    tst = {k: v/n for k, v in agg.items()}
    
    with open(Path(args.run_dir)/"final_test_metrics.json","w",encoding="utf-8") as f:
        json.dump(tst, f, ensure_ascii=False, indent=2)
    print("[test]", tst)

if __name__ == "__main__":
    main()
