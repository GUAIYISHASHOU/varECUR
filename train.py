# -*- coding: utf-8 -*-`r`nfrom __future__ import annotations
import argparse, os, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, LinearLR, SequentialLR

from utils import seed_everything, set_all_seeds, create_worker_init_fn, to_device, count_params, load_config_file
from dataset import build_loader
from models import IMURouteModel
from losses import nll_iso3_e2, nll_iso2_e2, mse_anchor_1d, nll_diag_axes, nll_diag_axes_weighted, nll_studentt_diag_axes, mse_anchor_axes, adaptive_nll_loss, nll_gauss_huber_iso3, nll_gauss_huber_iso2, nll_gaussian_e2_step, nll_gauss_huber_e2_step, nll_gauss_huber_e2_step_with_ema, vis_center_regularization, nll_vis_diag_2d, vis_center_regularization_2d
from metrics import route_metrics_imu, route_metrics_vis, route_metrics_gns_axes, DF_BY_ROUTE

# ==== 放在 train.py 顶部附近 ====
from torch.utils.data.dataloader import default_collate
import math

# ==== VIS Diag helpers ====
def huber_abs(x, delta: float):
    """Huber loss on absolute value of x."""
    ax = x.abs()
    quad = 0.5 * (ax ** 2)
    lin  = delta * (ax - 0.5 * delta)
    return torch.where(ax <= delta, quad, lin)

class AxisBalancerEMA:
    """Keep EMA of axis losses and return inverse-proportional weights."""
    def __init__(self, beta=0.9, gamma=1.0):
        self.beta = beta
        self.gamma = gamma
        self.Lx_ema = None
        self.Ly_ema = None

    def weights(self, Lx_mean: torch.Tensor, Ly_mean: torch.Tensor):
        lx = Lx_mean.detach().item()
        ly = Ly_mean.detach().item()
        if self.Lx_ema is None:
            self.Lx_ema, self.Ly_ema = lx, ly
        else:
            self.Lx_ema = self.beta * self.Lx_ema + (1 - self.beta) * lx
            self.Ly_ema = self.beta * self.Ly_ema + (1 - self.beta) * ly
        # inverse-proportional weights with gamma
        gx = self.Lx_ema ** self.gamma
        gy = self.Ly_ema ** self.gamma
        denom = gx + gy + 1e-12
        wx = gy / denom
        wy = gx / denom
        return float(wx), float(wy)

_vis_dbg_cnt = {"mask_broadcast": 0}
def collate_vis_debug(batch):
    # batch: list of dict
    # 先默认堆�?
    B = len(batch)
    # 处理每个样本�?E2/MASK 形状
    for s in batch:
        for k in ("E2", "MASK"):
            if k in s and hasattr(s[k], "ndim") and s[k].ndim == 2 and s[k].shape[-1] == 1:
                s[k] = s[k].squeeze(-1)  # (T,)
    # �?mask 是标�?单步�?e2 �?(T,) -> 广播
    for s in batch:
        if "E2" in s and "MASK" in s:
            e2, m = s["E2"], s["MASK"]
            if hasattr(e2, "shape") and hasattr(m, "shape"):
                if len(e2.shape) == 1 and len(m.shape) == 1 and m.shape[0] == 1 and e2.shape[0] > 1:
                    if _vis_dbg_cnt["mask_broadcast"] < 6:  # 限制打印次数
                        _vis_dbg_cnt["mask_broadcast"] += 1
                        print(f"[VIS][fix] collate: broadcast MASK (1,) -> (T={e2.shape[0]}). "
                              f"请从 Dataset 源头修正为逐步掩码")
                    import numpy as np_local
                    s["MASK"] = np_local.broadcast_to(m, e2.shape)
    return default_collate(batch)

def parse_args():
    # 先只解析 --config �?--route（不加载其它参数�?
    pre_cfg = argparse.ArgumentParser(add_help=False)
    pre_cfg.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    pre_route = argparse.ArgumentParser(add_help=False)
    pre_route.add_argument("--route", choices=["acc","gyr","vis","gns"], default=None)

    args_pre_cfg, _ = pre_cfg.parse_known_args()
    args_pre_route, _ = pre_route.parse_known_args()

    cfg = load_config_file(args_pre_cfg.config)

    # 根据"命令行的 --route（优先）"�?配置里是否存�?train_gns 段（兜底�?选择前缀
    if args_pre_route.route is not None:
        route_hint = args_pre_route.route
    else:
        # 配置文件没有 route 明示时，用是否存�?train_gns/model_gns 来猜�?
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
    ap.add_argument("--x_mode", choices=["both","route_only","imu","visual"], default=tr.get("x_mode","both"))
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
    ap.add_argument("--z2_center", type=float, default=tr.get("z2_center",0.0), help="z²居中正则化权")
    ap.add_argument("--z2_center_target", type=str, default=tr.get("z2_center_target","auto"), help="z²目标�? 'auto' 或数")
    ap.add_argument("--anchor_weight", type=float, default=tr.get("anchor_weight",0.0))
    ap.add_argument("--early_patience", type=int, default=tr.get("early_patience", 10))
    # 轴感�?& 自适应（只�?GNSS 有效，但参数照常解析�?
    ap.add_argument("--early_axis", action="store_true", default=tr.get("early_axis", True),
                    help="使用'最差轴 |E[z²]-1|'做早停监控（GNSS")
    ap.add_argument("--axis_auto_balance", action="store_true", default=tr.get("axis_auto_balance", True),
                    help="�?GNSS 逐轴 NLL 引入按轴权重，并按验证集 |E[z²]-1| 自适应更新")
    ap.add_argument("--axis_power", type=float, default=tr.get("axis_power", 1.0),
                    help="轴权�?~ dev^p 的指�?p")
    ap.add_argument("--axis_clip", type=str, default=tr.get("axis_clip", "0.5,2.0"),
                    help="权重裁剪区间 lo,hi")
    ap.add_argument("--student_nu", type=float, default=tr.get("student_nu", 0.0),
                    help="Student-t 自由度参数（0=使用高斯NLL�?0=使用t-NLL，推�?.0")
    ap.add_argument("--anchor_axes_weight", type=float, default=tr.get("anchor_axes_weight", 0.0),
                    help="GNSS 逐轴 vendor 软锚权重�? 关闭")
    ap.add_argument("--post_scale", action="store_true", default=tr.get("post_scale", False),
                    help="在验证集上做一次温度缩放，�?z² 拉回 1")
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
    ap.add_argument('--imu_loss', type=str, 
                    choices=['iso','gauss_huber','studentt_diag'], 
                    default=tr.get('imu_loss','gauss_huber'),
                    help="IMU loss: iso (scalar Gaussian), gauss_huber (Gaussian+Huber), or studentt_diag (per-axis t-NLL)")
    
    # Huber 损失的超�?
    ap.add_argument('--huber_delta', type=float, default=tr.get('huber_delta', 1.5))
    ap.add_argument('--cal_reg', type=float, default=tr.get('cal_reg', 5e-2), help="E[z^2] center regularizer weight")
    # 你已有的 anchor_weight 继续沿用（作为窗口尺度锚的权重）

    # Vision loss options
    ap.add_argument('--vis_loss', choices=['iso','gauss_huber'], default=tr.get('vis_loss','iso'))
    ap.add_argument('--vis_huber_delta', type=float, default=tr.get('vis_huber_delta', 1.5))
    ap.add_argument('--vis_center_weight', type=float, default=tr.get('vis_center_weight', 5e-2), 
                    help="VIS居中正则化权重，防止方差学得过小")
    ap.add_argument('--vis_center_weight_start', type=float, default=tr.get('vis_center_weight_start', tr.get('vis_center_weight', 5e-2)),
                    help="VIS center weight schedule start (high)")
    ap.add_argument('--vis_center_weight_end', type=float, default=tr.get('vis_center_weight_end', tr.get('vis_center_weight', 5e-2)),
                    help="VIS center weight schedule end (low)")
    ap.add_argument('--vis_ema_aux_alpha', type=float, default=tr.get('vis_ema_aux_alpha', 0.0),
                    help="EMA shape align auxiliary weight")
    ap.add_argument('--vis_ema_tau', type=int, default=tr.get('vis_ema_tau', 5),
                    help="EMA temporal window (steps)")
    ap.add_argument('--vis_2d_diag', action='store_true', default=tr.get('vis_2d_diag', False),
                    help="使用2维对角协方差(x,y)而非1维聚合误")
    ap.add_argument('--vis_axis_balance', type=str, default=tr.get('vis_axis_balance', 'ema'),
                    choices=['ema', 'none'], help="Axis balance mode for VIS 2D diag")
    ap.add_argument('--vis_axis_beta', type=float, default=tr.get('vis_axis_beta', 0.9),
                    help="EMA beta for axis balancing")
    ap.add_argument('--vis_axis_gamma', type=float, default=tr.get('vis_axis_gamma', 1.0),
                    help="Power for inverse-proportional weighting")
    ap.add_argument('--vis_huber_delta_x', type=float, default=tr.get('vis_huber_delta_x', None),
                    help="Huber delta for x-axis (default: use huber_delta)")
    ap.add_argument('--vis_huber_delta_y', type=float, default=tr.get('vis_huber_delta_y', None),
                    help="Huber delta for y-axis (default: use huber_delta)")

    args = ap.parse_args()

    # 启动时打印边界，防止再踩�?没有用上配置�?的坑
    if args.imu_loss == 'gauss_huber':
        nll_type = f"Gaussian+Huber(δ={args.huber_delta}, λ={args.cal_reg})"
    elif args.imu_loss == 'studentt_diag':
        nll_type = f"Student-t(ν={args.student_nu})"
    elif args.student_nu > 0:
        nll_type = f"Student-t(ν={args.student_nu})"
    else:
        nll_type = "Gaussian"
    print(f"[args] route={args.route}  logv_min={args.logv_min}  logv_max={args.logv_max}  NLL={nll_type}")
    # 明确记录 VIS 损失分支，便于排�?
    if args.route == "vis":
        print(f"[which-loss] VIS loss = {args.vis_loss}")

    return args

def main():
    args = parse_args()
    # 设置全家桶种子，确保完全可重现�?
    set_all_seeds(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    # 创建确定性Generator和worker_init_fn
    g = torch.Generator()
    g.manual_seed(args.seed)
    worker_init_fn = create_worker_init_fn(args.seed) if args.num_workers > 0 else None

    # Data
    if args.route == "vis":
        from torch.utils.data import DataLoader
        from dataset import IMURouteDataset
        train_ds = IMURouteDataset(args.train_npz, route=args.route, x_mode=args.x_mode)
        val_ds = IMURouteDataset(args.val_npz, route=args.route, x_mode=args.x_mode)
        test_ds = IMURouteDataset(args.test_npz, route=args.route, x_mode=args.x_mode)
        
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_vis_debug,
                              generator=g, worker_init_fn=worker_init_fn, pin_memory=True)
        val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_vis_debug,
                              generator=g, worker_init_fn=worker_init_fn, pin_memory=True)
        test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_vis_debug,
                              generator=g, worker_init_fn=worker_init_fn, pin_memory=True)
    else:
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

    # ==== DEBUG: VIS 数据 sanity（训�?验证各看一眼） ====
    if args.route == "vis":
        def _peek_vis_batch(dloader, tag):
            b = next(iter(dloader))
            e2 = b["E2"]; m = b["MASK"].float()
            if e2.dim()==3 and e2.size(-1)==1: e2 = e2.squeeze(-1)  # -> (B,T)
            if m.dim()==3  and m.size(-1)==1:  m  = m.squeeze(-1)   # -> (B,T)
            if m.size() != e2.size():
                if m.size(1) == 1 and e2.size(1) > 1:
                    print(f"[fix] {tag} MASK is (B,1); broadcasting to (B,T). Please fix dataset VIS mask.")
                    m = m.expand(-1, e2.size(1))
                else:
                    raise RuntimeError(f"{tag} shape mismatch: E2 {tuple(e2.shape)} vs MASK {tuple(m.shape)}")
            valid = (m>0.5)
            e2v = torch.nan_to_num(e2, 0.0, 0.0, 0.0)[valid]
            print(f"[sanity] {tag}: B={e2.shape[0]} T={e2.shape[1]}  mask.sum={int(valid.sum())}")
            if e2v.numel()>0:
                qs = torch.quantile(e2v, torch.tensor([0.0,0.5,0.9,0.99], device=e2v.device))
                print(f"[sanity] {tag}: e2(valid) min/med/p90/p99/max = "
                      f"{e2v.min().item():.3g}/{qs[1].item():.3g}/{qs[2].item():.3g}/{qs[3].item():.3g}/{e2v.max().item():.3g}")
                print(f"[sanity] {tag}: e2(valid)>0  frac = {(e2v>0).float().mean().item():.3%}")
            else:
                print(f"[sanity] {tag}: NO valid e2 after mask.")

        _peek_vis_batch(train_dl, "loader(train)")
        _peek_vis_batch(val_dl,   "loader(val)")

    # === 体检 1/2：npz 有效步统计（优先 VIS 键）===
    if args.route == "vis":
        try:
            import numpy as np
            z = np.load(args.train_npz, allow_pickle=False)
            E2z = z.get('E2_VIS', z.get('E2'))
            Mz  = z.get('MASK_VIS', z.get('MASK'))
            v   = ((Mz.reshape(-1)>0.5) & np.isfinite(E2z.reshape(-1)) & (E2z.reshape(-1)<999))
            print(f"[sanity] VIS npz={args.train_npz} valid={int(v.sum())}/{v.size} ({v.mean():.3%})")
            assert v.sum() > 0, "No valid VIS steps in npz �?abort."
        except Exception as e:
            print(f"[sanity] npz check failed: {e}")

    # 动态确定输�?输出维度
    if args.route == "gns":
        # 对于GNSS，从数据集直接获取维�?
        sample_batch = next(iter(train_dl))
        d_in = sample_batch["X"].shape[-1]
        d_out = 3                      # �?各向异性：ENU 三通道 logvar
    elif args.route == "vis":
        d_in = train_ds.X_all.shape[-1]
        # VIS: 检查是否启�?维对角协方差
        if getattr(args, 'vis_2d_diag', False):
            d_out = 2  # VIS: 2维对角协方差 (x,y)
        else:
            d_out = 1  # VIS: 1维聚合误�?
    else:
        d_in = train_ds.X_all.shape[-1] if args.x_mode=="both" else 3
        # IMU: 若使用逐轴 Student-t 且数据为步级标签，则输出3�?
        if getattr(train_ds, 'use_step_labels', False) and args.imu_loss == 'studentt_diag':
            d_out = 3
        else:
            d_out = 1  # 默认 1 �?
    
    model = IMURouteModel(d_in=d_in, d_out=d_out, d_model=args.d_model, n_tcn=args.n_tcn, kernel_size=args.kernel_size,
                          n_layers_tf=args.n_layers_tf, n_heads=args.n_heads, dropout=args.dropout, route=args.route).to(args.device)
    print(f"[model] params={count_params(model):,}  d_in={d_in}  d_out={d_out}")

    # ==== DEBUG: VIS warm-start（用"中位 e2 / 2"更稳�?===
    if args.route == "vis":
        import math
        with torch.no_grad():
            b = next(iter(train_dl))
            e2 = b["E2"]; m = b["MASK"].float()
            if e2.dim()==3 and e2.size(-1)==1: e2 = e2.squeeze(-1)  # -> (B,T)
            if m.dim()==3  and m.size(-1)==1:  m  = m.squeeze(-1)   # -> (B,T)
            if m.size() != e2.size():
                if m.size(1) == 1 and e2.size(1) > 1:
                    print("[fix] warm-start MASK is (B,1); broadcasting to (B,T). Please fix dataset VIS mask.")
                    m = m.expand(-1, e2.size(1))
                else:
                    raise RuntimeError(f"warm-start shape mismatch: E2 {tuple(e2.shape)} vs MASK {tuple(m.shape)}")
            # warm-start for VIS - 安全计算避免 no valid e2
            m = b["MASK"].float()
            e2 = b["E2"]
            if e2.ndim == 3: e2 = e2.squeeze(-1)
            valid = (m > 0.5)
            if valid.any():
                v = torch.nan_to_num(e2, 0.0, 0.0, 0.0)[valid]
                v = v[torch.isfinite(v)]
                v = v[v < 999]  # 与你�?sanity 条件一�?
                if v.numel() > 0:
                    df_vis = DF_BY_ROUTE["vis"]
                    var_hint = float(torch.quantile(v, torch.tensor(0.5, device=v.device))) / df_vis  # 使用统一的自由度
                    var_hint = float(min(max(var_hint, 1e-6), 1e3))
                    model.head.bias.data.fill_(math.log(var_hint))
                    print(f"[warm-start] VIS head bias = log(var) = {math.log(var_hint):+.2f}  "
                          f"(var≈{var_hint:.3e}) from {v.numel()} valid steps")
                else:
                    var0 = torch.tensor(1.0, device=e2.device)  # 兜底
                    model.head.bias.data.fill_(math.log(var0))
                    print("[warm-start] VIS: no finite valid e2 �?fallback to var=1.0")
            else:
                var0 = torch.tensor(1.0, device=e2.device)  # 兜底
                model.head.bias.data.fill_(math.log(var0))
                print("[warm-start] VIS: no valid mask �?fallback to var=1.0")

    # ---- 旧的 warm-start 逻辑已禁用，VIS 使用上面的专用版�?----

    base_lr = args.lr
    weight_decay = 1e-4
    main_params, bias_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "logv_bias" in name:
            bias_params.append(param)
        else:
            main_params.append(param)
    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": base_lr, "weight_decay": weight_decay})
    if bias_params:
        param_groups.append({"params": bias_params, "lr": base_lr * 0.1, "weight_decay": weight_decay})
    opt = AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    
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
    
    best_val = float('inf')   # 兼容原有基于 val_loss 的逻辑
    best_worst = float('inf') # 轴感知用
    epochs_since_improve = 0
    best_path = Path(args.run_dir) / "best.pt"
    last_path = Path(args.run_dir) / "last.pt"
    
    # 轴权重（�?GNSS 生效�?
    lo, hi = map(float, args.axis_clip.split(","))
    axis_w = torch.ones(3, device=args.device)

    # 仅打印一�?VIS 首批调试信息
    dbg_printed_once = False
    
    # VIS 2D diagonal: axis balancer（EMA 权重调整器）
    vis_axis_balancer = None

    def _prepare_vis_step(batch):
        """
        输出:
          e2_step: (B,T-1) 逐步平方误差 (跳过t=0)
          m_step : (B,T-1) 逐步掩码
        """
        # === 使用清洗后的通用键（dataset已处理好�?===
        e2 = batch["E2"].float()
        m  = batch["MASK"].float()

        # 形状统一�?(B,T)
        if e2.dim() == 3 and e2.size(-1) == 1: e2 = e2.squeeze(-1)  # -> (B,T)
        if m.dim()  == 3 and m.size(-1)  == 1: m  = m.squeeze(-1)   # -> (B,T)
        
        # 形状断言与修�?
        if m.size() != e2.size():
            if m.size(1) == 1 and e2.size(1) > 1:
                print("[fix] _prepare_vis_step MASK is (B,1); broadcasting to (B,T). Please fix dataset VIS mask.")
                m = m.expand(-1, e2.size(1))
            else:
                raise RuntimeError(f"_prepare_vis_step shape mismatch: E2 {tuple(e2.shape)} vs MASK {tuple(m.shape)}")
        
        assert e2.dim()==2 and m.dim()==2 and e2.size()==m.size(), f"E2 {e2.shape} vs M {m.shape}"

        # 丢弃 t=0（相邻帧度量�?t=0 本就无定义）
        return e2[:, 1:], m[:, 1:]

    # 体检 2/2：从 DataLoader 抽一批，验证到张量后的有效步（对比原始vs清洗后）
    if args.route == "vis":
        try:
            b = next(iter(train_dl))
            # 优先�?RAW（如果有的话�?
            e2_raw = b.get("E2_VIS_RAW", b["E2"]).detach().cpu().numpy()
            m_raw  = b.get("MASK_VIS_RAW", b["MASK"]).detach().cpu().numpy()
            e2_raw = e2_raw.squeeze()  # (B,T) or (B,T,1)->(B,T)
            m_raw  = m_raw.squeeze()   # (B,T) or (B,T,1)->(B,T)

            valid_raw = (m_raw > 0.5) & np.isfinite(e2_raw) & (e2_raw < 999.0)

            e2_step = b["E2"].detach().cpu().numpy().squeeze()
            m_step  = b["MASK"].detach().cpu().numpy().squeeze()
            valid_clean = (m_step > 0.5)

            print(f"[sanity] loader(raw):  valid_raw.sum={valid_raw.sum()}")
            print(f"[sanity] loader(clean):valid_clean.sum={valid_clean.sum()}")

            if valid_clean.sum() == 0 and valid_raw.sum() > 0:
                print("[sanity] loader check failed: cleaning wiped all valids �?please fix VIS cleaning in dataset.__getitem__")
            assert valid_clean.sum() > 0, "All VIS steps masked out �?check dataset clean path."
        except Exception as e:
            print(f"[sanity] loader check failed: {e}")

    def linear_schedule(start: float, end: float, cur: int, total: int) -> float:
        if total <= 1:
            return end
        cur = max(0, min(cur, total))
        return start + (end - start) * (cur / float(total))

    def run_epoch(loader, training: bool):
        nonlocal dbg_printed_once, vis_axis_balancer
        model.train(training)
        vis_center_weight = getattr(args, "vis_center_weight_current", args.vis_center_weight)
        alpha_aux = getattr(args, "vis_ema_aux_alpha", 0.0)
        ema_tau = getattr(args, "vis_ema_tau", 5)
        total_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, args.device)
            # 仅取通用的输�?X；其余键在各路由分支内按需获取
            x = batch["X"]
            logv = model(x)

            # ==== DEBUG: 看看 logv 贴边 & z^2 的分位数 ====
            if args.route == "vis" and training and (batch_idx == 0):
                logv_raw = logv  # 你的前向输出变量名
                e2 = batch["E2"]; m = batch["MASK"].float()
                if e2.dim()==3 and e2.size(-1)==1: e2 = e2.squeeze(-1)  # -> (B,T)
                if m.dim()==3  and m.size(-1)==1:  m  = m.squeeze(-1)   # -> (B,T)
                if m.size() != e2.size():
                    if m.size(1) == 1 and e2.size(1) > 1:
                        print("[fix] train MASK is (B,1); broadcasting to (B,T). Please fix dataset VIS mask.")
                        m = m.expand(-1, e2.size(1))
                    else:
                        raise RuntimeError(f"shape mismatch: E2 {tuple(e2.shape)} vs MASK {tuple(m.shape)}")
                
                # 处理 2D 对角模式
                if logv_raw.dim() == 3 and logv_raw.size(-1) == 2:
                    # 2D 模式：取平均作为调试信息
                    lv = torch.clamp(logv_raw.mean(dim=-1), min=args.logv_min, max=args.logv_max)  # (B,T)
                    print(f"[dbg] VIS-2D mode: logv shape={logv_raw.shape}")
                elif logv_raw.dim() == 3 and logv_raw.size(-1) == 1:
                    lv = torch.clamp(logv_raw.squeeze(-1), min=args.logv_min, max=args.logv_max)
                else:
                    lv = torch.clamp(logv_raw, min=args.logv_min, max=args.logv_max)
                
                sat_min = (lv <= args.logv_min).float().mean().item()
                sat_max = (lv >= args.logv_max).float().mean().item()
                z2 = torch.nan_to_num(e2, 0.0, 0.0, 0.0) / torch.exp(lv)
                # 保险：自动把 (B,1) 扩到 (B,T)
                m2 = m
                if m2.dim() == 3 and m2.size(-1) == 1:
                    m2 = m2.squeeze(-1)
                if m2.size(1) == 1 and z2.size(1) > 1:
                    m2 = m2.expand(-1, z2.size(1))
                z2v = z2[m2 > 0.5]
                if z2v.numel() > 0:
                    q = torch.quantile(z2v, torch.tensor([0.5, 0.9, 0.99], device=z2v.device))
                    print(f"[dbg] lv.mean={lv.mean().item():.2f}  sat(min/max)={sat_min:.3f}/{sat_max:.3f}  "
                          f"z2@p50/p90/p99={q[0].item():.3g}/{q[1].item():.3g}/{q[2].item():.3g}")
                else:
                    print(f"[dbg] no valid z2 in this batch")
            if args.route == "vis":
                # ---- VIS (step-wise) ----
                m_step = batch["MASK"].float()               # (B,T)
                e2_step = batch["E2"]                        # (B,T) �?(B,T,1)
                if e2_step.ndim == 3:                        # 兼容 (B,T,1)
                    e2_step = e2_step.squeeze(-1)

                # —�?调试：仅首批打印形状与前两个时间步的有效�?—�?
                if training and (not dbg_printed_once):
                    try:
                        print("[dbg] e2_step shape:", tuple(e2_step.shape), " m_step sum:", float(m_step.sum()))
                        m0 = m_step[:, 0].sum().item() if m_step.size(1) > 0 else -1
                        m1 = m_step[:, 1].sum().item() if m_step.size(1) > 1 else -1
                        print(f"[dbg] first two step valids per batch: m[:,0]={m0}, m[:,1]={m1}")
                        print("[dbg] mask>0.5 count=", int((m_step > 0.5).sum().item()))
                        if hasattr(args, 'vis_2d_diag') and args.vis_2d_diag:
                            print(f"[dbg] VIS 2D diagonal mode: logv shape={logv.shape}")
                    finally:
                        dbg_printed_once = True

                # 检查是否使用2D对角协方差
                use_vis_diag = bool(getattr(args, 'vis_2d_diag', False))
                
                if use_vis_diag:
                    # ==== VIS diagonal (2D) branch ====
                    # 模型输出： (B, T, 2) => logσx², logσy²
                    assert logv.dim() == 3 and logv.size(-1) == 2, f"VIS diag expects (B,T,2), got {logv.shape}"
                    lvx, lvy = torch.chunk(logv, 2, dim=-1)   # (B,T,1), (B,T,1)
                    lvx = lvx.squeeze(-1)                      # (B,T)
                    lvy = lvy.squeeze(-1)
                    
                    # 监督：优先用 per-axis；否则回退 E2/2
                    ex2 = batch.get("E2X", None)   # (B,T) or None
                    ey2 = batch.get("E2Y", None)
                    dfx = batch.get("DFX", None)   # (B,T) or None
                    dfy = batch.get("DFY", None)
                    
                    # isotropic 备选
                    e2_iso = e2_step                        # (B,T)
                    df_iso = DF_BY_ROUTE["vis"] * m_step    # fallback
                    
                    if (ex2 is not None) and (ey2 is not None):
                        ex2 = ex2.squeeze(-1) if ex2.dim() == 3 else ex2
                        ey2 = ey2.squeeze(-1) if ey2.dim() == 3 else ey2
                        use_axis_supervise = True
                    else:
                        use_axis_supervise = False
                    
                    # 掩码（每轴独立）：有 df 且 m_step 有效
                    if use_axis_supervise and (dfx is not None) and (dfy is not None):
                        if dfx.dim() == 3:
                            dfx = dfx.squeeze(-1)
                        if dfy.dim() == 3:
                            dfy = dfy.squeeze(-1)
                        mx = (dfx > 0) & m_step.bool()
                        my = (dfy > 0) & m_step.bool()
                        dfx = dfx.to(lvx.dtype)
                        dfy = dfy.to(lvy.dtype)
                    else:
                        # 回退：等分到两轴，掩码一致
                        mx = my = m_step.bool()
                        ex2 = e2_iso * 0.5
                        ey2 = e2_iso * 0.5
                        dfx = dfy = (df_iso * 0.5).to(lvx.dtype)
                    
                    # 数值安全
                    ex2 = torch.clamp(ex2, min=0.0)
                    ey2 = torch.clamp(ey2, min=0.0)
                    
                    # 轴向 Huber delta
                    delta_x = float(getattr(args, "vis_huber_delta_x", None) or getattr(args, "huber_delta", 1.2))
                    delta_y = float(getattr(args, "vis_huber_delta_y", None) or getattr(args, "huber_delta", 1.2))
                    
                    # 有效样本数
                    nx = mx.sum().clamp(min=1)
                    ny = my.sum().clamp(min=1)
                    
                    # NLL（每轴）: 0.5 * df * logσ² + Huber( sqrt( e² / σ² ) )
                    zx2 = ex2 * torch.exp(-lvx)   # (B,T)
                    zy2 = ey2 * torch.exp(-lvy)
                    
                    Lx = 0.5 * dfx * lvx + huber_abs(torch.sqrt(torch.clamp(zx2, min=1e-12)), delta_x)
                    Ly = 0.5 * dfy * lvy + huber_abs(torch.sqrt(torch.clamp(zy2, min=1e-12)), delta_y)
                    
                    Lx_mean = (Lx * mx).sum() / nx
                    Ly_mean = (Ly * my).sum() / ny
                    
                    # 轴向 center（把 E[z²] 拉到 1），各自独立
                    lam_center = float(vis_center_weight)
                    Lcenter = torch.tensor(0.0, device=lvx.device)
                    if lam_center > 0:
                        z2x_mean = (zx2 * mx).sum() / nx
                        z2y_mean = (zy2 * my).sum() / ny
                        Lcenter = lam_center * ((z2x_mean - 1.0).abs() + (z2y_mean - 1.0).abs())
                    
                    # EMA 自适应轴权，避免"弱轴拖强轴"
                    mode = str(getattr(args, "vis_axis_balance", "ema")).lower()
                    if vis_axis_balancer is None:
                        vis_axis_balancer = AxisBalancerEMA(
                            beta=float(getattr(args, "vis_axis_beta", 0.9)),
                            gamma=float(getattr(args, "vis_axis_gamma", 1.0))
                        )
                    
                    if mode == "ema":
                        wx, wy = vis_axis_balancer.weights(Lx_mean, Ly_mean)
                    else:
                        wx, wy = 0.5, 0.5
                    
                    loss = wx * Lx_mean + wy * Ly_mean + Lcenter
                    
                    # 日志
                    if training and (batch_idx % 100) == 0:
                        print(f"[VIS-2D] wx={wx:.3f} wy={wy:.3f} "
                              f"Lx={Lx_mean.item():.4f} Ly={Ly_mean.item():.4f} "
                              f"center={float(Lcenter):.4f}")
                else:
                    # 1D聚合误差模式（原有逻辑�?
                    lv_step = logv.squeeze(-1) if logv.dim() == 3 else logv  # (B,T)
                    
                    # 逐步损失
                    df_vis = DF_BY_ROUTE["vis"]
                    if args.vis_loss == "gauss_huber":
                        loss, _vis_info = nll_gauss_huber_e2_step_with_ema(
                            e2_step, lv_step, m_step,
                            logv_min=args.logv_min, logv_max=args.logv_max,
                            delta=args.vis_huber_delta,
                            df=df_vis,
                            lam_center=vis_center_weight,
                            alpha_aux=alpha_aux,
                            ema_tau=ema_tau
                        )
                    else:  # iso -> plain Gaussian
                        loss, lv_clamped = nll_gaussian_e2_step(
                            e2_step, lv_step, m_step,
                            logv_min=args.logv_min, logv_max=args.logv_max, df=df_vis
                        )
                        if vis_center_weight > 0:
                            center_loss = vis_center_regularization(
                                e2_step, lv_clamped, m_step,
                                df=df_vis, lam_center=vis_center_weight
                            )
                            loss = loss + center_loss
            elif args.route == "gns":
                # GNSS：逐轴各向异性（可选按轴加�?+ Student-t NLL�?
                if args.student_nu > 0:
                    # 使用稳健�?Student-t NLL（对异常值更稳健�?
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
                
                # —�?GNSS 逐轴 vendor 软锚（可选）
                if args.anchor_axes_weight > 0 and ("VENDOR_VAR_AXES" in batch):
                    loss = loss + mse_anchor_axes(logv, batch["VENDOR_VAR_AXES"], batch["MASK_AXES"], 
                                                 lam=args.anchor_axes_weight)
            else:
                # IMU (acc/gyr)
                m = batch["MASK"]
                e2 = batch["E2"]
                use_step_labels = (e2.size(-1) == 3)
                if args.imu_loss == 'gauss_huber':
                    # Gaussian+Huber 损失（推荐）
                    e2sum = e2.sum(dim=-1) if use_step_labels else e2.squeeze(-1)
                    y_anchor = batch.get("Y_anchor", None)
                    loss = nll_gauss_huber_iso3(
                        e2sum=e2sum, logv=logv, mask=m,
                        logv_min=args.logv_min, logv_max=args.logv_max,
                        delta=args.huber_delta, lam_center=args.cal_reg,
                        z2_target=(1.0 if args.z2_center_target=="auto" else float(args.z2_center_target)),
                        y_anchor=y_anchor, anchor_weight=args.anchor_weight,
                        df=DF_BY_ROUTE["acc"]
                    )
                elif args.imu_loss == 'studentt_diag' and use_step_labels:
                    # 逐轴 Student-t（对异常值稳健）
                    if logv.dim() == 2:
                        # 头是1维，扩展�?维以匹配 e2
                        logv = logv.unsqueeze(-1).expand_as(e2)
                    elif logv.size(-1) == 1:
                        logv = logv.expand_as(e2)
                    loss = nll_studentt_diag_axes(e2, logv, m.unsqueeze(-1).expand_as(e2),
                                                  nu=max(args.student_nu, 6.0),
                                                  logv_min=args.logv_min, logv_max=args.logv_max)
                elif args.imu_loss == 'iso':
                    # 传统ISO-3高斯NLL（含回退/锚点�?
                    y_anchor = batch.get("Y_anchor", None)
                    loss = adaptive_nll_loss(logv, e2, m,
                                             use_step_labels=use_step_labels,
                                             y_anchor=y_anchor,
                                             logv_min=args.logv_min,
                                             logv_max=args.logv_max,
                                             route=args.route)
                else:
                    raise ValueError(f"Unknown imu_loss: {args.imu_loss}")
            
            # z²居中正则化（通用于所有路由）
            if args.z2_center > 0:
                # �?NLL 一致地 clamp，再求方�?
                lv = torch.clamp(logv, min=args.logv_min, max=args.logv_max)
                v = torch.exp(lv).clamp_min(1e-12)
                
                # 居中目标：VIS/IMU 仍按聚合 df；GNSS（各向异性）按逐轴 z²
                if args.route == "gns" and logv.shape[-1] == 3:
                    e2_axes = batch["E2_AXES"]
                    m_axes  = batch["MASK_AXES"].float()
                    z2 = (e2_axes / v)                 # (B,T,3), 1D z²
                    m_float = m_axes
                else:
                    # VIS/IMU 路由的z²正则
                    df = DF_BY_ROUTE.get(args.route, 3)
                    
                    e2 = batch["E2"]  # (B,T,D)
                    m_step = m.float().unsqueeze(-1)  # (B,T,1)
                    
                    if e2.size(-1) == 3:  # 步级标签模式
                        # 三轴步级标签：先求和再除以df
                        e2_sum = e2.sum(dim=-1)  # (B,T)
                        v_avg = v.mean(dim=-1) if v.size(-1) > 1 else v.squeeze(-1)  # (B,T)
                        z2 = (e2_sum / v_avg) / df  # (B,T)
                        m_float = m.float()  # (B,T)
                    else:
                        # 回退模式：窗口标签扩�?
                        e2 = e2.squeeze(-1)  # (B,T)
                        v = v.squeeze(-1)  # (B,T)
                        z2 = (e2 / v) / df  # (B,T)
                        m_float = m.float()  # (B,T)
                        
                mean_z2 = (z2 * m_float).sum() / m_float.clamp_min(1.0).sum()
                
                # 目标值：高斯=1；若使用 Student-t �?ν>2，则 target=ν/(ν-2)
                if args.z2_center_target == "auto":
                    if args.student_nu and args.student_nu > 2.0:
                        target = args.student_nu / (args.student_nu - 2.0)
                    else:
                        target = 1.0
                else:
                    target = float(args.z2_center_target)
                
                loss = loss + args.z2_center * (mean_z2 - target).pow(2)
            # 兜底：空批直接跳�?
            if args.route == "vis" and (m_step.sum() < 1):
                if training and (epoch == 1):
                    print("[warn] empty batch (no valid VIS steps) �?skipping")
                continue

            if training:
                # 防NaN保险：检查loss是否为有限�?
                if not torch.isfinite(loss):
                    print(f"[warn] non-finite loss: {float(loss)} �?skip step")
                    opt.zero_grad(set_to_none=True)
                    continue
                
                opt.zero_grad(set_to_none=True)
                loss.backward()
                
                # 检查梯度是否有�?
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and (not torch.isfinite(p.grad).all()):
                        bad_grad = True; break
                if bad_grad:
                    print("[warn] non-finite grad �?zero and skip")
                    opt.zero_grad(set_to_none=True)
                    continue
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        if args.route == "vis":
            total_span = max(1, args.epochs - 1)
            vis_center_weight = linear_schedule(
                args.vis_center_weight_start,
                args.vis_center_weight_end,
                epoch - 1,
                total_span
            )
        else:
            vis_center_weight = getattr(args, "vis_center_weight", 0.0)
        args.vis_center_weight_current = vis_center_weight
        tr_loss = run_epoch(train_dl, True)
        val_loss = run_epoch(val_dl, False)

        # Validation metrics（收集整�?val 集再计算�?
        with torch.no_grad():
            model.eval()
            if args.route == "vis":
                # ===== Validate (VIS only) =====
                e2_all, m_all, lv_all = [], [], []
                for vb in val_dl:
                    vb = to_device(vb, args.device)
                    e2 = vb["E2"]; m = vb["MASK"].float()
                    if e2.dim()==3 and e2.size(-1)==1: e2 = e2.squeeze(-1)   # (B,T)
                    if m.dim()==3  and m.size(-1)==1:  m  = m.squeeze(-1)    # (B,T)
                    if m.size() != e2.size():
                        if m.size(1) == 1 and e2.size(1) > 1:
                            print(f"[VIS][fix] val MASK is (B,1) while E2 is (B,T={e2.size(1)}); broadcasting.")
                            m = m.expand(-1, e2.size(1))
                        else:
                            raise RuntimeError(f"[VIS] val shape mismatch: E2 {tuple(e2.shape)} vs MASK {tuple(m.shape)}")

                    lv = model(vb["X"]).squeeze(-1)    # (B,T)
                    e2_all.append(e2); m_all.append(m); lv_all.append(lv)

                # 沿时间维拼接（你当前 B=1�?
                e2_cat = torch.cat(e2_all, dim=1)
                m_cat  = torch.cat(m_all,  dim=1)
                lv_cat = torch.cat(lv_all, dim=1)

                # 可选：打印一次总长与有效步
                if _vis_dbg_cnt.get("val_summary_printed", 0) == 0:
                    _vis_dbg_cnt["val_summary_printed"] = 1
                    print(f"[VIS][val] concat along T: len={e2_cat.numel()}  valid={(m_cat>0.5).sum().item()}")

                stats = route_metrics_vis(e2_cat, lv_cat, m_cat, args.logv_min, args.logv_max)
            else:
                # 其他路由保持原有逻辑
                val_batch = next(iter(val_dl))
                val_batch = to_device(val_batch, args.device)
                logv = model(val_batch["X"])
                if args.route == "gns":
                    # GNSS：逐轴指标
                    stats = route_metrics_gns_axes(val_batch["E2_AXES"], logv, val_batch["MASK_AXES"],
                                                   args.logv_min, args.logv_max)
                else:
                    stats = route_metrics_imu(val_batch["E2"], logv, val_batch["MASK"], args.logv_min, args.logv_max)

        # === 轴感知统计（GNSS�?==
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

            # 按轴自适应权重（B）：谁偏得远谁更�?
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

        torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)

        if improved or epoch == 1:
            epochs_since_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.early_patience:
                print(f"[early-stop] No improvement for {args.early_patience} epochs. Stopping at epoch {epoch}.")
                break

        # Step the learning rate scheduler (�?optimizer.step() �?scheduler.step())
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, OneCycleLR):
                # OneCycleLR 在每�?batch 后调用，这里已经处理过了
                pass
            else:
                scheduler.step()

    # Final test - iterate over all batches like eval.py
    final_path = best_path if best_path.exists() else last_path
    ckpt = torch.load(final_path, map_location="cpu")
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





