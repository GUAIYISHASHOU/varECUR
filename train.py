# -*- coding: utf-8 -*-
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


def parse_args():
    # 先只解析 --config �?--route（不加载其它参数�?
    pre_cfg = argparse.ArgumentParser(add_help=False)
    pre_cfg.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    pre_route = argparse.ArgumentParser(add_help=False)
    pre_route.add_argument("--route", choices=["acc","gyr"], default=None)

    args_pre_cfg, _ = pre_cfg.parse_known_args()
    args_pre_route, _ = pre_route.parse_known_args()

    cfg = load_config_file(args_pre_cfg.config)

    route_hint = args_pre_route.route or "acc"
    tr = cfg.get("train", {})
    md = cfg.get("model", {})
    rt = cfg.get("runtime", {})

    # 真正的参数解析器
    ap = argparse.ArgumentParser("Train single-route IMU variance model", parents=[pre_cfg])
    ap.add_argument("--route", choices=["acc","gyr"], default=tr.get("route", route_hint),
                    help="Which route to train (IMU only)")
    ap.add_argument("--train_npz", required=(tr.get("train_npz") is None), default=tr.get("train_npz"))
    ap.add_argument("--val_npz", required=(tr.get("val_npz") is None), default=tr.get("val_npz"))
    ap.add_argument("--test_npz", required=(tr.get("test_npz") is None), default=tr.get("test_npz"))
    ap.add_argument("--x_mode", choices=["both","route_only","imu"], default=tr.get("x_mode","both"))
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
    # 混合早停模式（结合损失和校准质量）
    ap.add_argument("--early_mode", type=str, default=tr.get("early_mode", "loss"), 
                    choices=["loss", "hybrid"], help="Early stopping mode: 'loss' or 'hybrid' (loss + calibration)")
    ap.add_argument("--lam_calib", type=float, default=tr.get("lam_calib", 0.2),
                    help="Weight for calibration term in hybrid early stopping")
    ap.add_argument("--student_nu", type=float, default=tr.get("student_nu", 0.0),
                    help="Student-t 自由度参数（0=使用高斯NLL�?0=使用t-NLL，推�?.0")
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
    
    # Huber 损失的超参数
    ap.add_argument('--huber_delta', type=float, default=tr.get('huber_delta', 1.5))
    ap.add_argument('--cal_reg', type=float, default=tr.get('cal_reg', 5e-2), help="E[z^2] center regularizer weight")
    # 你已有的 anchor_weight 继续沿用（作为窗口尺度锚的权重）

    args = ap.parse_args()
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
    train_ds, train_dl = build_loader(args.train_npz, route=args.route, x_mode=args.x_mode, 
                                      batch_size=args.batch_size, shuffle=True, 
                                      num_workers=args.num_workers, generator=g, worker_init_fn=worker_init_fn)
    val_ds, val_dl = build_loader(args.val_npz, route=args.route, x_mode=args.x_mode,
                                  batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_ds, test_dl = build_loader(args.test_npz, route=args.route, x_mode=args.x_mode,
                                    batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 动态确定输入/输出维度
    d_in = train_ds.X_all.shape[-1] if args.x_mode=="both" else 3
    # IMU: 若使用逐轴 Student-t 且数据为步级标签，则输出3轴
    if getattr(train_ds, 'use_step_labels', False) and args.imu_loss == 'studentt_diag':
        d_out = 3
    else:
        d_out = 1  # 默认 1 轴
    
    model = IMURouteModel(d_in=d_in, d_out=d_out, d_model=args.d_model, n_tcn=args.n_tcn, kernel_size=args.kernel_size,
                          n_layers_tf=args.n_layers_tf, n_heads=args.n_heads, dropout=args.dropout, route=args.route).to(args.device)
    print(f"[model] params={count_params(model):,}  d_in={d_in}  d_out={d_out}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler setup
    if args.lr_sched == 'cosine':
        # Cosine annealing with warmup
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    elif args.lr_sched == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.plateau_factor, 
                                     patience=args.plateau_patience, min_lr=args.min_lr, verbose=True)
    elif args.lr_sched == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, 
                              steps_per_epoch=len(train_dl), pct_start=0.3)
    else:
        scheduler = None

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    dbg_printed_once = False

    # Import loss functions
    from losses import nll_iso3_e2, nll_gauss_huber_iso3, nll_studentt_diag_axes
    
    def run_epoch(loader, training: bool):
        nonlocal dbg_printed_once
        model.train(training)
        total_loss = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, args.device)
            x = batch["X"]
            logv = model(x)
            
            # IMU loss computation
            e2 = batch["E2"]
            mask = batch["MASK"]
            
            if args.imu_loss == 'iso':
                # Simple isotropic Gaussian
                e2sum = e2.sum(dim=-1) if e2.dim()==3 and e2.size(-1)==3 else e2.squeeze(-1)
                logv_avg = logv.mean(dim=-1) if logv.size(-1)==3 else logv.squeeze(-1)
                loss = nll_iso3_e2(e2sum, logv_avg, mask, args.logv_min, args.logv_max)
            elif args.imu_loss == 'gauss_huber':
                # Gaussian + Huber robustness
                e2sum = e2.sum(dim=-1) if e2.dim()==3 and e2.size(-1)==3 else e2.squeeze(-1)
                logv_avg = logv.mean(dim=-1) if logv.size(-1)==3 else logv.squeeze(-1)
                loss = nll_gauss_huber_iso3(
                    e2sum, logv_avg, mask, 
                    logv_min=args.logv_min, logv_max=args.logv_max,
                    delta=args.huber_delta, lam_center=args.cal_reg,
                    y_anchor=batch.get("Y"), anchor_weight=args.anchor_weight
                )
            elif args.imu_loss == 'studentt_diag':
                # Per-axis Student-t
                if e2.dim()==3 and e2.size(-1)==3:
                    e2_axes = e2  # (B,T,3)
                    logv_axes = logv if logv.size(-1)==3 else logv.expand(-1,-1,3)
                    mask_axes = mask.unsqueeze(-1).expand_as(e2_axes)
                    loss = nll_studentt_diag_axes(e2_axes, logv_axes, mask_axes, 
                                                 nu=args.student_nu if args.student_nu>0 else 3.0,
                                                 logv_min=args.logv_min, logv_max=args.logv_max)
                else:
                    # Fallback to iso
                    e2sum = e2.squeeze(-1)
                    logv_avg = logv.squeeze(-1)
                    loss = nll_iso3_e2(e2sum, logv_avg, mask, args.logv_min, args.logv_max)
            else:
                raise ValueError(f"Unknown imu_loss: {args.imu_loss}")
            
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if args.lr_sched == 'onecycle':
                    scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        
        train_loss = run_epoch(train_dl, training=True)
        val_loss = run_epoch(val_dl, training=False)
        
        # Learning rate scheduler step (except onecycle which steps per batch)
        if scheduler is not None and args.lr_sched != 'onecycle':
            if args.lr_sched == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"[{epoch:3d}/{args.epochs}] train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.2e} t={elapsed:.1f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, 
                      Path(args.run_dir)/"best.pt")
            print(f"  → new best @ epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_patience:
                print(f"[early stop] no improvement for {args.early_patience} epochs")
                break
    
    # Save final model
    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, 
              Path(args.run_dir)/"last.pt")
    
    # Test evaluation
    print(f"\n[test] loading best model from epoch {best_epoch}...")
    ckpt = torch.load(Path(args.run_dir)/"best.pt", map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test_loss = run_epoch(test_dl, training=False)
    print(f"[test] loss={test_loss:.4f}")
    
    # Compute detailed test metrics
    from metrics import route_metrics_imu
    agg = {}
    n = 0
    with torch.no_grad():
        model.eval()
        for batch in test_dl:
            batch = to_device(batch, args.device)
            logv = model(batch["X"])
            stats = route_metrics_imu(batch["E2"], logv, batch["MASK"], 
                                     logv_min=args.logv_min, logv_max=args.logv_max,
                                     yvar=batch.get("Y"))
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
    tst = {k: v/n for k, v in agg.items()}
    
    with open(Path(args.run_dir)/"final_test_metrics.json","w",encoding="utf-8") as f:
        json.dump(tst, f, ensure_ascii=False, indent=2)
    print("[test]", tst)

if __name__ == "__main__":
    main()





