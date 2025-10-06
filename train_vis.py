#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for visual uncertainty estimation.
Uses patch pairs + geometry to predict 2D uncertainty (σx², σy²).
"""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from vis.datasets.vis_pairs import VISPairs
from vis.models.uncert_head import UncertHead2D, UncertHead_ResNet_CrossAttention
from vis.losses.kendall import kendall_nll_2d, nll_studentt_2d, calib_reg_l1, ste_clamp

# Import quality-aware preprocessing (optional, degrades gracefully if not available)
try:
    from vis_preproc import quality_weight_from_images, quality_weight_from_meta
    HAS_QUALITY_AWARE = True
except ImportError:
    HAS_QUALITY_AWARE = False
    print("[warning] vis_preproc not found, quality-aware training disabled")

def parse_args():
    ap = argparse.ArgumentParser("Train visual uncertainty head")
    ap.add_argument("--train_npz", required=True, help="Training data NPZ")
    ap.add_argument("--val_npz",   required=True, help="Validation data NPZ")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--huber",  type=float, default=1.0, help="Huber delta for robustness")
    ap.add_argument("--lv_min", type=float, default=-10, help="Min log-variance clamp")
    ap.add_argument("--lv_max", type=float, default=4, help="Max log-variance clamp")
    ap.add_argument("--err_clip_px", type=float, default=20.0, help="Max pixel error for training (mask outliers)")
    ap.add_argument("--skip_loss", type=float, default=50.0, help="Skip batch if loss exceeds this (explosion protection)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", default="runs/vis_uncert", help="Directory to save models")
    
    # New: Loss function options
    ap.add_argument("--loss", type=str, choices=["gauss", "studentt"], default="gauss",
                   help="Loss function: gauss=Kendall+Huber, studentt=Student-t (more robust)")
    ap.add_argument("--nu", type=float, default=3.0, help="Student-t degrees of freedom (nu=3 is robust)")
    ap.add_argument("--calib_reg", type=float, default=0.0, 
                   help="Calibration regularizer coefficient λ for |E[z²]-1| (e.g., 1e-3)")
    
    # Learning rate scheduler
    ap.add_argument("--scheduler", type=str, choices=["none", "cosine", "step", "plateau"], 
                   default="cosine", help="LR scheduler: cosine (smooth decay), step (drop at milestones), plateau (on val loss)")
    ap.add_argument("--lr_min", type=float, default=1e-5, 
                   help="Min LR for cosine scheduler")
    ap.add_argument("--lr_step_epochs", type=str, default="10,15", 
                   help="Epochs to drop LR for step scheduler (comma-separated, e.g., '10,15')")
    ap.add_argument("--lr_gamma", type=float, default=0.1, 
                   help="LR decay factor for step scheduler (e.g., 0.1 = 10x reduction)")
    
    # Early stopping
    ap.add_argument("--early_stop", type=int, default=0, 
                   help="Early stopping patience (epochs). 0=disabled, 5-10 typical")
    
    # Image quality-aware training
    ap.add_argument("--photometric", type=str, default="none",
                   choices=["none","clahe","gamma"],
                   help="图像光度预处理方式")
    ap.add_argument("--clahe_tiles", type=int, default=8)
    ap.add_argument("--clahe_clip", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--ab_match", action="store_true", help="相邻帧仿射亮度匹配")
    
    ap.add_argument("--blur_thr", type=float, default=30.0, help="Laplacian方差阈值（低于→模糊）")
    ap.add_argument("--blur_scale", type=float, default=6.0)
    ap.add_argument("--grad_thr", type=float, default=0.02, help="平均梯度阈值（低于→低纹理）")
    ap.add_argument("--grad_scale", type=float, default=10.0)
    ap.add_argument("--skip_on_blur", action="store_true", help="极端模糊样本直接跳过loss")
    ap.add_argument("--w_lap", type=float, default=1.0)
    ap.add_argument("--w_grad", type=float, default=1.0)
    ap.add_argument("--w_gamma", type=float, default=2.0)
    
    # 没有图像时，基于元特征的权重
    ap.add_argument("--meta_thr", type=float, default=0.2)
    ap.add_argument("--meta_scale", type=float, default=6.0)
    
    # Two-stage training: freeze backbone for warmup, then fine-tune
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze ResNet backbone for head-only warmup")
    ap.add_argument("--load_model", type=str, default="",
                    help="Checkpoint path (.pt/.pth) to initialize weights")
    
    return ap.parse_args()

def main():
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    # Training set: enable photometric augmentation
    # Validation set: disable augmentation (evaluate on clean data)
    train_ds = VISPairs(args.train_npz, augment=True)
    val_ds = VISPairs(args.val_npz, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                             num_workers=0, drop_last=True)  # Windows: use 0 workers
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                           num_workers=0)  # Windows: use 0 workers
    
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}")
    print(f"[data] train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    
    # Initialize model - Use ResNet-18 + Cross-Attention architecture
    print("[model] Using ResNet-18 + Cross-Attention architecture.")
    model = UncertHead_ResNet_CrossAttention(
        in_ch=2, 
        geom_dim=train_ds.geom.shape[1], 
        d_model=128,      # ResNet-18 layer2 输出通道数为 128
        n_heads=4,        # 注意力头数
        out_dim=2,
        pretrained=True   # 关键：使用预训练权重
    ).to(args.device)
    
    # (可选) 载入权重做二阶段微调的起点
    if args.load_model:
        import os
        ckpt = torch.load(args.load_model, map_location=args.device)
        sd = ckpt.get("model", ckpt)  # 兼容纯 state_dict
        model.load_state_dict(sd, strict=False)
        print(f"[init] Loaded weights from {args.load_model}")
    
    # (可选) 冻结主干，仅训练头部（热身）
    if args.freeze_backbone:
        print("[train] Backbone FROZEN (head-only warmup).")
        freeze_keys = ("cnn_stem","cnn_body","backbone","resnet",
                       "conv1","bn1","layer1","layer2","layer3","layer4")
        for n, p in model.named_parameters():
            if any(k in n.lower() for k in freeze_keys):
                p.requires_grad = False
    
    # === Differential Learning Rates (差异化学习率) ===
    # 微调预训练模型的最佳实践：对不同部分使用不同的学习率
    # 1. 将模型参数分为两组：'backbone' 和 'head'
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # cnn_stem 和 cnn_body 属于ResNet主干（预训练）
        if 'cnn_stem' in name or 'cnn_body' in name:
            backbone_params.append(param)
        else:
            # geom_projector, cross_attn, norm1, mlp 属于新添加的头（随机初始化）
            head_params.append(param)
    
    # 2. 为不同的参数组设置不同的学习率
    # 主干：使用较小学习率微调（args.lr / 10）
    # 头部：使用正常学习率从零学习（args.lr）
    lr_backbone = args.lr / 10.0
    
    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': args.lr}
    ]
    
    print(f"[optimizer] Using AdamW with differential learning rates:")
    print(f"  - Backbone (ResNet) LR: {lr_backbone:.2e} ({len(backbone_params)} param tensors)")
    print(f"  - Head (Attn+MLP) LR:   {args.lr:.2e} ({len(head_params)} param tensors)")
    
    # 3. 使用参数组初始化优化器
    # 注意：这里不再传入 lr 参数，因为每个组都有自己的 lr
    # weight_decay提高到5e-4，加强正则化防止过拟合
    optimizer = torch.optim.AdamW(param_groups, weight_decay=5e-4)
    print(f"  - Weight decay: 5e-4 (increased for better regularization)")
    
    # Initialize learning rate scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
        print(f"[scheduler] CosineAnnealingLR (lr: {args.lr} → {args.lr_min})")
    elif args.scheduler == "step":
        milestones = [int(x) for x in args.lr_step_epochs.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_gamma
        )
        print(f"[scheduler] MultiStepLR (milestones={milestones}, gamma={args.lr_gamma})")
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        print(f"[scheduler] ReduceLROnPlateau (factor=0.5, patience=3)")
    else:
        print(f"[scheduler] None (fixed LR={args.lr})")
    
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")
    
    # Print training configuration
    print(f"[config] loss={args.loss}, nu={args.nu if args.loss=='studentt' else 'N/A'}, "
          f"huber={args.huber if args.loss=='gauss' else 'N/A'}, calib_reg={args.calib_reg}")
    print(f"[config] early_stop={'disabled' if args.early_stop == 0 else f'{args.early_stop} epochs'}")
    
    # Quality-aware training info
    if args.photometric != "none":
        print(f"[config] quality_aware=enabled (using geom gradient features)")
        print(f"[config]   grad_thr={args.grad_thr}, grad_scale={args.grad_scale}, "
              f"skip_on_blur={args.skip_on_blur}, w_gamma={args.w_gamma}")
    else:
        print(f"[config] quality_aware=disabled (set --photometric=clahe to enable)")
    
    def run_epoch(loader, training=True, epoch=1):
        """Run one epoch with diagnostic info."""
        model.train(training)
        
        total_loss = 0.0
        total_samples = 0
        z2x_sum, z2y_sum = 0.0, 0.0
        total_grad = 0.0
        steps = 0
        
        # Saturation tracking
        lvx_raw_list = []
        lvy_raw_list = []
        
        for batch in loader:  # ← 每个epoch都重新迭代，不要在外面缓存
            steps += 1
            
            patch2 = batch["patch2"].to(args.device)
            geom = batch["geom"].to(args.device)
            e2x = batch["e2x"].to(args.device)
            e2y = batch["e2y"].to(args.device)
            mask = batch["mask"].to(args.device)
            
            # Forward pass
            logv_raw = model(patch2, geom)  # [B,2] raw output
            
            # Apply STE clamp (preserves gradients)
            lvx = ste_clamp(logv_raw[:, 0], args.lv_min, args.lv_max)
            lvy = ste_clamp(logv_raw[:, 1], args.lv_min, args.lv_max)
            
            # Track raw values for saturation check
            if training:
                lvx_raw_list.append(logv_raw[:, 0].detach())
                lvy_raw_list.append(logv_raw[:, 1].detach())
            
            # Training-time outlier masking (don't clip validation set)
            if training:
                err_radius = torch.sqrt(e2x + e2y)
                mask_train = (err_radius < args.err_clip_px) & (mask > 0.5)
                loss_mask = mask_train.float()
            else:
                loss_mask = mask
            
            # Quality-aware weighting (if enabled)
            if training and args.photometric != "none":
                # Use gradient features from geom (more reliable than 32x32 patches)
                # geom format: [u0, v0, u2, v2, g0, g2, c0, c2, flow, baseline, parallax]
                if geom.shape[1] >= 6:  # Has gradient features
                    g0 = geom[:, 4]  # gradient magnitude at frame 0
                    g2 = geom[:, 5]  # gradient magnitude at frame 2
                    
                    # Normalize gradients to [0,1] weights
                    g_min = torch.tensor(args.grad_thr, device=geom.device)
                    g_max = torch.tensor(args.grad_thr * args.grad_scale, device=geom.device)
                    
                    def normalize_quality(g, g_min, g_max):
                        return torch.clamp((g - g_min) / (g_max - g_min + 1e-9), 0.0, 1.0)
                    
                    w0 = normalize_quality(g0, g_min, g_max)
                    w2 = normalize_quality(g2, g_min, g_max)
                    weights = torch.min(w0, w2)  # Take min (more conservative)
                    
                    # Apply gamma to suppress low-quality samples more
                    weights = weights ** args.w_gamma
                    
                    # Skip extremely low-gradient samples
                    if args.skip_on_blur:
                        keep = (weights > 0.01)  # Skip if weight < 1%
                        weights = weights * keep.float()
                    
                    # Apply to loss_mask
                    loss_mask = loss_mask * weights.unsqueeze(-1) if loss_mask.ndim > 1 else loss_mask * weights
            
            # Compute main loss (use clamped values)
            if args.loss == "studentt":
                loss_main, info = nll_studentt_2d(
                    e2x, e2y, lvx, lvy, loss_mask,
                    nu=args.nu,
                    lv_min=args.lv_min,
                    lv_max=args.lv_max
                )
            else:  # gauss (default Kendall + Huber)
                loss_main, info = kendall_nll_2d(
                    e2x, e2y, lvx, lvy, loss_mask,
                    huber_delta=args.huber, 
                    lv_min=args.lv_min,
                    lv_max=args.lv_max
                )
            
            loss = loss_main
            
            # Add calibration regularizer (only in later training epochs)
            if training and args.calib_reg > 0.0 and epoch > (args.epochs // 2):
                reg = calib_reg_l1(e2x, e2y, lvx, lvy, loss_mask)
                loss = loss + args.calib_reg * reg
            
            # Explosion protection: skip toxic batches
            if not torch.isfinite(loss) or loss.item() > args.skip_loss:
                if training:
                    print(f"  [skip batch] loss={loss.item():.2f} (non-finite or >{args.skip_loss})")
                continue
            
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping (stricter for stability)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                total_grad += float(grad_norm)
                optimizer.step()
            
            total_loss += float(loss.item()) * patch2.size(0)
            total_samples += patch2.size(0)
            z2x_sum += info.get("z2x", 0.0) * patch2.size(0)
            z2y_sum += info.get("z2y", 0.0) * patch2.size(0)
        
        # Assert that we actually processed data
        assert steps > 0, f"{'train' if training else 'val'} loader produced 0 batches!"
        
        avg_loss = total_loss / max(total_samples, 1)
        avg_z2x = z2x_sum / max(total_samples, 1)
        avg_z2y = z2y_sum / max(total_samples, 1)
        avg_grad = total_grad / max(steps, 1) if training else 0.0
        
        # Saturation diagnostics (training only)
        sat_info = {}
        if training and lvx_raw_list:
            lvx_all = torch.cat(lvx_raw_list)
            lvy_all = torch.cat(lvy_raw_list)
            sat_info['sat_hi_x'] = float((lvx_all > args.lv_max).float().mean())
            sat_info['sat_lo_x'] = float((lvx_all < args.lv_min).float().mean())
            sat_info['sat_hi_y'] = float((lvy_all > args.lv_max).float().mean())
            sat_info['sat_lo_y'] = float((lvy_all < args.lv_min).float().mean())
        
        return avg_loss, avg_z2x, avg_z2y, avg_grad, steps, sat_info
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0  # For early stopping
    
    print("\n[training] Starting...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs+1):
        # Track parameter changes
        with torch.no_grad():
            pnorm_before = torch.nn.utils.parameters_to_vector(
                [p for p in model.parameters() if p.requires_grad]
            ).norm().item()
        
        # Train epoch
        train_loss, train_z2x, train_z2y, train_grad, train_steps, sat_info = run_epoch(
            train_loader, training=True, epoch=epoch
        )
        
        # Validation epoch
        val_loss, val_z2x, val_z2y, _, val_steps, _ = run_epoch(
            val_loader, training=False, epoch=epoch
        )
        
        # Track parameter changes
        with torch.no_grad():
            pnorm_after = torch.nn.utils.parameters_to_vector(
                [p for p in model.parameters() if p.requires_grad]
            ).norm().item()
            param_delta = abs(pnorm_after - pnorm_before)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train={train_loss:.4f} (z2x={train_z2x:.3f}, z2y={train_z2y:.3f})  "
              f"val={val_loss:.4f} (z2x={val_z2x:.3f}, z2y={val_z2y:.3f})  "
              f"lr={current_lr:.2e}")
        
        # Diagnostic info
        print(f"  steps: train={train_steps}, val={val_steps}  "
              f"grad_norm={train_grad:.3f}  |Δparam|={param_delta:.6f}")
        
        # Saturation info (warn if too high)
        if sat_info:
            print(f"  saturation: "
                  f"hi_x={sat_info['sat_hi_x']:.3f}, lo_x={sat_info['sat_lo_x']:.3f}, "
                  f"hi_y={sat_info['sat_hi_y']:.3f}, lo_y={sat_info['sat_lo_y']:.3f}")
            if sat_info['sat_hi_x'] > 0.5 or sat_info['sat_hi_y'] > 0.5:
                print(f"  ⚠ Warning: >50% samples saturated at lv_max={args.lv_max}, consider increasing")
        
        # Save best model & early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0  # Reset patience
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }, save_dir / "best_vis_kendall.pt")
            print(f"  → saved best model (val_loss improved)")
        else:
            patience_counter += 1
            if args.early_stop > 0:
                print(f"  → no improvement for {patience_counter}/{args.early_stop} epochs")
        
        # Learning rate scheduler step
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)  # Plateau needs metric
            else:
                scheduler.step()  # Cosine/Step use epoch count
        
        # Early stopping check
        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n[early_stop] No improvement for {args.early_stop} epochs. Stopping training.")
            print(f"[early_stop] Best val loss: {best_val_loss:.4f} @ epoch {best_epoch}")
            break
        
        print()
    
    # Save final model
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': vars(args)
    }, save_dir / "last_vis_kendall.pt")
    
    print("=" * 60)
    print(f"[done] Best val loss: {best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"[done] Models saved to: {save_dir}")

if __name__ == "__main__":
    main()
