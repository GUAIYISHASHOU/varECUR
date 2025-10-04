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
from vis.models.uncert_head import UncertHead2D
from vis.losses.kendall import kendall_nll_2d

def ste_clamp(x, lo, hi):
    """Straight-through estimator clamp: forward uses clamped value, backward uses original gradient."""
    y = x.clamp(lo, hi)
    return x + (y - x).detach()

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
    return ap.parse_args()

def main():
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_ds = VISPairs(args.train_npz)
    val_ds = VISPairs(args.val_npz)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                             num_workers=0, drop_last=True)  # Windows: use 0 workers
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                           num_workers=0)  # Windows: use 0 workers
    
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}")
    print(f"[data] train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    
    # Initialize model (ONCE, outside epoch loop)
    model = UncertHead2D(in_ch=2, geom_dim=train_ds.geom.shape[1], out_dim=2).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")
    
    def run_epoch(loader, training=True):
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
            
            # Compute loss (use clamped values)
            loss, info = kendall_nll_2d(
                e2x, e2y, lvx, lvy, loss_mask,
                huber_delta=args.huber, 
                lv_min=args.lv_min,  # These are for internal sanity, already clamped above
                lv_max=args.lv_max
            )
            
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
            train_loader, training=True
        )
        
        # Validation epoch
        val_loss, val_z2x, val_z2y, _, val_steps, _ = run_epoch(
            val_loader, training=False
        )
        
        # Track parameter changes
        with torch.no_grad():
            pnorm_after = torch.nn.utils.parameters_to_vector(
                [p for p in model.parameters() if p.requires_grad]
            ).norm().item()
            param_delta = abs(pnorm_after - pnorm_before)
        
        # Print epoch summary
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train={train_loss:.4f} (z2x={train_z2x:.3f}, z2y={train_z2y:.3f})  "
              f"val={val_loss:.4f} (z2x={val_z2x:.3f}, z2y={val_z2y:.3f})")
        
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }, save_dir / "best_vis_kendall.pt")
            print(f"  → saved best model")
        
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
