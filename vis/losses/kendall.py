# -*- coding: utf-8 -*-
"""
Kendall-style heteroscedastic uncertainty loss for 2D visual features.
NLL with optional Huber robustness on errors.
Includes Student-t NLL and calibration regularizer.
"""
import torch

def ste_clamp(x, lo, hi):
    """Straight-Through Estimator clamp: forward限幅, backward保梯度"""
    y = x.clamp(lo, hi)
    return x + (y - x).detach()

@torch.no_grad()
def z2_mean_from_e2(e2x, e2y, lvx, lvy, mask=None):
    """Compute mean normalized squared error z²."""
    vx = torch.exp(lvx)
    vy = torch.exp(lvy)
    z2 = e2x / vx + e2y / vy
    if mask is not None:
        m = (mask > 0).float()
        return (z2 * m).sum() / m.sum().clamp_min(1.0)
    return z2.mean()

def kendall_nll_2d(e2x, e2y, lvx, lvy, mask=None, huber_delta=0.0, lv_min=-10.0, lv_max=4.0):
    """
    2D heteroscedastic negative log-likelihood loss.
    
    L = 0.5 * (e_x² / σx² + log σx²) + 0.5 * (e_y² / σy² + log σy²)
    
    Args:
        e2x, e2y: [N] squared errors per axis
        lvx, lvy: [N] predicted log-variances per axis (already clamped with STE)
        mask: [N] validity mask (optional)
        huber_delta: Huber threshold on errors (0 = no Huber)
        lv_min, lv_max: For reference only (caller should apply STE clamp)
    
    Returns:
        loss: scalar loss
        info: dict with diagnostic info (z2x, z2y)
    """
    def huber_e2(e2, d):
        if d <= 0: return e2
        e  = torch.sqrt(torch.clamp(e2, min=0.0) + 1e-12)
        return torch.where(e <= d, e2, 2.0*d*e - d*d)

    # Note: lvx, lvy are already clamped by caller using STE
    # No additional clamp here to preserve gradients
    e2x = huber_e2(e2x, huber_delta)
    e2y = huber_e2(e2y, huber_delta)

    L = 0.5*(torch.exp(-lvx)*e2x + lvx) + 0.5*(torch.exp(-lvy)*e2y + lvy)
    
    if mask is None: 
        return L.mean(), {}
    
    m = (mask>0).float()
    loss = (L*m).sum() / m.sum().clamp_min(1.0)
    
    with torch.no_grad():
        z2x = float(((e2x*torch.exp(-lvx))*m).sum() / m.sum().clamp_min(1.0))
        z2y = float(((e2y*torch.exp(-lvy))*m).sum() / m.sum().clamp_min(1.0))
    
    return loss, {"z2x": z2x, "z2y": z2y}

def nll_studentt_2d(e2x, e2y, lvx, lvy, mask=None, nu=3.0, lv_min=-10.0, lv_max=8.0):
    """
    二维对角 Student-t Negative Log-Likelihood.
    
    More robust to outliers than Gaussian. The Student-t distribution has heavier tails,
    making it less sensitive to extreme errors.
    
    L = 0.5 * (ν+2) * log(1 + z²/ν) + 0.5 * (log σx² + log σy²)
    where z² = e_x²/σx² + e_y²/σy²
    
    Args:
        e2x, e2y: [N] squared errors per axis
        lvx, lvy: [N] predicted log-variances (should be pre-clamped with STE)
        mask: [N] validity mask (optional)
        nu: degrees of freedom (3.0 is robust, lower = heavier tails)
        lv_min, lv_max: log-variance bounds (for reference)
    
    Returns:
        loss: scalar loss
        info: dict with diagnostic z²_x, z²_y
    """
    # Apply STE clamp to ensure numerical stability
    lvx = ste_clamp(lvx, lv_min, lv_max)
    lvy = ste_clamp(lvy, lv_min, lv_max)
    
    vx = torch.exp(lvx)
    vy = torch.exp(lvy)
    z = e2x / vx + e2y / vy  # Combined z²
    
    # Student-t NLL (constant terms omitted as they don't affect gradients)
    L = 0.5 * (nu + 2.0) * torch.log1p(z / nu) + 0.5 * (lvx + lvy)
    
    if mask is None:
        return L.mean(), {}
    
    m = (mask > 0).float()
    loss = (L * m).sum() / m.sum().clamp_min(1.0)
    
    # Diagnostics
    with torch.no_grad():
        z2x = float(((e2x / vx) * m).sum() / m.sum().clamp_min(1.0))
        z2y = float(((e2y / vy) * m).sum() / m.sum().clamp_min(1.0))
    
    return loss, {"z2x": z2x, "z2y": z2y}

def calib_reg_l1(e2x, e2y, lvx, lvy, mask=None):
    """
    Calibration regularizer: |E[z²] - 1|
    
    Encourages the model to be well-calibrated by penalizing deviations of 
    mean normalized squared error from 1.0 (the expected value for χ²(df=2)/2).
    
    Use a small coefficient (e.g., 1e-3) and only in later training epochs.
    
    Args:
        e2x, e2y: [N] squared errors
        lvx, lvy: [N] log-variances
        mask: [N] validity mask
    
    Returns:
        reg: scalar regularization term
    """
    vx = torch.exp(lvx)
    vy = torch.exp(lvy)
    z2 = (e2x / vx + e2y / vy) / 2.0  # Average of two axes
    
    if mask is not None:
        m = (mask > 0).float()
        mu = (z2 * m).sum() / m.sum().clamp_min(1.0)
    else:
        mu = z2.mean()
    
    return (mu - 1.0).abs()

