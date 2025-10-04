# -*- coding: utf-8 -*-
"""
Kendall-style heteroscedastic uncertainty loss for 2D visual features.
NLL with optional Huber robustness on errors.
"""
import torch

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

