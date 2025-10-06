# -*- coding: utf-8 -*-
"""
Dataset for visual uncertainty estimation from patch pairs.
Supports photometric augmentation (brightness, contrast, gamma, noise, blur)
WITHOUT changing geometric relationships.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def _rand_gamma(img, g):
    """
    Apply gamma correction to image.
    Args:
        img: float32 image in [0,1]
        g: gamma value (>1 darker, <1 brighter)
    Returns:
        gamma-corrected image
    """
    return np.clip(np.power(img, g), 0.0, 1.0).astype(np.float32)


class VISPairs(Dataset):
    """
    Load pre-extracted visual patch pairs with reprojection errors.
    
    Each sample contains:
    - I0, I1: paired patches at time t and t+Δ
    - geom: geometric context features (normalized coordinates)
    - e2x, e2y: ground-truth squared reprojection errors
    - mask: validity mask
    
    Args:
        npz_path: Path to NPZ file
        augment: If True, apply photometric augmentation (for training only)
    """
    def __init__(self, npz_path: str, augment: bool = False):
        D = np.load(npz_path, allow_pickle=True)
        self.I0   = D["I0"].astype(np.float32) / 255.0  # [N,1,H,W]
        self.I1   = D["I1"].astype(np.float32) / 255.0
        self.geom = D["geom"].astype(np.float32)        # [N,F]
        self.e2x  = D["e2x"].astype(np.float32)         # [N]
        self.e2y  = D["e2y"].astype(np.float32)
        self.mask = D["mask"].astype(np.float32)        # [N]
        self.augment = augment
        
        if self.augment:
            print("[data] Training-time photometric augmentation ENABLED.")
            print("[data]   - Contrast/Brightness: mild variations")
            print("[data]   - Gamma: 0.9-1.1 range")
            print("[data]   - Noise: 30% probability, std=0.01")
            print("[data]   - Blur: 20% probability, kernel=3")
            print("[data]   - Geometric transforms: DISABLED (preserve geom consistency)")

    def __len__(self): 
        return self.I0.shape[0]
    
    def __getitem__(self, i):
        # Extract patches (H, W)
        patch0 = self.I0[i, 0].copy()
        patch1 = self.I1[i, 0].copy()
        
        if self.augment:
            # ---- Contrast & Brightness (mild range) ----
            alpha = np.random.uniform(0.9, 1.1)  # contrast factor
            beta = np.random.uniform(-0.05, 0.05)  # brightness offset
            patch0 = np.clip(alpha * patch0 + beta, 0, 1)
            patch1 = np.clip(alpha * patch1 + beta, 0, 1)
            
            # ---- Gamma correction (mild range) ----
            gamma = np.random.uniform(0.9, 1.1)
            patch0 = _rand_gamma(patch0, gamma)
            patch1 = _rand_gamma(patch1, gamma)
            
            # ---- Light Gaussian noise (30% probability) ----
            if np.random.rand() < 0.3:
                nstd = np.random.uniform(0.0, 0.01)
                noise0 = np.random.normal(0.0, nstd, size=patch0.shape).astype(np.float32)
                noise1 = np.random.normal(0.0, nstd, size=patch1.shape).astype(np.float32)
                patch0 = np.clip(patch0 + noise0, 0, 1)
                patch1 = np.clip(patch1 + noise1, 0, 1)
            
            # ---- Light blur (20% probability, small kernel) ----
            if np.random.rand() < 0.2:
                k = 3  # small kernel size
                patch0 = cv2.GaussianBlur(patch0, (k, k), 0).astype(np.float32)
                patch1 = cv2.GaussianBlur(patch1, (k, k), 0).astype(np.float32)
            
            # ⚠️ NO flip/rotation/crop - these would break geom coordinate consistency
        
        # Stack to [2, H, W]
        patch2 = np.stack([patch0, patch1], axis=0).astype(np.float32)
        
        return {
            "patch2": torch.from_numpy(patch2),         # [2,H,W]
            "geom":   torch.from_numpy(self.geom[i]),   # [F]
            "e2x":    torch.tensor(self.e2x[i]),
            "e2y":    torch.tensor(self.e2y[i]),
            "mask":   torch.tensor(self.mask[i]),
        }

