# -*- coding: utf-8 -*-
"""
Dataset for visual uncertainty estimation from patch pairs.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class VISPairs(Dataset):
    """
    Load pre-extracted visual patch pairs with reprojection errors.
    
    Each sample contains:
    - I0, I1: paired patches at time t and t+Δ
    - geom: geometric context features (normalized coordinates)
    - e2x, e2y: ground-truth squared reprojection errors
    - mask: validity mask
    """
    def __init__(self, npz_path: str):
        D = np.load(npz_path, allow_pickle=True)
        self.I0   = D["I0"].astype(np.float32) / 255.0  # [N,1,H,W]
        self.I1   = D["I1"].astype(np.float32) / 255.0
        self.geom = D["geom"].astype(np.float32)        # [N,F]
        self.e2x  = D["e2x"].astype(np.float32)         # [N]
        self.e2y  = D["e2y"].astype(np.float32)
        self.mask = D["mask"].astype(np.float32)        # [N]

    def __len__(self): 
        return self.I0.shape[0]
    
    def __getitem__(self, i):
        # 把两帧patch沿通道拼一起 -> [2,H,W]
        patch2 = np.concatenate([self.I0[i], self.I1[i]], axis=0)
        return {
            "patch2": torch.from_numpy(patch2),    # [2,H,W]
            "geom":   torch.from_numpy(self.geom[i]),  # [F]
            "e2x":    torch.tensor(self.e2x[i]),
            "e2y":    torch.tensor(self.e2y[i]),
            "mask":   torch.tensor(self.mask[i]),
        }

