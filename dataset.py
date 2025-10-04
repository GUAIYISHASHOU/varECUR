from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict

# Debug limit helper
DBG_MAX = 8

def _dbg_limit(counter: dict, key: str, maxn=DBG_MAX) -> bool:
    """Return True if can still print; False if exceeded limit"""
    n = counter.get(key, 0)
    if n < maxn:
        counter[key] = n + 1
        return True
    return False

def _get(arrs: dict, keys, default=None):
    for k in keys:
        if k in arrs: 
            return arrs[k]
    return default

def _ensure_bool_mask(m):
    m = m.astype(np.float32)
    m = (m > 0.5).astype(np.float32)
    return m

class IMURouteDataset(Dataset):
    """Dataset for IMU routes only (acc, gyr). VIS route removed."""
    
    def __init__(self, npz_path: str | Path, route: str = "acc", x_mode: str = "both"):
        self.npz_path = str(npz_path)
        self.route = route
        self.x_mode = x_mode
        
        # Only IMU routes supported
        assert route in ("acc", "gyr"), f"Unsupported route: {route}. Only 'acc' and 'gyr' supported."
        
        valid_modes = {"both", "route_only", "imu"}
        if x_mode not in valid_modes:
            raise ValueError(f"Unsupported x_mode='{x_mode}', must be one of {sorted(valid_modes)}")
        
        data = np.load(self.npz_path, allow_pickle=True)
        
        def _pick(keys):
            for k in keys:
                if k in data.files:
                    return data[k]
            return None
        
        # Load X (features)
        if self.route == "acc":
            X = _pick(["X_IMU_ACC", "X_acc", "X"])
            TS = _pick(["TS_IMU"])
        elif self.route == "gyr":
            X = _pick(["X_IMU_GYR", "X_gyr", "X"])
            TS = _pick(["TS_IMU"])
        else:
            raise ValueError(f"Unknown route {self.route}")
        
        if X is None:
            raise ValueError(f"{self.npz_path}: no X found for route={self.route}")
        
        X = X.astype(np.float32)
        if X.ndim == 2:
            X = X[None, :, :]  # (T,D) -> (1,T,D)
        elif X.ndim == 1:
            X = X[None, :, None]  # (T,) -> (1,T,1)
        
        self.N, self.T, self.D = X.shape
        
        # Load labels (step-level or window-level)
        if self.route == "acc" and "E2_IMU_ACC" in data.files:
            E2_step = data["E2_IMU_ACC"].astype(np.float32)  # [N,T,3]
            self.use_step_labels = True
            self.d_out = 3
            Y_anchor = None
            DF_anchor = None
            E2 = E2_step
        elif self.route == "gyr" and "E2_IMU_GYR" in data.files:
            E2_step = data["E2_IMU_GYR"].astype(np.float32)
            self.use_step_labels = True
            self.d_out = 3
            Y_anchor = None
            DF_anchor = None
            E2 = E2_step
        else:
            # Fallback to window-level labels
            self.use_step_labels = False
            if self.route == "acc":
                Y_anchor = _pick(["Y_IMU_ACC", "E2_acc", "E2", "Y"])
            elif self.route == "gyr":
                Y_anchor = _pick(["Y_IMU_GYR", "E2_gyr", "E2", "Y"])
            
            if Y_anchor is None:
                raise ValueError(f"{self.npz_path}: no labels found for route={self.route}")
            
            Y_anchor = Y_anchor.astype(np.float32)
            DF_anchor = None
            self.df_all = DF_anchor
            self.d_out = 1
            
            # Convert window labels to per-timestep
            if Y_anchor.ndim == 2 and Y_anchor.shape[1] == 3:
                e2_scalar = np.linalg.norm(Y_anchor, axis=1).astype(np.float32)
            elif Y_anchor.ndim == 2 and Y_anchor.shape[1] == 1:
                e2_scalar = Y_anchor[:,0].astype(np.float32)
            else:
                raise ValueError(f"Unexpected Y shape {Y_anchor.shape} for IMU")
            
            E2 = np.repeat(e2_scalar[:, None, None], self.T, axis=1)  # [N,T,1]
        
        # Load mask
        M = _pick(["MASK_IMU", "MASK", "mask"])
        if M is None:
            M = np.ones((self.N, self.T), dtype=np.float32)
        else:
            M = M.astype(np.float32)
            if M.ndim == 1:
                M = M[None, :]
            M = _ensure_bool_mask(M)
        
        if M.ndim == 2:
            M = M[:, :, None]
        
        self.X_all = X
        self.E2_all = E2
        self.M_all = M
        self.DF_all = DF_anchor if 'DF_anchor' in locals() else None
        self.TS = TS if TS is not None else None
        self.Y_anchor = Y_anchor if 'Y_anchor' in locals() else None
        
        # Additional labels
        self.Y_acc = _get(data, ["Y_ACC","Y_acc","Yacc"], None)
        self.Y_gyr = _get(data, ["Y_GYR","Y_gyr","Ygyr"], None)
        
        if self.Y_acc is not None:
            self.Y_acc = self.Y_acc.astype(np.float32)
        if self.Y_gyr is not None:
            self.Y_gyr = self.Y_gyr.astype(np.float32)
        
        self.N, self.T, self.D = self.X_all.shape
        self._dbg_cnt = {}
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return dict with keys: X, E2, MASK, (DF if available)"""
        item = {
            "X": torch.from_numpy(self.X_all[idx]).float(),
            "E2": torch.from_numpy(self.E2_all[idx]).float(),
            "MASK": torch.from_numpy(self.M_all[idx]).float(),
        }
        
        if self.DF_all is not None:
            item["DF"] = torch.from_numpy(self.DF_all[idx]).float()
        
        return item


def build_loader(npz_path, route="acc", x_mode="both", batch_size=32, shuffle=True, **kwargs):
    """Build DataLoader for IMU routes"""
    ds = IMURouteDataset(npz_path, route=route, x_mode=x_mode)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, **kwargs
    )
    return ds, dl

