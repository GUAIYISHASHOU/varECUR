from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict

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
    def __init__(self, npz_path: str | Path, route: str = "acc", x_mode: str = "both"):
        self.npz_path = str(npz_path)
        self.route = route
        self.x_mode = x_mode
        assert route in ("acc","gyr","vis")
        assert x_mode in ("both","route_only")
        if self.route == "vis" and self.x_mode != "both":
            raise ValueError("Vision route only supports x_mode='both'")

        data = np.load(self.npz_path, allow_pickle=True)

        def _pick(keys):
            for k in keys:
                if k in data.files: return data[k]
            return None

        # 获取输入数据
        if self.route == "acc":
            X = _pick(["X_IMU_ACC", "X_acc", "X"])
            TS = _pick(["TS_IMU"])
        elif self.route == "gyr":
            X = _pick(["X_IMU_GYR", "X_gyr", "X"])
            TS = _pick(["TS_IMU"])
        elif self.route == "vis":
            X = _pick(["X_VIS", "X_vis", "X"])
            TS = _pick(["TS_VIS"])
        else:
            raise ValueError(f"unknown route {self.route}")

        if X is None:
            raise ValueError(f"{self.npz_path}: missing X keys for route={self.route}")

        X = X.astype(np.float32)
        self.N, self.T, self.D = X.shape

        # 1) 优先用步级标签（推荐）
        if self.route == "acc" and "E2_IMU_ACC" in data.files:
            E2_step = data["E2_IMU_ACC"].astype(np.float32)  # [N,T,3]
            self.use_step_labels = True
            self.d_out = 3  # 支持三轴输出
            E2 = E2_step  # [N,T,3]
            Y_anchor = None
        elif self.route == "gyr" and "E2_IMU_GYR" in data.files:
            E2_step = data["E2_IMU_GYR"].astype(np.float32)  # [N,T,3]
            self.use_step_labels = True
            self.d_out = 3  # 支持三轴输出
            E2 = E2_step  # [N,T,3]
            Y_anchor = None
        # 2) 否则回退：窗口锚 + 轻中心化（旧 EUROC npz 只有 Y_IMU_*）
        else:
            self.use_step_labels = False
            if self.route == "acc":
                Y_anchor = _pick(["Y_IMU_ACC", "E2_acc", "E2", "Y"])
            elif self.route == "gyr":
                Y_anchor = _pick(["Y_IMU_GYR", "E2_gyr", "E2", "Y"])
            else:  # vis
                Y_anchor = _pick(["Y_VIS", "E2_vis", "E2", "Y"])
            
            if Y_anchor is None:
                raise ValueError(f"{self.npz_path}: no labels found for route={self.route}")
            
            Y_anchor = Y_anchor.astype(np.float32)
            self.d_out = 1  # 回退模式使用标量输出
            
            # 把窗口标签变成逐时间步，用于与旧代码兼容
            if self.route in ("acc", "gyr"):
                if Y_anchor.ndim == 2 and Y_anchor.shape[1] == 3:
                    e2_scalar = Y_anchor.sum(axis=1).astype(np.float32) / 3.0  # 每轴均值
                elif Y_anchor.ndim == 2 and Y_anchor.shape[1] == 1:
                    e2_scalar = Y_anchor[:,0]
                else:
                    raise ValueError(f"unexpected Y shape {Y_anchor.shape} for IMU")
                E2 = np.repeat(e2_scalar[:, None, None], self.T, axis=1)  # [N,T,1]
            else:  # vis
                if Y_anchor.ndim == 2 and Y_anchor.shape[1] == 1:
                    e2_scalar = Y_anchor[:,0].astype(np.float32)
                else:
                    e2_scalar = Y_anchor.squeeze().astype(np.float32)
                E2 = np.repeat(e2_scalar[:, None, None], self.T, axis=1)  # [N,T,1]

        # 3) MASK：如果没有就置为全 1
        M = _pick(["MASK_IMU", "MASK", "mask", "mask_vis"])
        if M is None:
            M = np.ones((self.N, self.T), dtype=np.float32)
        else:
            M = (M.astype(np.float32) > 0.5).astype(np.float32)
            if M.ndim == 3:  # [N,T,1]→[N,T]
                M = M[...,0]

        self.X_all = X
        self.E2_all = E2  # [N,T,3] 或 [N,T,1]
        self.M_all = M    # [N,T]
        self.TS = TS if TS is not None else None
        self.Y_anchor = Y_anchor  # 窗口级标签（如果有的话）

        self.Y_acc = _get(data, ["Y_ACC","Y_acc","Yacc"], None)
        self.Y_gyr = _get(data, ["Y_GYR","Y_gyr","Ygyr"], None)
        if self.Y_acc is not None:
            self.Y_acc = self.Y_acc.astype(np.float32)
        if self.Y_gyr is not None:
            self.Y_gyr = self.Y_gyr.astype(np.float32)

        self.N, self.T, self.D = self.X_all.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        X = self.X_all[idx]     # [T, D]
        E2 = self.E2_all[idx]   # [T, 3] 或 [T, 1]
        M = self.M_all[idx]     # [T]

        # 处理路由特定的输入
        if self.route == "acc":
            Y = self.Y_acc[idx] if self.Y_acc is not None else None
            if self.x_mode == "route_only" and X.shape[-1] >= 6:
                X = X[..., :3]
        elif self.route == "gyr":
            Y = self.Y_gyr[idx] if self.Y_gyr is not None else None
            if self.x_mode == "route_only" and X.shape[-1] >= 6:
                X = X[..., 3:6]
        else:  # vis
            Y = None

        out = {
            "X": torch.from_numpy(X),
            "MASK": torch.from_numpy(M),
            "E2": torch.from_numpy(E2.astype(np.float32)),
            "use_step_labels": bool(self.use_step_labels),  # 确保是Python布尔值
        }
        
        # 添加窗口级标签（如果有的话）
        if hasattr(self, 'Y_anchor') and self.Y_anchor is not None:
            out["Y_anchor"] = torch.from_numpy(self.Y_anchor[idx].astype(np.float32))
        
        if Y is not None:
            out["Y"] = torch.from_numpy(Y)
        else:
            out["Y"] = torch.zeros_like(out["MASK"])
        
        return out

# === GNSS 数据集（ENU三维） ===
class GNSDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z['X'].astype(np.float32)     # (N, T, Din)
        self.Y = z['Y'].astype(np.float32)     # (N, T, 3)  ENU误差
        self.mask = z['mask'].astype(bool)     # (N, T, 3)
        self.meta = z.get('meta', None)
        assert self.X.shape[0] == self.Y.shape[0] == self.mask.shape[0]
        assert self.Y.shape[-1] == 3, "GNS Y should be (..,3) for ENU"
    
    def __len__(self):  
        return self.X.shape[0]
    
    def __getitem__(self, i):
        y_axes = self.Y[i].astype(np.float32)            # (T,3)
        e2_axes = (y_axes ** 2).astype(np.float32)       # (T,3)
        e2_sum  = e2_axes.sum(axis=-1, keepdims=True)    # (T,1)  ← 训练/评测用
        m_axes  = self.mask[i].astype(np.float32)        # (T,3)
        m_any   = (m_axes > 0.5).all(axis=-1, keepdims=True).astype(np.float32)  # (T,1)

        return {
            "X": torch.from_numpy(self.X[i]),            # (T,Din)
            "E2": torch.from_numpy(e2_sum),              # (T,1)  ← 配合 nll_iso3_e2
            "MASK": torch.from_numpy(m_any),             # (T,1)  ← 与上对齐
            # 下面是作图/逐维统计需要的"富信息"
            "Y": torch.from_numpy(y_axes),               # (T,3)
            "MASK_AXES": torch.from_numpy(m_axes),       # (T,3)
            "E2_AXES": torch.from_numpy(e2_axes),        # (T,3)
        }

def build_dataset(route: str, npz_path: str):
    """数据集工厂函数"""
    route = route.lower()
    if route in ('acc', 'gyr', 'vis'):
        return IMURouteDataset(npz_path, route=route, x_mode="both")
    elif route == 'gns':
        return GNSDataset(npz_path)
    else:
        raise ValueError(f"Unknown route {route}")

def build_loader(npz_path, route="acc", x_mode="both",
                 batch_size=32, shuffle=True, num_workers=0,
                 generator=None, worker_init_fn=None):
    if route.lower() == 'gns':
        ds = build_dataset(route, npz_path)
    else:
        ds = IMURouteDataset(npz_path, route=route, x_mode=x_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                    pin_memory=True, generator=generator, worker_init_fn=worker_init_fn)
    return ds, dl
