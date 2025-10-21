from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import List, Optional, Dict

# 约定：保持与原 IMU 项目一致的键名与形状
# X：X_IMU_ACC / X_IMU_GYR -> (N, T, D)
# Y：E2_IMU_ACC / E2_IMU_GYR -> (N, T, 3)   （优先步级；否则回退 Y_IMU_*）
# MASK：MASK_IMU -> (N, T)
# SEQ_NAME：SEQ_NAME -> (N,) 序列名（用于OOF分割）

class IMUDataset(Dataset):
    def __init__(self, npz_path: str, route: str, scaler: Optional[Dict] = None):
        """
        Args:
            npz_path: NPZ文件路径或目录（包含多个序列NPZ）
            route: "acc" | "gyr"
            scaler: 可选的scaler字典 {"mean": array, "std": array}
        """
        self.npz_path = str(npz_path)
        self.route = route  # "acc" | "gyr"
        self.scaler = scaler
        data = np.load(self.npz_path)

        def _pick(keys):
            for k in keys:
                if k in data.files:
                    return data[k]
            return None

        if route == "acc":
            X = _pick(["X_IMU_ACC", "X_acc", "X"])
        elif route == "gyr":
            X = _pick(["X_IMU_GYR", "X_gyr", "X"])
        else:
            raise ValueError(f"unknown route {route}")
        if X is None:
            raise ValueError(f"{npz_path}: missing X keys for route={route}")
        self.X = X.astype(np.float32)
        self.N, self.T, self.D = self.X.shape

        # 标签优先使用步级三轴 e2
        if route == "acc" and "E2_IMU_ACC" in data.files:
            self.E2 = data["E2_IMU_ACC"].astype(np.float32)  # (N,T,3)
            self.use_step = True
        elif route == "gyr" and "E2_IMU_GYR" in data.files:
            self.E2 = data["E2_IMU_GYR"].astype(np.float32)
            self.use_step = True
        else:
            # 回退：单通道锚（旧数据集）；在损失里按窗口均值约束
            key = "Y_IMU_ACC" if route == "acc" else "Y_IMU_GYR"
            if key not in data.files:
                raise ValueError(f"{npz_path}: missing step labels and {key}")
            y = data[key].astype(np.float32)  # (N,T) or (N,T,1) or (N,T,3)
            if y.ndim == 2:
                y = y[..., None]
            if y.shape[-1] == 1:
                y = np.repeat(y, 3, axis=-1)
            self.E2 = y.astype(np.float32)  # (N,T,3)
            self.use_step = False

        mask = _pick(["MASK_IMU", "mask"])
        if mask is None:
            mask = np.ones((self.N, self.T), np.float32)
        self.MASK = mask.astype(np.float32)
        
        # 序列名（用于OOF分割）
        seq_name = _pick(["SEQ_NAME", "seq_name", "seq"])
        if seq_name is not None:
            self.SEQ_NAME = seq_name
        else:
            self.SEQ_NAME = None
        
        # 应用scaler（如果提供）
        if self.scaler is not None:
            self.apply_scaler(self.scaler)

    def __len__(self):
        return self.N

    def fit_scaler(self) -> Dict:
        """
        拟合scaler：计算训练集的均值和标准差
        只在训练集上调用，验证集使用训练集的scaler
        """
        # 使用mask过滤有效数据
        mask_bool = self.MASK > 0.5  # (N, T)
        
        # 计算每个特征的均值和标准差
        valid_data = []
        for i in range(self.N):
            for t in range(self.T):
                if mask_bool[i, t]:
                    valid_data.append(self.X[i, t])
        
        if len(valid_data) == 0:
            # 如果没有有效数据，返回零均值单位方差
            return {
                "mean": np.zeros(self.D, dtype=np.float32),
                "std": np.ones(self.D, dtype=np.float32)
            }
        
        valid_data = np.array(valid_data, dtype=np.float32)  # (M, D)
        mean = valid_data.mean(axis=0)
        std = valid_data.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # 避免除零
        
        scaler = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32)
        }
        
        return scaler
    
    def apply_scaler(self, scaler: Dict):
        """应用scaler进行标准化"""
        mean = scaler["mean"]
        std = scaler["std"]
        self.X = (self.X - mean) / std
    
    def __getitem__(self, idx):
        x = self.X[idx]           # (T,D)
        e2 = self.E2[idx]         # (T,3)
        m = self.MASK[idx]        # (T,)
        return {
            "x": torch.from_numpy(x),
            "e2": torch.from_numpy(e2),
            "mask": torch.from_numpy(m),
        }


def build_loader(npz_path: str, route: str, batch_size: int, shuffle: bool, num_workers: int = 4, scaler: Optional[Dict] = None):
    ds = IMUDataset(npz_path, route, scaler=scaler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def build_dataset_from_seqs(seq_npz_dir: str, seq_list: List[str], route: str, scaler: Optional[Dict] = None) -> IMUDataset:
    """
    从序列列表构建数据集（用于OOF训练）
    
    Args:
        seq_npz_dir: 包含各序列NPZ文件的目录
        seq_list: 序列名列表
        route: "acc" | "gyr"
        scaler: 可选的scaler
    
    Returns:
        IMUDataset实例
    """
    seq_dir = Path(seq_npz_dir)
    
    # 收集所有序列的数据
    all_X, all_E2, all_MASK, all_SEQ = [], [], [], []
    
    for seq in seq_list:
        # 尝试多种可能的文件名格式
        npz_candidates = [
            seq_dir / f"{seq}.npz",
            seq_dir / f"{seq}_{route}.npz",
            seq_dir / seq / f"{route}.npz",
        ]
        
        npz_path = None
        for candidate in npz_candidates:
            if candidate.exists():
                npz_path = candidate
                break
        
        if npz_path is None:
            # 回退：匹配包含采样后缀的文件名，如 MH_02_easy_T512_S256.npz
            from glob import glob
            fallback_patterns = [
                str(seq_dir / f"{seq}_T*_S*.npz"),
                str(seq_dir / f"{seq}_*.npz"),
                str(seq_dir / f"{seq}*.npz"),
            ]
            for pat in fallback_patterns:
                matches = glob(pat)
                if matches:
                    npz_path = Path(sorted(matches)[0])
                    break
            if npz_path is None:
                print(f"Warning: NPZ not found for sequence {seq}, skipping")
                continue
        
        # 加载数据
        data = np.load(npz_path)
        
        def _pick(keys):
            for k in keys:
                if k in data.files:
                    return data[k]
            return None
        
        # 提取X
        if route == "acc":
            X = _pick(["X_IMU_ACC", "X_acc", "X"])
        elif route == "gyr":
            X = _pick(["X_IMU_GYR", "X_gyr", "X"])
        else:
            raise ValueError(f"unknown route {route}")
        
        if X is None:
            print(f"Warning: X not found for {seq}, skipping")
            continue
        
        # 提取E2
        if route == "acc" and "E2_IMU_ACC" in data.files:
            E2 = data["E2_IMU_ACC"]
        elif route == "gyr" and "E2_IMU_GYR" in data.files:
            E2 = data["E2_IMU_GYR"]
        else:
            key = "Y_IMU_ACC" if route == "acc" else "Y_IMU_GYR"
            if key in data.files:
                y = data[key]
                if y.ndim == 2:
                    y = y[..., None]
                if y.shape[-1] == 1:
                    y = np.repeat(y, 3, axis=-1)
                E2 = y
            else:
                print(f"Warning: E2/Y not found for {seq}, skipping")
                continue
        
        # 提取MASK
        mask = _pick(["MASK_IMU", "mask"])
        if mask is None:
            mask = np.ones((X.shape[0], X.shape[1]), np.float32)
        
        all_X.append(X.astype(np.float32))
        all_E2.append(E2.astype(np.float32))
        all_MASK.append(mask.astype(np.float32))
        all_SEQ.extend([seq] * X.shape[0])
    
    if len(all_X) == 0:
        raise ValueError(f"No valid data found for sequences: {seq_list}")
    
    # 合并数据
    X_merged = np.concatenate(all_X, axis=0)
    E2_merged = np.concatenate(all_E2, axis=0)
    MASK_merged = np.concatenate(all_MASK, axis=0)
    SEQ_merged = np.array(all_SEQ)
    
    # 创建临时NPZ并加载为Dataset
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
        if route == "acc":
            np.savez_compressed(tmp_path, 
                              X_IMU_ACC=X_merged, 
                              E2_IMU_ACC=E2_merged, 
                              MASK_IMU=MASK_merged,
                              SEQ_NAME=SEQ_merged)
        else:
            np.savez_compressed(tmp_path, 
                              X_IMU_GYR=X_merged, 
                              E2_IMU_GYR=E2_merged, 
                              MASK_IMU=MASK_merged,
                              SEQ_NAME=SEQ_merged)
    
    ds = IMUDataset(tmp_path, route, scaler=scaler)
    
    # 删除临时文件
    Path(tmp_path).unlink()
    
    return ds
