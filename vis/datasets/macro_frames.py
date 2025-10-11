# -*- coding: utf-8 -*-
# vis/datasets/macro_frames.py
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class MacroFrames(Dataset):
    """
    NPZ 字段约定：
      patches:(M,K,2,H,W) uint8  [0..255]
      geoms:  (M,K,G)     float32  # G维几何特征（默认模式20维，关键帧模式24维，自动从meta读取）
      y_true: (M,2)       float32  # [logσx², logσy²]
      num_tokens:(M,)     int32    # 每个样本真实点数（<=K）
      y_inlier:(M,1)      uint8    # 内点标签 [0 或 1] (存储用uint8，使用时转float32)
      meta:   dict                 # 包含 geom_dim, generation_mode, geoms_desc 等元信息
    """
    def __init__(self, npz_path: str, geom_stats_path: str = None):
        z = np.load(npz_path, allow_pickle=True)
        self.patches = z["patches"]
        self.geoms   = z["geoms"]
        self.y_true  = z["y_true"]
        self.num_tok = z["num_tokens"].astype(np.int64)
        
        # === 读取并显示元信息 ===
        if "meta" in z:
            meta = z["meta"].item() if hasattr(z["meta"], 'item') else z["meta"]
            self.meta = meta
            print(f"[Dataset] 加载数据: {os.path.basename(npz_path)}")
            print(f"  生成模式: {meta.get('generation_mode', 'unknown')}")
            print(f"  几何特征维度: {meta.get('geom_dim', self.geoms.shape[2])}")
            print(f"  样本数: {self.patches.shape[0]}")
            if 'kf_policy' in meta and meta['kf_policy'] is not None:
                print(f"  关键帧策略: parallax={meta['kf_policy'].get('parallax_px')}px, "
                      f"max_dt={meta['kf_policy'].get('max_interval_s')}s")
        else:
            self.meta = {}
            print(f"[Dataset] 警告: NPZ文件缺少meta信息，使用默认配置")
        
        # === 加载内点标签（uint8存储，节省空间）===
        # 如果旧的npz文件没有这个键，就默认所有都是内点
        if "y_inlier" in z:
            self.y_inlier = z["y_inlier"]  # uint8 或 float32
        else:
            # 旧NPZ兜底：全部标记为内点
            self.y_inlier = np.ones((len(self.y_true), 1), dtype=np.uint8)
            print(f"[Dataset] 警告: 旧格式NPZ，自动创建内点标签（全部标记为内点）")

        # === 新增：加载并准备归一化统计量 ===
        if geom_stats_path and os.path.exists(geom_stats_path):
            stats = np.load(geom_stats_path)
            self.geom_mean = torch.from_numpy(stats['mean'].astype(np.float32))
            self.geom_std = torch.from_numpy(stats['std'].astype(np.float32))
            print(f"[Dataset] 已加载特征统计量: {geom_stats_path}")
            print(f"  Mean shape: {self.geom_mean.shape}, Std shape: {self.geom_std.shape}")
        else:
            if geom_stats_path:
                print(f"[Dataset] 警告: 未找到统计量文件 {geom_stats_path}")
            print("[Dataset] 警告: 未提供特征统计量，geoms将不进行归一化")
            self.geom_mean = None
            self.geom_std = None

    def __len__(self): return self.patches.shape[0]

    def __getitem__(self, i: int):
        # 转换geoms为tensor
        geoms_tensor = torch.from_numpy(self.geoms[i])

        # === 新增：应用Z-score归一化 ===
        if self.geom_mean is not None and self.geom_std is not None:
            geoms_tensor = (geoms_tensor - self.geom_mean) / self.geom_std

        return {
            "patches": torch.from_numpy(self.patches[i].astype(np.float32) / 255.0),
            "geoms":   geoms_tensor,  # 已归一化（如果有统计量）
            "y_true":  torch.from_numpy(self.y_true[i]),
            "num_tok": torch.tensor(self.num_tok[i], dtype=torch.long),
            # === 返回内点标签（转为float32用于BCE loss）===
            "y_inlier": torch.from_numpy(self.y_inlier[i].astype(np.float32)),
        }

