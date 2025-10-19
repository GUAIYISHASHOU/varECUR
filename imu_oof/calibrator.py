# imu_oof/calibrator.py
from __future__ import annotations
import json
import math
import torch
import numpy as np
from pathlib import Path

class SA3AffineCalibrator:
    """
    三轴各向异性校准（SA域仿射）：
      s'    = α_s * s    + β_s
      a_xy' = α_xy * a_xy + β_xy
      a_z'  = α_z  * a_z  + β_z
    其中 (s, a_xy, a_z) 为对三轴 log-variance 的 SA3 分解。
    """
    def __init__(self,
                 alpha_s: float = 1.0, beta_s: float = 0.0,
                 alpha_a_xy: float = 1.0, beta_a_xy: float = 0.0,
                 alpha_a_z: float = 1.0, beta_a_z: float = 0.0,
                 dist: str = "studentt_diag_axes", nu: float = 5.0,
                 logv_min: float = -8.0, logv_max: float = 6.0):
        assert dist == "studentt_diag_axes", "Only anisotropic Student-t is supported"
        self.alpha_s = float(alpha_s);   self.beta_s = float(beta_s)
        self.alpha_a_xy = float(alpha_a_xy); self.beta_a_xy = float(beta_a_xy)
        self.alpha_a_z = float(alpha_a_z);   self.beta_a_z = float(beta_a_z)
        self.dist = dist
        self.nu = float(nu)
        self.logv_min = float(logv_min)
        self.logv_max = float(logv_max)

    def apply(self, logv_axes):
        """对 (..,3) 的三轴 log-variance 施加 SA 域仿射并裁剪到 [logv_min, logv_max]。
        支持 torch.Tensor 或 numpy.ndarray，保持输入类型。
        """
        if isinstance(logv_axes, torch.Tensor):
            from utils_sa3 import decompose_sa3_torch, reconstruct_sa3_torch
            s, a_xy, a_z = decompose_sa3_torch(logv_axes)
            s = self.alpha_s * s + self.beta_s
            a_xy = self.alpha_a_xy * a_xy + self.beta_a_xy
            a_z  = self.alpha_a_z  * a_z  + self.beta_a_z
            out = reconstruct_sa3_torch(s, a_xy, a_z)
            out = torch.clamp(out, self.logv_min, self.logv_max)
            return out
        else:
            from utils_sa3 import decompose_sa3_np, reconstruct_sa3_np
            x = np.asarray(logv_axes, dtype=np.float32)
            s, a_xy, a_z = decompose_sa3_np(x)
            s = self.alpha_s * s + self.beta_s
            a_xy = self.alpha_a_xy * a_xy + self.beta_a_xy
            a_z  = self.alpha_a_z  * a_z  + self.beta_a_z
            out = reconstruct_sa3_np(s, a_xy, a_z)
            out = np.clip(out, self.logv_min, self.logv_max)
            return out

    def to_dict(self):
        return {
            "sa_affine": {
                "alpha_s": self.alpha_s, "beta_s": self.beta_s,
                "alpha_a_xy": self.alpha_a_xy, "beta_a_xy": self.beta_a_xy,
                "alpha_a_z": self.alpha_a_z, "beta_a_z": self.beta_a_z,
            },
            "mode": "sa_affine",
            "dist": self.dist,
            "nu": self.nu,
            "logv_min": self.logv_min,
            "logv_max": self.logv_max,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "SA3AffineCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 统一 dist 口径
        dist_raw = str(cfg.get("dist", "studentt_diag_axes"))
        dist = "studentt_diag_axes" if ("studentt" in dist_raw) else "studentt_diag_axes"
        nu = float(cfg.get("nu", 5.0))
        logv_min = float(cfg.get("logv_min", -8.0))
        logv_max = float(cfg.get("logv_max", 6.0))

        if "sa_affine" in cfg:
            p = cfg["sa_affine"]
            return SA3AffineCalibrator(
                p.get("alpha_s", 1.0), p.get("beta_s", 0.0),
                p.get("alpha_a_xy", 1.0), p.get("beta_a_xy", 0.0),
                p.get("alpha_a_z", 1.0), p.get("beta_a_z", 0.0),
                dist=dist, nu=nu, logv_min=logv_min, logv_max=logv_max,
            )
        # 兼容旧温度校准（各向同性）：映射为仅对 s 做仿射，a_xy/a_z 不变
        if ("a" in cfg) and ("b" in cfg):
            a = float(cfg["a"]); b = float(cfg["b"])
            return SA3AffineCalibrator(
                alpha_s=a, beta_s=b,
                alpha_a_xy=1.0, beta_a_xy=0.0,
                alpha_a_z=1.0,  beta_a_z=0.0,
                dist=dist, nu=nu, logv_min=logv_min, logv_max=logv_max,
            )
        raise ValueError(f"Unsupported calibrator JSON schema: {path}")
