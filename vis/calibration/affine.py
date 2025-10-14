# -*- coding: utf-8 -*-
# vis/calibration/affine.py
import json
import numpy as np

class AffineCalibrator:
    """标定 logvar：lg ≈ α * lp + β。只在 val 拟合，test 仅应用。"""
    def __init__(self, name):
        self.name = name
        self.alpha = 1.0
        self.beta = 0.0

    def fit(self, lp: np.ndarray, lg: np.ndarray):
        """
        在验证集上拟合仿射变换参数
        
        Args:
            lp: 预测的 logvar
            lg: 真实的 logvar (GT)
        """
        # 允许鲁棒：对极端点做分位裁剪，避免被尾部拉偏
        p_low, p_high = np.percentile(lp, [1.0, 99.0])
        m = (lp >= p_low) & (lp <= p_high)
        A = np.c_[lp[m], np.ones_like(lp[m])]
        # 最小二乘：解 [α, β]
        theta, *_ = np.linalg.lstsq(A, lg[m], rcond=None)
        self.alpha, self.beta = float(theta[0]), float(theta[1])

    def apply(self, lp: np.ndarray) -> np.ndarray:
        """
        应用校准变换
        
        Args:
            lp: 预测的 logvar
            
        Returns:
            校准后的 logvar
        """
        return self.alpha * lp + self.beta

    def to_dict(self):
        """序列化为字典"""
        return {"type":"affine", "name":self.name,
                "alpha":self.alpha, "beta":self.beta}

    @staticmethod
    def from_dict(d):
        """从字典反序列化"""
        c = AffineCalibrator(d["name"])
        c.alpha, c.beta = float(d["alpha"]), float(d["beta"])
        return c
