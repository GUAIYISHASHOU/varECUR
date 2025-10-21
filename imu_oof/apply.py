from __future__ import annotations
import numpy as np
from pathlib import Path
from .calibrator import SA3AffineCalibrator


def apply_calibrator_if_any(logv: np.ndarray, calibrator_json: str | None):
    """
    应用校准器到 logv
    logv: (N,T,d_out) 或 (N,T)
    """
    if not calibrator_json or not Path(calibrator_json).exists():
        return logv
    # 仅在各向异性三轴时应用
    if (isinstance(logv, np.ndarray) and (logv.ndim >= 3) and (logv.shape[-1] == 3)):
        cal = SA3AffineCalibrator.load(calibrator_json)
        return cal.apply(logv)
    return logv
