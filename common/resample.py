# common/resample.py
import numpy as np

def interp_vec(t_src, y_src, t_tgt):
    """向量插值函数"""
    idx = np.searchsorted(t_src, t_tgt, 'left')
    i0 = (idx - 1).clip(0, len(t_src) - 1)
    i1 = idx.clip(0, len(t_src) - 1)
    t0, t1 = t_src[i0], t_src[i1]
    y0, y1 = y_src[i0], y_src[i1]
    denom = (t1 - t0)
    denom[denom == 0] = 1.0
    a = ((t_tgt - t0) / denom)[:, None]
    return y0 + a * (y1 - y0)

def interp_lin(t_src: np.ndarray, X_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """逐轴一维线插，X_src shape=(N,D)。"""
    t_src = np.asarray(t_src, dtype=np.float64)
    X_src = np.asarray(X_src, dtype=np.float64)
    t_tgt = np.asarray(t_tgt, dtype=np.float64)
    D = X_src.shape[1]
    X = np.empty((len(t_tgt), D), dtype=np.float64)
    for d in range(D):
        X[:, d] = np.interp(t_tgt, t_src, X_src[:, d])
    return X

def smooth_mavg(X: np.ndarray, win: int = 7) -> np.ndarray:
    """简单移动平均平滑，win 必须为奇数；win<=1 时返回原值。"""
    if win <= 1:
        return X
    assert win % 2 == 1
    pad = win // 2
    K = np.ones(win, dtype=np.float64) / float(win)
    Y = np.empty_like(X, dtype=np.float64)
    for d in range(X.shape[1]):
        x = np.pad(X[:, d], (pad, pad), mode='edge')
        Y[:, d] = np.convolve(x, K, mode='valid')
    return Y

def central_diff(X: np.ndarray, t: np.ndarray, dt_min: float = 1e-4) -> np.ndarray:
    """鲁棒中央差分；两端用一阶差分；对极小 dt 做保护。"""
    N, D = X.shape
    Y = np.zeros((N, D), dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if N <= 1:
        return Y
    # 中间点：用跨两步的差分
    if N >= 3:
        dt2 = (t[2:] - t[:-2])
        good = dt2 > (2.0 * dt_min)
        Y[1:-1][good] = (X[2:][good] - X[:-2][good]) / dt2[good, None]
    # 两端：一阶差分，夹一个 dt_min 下限
    d0 = max(t[1] - t[0], dt_min)
    d1 = max(t[-1] - t[-2], dt_min)
    Y[0] = (X[1] - X[0]) / d0
    Y[-1] = (X[-1] - X[-2]) / d1
    return Y