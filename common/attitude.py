# common/attitude.py
import numpy as np

def _q_continuous(q):
    """确保四元数序列连续性，避免符号翻转"""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12, None)
    out = q.copy()
    for i in range(1, len(q)):
        if np.dot(out[i-1], out[i]) < 0:  # 避免符号翻转
            out[i] = -out[i]
    return out

def q_to_R(q):
    """四元数转旋转矩阵 [w,x,y,z] -> (N,3,3)"""
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def so3_log(R):
    """SO(3) logarithmic map - 单个旋转矩阵"""
    c = ((np.trace(R) - 1.0) * 0.5).clip(-1, 1)
    th = np.arccos(c)
    if th < 1e-8:
        return np.zeros(3)
    W = (R - R.T) / (2 * np.sin(th))
    return th * np.array([W[2, 1], W[0, 2], W[1, 0]])

def quat_to_R(q):
    """别名：与代码中其它模块保持一致的函数名。
    等价于 q_to_R(q)。
    """
    return q_to_R(q)

def _fix_quat_sign(q: np.ndarray) -> np.ndarray:
    """连续化四元数符号，避免相邻采样点的符号翻转导致大角增量。
    q: (N,4) in [w,x,y,z]
    """
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0:
            out[i] = -out[i]
    return out

def slerp_many(t_src: np.ndarray, q_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """批量 SLERP：把 (t_src, q_src) 插值到 t_tgt。
    - t_src: (N,)
    - q_src: (N,4) [w,x,y,z]
    - t_tgt: (M,)
    返回: (M,4)
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    q_src = np.asarray(q_src, dtype=np.float64)
    t_tgt = np.asarray(t_tgt, dtype=np.float64)

    q_src = _fix_quat_sign(q_src)
    idx = np.searchsorted(t_src, t_tgt, side='right') - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    t0, t1 = t_src[idx], t_src[idx + 1]
    q0, q1 = q_src[idx], q_src[idx + 1]

    # SLERP 计算
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    u = ((t_tgt - t0) / (t1 - t0 + 1e-12)).reshape(-1, 1)
    small = (sin_omega <= 1e-6)
    a = np.where(~small, np.sin((1 - u) * omega) / (sin_omega + 1e-12), 1 - u)
    b = np.where(~small, np.sin(u * omega) / (sin_omega + 1e-12), u)
    q = a * q0 + b * q1
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return q

def quat_to_omega_body(q, t_s, dt_min=1e-4):
    """
    四元数序列转体坐标系角速度
    使用Log映射: ω = log(R_i^T * R_{i+1}) / dt
    """
    q = _q_continuous(q)
    R = q_to_R(q)
    w = np.zeros((len(q), 3))
    
    # 预计算 dt 序列用于日志
    dt_vals = np.diff(t_s)

    for i in range(len(q) - 1):
        dt = t_s[i + 1] - t_s[i]
        if dt <= dt_min:  # 避免"假尖峰"
            w[i] = w[i - 1] if i > 0 else 0.0
            continue
        dR = R[i].T @ R[i + 1]
        w[i] = so3_log(dR) / dt  # rad/s
    
    w[-1] = w[-2] if len(q) > 1 else w[-1]

    # ====== 调试日志：dt 统计与被标记比例 ======
    if dt_vals.size > 0:
        try:
            print(f"[omega] dt_median={np.median(dt_vals):.6g}s, flagged={(dt_vals<=dt_min).mean():.4%} (dt_min={dt_min})")
        except Exception:
            pass
    return w
