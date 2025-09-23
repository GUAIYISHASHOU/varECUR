# common/euroc_state_gt.py
import numpy as np
import pandas as pd
from .attitude import quat_to_omega_body, slerp_many, quat_to_R
from .resample import interp_lin, smooth_mavg, central_diff

def _canon_cols(df):
    # 去掉BOM/首尾空格；统一小写；去空格与中括号内容做匹配
    new = []
    for c in df.columns:
        c2 = c.replace('\ufeff','').strip()
        new.append(c2)
    df.columns = new
    return df

def _find(df, *frags, required=True):
    """按子串模糊匹配列名，比如 ('p_RS_R','x')"""
    for c in df.columns:
        k = c.lower().replace(' ', '')
        if all(f.lower().replace(' ', '') in k for f in frags):
            return c
    if required:
        raise KeyError(f"missing column like: {frags}, have={list(df.columns)[:10]} ...")
    return None

def load_state_gt(csv_path):
    # 1) 自适应分隔符 + 去BOM
    df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')
    df = _canon_cols(df)

    # 2) 找各列（允许轻微命名差异）
    t_col = (_find(df, 'timestamp', 'ns', required=False)
             or _find(df, '#timestamp', required=False)
             or _find(df, 'timestamp', required=False)
             or df.columns[0])   # 兜底：用第一列
    p_cols = [_find(df, 'p_rs_r', ax) for ax in ['x','y','z']]
    q_cols = [_find(df, 'q_rs', comp) for comp in ['w','x','y','z']]
    v_cols = [_find(df, 'v_rs_r', ax) for ax in ['x','y','z']]

    # 可选的IMU bias，缺失则返回None
    bw_cols = [_find(df, 'b_w_rs_s', ax, required=False) for ax in ['x','y','z']]
    ba_cols = [_find(df, 'b_a_rs_s', ax, required=False) for ax in ['x','y','z']]
    has_bw = all(c is not None for c in bw_cols)
    has_ba = all(c is not None for c in ba_cols)

    # 3) 取值并做时间单位转换（自适应识别 ns/µs/ms/s）
    # 读原始时间列 -> 自动识别单位 -> 换算到秒
    t_raw = df[t_col].to_numpy(np.float64)
    med_abs = np.median(np.abs(t_raw))
    if med_abs > 1e16:
        scale = 1e-9   # ns -> s
    elif med_abs > 1e13:
        scale = 1e-6   # µs -> s
    elif med_abs > 1e10:
        scale = 1e-3   # ms -> s
    else:
        scale = 1.0    # s
    t_s = t_raw * scale

    # 严格递增去重（别在整数化之前做，保留小数精度）
    keep = np.r_[True, np.diff(t_s) > 0.0]
    t_s = t_s[keep]
    df = df.loc[keep]

    # ====== 调试日志：单位与dt统计 ======
    dt_min = 1e-4  # 秒；按你的IMU频率调（200Hz≈0.005s）
    if len(t_s) >= 2:
        dt = np.diff(t_s)
        try:
            print(f"[gt] unit scale={scale}, dt_median={np.median(dt):.6g}s, dt_min={dt.min():.6g}s, bad_dt_ratio={(dt<=dt_min).mean():.4%}")
        except Exception:
            # 打印失败不影响主流程
            pass

    p = df[p_cols].to_numpy(np.float64)
    q = df[q_cols].to_numpy(np.float64)
    v = df[v_cols].to_numpy(np.float64)
    bw = df[bw_cols].to_numpy(np.float64) if has_bw else None
    ba = df[ba_cols].to_numpy(np.float64) if has_ba else None

    # 4) 用Lie-Log在GT时间轴上求角速度
    w = quat_to_omega_body(q, t_s, dt_min=1e-4)  # rad/s

    return dict(t=t_s, p=p, q=q, v=v, w=w, bw=bw, ba=ba)

def make_acc_body_gt(gt: dict, t_imu, smooth_win: int = 7, g_sign: int = -1, edge_mask: int = 3):
    """
    从 state_groundtruth_estimate0 生成体坐标 ACC 真值以及边界 mask。
    - gt: dict(t, q, v)
    - t_imu: (M,) 秒
    返回: a_body_gt (M,3), mask (M,)
    """
    t_imu = np.asarray(t_imu, dtype=np.float64)

    # 1) 插值到 IMU 时间轴
    q_imu = slerp_many(gt['t'], gt['q'], t_imu)
    R_imu = quat_to_R(q_imu)  # world->body
    v_imu = interp_lin(gt['t'], gt['v'], t_imu)

    # 2) 轻度平滑 + 鲁棒中央差分（世界系）
    win = int(smooth_win)
    if win <= 1:
        v_s = v_imu
    else:
        if win % 2 == 0:
            win += 1
        v_s = smooth_mavg(v_imu, win=win)
    a_w = central_diff(v_s, t_imu, dt_min=1e-4)

    # 3) 减重力并旋到体轴
    g_world = np.array([0.0, 0.0, g_sign * 9.80665], dtype=np.float64)
    a_body_gt = np.einsum('bij,bj->bi', np.transpose(R_imu, (0, 2, 1)), a_w - g_world)

    # 4) 边界 mask（中央差分两端不稳）
    M = len(t_imu)
    mask = np.ones(M, dtype=bool)
    k = max(0, int(edge_mask))
    if k > 0:
        mask[:k] = False
        mask[-k:] = False

    return a_body_gt.astype(np.float64), mask
