#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EUROC -> NPZ (per-timestep IMU labels)
生成包含步级标签的 NPZ：ERR_IMU_ACC/GYR、E2_IMU_ACC/GYR、MASK_IMU、TS_*、X_*。
- 依赖：numpy, pandas
- 输入：EUROC 序列根目录（含 mav0/imu0/data.csv 与 mav0/state_groundtruth0/data.csv）
- 标签生成：优先使用 state_groundtruth_estimate0，回退到 vicon0
           -> 使用Log映射计算角速度真值，插值到IMU时间轴
           -> 支持GT bias补偿（可选）
注意：EUROC GT 世界系 z 轴向上（z-up），重力向下 => g_world = [0,0,-9.80665]
"""

from pathlib import Path
import argparse, csv, math, json
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from common.euroc_state_gt import load_state_gt, make_acc_body_gt
    from common.resample import interp_vec
    USE_NEW_GT = True
except ImportError as e:
    print(f"Warning: Could not import new GT modules ({e}), falling back to old method")
    USE_NEW_GT = False

def load_imu_csv(path):
    # EUROC imu: timestamp[nanosec], w_x, w_y, w_z [rad/s], a_x, a_y, a_z [m/s^2]
    t=[]; w=[]; a=[]
    with open(path, "r") as f:
        rd = csv.reader(f)
        for r in rd:
            if not r or r[0].startswith("#") or r[0]=="timestamp":
                continue
            ts = int(r[0]) * 1e-9
            wx,wy,wz = float(r[1]), float(r[2]), float(r[3])
            ax,ay,az = float(r[4]), float(r[5]), float(r[6])
            t.append(ts); w.append([wx,wy,wz]); a.append([ax,ay,az])
    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    return t, w, a

def load_gt_csv(path):
    # EUROC gt(state_groundtruth0): timestamp[nanosec], p_RS_R_[xyz], q_RS_[wxyz], v_RS_R_[xyz]
    t=[]; p=[]; q=[]; v=[]
    with open(path, "r") as f:
        rd = csv.reader(f)
        for r in rd:
            if not r or r[0].startswith("#") or r[0]=="timestamp":
                continue
            ts = int(r[0]) * 1e-9
            px,py,pz = float(r[1]), float(r[2]), float(r[3])
            qw,qx,qy,qz = float(r[4]), float(r[5]), float(r[6]), float(r[7])
            vx,vy,vz = float(r[8]), float(r[9]), float(r[10])
            t.append(ts); p.append([px,py,pz]); q.append([qw,qx,qy,qz]); v.append([vx,vy,vz])
    return (np.asarray(t, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
            np.asarray(q, dtype=np.float64),
            np.asarray(v, dtype=np.float64))

def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)

def q_mul(q1, q2):
    # [w,x,y,z]
    w1,x1,y1,z1 = np.moveaxis(q1, -1, 0)
    w2,x2,y2,z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w,x,y,z], axis=-1)

def q_conj(q):
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack([w,-x,-y,-z], axis=-1)

def q_slerp(q0, q1, u):
    # u in [0,1], q0,q1 shape (...,4)
    q0 = q_normalize(q0); q1 = q_normalize(q1)
    dot = np.sum(q0*q1, axis=-1, keepdims=True)
    # 避免长路径
    q1 = np.where(dot<0, -q1, q1)
    dot = np.abs(dot)
    omega = np.arccos(np.clip(dot, 0.0, 1.0))
    sin_omega = np.sin(omega)
    small = sin_omega < 1e-8
    
    # 确保形状匹配
    u = u[..., None]  # 添加最后一维以匹配四元数
    omega = omega  # 保持omega的形状
    
    out = np.where(small, (1-u)*q0 + u*q1,
                   (np.sin((1-u)*omega)/sin_omega)*q0 + (np.sin(u*omega)/sin_omega)*q1)
    return q_normalize(out)

def R_from_q(q):
    # q: (...,4) [w,x,y,z]
    w,x,y,z = np.moveaxis(q, -1, 0)
    ww,xx,yy,zz = w*w, x*x, y*y, z*z
    wx,wy,wz = w*x, w*y, w*z
    xy,xz,yz = x*y, x*z, y*z
    R = np.stack([
        ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz
    ], axis=-1)
    return R.reshape(q.shape[:-1]+(3,3))

def interp_linear(t_ref, t_src, x_src):
    # 对每列线性插值；t_ref,t_src 1D 升序
    idx = np.searchsorted(t_src, t_ref, side="left")
    idx0 = np.clip(idx-1, 0, len(t_src)-1)
    idx1 = np.clip(idx,   0, len(t_src)-1)
    t0, t1 = t_src[idx0], t_src[idx1]
    w = np.zeros_like(t_ref)
    denom = (t1 - t0)
    mask = denom > 0
    w[mask] = (t_ref[mask] - t0[mask]) / denom[mask]
    x0 = x_src[idx0]
    x1 = x_src[idx1]
    return (1-w)[:,None]*x0 + w[:,None]*x1

def interp_quat(t_ref, t_src, q_src):
    idx = np.searchsorted(t_src, t_ref, side="left")
    idx0 = np.clip(idx-1, 0, len(t_src)-1)
    idx1 = np.clip(idx,   0, len(t_src)-1)
    t0, t1 = t_src[idx0], t_src[idx1]
    w = np.zeros_like(t_ref)
    denom = (t1 - t0)
    mask = denom > 0
    w[mask] = (t_ref[mask] - t0[mask]) / denom[mask]
    q0 = q_src[idx0]
    q1 = q_src[idx1]
    return q_slerp(q0, q1, w)

def central_diff(y, t):
    # y: (N,3), t: (N,)
    N = len(t)
    dy = np.zeros_like(y)
    if N>=2:
        dy[0]   = (y[1]-y[0]) / max(t[1]-t[0], 1e-9)
        dy[-1]  = (y[-1]-y[-2]) / max(t[-1]-t[-2], 1e-9)
    for i in range(1, N-1):
        dt = t[i+1]-t[i-1]
        dy[i] = (y[i+1]-y[i-1]) / max(dt, 1e-9)
    return dy

def quat_to_omega_body(q, t):
    # ω ≈ Log( R_t^T R_{t+1} )/dt  (body-frame)
    R = R_from_q(q)  # (N,3,3)
    N = R.shape[0]
    w = np.zeros((N,3), dtype=np.float64)
    for i in range(N-1):
        Rt = R[i].T
        Rnext = R[i+1]
        dR = Rt @ Rnext
        # so(3) log
        theta = np.arccos(np.clip((np.trace(dR)-1)/2.0, -1, 1))
        if theta < 1e-8:
            vec = np.zeros(3)
        else:
            v = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / (2*np.sin(theta))
            vec = theta * v
        dt = max(t[i+1]-t[i], 1e-9)
        w[i] = vec / dt
    w[-1] = w[-2] if N >= 2 else w[-1]
    return w

# ======== Gyro helpers (moving average, log-map central diff, delay alignment) ========
def moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    if x.ndim == 1:
        xpad = np.pad(x, (pad, pad), mode='edge')
        ker = np.ones((k,), dtype=np.float64) / float(k)
        y = np.convolve(xpad, ker, mode='valid')
        return y
    else:
        ker = np.ones((k,), dtype=np.float64) / float(k)
        y = np.empty_like(x, dtype=np.float64)
        for i in range(x.shape[1]):
            xpad = np.pad(x[:, i], (pad, pad), mode='edge')
            y[:, i] = np.convolve(xpad, ker, mode='valid')
        return y

def omega_from_quat_cdiff(q_seq: np.ndarray, t_seq: np.ndarray, step: int = 1) -> np.ndarray:
    """Central-difference + SO(3) log-map to compute omega on the target axis.
    Returns NaN at boundaries for masking.
    """
    N = len(q_seq)
    w = np.full((N, 3), np.nan, dtype=np.float64)
    if N < 2 * step + 1:
        return w
    for i in range(step, N - step):
        dt = float(t_seq[i + step] - t_seq[i - step])
        if dt <= 0:
            continue
        Rm = R_from_q(q_seq[i - step])
        Rp = R_from_q(q_seq[i + step])
        dR = Rp @ Rm.T
        theta = np.arccos(np.clip((np.trace(dR) - 1) / 2.0, -1.0, 1.0))
        if theta < 1e-8:
            vec = np.zeros(3)
        else:
            v = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / (2*np.sin(theta))
            vec = theta * v
        w[i] = vec / max(dt, 1e-9)
    return w

def best_shift_by_corr(a: np.ndarray, b: np.ndarray, max_k: int) -> int:
    """Estimate integer delay that maximizes correlation between |a| and |b|.
    Returns best k (b delayed forward by k samples)."""
    if max_k is None or max_k <= 0:
        return 0
    a_mag = np.linalg.norm(a, axis=1)
    b_mag = np.linalg.norm(b, axis=1)
    N = len(a_mag)
    best_k, best_c = 0, -1.0
    for k in range(-max_k, max_k + 1):
        if k < 0:
            aa = a_mag[-k:N]
            bb = b_mag[0:N + k]
        elif k > 0:
            aa = a_mag[0:N - k]
            bb = b_mag[k:N]
        else:
            aa = a_mag
            bb = b_mag
        if len(aa) < max(16, 2 * max_k + 1):
            continue
        c = np.corrcoef(aa, bb)[0, 1]
        if not np.isnan(c) and c > best_c:
            best_c, best_k = c, k
    return best_k

def roll_with_nan(x: np.ndarray, k: int) -> np.ndarray:
    y = np.full_like(x, np.nan)
    if k == 0:
        return x.copy()
    if k > 0:
        y[k:] = x[:-k]
    else:
        y[:k] = x[-k:]
    return y

def build_windows(ts, acc, gyr, E_acc, E_gyr, T=512, stride=256):
    N = ts.shape[0]
    starts = list(range(0, max(N-T+1, 0), stride))
    nW = len(starts)
    TS_IMU = np.zeros((nW, T), dtype=np.float64)
    X_ACC  = np.zeros((nW, T, 3), dtype=np.float32)
    X_GYR  = np.zeros((nW, T, 3), dtype=np.float32)
    E2A    = np.zeros((nW, T, 3), dtype=np.float32)
    E2G    = np.zeros((nW, T, 3), dtype=np.float32)
    MASK   = np.ones((nW, T), dtype=np.uint8)
    for k,s in enumerate(starts):
        sl = slice(s, s+T)
        TS_IMU[k] = ts[sl]
        X_ACC[k]  = acc[sl]
        X_GYR[k]  = gyr[sl]
        E2A[k]    = E_acc[sl]**2
        E2G[k]    = E_gyr[sl]**2
        # MASK 可按需要做边界或缺测；此处全 1
    return TS_IMU, X_ACC, X_GYR, E2A, E2G, MASK

def process_seq(seq_root, out_dir, T=512, stride=256, g=9.80665, g_sign=-1, use_gt_bias=False,
                acc_smooth=7, mask_edge=3, est_tau=0,
                gyr_smooth=0, gyr_mask_edge=0, w_logmap=True):
    seq_root = Path(seq_root)
    imu_csv = seq_root/"mav0"/"imu0"/"data.csv"

    # 优先使用 state_groundtruth_estimate0，回退到其他GT源
    state_gt_csv = seq_root/"mav0"/"state_groundtruth_estimate0"/"data.csv"
    gt_candidates = [
        state_gt_csv,
        seq_root/"mav0"/"state_groundtruth0"/"data.csv",
        seq_root/"mav0"/"vicon0"/"data.csv",
        seq_root/"mav0"/"leica0"/"data.csv",   # 以防某些整理版
    ]
    gt_csv = next((p for p in gt_candidates if p.exists()), None)

    if not imu_csv.exists() or gt_csv is None:
        print(f"[skip] {seq_root}  (imu or gt missing)")
        return

    t_imu, w_meas, a_meas = load_imu_csv(imu_csv)
    
    # 优先使用新的GT加载器
    if USE_NEW_GT and state_gt_csv.exists():
        print(f"[info] Using state_groundtruth_estimate0 for {seq_root.name}")
        gt = load_state_gt(state_gt_csv)
        
        # 陀螺监督
        if w_logmap:
            # 在 IMU 时间轴上由姿态增量求 omega_gt（Log 映射 + 中央差分）
            q_i_log = interp_quat(t_imu, gt['t'], gt['q'])
            w_gt = omega_from_quat_cdiff(q_i_log, t_imu, step=1)
            # 平滑
            if int(gyr_smooth) > 1:
                w_gt = moving_average_1d(w_gt, int(gyr_smooth))
            # 边界与NaN mask
            mask_gyr = np.all(np.isfinite(w_gt), axis=1)
            if int(gyr_mask_edge) > 0:
                m = int(gyr_mask_edge)
                mask_gyr[:m] = False
                mask_gyr[-m:] = False
            # 整数样本延迟对齐
            if int(est_tau) > 0:
                best_k = best_shift_by_corr(w_meas, w_gt, int(est_tau))
                if best_k != 0:
                    w_gt = roll_with_nan(w_gt, best_k)
                    mask_gyr &= np.all(np.isfinite(w_gt), axis=1)
        else:
            # 回退：插值 GT 的 omega 到 IMU 轴
            w_gt = interp_vec(gt['t'], gt['w'], t_imu)
            mask_gyr = np.ones_like(t_imu, dtype=bool)
        if use_gt_bias and gt['bw'] is not None:
            bw = interp_vec(gt['t'], gt['bw'], t_imu)
            w_err = (w_meas - bw) - w_gt
        else:
            w_err = w_meas - w_gt
        e_gyr = w_err
        
        # 加计监督 - 采用 make_acc_body_gt（平滑+中央差分+重力+体轴）
        a_pred_body, mask_acc = make_acc_body_gt(gt, t_imu, smooth_win=acc_smooth, g_sign=g_sign, edge_mask=mask_edge)
        
        if use_gt_bias and gt['ba'] is not None:
            ba = interp_vec(gt['t'], gt['ba'], t_imu)
            a_err = (a_meas - ba) - a_pred_body
        else:
            a_err = a_meas - a_pred_body
        e_acc = a_err
        
        used_method = "state_groundtruth_estimate0"
    else:
        # 回退到原始方法
        print(f"[info] Using legacy GT method for {seq_root.name}")
        t_gt, p_gt, q_gt, v_gt = load_gt_csv(gt_csv)
        # 插值到 IMU 时刻
        q_i  = interp_quat(t_imu, t_gt, q_gt)         # 姿态（世界->体）
        v_i  = interp_linear(t_imu, t_gt, v_gt)       # 线速度（世界系）
        a_w  = central_diff(v_i, t_imu)               # 世界加速度
        R_i  = R_from_q(q_i)                          # R_wb: 世界->体
        g_world = np.array([0,0,g_sign*g], dtype=np.float64)  # z-up, 向下负号
        # 预测的体坐标加速度（不含噪声）
        a_pred_body = np.einsum('bij,bj->bi', np.transpose(R_i, (0,2,1)), a_w - g_world)
        # 预测的体坐标角速度（由姿态增量）
        w_pred_body = quat_to_omega_body(q_i, t_imu)
        
        e_acc = a_meas - a_pred_body
        e_gyr = w_meas - w_pred_body
        
        used_method = "legacy"

    TS_IMU, X_ACC, X_GYR, E2A, E2G, MASK = build_windows(t_imu, a_meas, w_meas, e_acc, e_gyr, T=T, stride=stride)
    # 把 ACC/GYR 的边界 mask 融合进窗口 MASK
    starts = list(range(0, max(len(t_imu)-T+1, 0), stride))
    have_acc = ('mask_acc' in locals()) and isinstance(mask_acc, np.ndarray)
    have_gyr = ('mask_gyr' in locals()) and isinstance(mask_gyr, np.ndarray)
    if have_acc or have_gyr:
        for k, s in enumerate(starts):
            sl = slice(s, s+T)
            cur = MASK[k]
            if have_acc and sl.stop <= len(mask_acc):
                cur = cur & (mask_acc[sl].astype(np.uint8))
            if have_gyr and sl.stop <= len(mask_gyr):
                cur = cur & (mask_gyr[sl].astype(np.uint8))
            MASK[k] = cur

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (seq_root.name + f"_T{T}_S{stride}.npz")
    np.savez(out,
        TS_IMU=TS_IMU,
        X_IMU_ACC=X_ACC, X_IMU_GYR=X_GYR,
        ERR_IMU_ACC=e_acc.reshape(1,-1,3)[:, :1, :],  # 防止某些加载器要求键存在；真正用 E2*
        ERR_IMU_GYR=e_gyr.reshape(1,-1,3)[:, :1, :],
        E2_IMU_ACC=E2A, E2_IMU_GYR=E2G,
        MASK_IMU=MASK
    )
    # 也可写一个汇总 JSON
    meta = {
        "seq": seq_root.name, "T": T, "stride": stride,
        "imu_len": int(len(t_imu)), "n_windows": int(TS_IMU.shape[0]),
        "g_sign": g_sign, "g": g, "use_gt_bias": use_gt_bias, "method": used_method
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote:", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc_root", required=True, help="EUROC 根目录，下面是 MH_01_easy 等子目录")
    ap.add_argument("--seqs", type=str, default="MH_01_easy,MH_02_easy,MH_03_medium,MH_04_difficult,MH_05_difficult",
                    help="逗号分隔的序列名")
    ap.add_argument("--out", required=True, help="输出目录（会保存多个 .npz）")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--g_sign", type=int, default=-1, help="z-up 世界，重力向下 => -1；如你的世界系向下为正可用 +1")
    ap.add_argument("--use_gt_bias", action="store_true", help="是否使用GT中的bias进行补偿（需要保证训练/评测一致）")
    ap.add_argument('--acc_smooth', type=int, default=7, help='moving-average window (odd); 0/1 to disable')
    ap.add_argument('--mask_edge', type=int, default=3, help='mask this many samples at both ends for ACC GT')
    ap.add_argument('--est_tau', type=int, default=0, help='estimate constant IMU-GT lag (in samples); 0=off')
    # Gyro options
    ap.add_argument('--gyr_smooth', type=int, default=0, help='moving-average window for gyro GT (odd, e.g., 5/7)')
    ap.add_argument('--gyr_mask_edge', type=int, default=0, help='mask N samples at both ends for gyro')
    ap.add_argument('--w_logmap', action='store_true', default=True, help='use SO(3) log-map on IMU axis to get omega_gt')
    ap.add_argument('--no_w_logmap', dest='w_logmap', action='store_false')
    args = ap.parse_args()
    root = Path(args.euroc_root)
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    for s in seqs:
        process_seq(root/s, args.out, T=args.T, stride=args.stride,
                    g_sign=args.g_sign, use_gt_bias=args.use_gt_bias,
                    acc_smooth=args.acc_smooth, mask_edge=args.mask_edge, est_tau=args.est_tau,
                    gyr_smooth=args.gyr_smooth, gyr_mask_edge=args.gyr_mask_edge, w_logmap=args.w_logmap)

if __name__ == "__main__":
    main()
