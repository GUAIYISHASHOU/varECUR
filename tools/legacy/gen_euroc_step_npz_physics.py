#!/usr/bin/env python3
"""
使用物理方法生成EuRoC IMU标签NPZ
关键改进：标签 = 测量值 - Ground Truth物理真值
增强功能：
  - 整数样本延迟估计与对齐 (--est_tau)
  - 可选使用GT bias (--use_gt_bias)
  - 陀螺仪平滑与边界掩码 (--gyr_smooth, --gyr_mask_edge)
"""
from pathlib import Path
import argparse, csv, json
import numpy as np
import sys

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.euroc_state_gt import load_state_gt, make_acc_body_gt
from common.attitude import slerp_many, quat_to_omega_body
from common.resample import interp_lin, smooth_mavg

# ============ 时间延迟对齐工具函数 ============

def best_shift_by_corr(a: np.ndarray, b: np.ndarray, max_k: int) -> int:
    """
    通过互相关估计整数样本延迟，使 |a| 和 |b| 的相关性最大化
    返回 k，表示将 b 向前移动 k 个样本 (b[t] -> b[t+k])
    
    Args:
        a, b: shape (N, D)，可能包含 NaN
        max_k: 最大搜索延迟（样本数）
    Returns:
        best_k: 最佳延迟值
    """
    if max_k is None or max_k <= 0:
        return 0
    
    # 计算模长
    a_mag = np.linalg.norm(a, axis=1)
    b_mag = np.linalg.norm(b, axis=1)
    
    # 处理 NaN
    a_mag = np.where(np.isfinite(a_mag), a_mag, 0.0)
    b_mag = np.where(np.isfinite(b_mag), b_mag, 0.0)
    
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
        
        if len(aa) < 4:  # 太短无意义
            continue
        
        # 零均值归一化（Pearson相关）
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        denom = (np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom <= 0:
            c = -1.0
        else:
            c = float(np.dot(aa, bb) / denom)
        
        if c > best_c:
            best_c = c
            best_k = k
    
    return best_k


def roll_with_nan(x: np.ndarray, k: int) -> np.ndarray:
    """
    沿 axis=0 滚动数组，但用 NaN 填充空缺部分（而非循环）
    
    Args:
        x: 输入数组
        k: 滚动量（正数向前，负数向后）
    Returns:
        滚动后的数组
    """
    y = np.full_like(x, np.nan)
    if k == 0:
        return x.copy()
    if k > 0:
        y[k:] = x[:-k]
    else:
        y[:k] = x[-k:]
    return y

# ============ 数据加载 ============

def load_imu_csv(csv_path):
    """加载IMU CSV：timestamp[ns], wx, wy, wz [rad/s], ax, ay, az [m/s^2]"""
    t, gyr, acc = [], [], []
    with open(csv_path, 'r', newline='') as f:
        rd = csv.reader(f)
        for row in rd:
            if not row or row[0].startswith('#') or row[0] == 'timestamp':
                continue
            try:
                ts = int(row[0]) * 1e-9  # ns -> s
                w = [float(row[i]) for i in [1,2,3]]
                a = [float(row[i]) for i in [4,5,6]]
                t.append(ts); gyr.append(w); acc.append(a)
            except (ValueError, IndexError):
                continue
    t = np.asarray(t, dtype=np.float64)
    gyr = np.asarray(gyr, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    
    # 确保时间戳严格递增
    keep = np.r_[True, np.diff(t) > 0.0]
    return t[keep], gyr[keep], acc[keep]

def make_windows(t, acc, gyr, e_acc, e_gyr, T=512, stride=256):
    """滑动窗口切分"""
    N = len(t)
    starts = list(range(0, max(N - T + 1, 0), stride))
    nW = len(starts)
    
    TS = np.zeros((nW, T))
    X_ACC = np.zeros((nW, T, 3), dtype=np.float32)
    X_GYR = np.zeros((nW, T, 3), dtype=np.float32)
    E2_ACC = np.zeros((nW, T, 3), dtype=np.float32)
    E2_GYR = np.zeros((nW, T, 3), dtype=np.float32)
    MASK = np.ones((nW, T), dtype=np.uint8)
    
    for k, s in enumerate(starts):
        sl = slice(s, s + T)
        TS[k] = t[sl]
        X_ACC[k] = acc[sl]
        X_GYR[k] = gyr[sl]
        E2_ACC[k] = e_acc[sl]**2
        E2_GYR[k] = e_gyr[sl]**2
    
    return TS, X_ACC, X_GYR, E2_ACC, E2_GYR, MASK

def process_sequence(seq_root, out_dir, T=512, stride=256, g_sign=-1, 
                     acc_smooth=7, edge_mask=3,
                     est_tau=2, use_gt_bias=False, gyr_smooth=0, gyr_mask_edge=0):
    """
    处理单个EuRoC序列，生成物理标签
    
    新增参数：
        est_tau: 最大整数样本延迟搜索范围（0=禁用）
        use_gt_bias: 是否使用GT bias（bw/ba）
        gyr_smooth: 陀螺仪GT平滑窗口（奇数，<=1禁用）
        gyr_mask_edge: 陀螺仪边界掩码样本数
    """
    seq_root = Path(seq_root)
    imu_csv = seq_root / 'mav0' / 'imu0' / 'data.csv'
    
    # 优先使用 state_groundtruth_estimate0
    gt_candidates = [
        seq_root / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv',
        seq_root / 'mav0' / 'state_groundtruth0' / 'data.csv',
        seq_root / 'mav0' / 'vicon0' / 'data.csv',
        seq_root / 'mav0' / 'leica0' / 'data.csv',
    ]
    gt_csv = next((p for p in gt_candidates if p.exists()), None)
    
    if not imu_csv.exists() or gt_csv is None:
        print(f"[SKIP] {seq_root.name}: IMU或GT缺失")
        return
    
    print(f"[{seq_root.name}] build step-labels (T={T}, S={stride})")
    
    # 1. 加载IMU测量值
    t_imu, w_meas, a_meas = load_imu_csv(imu_csv)
    
    # 2. 加载Ground Truth
    gt = load_state_gt(gt_csv)
    
    # 3. 计算加速度计真值（物理方法）
    a_gt_body, mask_acc = make_acc_body_gt(
        gt, t_imu, 
        smooth_win=acc_smooth, 
        g_sign=g_sign, 
        edge_mask=edge_mask
    )
    
    # 可选：使用GT bias（加速度计）
    if use_gt_bias and ('ba' in gt) and (gt['ba'] is not None):
        ba_i = interp_lin(gt['t'], gt['ba'], t_imu)
        e_acc = (a_meas - ba_i) - a_gt_body
    else:
        e_acc = a_meas - a_gt_body  # ✅ 真实误差
    
    # 4. 计算陀螺仪真值（SO(3) Log映射）
    q_imu = slerp_many(gt['t'], gt['q'], t_imu)
    w_gt_body = quat_to_omega_body(q_imu, t_imu, dt_min=1e-4)
    
    # 可选：陀螺仪GT平滑
    gwin = int(gyr_smooth)
    if gwin > 1:
        if gwin % 2 == 0:
            gwin += 1
        w_gt_body = smooth_mavg(w_gt_body, win=gwin)
    
    # 边界mask（陀螺仪）
    mask_gyr = np.isfinite(w_gt_body).all(axis=1)
    if int(gyr_mask_edge) > 0:
        m = int(gyr_mask_edge)
        mask_gyr[:m] = False
        mask_gyr[-m:] = False
    elif edge_mask > 0:  # 回退到通用边界掩码
        mask_gyr[:edge_mask] = False
        mask_gyr[-edge_mask:] = False
    
    # ✅ 关键增强：整数样本延迟对齐
    K = int(est_tau)
    if K > 0:
        k_best = best_shift_by_corr(w_meas, w_gt_body, K)
        if k_best != 0:
            print(f"  [lag-align] shift gyro GT by {k_best} samples")
            w_gt_body = roll_with_nan(w_gt_body, k_best)
            mask_gyr &= np.isfinite(w_gt_body).all(axis=1)
    
    # 可选：使用GT bias（陀螺仪）
    if use_gt_bias and ('bw' in gt) and (gt['bw'] is not None):
        bw_i = interp_lin(gt['t'], gt['bw'], t_imu)
        e_gyr = (w_meas - bw_i) - w_gt_body
    else:
        e_gyr = w_meas - w_gt_body  # ✅ 真实误差
    
    # 统一mask（AND操作）
    mask_all = (mask_acc.astype(bool) & mask_gyr.astype(bool))
    # 将无效样本的误差置零（它们会被mask掉）
    e_acc[~mask_all] = 0.0
    e_gyr[~mask_all] = 0.0
    
    # 5. 生成滑动窗口
    TS, X_ACC, X_GYR, E2_ACC, E2_GYR, MASK = make_windows(
        t_imu, a_meas.astype(np.float32), w_meas.astype(np.float32), 
        e_acc.astype(np.float32), e_gyr.astype(np.float32), 
        T=T, stride=stride
    )
    
    # 6. 融合mask（基于统一的mask_all）
    starts = list(range(0, max(len(t_imu) - T + 1, 0), stride))
    for k, s in enumerate(starts):
        sl = slice(s, s + T)
        if sl.stop <= len(mask_all):
            MASK[k] &= mask_all[sl].astype(np.uint8)
    
    # 7. 保存
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_root.name}_T{T}_S{stride}.npz"
    
    np.savez_compressed(
        out_path,
        TS_IMU=TS,
        X_IMU_ACC=X_ACC,
        X_IMU_GYR=X_GYR,
        E2_IMU_ACC=E2_ACC,
        E2_IMU_GYR=E2_GYR,
        MASK_IMU=MASK,
        ERR_IMU_ACC=e_acc.reshape(1, -1, 3)[:, :1, :],
        ERR_IMU_GYR=e_gyr.reshape(1, -1, 3)[:, :1, :],
    )

    meta = dict(seq=seq_root.name, T=int(T), stride=int(stride),
                imu_len=int(len(t_imu)), n_windows=int(TS.shape[0]),
                g_sign=int(g_sign), use_gt_bias=bool(use_gt_bias),
                acc_smooth=int(acc_smooth), mask_edge=int(edge_mask),
                gyr_smooth=int(gyr_smooth), gyr_mask_edge=int(gyr_mask_edge),
                est_tau=int(est_tau))
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  ✓ wrote: {out_path.name}  (nW={TS.shape[0]})")

def main():
    parser = argparse.ArgumentParser(description="Generate physics-based IMU labels (enhanced)")
    parser.add_argument('--euroc_root', required=True, help='EuRoC dataset root')
    parser.add_argument('--seqs', nargs='+', required=True, help='Sequence names')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--T', type=int, default=512, help='Window length (final uses 512)')
    parser.add_argument('--stride', type=int, default=256, help='Window stride (final uses 256)')
    parser.add_argument('--g_sign', type=int, default=-1, help='Gravity sign (+/-1)')
    parser.add_argument('--acc_smooth', type=int, default=7, help='ACC smoothing window (odd, <=1 to disable)')
    parser.add_argument('--edge_mask', type=int, default=3, help='ACC edge masking samples')
    parser.add_argument('--mask_edge', type=int, default=None, help='同 --edge_mask（别名）')
    
    # 新增参数（与imuFINAL对齐）
    parser.add_argument('--est_tau', type=int, default=2, 
                        help='Max integer-sample delay to search via correlation (0=disable, recommend 1-3)')
    parser.add_argument('--use_gt_bias', action='store_true', 
                        help='Use GT bias bw/ba if available')
    parser.add_argument('--gyr_smooth', type=int, default=0, 
                        help='GYR GT smoothing window (odd, <=1 to disable)')
    parser.add_argument('--gyr_mask_edge', type=int, default=0, 
                        help='Mask N samples at both ends for gyro GT')
    
    args = parser.parse_args()
    edge_mask = args.edge_mask if args.mask_edge is None else int(args.mask_edge)
    
    root = Path(args.euroc_root)
    
    for seq_name in args.seqs:
        seq_path = root / seq_name
        if not seq_path.exists():
            print(f"[SKIP] {seq_name}: 目录不存在")
            continue
        
        process_sequence(
            seq_path, args.out_dir, 
            T=args.T, stride=args.stride,
            g_sign=args.g_sign,
            acc_smooth=args.acc_smooth,
            edge_mask=edge_mask,
            est_tau=args.est_tau,
            use_gt_bias=args.use_gt_bias,
            gyr_smooth=args.gyr_smooth,
            gyr_mask_edge=args.gyr_mask_edge
        )
    
    print("\n✓ All done")

if __name__ == '__main__':
    main()
