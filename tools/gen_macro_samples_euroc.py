# -*- coding: utf-8 -*-
# tools/gen_macro_samples_euroc.py
"""
把 EuRoC 每个帧对(i, i+Δ)的所有点匹配聚成一个"宏观样本"
输出：train/val/test 三个 npz（或单个npz由 merge 脚本再切分）
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# 导入现有的工具函数
sys.path.insert(0, str(Path(__file__).parent.parent))

def robust_var_mad(err):
    """使用 MAD (Median Absolute Deviation) 估计稳健方差"""
    err = np.asarray(err).astype(np.float64)
    if err.size == 0: return 1e-12
    med = np.median(err)
    mad = np.median(np.abs(err - med))
    sigma = 1.4826 * mad
    return max(1e-12, float(sigma**2))

def sample_or_pad(arr, K, pad_value=0):
    """采样或填充数组到固定长度K"""
    N = len(arr)
    if N >= K:
        idx = np.random.choice(N, K, replace=False)
        return arr[idx], N
    pad = K - N
    if arr.ndim == 1:
        return np.pad(arr, (0,pad), constant_values=pad_value), N
    if arr.ndim == 2:
        return np.pad(arr, ((0,pad),(0,0)), constant_values=pad_value), N
    if arr.ndim == 4:
        return np.pad(arr, ((0,pad),(0,0),(0,0),(0,0)), constant_values=pad_value), N
    raise ValueError("unexpected ndim")

def median_pixel_parallax(uv1, uv2):
    """计算两组像素点之间的中位数视差"""
    if uv1 is None or uv2 is None or len(uv1) == 0:
        return 0.0
    diff = uv2 - uv1
    parallax = np.linalg.norm(diff, axis=1)
    return float(np.median(parallax))

# ===== 从 gen_vis_pairs_euroc_strict.py 复制核心函数 =====
def read_cam_csv(csv_path):
    """Read camera timestamps and image paths from EuRoC CSV."""
    rows = []
    with open(csv_path, 'r') as f:
        for r in f:
            if not r or r[0] == '#' or 'timestamp' in r: 
                continue
            ts, rel = r.strip().split(',')
            rows.append((int(ts), rel))
    ts = np.array([r[0] for r in rows], np.int64)
    files = [Path(csv_path).parent / 'data' / r[1] for r in rows]
    return ts, files

def quat_to_R(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    import math
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)
    return R

def SE3(R, t):
    """Construct 4x4 SE3 matrix from R and t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invSE3(T):
    """Invert SE3 transformation."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def read_gt_states(gt_csv):
    """Read ground-truth states from EuRoC CSV."""
    import csv
    with open(gt_csv, 'r') as f:
        rdr = csv.reader(f)
        head = next(rdr)
        idx = {name: i for i, name in enumerate(head)}
        
        def get_idx(keys):
            for k in keys:
                if k in idx: return idx[k]
            raise KeyError(f"None of {keys} found in CSV header")
        
        t_i = get_idx(['#timestamp [ns]', '#timestamp', 'timestamp'])
        px_i = get_idx([' p_RS_R_x [m]', 'p_RS_R_x [m]', 'p_RS_R_x'])
        py_i = get_idx([' p_RS_R_y [m]', 'p_RS_R_y [m]', 'p_RS_R_y'])
        pz_i = get_idx([' p_RS_R_z [m]', 'p_RS_R_z [m]', 'p_RS_R_z'])
        qw_i = get_idx([' q_RS_w []', 'q_RS_w []', 'q_RS_w'])
        qx_i = get_idx([' q_RS_x []', 'q_RS_x []', 'q_RS_x'])
        qy_i = get_idx([' q_RS_y []', 'q_RS_y []', 'q_RS_y'])
        qz_i = get_idx([' q_RS_z []', 'q_RS_z []', 'q_RS_z'])
        
        T, P, Q = [], [], []
        for row in rdr:
            if not row or row[0].startswith('#'): continue
            T.append(int(row[t_i]))
            P.append([float(row[px_i]), float(row[py_i]), float(row[pz_i])])
            Q.append([float(row[qw_i]), float(row[qx_i]), float(row[qy_i]), float(row[qz_i])])
    
    return np.array(T, np.int64), np.array(P, np.float64), np.array(Q, np.float64)

def interp_pose(ts_all, T_gt, P, Q):
    """Interpolate poses at given timestamps."""
    import math
    def slerp(q1, q2, alpha):
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        if dot < 0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            q = (1 - alpha) * q1 + alpha * q2
            return q / np.linalg.norm(q)
        theta0 = math.acos(dot)
        s0 = math.sin((1 - alpha) * theta0)
        s1 = math.sin(alpha * theta0)
        s = math.sin(theta0)
        return (s0 * q1 + s1 * q2) / s
    
    out = []
    for t in ts_all:
        i = np.searchsorted(T_gt, t)
        if i <= 0:
            i0 = i1 = 0
            a = 0.0
        elif i >= len(T_gt):
            i0 = i1 = len(T_gt) - 1
            a = 0.0
        else:
            i0, i1 = i - 1, i
            a = (t - T_gt[i0]) / (T_gt[i1] - T_gt[i0] + 1e-12)
        p = (1 - a) * P[i0] + a * P[i1]
        q = slerp(Q[i0], Q[i1], a)
        out.append(SE3(quat_to_R(*q), p))
    return out

def load_cam_yaml(yaml_path):
    """Load camera intrinsics, distortion, and extrinsics from EuRoC YAML."""
    import yaml
    Y = yaml.safe_load(open(yaml_path, 'r'))
    if 'intrinsics' in Y:
        fx, fy, cx, cy = Y['intrinsics']
    else:
        fx = Y['camera_matrix']['data'][0]
        fy = Y['camera_matrix']['data'][4]
        cx = Y['camera_matrix']['data'][2]
        cy = Y['camera_matrix']['data'][5]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    
    if 'distortion_coefficients' in Y:
        if isinstance(Y['distortion_coefficients'], dict):
            D = np.array(Y['distortion_coefficients']['data'], np.float64)
        else:
            D = np.array(Y['distortion_coefficients'], np.float64)
    elif 'distortion_parameters' in Y:
        D = np.array(list(Y['distortion_parameters'].values()), np.float64)
    else:
        D = np.zeros((5,), np.float64)
    
    T_BS = np.eye(4)
    if 'T_BS' in Y:
        data = Y['T_BS']['data'] if isinstance(Y['T_BS'], dict) else Y['T_BS']
        T_BS = np.array(data, dtype=np.float64).reshape(4, 4)
    
    return K, D, T_BS

def undist_norm(K, D, pts):
    """Undistort points and convert to normalized coordinates."""
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, D)
    return und.reshape(-1, 2)

def triangulate(K0, K1, T_cam1_cam0, x0n, x1n):
    """Triangulate 3D points from stereo normalized coordinates."""
    R10 = T_cam1_cam0[:3, :3]
    t10 = T_cam1_cam0[:3, 3:4]
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R10, t10])
    X_h = cv2.triangulatePoints(P0, P1, x0n.T, x1n.T)
    X = (X_h[:3] / (X_h[3] + 1e-12)).T
    return X

def project(K, Xc):
    """Project 3D points in camera frame to image pixels."""
    x = (Xc[:, :2] / (Xc[:, 2:3] + 1e-12))
    uv = (K[:2, :2] @ x.T + K[:2, 2:3]).T
    return uv

def crop_patches(img, pts, patch=32):
    """Crop square patches around given points."""
    H, W = img.shape[:2]
    h = patch // 2
    out, ok = [], []
    for (u, v) in pts:
        x = int(round(u))
        y = int(round(v))
        if x - h < 0 or y - h < 0 or x + h >= W or y + h >= H:
            ok.append(False)
            continue
        out.append(img[y - h:y + h, x - h:x + h].copy())
        ok.append(True)
    return np.array(out, np.uint8), np.array(ok, bool)

def orb_det_desc(img, n=1200):
    """Detect and describe ORB features."""
    orb = cv2.ORB_create(nfeatures=n, fastThreshold=7)
    kps, des = orb.detectAndCompute(img, None)
    if des is None or len(kps) < 16:
        return None, None, None
    pts = np.array([k.pt for k in kps], np.float64)
    sc = np.array([k.response for k in kps], np.float32)
    return pts, des, sc

def match_bf(d0, d1):
    """Brute-force matching with cross-check."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d0, d1)
    matches = sorted(matches, key=lambda r: r.distance)
    return np.array([[m.queryIdx, m.trainIdx] for m in matches], int)

# ===== 新增：辅助函数用于高级特征 =====
def skew3(v):
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)

def sampson_err_norm(x1n, x0n, R, t):
    """
    计算 Sampson (一阶几何) 极线误差
    Args:
        x1n, x0n: [N,2] 归一化坐标（已去畸变）
        R, t: 相对旋转和平移 (cam1 ← cam0)
    Returns:
        err: [N] Sampson误差
    """
    x0 = np.hstack([x0n, np.ones((x0n.shape[0], 1))])
    x1 = np.hstack([x1n, np.ones((x1n.shape[0], 1))])
    E = skew3(t) @ R  # 本质矩阵
    Ex0 = (E @ x0.T).T
    ETx1 = (E.T @ x1.T).T
    x1Ex0 = np.sum(x1 * Ex0, axis=1)
    num = x1Ex0 ** 2
    den = Ex0[:, 0]**2 + Ex0[:, 1]**2 + ETx1[:, 0]**2 + ETx1[:, 1]**2 + 1e-12
    return np.sqrt(num / den)

def compute_geom_features_v2(xy0_px, xy2_px, x0n, x2n, X_c0, T_c2_c0, P0, P2, delta_t, K_tokens):
    """
    计算完整的20维几何与图像特征向量
    Args:
        xy0_px, xy2_px: 像素坐标 [N,2]
        x0n, x2n: 归一化坐标 [N,2]
        X_c0: 3D点在cam0坐标系 [N,3]
        T_c2_c0: 相对位姿变换 [4,4]
        P0, P2: 图像patch [N,H,W]
        delta_t: 时间间隔（秒）
        K_tokens: 最大token数
    Returns:
        feat: [N, 20] 特征向量
    """
    N = len(xy0_px)
    R, t = T_c2_c0[:3,:3], T_c2_c0[:3,3]

    # 1-4. (u1,v1,u2,v2) 归一化坐标
    u1n, v1n = x0n[:, 0], x0n[:, 1]
    u2n, v2n = x2n[:, 0], x2n[:, 1]

    # 5-6. 到各自图像中心的半径
    r1 = np.sqrt(u1n**2 + v1n**2)
    r2 = np.sqrt(u2n**2 + v2n**2)

    # 7. 视差像素 (clipped)
    parallax_px = np.linalg.norm(xy2_px - xy0_px, axis=1)
    parallax_px = np.clip(parallax_px, 0, 40)

    # 8. 视差角 (clipped)
    v1_dir = np.hstack([x0n, np.ones((N, 1))])
    v1_dir /= np.linalg.norm(v1_dir, axis=1, keepdims=True)
    X_c2 = (R @ X_c0.T + t[:, None]).T
    v2_dir = X_c2 / (np.linalg.norm(X_c2, axis=1, keepdims=True) + 1e-12)
    cos_parallax = np.clip(np.sum(v1_dir * v2_dir, axis=1), -1.0, 1.0)
    parallax_deg = np.rad2deg(np.arccos(cos_parallax))
    parallax_deg = np.clip(parallax_deg, 0, 10)

    # 9. 深度倒数 (1/z)
    depth = X_c0[:, 2]
    inv_depth = 1.0 / np.clip(depth, 0.1, 100.0)
    inv_depth[depth < 0.1] = 0  # 处理无效深度

    # 10. 视线与基线夹角余弦
    baseline_dir = t / (np.linalg.norm(t) + 1e-9)
    cos_theta = np.clip(np.sum(v1_dir * baseline_dir, axis=1), -1.0, 1.0)

    # 11-14. 梯度与角点强度 (patch均值)
    grad1, grad2, corn1, corn2 = [], [], [], []
    for p in P0:
        gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
        grad1.append(np.mean(np.sqrt(gx*gx + gy*gy)))
        corn1.append(np.mean(cv2.cornerMinEigenVal(p, blockSize=3, ksize=3)))
    for p in P2:
        gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
        grad2.append(np.mean(np.sqrt(gx*gx + gy*gy)))
        corn2.append(np.mean(cv2.cornerMinEigenVal(p, blockSize=3, ksize=3)))

    # 15. 尺度变化 proxy
    scale_change = np.log(r2 / (r1 + 1e-9) + 1e-9)

    # 16. Sampson/极线残差
    sampson = sampson_err_norm(x2n, x0n, R, t)

    # 17. token rank/K
    token_rank = np.arange(N) / float(K_tokens)

    # 18. Δt (normalized)
    delta_t_norm = np.full(N, fill_value=delta_t, dtype=np.float32)

    # 19-20. 曝光/亮度差 proxy
    mean_lum1 = np.array([p.mean() for p in P0])
    mean_lum2 = np.array([p.mean() for p in P2])

    # Stack all 20 features
    feat = np.stack([
        u1n, v1n, u2n, v2n,                     # 1-4
        r1, r2,                                 # 5-6
        parallax_px, parallax_deg,              # 7-8
        inv_depth, cos_theta,                   # 9-10
        np.array(grad1), np.array(grad2),       # 11-12
        np.array(corn1), np.array(corn2),       # 13-14
        scale_change, sampson,                  # 15-16
        token_rank, delta_t_norm,               # 17-18
        mean_lum1, mean_lum2,                   # 19-20
    ], axis=1).astype(np.float32)

    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc_root", type=str, required=True)
    ap.add_argument("--seqs", type=str, nargs="+", required=True)
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--deltas", type=int, nargs="+", default=[1])
    ap.add_argument("--frame_step", type=int, default=2)
    ap.add_argument("--K_tokens", type=int, default=256)
    ap.add_argument("--min_matches", type=int, default=24)
    ap.add_argument("--err_clip_px", type=float, default=20.0,
                    help="Robust tail clipping threshold in pixels")
    ap.add_argument("--pos_thr_px", type=float, default=3.0,
                    help="Base positive/negative decision threshold (pixels)")
    ap.add_argument("--pos_thr_px_v1", type=float, default=3.0,
                    help="Positive threshold for V1 sequences")
    ap.add_argument("--pos_thr_px_v2", type=float, default=3.0,
                    help="Positive threshold for V2 sequences")
    ap.add_argument("--pos_thr_px_mh", type=float, default=3.0,
                    help="Positive threshold for MH sequences")
    ap.add_argument("--inlier_thr_px", type=float, default=2.0,
                    help="Threshold for inlier label: median GT reprojection error < this = inlier")
    
    # 关键帧选取参数（IC-GVINS 对齐模式）
    kf = ap.add_argument_group("Keyframe Selection (IC-GVINS aligned)")
    kf.add_argument("--kf_enable", action="store_true",
                    help="Enable keyframe selection mode (IC-GVINS aligned)")
    kf.add_argument("--kf_parallax_px", type=float, default=20.0,
                    help="Pixel parallax threshold for keyframe selection")
    kf.add_argument("--kf_max_interval_s", type=float, default=0.5,
                    help="Maximum time interval (seconds) between keyframes")
    kf.add_argument("--kf_min_interval_s", type=float, default=0.08,
                    help="Minimum time interval (seconds) between keyframes")
    kf.add_argument("--emit_non_kf_ratio", type=float, default=0.2,
                    help="Ratio of non-keyframe pairs to emit (for diversity)")
    
    args = ap.parse_args()

    # === 修改：增加一个列表来存储内点标签 ===
    patches_all, geoms_all, ytrue_all, numtok_all, yinlier_all = [], [], [], [], []

    for seq in args.seqs:
        seq_root = Path(args.euroc_root) / seq / "mav0"
        
        # Read camera and ground-truth data
        ts0, p0 = read_cam_csv(seq_root / "cam0" / "data.csv")
        ts1, p1 = read_cam_csv(seq_root / "cam1" / "data.csv")
        gt_t, gt_p, gt_q = read_gt_states(seq_root / "state_groundtruth_estimate0" / "data.csv")
        
        # Load camera parameters
        K0, D0, Timu_cam0 = load_cam_yaml(seq_root / "cam0" / "sensor.yaml")
        K1, D1, Timu_cam1 = load_cam_yaml(seq_root / "cam1" / "sensor.yaml")
        
        # Compute transformations
        Tcam0_imu = invSE3(Timu_cam0)
        Tcam1_imu = invSE3(Timu_cam1)
        Tcam1_cam0 = Tcam1_imu @ Timu_cam0
        
        # Align lengths
        max_delta = max(args.deltas)
        n = min(len(ts0), len(ts1)) - max_delta
        ts0 = ts0[:n]
        p0 = p0[:n]
        p1 = p1[:n]
        
        # Interpolate poses
        T_w_imu_all = interp_pose(ts0, gt_t, gt_p, gt_q)
        T_w_cam0_all = [Twi @ Timu_cam0 for Twi in T_w_imu_all]
        
        # 选择帧对：关键帧模式 或 默认间隔模式
        pairs_to_process = []  # [(i, j, pair_type, parallax)]
        
        if args.kf_enable:
            # ===== 关键帧选取模式（IC-GVINS 对齐）=====
            print(f"[{seq}] Using keyframe selection mode (IC-GVINS aligned)")
            
            # 预先计算所有帧的 ORB 特征（用于快速关键帧判定）
            print(f"[{seq}] Pre-computing ORB features for {n} frames...")
            orb_cache = {}
            for idx in tqdm(range(n), desc="ORB cache"):
                img = cv2.imread(str(p0[idx]), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    orb_cache[idx] = (None, None, None)
                    continue
                pts, des, sc = orb_det_desc(img)
                orb_cache[idx] = (pts, des, sc)
            
            # 关键帧选取
            last_kf = 0
            last_emit_ts = ts0[0] * 1e-9  # 转换为秒
            ts0_s = ts0 * 1e-9
            
            for j in range(1, n):
                dt_emit = ts0_s[j] - last_emit_ts
                if dt_emit < args.kf_min_interval_s:
                    continue
                
                # 获取缓存的特征
                pts_i, des_i, _ = orb_cache[last_kf]
                pts_j, des_j, _ = orb_cache[j]
                
                if des_i is None or des_j is None:
                    continue
                
                # 匹配特征点
                m_idx = match_bf(des_i, des_j)
                if len(m_idx) < args.min_matches:
                    continue
                
                # 计算视差
                uv_i = pts_i[m_idx[:, 0]]
                uv_j = pts_j[m_idx[:, 1]]
                para = median_pixel_parallax(uv_i, uv_j)
                
                dt_kf = ts0_s[j] - ts0_s[last_kf]
                
                # 判断是否为关键帧
                if para > args.kf_parallax_px:
                    # 满足视差阈值 -> 关键帧
                    pairs_to_process.append((last_kf, j, 1, para))  # pair_type=1表示KF
                    last_kf = j
                    last_emit_ts = ts0_s[j]
                elif dt_kf > args.kf_max_interval_s:
                    # 超过最大时间间隔 -> 强制作为观测帧
                    pairs_to_process.append((last_kf, j, 0, para))  # pair_type=0表示观测帧
                    last_emit_ts = ts0_s[j]
            
            # 添加一些非关键帧对以增加多样性
            num_kf = sum(1 for _, _, pt, _ in pairs_to_process if pt == 1)
            extra = int(num_kf * args.emit_non_kf_ratio)
            import random
            random.seed(42)
            cand = list(range(0, n - 5, 5))
            random.shuffle(cand)
            
            for idx in cand[:extra]:
                if idx + 1 >= n:
                    continue
                pts0, des0, _ = orb_cache[idx]
                pts1, des1, _ = orb_cache[idx + 1]
                if des0 is None or des1 is None:
                    continue
                m = match_bf(des0, des1)
                if len(m) < args.min_matches:
                    continue
                para = median_pixel_parallax(pts0[m[:, 0]], pts1[m[:, 1]])
                pairs_to_process.append((idx, idx + 1, 0, para))
            
            print(f"[{seq}] Selected {len(pairs_to_process)} pairs ({num_kf} KF, {len(pairs_to_process)-num_kf} obs)")
        else:
            # ===== 默认间隔帧模式 =====
            print(f"[{seq}] Using default delta/step mode")
            for i in range(0, n - max_delta - 1, args.frame_step):
                for d in args.deltas:
                    j = i + d
                    if j < n:
                        pairs_to_process.append((i, j, -1, -1.0))  # pair_type=-1表示默认模式
        
        # ===== 处理选定的帧对 =====
        for (i, j, pair_type, para_pair) in tqdm(pairs_to_process, desc=f"[{seq}] processing pairs"):
            # Read reference images
            I0 = cv2.imread(str(p0[i]), cv2.IMREAD_GRAYSCALE)
            I1 = cv2.imread(str(p1[i]), cv2.IMREAD_GRAYSCALE)
            
            if I0 is None or I1 is None:
                continue
            
            # Stereo matching @ t
            x0, d0, s0 = orb_det_desc(I0)
            x1, d1, s1 = orb_det_desc(I1)
            
            if x0 is None or x1 is None:
                continue
            
            idx = match_bf(d0, d1)
            if len(idx) < 40:
                continue
            
            x0m = x0[idx[:, 0]]
            x1m = x1[idx[:, 1]]
            
            # Undistort → normalize → triangulate
            x0n = undist_norm(K0, D0, x0m)
            x1n = undist_norm(K1, D1, x1m)
            X3d = triangulate(K0, K1, Tcam1_cam0, x0n, x1n)
            
            # 读取第二帧图像
            if j >= n:
                continue
            
            I2 = cv2.imread(str(p0[j]), cv2.IMREAD_GRAYSCALE)
            if I2 is None:
                continue
            
            # Compute relative pose
            T_w_cam0_i = T_w_cam0_all[i]
            T_w_cam0_j = T_w_cam0_all[j]
            T_cj_ci = invSE3(T_w_cam0_j) @ T_w_cam0_i
            
            R01 = T_cj_ci[:3, :3]
            t01 = T_cj_ci[:3, 3]
            Xj = (R01 @ X3d.T + t01.reshape(3, 1)).T
            
            # === 新增：保存GT位姿变换的3D点用于后续计算内点标签 ===
            Xj_gt = Xj.copy()
            
            # Temporal matching
            x2, d2, s2 = orb_det_desc(I2)
            if x2 is None:
                continue
            
            m02 = match_bf(d0, d2)
            
            # RANSAC filtering
            if m02 is not None and len(m02) >= 8:
                p0_px = x0[m02[:, 0]].astype(np.float32)
                p2_px = x2[m02[:, 1]].astype(np.float32)
                E, inlier = cv2.findEssentialMat(
                    p0_px, p2_px, cameraMatrix=K0,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                if inlier is not None:
                    m02 = m02[inlier.ravel().astype(bool)]
            
            if m02 is None or len(m02) < 20:
                continue
            
            q0 = x0[m02[:, 0]]
            u2_obs = x2[m02[:, 1]]
            
            # Align 3D points
            if len(q0) == 0 or len(Xj) == 0:
                continue
            
            diff = q0[:, None, :] - x0m[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            nn = np.argmin(dist, axis=1)
            thr = 4.0 if args.kf_enable else 2.5  # 关键帧模式下使用更宽松的阈值
            sel = (dist[np.arange(len(q0)), nn] < thr)
            
            if sel.sum() < 20:
                continue
            
            X_sel = Xj[nn[sel]]
            u2_obs_final = u2_obs[sel]
            
            # Predict and compute errors
            u2_pred = project(K0, X_sel)
            e = (u2_obs_final - u2_pred).astype(np.float32)
            e2 = e * e
            m = np.isfinite(e2).all(axis=1)
            
            if m.sum() == 0:
                continue
            
            # Crop patches
            P0, ok0 = crop_patches(I0, q0[sel], patch=args.patch)
            P2, ok2 = crop_patches(I2, u2_obs_final, patch=args.patch)
            
            # Filter by reprojection error (per-sequence adaptive threshold)
            err_px = np.sqrt(e2.sum(axis=1))
            seq_upper = seq.upper()
            pos_thr = args.pos_thr_px
            if seq_upper.startswith("V1"):
                pos_thr = min(pos_thr, args.pos_thr_px_v1)
            elif seq_upper.startswith("V2"):
                pos_thr = min(pos_thr, args.pos_thr_px_v2)
            elif seq_upper.startswith("MH"):
                pos_thr = max(pos_thr, args.pos_thr_px_mh)
            good = err_px < pos_thr
            keep = m & ok0 & ok2 & good
            
            if keep.sum() < args.min_matches:
                continue
            
            # === 关键修改：计算 y_inlier 标签（基于GT位姿） ===
            # 1. 找到keep中的点对应的原始3D点索引
            nn_kept = nn[sel][keep]
            
            # 2. 用GT位姿的3D点计算重投影
            X_gt_sel = Xj_gt[nn_kept]
            u2_pred_gt = project(K0, X_gt_sel)
            
            # 3. 计算GT重投影误差
            u2_obs_kept = u2_obs_final[keep]
            e_gt = u2_obs_kept - u2_pred_gt
            e_norm_gt = np.linalg.norm(e_gt, axis=1)
            
            # 4. 定义宏观内点标签：中位数GT误差小于阈值
            median_error = np.median(e_norm_gt)
            is_inlier_sample = 1.0 if median_error < args.inlier_thr_px else 0.0
            
            # Apply filters
            e2 = e2[keep]
            X_sel = X_sel[keep]
            u2_obs_final = u2_obs_final[keep]
            q0_final = q0[sel][keep]
            P0 = P0[keep]
            P2 = P2[keep]
            
            # Clip extreme outliers for robustness when aggregating labels
            if e2.shape[0] == 0:
                continue
            err_px_kept = np.sqrt(e2.sum(axis=1))
            if args.err_clip_px > 0:
                clip_mask = err_px_kept > args.err_clip_px
                if np.any(clip_mask):
                    scale = args.err_clip_px / np.maximum(err_px_kept[clip_mask], 1e-6)
                    e2[clip_mask] *= (scale[:, None] ** 2)
            
            # Aggregate to macro labels (MAD-robust)
            var_x = robust_var_mad(e2[:, 0]**0.5)
            var_y = robust_var_mad(e2[:, 1]**0.5)
            y_true = np.array([np.log(var_x), np.log(var_y)], dtype=np.float32)
            
            # === 修改：计算20维几何特征 + 4维帧对上下文 ===
            # 准备归一化坐标和时间间隔
            x0n_final = undist_norm(K0, D0, q0_final)
            x2n_final = undist_norm(K0, D0, u2_obs_final)
            delta_t = (ts0[j] - ts0[i]) * 1e-9  # 转换为秒
            
            # 调用特征计算函数 (20维)
            geom_20d = compute_geom_features_v2(
                q0_final, u2_obs_final,
                x0n_final, x2n_final,
                X_sel, T_cj_ci,
                P0, P2,
                delta_t, args.K_tokens
            )
            
            # 添加4维帧对上下文：[pair_type, delta_frames, delta_t, parallax_median]
            # 如果 para_pair 未计算（=-1），则现在计算
            if para_pair < 0:
                para_pair = median_pixel_parallax(q0_final, u2_obs_final)
            
            frame_context = np.array([
                pair_type,           # 1=KF, 0=Obs, -1=Default
                j - i,               # 帧间隔
                delta_t,             # 时间间隔(秒)
                para_pair            # 帧对级中位数视差
            ], dtype=np.float32)
            
            # 扩展到每个 token（广播）
            frame_context_expanded = np.tile(frame_context, (geom_20d.shape[0], 1))
            geom_24d = np.concatenate([geom_20d, frame_context_expanded], axis=1)  # (N, 24)
            
            # Stack patches (2 channels)
            patches = np.stack([P0, P2], axis=1)  # (n,2,H,W)
            
            # Sample or pad to K_tokens
            pK, realN = sample_or_pad(patches, args.K_tokens, pad_value=0)
            gK, _     = sample_or_pad(geom_24d,  args.K_tokens, pad_value=0.0)
            
            patches_all.append(pK.astype(np.uint8))
            geoms_all.append(gK.astype(np.float32))
            ytrue_all.append(y_true)
            numtok_all.append(np.int32(realN))
            # === 新增：存储内点标签（用uint8节省空间）===
            yinlier_all.append(np.array([int(is_inlier_sample)], dtype=np.uint8))

    # === 修改：处理并保存新的 yinlier_all 数组 ===
    patches_all = np.stack(patches_all, 0) if patches_all else np.empty((0,args.K_tokens,2,args.patch,args.patch), np.uint8)
    geoms_all   = np.stack(geoms_all,   0) if geoms_all   else np.empty((0,args.K_tokens,24), np.float32)  # 改为24维
    ytrue_all   = np.stack(ytrue_all,   0) if ytrue_all   else np.empty((0,2), np.float32)
    numtok_all  = np.stack(numtok_all,  0) if numtok_all  else np.empty((0,),  np.int32)
    yinlier_all = np.stack(yinlier_all, 0) if yinlier_all else np.empty((0,1), np.uint8)  # 改为uint8节省空间

    os.makedirs(Path(args.out_npz).parent, exist_ok=True)
    
    # === 新增：24维几何特征描述（20维token级 + 4维帧对级） ===
    geoms_desc = [
        # Token级特征（1-20）
        "u1_norm", "v1_norm", "u2_norm", "v2_norm",         # 1-4
        "radius1_norm", "radius2_norm",                     # 5-6
        "parallax_px_token", "parallax_deg_token",          # 7-8
        "inv_depth", "cos_theta_baseline",                  # 9-10
        "grad_mean1", "grad_mean2",                         # 11-12
        "corner_score1", "corner_score2",                   # 13-14
        "scale_change_log", "sampson_error",                # 15-16
        "token_rank_norm", "delta_t_token_sec",             # 17-18
        "mean_luminance1", "mean_luminance2",               # 19-20
        # 帧对级特征（21-24）
        "pair_type(1=KF,0=Obs,-1=Default)",                 # 21
        "delta_frames_pair",                                # 22
        "delta_t_pair_sec",                                 # 23
        "parallax_px_median_pair"                           # 24
    ]
    
    # Build metadata for reproducibility and debugging
    meta = {
        'seqs': args.seqs,
        'deltas': args.deltas if not args.kf_enable else None,
        'patch': args.patch,
        'K_tokens': args.K_tokens,
        'frame_step': args.frame_step if not args.kf_enable else None,
        'min_matches': args.min_matches,
        'pos_thr_px': args.pos_thr_px,
        'pos_thr_px_v1': args.pos_thr_px_v1,
        'pos_thr_px_v2': args.pos_thr_px_v2,
        'pos_thr_px_mh': args.pos_thr_px_mh,
        'err_clip_px': args.err_clip_px,
        'inlier_thr_px': args.inlier_thr_px,
        'geom_dim': 24,  # 更新为24维（20维token级 + 4维帧对级）
        'geoms_desc': geoms_desc,
        'y_format': 'log(MAD_var_x), log(MAD_var_y)',
        'y_inlier_format': 'median_gt_reproj_error < inlier_thr_px',
        'generation_mode': 'icgvins_aligned_kf' if args.kf_enable else 'default_delta_step',
        'kf_policy': {
            'parallax_px': args.kf_parallax_px,
            'max_interval_s': args.kf_max_interval_s,
            'min_interval_s': args.kf_min_interval_s,
            'emit_non_kf_ratio': args.emit_non_kf_ratio
        } if args.kf_enable else None,
        'pipeline': 'macro_frame_v4_geoms24_kf',  # 更新版本号
    }
    
    # === 修改：在保存时加入 y_inlier ===
    np.savez_compressed(args.out_npz,
                        patches=patches_all, geoms=geoms_all,
                        y_true=ytrue_all, num_tokens=numtok_all,
                        y_inlier=yinlier_all,  # 新增
                        meta=meta)
    
    # 打印统计信息
    inlier_count = (yinlier_all > 0.5).sum()
    inlier_rate = inlier_count / len(yinlier_all) if len(yinlier_all) > 0 else 0.0
    print(f"[done] saved: {args.out_npz}  samples={len(numtok_all)}  inliers={inlier_count} ({inlier_rate*100:.1f}%)")

if __name__ == "__main__":
    main()

