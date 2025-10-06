#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visual uncertainty training data from EuRoC with STRICT GT geometry.
Uses ground-truth poses + camera-IMU extrinsics for accurate reprojection supervision.
"""
import argparse, os, csv, math, yaml, json
from pathlib import Path
import numpy as np
import cv2

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可显示进度条 (pip install tqdm)")

# ---------- SE3 Transformations & Quaternions ----------
def quat_to_R(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
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

# ---------- EuRoC Data Reading ----------
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

def read_gt_states(gt_csv):
    """
    Read ground-truth states from EuRoC CSV.
    Supports different column name variations.
    Returns: timestamps, positions, quaternions (world←imu)
    """
    with open(gt_csv, 'r') as f:
        rdr = csv.reader(f)
        head = next(rdr)
        idx = {name: i for i, name in enumerate(head)}
        
        def get_idx(keys):
            """Find first matching key."""
            for k in keys:
                if k in idx: 
                    return idx[k]
            raise KeyError(f"None of {keys} found in CSV header")
        
        # Find column indices (handle different EuRoC versions)
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
            if not row or row[0].startswith('#'): 
                continue
            T.append(int(row[t_i]))
            P.append([float(row[px_i]), float(row[py_i]), float(row[pz_i])])
            Q.append([float(row[qw_i]), float(row[qx_i]), float(row[qy_i]), float(row[qz_i])])
    
    T = np.array(T, np.int64)
    P = np.array(P, np.float64)
    Q = np.array(Q, np.float64)
    return T, P, Q

def interp_pose(ts_all, T_gt, P, Q):
    """
    Interpolate poses at given timestamps.
    Uses linear interpolation for position and SLERP for rotation.
    """
    def slerp(q1, q2, alpha):
        """Spherical linear interpolation between quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:  # Nearly parallel
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
        
        # Linear interpolation for position
        p = (1 - a) * P[i0] + a * P[i1]
        # SLERP for rotation
        q = slerp(Q[i0], Q[i1], a)
        
        out.append(SE3(quat_to_R(*q), p))
    
    return out  # list of 4x4 SE3 matrices

def load_cam_yaml(yaml_path):
    """
    Load camera intrinsics, distortion, and extrinsics from EuRoC YAML.
    Returns: K, D, T_BS (IMU←Camera)
    """
    Y = yaml.safe_load(open(yaml_path, 'r'))
    
    # Intrinsics
    if 'intrinsics' in Y:
        fx, fy, cx, cy = Y['intrinsics']
    else:
        fx = Y['camera_matrix']['data'][0]
        fy = Y['camera_matrix']['data'][4]
        cx = Y['camera_matrix']['data'][2]
        cy = Y['camera_matrix']['data'][5]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    
    # Distortion coefficients
    if 'distortion_coefficients' in Y:
        if isinstance(Y['distortion_coefficients'], dict):
            D = np.array(Y['distortion_coefficients']['data'], np.float64)
        else:
            D = np.array(Y['distortion_coefficients'], np.float64)
    elif 'distortion_parameters' in Y:
        D = np.array(list(Y['distortion_parameters'].values()), np.float64)
    else:
        D = np.zeros((5,), np.float64)
    
    # T_BS: body(IMU) ← sensor(Camera)
    T_BS = np.eye(4)
    if 'T_BS' in Y:
        data = Y['T_BS']['data'] if isinstance(Y['T_BS'], dict) else Y['T_BS']
        T_BS = np.array(data, dtype=np.float64).reshape(4, 4)
    
    return K, D, T_BS

# ---------- Computer Vision Functions ----------
def undist_norm(K, D, pts):
    """Undistort points and convert to normalized coordinates."""
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, D)
    return und.reshape(-1, 2)

def triangulate(K0, K1, T_cam1_cam0, x0n, x1n):
    """
    Triangulate 3D points from stereo normalized coordinates.
    
    CRITICAL FIX: x0n, x1n are already undistorted & normalized (unit-focal-length) points,
    so we must use projection matrices WITHOUT intrinsics K.
    
    Previous bug: Was using P0=K0@[I|0], P1=K1@[R|t] with normalized coords,
    causing incorrect 3D reconstruction → huge reprojection errors → network saturates at lv_max.
    
    Args:
        K0, K1: Kept for compatibility but NOT used in triangulation
        T_cam1_cam0: 4x4 transform from cam0 to cam1
        x0n, x1n: Nx2 normalized coordinates (already undistorted)
    
    Returns:
        X: Nx3 3D points in cam0 frame
    """
    R10 = T_cam1_cam0[:3, :3]
    t10 = T_cam1_cam0[:3, 3:4]
    # Use identity projection matrices for normalized coordinates
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])  # [I|0], no K!
    P1 = np.hstack([R10, t10])                      # [R|t], no K!
    X_h = cv2.triangulatePoints(P0, P1, x0n.T, x1n.T)  # 4×N
    X = (X_h[:3] / (X_h[3] + 1e-12)).T  # N×3 in cam0@t
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

# ---------- Advanced Filtering ----------
def skew3(v):
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]], dtype=np.float64)

def sampson_err_norm(x1n, x0n, R, t):
    """
    Compute Sampson (first-order geometric) epipolar error.
    
    Args:
        x1n, x0n: [N,2] normalized coordinates (undistorted)
        R, t: Relative rotation and translation (cam1 ← cam0)
    
    Returns:
        err: [N] Sampson error in normalized image plane (≈radians for small errors)
    """
    x0 = np.hstack([x0n, np.ones((x0n.shape[0], 1))])
    x1 = np.hstack([x1n, np.ones((x1n.shape[0], 1))])
    
    E = skew3(t) @ R  # Essential matrix
    Ex0 = (E @ x0.T).T
    ETx1 = (E.T @ x1.T).T
    x1Ex0 = np.sum(x1 * (E @ x0.T).T, axis=1)
    
    num = x1Ex0 ** 2
    den = Ex0[:, 0]**2 + Ex0[:, 1]**2 + ETx1[:, 0]**2 + ETx1[:, 1]**2 + 1e-12
    
    return np.sqrt(num / den)

def precompute_grad_corner(path_list):
    """
    Precompute gradient magnitude and corner response for all frames.
    
    Returns:
        grads: list of gradient magnitude images (H×W float32)
        corners: list of corner response images (H×W float32)
    """
    grads, corners = [], []
    print(f"[precompute] Computing gradients and corners for {len(path_list)} frames...")
    
    for p in path_list:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            grads.append(None)
            corners.append(None)
            continue
        
        # Sobel gradient magnitude
        gX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gX*gX + gY*gY)
        
        # Corner response (eigenvalue-based)
        corn = cv2.cornerMinEigenVal(img, blockSize=3, ksize=3)
        
        grads.append(gmag.astype(np.float32))
        corners.append(corn.astype(np.float32))
    
    return grads, corners

# ---------- Main Pipeline ----------
def main():
    ap = argparse.ArgumentParser("Generate VIS pairs with strict GT geometry")
    ap.add_argument("--euroc_root", required=True, help="Path to EuRoC dataset root")
    ap.add_argument("--seq", required=True, help="Sequence name (e.g., MH_01_easy)")
    ap.add_argument("--delta", type=int, default=1, help="Time step delta (frames)")
    ap.add_argument("--patch", type=int, default=32, help="Patch size")
    ap.add_argument("--max_pairs", type=int, default=60000, help="Max pairs to collect")
    ap.add_argument("--out_npz", required=True, help="Output NPZ file path")
    
    # Sampling strategy parameters (for better diversity)
    ap.add_argument("--per_frame_cap", type=int, default=0, 
                   help="Max pairs per frame (0=unlimited, e.g., 200 for uniform coverage)")
    ap.add_argument("--frame_step", type=int, default=1,
                   help="Frame sampling step (1=every frame, 2=every other frame)")
    ap.add_argument("--min_frames", type=int, default=0,
                   help="Minimum frames to cover before stopping (0=no requirement)")
    
    # NEW: Multi-delta support
    ap.add_argument("--deltas", type=str, default=None,
                   help="Comma-separated frame intervals (e.g., '1,2'); if empty, use --delta single value")
    
    # NEW: Filtering thresholds
    ap.add_argument("--err_clip_px", type=float, default=15.0, 
                   help="Pixel error threshold (filter extreme outliers)")
    ap.add_argument("--depth_min", type=float, default=0.1, help="Min depth (m)")
    ap.add_argument("--depth_max", type=float, default=80.0, help="Max depth (m)")
    ap.add_argument("--epi_thr_px", type=float, default=1.5, 
                   help="Sampson epipolar error threshold (normalized image plane)")
    
    # NEW: Texture stratification
    ap.add_argument("--texture_strat", action="store_true",
                   help="Enable texture stratification (7:3 high:low gradient)")
    
    args = ap.parse_args()
    
    # Parse multi-delta
    deltas = [int(x) for x in args.deltas.split(",")] if args.deltas else [args.delta]
    print(f"[config] deltas={deltas}, err_clip={args.err_clip_px}px, epi_thr={args.epi_thr_px}, "
          f"depth=[{args.depth_min}, {args.depth_max}]m")
    
    root = Path(args.euroc_root) / args.seq / "mav0"
    
    # Read camera and ground-truth data
    ts0, p0 = read_cam_csv(root / "cam0" / "data.csv")
    ts1, p1 = read_cam_csv(root / "cam1" / "data.csv")
    gt_t, gt_p, gt_q = read_gt_states(root / "state_groundtruth_estimate0" / "data.csv")
    
    # Load camera parameters
    K0, D0, Timu_cam0 = load_cam_yaml(root / "cam0" / "sensor.yaml")
    K1, D1, Timu_cam1 = load_cam_yaml(root / "cam1" / "sensor.yaml")
    
    # Compute transformations
    Tcam0_imu = invSE3(Timu_cam0)  # Cam0 ← IMU
    Tcam1_imu = invSE3(Timu_cam1)
    Tcam1_cam0 = Tcam1_imu @ Timu_cam0  # Cam1 ← Cam0
    
    # Align lengths (use max delta for safety)
    max_delta = max(deltas)
    n = min(len(ts0), len(ts1)) - max_delta
    ts0 = ts0[:n]
    p0 = p0[:n]
    p1 = p1[:n]
    
    # Pre-interpolate IMU poses at t for each delta
    print(f"[interp] Computing GT poses for {n} frames at deltas={deltas}...")
    T_w_imu_t = interp_pose(ts0, gt_t, gt_p, gt_q)
    
    # Create dict of poses for each delta
    frame_dt = ts0[1] - ts0[0] if len(ts0) > 1 else 1e9
    T_w_imu_deltas = {}
    for d in deltas:
        T_w_imu_deltas[d] = interp_pose(ts0 + frame_dt * d, gt_t, gt_p, gt_q)
    
    # NEW: Precompute gradients and corners for texture stratification
    grads0, corners0 = None, None
    if args.texture_strat:
        grads0, corners0 = precompute_grad_corner(p0)
        print(f"[precompute] Gradient/corner maps ready for texture stratification")
    
    I0P, I2P, GEOM, E2X, E2Y, MASK = [], [], [], [], [], []
    
    print(f"[processing] Extracting pairs from {args.seq}...")
    if args.per_frame_cap > 0:
        print(f"  Sampling strategy: max {args.per_frame_cap} pairs/frame, step={args.frame_step}, min_frames={args.min_frames}")
    
    frames_covered = 0
    
    # 创建进度条 (with frame_step)
    frame_indices = range(0, n, args.frame_step)
    if HAS_TQDM:
        pbar = tqdm(frame_indices, desc="Processing frames", unit="frame")
    else:
        pbar = frame_indices
    
    for i in pbar:
        # Read reference images (stereo at time t)
        I0 = cv2.imread(str(p0[i]), cv2.IMREAD_GRAYSCALE)
        I1 = cv2.imread(str(p1[i]), cv2.IMREAD_GRAYSCALE)
        
        if I0 is None or I1 is None:
            continue
        
        # Stereo matching @ t (done once per frame)
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
        X3d = triangulate(K0, K1, Tcam1_cam0, x0n, x1n)  # in cam0@t
        
        Twi = T_w_imu_t[i]
        
        # === Loop over deltas (multi-temporal) ===
        for d in deltas:
            if i + d >= n:
                continue
            
            # Read target image at t+delta
            I2 = cv2.imread(str(p0[i + d]), cv2.IMREAD_GRAYSCALE)
            if I2 is None:
                continue
            
            # Compute relative pose: Cam0(t+Δ) ← Cam0(t)
            Twj = T_w_imu_deltas[d][i]
            T_cj_ci = invSE3(Twj @ Timu_cam0) @ (Twi @ Timu_cam0)
            
            R01 = T_cj_ci[:3, :3]
            t01 = T_cj_ci[:3, 3]
            Xj = (R01 @ X3d.T + t01.reshape(3, 1)).T  # Transform to cam0@(t+Δ)
            
            # Temporal matching: left t → left t+Δ
            x2, d2, s2 = orb_det_desc(I2)
            if x2 is None:
                continue
            
            m02 = match_bf(d0, d2)
            if len(m02) < 40:
                continue
            
            q0 = x0[m02[:, 0]]
            u2_obs = x2[m02[:, 1]]
            
            # Align 3D points with temporal matches using nearest neighbor
            if len(q0) == 0 or len(Xj) == 0:
                continue
            
            # Vectorized nearest neighbor (O(NM), sufficient for typical patch counts)
            diff = q0[:, None, :] - x0m[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            nn = np.argmin(dist, axis=1)
            # Adaptive threshold: looser when skipping frames (more motion)
            thr = 4.0 if args.frame_step > 1 else 2.5
            sel = (dist[np.arange(len(q0)), nn] < thr)
            
            if sel.sum() < 20:
                continue
            
            X_sel = Xj[nn[sel]]
            u2_obs_final = u2_obs[sel]
            
            # Predict pixel locations and compute residuals
            u2_pred = project(K0, X_sel)
            e = (u2_obs_final - u2_pred).astype(np.float32)
            e2 = e * e
            m = np.isfinite(e2).all(axis=1)
            
            if m.sum() == 0:
                continue
            
            # === Advanced Outlier Filtering (NEW) ===
            err = np.sqrt(e2.sum(axis=1))  # Pixel error norm
            
            # 1. Depth check: parameterized range
            depth_ok = (X_sel[:, 2] > args.depth_min) & (X_sel[:, 2] < args.depth_max)
            
            # 2. In-image check: predicted projection within bounds
            H, W = I2.shape[:2]
            in_img = (u2_pred[:,0] >= 0) & (u2_pred[:,0] < W) & \
                     (u2_pred[:,1] >= 0) & (u2_pred[:,1] < H)
            
            # 3. Reprojection error check: parameterized threshold
            err_ok = (err < args.err_clip_px)
            
            # 4. NEW: Sampson epipolar error (geometric consistency)
            x0n_temp = undist_norm(K0, D0, q0[sel])
            x2n_temp = undist_norm(K0, D0, u2_obs_final)
            epi_err = sampson_err_norm(x2n_temp, x0n_temp, R01, t01)
            epi_ok = (epi_err < args.epi_thr_px)
            
            # Combined validity mask
            good_pair = depth_ok & in_img & err_ok & epi_ok
            
            # Crop patches
            P0, ok0 = crop_patches(I0, q0[sel], patch=args.patch)
            P2, ok2 = crop_patches(I2, u2_obs_final, patch=args.patch)
            keep = m & ok0 & ok2 & good_pair
            
            if keep.sum() == 0:
                continue
            
            # === NEW: Texture stratification (7:3 high:low gradient) ===
            if args.texture_strat and grads0 is not None and grads0[i] is not None:
                g0_vals = grads0[i]
                # Sample gradient values at q0 locations
                xy = q0[sel][keep].astype(np.float32)
                mapx = xy[:, 0].reshape(-1, 1)
                mapy = xy[:, 1].reshape(-1, 1)
                g_samp = cv2.remap(g0_vals, mapx, mapy, cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REPLICATE).reshape(-1)
                
                thr = np.quantile(g_samp, 0.7) if g_samp.size > 10 else 0.0
                hi = (g_samp >= thr)
                lo = ~hi
                
                idx_all = np.flatnonzero(keep)
                if args.per_frame_cap > 0 and len(idx_all) > args.per_frame_cap:
                    k_hi = int(args.per_frame_cap * 0.7)
                    k_lo = args.per_frame_cap - k_hi
                    
                    def choice(mask, k):
                        I = np.flatnonzero(mask)
                        if I.size <= k: return I
                        return np.random.choice(I, size=k, replace=False)
                    
                    pick = np.concatenate([choice(hi, k_hi), choice(lo, k_lo)], axis=0)
                    tmp = np.zeros_like(hi, dtype=bool)
                    tmp[pick] = True
                    keep_new = np.zeros_like(keep, dtype=bool)
                    keep_new[idx_all[tmp]] = True
                    keep = keep_new
            else:
                # Per-frame sampling cap (uniform, no stratification)
                if args.per_frame_cap > 0 and keep.sum() > args.per_frame_cap:
                    idx = np.flatnonzero(keep)
                    sel_idx = np.random.choice(idx, size=args.per_frame_cap, replace=False)
                    keep_sampled = np.zeros_like(keep, dtype=bool)
                    keep_sampled[sel_idx] = True
                    keep = keep_sampled
            
            if keep.sum() == 0:
                continue
            
            frames_covered += 1
            
            # Store basic data
            I0P.append(P0[keep][:, None, :, :])
            I2P.append(P2[keep][:, None, :, :])
            E2X.append(e2[keep, 0])
            E2Y.append(e2[keep, 1])
            MASK.append(np.ones((keep.sum(),), np.float32))
            
            # === NEW: Enhanced geometric context with texture features ===
            xy0 = q0[sel][keep].astype(np.float32)
            xy2 = u2_obs_final[keep].astype(np.float32)
            
            # Normalized pixel coordinates ([-1, 1])
            u0n_norm = xy0[:, 0] / I0.shape[1] * 2 - 1
            v0n_norm = xy0[:, 1] / I0.shape[0] * 2 - 1
            u2n_norm = xy2[:, 0] / I2.shape[1] * 2 - 1
            v2n_norm = xy2[:, 1] / I2.shape[0] * 2 - 1
            
            # NEW: Sample gradient and corner features
            if grads0 is not None and grads0[i] is not None:
                mapx0 = xy0[:, 0].reshape(-1, 1)
                mapy0 = xy0[:, 1].reshape(-1, 1)
                g0s = cv2.remap(grads0[i], mapx0, mapy0, cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_REPLICATE).reshape(-1)
                c0s = cv2.remap(corners0[i], mapx0, mapy0, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).reshape(-1)
                
                if grads0[i+d] is not None:
                    mapx2 = xy2[:, 0].reshape(-1, 1)
                    mapy2 = xy2[:, 1].reshape(-1, 1)
                    g2s = cv2.remap(grads0[i+d], mapx2, mapy2, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE).reshape(-1)
                    c2s = cv2.remap(corners0[i+d], mapx2, mapy2, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE).reshape(-1)
                else:
                    g2s = np.zeros_like(g0s)
                    c2s = np.zeros_like(c0s)
            else:
                # Fallback: zero features
                g0s = np.zeros(len(xy0), dtype=np.float32)
                g2s = np.zeros(len(xy0), dtype=np.float32)
                c0s = np.zeros(len(xy0), dtype=np.float32)
                c2s = np.zeros(len(xy0), dtype=np.float32)
            
            # Optical flow magnitude
            flow = np.linalg.norm(xy2 - xy0, axis=1).astype(np.float32)
            
            # Baseline (translation norm)
            baseline = np.full_like(flow, np.linalg.norm(t01).astype(np.float32))
            
            # Parallax angle (viewing angle change)
            X_sel_keep = X_sel[keep]
            v0 = X_sel_keep / (np.linalg.norm(X_sel_keep, axis=1, keepdims=True) + 1e-12)
            v1 = (R01 @ X_sel_keep.T + t01.reshape(3, 1)).T
            v1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)
            cosang = np.clip(np.sum(v0 * v1, axis=1), -1.0, 1.0)
            parallax = np.arccos(cosang).astype(np.float32)
            
            # Stack all 11 features: [u0n, v0n, u2n, v2n, g0, g2, c0, c2, flow, baseline, parallax]
            feat = np.stack([u0n_norm, v0n_norm, u2n_norm, v2n_norm, 
                            g0s, g2s, c0s, c2s, flow, baseline, parallax], axis=1).astype(np.float32)
            GEOM.append(feat)
        
        # 更新进度条信息
        current_pairs = sum(len(x) for x in E2X)
        if HAS_TQDM:
            pbar.set_postfix({'pairs': current_pairs, 'frames': frames_covered})
        
        # Stop条件：达到max_pairs且满足min_frames要求
        if current_pairs >= args.max_pairs:
            if args.min_frames > 0 and frames_covered < args.min_frames:
                # 已达到pairs上限但帧数不够，继续收集
                continue
            else:
                if HAS_TQDM:
                    pbar.close()
                print(f"\n[stop] Reached max_pairs={args.max_pairs}, covered {frames_covered} frames")
                break
    
    if HAS_TQDM:
        pbar.close()
    
    if not E2X:
        print("[WARN] No pairs collected!")
        return
    
    # Save to NPZ with robust concatenation
    def _cat(xs, name):
        """Safe concatenation with error diagnostics."""
        xs = [x for x in xs if x is not None and len(x) > 0]
        if len(xs) == 0:
            raise RuntimeError(f"{name} is empty before save()")
        try:
            return np.concatenate(xs, axis=0)
        except Exception as err:
            shapes = [tuple(x.shape) for x in xs[:5]]
            lens = [len(x) for x in xs[:5]]
            print(f"[concat-error] {name}: head_shapes={shapes}, head_lens={lens}, n_chunks={len(xs)}")
            raise
    
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    total_pairs = sum(len(x) for x in E2X)
    
    try:
        # Enhanced metadata
        meta = dict(
            seq=args.seq,
            patch=args.patch,
            deltas=deltas,
            err_clip_px=args.err_clip_px,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            epi_thr_px=args.epi_thr_px,
            texture_strat=args.texture_strat,
            per_frame_cap=args.per_frame_cap,
            frame_step=args.frame_step,
            geom_dim=11 if args.texture_strat or grads0 is not None else 11,  # Always 11 now
            # Optional: Camera parameters for future factor graph integration
            # K0=K0.astype(np.float32),
            # D0=D0.astype(np.float32),
            # T_cam0_imu=Tcam0_imu.astype(np.float32)
        )
        
        np.savez_compressed(
            out,
            I0=_cat(I0P, "I0P"),
            I1=_cat(I2P, "I2P"),
            geom=_cat(GEOM, "GEOM"),
            e2x=_cat(E2X, "E2X"),
            e2y=_cat(E2Y, "E2Y"),
            mask=_cat(MASK, "MASK"),
            meta=meta,
        )
        print(f"[OK] Wrote {out}  | N={total_pairs} pairs from {frames_covered} frames")
        print(f"[OK] geom features: {meta['geom_dim']} dimensions")
    except PermissionError as e:
        print(f"[ERROR] Failed to save NPZ: {e}. 可能是文件被占用（Excel/压缩软件）或无写权限。")
        raise

if __name__ == "__main__":
    main()

