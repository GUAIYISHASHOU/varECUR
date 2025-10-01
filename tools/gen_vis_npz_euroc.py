#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import csv
import yaml
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Import new feature extraction functions
from vis_features import (
    rot_angle_deg, laplacian_var, hist_entropy, count_corners,
    normalize_points, fundamental_from_essential, sampson_dist_sq,
    parallax_angles_deg, triangulate_and_reproj_stats, fb_flow_error, SlidingAgg
)


# ======= Sampson error (归一化平面坐标) =======
def sampson_errors_normed(pts1n: np.ndarray, pts2n: np.ndarray, E: np.ndarray) -> np.ndarray:
    """pts1n, pts2n: (N,2) in normalized camera coords; E: (3,3)"""
    N = pts1n.shape[0]
    x1 = np.hstack([pts1n, np.ones((N,1))])
    x2 = np.hstack([pts2n, np.ones((N,1))])
    Ex1  = (E @ x1.T).T           # (N,3)
    Etx2 = (E.T @ x2.T).T         # (N,3)
    x2Ex1 = np.sum(x2 * (E @ x1.T).T, axis=1)  # (N,)
    num = x2Ex1 ** 2
    den = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2 + 1e-12
    return num / den

# ======= IO helpers =======
def load_euroc_intrinsics(sensor_yaml: str, cam_id: int | None = None):
    """
    兼容三种来源：
    1) OpenCV/TUM 风格: camera_matrix.data 或 K
    2) EuRoC(Kalibr) 风格: intrinsics / distortion_model / distortion_coefficients
    3) camchain-imucam.yaml 中的 cam{cam_id} 节点（兜底）
    返回: K(3x3), D(1xN), dist_model(str)
    """
    p = Path(sensor_yaml)
    with open(p, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f) or {}

    K, D, dist_model = None, None, None

    # --- 1) OpenCV/TUM 风格 ---
    cm = y.get("camera_matrix", None)
    if isinstance(cm, dict) and "data" in cm:
        data = np.array(cm["data"], dtype=float).reshape(3, 3)
        K = data
    if K is None and "K" in y:
        K = np.array(y["K"], dtype=float).reshape(3, 3)

    # --- 2) EuRoC(Kalibr) 风格: sensor.yaml ---
    if K is None and "intrinsics" in y:
        fx, fy, cx, cy = [float(v) for v in y["intrinsics"]]
        K = np.array([[fx, 0, cx],
                      [0,  fy, cy],
                      [0,  0,  1]], dtype=float)
    coeffs = y.get("distortion_coefficients", y.get("distortion_parameters", []))
    if coeffs is None:
        coeffs = []
    D = np.array(coeffs, dtype=float).reshape(-1)
    dist_model = str(y.get("distortion_model", "radtan")).lower()

    # --- 3) 兜底: camchain-imucam.yaml ---
    if K is None:
        cc = p.parent.parent / "camchain-imucam.yaml"
        if cc.exists():
            with open(cc, 'r', encoding='utf-8') as f:
                chain = yaml.safe_load(f) or {}
            node = None
            if cam_id is not None:
                node = chain.get(f"cam{cam_id}", None)
            if node is None:
                for k in ("cam0", "cam1"):
                    if k in chain:
                        node = chain[k]
                        break
                if node is None and isinstance(chain, dict) and len(chain):
                    node = list(chain.values())[0]
            if node and "intrinsics" in node:
                fx, fy, cx, cy = [float(v) for v in node["intrinsics"]]
                K = np.array([[fx, 0, cx],
                              [0,  fy, cy],
                              [0,  0,  1]], dtype=float)
                coeffs = node.get("distortion_coefficients", node.get("distortion_parameters", []))
                coeffs = coeffs if coeffs is not None else []
                D = np.array(coeffs, dtype=float).reshape(-1)
                dist_model = str(node.get("distortion_model", "radtan")).lower()

    if K is None:
        raise ValueError(
            "未在 sensor.yaml / camchain-imucam.yaml 找到相机内参：需要 camera_matrix.data / K 或 Kalibr 的 intrinsics。"
        )

    if D is None:
        D = np.zeros((0,), dtype=float)
    if dist_model is None:
        dist_model = "radtan"

    return K.astype(np.float64), D.astype(np.float64), dist_model


def undistort_points_uv(uv: np.ndarray, K: np.ndarray, D: np.ndarray, dist_model: str) -> np.ndarray:
    """
    uv: (N,2) 像素坐标
    dist_model: 'radtan'/'plumb_bob' 用 cv2.undistortPoints
                'equidistant'/'fisheye' 用 cv2.fisheye.undistortPoints
    """
    if D is None or len(D) == 0 or np.allclose(D, 0):
        return uv
    pts = uv.reshape(-1, 1, 2).astype(np.float32)
    m = dist_model.lower()
    if m in ("equidistant", "fisheye"):
        out = cv2.fisheye.undistortPoints(pts, K, D, P=K)
    else:
        out = cv2.undistortPoints(pts, K, D, P=K)
    return out.reshape(-1, 2)

def load_euroc_timestamps(data_csv: Path) -> tuple[list[Path], np.ndarray]:
    # data.csv 格式：#timestamp [ns],filename
    imgs, ts = [], []
    with open(data_csv, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].startswith("#"):  # 跳过表头
                continue
            t_ns = int(row[0])
            fn = row[1]
            imgs.append(Path(fn))
            ts.append(t_ns)
    return imgs, np.array(ts, dtype=np.int64)

def list_images_fallback(img_dir: Path):
    files = sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg"), *img_dir.glob("*.tiff")])
    ts = np.arange(len(files), dtype=np.int64)
    names = [p.name for p in files]
    return names, ts

def agg_frame_error(abs_err: np.ndarray, p: float) -> float:
    if abs_err.size == 0:
        return np.nan
    if p == 2.0:
        return float(np.mean(abs_err ** 2))
    return float(np.mean(abs_err ** p))

def features_from_flow(flow: np.ndarray, inlier_mask: np.ndarray | None, baseline_norm: float | None) -> np.ndarray:
    if flow.size == 0:
        return np.array([0.,0.,0., float(baseline_norm or 0.)], dtype=np.float32)
    mag = np.linalg.norm(flow, axis=1)
    inl = float(np.mean(inlier_mask.astype(np.float32))) if inlier_mask is not None else 0.0
    return np.array([inl, float(np.mean(mag)), float(np.std(mag)), float(baseline_norm or 0.)], dtype=np.float32)

def save_npz(out_npz: Path, X, E2, DF, M, TS, E2X=None, E2Y=None, DFX=None, DFY=None):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'X_VIS': X.astype(np.float32),
        'E2_VIS': E2.reshape(-1,1).astype(np.float32),
        'DF_VIS': DF.reshape(-1,1).astype(np.int32),
        'MASK_VIS': M.reshape(-1,1).astype(np.uint8),
        'TS_VIS': TS.astype(np.int64)
    }
    # Add diagonal supervision if provided
    if E2X is not None:
        data['E2X_VIS'] = E2X.reshape(-1,1).astype(np.float32)
    if E2Y is not None:
        data['E2Y_VIS'] = E2Y.reshape(-1,1).astype(np.float32)
    if DFX is not None:
        data['DFX_VIS'] = DFX.reshape(-1,1).astype(np.int32)
    if DFY is not None:
        data['DFY_VIS'] = DFY.reshape(-1,1).astype(np.int32)
    np.savez(out_npz, **data)
    print(f"[ok] wrote {out_npz}  (T={len(X)}, d_in={X.shape[-1]})")

# ======= Main pipeline =======
def process_sequence(seq_root: Path, cam_id: int, out_dir: Path,
                     nfeatures=2000, topk=1200, min_match=20, ransac_thr=1.0, lp_p=3.0):
    cam_dir = seq_root / "mav0" / f"cam{cam_id}"
    img_dir = cam_dir / "data"
    data_csv = cam_dir / "data.csv"
    sensor_yaml = cam_dir / "sensor.yaml"

    assert img_dir.exists(), f"not found: {img_dir}"
    assert sensor_yaml.exists(), f"not found: {sensor_yaml}"
    K, D, dist_model = load_euroc_intrinsics(sensor_yaml, cam_id)

    if data_csv.exists():
        names, ts = load_euroc_timestamps(data_csv)
        img_paths = [img_dir / n for n in names]
    else:
        names, ts = list_images_fallback(img_dir)
        img_paths = [img_dir / n for n in names]
    assert len(img_paths) >= 2, "need at least 2 images"

    T = len(img_paths)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    Kinv = np.linalg.inv(K)

    # Initialize sliding aggregator for temporal features
    agg = SlidingAgg(k=5)
    
    # Feature names (17 raw features)
    feat_names = [
        "inlier_ratio", "flow_mean", "flow_std", "baseline_norm",
        "inlier_count", "flow_median", "flow_iqr",
        "num_corners", "lap_var", "hist_entropy",
        "dtheta_deg", "parallax_deg_med",
        "sampson_med", "sampson_p90",
        "reproj_mean_px", "reproj_med_px",
        "fb_flow_err"
    ]
    n_raw = len(feat_names)
    n_agg = n_raw * 3  # mean, p90, min for each
    n_total = n_raw + n_agg

    X_rows = [np.zeros(n_total, np.float32)]   # 第一个帧占位
    E2_rows = [np.nan]                   # 首帧无相对量 -> 设为 NaN
    DF_rows = [0]                        # 首帧自由度
    M_rows  = [0]                        # 首帧 mask=0
    E2X_rows = [np.nan]                  # 对角监督：X轴
    E2Y_rows = [np.nan]                  # 对角监督：Y轴
    DFX_rows = [0]                       # X轴自由度
    DFY_rows = [0]                       # Y轴自由度

    frame_iter = range(T-1)
    if tqdm is not None:
        desc = f"{seq_root.name} cam{cam_id}"
        frame_iter = tqdm(frame_iter, desc=desc, unit="pair")

    for i in frame_iter:
        im1 = cv2.imread(str(img_paths[i]), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(str(img_paths[i+1]), cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None:
            X_rows.append(np.zeros(n_total, np.float32)); E2_rows.append(0.0); DF_rows.append(0); M_rows.append(0)
            E2X_rows.append(np.nan); E2Y_rows.append(np.nan); DFX_rows.append(0); DFY_rows.append(0)
            continue

        k1, d1 = orb.detectAndCompute(im1, None)
        k2, d2 = orb.detectAndCompute(im2, None)
        if d1 is None or d2 is None or len(k1) < min_match or len(k2) < min_match:
            X_rows.append(np.zeros(n_total, np.float32)); E2_rows.append(0.0); DF_rows.append(0); M_rows.append(0)
            E2X_rows.append(np.nan); E2Y_rows.append(np.nan); DFX_rows.append(0); DFY_rows.append(0)
            continue

        matches = bf.match(d1, d2)
        matches = sorted(matches, key=lambda m: m.distance)[:topk]
        if len(matches) < min_match:
            X_rows.append(np.zeros(n_total, np.float32)); E2_rows.append(0.0); DF_rows.append(0); M_rows.append(0)
            E2X_rows.append(np.nan); E2Y_rows.append(np.nan); DFX_rows.append(0); DFY_rows.append(0)
            continue

        pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches])

        # RANSAC 估 E（像素坐标 + K）
        E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thr)
        if E is None:
            X_rows.append(np.zeros(n_total, np.float32)); E2_rows.append(0.0); DF_rows.append(0); M_rows.append(0)
            E2X_rows.append(np.nan); E2Y_rows.append(np.nan); DFX_rows.append(0); DFY_rows.append(0)
            continue
        inl_mask = inliers.ravel().astype(bool) if inliers is not None else None

        # 归一化点（去畸变 + K^-1），用于 Sampson 误差
        pts1n = undistort_points_uv(pts1, K, D, dist_model)
        pts2n = undistort_points_uv(pts2, K, D, dist_model)

        # Recover pose to get R, t
        try:
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
            baseline = float(np.linalg.norm(t))
        except Exception:
            R = np.eye(3, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
            baseline = 0.0

        # === NEW: Expanded feature computation ===
        # Basic flow statistics
        flow = pts2 - pts1
        flow_mag = np.linalg.norm(flow, axis=1)
        flow_median = float(np.median(flow_mag)) if flow_mag.size > 0 else np.nan
        flow_iqr = float(np.percentile(flow_mag, 75) - np.percentile(flow_mag, 25)) if flow_mag.size > 0 else np.nan
        
        # Image quality metrics
        num_c = count_corners(im2)
        lap_v = laplacian_var(im2)
        ent = hist_entropy(im2)
        
        # Rotation angle
        dtheta = rot_angle_deg(R)
        
        # Fundamental matrix and Sampson distance
        F = fundamental_from_essential(E, K)
        sampson2 = sampson_dist_sq(F, pts1n, pts2n)  # (N,)
        
        # Inlier statistics
        inl = inl_mask.astype(bool) if inl_mask is not None else np.zeros(len(pts1), dtype=bool)
        ninl = int(inl.sum())
        inlier_ratio = float(ninl / len(pts1)) if len(pts1) > 0 else 0.0
        samp_med = float(np.median(sampson2[inl])) if ninl > 0 else np.nan
        samp_p90 = float(np.percentile(sampson2[inl], 90)) if ninl > 0 else np.nan
        
        # Parallax angle
        parallax_deg_med = float(np.median(parallax_angles_deg(R, pts1n, pts2n))) if ninl > 0 else np.nan
        
        # Reprojection stats + diagonal supervision
        reproj_mean_px, reproj_med_px, e2x_mean, e2y_mean, dfx, dfy = \
            triangulate_and_reproj_stats(K, R, t, pts1, pts2, inl_mask if inl_mask is not None else np.ones(len(pts1), dtype=bool))
        
        # Forward-backward flow consistency
        fb_err = fb_flow_error(im1, im2, pts1.astype(np.float32)) if pts1.shape[0] >= 8 else np.nan
        
        # === Assemble feature vector ===
        feat_values = [
            float(inlier_ratio), float(np.mean(flow_mag)) if flow_mag.size > 0 else np.nan, 
            float(np.std(flow_mag)) if flow_mag.size > 0 else np.nan, float(baseline),
            float(ninl), flow_median, flow_iqr,
            float(num_c), lap_v, ent,
            dtheta, parallax_deg_med,
            samp_med, samp_p90,
            float(reproj_mean_px), float(reproj_med_px),
            float(fb_err)
        ]
        
        # Sliding window aggregation
        for n, v in zip(feat_names, feat_values):
            agg.push(n, v)
        agg_block = agg.featurize(feat_names)  # 3 * len(feat_names)
        
        # Combine raw + aggregated features
        x_step = feat_values + agg_block
        
        # === Supervision: Original Sampson-based E2 ===
        valid = int(ninl >= min_match)
        if ninl > 0:
            pts1n_inl = pts1n[inl]
            pts2n_inl = pts2n[inl]
            errs_inl = sampson_errors_normed(pts1n_inl, pts2n_inl, E)
            errs_inl = np.abs(errs_inl)
            mse = float(np.mean(errs_inl)) if ninl > 0 else np.nan
            df = int(2 * ninl)
            
            # Debug output
            if i % max((T // 10), 1) == 0:
                H, W = im1.shape
                residuals_px = pts2[inl] - pts1[inl]
                sse_px2 = float((residuals_px ** 2).sum())
                mse_px2 = float((residuals_px ** 2).mean()) if residuals_px.size else 0.0
                rms_px = float(np.sqrt(sse_px2 / max(2 * ninl, 1)))
                print("[vis-debug] win", i,
                      "img_wh=", (W, H),
                      "#inliers=", ninl,
                      "SSE(px^2)=", sse_px2,
                      "MSE(px^2)=", mse_px2,
                      "per-pt RMS(px)~", rms_px,
                      "d_in=", n_total)
        else:
            mse = np.nan
            df = 0
        
        # Validity check
        e2_clip = 1000.0
        mask_vis = bool(valid and np.isfinite(mse) and (mse < e2_clip - 1e-6) and df >= 6)
        e2_out = np.nan if not mask_vis else float(mse)
        
        X_rows.append(np.array(x_step, dtype=np.float32))
        E2_rows.append(e2_out)
        DF_rows.append(df if mask_vis else 0)
        M_rows.append(1 if mask_vis else 0)
        E2X_rows.append(float(e2x_mean) if dfx > 0 else np.nan)
        E2Y_rows.append(float(e2y_mean) if dfy > 0 else np.nan)
        DFX_rows.append(int(dfx))
        DFY_rows.append(int(dfy))

    X  = np.stack(X_rows, axis=0)                 # (T, n_total)
    E2 = np.array(E2_rows, dtype=np.float32)      # (T,)
    DF = np.array(DF_rows, dtype=np.int32)        # (T,)
    M  = np.array(M_rows, dtype=np.uint8)         # (T,)
    E2X = np.array(E2X_rows, dtype=np.float32)    # (T,)
    E2Y = np.array(E2Y_rows, dtype=np.float32)    # (T,)
    DFX = np.array(DFX_rows, dtype=np.int32)      # (T,)
    DFY = np.array(DFY_rows, dtype=np.int32)      # (T,)
    out_npz = out_dir / f"{seq_root.name}_cam{cam_id}_vis.npz"
    save_npz(out_npz, X, E2, DF, M, TS=ts, E2X=E2X, E2Y=E2Y, DFX=DFX, DFY=DFY)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc_root", required=True, help="例如 /path/to/EuRoC/V1_01_easy")
    ap.add_argument("--cam_id", type=int, default=0, choices=[0,1])
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--nfeatures", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=1200)
    ap.add_argument("--min_match", type=int, default=20)
    ap.add_argument("--ransac_thr", type=float, default=1.0)
    ap.add_argument("--lp_p", type=float, default=3.0)
    args = ap.parse_args()

    seq_root = Path(args.euroc_root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    process_sequence(seq_root, args.cam_id, out_dir,
                     nfeatures=args.nfeatures, topk=args.topk, min_match=args.min_match,
                     ransac_thr=args.ransac_thr, lp_p=args.lp_p)

if __name__ == "__main__":
    main()

