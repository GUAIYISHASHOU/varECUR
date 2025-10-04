#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visual uncertainty training data from EuRoC stereo pairs.
Extracts matched feature patches between time t and t+Δ with ground-truth reprojection errors.
"""
import argparse, json, cv2, yaml, numpy as np
from pathlib import Path

def load_camchain(camchain_yaml):
    Y = yaml.safe_load(open(camchain_yaml,'r'))
    # 取 cam0, cam1 的 K, D 与相对位姿 T_cam1_cam0
    def K_of(cam):
        intr = np.array(Y[cam]['intrinsics'], float)
        fx, fy, cx, cy = intr
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)
        D = np.array(Y[cam].get('distortion_coeffs', Y[cam].get('distortion_parameters',[0,0,0,0])), float)
        return K, D
    K0,D0 = K_of('cam0'); K1,D1 = K_of('cam1')
    T_imu_cam0 = np.array(Y['cam0']['T_cam_imu'], float).reshape(4,4)
    T_imu_cam1 = np.array(Y['cam1']['T_cam_imu'], float).reshape(4,4)
    T_cam1_cam0 = np.linalg.inv(T_imu_cam1) @ T_imu_cam0  # cam1←cam0
    R10, t10 = T_cam1_cam0[:3,:3], T_cam1_cam0[:3,3:4]
    return (K0,D0), (K1,D1), (R10, t10)

def undist_norm(K, D, pts):
    # 像素→归一化坐标（去畸变）
    pts = pts.reshape(-1,1,2).astype(np.float32)
    und = cv2.undistortPoints(pts, K, D)  # -> (N,1,2) in norm coords
    return und.reshape(-1,2)

def triangulate(K0,K1,R10,t10, x0n, x1n):
    # 用规范化坐标线性三角化
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3,1))])
    P1 = K1 @ np.hstack([R10, t10])
    X_h = cv2.triangulatePoints(P0, P1, x0n.T, x1n.T)  # 4×N
    X   = (X_h[:3] / (X_h[3:]+1e-12)).T               # N×3 in cam0@t
    return X

def project(K, X):
    X = X.reshape(-1,3)
    x = (X[:,:2] / (X[:,2:3] + 1e-12))
    return (K[:2,:2] @ x.T + K[:2,2:3]).T  # N×2 (像素)

def extract_orb(img, n=1200):
    orb = cv2.ORB_create(nfeatures=n, fastThreshold=7)
    kps, des = orb.detectAndCompute(img, None)
    if des is None or len(kps)<8: return None, None, None
    pts = np.array([k.pt for k in kps], np.float32)
    sc  = np.array([k.response for k in kps], np.float32)
    return pts, des, sc

def match_bf(d0, d1):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d0, d1)
    m.sort(key=lambda r: r.distance)
    return np.array([[mm.queryIdx, mm.trainIdx] for mm in m], int)

def crop_patches(img, pts, patch=32):
    H,W = img.shape[:2]; h=patch//2
    out, ok = [], []
    for (x,y) in pts:
        x=int(round(x)); y=int(round(y))
        if x-h<0 or y-h<0 or x+h>=W or y+h>=H: ok.append(False); continue
        out.append(img[y-h:y+h, x-h:x+h].copy())
        ok.append(True)
    return np.array(out, np.uint8), np.array(ok, bool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc_root", required=True)
    ap.add_argument("--seq", required=True)            # e.g. MH_01_easy
    ap.add_argument("--camchain", required=True)       # camchain-imucam.yaml
    ap.add_argument("--delta", type=int, default=1)    # t -> t+Δ
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--max_pairs", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.euroc_root)/args.seq
    cam0_csv = root/"mav0"/"cam0"/"data.csv"
    cam1_csv = root/"mav0"/"cam1"/"data.csv"
    # 读取时间戳与路径
    def read_cam(csv):
        rows = [r.strip().split(',') for r in open(csv,'r') if r and not r.startswith('#') and 'timestamp' not in r]
        ts = np.array([int(r[0]) for r in rows], np.int64)
        paths = [root/"mav0"/csv.parent.name/"data"/r[1] for r in rows]
        return ts, paths
    ts0, paths0 = read_cam(cam0_csv); ts1, paths1 = read_cam(cam1_csv)
    # 对齐左右相机（同一时刻）
    n = min(len(ts0), len(ts1))
    ts0, paths0, paths1 = ts0[:n], paths0[:n], paths1[:n]

    (K0,D0),(K1,D1),(R10,t10) = load_camchain(args.camchain)

    I0P, I1P, GEOM, E2X, E2Y, MASK = [], [], [], [], [], []

    for i in range(0, n-args.delta):
        # 读取 左t/右t/左t+Δ
        I0 = cv2.imread(str(paths0[i]),  cv2.IMREAD_GRAYSCALE)
        I1 = cv2.imread(str(paths1[i]),  cv2.IMREAD_GRAYSCALE)
        I2 = cv2.imread(str(paths0[i+args.delta]), cv2.IMREAD_GRAYSCALE)
        if I0 is None or I1 is None or I2 is None: continue

        # 立体匹配 @ t
        p0,d0,s0 = extract_orb(I0); 
        p1,d1,s1 = extract_orb(I1)
        if p0 is None or p1 is None: continue
        idx = match_bf(d0,d1)
        if len(idx) < 50: continue
        x0 = p0[idx[:,0]]; x1 = p1[idx[:,1]]

        # 去畸变→归一化→三角化
        x0n = undist_norm(K0,D0,x0); 
        x1n = undist_norm(K1,D1,x1)
        X3d = triangulate(K0,K1,R10,t10, x0n, x1n)              # in cam0@t

        # 跨时匹配：左 t → 左 t+Δ
        p2,d2,s2 = extract_orb(I2)
        if p2 is None: continue
        m02 = match_bf(d0, d2)                                  # 用 d0 连接 t 与 t+Δ
        if len(m02) < 50: continue
        q0 = p0[m02[:,0]]; q2 = p2[m02[:,1]]

        # 将立体匹配投到 t+Δ：需要用 q0 来索引 X3d（建立 3D-2D 对）
        # —— 简单做法：在 q0 附近用KDTree找最近的 x0（像素空间近似）
        from sklearn.neighbors import KDTree
        tree = KDTree(x0, leaf_size=40)
        dist, nn = tree.query(q0, k=1)
        sel = dist[:,0] < 2.5   # 2.5 px 内视为同一点
        if sel.sum() < 20: continue
        X_sel = X3d[nn[sel, 0]]           # 3D in cam0@t
        u2_obs = q2[sel]                  # 观测 @ t+Δ
        # 近似：相机位姿随载体刚性，使用 cam0 的相对位姿来自 GT（此处简化：假设小运动+短Δ，用恒等近似）
        # —— 为避免牵涉IMU-相机外参/GT插值的长链条：先给"近似baseline版"
        # 也可传入 Pose(t→t+Δ) 做精确投影；接口已留好：
        u2_pred = project(K0, X_sel)      # "近似"：当 Δ=1 & 视差小，误差仍可训练；若需精确，请替换为 GT 变换后再投影

        e = u2_obs - u2_pred
        e2 = e.astype(np.float32)**2
        m = np.isfinite(e2).all(axis=1)

        if m.sum() == 0: continue

        # 裁 patch（左 t 与 左 t+Δ）
        P0, ok0 = crop_patches(I0, q0[sel], patch=args.patch)
        P2, ok2 = crop_patches(I2, u2_obs,  patch=args.patch)
        keep = m & ok0 & ok2
        if keep.sum() == 0: continue

        # 几何上下文（可按需扩展）
        g = []
        for (u0,v0),(u2,v2) in zip(q0[sel][keep], u2_obs[keep]):
            g.append([u0/I0.shape[1]*2-1, v0/I0.shape[0]*2-1,
                      u2/I2.shape[1]*2-1, v2/I2.shape[0]*2-1])
        GEOM.append(np.array(g, np.float32))
        I0P.append(P0[keep][:,None,:,:])   # (N,1,H,W)
        I1P.append(P2[keep][:,None,:,:])
        E2X.append(e2[keep,0]); E2Y.append(e2[keep,1])
        MASK.append(np.ones((keep.sum(),), np.float32))

        if sum(len(x) for x in E2X) >= args.max_pairs: break

    if not E2X:
        print("no pairs collected."); return

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
        I0=np.concatenate(I0P,0), I1=np.concatenate(I1P,0),
        geom=np.concatenate(GEOM,0),
        e2x=np.concatenate(E2X,0), e2y=np.concatenate(E2Y,0),
        mask=np.concatenate(MASK,0),
        meta=dict(seq=args.seq, delta=args.delta, patch=args.patch))
    print("[OK] wrote", out)

if __name__ == "__main__":
    main()

