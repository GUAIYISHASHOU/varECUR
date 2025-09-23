# vision/vis_metrics.py
import cv2, numpy as np
from typing import List, Tuple

def _K_mat(fx, fy, cx, cy):
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64); return K

def pair_metrics(img1, img2, K_tuple):
    fx, fy, cx, cy = K_tuple
    K = _K_mat(fx,fy,cx,cy)

    orb = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1)<8 or len(k2)<8:
        return dict(n_match=0, inlier=0, inlier_ratio=0.0, sampson_mean=np.inf, sampson_med=np.inf)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = matcher.match(d1, d2)
    if len(m) < 8:
        return dict(n_match=len(m), inlier=0, inlier_ratio=0.0, sampson_mean=np.inf, sampson_med=np.inf)

    pts1 = np.float32([k1[i.queryIdx].pt for i in m])
    pts2 = np.float32([k2[i.trainIdx].pt for i in m])

    E, inliers = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx,cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return dict(n_match=len(m), inlier=0, inlier_ratio=0.0, sampson_mean=np.inf, sampson_med=np.inf)
    mask = inliers.ravel().astype(bool)
    p1 = cv2.undistortPoints(pts1.reshape(-1,1,2), _K_mat(fx,fy,cx,cy), None).reshape(-1,2)
    p2 = cv2.undistortPoints(pts2.reshape(-1,1,2), _K_mat(fx,fy,cx,cy), None).reshape(-1,2)

    # Sampson 误差（已归一化相机坐标）
    def sampson(E, x1, x2):
        x1h = np.hstack([x1, np.ones((x1.shape[0],1))])
        x2h = np.hstack([x2, np.ones((x2.shape[0],1))])
        Ex1 = (E @ x1h.T).T
        Etx2 = (E.T @ x2h.T).T
        x2tEx1 = np.sum(x2h * (E @ x1h.T).T, axis=1)
        num = x2tEx1**2
        den = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2 + 1e-12
        return num / den

    s_all = sampson(E, p1, p2)
    s_in = s_all[mask]
    return dict(
        n_match=len(m),
        inlier=int(mask.sum()),
        inlier_ratio=float(mask.mean()),
        sampson_mean=float(np.mean(s_in)) if len(s_in)>0 else float('inf'),
        sampson_med=float(np.median(s_in)) if len(s_in)>0 else float('inf'),
    )
