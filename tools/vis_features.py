# tools/vis_features.py
import numpy as np
import cv2
from collections import deque

def rot_angle_deg(R):
    """|Δθ| from relative rotation matrix (degrees)."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))

def laplacian_var(gray):
    """Image sharpness proxy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def hist_entropy(gray):
    """8bit grayscale histogram entropy in bits."""
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    nz = p[p>0]
    return float(-(nz * np.log2(nz)).sum())

def count_corners(gray, max_corners=500, qlevel=0.01, mindist=7):
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=qlevel, minDistance=mindist)
    return 0 if pts is None else int(pts.shape[0])

def normalize_points(pts_px, Kinv):
    """pts_px: (N,2) pixel → normalized (N,2)"""
    pts_h = np.concatenate([pts_px, np.ones((pts_px.shape[0],1))], axis=1)  # (N,3)
    pts_n = (Kinv @ pts_h.T).T
    return pts_n[:, :2] / (pts_n[:, 2:3] + 1e-12)

def fundamental_from_essential(E, K):
    """F = K^{-T} E K^{-1}"""
    Kinv = np.linalg.inv(K)
    return Kinv.T @ E @ Kinv

def sampson_dist_sq(F, x0n, x1n):
    """
    x0n/x1n: normalized coords (N,2) in cam0/cam1.
    Return Sampson squared distance (N,)
    """
    # homo
    x0 = np.concatenate([x0n, np.ones((x0n.shape[0],1))], axis=1)  # (N,3)
    x1 = np.concatenate([x1n, np.ones((x1n.shape[0],1))], axis=1)
    Fx0 = (F @ x0.T).T                  # (N,3)
    FTx1 = (F.T @ x1.T).T               # (N,3)
    x1Fx0 = np.sum(x1 * (F @ x0.T).T, axis=1)  # (N,)
    denom = Fx0[:,0]**2 + Fx0[:,1]**2 + FTx1[:,0]**2 + FTx1[:,1]**2 + 1e-12
    return (x1Fx0**2) / denom

def parallax_angles_deg(R, x0n, x1n):
    """
    Approx parallax between bearing rays, both in cam0 frame.
    x0n, x1n: normalized (N,2). Return (N,) degrees.
    """
    b0 = np.concatenate([x0n, np.ones((x0n.shape[0],1))], axis=1)
    b0 = b0 / (np.linalg.norm(b0, axis=1, keepdims=True) + 1e-12)
    b1 = np.concatenate([x1n, np.ones((x1n.shape[0],1))], axis=1)
    # rotate second ray into cam0 frame
    b1_in0 = (R.T @ b1.T).T
    b1_in0 = b1_in0 / (np.linalg.norm(b1_in0, axis=1, keepdims=True) + 1e-12)
    cosang = np.sum(b0 * b1_in0, axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def triangulate_and_reproj_stats(K, R, t, x0_px, x1_px, inlier_mask):
    """
    Return:
      reproj_mean_px, reproj_med_px,  e2x_mean, e2y_mean (per-axis mean of squared reproj residual, pixels), df_x, df_y
    """
    m = inlier_mask.astype(bool)
    if m.sum() < 8:
        return np.nan, np.nan, np.nan, np.nan, 0, 0
    x0 = x0_px[m].astype(np.float32); x1 = x1_px[m].astype(np.float32)
    # Projection matrices
    P0 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P1 = K @ np.hstack([R, t.reshape(3,1)])
    # normalize to homogeneous
    x0n = cv2.undistortPoints(x0.reshape(-1,1,2), K, None).reshape(-1,2)
    x1n = cv2.undistortPoints(x1.reshape(-1,1,2), K, None).reshape(-1,2)
    X_h = cv2.triangulatePoints(P0, P1, x0.T, x1.T)  # (4,N)
    X = (X_h[:3] / (X_h[3:4] + 1e-12)).T            # (N,3)

    # Reproject
    def proj(P, X):
        X_h = np.hstack([X, np.ones((X.shape[0],1))])  # (N,4)
        uvw = (P @ X_h.T).T
        uv = uvw[:, :2] / (uvw[:, 2:3] + 1e-12)
        return uv

    u0_hat = proj(P0, X); u1_hat = proj(P1, X)
    r0 = x0 - u0_hat
    r1 = x1 - u1_hat
    # combine two views
    rx = np.concatenate([r0[:,0], r1[:,0]])
    ry = np.concatenate([r0[:,1], r1[:,1]])
    rn = np.linalg.norm(np.vstack([r0, r1]), axis=1)
    reproj_mean = float(rn.mean())
    reproj_med  = float(np.median(rn))
    e2x_mean = float((rx**2).mean())
    e2y_mean = float((ry**2).mean())
    df = rx.shape[0]  # 2*#inliers
    return reproj_mean, reproj_med, e2x_mean, e2y_mean, df//2, df//2

def fb_flow_error(prev_gray, next_gray, pts0):
    """forward-backward LK flow consistency (px). pts0: (N,2) float32 pixels."""
    if pts0.shape[0] < 8:
        return np.nan
    p0 = pts0.reshape(-1,1,2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None,
                                         winSize=(21,21), maxLevel=3,
                                         criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    st = st.reshape(-1).astype(bool)
    if st.sum() < 8:
        return np.nan
    p0b, stb, _ = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, p1, None,
                                           winSize=(21,21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    stb = stb.reshape(-1).astype(bool)
    ok = st & stb
    if ok.sum() < 8:
        return np.nan
    err = np.linalg.norm(p0[ok,:, :]-p0b[ok,:, :], axis=2).reshape(-1)
    return float(err.mean())

class SlidingAgg:
    """
    Keep last k scalars per feature id and emit [mean, p90, min] for each.
    """
    def __init__(self, k=5):
        self.k = k
        self.buf = {}  # name -> deque
    def push(self, name, value):
        dq = self.buf.setdefault(name, deque(maxlen=self.k))
        dq.append(float(value))
    def featurize(self, ordered_names):
        out = []
        for n in ordered_names:
            dq = self.buf.get(n, deque())
            if len(dq)==0:
                out.extend([np.nan, np.nan, np.nan])
            else:
                arr = np.array(dq, dtype=float)
                out.extend([float(np.nanmean(arr)),
                            float(np.nanpercentile(arr, 90)),
                            float(np.nanmin(arr))])
        return out

