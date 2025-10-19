# datasets/euroc_io.py
import numpy as np, pandas as pd, cv2, yaml
from pathlib import Path

GRAV = np.array([0, 0, 9.81], dtype=np.float64)  # 世界坐标系向下为 +Z（EuRoC 的 R 系）

def read_imu_csv(csv_path: Path):
    df = pd.read_csv(csv_path, comment='#', header=None)
    df = df.iloc[:, :7]
    df.columns = ['t_ns','wx','wy','wz','ax','ay','az']
    df = df.sort_values('t_ns').reset_index(drop=True)
    return df

def read_gt_csv(csv_path: Path):
    # EuRoC GT: timestamp, p_RS_R_*, q_RS_*, v_RS_R_*, b_w_RS_S_*, b_a_RS_S_*
    cols = ['t_ns','px','py','pz','qw','qx','qy','qz','vx','vy','vz',
            'bgx','bgy','bgz','bax','bay','baz']
    df = pd.read_csv(csv_path, comment='#', header=None)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    df = df.sort_values('t_ns').reset_index(drop=True)
    return df

def quat_to_R(qw,qx,qy,qz):
    q = np.array([qw,qx,qy,qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    return R

def interp_gt_to(times_ns, gt_df):
    # 线性插值 p、v、bias；姿态做球面插值再求角速度
    t = gt_df['t_ns'].to_numpy().astype(np.int64)
    p = gt_df[['px','py','pz']].to_numpy(np.float64)
    v = gt_df[['vx','vy','vz']].to_numpy(np.float64)
    q = gt_df[['qw','qx','qy','qz']].to_numpy(np.float64)
    bg = gt_df[['bgx','bgy','bgz']].to_numpy(np.float64)
    ba = gt_df[['bax','bay','baz']].to_numpy(np.float64)

    # 速度插值得到 a_R（中央差分更稳）
    # 1) 先把 v_R(t) 插到 IMU 时间戳
    ts = times_ns.astype(np.int64)
    def lin(tgt, src_t, src_val):
        return np.vstack([np.interp(tgt, src_t, src_val[:,i]) for i in range(src_val.shape[1])]).T
    v_i = lin(ts, t, v)

    # 2) 用中央差分在 R 系求 a_R
    ts_s = ts * 1e-9
    aR = np.zeros_like(v_i)
    aR[1:-1] = (v_i[2:] - v_i[:-2]) / (ts_s[2:,None] - ts_s[:-2,None])
    aR[0] = aR[1]; aR[-1] = aR[-2]

    # 姿态插值（简化：最近邻 + 小角度差近似求 ω）
    # 为稳妥，直接用邻近两帧姿态做 q_{k->k+1} ≈ [0, ω*dt/2] ⊗ q_k
    # 数值上：ω ≈ 2 * vec( (q_k^{-1} ⊗ q_{k+1}) ) / dt
    # 先把 GT 姿态插到 IMU 时间最近的两边索引
    idx = np.searchsorted(t, ts, side='left')
    idx0 = np.clip(idx-1, 0, len(t)-1)
    idx1 = np.clip(idx,   0, len(t)-1)
    q0 = q[idx0]; q1 = q[idx1]
    t0 = t[idx0]*1e-9; t1 = t[idx1]*1e-9
    dt = (t1 - t0); dt[dt==0] = 1e-3

    def q_inv(q):
        s = np.sum(q*q, axis=1, keepdims=True)
        return q * np.array([[1,-1,-1,-1]]) / s
    def q_mul(a,b):
        aw,ax,ay,az = a.T; bw,bx,by,bz = b.T
        return np.vstack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        ]).T

    dq = q_mul(q_inv(q0), q1)
    # 角速度近似
    omega_S = 2.0 * dq[:,1:4] / dt[:,None]  # 已在 S（IMU）系
    # 旋转矩阵 R_RS：S->R
    R_RS = np.stack([quat_to_R(*qq) for qq in q0], axis=0)

    # 插值 bias
    bg_i = lin(ts, t, bg)
    ba_i = lin(ts, t, ba)

    return {
        'R_RS': R_RS,     # S->R
        'omega_S': omega_S,   # 真值角速度（S系）
        'a_R': aR,        # 世界系加速度
        'bg': bg_i,
        'ba': ba_i,
        't_ns': ts
    }

def ideal_imu_from_gt(gti):
    # 理想测量模型：
    #   ω_meas ≈ ω_true + bg
    #   a_meas ≈ R_SR * (a_R - g) + ba
    R_SR = np.transpose(gti['R_RS'], (0,2,1))
    a_S = (R_SR @ (gti['a_R'] - GRAV)[...,None]).squeeze(-1)
    return {
        'omega_S': gti['omega_S'] + gti['bg'],
        'acc_S':   a_S + gti['ba']
    }

def read_cam_timestamps(cam_dir: Path):
    csv = cam_dir/'data.csv'
    df = pd.read_csv(csv, comment='#', header=None)
    df = df.iloc[:, :2]
    df.columns = ['t_ns','relpath']
    paths = [cam_dir/'data'/p for p in df['relpath'].tolist()]
    ts = df['t_ns'].to_numpy(np.int64)
    return ts, [str(p) for p in paths]

def load_cam_yaml(cam_yaml: Path):
    y = yaml.safe_load(open(cam_yaml,'r'))
    K = np.array(y['intrinsics'], dtype=np.float64).reshape(4)  # [fx, fy, cx, cy] (EuRoC)
    D = np.array(y.get('distortion_parameters', [0,0,0,0]), dtype=np.float64)
    model = y.get('distortion_model','radtan')
    return K, D, model
