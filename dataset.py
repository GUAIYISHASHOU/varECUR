from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict
# ==== 放到文件顶部（或类外部） ====
DBG_MAX = 8  # 每类告警最多打印次数，避免刷屏

def _dbg_limit(counter: dict, key: str, maxn=DBG_MAX) -> bool:
    """返回 True 表示还可以打印；False 表示超过上限不要再打印"""
    n = counter.get(key, 0)
    if n < maxn:
        counter[key] = n + 1
        return True
    return False

def _get(arrs: dict, keys, default=None):
    for k in keys:
        if k in arrs: 
            return arrs[k]
    return default

def _ensure_bool_mask(m):
    m = m.astype(np.float32)
    m = (m > 0.5).astype(np.float32)
    return m

class IMURouteDataset(Dataset):
    def __init__(self, npz_path: str | Path, route: str = "acc", x_mode: str = "both"):
        self.npz_path = str(npz_path)
        self.route = route
        self.x_mode = x_mode
        assert route in ("acc","gyr","vis")
        valid_modes = {"both", "route_only", "imu", "visual"}
        if x_mode not in valid_modes:
            raise ValueError(f"Unsupported x_mode='{x_mode}', must be one of {sorted(valid_modes)}")
        # 视觉路由允许 'both'（默认）或 'visual'（纯视觉别名）
        if self.route == "vis":
            if x_mode not in ("both", "visual"):
                raise ValueError("Vision route only supports x_mode in {'both','visual'}")
        else:
            # IMU 路由允许 'both'、'route_only'，以及更直观的别名 'imu'
            if x_mode not in ("both", "route_only", "imu"):
                raise ValueError("IMU route only supports x_mode in {'both','route_only','imu'}")
        data = np.load(self.npz_path, allow_pickle=True)
        def _pick(keys):
            for k in keys:
                if k in data.files: return data[k]
            return None
        # 获取输入数据
        if self.route == "acc":
            X = _pick(["X_IMU_ACC", "X_acc", "X"])
            TS = _pick(["TS_IMU"])
        elif self.route == "gyr":
            X = _pick(["X_IMU_GYR", "X_gyr", "X"])
            TS = _pick(["TS_IMU"])
        elif self.route == "vis":
            X = _pick(["X_VIS", "X_vis", "X"])
            TS = _pick(["TS_VIS"])
        else:
            raise ValueError(f"unknown route {self.route}")
        if X is None:
            raise ValueError(f"{self.npz_path}: missing X keys for route={self.route}")
        X = X.astype(np.float32)
        if X.ndim == 2:
            X = X[None, ...]  # (T,D) -> (1,T,D)
        elif X.ndim == 1:
            X = X[None, :, None]  # (T,) -> (1,T,1)
        self.N, self.T, self.D = X.shape
        # 1) 优先用步级标签（推荐）
        if self.route == "vis" and ("E2_VIS" in data.files):
            # VIS: 步级 e2（像素^2），输出维度=1
            E2_step = data["E2_VIS"].astype(np.float32)
            if E2_step.ndim == 1:   # (T,) -> (1,T,1)
                E2 = E2_step[None, :, None]
            elif E2_step.ndim == 2: # (N,T) -> (N,T,1)
                E2 = E2_step[:, :, None]
            elif E2_step.ndim == 3 and E2_step.shape[-1] == 1:
                E2 = E2_step
            else:
                raise ValueError(f"unexpected E2_VIS shape {E2_step.shape}")
            self.use_step_labels = True
            self.d_out = 1
            Y_anchor = None
            DF_anchor = data["DF_VIS"].astype(np.int32) if "DF_VIS" in data.files else None
        elif self.route == "acc" and "E2_IMU_ACC" in data.files:
            E2_step = data["E2_IMU_ACC"].astype(np.float32)  # [N,T,3]
            self.use_step_labels = True
            self.d_out = 3  # 支持三轴输出
            E2 = E2_step  # [N,T,3]
            Y_anchor = None
        elif self.route == "gyr" and "E2_IMU_GYR" in data.files:
            E2_step = data["E2_IMU_GYR"].astype(np.float32)  # [N,T,3]
            self.use_step_labels = True
            self.d_out = 3  # 支持三轴输出
            E2 = E2_step  # [N,T,3]
            Y_anchor = None
        # 2) 否则回退：窗口锚 + 轻中心化（旧 EUROC npz 只有 Y_IMU_*）
        else:
            self.use_step_labels = False
            if self.route == "acc":
                Y_anchor = _pick(["Y_IMU_ACC", "E2_acc", "E2", "Y"])
            elif self.route == "gyr":
                Y_anchor = _pick(["Y_IMU_GYR", "E2_gyr", "E2", "Y"])
            else:  # vis（旧 npz 兜底才会走到这里）
                Y_anchor = _pick(["Y_VIS", "E2_vis", "E2", "Y"])
            if Y_anchor is None:
                raise ValueError(f"{self.npz_path}: no labels found for route={self.route}")
            Y_anchor = Y_anchor.astype(np.float32)
            DF_anchor = None
            if self.route == "vis" and "DF_VIS" in data.files:
                DF_anchor = data["DF_VIS"].astype(np.int32)
            self.df_all = DF_anchor
            self.d_out = 1  # 回退模式使用标量输出
            # 把窗口标签变成逐时间步，用于与旧代码兼容
            if self.route in ("acc", "gyr"):
                if Y_anchor.ndim == 2 and Y_anchor.shape[1] == 3:
                    e2_scalar = Y_anchor.sum(axis=1).astype(np.float32) / 3.0  # 每轴均值
                elif Y_anchor.ndim == 2 and Y_anchor.shape[1] == 1:
                    e2_scalar = Y_anchor[:,0]
                else:
                    raise ValueError(f"unexpected Y shape {Y_anchor.shape} for IMU")
                E2 = np.repeat(e2_scalar[:, None, None], self.T, axis=1)  # [N,T,1]
            else:  # vis
                if Y_anchor.ndim == 2 and Y_anchor.shape[1] == 1:
                    e2_scalar = Y_anchor[:,0].astype(np.float32)
                else:
                    e2_scalar = Y_anchor.reshape(-1).astype(np.float32)
                E2 = np.repeat(e2_scalar[:, None, None], self.T, axis=1)  # [N,T,1]
                if DF_anchor is not None:
                    DF_anchor = DF_anchor.reshape(self.N, self.T)
        # 3) MASK：如果没有就置为全 1
        if self.route == "vis":
            M = _pick(["MASK_VIS", "mask_vis", "MASK", "mask"])
        else:
            M = _pick(["MASK_IMU", "MASK", "mask", "mask_vis"])
        if M is None:
            M = np.ones((self.N, self.T), dtype=np.float32)
        else:
            M = (M.astype(np.float32) > 0.5).astype(np.float32)
            if M.ndim == 3:  # [N,T,1]→[N,T]
                M = M[...,0]
            if M.ndim == 2 and M.shape[0] != self.N:
                M = np.expand_dims(M, axis=0)
            if M.ndim == 2 and M.shape[0] == self.N:
                pass
            elif M.ndim == 1:
                M = np.repeat(M[None, :], self.N, axis=0)
        self.X_all = X
        self.E2_all = E2  # [N,T,3] 或 [N,T,1]
        self.M_all = M    # [N,T]
        self.DF_all = DF_anchor if 'DF_anchor' in locals() else None
        self.TS = TS if TS is not None else None
        self.Y_anchor = Y_anchor  # 窗口级标签（如果有的话）
        # 为 VIS 路由保存原始 E2_VIS/MASK_VIS 数据
        if self.route == "vis" and ("E2_VIS" in data.files):
            # ==== 在 __init__ 里准备窗口起点 ====
            # 1) 读入 1D 的步级数组
            self.E2_VIS   = np.asarray(data["E2_VIS"])
            self.MASK_VIS = np.asarray(data.get("MASK_VIS", np.ones_like(self.E2_VIS, dtype=np.float32)))
            if "DF_VIS" in data.files:
                self.DF_VIS = np.asarray(data["DF_VIS"])
            else:
                self.DF_VIS = np.full_like(self.E2_VIS, 2.0, dtype=np.float32)
            
            # ==== 可选的对角监督（用于 vis_2d_diag 模式） ====
            self.E2X_VIS = np.asarray(data["E2X_VIS"]) if "E2X_VIS" in data.files else None
            self.E2Y_VIS = np.asarray(data["E2Y_VIS"]) if "E2Y_VIS" in data.files else None
            self.DFX_VIS = np.asarray(data["DFX_VIS"]) if "DFX_VIS" in data.files else None
            self.DFY_VIS = np.asarray(data["DFY_VIS"]) if "DFY_VIS" in data.files else None
            L = int(self.E2_VIS.shape[0])
            # 2) 窗口参数（可从 config/args 取，取不到就兜底 64/32）
            self.vis_window = int(getattr(self, "vis_window", 0) or 64)
            self.vis_stride = int(getattr(self, "vis_stride", 0) or 32)
            # 3) 生成窗口起点列表（保证至少包含最后一窗）
            starts = list(range(0, max(L - self.vis_window + 1, 0), self.vis_stride))
            if starts == [] and L >= self.vis_window:
                starts = [0]
            # （可选）强制包含尾窗
            if L > 0 and (len(starts) == 0 or starts[-1] != L - self.vis_window):
                if L - self.vis_window >= 0:
                    starts.append(L - self.vis_window)
            self._vis_starts = np.array(starts, dtype=np.int64)
        if self.route == "vis" and getattr(self, "use_step_labels", False) and hasattr(self, "E2_VIS"):
            self._prepare_vis_windows()
        self.Y_acc = _get(data, ["Y_ACC","Y_acc","Yacc"], None)
        self.Y_gyr = _get(data, ["Y_GYR","Y_gyr","Ygyr"], None)
        if self.Y_acc is not None:
            self.Y_acc = self.Y_acc.astype(np.float32)
        if self.Y_gyr is not None:
            self.Y_gyr = self.Y_gyr.astype(np.float32)
        self.N, self.T, self.D = self.X_all.shape
        # ==== 在 IMURouteDataset.__init__ 里（最后）添加 ====
        self._dbg_cnt = {}   # 各类调试信息的计数器
    def _prepare_vis_windows(self):
        """Convert VIS route into fixed-length cleaned windows."""
        X_src = np.asarray(self.X_all, dtype=np.float32)
        if X_src.ndim != 3:
            raise ValueError(f"VIS dataset expects X_all to be 3D, got shape {X_src.shape}")
        N, T_total, D = X_src.shape
        window = int(getattr(self, 'vis_window', 0) or 64)
        stride = int(getattr(self, 'vis_stride', 0) or 32)
        window = max(2, window)
        stride = max(1, stride)
        def _normalize(arr, default_value, dtype):
            if arr is None:
                return [np.full((T_total,), default_value, dtype=dtype) for _ in range(N)]
            a = np.asarray(arr)
            if a.ndim == 3 and a.shape[0] == N and a.shape[1] == T_total:
                if a.shape[2] == 1:
                    a = a[..., 0]
                return [a[i].astype(dtype, copy=False) for i in range(N)]
            if a.ndim == 2:
                if a.shape == (N, T_total):
                    return [a[i].astype(dtype, copy=False) for i in range(N)]
                if a.shape == (T_total, 1) and N == 1:
                    return [a[:, 0].astype(dtype, copy=False)]
                if a.shape == (T_total,) and N == 1:
                    return [a.astype(dtype, copy=False)]
                if a.shape == (N, 1):
                    filled = np.repeat(a.astype(dtype, copy=False), T_total, axis=1)
                    return [filled[i] for i in range(N)]
            if a.ndim == 1 and N == 1 and a.shape[0] == T_total:
                return [a.astype(dtype, copy=False)]
            raise ValueError(f"VIS array shape {a.shape} incompatible with X shape {(N, T_total)}")
        e2_sequences = _normalize(getattr(self, 'E2_VIS', None), np.nan, np.float32)
        mask_sequences = _normalize(getattr(self, 'MASK_VIS', None), 1.0, np.float32)
        df_source = getattr(self, 'DF_VIS', None)
        if df_source is not None:
            from metrics import DF_BY_ROUTE
            df_sequences = _normalize(df_source, float(DF_BY_ROUTE["vis"]), np.float32)
            df_windows = []
        else:
            df_sequences = [None] * N
            df_windows = None
        
        # ==== 对角监督（可选） ====
        e2x_source = getattr(self, 'E2X_VIS', None)
        e2y_source = getattr(self, 'E2Y_VIS', None)
        dfx_source = getattr(self, 'DFX_VIS', None)
        dfy_source = getattr(self, 'DFY_VIS', None)
        
        if e2x_source is not None:
            e2x_sequences = _normalize(e2x_source, np.nan, np.float32)
            e2x_windows = []
        else:
            e2x_sequences = [None] * N
            e2x_windows = None
        
        if e2y_source is not None:
            e2y_sequences = _normalize(e2y_source, np.nan, np.float32)
            e2y_windows = []
        else:
            e2y_sequences = [None] * N
            e2y_windows = None
        
        if dfx_source is not None:
            dfx_sequences = _normalize(dfx_source, 0.0, np.float32)
            dfx_windows = []
        else:
            dfx_sequences = [None] * N
            dfx_windows = None
        
        if dfy_source is not None:
            dfy_sequences = _normalize(dfy_source, 0.0, np.float32)
            dfy_windows = []
        else:
            dfy_sequences = [None] * N
            dfy_windows = None
        
        X_windows, E2_clean_windows, M_clean_windows = [], [], []
        E2_raw_windows, M_raw_windows = [], []
        seq_ids, starts, lengths = [], [], []
        empty_windows = 0
        for seq_idx in range(N):
            x_seq = X_src[seq_idx]
            e2_seq = e2_sequences[seq_idx]
            mask_seq = mask_sequences[seq_idx]
            df_seq = df_sequences[seq_idx]
            e2x_seq = e2x_sequences[seq_idx]
            e2y_seq = e2y_sequences[seq_idx]
            dfx_seq = dfx_sequences[seq_idx]
            dfy_seq = dfy_sequences[seq_idx]
            if x_seq.shape[0] != e2_seq.shape[0] or mask_seq.shape[0] != x_seq.shape[0]:
                raise ValueError("Length mismatch between VIS arrays")
            T_seq = x_seq.shape[0]
            if T_seq == 0:
                continue
            if T_seq <= window:
                seq_starts = [0]
            else:
                seq_starts = list(range(0, T_seq - window + 1, stride))
                tail = T_seq - window
                if seq_starts[-1] != tail:
                    seq_starts.append(tail)
            for start in seq_starts:
                end = min(start + window, T_seq)
                actual_len = end - start
                X_slice = x_seq[start:end].astype(np.float32, copy=False)
                e2_slice_raw = e2_seq[start:end].astype(np.float32, copy=False)
                mask_slice_raw = mask_seq[start:end].astype(np.float32, copy=False)
                if df_seq is not None:
                    df_slice_raw = df_seq[start:end].astype(np.float32, copy=False)
                else:
                    df_slice_raw = None
                
                # 对角监督的切片
                e2x_slice_raw = e2x_seq[start:end].astype(np.float32, copy=False) if e2x_seq is not None else None
                e2y_slice_raw = e2y_seq[start:end].astype(np.float32, copy=False) if e2y_seq is not None else None
                dfx_slice_raw = dfx_seq[start:end].astype(np.float32, copy=False) if dfx_seq is not None else None
                dfy_slice_raw = dfy_seq[start:end].astype(np.float32, copy=False) if dfy_seq is not None else None
                pad = window - actual_len
                if pad > 0:
                    pad_x = X_slice[-1:] if actual_len > 0 else np.zeros((1, D), dtype=np.float32)
                    X_slice = np.concatenate([X_slice, np.repeat(pad_x, pad, axis=0)], axis=0)
                    e2_slice_raw = np.concatenate([e2_slice_raw, np.full((pad,), np.nan, dtype=np.float32)], axis=0)
                    mask_slice_raw = np.concatenate([mask_slice_raw, np.zeros((pad,), dtype=np.float32)], axis=0)
                    if df_slice_raw is not None:
                        df_pad_val = df_slice_raw[-1] if actual_len > 0 else 0.0
                        df_slice_raw = np.concatenate([df_slice_raw, np.full((pad,), df_pad_val, dtype=np.float32)], axis=0)
                    
                    # 对角监督的 padding
                    if e2x_slice_raw is not None:
                        e2x_slice_raw = np.concatenate([e2x_slice_raw, np.full((pad,), np.nan, dtype=np.float32)], axis=0)
                    if e2y_slice_raw is not None:
                        e2y_slice_raw = np.concatenate([e2y_slice_raw, np.full((pad,), np.nan, dtype=np.float32)], axis=0)
                    if dfx_slice_raw is not None:
                        dfx_pad_val = dfx_slice_raw[-1] if actual_len > 0 else 0.0
                        dfx_slice_raw = np.concatenate([dfx_slice_raw, np.full((pad,), dfx_pad_val, dtype=np.float32)], axis=0)
                    if dfy_slice_raw is not None:
                        dfy_pad_val = dfy_slice_raw[-1] if actual_len > 0 else 0.0
                        dfy_slice_raw = np.concatenate([dfy_slice_raw, np.full((pad,), dfy_pad_val, dtype=np.float32)], axis=0)
                e2_numeric = np.nan_to_num(e2_slice_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                valid = (mask_slice_raw > 0.5) & np.isfinite(e2_slice_raw) & (e2_slice_raw < 999.0)
                e2_clean = np.where(valid, e2_numeric, 0.0).astype(np.float32, copy=False)
                mask_clean = valid.astype(np.float32, copy=False)
                if mask_clean.sum() <= 0:
                    empty_windows += 1
                    continue
                X_windows.append(np.ascontiguousarray(X_slice))
                E2_clean_windows.append(np.ascontiguousarray(e2_clean.reshape(window, 1)))
                M_clean_windows.append(np.ascontiguousarray(mask_clean))
                E2_raw_windows.append(np.ascontiguousarray(e2_slice_raw.reshape(window, 1)))
                M_raw_windows.append(np.ascontiguousarray(mask_slice_raw))
                if df_windows is not None and df_slice_raw is not None:
                    df_windows.append(np.ascontiguousarray(df_slice_raw))
                
                # 对角监督的窗口
                if e2x_windows is not None and e2x_slice_raw is not None:
                    e2x_windows.append(np.ascontiguousarray(e2x_slice_raw.reshape(window, 1)))
                if e2y_windows is not None and e2y_slice_raw is not None:
                    e2y_windows.append(np.ascontiguousarray(e2y_slice_raw.reshape(window, 1)))
                if dfx_windows is not None and dfx_slice_raw is not None:
                    dfx_windows.append(np.ascontiguousarray(dfx_slice_raw))
                if dfy_windows is not None and dfy_slice_raw is not None:
                    dfy_windows.append(np.ascontiguousarray(dfy_slice_raw))
                
                seq_ids.append(seq_idx)
                starts.append(int(start))
                lengths.append(int(actual_len))
        if len(X_windows) == 0:
            raise ValueError("No VIS windows with valid steps were constructed")
        self.X_all = np.stack(X_windows, axis=0).astype(np.float32, copy=False)
        self.E2_all = np.stack(E2_clean_windows, axis=0).astype(np.float32, copy=False)
        self.M_all = np.stack(M_clean_windows, axis=0).astype(np.float32, copy=False)
        self._vis_e2_raw_windows = np.stack(E2_raw_windows, axis=0).astype(np.float32, copy=False)
        self._vis_mask_raw_windows = np.stack(M_raw_windows, axis=0).astype(np.float32, copy=False)
        if df_windows is not None and len(df_windows) == len(X_windows):
            self.DF_all = np.stack(df_windows, axis=0).astype(np.float32, copy=False)
        else:
            self.DF_all = None
        
        # 对角监督窗口
        if e2x_windows is not None and len(e2x_windows) == len(X_windows):
            self._vis_e2x_windows = np.stack(e2x_windows, axis=0).astype(np.float32, copy=False)
        else:
            self._vis_e2x_windows = None
        if e2y_windows is not None and len(e2y_windows) == len(X_windows):
            self._vis_e2y_windows = np.stack(e2y_windows, axis=0).astype(np.float32, copy=False)
        else:
            self._vis_e2y_windows = None
        if dfx_windows is not None and len(dfx_windows) == len(X_windows):
            self._vis_dfx_windows = np.stack(dfx_windows, axis=0).astype(np.float32, copy=False)
        else:
            self._vis_dfx_windows = None
        if dfy_windows is not None and len(dfy_windows) == len(X_windows):
            self._vis_dfy_windows = np.stack(dfy_windows, axis=0).astype(np.float32, copy=False)
        else:
            self._vis_dfy_windows = None
        self._vis_seq_ids = np.array(seq_ids, dtype=np.int64)
        self._vis_starts = np.array(starts, dtype=np.int64)
        self._vis_lengths = np.array(lengths, dtype=np.int64)
        self._vis_windows_ready = True
        self._vis_dropped = int(empty_windows)
        self.vis_window = window
        self.vis_stride = stride
        self.N, self.T, self.D = self.X_all.shape
    def __len__(self):
        return int(self.X_all.shape[0])
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        X = np.ascontiguousarray(self.X_all[idx].astype(np.float32, copy=False))
        E2 = self.E2_all[idx]
        M = self.M_all[idx]
        # Route-specific preprocessing
        if self.route == "acc":
            Y = self.Y_acc[idx] if self.Y_acc is not None else None
            if self.x_mode in ("route_only", "imu") and X.shape[-1] >= 6:
                X = X[..., :3]
        elif self.route == "gyr":
            Y = self.Y_gyr[idx] if self.Y_gyr is not None else None
            if self.x_mode in ("route_only", "imu") and X.shape[-1] >= 6:
                X = X[..., 3:6]
        else:  # vis
            Y = None
            if self.x_mode == "visual" and X.shape[-1] > 4:
                pass
        if self.route == "vis":
            if getattr(self, "_vis_windows_ready", False):
                E2 = np.ascontiguousarray(E2.astype(np.float32, copy=False))
                if E2.ndim == 1:
                    E2 = E2[:, None]
                mask_clean = np.ascontiguousarray(M.astype(np.float32, copy=False).reshape(-1))
                e2_raw = np.asarray(self._vis_e2_raw_windows[idx], dtype=np.float32).reshape(-1)
                mask_raw = np.asarray(self._vis_mask_raw_windows[idx], dtype=np.float32).reshape(-1)
                T_raw = int(mask_raw.shape[0])
                T_clean = int(mask_clean.shape[0])
                if T_clean < 2 and _dbg_limit(self._dbg_cnt, "vis_T_lt_2"):
                    print(f"[VIS][warn] window T={T_clean} after cleaning (raw T={T_raw}). T>=2 is required; please check window/stride and dropping the t=0 step")
                msum = float(mask_clean.sum())
                if msum <= 0 and _dbg_limit(self._dbg_cnt, "vis_all_masked"):
                    try:
                        nan_cnt = int(np.isnan(e2_raw).sum())
                        inf_cnt = int(np.isinf(e2_raw).sum())
                        big_cnt = int((e2_raw >= 999).sum())
                        m0_cnt = int((mask_raw <= 0.5).sum())
                    except Exception:
                        nan_cnt = inf_cnt = big_cnt = m0_cnt = -1
                    print(f"[VIS][error] all steps masked out after cleaning. T={T_clean}  nan={nan_cnt} inf={inf_cnt} e2>=999={big_cnt} mask_raw<=0.5={m0_cnt}")
                if T_clean >= 2 and mask_clean.shape[0] == 1 and _dbg_limit(self._dbg_cnt, "vis_mask_B1"):
                    print(f"[VIS][warn] mask_step is shape (1,) while T={T_clean}. the mask will broadcast but should be per-step (T,) from data source")
                M = mask_clean
            else:
                e2_flat = np.ascontiguousarray(E2.reshape(-1).astype(np.float32, copy=False))
                m_flat = np.ascontiguousarray(M.reshape(-1).astype(np.float32, copy=False))
                valid = (m_flat > 0.5)
                valid &= np.isfinite(e2_flat)
                valid &= (e2_flat < 999.0)
                e2_flat = np.nan_to_num(e2_flat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                e2_clean = np.where(valid, e2_flat, 0.0).astype(np.float32, copy=False)
                mask_clean = valid.astype(np.float32, copy=False)
                E2 = e2_clean[:, None]
                M = mask_clean
        out = {
            "X": torch.from_numpy(np.ascontiguousarray(X)),
            "MASK": torch.from_numpy(np.ascontiguousarray(M.astype(np.float32, copy=False))),
            "E2": torch.from_numpy(np.ascontiguousarray(E2.astype(np.float32, copy=False))),
            "use_step_labels": bool(self.use_step_labels),
        }
        if self.route == "vis" and getattr(self, '_vis_windows_ready', False):
            out["E2_VIS_RAW"] = torch.from_numpy(
                np.ascontiguousarray(self._vis_e2_raw_windows[idx].astype(np.float32, copy=False))
            )
            out["MASK_VIS_RAW"] = torch.from_numpy(
                np.ascontiguousarray(self._vis_mask_raw_windows[idx].astype(np.float32, copy=False))
            )
            # 可选的 per-axis 监督（若生成器写了这些键）
            if hasattr(self, "_vis_e2x_windows") and self._vis_e2x_windows is not None:
                out["E2X"] = torch.from_numpy(
                    np.ascontiguousarray(self._vis_e2x_windows[idx].astype(np.float32, copy=False))
                )
            if hasattr(self, "_vis_e2y_windows") and self._vis_e2y_windows is not None:
                out["E2Y"] = torch.from_numpy(
                    np.ascontiguousarray(self._vis_e2y_windows[idx].astype(np.float32, copy=False))
                )
            if hasattr(self, "_vis_dfx_windows") and self._vis_dfx_windows is not None:
                out["DFX"] = torch.from_numpy(
                    np.ascontiguousarray(self._vis_dfx_windows[idx].astype(np.float32, copy=False))
                )
            if hasattr(self, "_vis_dfy_windows") and self._vis_dfy_windows is not None:
                out["DFY"] = torch.from_numpy(
                    np.ascontiguousarray(self._vis_dfy_windows[idx].astype(np.float32, copy=False))
                )
        if self.DF_all is not None:
            out["DF"] = torch.from_numpy(np.ascontiguousarray(self.DF_all[idx].astype(np.float32, copy=False)))
        if hasattr(self, 'Y_anchor') and self.Y_anchor is not None:
            out["Y_anchor"] = torch.from_numpy(np.ascontiguousarray(self.Y_anchor[idx].astype(np.float32, copy=False)))
        if Y is not None:
            out["Y"] = torch.from_numpy(np.ascontiguousarray(Y.astype(np.float32, copy=False)))
        else:
            out["Y"] = torch.zeros_like(out["MASK"])
        return out
# === GNSS 数据集（ENU三维） ===

class GNSDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z['X'].astype(np.float32)     # (N, T, Din)
        self.Y = z['Y'].astype(np.float32)     # (N, T, 3)  ENU误差
        self.mask = z['mask'].astype(bool)     # (N, T, 3)
        self.meta = z.get('meta', None)
        assert self.X.shape[0] == self.Y.shape[0] == self.mask.shape[0]
        assert self.Y.shape[-1] == 3, "GNS Y should be (..,3) for ENU"
    def __len__(self):  
        return self.X.shape[0]
    def __getitem__(self, i):
        y_axes = self.Y[i].astype(np.float32)            # (T,3)
        e2_axes = (y_axes ** 2).astype(np.float32)       # (T,3)
        e2_sum  = e2_axes.sum(axis=-1, keepdims=True)    # (T,1)  ← 训练/评测用
        m_axes  = self.mask[i].astype(np.float32)        # (T,3)
        m_any   = (m_axes > 0.5).all(axis=-1, keepdims=True).astype(np.float32)  # (T,1)
        return {
            "X": torch.from_numpy(self.X[i]),            # (T,Din)
            "E2": torch.from_numpy(e2_sum),              # (T,1)  ← 配合 nll_iso3_e2
            "MASK": torch.from_numpy(m_any),             # (T,1)  ← 与上对齐
            # 下面是作图/逐维统计需要的"富信息"
            "Y": torch.from_numpy(y_axes),               # (T,3)
            "MASK_AXES": torch.from_numpy(m_axes),       # (T,3)
            "E2_AXES": torch.from_numpy(e2_axes),        # (T,3)
        }

def build_dataset(route: str, npz_path: str):
    """数据集工厂函数"""
    route = route.lower()
    if route in ('acc', 'gyr', 'vis'):
        return IMURouteDataset(npz_path, route=route, x_mode="both")
    elif route == 'gns':
        return GNSDataset(npz_path)
    else:
        raise ValueError(f"Unknown route {route}")

def build_loader(npz_path, route="acc", x_mode="both",
                 batch_size=32, shuffle=True, num_workers=0,
                 generator=None, worker_init_fn=None):
    if route.lower() == 'gns':
        ds = build_dataset(route, npz_path)
    else:
        ds = IMURouteDataset(npz_path, route=route, x_mode=x_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                    pin_memory=True, generator=generator, worker_init_fn=worker_init_fn)
    return ds, dl
