# tools/check_npz.py
import argparse, math
from pathlib import Path
import numpy as np

def q(x, ps=(0.5, 0.68, 0.95, 0.99)):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [np.nan]*len(ps)
    return np.quantile(x, ps).tolist()

def try_get(arrs, names):
    for n in names:
        if n in arrs: return arrs[n]
    return None

def get_e2(arrs):
    # 兼容多种打包方式：优先直接用 e2；否则用 err 还原 e2=sum(err^2,axis=-1)
    e2 = try_get(arrs, ["e2","E2","e_sq","err2"])
    if e2 is not None: return np.asarray(e2)
    err = try_get(arrs, ["err","E","res","residual"])
    if err is not None:
        err = np.asarray(err)
        # err 形状可能是 [..., 3] 或 [..., 2]；把最后一维平方求和
        return np.sum(err**2, axis=-1)
    return None

def get_mask(arrs):
    m = try_get(arrs, ["m","mask","valid","valid_mask"])
    if m is None: return None
    m = np.asarray(m)
    # 允许 float/bool，转成 0/1
    if m.dtype != np.bool_:
        m = (m > 0).astype(np.uint8)
    return m

def check_one(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    arrs = {k: d[k] for k in d.files}

    warn = []

    # 1) 基础体检
    keys = sorted(arrs.keys())
    shapes = {k: tuple(arrs[k].shape) for k in keys}
    dtypes = {k: str(arrs[k].dtype) for k in keys}

    # 2) NaN/Inf 检查 & 简要统计
    naninf = {}
    basic_stats = {}
    for k,v in arrs.items():
        a = np.asarray(v)
        if a.size == 0: 
            naninf[k] = (0,0); basic_stats[k] = None; 
            continue
        n_nan = np.isnan(a).sum()
        n_inf = np.isinf(a).sum()
        naninf[k] = (int(n_nan), int(n_inf))
        if a.dtype.kind in "fiu":
            aa = a[np.isfinite(a)]
            if aa.size:
                basic_stats[k] = dict(min=float(aa.min()), max=float(aa.max()),
                                      mean=float(aa.mean()), std=float(aa.std()))
            else:
                basic_stats[k] = None

    # 3) 时间戳/单调性（若存在）
    ts = try_get(arrs, ["ts","time","timestamp","timestamps"])
    ts_info = None
    if ts is not None:
        t = np.asarray(ts).astype(np.float64).reshape(-1)
        if t.size >= 2:
            dt = np.diff(t)
            mono = bool(np.all(dt>0))
            ts_info = dict(n=len(t), mono=mono,
                           dt_median=float(np.median(dt)),
                           dt_p10=float(np.quantile(dt,0.1)),
                           dt_p90=float(np.quantile(dt,0.9)))
            if not mono:
                warn.append("timestamps are NOT strictly increasing")

    # 4) 掩码合规性（若存在）
    m = get_mask(arrs)
    mask_info = None
    if m is not None:
        mm = m.reshape(-1)
        frac01 = (mm==0).mean(), (mm==1).mean()
        uniq = np.unique(mm)
        mask_info = dict(shape=tuple(m.shape), unique=uniq.tolist(),
                         frac0=float(frac01[0]), frac1=float(frac01[1]))
        if not set(uniq).issubset({0,1}):
            warn.append("mask contains values other than {0,1}")
        if mm.mean()==0.0:
            warn.append("mask is all zeros")

    # 5) e² 时间异方差检查（关键）
    e2 = get_e2(arrs)
    e2_info = None
    if e2 is not None:
        e2 = np.asarray(e2)
        # 支持 [B,T] 或 [T]；统一成二维 [N,T]
        if e2.ndim==1:
            E = e2[None,:]
        elif e2.ndim>=2:
            E = e2.reshape(e2.shape[0], -1)
        else:
            E = e2

        # 每段时间维的方差
        var_t = np.var(E, axis=1)
        frac_const = float(np.mean(var_t < 1e-8))
        p50,p68,p95,p99 = q(E, (0.5,0.68,0.95,0.99))
        e2_info = dict(shape=tuple(e2.shape),
                       var_t_median=float(np.median(var_t)),
                       frac_time_const=frac_const,
                       q50=float(p50), q68=float(p68), q95=float(p95), q99=float(p99))
        if frac_const > 0.5:
            warn.append(f"e² is nearly constant along time in {frac_const*100:.1f}% windows")

        if not math.isnan(p68) and abs(p68-1.17)<0.05 and abs(p95-2.60)<0.10:
            # df=3 的 χ²/df 分位数 ~ 1.17/2.60；若跟它过于接近 + var_t 很小，通常是“常数 σ² + 常数 e²”
            if np.median(var_t) < 1e-6:
                warn.append("z² quantiles look collapsed near theoretical line with tiny time-variance")

    return {
        "path": str(npz_path),
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "naninf": naninf,
        "ts": ts_info,
        "mask": mask_info,
        "e2": e2_info,
        "warn": warn
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="目录或单个 .npz 文件")
    ap.add_argument("--max_files", type=int, default=50, help="最多检查多少个文件")
    args = ap.parse_args()

    p = Path(args.root)
    files = [p] if p.is_file() else sorted(list(p.rglob("*.npz")))
    files = files[:args.max_files]
    if not files:
        print("No .npz found."); return

    bad = 0
    for f in files:
        R = check_one(f)
        print("\n====", R["path"], "====")
        print("keys:", R["keys"])
        print("shapes:", R["shapes"])
        if R["ts"]:   print("ts:", R["ts"])
        if R["mask"]: print("mask:", R["mask"])
        if R["e2"]:   print("e2:", R["e2"])
        if any(R["naninf"][k][0] or R["naninf"][k][1] for k in R["naninf"]):
            print("NaN/Inf:", {k:v for k,v in R["naninf"].items() if v!=(0,0)})
        if R["warn"]:
            bad += 1
            print("WARN:", "; ".join(R["warn"]))
    print(f"\nChecked {len(files)} files. Files with warnings: {bad}")

if __name__=="__main__":
    main()
