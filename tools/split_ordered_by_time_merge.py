# -*- coding: utf-8 -*-
import os, glob, math, argparse
import numpy as np

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    # 取"窗口个数"N：所有首维一致的最小长度
    N = None
    for k in keys:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            N = len(arr) if N is None else min(N, len(arr))
    return data, keys, N

def pick_time_vector(data, N):
    # 优先时间戳；若有 t0/t1 取中点；都没有就用索引
    keys = {k.lower(): k for k in data.keys()}
    if 't0' in keys and 't1' in keys and len(data[keys['t0']]) == N and len(data[keys['t1']]) == N:
        t = (data[keys['t0']].astype(np.float64) + data[keys['t1']].astype(np.float64)) * 0.5
    else:
        for cand in ['ts','time','times','timestamp','timestamps','t']:
            if cand in keys and len(data[keys[cand]]) == N:
                t = data[keys[cand]].astype(np.float64)
                break
        else:
            t = np.arange(N, dtype=np.float64)
    return t

def masks_time_ordered(t, train_ratio, val_ratio, embargo_sec):
    t = t.astype(np.float64)
    # 自适应时间单位：用中位 dt 把秒转为"时间轴单位"
    dt = np.median(np.diff(np.sort(t))) if len(t) > 1 else 1.0
    emb = embargo_sec / dt
    # 切点用分位数更稳
    cut_train = np.quantile(t, train_ratio)
    cut_val   = np.quantile(t, train_ratio + val_ratio)
    m_tr = (t <= cut_train - emb)
    m_va = (t >  cut_train + emb) & (t <= cut_val - emb)
    m_te = (t >  cut_val   + emb)
    # 互斥检查
    if np.any((m_tr & m_va) | (m_tr & m_te) | (m_va & m_te)):
        raise RuntimeError("mask overlap; check timestamps/embargo.")
    return m_tr, m_va, m_te

def append_split(buffers, data, keys, N, mask, seq_name):
    for k in keys:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == N:
            buffers.setdefault(k, []).append(arr[mask])
    # 额外附加一个 seq_id（可选）
    if 'seq_id' not in buffers:
        buffers['seq_id'] = []
    buffers['seq_id'].append(np.full(mask.sum(), hash(seq_name) & 0x7fffffff, dtype=np.int64))

def save_merged(buffers, out_path):
    merged = {}
    for k, parts in buffers.items():
        # 兼容 object/变长：尽量按轴0拼；失败就存为 object
        try:
            merged[k] = np.concatenate(parts, axis=0)
        except Exception:
            merged[k] = np.array(parts, dtype=object)
    np.savez_compressed(out_path, **merged)
    print(f"[save] {out_path}  N={len(next(iter(merged.values())))}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="folder of per-sequence npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio",   type=float, default=0.15)
    ap.add_argument("--embargo_sec", type=float, default=3.2)  # 64@20Hz → 3.2s
    ap.add_argument("--pattern", type=str, default="*.npz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No NPZ found under {args.in_dir}")

    tr_buf, va_buf, te_buf = {}, {}, {}
    total = {"tr":0, "va":0, "te":0}

    for f in files:
        data, keys, N = load_npz(f)
        if N is None or N == 0:
            print(f"[skip] {f} (no window-like arrays)")
            continue
        t = pick_time_vector(data, N)
        m_tr, m_va, m_te = masks_time_ordered(t, args.train_ratio, args.val_ratio, args.embargo_sec)
        seq = os.path.splitext(os.path.basename(f))[0]
        append_split(tr_buf, data, keys, N, m_tr, seq); total["tr"] += m_tr.sum()
        append_split(va_buf, data, keys, N, m_va, seq); total["va"] += m_va.sum()
        append_split(te_buf, data, keys, N, m_te, seq); total["te"] += m_te.sum()
        print(f"[seq] {seq:20s}  N={N:6d}  tr/va/te={m_tr.sum():5d}/{m_va.sum():5d}/{m_te.sum():5d}")

    save_merged(tr_buf, os.path.join(args.out_dir, "train.npz"))
    save_merged(va_buf, os.path.join(args.out_dir, "val.npz"))
    save_merged(te_buf, os.path.join(args.out_dir, "test.npz"))
    print(f"[done] total  tr/va/te={total['tr']}/{total['va']}/{total['te']}")

if __name__ == "__main__":
    main()

