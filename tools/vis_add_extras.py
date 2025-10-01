#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Augment VIS NPZ files with inexpensive statistics."""
import argparse
import numpy as np


def robust_stats(x, axis=-1):
    x = np.asarray(x)
    med = np.nanmedian(x, axis=axis)
    p50 = med
    p90 = np.nanpercentile(x, 90, axis=axis)
    mu = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    return mu, std, p50, p90


def grid_entropy(points_xy, H=480, W=752, bins=8):
    if points_xy is None or len(points_xy) == 0:
        return np.nan
    pts = np.asarray(points_xy)
    x = np.clip(pts[:, 0], 0, W - 1)
    y = np.clip(pts[:, 1], 0, H - 1)
    gx = np.minimum((x / W * bins).astype(int), bins - 1)
    gy = np.minimum((y / H * bins).astype(int), bins - 1)
    hist = np.zeros((bins, bins), dtype=np.float64)
    for i in range(len(x)):
        hist[gy[i], gx[i]] += 1.0
    prob = hist / (hist.sum() + 1e-12)
    mask = prob > 0
    return -(prob[mask] * np.log(prob[mask])).sum()


def main():
    parser = argparse.ArgumentParser(description="Append cheap VIS statistics to an existing NPZ")
    parser.add_argument('--in', dest='npz_in', required=True, help='Input NPZ path')
    parser.add_argument('--out', dest='npz_out', required=True, help='Output NPZ path')
    parser.add_argument('--img_h', type=int, default=480)
    parser.add_argument('--img_w', type=int, default=752)
    args = parser.parse_args()

    data = np.load(args.npz_in, allow_pickle=True)
    keys = set(data.keys())

    X = data['X_VIS'] if 'X_VIS' in keys else None
    if X is None:
        print('[warn] no X_VIS in NPZ; copy only.')
        np.savez_compressed(args.npz_out, **data)
        return

    T = X.shape[0]

    inlier_mask = data['inlier_mask'] if 'inlier_mask' in keys else None
    flow_u = data['flow_u'] if 'flow_u' in keys else None
    flow_v = data['flow_v'] if 'flow_v' in keys else None
    keypoints = data['keypoints'] if 'keypoints' in keys else None
    inlier_ratio = data['inlier_ratio'] if 'inlier_ratio' in keys else None
    flow_mean = data['flow_mean'] if 'flow_mean' in keys else None
    flow_std = data['flow_std'] if 'flow_std' in keys else None

    extras = []
    for t in range(T):
        if inlier_mask is not None:
            mask = inlier_mask[t].astype(bool)
            track_count = float(mask.sum())
            outlier_ratio = float(1.0 - (mask.mean() if mask.size > 0 else 0.0))
        else:
            ir = float(inlier_ratio[t]) if inlier_ratio is not None else np.nan
            track_count = np.nan
            outlier_ratio = float(1.0 - ir) if not np.isnan(ir) else np.nan

        if flow_u is not None and flow_v is not None:
            fu = flow_u[t]
            fv = flow_v[t]
            if inlier_mask is not None:
                mask = inlier_mask[t].astype(bool)
                fu = fu[mask]
                fv = fv[mask]
            flow_mag = np.sqrt(fu ** 2 + fv ** 2)
            mu, sd, p50, p90 = robust_stats(flow_mag, axis=0)
            var_u = np.nanvar(fu)
            var_v = np.nanvar(fv)
            aniso = float((var_u + 1e-12) / (var_v + 1e-12))
        else:
            mu = float(flow_mean[t]) if flow_mean is not None else np.nan
            sd = float(flow_std[t]) if flow_std is not None else np.nan
            p50, p90, aniso = np.nan, np.nan, np.nan

        if keypoints is not None:
            pts = keypoints[t] if keypoints.ndim == 3 else keypoints
            ent = float(grid_entropy(pts, H=args.img_h, W=args.img_w, bins=8))
        else:
            ent = np.nan

        extras.append([track_count, outlier_ratio, mu, sd, p50, p90, aniso, ent])

    extras = np.asarray(extras, dtype=np.float32)
    X_new = np.concatenate([X, extras], axis=1)

    out_dict = {k: data[k] for k in keys}
    out_dict['X_VIS'] = X_new
    feature_names = list(data['vis_feature_names']) if 'vis_feature_names' in keys else []
    feature_names.extend(['track_count', 'outlier_ratio', 'flow_mu', 'flow_sd', 'flow_p50', 'flow_p90', 'flow_aniso', 'kp_grid_entropy'])
    out_dict['vis_feature_names'] = np.asarray(feature_names)

    np.savez_compressed(args.npz_out, **out_dict)
    print('[ok] X_VIS columns: %d -> %d' % (X.shape[1], X_new.shape[1]))


if __name__ == '__main__':
    main()
