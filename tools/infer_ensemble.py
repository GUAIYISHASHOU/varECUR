#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import torch

# Ensure relative imports work when run from repo root
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import load_config_file, to_device
from dataset import build_loader
from models import IMURouteModel
from metrics import z2_cov_studentt_diag_axes, C68_GAUSS, C95_GAUSS
from imu_oof.calibrator import SA3AffineCalibrator


def build_model_from_ckpt(ckpt: dict, device: str = 'cpu'):
    cfg = ckpt.get('cfg', {})
    model_cfg = cfg.get('model', {})
    tr = cfg.get('train', {})
    d_in = ckpt.get('d_in', None)
    if d_in is None:
        raise ValueError('Checkpoint missing d_in')
    d_out = ckpt.get('d_out', 1)
    variance_param = ckpt.get('variance_param', tr.get('variance_param', 'direct'))
    aniso = ckpt.get('aniso', None)
    use_bounded = tr.get('use_bounded', True)
    model = IMURouteModel(
        d_in=d_in,
        d_model=model_cfg.get('d_model', 128),
        d_out=d_out,
        n_tcn=model_cfg.get('n_tcn', 4),
        kernel_size=model_cfg.get('kernel_size', 3),
        n_layers_tf=model_cfg.get('n_layers_tf', 0),
        n_heads=model_cfg.get('n_heads', 4),
        dropout=model_cfg.get('dropout', 0.1),
        logv_min=tr.get('logv_min', -8.0),
        logv_max=tr.get('logv_max', 6.0),
        use_bounded=use_bounded,
        variance_param=variance_param,
        aniso=aniso,
    ).to(device)
    state_dict = ckpt.get('state_dict', ckpt.get('model', None))
    if state_dict is None:
        raise ValueError('Checkpoint missing state_dict')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description='Ensemble inference across best_fold*.pt and evaluate on a test NPZ')
    ap.add_argument('--models_glob', type=str, default='runs/oof_acc/best_fold*.pt')
    ap.add_argument('--models', type=str, default=None, help='Space/comma-separated list of model paths')
    ap.add_argument('--npz', type=str, required=True)
    ap.add_argument('--route', choices=['acc','gyr'], default='acc')
    ap.add_argument('--config', type=str, default='config_oof.yaml')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--calibrator_json', type=str, default=None)
    ap.add_argument('--apply_before_ensemble', action='store_true', help='Apply calibrator per-model before averaging')
    args = ap.parse_args()

    cfg = load_config_file(args.config)
    device = args.device or cfg.get('common', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = cfg.get('common', {}).get('num_workers', 0)

    if args.models is not None:
        paths = [p.strip() for p in args.models.replace(',', ' ').split() if p.strip()]
    else:
        paths = sorted(glob.glob(args.models_glob))
    assert paths, f'No models found by {args.models or args.models_glob}'

    cal = None
    if args.calibrator_json and Path(args.calibrator_json).exists():
        cal = SA3AffineCalibrator.load(args.calibrator_json)

    pred_list = []
    e2_ref = None
    mask_ref = None
    nu = None
    logv_min = -8.0
    logv_max = 6.0

    for i, p in enumerate(paths):
        ckpt = torch.load(p, map_location='cpu')
        scaler = ckpt.get('scaler', None)
        d_out = ckpt.get('d_out', 1)
        tr = ckpt.get('cfg', {}).get('train', {})
        nu = nu if nu is not None else tr.get('nu', ckpt.get('nu', 5.0))
        logv_min = tr.get('logv_min', ckpt.get('logv_min', logv_min))
        logv_max = tr.get('logv_max', ckpt.get('logv_max', logv_max))

        dl = build_loader(args.npz, args.route, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, scaler=scaler)
        model = build_model_from_ckpt(ckpt, device=device)

        all_lv, all_e2, all_mask = [], [], []
        with torch.no_grad():
            for batch in dl:
                batch = to_device(batch, device)
                x = batch['x']
                e2 = batch['e2']
                m  = batch['mask']
                logv = model(x)
                all_lv.append(logv.detach().cpu().numpy())
                if e2_ref is None:
                    all_e2.append(e2.cpu().numpy())
                    all_mask.append(m.cpu().numpy())
        lv = np.concatenate(all_lv, 0)  # (N,T,3)
        if d_out != 3 or lv.shape[-1] != 3:
            raise RuntimeError(f'Model {p} does not output 3 channels; got shape {lv.shape}')
        pred_list.append(lv)
        if e2_ref is None:
            e2_ref = np.concatenate(all_e2, 0)
            mask_ref = np.concatenate(all_mask, 0)

    # Optional: per-model calibration before ensemble
    if cal is not None and args.apply_before_ensemble:
        pred_list = [cal.apply(lv) for lv in pred_list]

    # Ensemble in variance domain
    s2_list = [np.exp(lv) for lv in pred_list]
    s2_ens = np.mean(np.stack(s2_list, axis=0), axis=0)
    logv_ens = np.log(np.clip(s2_ens, 1e-12, None))

    # Or apply calibrator after ensemble
    if cal is not None and not args.apply_before_ensemble:
        logv_ens = cal.apply(logv_ens)

    # Metrics (Student-t diag axes)
    e_axes = np.sqrt(np.clip(e2_ref, 0.0, None)).astype(np.float32)
    mask_axes = np.repeat(mask_ref[..., None], 3, axis=-1)
    z2_mean, cov68, cov95 = z2_cov_studentt_diag_axes(logv_ens, e_axes, mask_axes, nu=nu, logv_min=logv_min, logv_max=logv_max)

    # Spearman (var_mean vs e2_mean)
    s2 = np.exp(logv_ens)
    v_eff = s2 * (nu / (nu - 2.0)) if (nu is not None and nu > 2.0) else s2
    e2sum = e2_ref.sum(axis=-1)
    var_mean = v_eff.mean(axis=-1)
    m_bool = mask_ref > 0.5
    if m_bool.any():
        ra = np.argsort(np.argsort(e2sum[m_bool].reshape(-1)))
        rb = np.argsort(np.argsort(var_mean[m_bool].reshape(-1)))
        spearman = float(np.corrcoef(ra, rb)[0, 1])
    else:
        spearman = 0.0

    print('\n[Ensemble metrics]')
    print(f'  models: {len(paths)}')
    print(f'  z2_mean: {z2_mean:.4f}')
    print(f'  cov68:   {cov68:.4f}')
    print(f'  cov95:   {cov95:.4f}')
    print(f'  spearman:{spearman:.4f}')


if __name__ == '__main__':
    main()
