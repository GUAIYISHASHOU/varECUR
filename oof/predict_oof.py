#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import re
import json
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when running from subdirectory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import build_dataset_from_seqs
from models import IMURouteModel


def load_splits(splits_path: str):
    with open(splits_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'folds' in raw:
        folds_list = raw['folds']
        return {str(i): {"train": folds_list[i]["train"], "val": folds_list[i]["val"]}
                for i in range(len(folds_list))}
    return raw


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
    ap = argparse.ArgumentParser(description='Predict OOF per fold and dump flattened arrays')
    ap.add_argument('--runs_dir', type=str, default='runs/oof_acc')
    ap.add_argument('--out_dir', type=str, default=None)
    ap.add_argument('--splits', type=str, default='splits_oof.json')
    ap.add_argument('--route', type=str, choices=['acc','gyr'], default='acc')
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    runs_dir = args.runs_dir
    out_dir = args.out_dir or os.path.join(runs_dir, 'oof')
    os.makedirs(out_dir, exist_ok=True)

    split_map = load_splits(args.splits)

    for p in sorted(glob.glob(os.path.join(runs_dir, 'best_fold*.pt'))):
        m = re.search(r"best_fold(\d+)\.pt$", os.path.basename(p))
        if not m:
            continue
        fold = int(m.group(1))
        ckpt = torch.load(p, map_location='cpu')
        cfg = ckpt.get('cfg', {})
        seq_dir = cfg.get('oof', {}).get('seq_npz_dir', None)
        if not seq_dir:
            raise ValueError('seq_npz_dir not found in checkpoint cfg[oof]')
        scaler = ckpt.get('scaler', None)

        val_seqs = split_map.get(str(fold), {}).get('val', None)
        if val_seqs is None:
            raise ValueError(f'Fold {fold} not found in splits: {args.splits}')

        ds = build_dataset_from_seqs(seq_dir, val_seqs, args.route, scaler=scaler)
        dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

        model = build_model_from_ckpt(ckpt, device=args.device)

        d_out = ckpt.get('d_out', 1)
        dist = ckpt.get('dist', 'gauss')
        logv_all, logv_axes_all, e2_all = [], [], []
        with torch.no_grad():
            for batch in dl:
                x = batch['x'].to(args.device)
                e2 = batch['e2'].cpu().numpy()  # (B,T,3)
                logv = model(x)
                if d_out == 3 and logv.dim() == 3 and logv.size(-1) == 3:
                    # 保存轴向logv与等效各向同性logv
                    lv_axes = logv.detach().cpu().numpy()  # (B,T,3)
                    lv_iso = np.log(np.clip(np.exp(lv_axes), 1e-12, None).mean(axis=-1))  # (B,T)
                    logv_axes_all.append(lv_axes.reshape(-1, 3))
                    logv_all.append(lv_iso.reshape(-1))
                else:
                    if logv.dim() == 3 and logv.size(-1) == 1:
                        logv = logv.squeeze(-1)
                    logv = logv.detach().cpu().numpy()  # (B,T)
                    logv_all.append(logv.reshape(-1))
                e2_all.append(e2.reshape(-1, 3))

        meta = {
            'd_out': int(d_out),
            'dist': dist,
            'nu': ckpt.get('nu', None),
            'logv_min': ckpt.get('logv_min', -8.0),
            'logv_max': ckpt.get('logv_max', 6.0),
            'label_version': ckpt.get('label_version', None),
            'variance_param': ckpt.get('variance_param', 'direct'),
            'aniso': ckpt.get('aniso', None),
        }

        out_path = os.path.join(out_dir, f'fold{fold}_pred.npz')
        if d_out == 3 and len(logv_axes_all) > 0:
            np.savez(out_path,
                     logv=np.concatenate(logv_all),
                     logv_axes=np.concatenate(logv_axes_all),
                     e2=np.concatenate(e2_all),
                     meta=json.dumps(meta))
        else:
            np.savez(out_path,
                     logv=np.concatenate(logv_all),
                     e2=np.concatenate(e2_all),
                     meta=json.dumps(meta))
        print(f'[fold {fold}] saved {out_path}')


if __name__ == '__main__':
    main()
