from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import load_config_file, to_device
from dataset import build_loader, IMUDataset
from models import IMURouteModel
from metrics import route_metrics_imu, z2_cov_studentt_diag_axes, C68_GAUSS, C95_GAUSS, studentt_z2_threshold
from tools.plot_results_imu import vis_plot_all


def main():
    ap = argparse.ArgumentParser("Evaluate a trained single-route model")
    ap.add_argument("--route", choices=["acc","gyr"], required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--calibrator_json", type=str, default=None)
    ap.add_argument("--plots_dir", type=str, default=None)
    ap.add_argument("--dump_preds_npz", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config_file(args.config)
    device = cfg.get("common", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    probe = IMUDataset(args.npz, args.route)
    d_in = probe.D
    
    # 从 checkpoint 中读取模型配置
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        d_out = ckpt.get("d_out", 1)
        logv_min = ckpt.get("logv_min", -12.0)
        logv_max = ckpt.get("logv_max", 6.0)
    else:
        # 直接是state_dict
        state_dict = ckpt
        d_out = 1
        logv_min = -12.0
        logv_max = 6.0
    
    # 获取参数（兼容旧模型）
    use_bounded = ckpt.get("use_bounded", True) if isinstance(ckpt, dict) else True
    variance_param = ckpt.get("variance_param", "direct") if isinstance(ckpt, dict) else "direct"
    aniso = ckpt.get("aniso", None) if isinstance(ckpt, dict) else None
    scaler = ckpt.get("scaler", None) if isinstance(ckpt, dict) else None
    # 分布与自由度（若 checkpoint 未包含，则回退到配置）
    dist = ckpt.get("dist", "gauss") if isinstance(ckpt, dict) else "gauss"
    nu = (ckpt.get("nu", cfg.get("train", {}).get("nu", 5.0))
          if dist == "studentt" else None)
    
    model = IMURouteModel(d_in=d_in, d_out=d_out, d_model=cfg.get("model",{}).get("d_model",128),
                          n_tcn=cfg.get("model",{}).get("n_tcn",4), kernel_size=cfg.get("model",{}).get("kernel_size",3),
                          n_layers_tf=cfg.get("model",{}).get("n_layers_tf",0), n_heads=cfg.get("model",{}).get("n_heads",4),
                          dropout=cfg.get("model",{}).get("dropout",0.1),
                          logv_min=logv_min, logv_max=logv_max, use_bounded=use_bounded,
                          variance_param=variance_param, aniso=aniso).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 加载校准器（仅各向异性SA3）
    calibrator = None
    if args.calibrator_json and Path(args.calibrator_json).exists():
        from imu_oof.calibrator import SA3AffineCalibrator
        calibrator = SA3AffineCalibrator.load(args.calibrator_json)

    # 前向
    loader = build_loader(args.npz, args.route, batch_size=16, shuffle=False,
                          num_workers=cfg.get("common",{}).get("num_workers",4),
                          scaler=scaler)
    all_e2, all_logv, all_mask = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            x = batch["x"]
            e2 = batch["e2"]
            m  = batch["mask"]
            logv = model(x)
            
            # 仅当输出为三轴时应用各向异性校准
            if calibrator is not None and d_out == 3:
                logv = calibrator.apply(logv)
            
            all_e2.append(e2.cpu().numpy())
            all_logv.append(logv.cpu().numpy())
            all_mask.append(m.cpu().numpy())

    e2 = np.concatenate(all_e2, 0)
    logv = np.concatenate(all_logv, 0)
    mask = np.concatenate(all_mask, 0)
    m_bool = mask > 0.5
    if d_out == 3:
        pred_lv_flat = logv[m_bool]
        gt_lv_flat = np.log(np.clip(e2[m_bool], 1e-12, None))
    else:
        lv = logv.squeeze(-1) if logv.ndim == 3 else logv
        pred_lv_flat = lv[m_bool].reshape(-1, 1)
        e2sum = e2.sum(axis=-1)
        gt_lv_flat = np.log(np.clip(e2sum[m_bool] / 3.0, 1e-12, None)).reshape(-1, 1)

    if args.dump_preds_npz:
        out_dir_npz = os.path.dirname(args.dump_preds_npz)
        if out_dir_npz:
            Path(out_dir_npz).mkdir(parents=True, exist_ok=True)
        np.savez(args.dump_preds_npz,
                 pred_logvar=pred_lv_flat,
                 gt_logvar=gt_lv_flat,
                 idx=np.arange(pred_lv_flat.shape[0], dtype=np.int32))

    # 指标
    if d_out == 3:
        e_axes = np.sqrt(np.clip(e2, 0.0, None)).astype(np.float32)
        z2_mean, cov68, cov95 = z2_cov_studentt_diag_axes(logv, e_axes, mask[..., None].repeat(3, axis=-1), nu, logv_min, logv_max)
        s2 = np.exp(logv)  # logv 已经通过 tanh 有界，无需 clip
        v_eff = s2 * (nu / (nu - 2.0))
        e2sum = e2.sum(axis=-1)
        v_sum = v_eff.sum(axis=-1)
        m_bool = mask > 0.5
        if m_bool.sum() > 0:
            ra = np.argsort(np.argsort(e2sum[m_bool].reshape(-1)))
            rb = np.argsort(np.argsort(v_sum[m_bool].reshape(-1)))
            sp = float(np.corrcoef(ra, rb)[0, 1])
        else:
            sp = 0.0
        # 饱和度统计：接近边界的比例
        m_axes = mask[..., None].repeat(3, axis=-1) > 0.5
        boundary_margin = 0.05 * (logv_max - logv_min)
        sat_min = float(((logv <= (logv_min + boundary_margin)) & m_axes).mean()) if m_axes.any() else 0.0
        sat_max = float(((logv >= (logv_max - boundary_margin)) & m_axes).mean()) if m_axes.any() else 0.0
        stats = {
            "z2_mean": z2_mean,
            "cov68": cov68,
            "cov95": cov95,
            "spearman": sp,
            "saturation": sat_min + sat_max,
            "sat_min": sat_min,
            "sat_max": sat_max,
        }
    else:
        # 各向同性路径：根据分布选择阈值（Student-t 按 ν 动态求阈）
        lv = logv.squeeze(-1) if logv.ndim == 3 else logv  # (N,T)
        v = np.exp(lv)
        e2sum = e2.sum(axis=-1)
        m = (mask > 0.5)
        z2_full = e2sum / (np.clip(v, 1e-12, None) * 3.0)
        z2 = z2_full[m]
        z2_mean = float(z2.mean()) if z2.size > 0 else 0.0
        if dist == 'studentt' and (nu is not None) and (nu > 2.0):
            thr68 = float(studentt_z2_threshold(0.68, float(nu), 3))
            thr95 = float(studentt_z2_threshold(0.95, float(nu), 3))
        else:
            thr68, thr95 = C68_GAUSS, C95_GAUSS
        cov68 = float((z2 <= thr68).mean()) if z2.size > 0 else 0.0
        cov95 = float((z2 <= thr95).mean()) if z2.size > 0 else 0.0
        if m.sum() > 0:
            ra = np.argsort(np.argsort(e2sum[m].reshape(-1)))
            rb = np.argsort(np.argsort(v[m].reshape(-1)))
            sp = float(np.corrcoef(ra, rb)[0, 1])
        else:
            sp = 0.0
        boundary_margin = 0.05 * (logv_max - logv_min)
        sat_min = float(((lv <= (logv_min + boundary_margin)) & m).mean()) if m.any() else 0.0
        sat_max = float(((lv >= (logv_max - boundary_margin)) & m).mean()) if m.any() else 0.0
        stats = {
            "z2_mean": z2_mean,
            "cov68": cov68,
            "cov95": cov95,
            "spearman": sp,
            "saturation": sat_min + sat_max,
            "sat_min": sat_min,
            "sat_max": sat_max,
        }
    print("[metrics]")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 可视化（直方图/均值时间序列/覆盖率条形图）
    if args.plots_dir:
        Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
        DF = 3
        if d_out == 3:
            nu = cfg.get("train", {}).get("nu", 5.0)
            s2 = np.exp(logv)                       # (N,T,3)
            v_eff = s2 * (nu / (nu - 2.0))          # (N,T,3)
            
            # ✅ 逐轴归一化 → 求和 → 除 df（与 metrics.py 一致）
            z2_full = ((e2 / np.clip(v_eff, 1e-12, None)).sum(axis=-1) / DF)  # (N,T)
            z2 = z2_full[mask > 0]
            
            plt.figure(); plt.hist(z2, bins=80); plt.title("z² histogram (Student-t)"); plt.savefig(os.path.join(args.plots_dir, "z2_hist.png"), dpi=150); plt.close()
            m = mask.astype(np.float32)
            # 时间序列：每个时间步的平均 z²
            z2_mean_t = (z2_full * m).sum(0) / np.clip(m.sum(0), 1.0, None)
            plt.figure(); plt.plot(z2_mean_t, label="mean z²"); plt.axhline(y=1.0, color='r', linestyle='--', label='target=1.0'); plt.legend(); plt.title("z² time series (Student-t)"); plt.savefig(os.path.join(args.plots_dir, "z2_timeseries.png"), dpi=150); plt.close()
            # Thresholds: Student-t for diag-axes when using Student-t
            thr68 = float(studentt_z2_threshold(0.68, float(nu), DF))
            thr95 = float(studentt_z2_threshold(0.95, float(nu), DF))
            c68 = float(np.mean(z2_full[mask>0] <= thr68))
            c95 = float(np.mean(z2_full[mask>0] <= thr95))
            print(f"[Student-t thresholds] C68={thr68:.4f}, C95={thr95:.4f}")
            plt.figure(); plt.bar(["68%","95%"], [c68, c95]); plt.ylim(0,1); plt.title("coverage (Student-t)"); plt.savefig(os.path.join(args.plots_dir, "coverage.png"), dpi=150); plt.close()
        else:
            lv = logv.squeeze(-1) if logv.ndim == 3 else logv
            v = np.exp(lv)                    # (N,T)
            e2sum = e2.sum(axis=-1)           # (N,T)
            z2_full = e2sum / (np.clip(v,1e-12,None) * DF)
            z2 = z2_full[mask>0]
            plt.figure(); plt.hist(z2, bins=80); plt.title("z² histogram"); plt.savefig(os.path.join(args.plots_dir, "z2_hist.png"), dpi=150); plt.close()
            m = mask.astype(np.float32)
            e2_mean_t = ((e2sum / DF) * m).sum(0) / np.clip(m.sum(0), 1.0, None)
            v_mean_t  = (v * m).sum(0)        / np.clip(m.sum(0), 1.0, None)
            plt.figure(); plt.plot(e2_mean_t, label="mean e²/df"); plt.plot(v_mean_t, label="mean σ²"); plt.legend(); plt.title("mean curves"); plt.savefig(os.path.join(args.plots_dir, "mean_timeseries.png"), dpi=150); plt.close()
            # 阈值：Student-t 下按 ν 动态计算，否则用高斯常数
            if dist == 'studentt' and (nu is not None) and (nu > 2.0):
                thr68 = float(studentt_z2_threshold(0.68, float(nu), DF))
                thr95 = float(studentt_z2_threshold(0.95, float(nu), DF))
                c68 = float(np.mean(z2_full[mask>0] <= thr68))
                c95 = float(np.mean(z2_full[mask>0] <= thr95))
                print(f"[Student-t thresholds] C68={thr68:.4f}, C95={thr95:.4f}")
            else:
                c68 = float(np.mean(z2_full[mask>0] <= C68_GAUSS))
                c95 = float(np.mean(z2_full[mask>0] <= C95_GAUSS))
                print(f"[Gaussian thresholds] C68={C68_GAUSS:.4f}, C95={C95_GAUSS:.4f}")
            plt.figure(); plt.bar(["68%","95%"], [c68, c95]); plt.ylim(0,1); plt.title("coverage"); plt.savefig(os.path.join(args.plots_dir, "coverage.png"), dpi=150); plt.close()
        vis_plot_all(pred_lv_flat, gt_lv_flat, None, None, args.plots_dir, True)

if __name__ == "__main__":
    main()
