from __future__ import annotations
import argparse, json as jsonlib
from pathlib import Path
import torch
from utils import to_device, load_config_file
import dataset  # 避免同名遮蔽
from models import IMURouteModel
from metrics import route_metrics_imu, DF_BY_ROUTE

def parse_args():
    # 先解析 --route 参数来确定配置段
    pre_route = argparse.ArgumentParser(add_help=False)
    pre_route.add_argument("--route", choices=["acc","gyr"], default=None)
    args_route, _ = pre_route.parse_known_args()
    
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    
    # IMU only configuration
    route = args_route.route or "acc"
    ev = cfg.get("eval", {})
    rt = cfg.get("runtime", {})

    ap = argparse.ArgumentParser("Evaluate a trained single-route model", parents=[pre])
    ap.add_argument("--route", choices=["acc","gyr"], default=ev.get("route","acc"))
    ap.add_argument("--npz", required=(ev.get("npz") is None), default=ev.get("npz"))
    ap.add_argument("--model", required=(ev.get("model") is None), default=ev.get("model"))
    ap.add_argument("--x_mode", choices=["both","route_only","imu"], default=ev.get("x_mode","both"))
    ap.add_argument("--device", default=rt.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    # 增加新参数
    ap.add_argument("--nu", type=float, default=0.0, help="Student-t 自由度（评测口径）；0 表示用高斯口径")
    ap.add_argument("--manifest", type=str, default=None, help="manifest.json from split_eachseq_merge_npz.py")
    ap.add_argument("--plots_dir", type=str, default=None, help="where to save per-sequence plots (.png)")
    
    # 参数规范化：与train保持一致
    ap.add_argument('--logv_min', type=float, default=-12.0,
                   help='lower clamp for predicted log-variance')
    ap.add_argument('--logv_max', type=float, default=6.0,
                   help='upper clamp for predicted log-variance')
    
    # 温度校正参数
    ap.add_argument('--auto_temp', choices=['off','global'], default='off',
                   help='Temperature calibration mode')
    ap.add_argument('--calib_npz', type=str, default=None,
                   help='NPZ file for calibration (defaults to test set)')
    ap.add_argument('--logvar_offset', type=float, default=0.0,
                   help='additive offset to log-variance (applied after auto_temp)')
    
    return ap.parse_args()

def main():
    args = parse_args()
    ds, dl = dataset.build_loader(args.npz, route=args.route, x_mode=args.x_mode, batch_size=64, shuffle=False, num_workers=0)

    # 动态确定输入/输出维度
    d_in = ds.X_all.shape[-1] if args.x_mode=="both" else 3
    d_out = 1
    
    # Load checkpoint (weights_only for newer torch; fallback for older)
    try:
        ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.model, map_location="cpu")
    state = ckpt.get("model", ckpt)
    md_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    # Infer output dim from head weights (3 for diag, 1 for iso) and override
    if isinstance(state, dict) and ("head.weight" in state or "head.bias" in state):
        out_dim = int(state.get("head.weight", state.get("head.bias")).shape[0])
    else:
        out_dim = d_out

    model = IMURouteModel(d_in=d_in,
                          d_out=out_dim,
                          d_model=md_args.get("d_model",128),
                          n_tcn=md_args.get("n_tcn",4),
                          kernel_size=md_args.get("kernel_size",3),
                          n_layers_tf=md_args.get("n_layers_tf",2),
                          n_heads=md_args.get("n_heads",4),
                          dropout=md_args.get("dropout",0.1),
                          route=args.route)
    # Load weights (support both full dict and raw state dict)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    all_stats = []
    # 收集每窗口数据用于按序列分析
    all_e2sum, all_logv, all_mask = [], [], []
    
    # 收集所有原始数据用于全局温度校正
    if args.auto_temp == "global":
        all_e2sum_raw, all_logv_raw, all_mask_raw = [], [], []
    
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, args.device)
            logv = model(batch["X"])
            
            # 收集原始数据用于全局温度校正
            if args.auto_temp == "global":
                e2_data = batch["E2"]
                if e2_data.size(-1) == 3:  # 步级标签，三轴求和
                    e2sum_temp = e2_data.sum(dim=-1)  # (B,T)
                else:  # 回退模式
                    e2sum_temp = e2_data.squeeze(-1)  # (B,T)
                all_e2sum_raw.append(e2sum_temp.detach().cpu().numpy())
                all_logv_raw.append(logv.detach().cpu().numpy())
                all_mask_raw.append(batch["MASK"].detach().cpu().numpy())
            # 应用手工偏置（auto_temp会在后面重新计算时应用）
            if args.auto_temp != "global":
                logv_applied = logv + args.logvar_offset
            else:
                logv_applied = logv  # 温度校正模式下先不应用偏置
            
            # 统一先算一次指标
            st = route_metrics_imu(batch["E2"], logv_applied, batch["MASK"],
                                 logv_min=args.logv_min, logv_max=args.logv_max,
                                 yvar=batch.get("Y", None))
            all_stats.append(st)
            
            # 若需要按序列画图，再额外收集原始数据
            if args.manifest and args.plots_dir:
                # IMU路由：处理步级标签
                e2_data = batch["E2"]
                if e2_data.size(-1) == 3:  # 步级标签，三轴求和
                    e2sum = e2_data.sum(dim=-1)  # (B,T)
                else:  # 回退模式
                    e2sum = e2_data.squeeze(-1)  # (B,T)
                mask = batch["MASK"]
                
                all_e2sum.append(e2sum.detach().cpu().numpy())
                all_logv.append(logv.detach().cpu().numpy())  # 保存原始logv用于后续校正
                all_mask.append(mask.detach().cpu().numpy())
    
    # 准备按序列分析的数据
    if args.manifest and args.plots_dir and all_e2sum:
        import numpy as np
        E2SUM = np.concatenate(all_e2sum, axis=0)  # (N,T)
        LOGV  = np.concatenate(all_logv,  axis=0)  # (N,T,1) 或 (N,T)
        MASK  = np.concatenate(all_mask,  axis=0)  # (N,T)
        if LOGV.ndim == 2:  # 统一成 (N,T,1)
            LOGV = LOGV[..., None]

    # ===== 全局温度校正（IMU）=====
    delta_logvar = 0.0
    if args.auto_temp == "global" and args.route in ("acc", "gyr"):
        import numpy as np
        print("[global_temp] 计算全局温度校正（均值z²法）...")

        calib_npz = args.calib_npz or args.npz
        print(f"[global_temp] 使用校准数据: {calib_npz}")

        # 取 val 集或当前集的数据（和你原逻辑一致）
        if args.calib_npz:
            calib_ds, calib_dl = dataset.build_loader(
                calib_npz, route=args.route, x_mode=args.x_mode,
                batch_size=64, shuffle=False, num_workers=0
            )
            calib_e2sum_raw, calib_logv_raw, calib_mask_raw = [], [], []
            with torch.no_grad():
                for batch in calib_dl:
                    b = to_device(batch, args.device)
                    logv = model(b["X"])
                    e2 = b["E2"]
                    e2sum = e2.sum(dim=-1) if (e2.dim()==3 and e2.size(-1)==3) else e2.squeeze(-1)
                    calib_e2sum_raw.append(e2sum.detach().cpu().numpy())
                    calib_logv_raw.append(logv.detach().cpu().numpy())
                    calib_mask_raw.append(b["MASK"].detach().cpu().numpy())
            e2sum_flat = np.concatenate(calib_e2sum_raw, axis=0)
            logv_flat  = np.concatenate(calib_logv_raw,  axis=0)
            mask_flat  = np.concatenate(calib_mask_raw,  axis=0)
        else:
            e2sum_flat = np.concatenate(all_e2sum_raw, axis=0)
            logv_flat  = np.concatenate(all_logv_raw,  axis=0)
            mask_flat  = np.concatenate(all_mask_raw,  axis=0)

        # Handle logv dimensions
        if logv_flat.ndim == 3 and logv_flat.shape[-1] == 1:
            logv_flat_for_temp = logv_flat.squeeze(-1)
        else:
            logv_flat_for_temp = logv_flat

        valid = mask_flat > 0
        e2sum_valid = e2sum_flat[valid]
        logv_valid  = logv_flat_for_temp[valid]

        # 与评测一致：先 clamp，再算 var 和 z²
        sigma2 = np.exp(np.clip(logv_valid, args.logv_min, args.logv_max))
        df = float(DF_BY_ROUTE.get(args.route, 3))  # 使用统一的自由度定义
        z2 = (e2sum_valid / sigma2) / df

        # 稳健一点：去掉顶部 1% 的极端点（可选）
        if z2.size > 100:
            hi = np.percentile(z2, 99.0)
            z2 = z2[z2 <= hi]

        # 目标：E[z²] -> 1  ⇒  给 logσ² 加上 log(E[z²])
        mu = float(np.mean(z2))
        delta_logvar = float(np.log(mu))

        n_used = int(valid.sum()); n_tot = int(valid.size)
        print(f"[global_temp] 使用 {n_used}/{n_tot} 数据点")
        print(f"[global_temp] delta_logvar = {delta_logvar:+.4f}  (σ×={np.exp(0.5*delta_logvar):.3f})")
        
        # 用校正后的 logv 重算指标
        print("[global_temp] 重新计算校正后指标...")
        all_stats_corrected = []
        with torch.no_grad():
            for batch in dl:
                b = to_device(batch, args.device)
                logv = model(b["X"])
                
                # 应用温度校正（IMU 路由）
                st = route_metrics_imu(
                    b["E2"], logv + delta_logvar + args.logvar_offset, b["MASK"],
                    logv_min=args.logv_min, logv_max=args.logv_max,
                    yvar=b.get("Y", None)
                )
                all_stats_corrected.append(st)
        all_stats = all_stats_corrected
        print("[global_temp] 温度校正完成！")
    
    # 如果没有温度校正但有手工偏置，也要应用到按序列数据
    if args.logvar_offset != 0.0 and args.auto_temp == "off":
        print(f"[logvar_offset] 应用手工偏置: {args.logvar_offset:+.4f}")

    # Average - 只对数值类型进行聚合
    if all_stats:
        keys = all_stats[0].keys()
        agg = {}
        for k in keys:
            values = [d[k] for d in all_stats]
            # 只对数值类型进行平均
            if all(isinstance(v, (int, float)) for v in values):
                agg[k] = float(sum(values) / len(values))
            else:
                # 对于非数值类型，取第一个值（如列表、字符串等）
                agg[k] = values[0]
    else:
        agg = {}
    
    print(jsonlib.dumps(agg, indent=2, ensure_ascii=False))

    # ===== 按序列分析和画图 =====
    if args.manifest and args.plots_dir and 'E2SUM' in locals():
        import os
        import matplotlib.pyplot as plt
        os.makedirs(args.plots_dir, exist_ok=True)

        with open(args.manifest, "r", encoding="utf-8") as f:
            mani = jsonlib.load(f)

        # Derive per-sequence window counts; support both legacy and new manifest layouts.
        if "per_sequence" in mani:
            seq_items = [
                (fname, info["counts"].get("test", 0))
                for fname, info in mani["per_sequence"].items()
            ]
        else:
            splits = mani.get("splits", {})
            test_entries = splits.get("test", [])
            seq_lengths = [
                (entry["file"], max(int(entry["idx"][1]) - int(entry["idx"][0]), 0))
                for entry in test_entries
            ]
            vis_window = getattr(ds, "vis_window", E2SUM.shape[1] if E2SUM.ndim == 2 else 64)
            vis_stride = getattr(ds, "vis_stride", vis_window)
            vis_window = max(int(vis_window), 1)
            vis_stride = max(int(vis_stride), 1)
            total_len = sum(length for _, length in seq_lengths)
            if total_len <= 0:
                seq_items = [(fname, 0) for fname, _ in seq_lengths]
            else:
                if total_len <= vis_window:
                    window_starts = [0]
                else:
                    last_possible = max(total_len - vis_window, 0)
                    window_starts = list(range(0, last_possible + 1, vis_stride))
                    if window_starts[-1] != last_possible:
                        window_starts.append(last_possible)
                boundaries = []
                offset = 0
                for fname, length in seq_lengths:
                    boundaries.append((fname, offset, offset + length))
                    offset += length
                counts = {fname: 0 for fname, _ in seq_lengths}
                for start_idx in window_starts:
                    window_end = min(start_idx + vis_window, total_len)
                    center = start_idx + (window_end - start_idx) / 2.0
                    for fname, lo, hi in boundaries:
                        if center < hi or fname == boundaries[-1][0]:
                            counts[fname] += 1
                            break
                seq_items = [(fname, counts[fname]) for fname, _ in seq_lengths]

        start = 0
        print("E2SUM:", E2SUM.shape)
        print("sum(manifest_counts)=", sum(cnt for _, cnt in seq_items))
        per_seq_metrics = {}

        for fname, n_test in seq_items:
            if n_test <= 0:
                continue
            sl = slice(start, start + n_test)
            start += n_test


            e2sum = E2SUM[sl]             # (n,T)
            logv  = LOGV[sl]              # (n,T,1) or (n,T,3) for per-axis
            mask  = MASK[sl]              # (n,T)

            # 应用温度校正 + 手工偏置到按序列数据
            logv_corrected = logv + delta_logvar + args.logvar_offset
            
            # 计算该序列的指标
            t_e2 = torch.from_numpy(e2sum)
            t_lv = torch.from_numpy(logv_corrected)  # 使用校正后的logv
            t_mk = torch.from_numpy(mask)
            stats = route_metrics_imu(t_e2, t_lv, t_mk, logv_min=args.logv_min, logv_max=args.logv_max)
            stats = {k: (v.item() if hasattr(v, "item") else float(v)) for k,v in stats.items()}
            per_seq_metrics[fname] = stats

            # ---------- 画图（3张） ----------
            # 1) z^2 直方图（掩码内）
            df = float(DF_BY_ROUTE.get(args.route, 3))  # 使用统一的自由度定义
            
            # Handle logv dimensions
            if logv_corrected.ndim == 3 and logv_corrected.shape[-1] == 1:
                v = np.exp(np.clip(logv_corrected, args.logv_min, args.logv_max))[:, :, 0]  # (n,T)
            else:
                v = np.exp(np.clip(logv_corrected, args.logv_min, args.logv_max))  # (n,T)
            z2 = (e2sum / (np.clip(v, 1e-12, None) * df))[mask > 0]  # ← 归一化
            plt.figure()
            plt.hist(z2, bins=80)
            plt.title(f"{fname}  z² histogram")
            plt.xlabel("z²"); plt.ylabel("count")
            plt.savefig(os.path.join(args.plots_dir, f"{fname}_z2_hist.png"), dpi=150, bbox_inches="tight"); plt.close()

            # 2) 平均时间序列（看校准趋势）
            m = mask.astype(np.float32)
            e2_mean_t = ((e2sum / df) * m).sum(0) / np.clip(m.sum(0), 1.0, None)  # ← /df
            v_mean_t  = (v * m).sum(0) / np.clip(m.sum(0), 1.0, None)
            plt.figure()
            plt.plot(e2_mean_t, label="mean e² / df")  # 改个label避免误解
            plt.plot(v_mean_t,  label="mean σ² (pred)")
            plt.legend(); plt.title(f"{fname}  mean e² vs mean σ²")
            plt.xlabel("time idx"); plt.ylabel("value")
            plt.savefig(os.path.join(args.plots_dir, f"{fname}_timeseries.png"), dpi=150, bbox_inches="tight"); plt.close()

            # 3) 覆盖率小条形图（68/95）
            cov68 = stats["cov68"]; cov95 = stats["cov95"]
            plt.figure()
            xs = [0,1]; gt = [0.68,0.95]; got = [cov68, cov95]
            plt.bar([x-0.15 for x in xs], gt,  width=0.3, label="target")
            plt.bar([x+0.15 for x in xs], got, width=0.3, label="empirical")
            plt.xticks(xs, ["68%", "95%"]); plt.ylim(0,1)
            plt.title(f"{fname}  coverage")
            plt.legend()
            plt.savefig(os.path.join(args.plots_dir, f"{fname}_coverage.png"), dpi=150, bbox_inches="tight"); plt.close()

        # 存一份每序列指标
        with open(os.path.join(args.plots_dir, "per_sequence_metrics.json"), "w", encoding="utf-8") as f:
            jsonlib.dump(per_seq_metrics, f, indent=2, ensure_ascii=False)

        print(f"[per-seq] wrote plots & metrics to {args.plots_dir}")

if __name__ == "__main__":
    main()
