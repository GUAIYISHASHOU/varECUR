#!/usr/bin/env python3
"""
诊断OOF分割质量：检查每折的数据分布是否均衡
用于定位"某一折总是不降"的根因
"""
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def parse_meta(seq_name: str):
    """解析EuRoC风格序列名：MH_01_easy / V1_03_difficult / V2_02_medium"""
    m = re.match(r'(MH|V1|V2)_(\d+)_(easy|medium|difficult)', seq_name)
    if m:
        env = m.group(1)
        diff = m.group(3)
    else:
        env = 'UNK'
        diff = 'UNK'
    return env, diff


def load_manifest(manifest_path: str):
    """加载manifest文件"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_splits(splits_path: str):
    """加载splits文件，支持两种格式：
    1) {"0": {"train": [...], "val": [...]}, ...}
    2) {"folds": [{"train": [...], "val": [...]}, ...], "test": [...]}  # test可选
    返回统一的 dict 格式：{"0": {"train": [...], "val": [...]}, ...}
    """
    with open(splits_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "folds" in raw:
        folds_list = raw["folds"]
        norm = {str(i): {"train": folds_list[i]["train"], "val": folds_list[i]["val"]}
                for i in range(len(folds_list))}
        return norm
    return raw


def summarize(splits_json: str, manifest_json: str, route: str = 'acc'):
    """
    汇总每折的统计信息
    
    Args:
        splits_json: splits文件路径
        manifest_json: manifest文件路径
        route: 路由类型
    """
    mani = load_manifest(manifest_json)
    
    # 统计每个序列的窗口数（按route过滤）
    nwin = {}
    for d in mani:
        seq = d['seq']
        nwin_key = f'nwin_{route}'
        if nwin_key in d:
            nwin[seq] = d[nwin_key]
        elif 'nwin' in d:
            nwin[seq] = d['nwin']
        elif 'n_windows' in d:
            nwin[seq] = d['n_windows']
        else:
            nwin[seq] = 0
    
    # 解析环境和难度
    envd = {d['seq']: parse_meta(d['seq'])[0] for d in mani}
    diffd = {d['seq']: parse_meta(d['seq'])[1] for d in mani}
    
    splits = load_splits(splits_json)
    
    print(f"\n{'='*80}")
    print(f"Fold Diagnosis Report (route={route})")
    print(f"{'='*80}\n")
    
    # 收集所有折的窗口数，用于计算不平衡度
    all_val_wins = []
    all_trn_wins = []
    
    for k in sorted(map(int, splits.keys())):
        val = splits[str(k)]['val']
        trn = splits[str(k)]['train']
        
        def stat(seqs):
            W = sum(nwin.get(s, 0) for s in seqs)
            env = Counter(envd.get(s, '?') for s in seqs)
            dif = Counter(diffd.get(s, '?') for s in seqs)
            return W, env, dif, len(seqs)
        
        Wv, Ev, Dv, Nv = stat(val)
        Wt, Et, Dt, Nt = stat(trn)
        
        all_val_wins.append(Wv)
        all_trn_wins.append(Wt)
        
        print(f"Fold {k}:")
        print(f"  Val:   seqs={Nv:2d}, windows={Wv:5d}, env={dict(Ev)}, diff={dict(Dv)}")
        print(f"  Train: seqs={Nt:2d}, windows={Wt:5d}, env={dict(Et)}, diff={dict(Dt)}")
        
        # 红旗检测
        warnings = []
        if Wv == 0:
            warnings.append("⚠️  Val窗口数为0！")
        if Wt == 0:
            warnings.append("⚠️  Train窗口数为0！")
        if len(Ev) == 1:
            warnings.append(f"⚠️  Val只有单一环境: {list(Ev.keys())[0]}")
        if len(Dv) == 1:
            warnings.append(f"⚠️  Val只有单一难度: {list(Dv.keys())[0]}")
        if Dv.get('difficult', 0) == Nv and Nv > 0:
            warnings.append("⚠️  Val全是difficult序列！")
        
        if warnings:
            for w in warnings:
                print(f"    {w}")
        print()
    
    # 计算不平衡度
    if all_val_wins:
        max_val = max(all_val_wins)
        min_val = min(all_val_wins)
        avg_val = sum(all_val_wins) / len(all_val_wins)
        imbalance = (max_val - min_val) / avg_val if avg_val > 0 else 0
        
        print(f"{'='*80}")
        print(f"Overall Statistics:")
        print(f"  Val windows: min={min_val}, max={max_val}, avg={avg_val:.1f}")
        print(f"  Imbalance ratio: {imbalance:.2%} (建议 <15%)")
        
        if imbalance > 0.15:
            print(f"  ⚠️  WARNING: 窗口数不平衡度过高！建议重新生成splits")
        else:
            print(f"  ✓ 窗口数分布较为均衡")
        print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser(description="诊断OOF分割质量")
    ap.add_argument("--splits", required=True, help="Splits JSON文件路径")
    ap.add_argument("--manifest", required=True, help="Manifest JSON文件路径")
    ap.add_argument("--route", choices=["acc", "gyr"], default="acc", help="路由类型")
    args = ap.parse_args()
    
    summarize(args.splits, args.manifest, args.route)


if __name__ == "__main__":
    main()
