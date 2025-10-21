#!/usr/bin/env python3
"""
构建分层GroupKFold分割：按序列分组，平衡环境、难度和窗口数
用于OOF训练，避免数据泄漏和分布失衡
"""
import json
import re
import random
import argparse
from collections import Counter
from pathlib import Path


def parse_meta(seq: str):
    """解析EuRoC序列名，提取环境和难度"""
    m = re.match(r'(MH|V1|V2)_(\d+)_(easy|medium|difficult)', seq)
    if m:
        env = m.group(1)
        diff = m.group(3)
    else:
        env = 'UNK'
        diff = 'UNK'
    return env, diff


def load_manifest(manifest_path: str):
    """加载manifest文件（JSON格式）"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_folds(manifest_json: str, k: int = 5, seed: int = 42, route: str = 'acc'):
    """
    构建分层GroupKFold分割
    
    Args:
        manifest_json: manifest文件路径，包含每个序列的窗口数信息
        k: 折数
        seed: 随机种子
        route: 路由类型 ('acc' 或 'gyr')
    
    Returns:
        dict: {fold_id: {"train": [seqs], "val": [seqs]}}
    """
    rng = random.Random(seed)
    mani = load_manifest(manifest_json)
    
    # 每个序列的"重量" = 窗口数（按route）
    weight = {}
    label = {}
    
    for d in mani:
        seq = d['seq']
        # 尝试多种可能的键名
        nwin_key = f'nwin_{route}'
        if nwin_key in d:
            w = max(1, int(d[nwin_key]))
        elif 'nwin' in d:
            w = max(1, int(d['nwin']))
        elif 'n_windows' in d:
            w = max(1, int(d['n_windows']))
        else:
            w = 1  # 默认权重
        
        weight[seq] = w
        label[seq] = parse_meta(seq)
    
    seqs = list(weight.keys())
    
    # 先打乱序列，避免系统性顺序偏差
    rng.shuffle(seqs)
    
    # 按权重降序排序（贪心装箱）
    seqs.sort(key=lambda s: weight[s], reverse=True)
    
    # 初始化k个折
    folds = [dict(weight=0, seqs=[], lab=Counter()) for _ in range(k)]
    
    # 贪心分配：每次把序列放到"最不破坏均衡"的折
    for s in seqs:
        cand = []
        for i in range(k):
            # 计算放入该折后的状态
            w_after = folds[i]['weight'] + weight[s]
            lab_after = folds[i]['lab'].copy()
            lab_after[label[s]] += 1
            
            # 评分：窗口数方差 + 标签不平衡惩罚
            w_list = [
                (folds[j]['weight'] + (weight[s] if j == i else 0))
                for j in range(k)
            ]
            w_var = max(w_list) - min(w_list)
            
            # 标签平衡：惩罚该折的某类比例过高
            total_lab = sum(lab_after.values())
            lab_pen = max(lab_after.values()) / total_lab if total_lab > 0 else 0
            
            score = (w_var, lab_pen)  # tuple比较：先比较w_var，再比较lab_pen
            cand.append((score, i))
        
        # 选择得分最低的折
        cand.sort(key=lambda x: x[0])
        best_i = cand[0][1]
        
        folds[best_i]['seqs'].append(s)
        folds[best_i]['weight'] += weight[s]
        folds[best_i]['lab'][label[s]] += 1
    
    # 生成train/val列表
    out = {}
    for i in range(k):
        val = folds[i]['seqs']
        trn = [s for j in range(k) if j != i for s in folds[j]['seqs']]
        out[str(i)] = {"train": trn, "val": val}
    
    # 打印统计信息
    print(f"\n=== Fold Statistics (route={route}) ===")
    for i in range(k):
        val_seqs = folds[i]['seqs']
        val_w = folds[i]['weight']
        val_lab = folds[i]['lab']
        trn_w = sum(folds[j]['weight'] for j in range(k) if j != i)
        
        print(f"\nFold {i}:")
        print(f"  Val:   {len(val_seqs):2d} seqs, {val_w:5d} windows, labels={dict(val_lab)}")
        print(f"  Train: {len(out[str(i)]['train']):2d} seqs, {trn_w:5d} windows")
    
    return out


def main():
    ap = argparse.ArgumentParser(description="构建分层GroupKFold分割")
    ap.add_argument("--manifest", required=True, help="Manifest JSON文件路径")
    ap.add_argument("--out", default="splits_oof.json", help="输出splits文件路径")
    ap.add_argument("--k", type=int, default=5, help="折数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--route", choices=["acc", "gyr"], default="acc", help="路由类型")
    args = ap.parse_args()
    
    folds = build_folds(args.manifest, k=args.k, seed=args.seed, route=args.route)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved splits to: {out_path}")


if __name__ == "__main__":
    main()
