# -*- coding: utf-8 -*-
# tools/make_kfold_splits.py
"""
生成 K 折交叉验证的索引切分文件
只需要运行一次，生成的 JSON 文件可以重复使用
"""
import argparse
import os
import json
import numpy as np
from sklearn.model_selection import KFold

def main():
    ap = argparse.ArgumentParser(description="生成 K 折交叉验证索引")
    ap.add_argument("--train_npz", required=True, help="训练集 npz 文件路径")
    ap.add_argument("--out_json", required=True, help="输出的折叠索引 JSON 文件")
    ap.add_argument("--k", type=int, default=5, help="折数（默认 5）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    # 读取训练集获取样本数
    data = np.load(args.train_npz)
    M = int(data["y_true"].shape[0])  # 样本数
    print(f"训练集样本数: {M}")
    
    idx = np.arange(M)

    # K 折切分
    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    folds = []
    for fold_idx, (tr, va) in enumerate(kf.split(idx)):
        folds.append({
            "fold": fold_idx,
            "train_idx": tr.tolist(),
            "val_idx": va.tolist(),
            "n_train": len(tr),
            "n_val": len(va)
        })
        print(f"Fold {fold_idx}: train={len(tr)}, val={len(va)}")

    # 保存到 JSON
    output = {
        "k": args.k,
        "seed": args.seed,
        "total_samples": M,
        "folds": folds
    }
    
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ K 折切分已保存到: {args.out_json}")

if __name__ == "__main__":
    main()
