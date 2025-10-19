#!/usr/bin/env python3
"""
生成Manifest文件：扫描序列NPZ目录，统计每个序列的窗口数
用于OOF分层分折
"""
import argparse
import json
import numpy as np
from pathlib import Path


def generate_manifest(seq_dir: str, routes: list = None):
    """
    扫描序列目录，生成manifest
    
    Args:
        seq_dir: 序列NPZ文件目录
        routes: 要统计的路由列表，默认 ['acc', 'gyr']
    
    Returns:
        manifest: list of dict
    """
    if routes is None:
        routes = ['acc', 'gyr']
    
    seq_dir = Path(seq_dir)
    npz_files = sorted(seq_dir.glob('*.npz'))
    
    if not npz_files:
        print(f"Warning: No NPZ files found in {seq_dir}")
        return []
    
    manifest = []
    
    print(f"Scanning {len(npz_files)} NPZ files...")
    
    for npz_path in npz_files:
        seq_name = npz_path.stem
        
        try:
            data = np.load(npz_path)
            
            entry = {"seq": seq_name}
            
            # 统计每个route的窗口数
            for route in routes:
                if route == 'acc':
                    keys = ['X_IMU_ACC', 'X_acc', 'X']
                elif route == 'gyr':
                    keys = ['X_IMU_GYR', 'X_gyr', 'X']
                else:
                    continue
                
                # 找到对应的键
                X = None
                for k in keys:
                    if k in data.files:
                        X = data[k]
                        break
                
                if X is not None:
                    nwin = X.shape[0]
                    entry[f'nwin_{route}'] = int(nwin)
                else:
                    entry[f'nwin_{route}'] = 0
            
            manifest.append(entry)
            
            # 打印进度
            nwin_str = ', '.join([f"{route}={entry.get(f'nwin_{route}', 0)}" 
                                 for route in routes])
            print(f"  {seq_name}: {nwin_str}")
            
        except Exception as e:
            print(f"  Error loading {npz_path.name}: {e}")
            continue
    
    return manifest


def main():
    ap = argparse.ArgumentParser(description="生成Manifest文件")
    ap.add_argument("--seq_dir", required=True, help="序列NPZ文件目录")
    ap.add_argument("--out", default="manifest.json", help="输出manifest文件路径")
    ap.add_argument("--routes", nargs='+', default=['acc', 'gyr'], 
                   help="要统计的路由列表")
    args = ap.parse_args()
    
    manifest = generate_manifest(args.seq_dir, args.routes)
    
    if not manifest:
        print("No data found, manifest not created")
        return
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved manifest with {len(manifest)} sequences to: {out_path}")
    
    # 打印统计
    total_acc = sum(e.get('nwin_acc', 0) for e in manifest)
    total_gyr = sum(e.get('nwin_gyr', 0) for e in manifest)
    print(f"\nTotal windows: acc={total_acc}, gyr={total_gyr}")


if __name__ == "__main__":
    main()
