#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick sanity check for VIS NPZ files:
- Verify presence of all required keys
- Check shapes and data validity
- Report on diagonal supervision availability

Usage:
  python tools/check_vis_npz.py path/to/sequence_cam0_vis.npz
"""
import numpy as np
import sys

def check_vis_npz(path):
    print(f"\n{'='*60}")
    print(f"Checking VIS NPZ: {path}")
    print('='*60)
    
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False
    
    def has_key(k):
        return k in z.files
    
    # Core fields
    print("\n📋 Core Fields:")
    if has_key("X_VIS"):
        shape = z["X_VIS"].shape
        print(f"  ✅ X_VIS: shape={shape}, d_in={shape[-1] if len(shape) > 1 else shape[0]}")
        if len(shape) > 1 and shape[-1] not in [4, 68]:
            print(f"     ⚠️  Unexpected d_in={shape[-1]} (expected 4 or 68)")
    else:
        print("  ❌ X_VIS: MISSING")
        return False
    
    # Standard supervision
    print("\n📊 Standard Supervision:")
    for k in ["E2_VIS", "DF_VIS", "MASK_VIS"]:
        if has_key(k):
            shape = z[k].shape
            valid = np.isfinite(z[k]).sum()
            total = z[k].size
            print(f"  ✅ {k}: shape={shape}, valid={valid}/{total}")
        else:
            print(f"  ❌ {k}: MISSING")
    
    # Diagonal supervision (optional)
    print("\n🔲 Diagonal Supervision (optional for vis_2d_diag mode):")
    diag_complete = True
    for k in ["E2X_VIS", "E2Y_VIS", "DFX_VIS", "DFY_VIS"]:
        if has_key(k):
            shape = z[k].shape
            valid = np.isfinite(z[k]).sum()
            total = z[k].size
            print(f"  ✅ {k}: shape={shape}, valid={valid}/{total}")
        else:
            print(f"  ⚠️  {k}: MISSING (will fall back to E2/2 approximation)")
            diag_complete = False
    
    # Timestamps
    print("\n⏱️  Timestamps:")
    if has_key("TS_VIS"):
        ts = z["TS_VIS"]
        print(f"  ✅ TS_VIS: shape={ts.shape}, range=[{ts.min()}, {ts.max()}]")
    else:
        print("  ⚠️  TS_VIS: MISSING")
    
    # Summary
    print(f"\n{'='*60}")
    if diag_complete:
        print("✅ All fields present! Ready for vis_2d_diag training.")
    else:
        print("⚠️  Diagonal supervision incomplete. Will use E2/2 approximation.")
    print('='*60 + '\n')
    
    z.close()
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_vis.npz>")
        sys.exit(1)
    
    success = check_vis_npz(sys.argv[1])
    sys.exit(0 if success else 1)
