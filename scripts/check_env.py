#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查VIS数据生成所需的环境"""

import sys

print("=" * 50)
print("检查VIS数据生成环境")
print("=" * 50)
print()

# 检查Python版本
print(f"Python版本: {sys.version}")
print()

# 检查必要的包
packages = {
    'numpy': 'NumPy',
    'yaml': 'PyYAML', 
    'cv2': 'OpenCV',
    'sklearn': 'scikit-learn'
}

missing = []
print("检查Python包:")
for module, name in packages.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ✗ {name} - 未安装")
        print(f"    错误: {e}")
        missing.append(name)

print()

if missing:
    print("=" * 50)
    print("缺少以下包，请安装:")
    print("=" * 50)
    for pkg in missing:
        if pkg == 'PyYAML':
            print(f"  pip install pyyaml")
        elif pkg == 'OpenCV':
            print(f"  pip install opencv-python")
        elif pkg == 'scikit-learn':
            print(f"  pip install scikit-learn")
        else:
            print(f"  pip install {pkg.lower()}")
    print()
    sys.exit(1)
else:
    print("=" * 50)
    print("✓ 所有依赖包已安装！")
    print("=" * 50)
    print()
    print("可以开始生成VIS数据:")
    print("  python scripts/run_vis_data_generation.py")
    sys.exit(0)

