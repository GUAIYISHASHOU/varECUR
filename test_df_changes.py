#!/usr/bin/env python3
"""测试统一的自由度定义是否正确工作"""

import torch
import numpy as np

# 导入修改后的模块
from metrics import DF_BY_ROUTE, z2_from_residual, route_metrics_vis, route_metrics_imu

def test_df_by_route():
    """测试统一的自由度定义"""
    print("测试 DF_BY_ROUTE:")
    print(f"  vis: {DF_BY_ROUTE['vis']}")
    print(f"  acc: {DF_BY_ROUTE['acc']}")  
    print(f"  gyr: {DF_BY_ROUTE['gyr']}")
    assert DF_BY_ROUTE['vis'] == 2
    assert DF_BY_ROUTE['acc'] == 3
    assert DF_BY_ROUTE['gyr'] == 3
    print("✓ DF_BY_ROUTE 定义正确")

def test_z2_from_residual():
    """测试统一的z²计算函数"""
    print("\n测试 z2_from_residual:")
    
    # 模拟2D视觉残差
    residual_2d = torch.tensor([[[1.0, 1.0], [2.0, 0.0]]])  # (B=1, T=2, D=2)
    logvar_vis = torch.tensor([[[0.0], [0.0]]])  # (B=1, T=2, 1) - 各向同性
    df_vis = DF_BY_ROUTE['vis']
    
    z2_vis = z2_from_residual(residual_2d, logvar_vis, df_vis)
    print(f"  VIS z2: {z2_vis}")
    expected_vis = torch.tensor([[2.0/2, 4.0/2]])  # (1²+1²)/2, (2²+0²)/2
    assert torch.allclose(z2_vis, expected_vis)
    
    # 模拟3D IMU残差
    residual_3d = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]])  # (B=1, T=2, D=3)
    logvar_imu = torch.tensor([[[0.0], [0.0]]])  # (B=1, T=2, 1) - 各向同性
    df_imu = DF_BY_ROUTE['acc']
    
    z2_imu = z2_from_residual(residual_3d, logvar_imu, df_imu)
    print(f"  IMU z2: {z2_imu}")
    expected_imu = torch.tensor([[3.0/3, 4.0/3]])  # (1²+1²+1²)/3, (2²+0²+0²)/3
    assert torch.allclose(z2_imu, expected_imu)
    
    print("✓ z2_from_residual 计算正确")

def test_route_metrics():
    """测试路由指标函数使用正确的df"""
    print("\n测试路由指标函数:")
    
    # 创建测试数据
    B, T = 2, 10
    e2sum = torch.rand(B, T) * 2  # 随机误差
    logv = torch.randn(B, T) * 0.5  # 随机对数方差
    mask = torch.ones(B, T)  # 全部有效
    
    # 测试视觉指标
    vis_stats = route_metrics_vis(e2sum, logv, mask, -10.0, 10.0)
    print(f"  VIS stats keys: {list(vis_stats.keys())}")
    print(f"  VIS z2_mean: {vis_stats['z2_mean']:.4f}")
    
    # 测试IMU指标
    imu_stats = route_metrics_imu(e2sum, logv, mask, -10.0, 10.0)
    print(f"  IMU stats keys: {list(imu_stats.keys())}")
    print(f"  IMU z2_mean: {imu_stats['z2_mean']:.4f}")
    
    print("✓ 路由指标函数工作正常")

if __name__ == "__main__":
    test_df_by_route()
    test_z2_from_residual() 
    test_route_metrics()
    print("\n所有测试通过! 统一的自由度定义工作正常。")
