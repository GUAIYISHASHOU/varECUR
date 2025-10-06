#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像质量感知工具：光度预处理 + 质量评估 + 损失加权
用于降低模糊/低纹理/曝光异常样本的训练权重
"""
import numpy as np

try:
    import cv2  # 用于CLAHE/拉普拉斯
except Exception:
    cv2 = None

# ---------- 光度预处理 ----------
def photometric_preprocess(img, mode="none", clahe_tiles=8, clahe_clip=3.0, gamma=1.0):
    """
    img: HxW 或 HxWx1/3, uint8/float32, [0,255]或[0,1]
    return: float32, 单通道[0,1]
    """
    if img is None:
        return None
    if img.ndim == 3 and img.shape[-1] == 3:
        # 转灰度
        img = 0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]
    img = img.astype(np.float32)
    # 归一化到[0,1]
    if img.max() > 1.5:
        img = img / 255.0

    if mode == "clahe" and cv2 is not None:
        # OpenCV CLAHE 需要uint8
        u8 = np.clip(img*255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_tiles), int(clahe_tiles)))
        u8 = clahe.apply(u8)
        img = u8.astype(np.float32)/255.0
    elif mode == "gamma":
        img = np.power(np.clip(img, 0.0, 1.0), float(gamma))
    # mode=="none" 直接返回
    return img

# ---------- 帧间仿射亮度匹配（可选） ----------
def affine_brightness_match(I1, I2, robust_trim=0.01):
    """
    估计 a,b 使 I2 ≈ a*I1 + b，返回匹配后的 I2'
    """
    x = I1.reshape(-1); y = I2.reshape(-1)
    n = x.size
    k = int(n * float(robust_trim))
    if k > 0:
        # 去除两端1%强异常
        lo = k; hi = n - k
        idx = np.argsort(x)
        x = x[idx][lo:hi]; y = y[idx][lo:hi]
    vx = np.var(x) + 1e-12
    a = float(np.cov(x, y, bias=True)[0,1] / vx)
    b = float(np.mean(y) - a * np.mean(x))
    return np.clip(a*I2 + b, 0.0, 1.0), a, b

# ---------- 清晰度/纹理质量 ----------
def laplacian_var(img):
    """无参考清晰度，Laplacian 方差"""
    if cv2 is None:
        # 退化实现（简单卷积核）
        K = np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)
        f = np.pad(img, 1, mode='reflect')
        L = (f[1:-1,2:] + f[1:-1,:-2] + f[2:,1:-1] + f[:-2,1:-1] - 4*f[1:-1,1:-1])
        return float(np.var(L))
    L = cv2.Laplacian((img*255).astype(np.uint8), cv2.CV_32F, ksize=3)
    return float(L.var())

def sobel_grad_mean(img):
    """平均梯度强度（衡量纹理）"""
    if cv2 is None:
        # 退化实现
        Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
        Ky = Kx.T
        f = np.pad(img, 1, mode='reflect')
        Gx = (f[1:-1,2:]*1 + f[1:-1,:-2]*-1 + f[2:,2:]*2 + f[2:,:-2]*-2 + f[:-2,2:]*1 + f[:-2,:-2]*-1)
        Gy = (f[2:,1:-1]*2 + f[:-2,1:-1]*-2 + f[2:,2:]*1 + f[:-2,2:]*-1 + f[2:,:-2]*1 + f[:-2,:-2]*-1)
    else:
        Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(Gx*Gx + Gy*Gy)
    return float(mag.mean())

# ---------- 将图像质量变成训练/评估用权重 ----------
def quality_weight_from_images(img0, img1, args):
    """
    基于两帧质量→返回 w in [0,1]，低质→小权重；以及是否跳样本的布尔量
    """
    # 预处理
    I0 = photometric_preprocess(img0, args.photometric, args.clahe_tiles, args.clahe_clip, args.gamma)
    I1 = photometric_preprocess(img1, args.photometric, args.clahe_tiles, args.clahe_clip, args.gamma)
    if args.ab_match and I0 is not None and I1 is not None:
        I1, _, _ = affine_brightness_match(I0, I1)

    # 清晰度/纹理
    l0 = laplacian_var(I0) if I0 is not None else None
    l1 = laplacian_var(I1) if I1 is not None else None
    g0 = sobel_grad_mean(I0) if I0 is not None else None
    g1 = sobel_grad_mean(I1) if I1 is not None else None

    # 合成得分（归一化到[0,1]）
    # 阈值/尺度来自命令行，可根据你的训练集统计微调
    lmin, lmax = args.blur_thr, args.blur_thr*args.blur_scale
    def norm(v, vmin, vmax):
        if v is None: return 1.0
        return float(np.clip((v - vmin) / max(1e-9, (vmax - vmin)), 0.0, 1.0))
    sL = min(norm(l0, lmin, lmax), norm(l1, lmin, lmax))
    # 梯度分数
    gmin, gmax = args.grad_thr, args.grad_thr*args.grad_scale
    sG = min(norm(g0, gmin, gmax), norm(g1, gmin, gmax))

    # 合成权重（指数抑制低质样本）
    w = (args.w_lap*sL + args.w_grad*sG) / max(1e-6, (args.w_lap + args.w_grad))
    w = w ** args.w_gamma  # γ>1 更强抑制
    skip = (sL <= 0.0 and args.skip_on_blur)  # 超阈可选择直接跳样本
    return float(np.clip(w, 0.0, 1.0)), bool(skip)

def quality_weight_from_meta(sample, args):
    """
    没有图像时，从npz里已有字段推一个权重（自动适配）
    支持的常见key: 'grad_mean','ncc','inliers','tex','sharpness'...
    """
    # 越大越好
    for k in ['grad_mean','tex','sharpness','inliers','ncc']:
        if k in sample:
            v = float(sample[k])
            vmin, vmax = args.meta_thr, args.meta_thr*args.meta_scale
            s = np.clip((v - vmin)/max(1e-9,(vmax - vmin)), 0.0, 1.0)
            return float(s ** args.w_gamma), False
    return 1.0, False

