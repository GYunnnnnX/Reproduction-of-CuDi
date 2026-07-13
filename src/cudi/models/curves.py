"""Curve operations for Zero-DCE and CuDi."""

from __future__ import annotations

import torch
from torch import Tensor


def split_curve_params(params: Tensor, iterations: int = 8) -> tuple[Tensor, ...]:
    """将教师网络输出的曲线参数按迭代次数拆分。
    CuDi 教师分支继承 Zero-DCE 的高阶曲线设计：每次迭代需要 RGB 三个
    通道各自的参数图，因此参数通道数为iterations * 3。
    """
    if params.dim() != 4:
        raise ValueError("curve parameters must be a 4D tensor")
    expected_channels = iterations * 3
    if params.size(1) != expected_channels:
        raise ValueError(f"expected {expected_channels} channels, got {params.size(1)}")
    return torch.chunk(params, iterations, dim=1)


def apply_le_curve(image: Tensor, curve_params: Tensor, iterations: int = 8) -> Tensor:
    """应用 Zero-DCE 的逐像素高阶 LE 曲线。
    用于教师分支：教师网络预测 8 组曲线参数 A_n，再迭代生成
    高阶曲线调整结果，作为后续学生网络蒸馏的监督目标。
    """
    result = image
    for alpha in split_curve_params(curve_params, iterations):
        # 对应于公式：LE_n = LE_{n-1} + A_n * LE_{n-1} * (1 - LE_{n-1})
        result = result + alpha * result * (1.0 - result)
    return result.clamp(0.0, 1.0)


def apply_tangent_curve(image: Tensor, tangent_params: Tensor) -> Tensor:
    """应用 CuDi 学生分支的切线近似映射。
    CuDi 用高阶曲线在输入点附近的切线近似教师分支的 LE_8 结果，
    映射形式为 TL(I) = K * I + B。学生网络输出 6 个通道：前 3 个
    通道是 RGB 斜率图 K，后 3 个通道是 RGB 截距图 B。
    """
    if tangent_params.dim() != 4:
        raise ValueError("tangent parameters must be a 4D tensor")
    if tangent_params.size(1) != 6:
        raise ValueError(f"expected 6 channels, got {tangent_params.size(1)}")
    slope, intercept = torch.chunk(tangent_params, 2, dim=1)
    return (slope * image + intercept).clamp(0.0, 1.0)


def split_tangent_params(tangent_params: Tensor) -> tuple[Tensor, Tensor]:
    """将学生网络输出拆分为斜率图 K 和截距图 B。"""
    if tangent_params.dim() != 4:
        raise ValueError("tangent parameters must be a 4D tensor")
    if tangent_params.size(1) != 6:
        raise ValueError(f"expected 6 channels, got {tangent_params.size(1)}")
    return torch.chunk(tangent_params, 2, dim=1)
