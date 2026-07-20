"""CuDi dataset definitions."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset[Tensor]):
    """从文件夹递归读取图像的数据集。
    """

    def __init__(self, root: str | Path, image_size: int = 256) -> None:
        self.root = Path(root)
        # 递归收集 root 下所有支持格式的图像路径。
        self.paths = sorted(path for path in self.root.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES)
        if not self.paths:
            raise FileNotFoundError(f"no images found in {self.root}")
        self.transform = transforms.Compose(
            [
                # 训练时统一图像尺寸，便于 batch 组合。
                transforms.Resize((image_size, image_size)),
                # 转成 [0, 1] 范围的张量，形状为 [C, H, W]。
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        """返回数据集中图像数量。"""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        """读取第 index 张图像，并转换成 RGB tensor。"""
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


def random_exposure_map(batch_size: int, height: int, width: int, device: torch.device | str) -> Tensor:
    """生成训练阶段使用的随机条件曝光图 E。

    论文中训练时会在曝光图中随机选取一个任意形状区域，并给区域内外赋予不同的随机
    曝光值。这里将取值范围扩展到 [0.0, 1.0]，覆盖更极端、全面的曝光控制条件。
    通过低分辨率随机噪声上采样并阈值化生成不规则二值区域 mask，再分别为 mask 内外赋值。
    """
    exposure = torch.empty(batch_size, 1, height, width, device=device)
    mask_height = max(4, height // 32)
    mask_width = max(4, width // 32)

    for index in range(batch_size):
        inside = random.uniform(0.0, 1.0)
        outside = random.uniform(0.0, 1.0)
        noise = torch.rand(1, 1, mask_height, mask_width, device=device)
        noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
        threshold = random.uniform(0.35, 0.65)
        mask = noise > threshold
        exposure[index].fill_(outside)
        exposure[index].masked_fill_(mask[0], inside)

    return exposure


def uniform_exposure_map(batch_size: int, height: int, width: int, value: float, device: torch.device | str) -> Tensor:
    """生成全图统一曝光值的条件曝光图，常用于推理阶段的全局曝光控制。"""
    return torch.full((batch_size, 1, height, width), value, device=device)


def spatial_exposure_map(image: Tensor, base: float, amplitude: float = 0.15) -> Tensor:
    """根据输入图像亮度生成空间变化的条件曝光图。

    对应论文中的 S + A * Norm(L_avg - L)：暗区域会得到更大的曝光值，亮区域
    会得到更小的曝光值，从而实现局部自适应曝光调节。
    """
    luminance = image.mean(dim=1, keepdim=True)
    luminance_avg = luminance.mean(dim=(2, 3), keepdim=True)
    delta = luminance_avg - luminance
    min_value = delta.amin(dim=(2, 3), keepdim=True)
    max_value = delta.amax(dim=(2, 3), keepdim=True)
    normalized = 2.0 * (delta - min_value) / (max_value - min_value + 1e-6) - 1.0
    return (base + amplitude * normalized).clamp(0.0, 1.0)
