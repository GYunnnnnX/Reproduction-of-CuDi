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

    论文中训练时会在曝光图中随机选取一个区域，并给区域内外赋予不同的随机
    曝光值，取值范围为 [0.2, 0.8]。这里用随机矩形近似“任意形状区域”。
    """
    exposure = torch.empty(batch_size, 1, height, width, device=device)
    for index in range(batch_size):
        inside = random.uniform(0.2, 0.8)
        outside = random.uniform(0.2, 0.8)
        y1 = random.randint(0, max(height - 1, 0))
        y2 = random.randint(y1 + 1, height)
        x1 = random.randint(0, max(width - 1, 0))
        x2 = random.randint(x1 + 1, width)
        exposure[index].fill_(outside)
        exposure[index, :, y1:y2, x1:x2] = inside

    # 用平均池化平滑曝光图边界，避免区域边缘过于生硬。
    radius = max(3, min(height, width) // 8)
    if radius % 2 == 0:
        radius += 1
    exposure = F.avg_pool2d(exposure, kernel_size=radius, stride=1, padding=radius // 2)
    return exposure.clamp(0.2, 0.8)


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
