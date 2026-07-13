"""CuDi student network."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from cudi.models.curves import apply_tangent_curve

'''深度可分离卷积层'''
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

'''学生网络块，包含两个深度可分离卷积层'''
def _student_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        DepthwiseSeparableConv(in_channels, in_channels, 3, in_channels),
        DepthwiseSeparableConv(in_channels, out_channels, 1, 1),
        nn.ReLU(inplace=True),
    )


'''CuDi 学生网络'''
class CuDiStudent(nn.Module):
    def __init__(self, in_channels: int = 4, downsample: int = 4) -> None:
        super().__init__()
        self.downsample = downsample

        self.l1 = _student_block(in_channels, 16)
        self.l2 = _student_block(16, 16)
        self.l3 = _student_block(16, 16)
        self.l4 = _student_block(16, 16)
        self.l5 = _student_block(32, 16)
        self.l6 = _student_block(32, 16)
        self.l7 = nn.Sequential(
            DepthwiseSeparableConv(32, 32, 3, 32),
            DepthwiseSeparableConv(32, 6, 1, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward_params(self, image: Tensor, exposure_map: Tensor) -> Tensor:
        x = torch.cat([image, exposure_map], dim=1)
        original_size = x.shape[-2:]
        if self.downsample > 1:
            x = F.interpolate(x, scale_factor=1.0 / self.downsample, mode="bilinear", align_corners=False)

        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        l5 = self.l5(torch.cat([l4, l3], dim=1))
        l6 = self.l6(torch.cat([l5, l2], dim=1))
        params = self.l7(torch.cat([l6, l1], dim=1))

        if params.shape[-2:] != original_size:
            params = F.interpolate(params, size=original_size, mode="bilinear", align_corners=False)
        return params

    def forward(self, image: Tensor, exposure_map: Tensor) -> tuple[Tensor, Tensor]:
        tangent_params = self.forward_params(image, exposure_map)
        '''应用学生网络切线近似映射'''
        enhanced = apply_tangent_curve(image, tangent_params)
        return enhanced, tangent_params
