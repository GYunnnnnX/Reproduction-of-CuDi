"""CuDi teacher network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cudi.models.curves import apply_le_curve

'''一个卷积块包含三层卷积层'''
def _conv_block(in_channels: int, out_channels: int, layers: int = 3) -> nn.Sequential:
    modules: list[nn.Module] = []
    channels = in_channels
    for _ in range(layers):
        modules.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
        modules.append(nn.ReLU(inplace=True))
        channels = out_channels
    return nn.Sequential(*modules)

'''CuDi 教师网络'''
class CuDiTeacher(nn.Module):
    def __init__(self, in_channels: int = 4, iterations: int = 8) -> None:
        super().__init__()
        self.iterations = iterations
        '''原文的Unet-like network'''
        self.l1 = _conv_block(in_channels, 32)
        self.l2 = _conv_block(32, 64)
        self.l3 = _conv_block(64, 128)
        self.l4 = _conv_block(128, 256)
        self.l5 = _conv_block(256, 256)
        self.l6 = _conv_block(384, 128)
        self.l7 = _conv_block(192, 64)
        self.l8 = _conv_block(96, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, iterations * 3, kernel_size=3, padding=1),
            nn.Tanh(),
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

        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        l5 = self.l5(l4)
        l6 = self.l6(torch.cat([l5, l3], dim=1))
        l7 = self.l7(torch.cat([l6, l2], dim=1))
        l8 = self.l8(torch.cat([l7, l1], dim=1))
        return self.out(l8)

    def forward(self, image: Tensor, exposure_map: Tensor) -> tuple[Tensor, Tensor]:
        curve_params = self.forward_params(image, exposure_map)
        '''应用 Zero-DCE 的逐像素高阶 LE 曲线'''
        enhanced = apply_le_curve(image, curve_params, iterations=self.iterations)
        return enhanced, curve_params
