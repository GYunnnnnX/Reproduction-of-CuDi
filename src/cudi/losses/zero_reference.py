"""Zero-reference losses used by CuDi."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def rgb_to_intensity(image: Tensor) -> Tensor:
    """把 RGB 图像转换成单通道亮度图。
    使用 RGB 三通道的简单平均值作为亮度近似，后续曝光控制损失和空间
    一致性损失都只关心亮度变化，而不是直接在 RGB 三通道上计算。
    """
    return image.mean(dim=1, keepdim=True)


class SelfSupervisedExposureControlLoss(nn.Module):
    """自监督空间曝光控制损失（L_sec）。
    它把增强结果的局部平均亮度拉向条件曝光图 E 的局部平均值，使网络输出的
    亮度分布能够受 exposure_map 控制。
    """

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, result: Tensor, exposure_map: Tensor) -> Tensor:
        """计算 16×16 非重叠局部 patch 平均亮度之间的 L1 距离。"""
        result_mean = F.avg_pool2d(rgb_to_intensity(result), kernel_size=self.patch_size, stride=self.patch_size)
        exposure_mean = F.avg_pool2d(exposure_map, kernel_size=self.patch_size, stride=self.patch_size)
        return F.l1_loss(result_mean, exposure_mean)


class SpatialConsistencyLoss(nn.Module):
    """空间一致性损失（L_sc）。
    四方向差分核，约束增强前后局部亮度梯度保持一致。
    """

    def __init__(self, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.register_buffer("weight_left", torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("weight_right", torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("weight_up", torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("weight_down", torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, image: Tensor, result: Tensor) -> Tensor:
        """比较输入图和增强图的四方向局部亮度差。"""
        image_pool = F.avg_pool2d(rgb_to_intensity(image), kernel_size=self.patch_size, stride=self.patch_size)
        result_pool = F.avg_pool2d(rgb_to_intensity(result), kernel_size=self.patch_size, stride=self.patch_size)

        diff_left = F.conv2d(result_pool, self.weight_left, padding=1) - F.conv2d(image_pool, self.weight_left, padding=1)
        diff_right = F.conv2d(result_pool, self.weight_right, padding=1) - F.conv2d(image_pool, self.weight_right, padding=1)
        diff_up = F.conv2d(result_pool, self.weight_up, padding=1) - F.conv2d(image_pool, self.weight_up, padding=1)
        diff_down = F.conv2d(result_pool, self.weight_down, padding=1) - F.conv2d(image_pool, self.weight_down, padding=1)

        return diff_left.pow(2).mean() + diff_right.pow(2).mean() + diff_up.pow(2).mean() + diff_down.pow(2).mean()


class ColorConstancyLoss(nn.Module):
    """颜色恒常性损失（L_cc）。
    它约束增强结果的 RGB 三个通道均值不要相差过大，用来减少明显偏色。
    """

    def forward(self, result: Tensor) -> Tensor:
        """计算增强图 RGB 通道均值之间的颜色恒常性损失。"""
        mean_rgb = result.mean(dim=(2, 3), keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = (mr - mg).pow(2)
        drb = (mr - mb).pow(2)
        dgb = (mb - mg).pow(2)
        return torch.sqrt(drg.pow(2) + drb.pow(2) + dgb.pow(2)).mean()


class IlluminationSmoothnessLoss(nn.Module):
    """光照平滑损失（L_is）。
    对整张曲线参数图做平滑约束。
    """

    def __init__(self, TVLoss_weight: float = 1.0) -> None:
        super().__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, curve_params: Tensor) -> Tensor:
        """计算曲线参数图的光照平滑损失。"""
        batch_size = curve_params.size(0)
        h_x = curve_params.size(2)
        w_x = curve_params.size(3)
        count_h = max((h_x - 1) * w_x, 1)
        count_w = max(h_x * (w_x - 1), 1)
        h_tv = torch.pow(curve_params[:, :, 1:, :] - curve_params[:, :, : h_x - 1, :], 2).sum()
        w_tv = torch.pow(curve_params[:, :, :, 1:] - curve_params[:, :, :, : w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class CuDiTeacherLoss(nn.Module):
    """教师网络总损失。
    教师网络沿用 Zero-DCE 的高阶曲线增强形式，并加入 CuDi 的条件曝光控制。
    总损失为：10 * L_sec + 1 * L_sc + 5 * L_cc + 200 * L_is。
    """

    def __init__(
        self,
        lambda_sec: float = 10.0,
        lambda_sc: float = 1.0,
        lambda_cc: float = 5.0,
        lambda_is: float = 200.0,
    ) -> None:
        super().__init__()
        self.lambda_sec = lambda_sec
        self.lambda_sc = lambda_sc
        self.lambda_cc = lambda_cc
        self.lambda_is = lambda_is
        self.sec = SelfSupervisedExposureControlLoss()
        self.sc = SpatialConsistencyLoss()
        self.cc = ColorConstancyLoss()
        self.is_loss = IlluminationSmoothnessLoss()

    def forward(self, image: Tensor, exposure_map: Tensor, result: Tensor, curve_params: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """计算教师网络训练时的总损失，并返回各个子损失用于日志记录。"""
        sec = self.sec(result, exposure_map)
        sc = self.sc(image, result)
        cc = self.cc(result)
        is_loss = self.is_loss(curve_params)
        total = self.lambda_sec * sec + self.lambda_sc * sc + self.lambda_cc * cc + self.lambda_is * is_loss
        return total, {
            "total": total.detach(),
            "sec": sec.detach(),
            "sc": sc.detach(),
            "cc": cc.detach(),
            "is": is_loss.detach(),
        }


class CuDiStudentLoss(nn.Module):
    """学生网络蒸馏损失。

    训练学生网络时，教师网络权重固定。学生输出的切线映射结果与
    教师高阶 LE 曲线增强结果直接使用 L1 损失。
    """

    def forward(self, student_result: Tensor, teacher_result: Tensor) -> Tensor:
        """计算学生增强结果和教师增强结果之间的 L1 距离。"""
        return F.l1_loss(student_result, teacher_result)
