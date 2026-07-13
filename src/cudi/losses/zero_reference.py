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
        """计算增强结果和曝光图在局部 patch 平均亮度上的 L1 距离。"""
        result_mean = F.avg_pool2d(rgb_to_intensity(result), self.patch_size)
        exposure_mean = F.avg_pool2d(exposure_map, self.patch_size)
        return F.l1_loss(result_mean, exposure_mean)


class SpatialConsistencyLoss(nn.Module):
    """空间一致性损失（L_sc）。
    它约束增强前后相邻 patch 的亮度差异关系保持一致，避免增强结果破坏原图中
    的局部结构、边缘和明暗相对关系。
    """

    def __init__(self, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, image: Tensor, result: Tensor) -> Tensor:
        """比较输入图和增强图在水平、垂直相邻 patch 上的亮度差。"""
        image_pool = F.avg_pool2d(rgb_to_intensity(image), self.patch_size)
        result_pool = F.avg_pool2d(rgb_to_intensity(result), self.patch_size)

        # 输入图中相邻 patch 的亮度，用于表示原始空间结构。
        image_left = image_pool[:, :, :, :-1]
        image_right = image_pool[:, :, :, 1:]
        image_up = image_pool[:, :, :-1, :]
        image_down = image_pool[:, :, 1:, :]

        # 增强图中相邻 patch 的亮度，用于和输入图的结构关系进行对齐。
        result_left = result_pool[:, :, :, :-1]
        result_right = result_pool[:, :, :, 1:]
        result_up = result_pool[:, :, :-1, :]
        result_down = result_pool[:, :, 1:, :]

        horizontal = (torch.abs(result_left - result_right) - torch.abs(image_left - image_right)).pow(2)
        vertical = (torch.abs(result_up - result_down) - torch.abs(image_up - image_down)).pow(2)
        return horizontal.mean() + vertical.mean()


class ColorConstancyLoss(nn.Module):
    """颜色恒常性损失（L_cc）。
    它约束增强结果的 RGB 三个通道均值不要相差过大，用来减少明显偏色。
    """

    def forward(self, result: Tensor) -> Tensor:
        """计算增强图 RGB 通道均值两两之间的平方差。"""
        # 假设 result 的形状是[B, 3, H, W]
        channel_mean = result.mean(dim=(2, 3))
        r, g, b = channel_mean[:, 0], channel_mean[:, 1], channel_mean[:, 2]
        return (r - g).pow(2).mean() + (r - b).pow(2).mean() + (g - b).pow(2).mean()


class IlluminationSmoothnessLoss(nn.Module):
    """光照平滑损失（L_is）。
    教师网络会输出多组曲线参数 A_n。这个损失约束每组参数图在空间上平滑，
    避免相邻像素的曲线参数剧烈变化，从而减少增强结果中的噪声和伪影。
    """

    def __init__(self, iterations: int = 8) -> None:
        super().__init__()
        self.iterations = iterations

    def forward(self, curve_params: Tensor) -> Tensor:
        """对每次迭代的 3 通道曲线参数分别计算水平和垂直方向平滑项。"""
        if curve_params.size(1) != self.iterations * 3:
            raise ValueError(f"expected {self.iterations * 3} channels, got {curve_params.size(1)}")

        loss = curve_params.new_tensor(0.0)
        for params in torch.chunk(curve_params, self.iterations, dim=1):
            grad_x = torch.abs(params[:, :, :, 1:] - params[:, :, :, :-1])
            grad_y = torch.abs(params[:, :, 1:, :] - params[:, :, :-1, :])

            # 对每个 RGB 通道分别计算水平/垂直梯度的 L1 范数，对应公式中的 c ∈ {r,g,b}。
            grad_x_l1 = grad_x.mean(dim=(2, 3))
            grad_y_l1 = grad_y.mean(dim=(2, 3))
            loss = loss + (grad_x_l1 + grad_y_l1).pow(2).sum(dim=1).mean()
        return loss / self.iterations


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
