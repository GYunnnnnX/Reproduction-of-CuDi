"""Run CuDi inference."""

"""使用蒸馏后的 CuDi 学生网络进行推理。
训练完成后只保留学生网络。推理时给定输入图像 I 和条件曝光图 E，
学生网络预测切线参数 K/B，并用 Y = K * I + B 得到最终增强图像。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cudi.data.dataset import spatial_exposure_map, uniform_exposure_map
from cudi.models.student import CuDiStudent
from cudi.utils.image import load_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/student.pt")
    parser.add_argument("--exposure", type=float, default=0.65)
    parser.add_argument("--mode", choices=["uniform", "spatial"], default="uniform")
    parser.add_argument("--base", type=float, default=None)
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    image = load_image(args.input, device)

    # 推理阶段可以使用统一曝光图，也可以使用根据图像亮度生成的空间变化曝光图。
    if args.mode == "spatial":
        exposure = spatial_exposure_map(image, base=args.base if args.base is not None else args.exposure, amplitude=args.amplitude)
    else:
        exposure = uniform_exposure_map(image.size(0), image.size(2), image.size(3), args.exposure, device)

    # 推理阶段只加载学生网络
    model = CuDiStudent().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    with torch.no_grad():
        result, _ = model(image, exposure)

    save_image(result, Path(args.output))


if __name__ == "__main__":
    main()
