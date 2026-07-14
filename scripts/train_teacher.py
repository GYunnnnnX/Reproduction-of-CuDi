"""Train CuDi teacher network."""

"""训练 CuDi 教师网络。
教师网络负责学习 Zero-DCE 风格的高阶曲线参数 A，并通过 LE_8(I) 得到增强结果。
训练教师时依赖零参考损失和条件曝光图 E。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cudi.data.dataset import ImageFolderDataset, random_exposure_map
from cudi.losses.zero_reference import CuDiTeacherLoss
from cudi.models.teacher import CuDiTeacher
from cudi.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--output", default="checkpoints/teacher.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    train_dir = args.train_dir or config["paths"]["train_dir"]

    # 零参考训练，只需要读取单张正常亮度图像，不需要成对标签。
    dataset = ImageFolderDataset(train_dir, image_size=config["training"]["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # 教师网络输出 8 组 RGB 曲线参数，并通过高阶 LE 曲线生成增强结果。
    model = CuDiTeacher().to(device)
    criterion = CuDiTeacherLoss(
        lambda_sec=config["teacher"]["lambda_sec"],
        lambda_sc=config["teacher"]["lambda_sc"],
        lambda_cc=config["teacher"]["lambda_cc"],
        lambda_is=config["teacher"]["lambda_is"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["teacher"]["lr"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["teacher"]["epochs"]):
        model.train()
        progress = tqdm(loader, desc=f"teacher {epoch + 1}/{config['teacher']['epochs']}")
        for image in progress:
            image = image.to(device)
            # 训练阶段随机生成条件曝光图 E，用来控制目标曝光分布。
            exposure = random_exposure_map(image.size(0), image.size(2), image.size(3), device)

            # result 是教师高阶曲线输出 R，params 是曲线参数 A。
            result, params = model(image, exposure)
            loss, logs = criterion(image, exposure, result, params)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            progress.set_postfix({key: f"{value.item():.4f}" for key, value in logs.items()})

        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, output)


if __name__ == "__main__":
    main()
