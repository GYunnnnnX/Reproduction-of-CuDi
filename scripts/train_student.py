"""Distill CuDi student network from teacher."""

"""从训练好的 CuDi 教师网络蒸馏学生网络。
学生网络学习拟合教师网络的高阶曲线输出。
训练完成后，推理阶段只保留学生网络，更快、更轻量。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cudi.data.dataset import ImageFolderDataset, random_exposure_map
from cudi.losses.zero_reference import CuDiStudentLoss
from cudi.models.student import CuDiStudent
from cudi.models.teacher import CuDiTeacher
from cudi.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--teacher", default="checkpoints/teacher.pt")
    parser.add_argument("--output", default="checkpoints/student.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    train_dir = args.train_dir or config["paths"]["train_dir"]

    # 学生蒸馏阶段使用和教师训练相同类型的无标签图像。
    dataset = ImageFolderDataset(train_dir, image_size=config["training"]["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # 加载已经训练好的教师网络，并固定其参数，当作监督信号生成器。
    teacher = CuDiTeacher().to(device)
    checkpoint = torch.load(args.teacher, map_location=device)
    teacher.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    # 学生网络预测切线参数 K/B
    student = CuDiStudent(downsample=config["student"]["downsample"]).to(device)
    criterion = CuDiStudentLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=config["student"]["lr"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["student"]["epochs"]):
        student.train()
        progress = tqdm(loader, desc=f"student {epoch + 1}/{config['student']['epochs']}")
        for image in progress:
            image = image.to(device)
            exposure = random_exposure_map(image.size(0), image.size(2), image.size(3), device)

            # 教师权重固定，teacher_result 作为学生蒸馏目标。
            with torch.no_grad():
                teacher_result, _ = teacher(image, exposure)
            student_result, _ = student(image, exposure)
            loss = criterion(student_result, teacher_result)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.save({"model": student.state_dict(), "epoch": epoch + 1}, output)


if __name__ == "__main__":
    main()
