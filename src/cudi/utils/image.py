"""Image IO helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor


def load_image(path: str | Path, device: torch.device | str) -> Tensor:
    image = Image.open(path).convert("RGB")
    return to_tensor(image).unsqueeze(0).to(device)


def save_image(tensor: Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).cpu()
    if image.dim() == 4:
        image = image[0]
    to_pil_image(image).save(path)
