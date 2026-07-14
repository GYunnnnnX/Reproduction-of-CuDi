"""Basic smoke tests for CuDi modules."""

from __future__ import annotations

import torch

from cudi.data.dataset import random_exposure_map
from cudi.losses.zero_reference import CuDiTeacherLoss
from cudi.models.student import CuDiStudent
from cudi.models.teacher import CuDiTeacher


def test_teacher_student_forward() -> None:
    image = torch.rand(2, 3, 64, 64)
    exposure = random_exposure_map(2, 64, 64, image.device)

    teacher = CuDiTeacher()
    teacher_result, curve_params = teacher(image, exposure)
    assert teacher_result.shape == image.shape
    assert curve_params.shape == (2, 24, 64, 64)

    loss, logs = CuDiTeacherLoss()(image, exposure, teacher_result, curve_params)
    assert loss.ndim == 0
    assert set(logs) == {"total", "sec", "sc", "cc", "is"}

    student = CuDiStudent()
    student_result, tangent_params = student(image, exposure)
    assert student_result.shape == image.shape
    assert tangent_params.shape == (2, 6, 64, 64)
