"""质量指标计算工具 (供 analyzer 使用).

亮度、对比度、模糊(Laplacian 方差)、饱和度、空帧、欠/过曝、
分辨率异常、宽高比异常、颜色通道异常、压缩伪影、时间戳异常等。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    width: int
    height: int
    aspect_ratio: float
    brightness_mean: float
    brightness_std: float
    contrast_score: float
    blur_score: float
    saturation_mean: float
    is_empty: bool
    is_corrupted: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def compute_frame_metrics(image: np.ndarray | None) -> FrameMetrics:
    """Compute frame-level quality metrics from a BGR image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return _empty_metrics(is_corrupted=True)

    if image.ndim not in {2, 3}:
        return _empty_metrics(is_corrupted=True)

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return _empty_metrics(is_corrupted=True)

    try:
        if image.ndim == 2:
            gray = image
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            gray = image[:, :, 0]
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] >= 3:
            bgr = image[:, :, :3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            return _empty_metrics(is_corrupted=True)

        gray_u8 = _to_uint8(gray)
        bgr_u8 = _to_uint8(bgr)
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
        brightness_mean = float(np.mean(gray_u8))
        brightness_std = float(np.std(gray_u8))
        blur_score = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        saturation_mean = float(np.mean(hsv[:, :, 1]))
        is_empty = brightness_std < 1.0 and (brightness_mean < 2.0 or brightness_mean > 253.0)
        return FrameMetrics(
            width=int(width),
            height=int(height),
            aspect_ratio=float(width / height),
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            contrast_score=brightness_std,
            blur_score=blur_score,
            saturation_mean=saturation_mean,
            is_empty=bool(is_empty),
            is_corrupted=bool(is_empty),
        )
    except Exception:
        return _empty_metrics(is_corrupted=True)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if image.size == 0:
        return np.asarray(image, dtype=np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _empty_metrics(is_corrupted: bool) -> FrameMetrics:
    return FrameMetrics(
        width=0,
        height=0,
        aspect_ratio=0.0,
        brightness_mean=0.0,
        brightness_std=0.0,
        contrast_score=0.0,
        blur_score=0.0,
        saturation_mean=0.0,
        is_empty=True,
        is_corrupted=is_corrupted,
    )
