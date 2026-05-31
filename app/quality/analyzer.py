"""单帧图像质量检测.

每帧至少计算: brightness_mean, brightness_std, blur_score(拉普拉斯方差),
contrast_score, saturation_mean, width, height。
问题检测标志: is_too_dark, is_too_bright, is_blurry, is_low_contrast,
is_low_resolution, is_corrupted。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.quality.metrics import FrameMetrics, compute_frame_metrics


@dataclass(frozen=True)
class QualityThresholds:
    dark_mean: float = 35.0
    bright_mean: float = 245.0
    blur_score: float = 80.0
    contrast_score: float = 18.0
    min_width: int = 320
    min_height: int = 240


@dataclass(frozen=True)
class FrameQuality:
    metrics: FrameMetrics
    is_too_dark: bool
    is_too_bright: bool
    is_blurry: bool
    is_low_contrast: bool
    is_low_resolution: bool
    is_corrupted: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


class QualityAnalyzer:
    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def analyze(self, image: np.ndarray | None) -> FrameQuality:
        metrics = compute_frame_metrics(image)
        is_low_resolution = metrics.width < self.thresholds.min_width or metrics.height < self.thresholds.min_height
        is_too_dark = metrics.brightness_mean <= self.thresholds.dark_mean
        is_too_bright = metrics.brightness_mean >= self.thresholds.bright_mean
        is_low_contrast = metrics.contrast_score < self.thresholds.contrast_score
        is_blurry = metrics.blur_score < self.thresholds.blur_score and not metrics.is_empty
        is_corrupted = metrics.is_corrupted or metrics.is_empty
        return FrameQuality(
            metrics=metrics,
            is_too_dark=is_too_dark,
            is_too_bright=is_too_bright,
            is_blurry=is_blurry,
            is_low_contrast=is_low_contrast,
            is_low_resolution=is_low_resolution,
            is_corrupted=is_corrupted,
        )


def analyze_frame(image: np.ndarray | None, thresholds: QualityThresholds | None = None) -> FrameQuality:
    return QualityAnalyzer(thresholds=thresholds).analyze(image)
