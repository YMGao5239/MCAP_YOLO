"""质量评估单测。"""
import cv2
import numpy as np

from app.quality.analyzer import QualityAnalyzer
from app.quality.metrics import compute_frame_metrics


def textured_frame(width: int = 640, height: int = 480) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(0, width, 20):
        color = (40 + x % 180, 180 - x % 120, 80 + x % 140)
        cv2.rectangle(image, (x, 0), (min(width - 1, x + 10), height - 1), color, -1)
    cv2.putText(image, "sharp frame", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    return image


def test_compute_frame_metrics_for_normal_image():
    image = textured_frame()

    metrics = compute_frame_metrics(image)

    assert metrics.width == 640
    assert metrics.height == 480
    assert 20.0 < metrics.brightness_mean < 230.0
    assert metrics.brightness_std > 20.0
    assert metrics.contrast_score > 20.0
    assert metrics.blur_score > 100.0
    assert 0.0 <= metrics.saturation_mean <= 255.0
    assert metrics.aspect_ratio == 640 / 480
    assert metrics.is_empty is False
    assert metrics.is_corrupted is False


def test_analyzer_flags_dark_blurry_low_resolution_and_low_contrast_frames():
    analyzer = QualityAnalyzer()
    dark = np.zeros((480, 640, 3), dtype=np.uint8)
    low_resolution = textured_frame(120, 90)
    blurry = cv2.GaussianBlur(textured_frame(), (41, 41), 0)
    low_contrast = np.full((480, 640, 3), 118, dtype=np.uint8)

    dark_result = analyzer.analyze(dark)
    low_res_result = analyzer.analyze(low_resolution)
    blurry_result = analyzer.analyze(blurry)
    low_contrast_result = analyzer.analyze(low_contrast)

    assert dark_result.is_too_dark is True
    assert dark_result.is_corrupted is True
    assert low_res_result.is_low_resolution is True
    assert blurry_result.is_blurry is True
    assert low_contrast_result.is_low_contrast is True


def test_analyzer_flags_bright_and_invalid_frames_without_crashing():
    analyzer = QualityAnalyzer()
    bright = np.full((480, 640, 3), 255, dtype=np.uint8)
    invalid = np.asarray([], dtype=np.uint8)

    bright_result = analyzer.analyze(bright)
    invalid_result = analyzer.analyze(invalid)

    assert bright_result.is_too_bright is True
    assert bright_result.is_corrupted is True
    assert invalid_result.is_corrupted is True
    assert invalid_result.metrics.width == 0
    assert invalid_result.metrics.height == 0
