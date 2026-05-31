"""YOLO 前处理单测: letterbox 形状、归一化范围、ratio/pad 正确性."""
import numpy as np

from app.yolo.preprocess import preprocess


def test_preprocess_letterboxes_to_chw_float_batch():
    image = np.full((480, 640, 3), (10, 20, 30), dtype=np.uint8)

    result = preprocess(image, image_size=640)

    assert result.tensor.shape == (1, 3, 640, 640)
    assert result.tensor.dtype == np.float32
    assert 0.0 <= float(result.tensor.min()) <= float(result.tensor.max()) <= 1.0
    assert result.ratio == 1.0
    assert result.pad == (0.0, 80.0)
    assert result.original_shape == (480, 640)
    assert result.input_shape == (640, 640)


def test_preprocess_preserves_aspect_ratio_for_tall_image():
    image = np.zeros((800, 400, 3), dtype=np.uint8)

    result = preprocess(image, image_size=640)

    assert result.tensor.shape == (1, 3, 640, 640)
    assert result.ratio == 0.8
    assert result.pad == (160.0, 0.0)
