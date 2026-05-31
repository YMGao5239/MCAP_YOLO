"""YOLO 前处理.

letterbox resize → BGR2RGB → /255 归一化 → HWC2CHW → 增加 batch 维。
记录缩放比例与 padding,供后处理坐标还原使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    ratio: float
    pad: tuple[float, float]
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]


@dataclass(frozen=True)
class PreprocessResult:
    tensor: np.ndarray
    image: np.ndarray
    ratio: float
    pad: tuple[float, float]
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]

    @property
    def letterbox(self) -> LetterboxInfo:
        return LetterboxInfo(self.ratio, self.pad, self.original_shape, self.input_shape)


def preprocess(image: np.ndarray, image_size: int = 640, pad_value: int = 114) -> PreprocessResult:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty ndarray")
    height, width = image.shape[:2]
    ratio = min(image_size / height, image_size / width)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_w = image_size - new_width
    pad_h = image_size - new_height
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )

    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return PreprocessResult(
        tensor=np.ascontiguousarray(tensor),
        image=letterboxed,
        ratio=round(float(ratio), 6),
        pad=(float(left), float(top)),
        original_shape=(height, width),
        input_shape=(image_size, image_size),
    )
