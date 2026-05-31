"""检测结果可视化.

在原图上绘制 bbox + label + confidence,用于 detection_samples/ 和
单帧 YOLO 预览接口。
"""
from __future__ import annotations

import cv2
import numpy as np

from app.yolo.postprocess import Detection


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        color = _color_for_class(det.class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(output, (x1, top), (x1 + text_w + 4, top + text_h + baseline + 4), color, -1)
        cv2.putText(output, label, (x1 + 2, top + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 180, 255),
        (80, 200, 80),
        (255, 120, 80),
        (220, 80, 220),
        (80, 160, 255),
    ]
    return palette[class_id % len(palette)]
