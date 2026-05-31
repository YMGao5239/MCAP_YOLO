"""YOLO 后处理.

 - 解析模型原始输出 → 候选框 (xywh/xyxy + obj/cls 置信度)
 - 置信度过滤 (conf-threshold)
 - 坐标映射回原图 (反 letterbox)
 - 关键目标类别过滤 (--target-classes person,car,truck,bus)
 - 输出 bbox, class_id, label, confidence
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.yolo.nms import nms
from app.yolo.preprocess import LetterboxInfo


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: list[float]
    class_id: int
    label: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def postprocess(
    raw_output: np.ndarray | list[np.ndarray],
    letterbox: LetterboxInfo,
    labels: list[str],
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    target_classes: set[str] | None = None,
) -> list[Detection]:
    predictions = _normalize_output(raw_output)
    if predictions.size == 0:
        return []

    boxes_xywh = predictions[:, :4]
    if predictions.shape[1] == 85:  # YOLOv5: xywh + objectness + class probs
        class_scores = predictions[:, 5:] * predictions[:, 4:5]
    else:  # YOLOv8/v11 export: xywh + class scores
        class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]
    if boxes_xywh.size == 0:
        return []

    boxes = _xywh_to_xyxy(boxes_xywh)
    boxes = _restore_boxes(boxes, letterbox)
    valid = np.asarray([0 <= class_id < len(labels) for class_id in class_ids], dtype=bool)
    boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]

    if target_classes is not None:
        target_mask = np.asarray([labels[int(class_id)] in target_classes for class_id in class_ids], dtype=bool)
        boxes, class_ids, confidences = boxes[target_mask], class_ids[target_mask], confidences[target_mask]

    if boxes.size == 0:
        return []

    detections: list[Detection] = []
    for class_id in sorted(set(int(item) for item in class_ids)):
        mask = class_ids == class_id
        local_indices = np.where(mask)[0]
        keep_indices = nms(boxes[mask], confidences[mask], iou_threshold=nms_threshold)
        for keep_idx in keep_indices:
            idx = local_indices[keep_idx]
            detections.append(
                Detection(
                    bbox_xyxy=[round(float(v), 2) for v in boxes[idx].tolist()],
                    class_id=int(class_ids[idx]),
                    label=labels[int(class_ids[idx])],
                    confidence=round(float(confidences[idx]), 4),
                )
            )
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def _normalize_output(raw_output: np.ndarray | list[np.ndarray]) -> np.ndarray:
    output = raw_output[0] if isinstance(raw_output, list) else raw_output
    output = np.asarray(output)
    if output.ndim == 3:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"Unsupported YOLO output shape: {output.shape}")
    if output.shape[0] in {84, 85}:
        output = output.T
    return output.astype(np.float32)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _restore_boxes(boxes: np.ndarray, letterbox: LetterboxInfo) -> np.ndarray:
    restored = boxes.copy().astype(np.float32)
    pad_x, pad_y = letterbox.pad
    restored[:, [0, 2]] = (restored[:, [0, 2]] - pad_x) / letterbox.ratio
    restored[:, [1, 3]] = (restored[:, [1, 3]] - pad_y) / letterbox.ratio
    original_h, original_w = letterbox.original_shape
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, original_w)
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, original_h)
    return restored
