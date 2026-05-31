"""YOLO 后处理单测: 坐标还原回原图、类别过滤、置信度过滤."""
import numpy as np

from app.yolo.nms import nms
from app.yolo.postprocess import postprocess
from app.yolo.preprocess import LetterboxInfo


def test_nms_keeps_highest_scored_overlapping_boxes():
    boxes = np.asarray([[0, 0, 100, 100], [10, 10, 110, 110], [200, 200, 250, 250]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    keep = nms(boxes, scores, iou_threshold=0.5)

    assert keep == [0, 2]


def test_postprocess_filters_classes_and_restores_letterbox_coordinates():
    labels = ["person", "bicycle", "car"]
    info = LetterboxInfo(ratio=1.0, pad=(0.0, 80.0), original_shape=(480, 640), input_shape=(640, 640))
    raw = np.zeros((1, 84, 3), dtype=np.float32)
    # YOLOv8 format: cx, cy, w, h + class scores. The first candidate is a car
    # in padded 640x640 coordinates, so y must subtract top pad during restore.
    raw[0, 0:4, 0] = [320, 320, 100, 80]
    raw[0, 4 + 2, 0] = 0.91
    raw[0, 0:4, 1] = [320, 320, 100, 80]
    raw[0, 4 + 0, 1] = 0.95
    raw[0, 0:4, 2] = [50, 50, 20, 20]
    raw[0, 4 + 2, 2] = 0.1

    detections = postprocess(raw, info, labels, conf_threshold=0.25, nms_threshold=0.45, target_classes={"car"})

    assert len(detections) == 1
    det = detections[0]
    assert det.label == "car"
    assert det.class_id == 2
    assert det.confidence == 0.91
    assert det.bbox_xyxy == [270.0, 200.0, 370.0, 280.0]
