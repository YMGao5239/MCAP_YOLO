"""集中配置.

阈值、默认路径、Topic 等通过此处或环境变量管理,避免硬编码。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppDefaults:
    quality_threshold: float = 0.6
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    sample_every_n: int = 1
    output_dir: str = "outputs"
    model_path: str = "models/yolov8n.onnx"
    labels_path: str = "models/coco_classes.txt"


DEFAULTS = AppDefaults()
