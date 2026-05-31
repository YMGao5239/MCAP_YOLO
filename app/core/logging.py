"""结构化日志.

日志字段:
  mcap_file, topic, message_type, frame_seq, decode_ms, preprocess_ms,
  inference_ms, postprocess_ms, quality_score, quality_tags, object_count,
  target_object_count, processed_frames, decode_failed_frames,
  skipped_low_quality_frames
示例: [metrics] file=sample.mcap topic=... decode_ms=...
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

import numpy as np


METRIC_FIELDS = [
    "mcap_file",
    "topic",
    "message_type",
    "frame_seq",
    "decode_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "quality_score",
    "quality_tags",
    "object_count",
    "target_object_count",
    "processed_frames",
    "decode_failed_frames",
    "skipped_low_quality_frames",
    "bad_quality_frames",
]


@dataclass
class StageTimer:
    started_at: float = field(default_factory=time.perf_counter)

    def ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0


def summarize_ms(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return round(float(mean(values)), 4), round(float(np.percentile(values, 95)), 4)


def log_metrics_line(**fields: Any) -> None:
    payload = {name: fields.get(name, "-") for name in METRIC_FIELDS}
    text = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[metrics] {text}")
