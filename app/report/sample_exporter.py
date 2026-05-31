"""样本导出.

 - bad_samples/      : 低质量帧图像 + 对应 index json (quality_score, tags)
 - detection_samples/: 带检测框可视化图像 + index json (detections)
 - 支持 --max-bad-samples 上限;按 Topic 命名,异常不影响其他。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2

from app.yolo.visualizer import draw_detections


def export_bad_samples(decoded_items: list[dict[str, Any]], output_dir: str | Path, max_bad_samples: int = 20) -> dict[str, Any]:
    root = Path(output_dir) / "bad_samples"
    root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for item in decoded_items:
        score = item["score"]
        if "bad_quality" not in score.get("quality_tags", []):
            continue
        frame = item["frame"]
        rel = Path("bad_samples") / f"{_safe(frame.topic)}_{frame.frame_seq:05d}.jpg"
        cv2.imwrite(str(Path(output_dir) / rel), frame.image)
        samples.append(
            {
                "topic": frame.topic,
                "frame_seq": frame.frame_seq,
                "image_path": rel.as_posix(),
                "quality_score": score.get("score"),
                "quality_tags": score.get("quality_tags", []),
                "penalties": score.get("penalties", {}),
            }
        )
        if len(samples) >= max_bad_samples:
            break
    return _write_index(root / "index.json", samples)


def export_detection_samples(prediction_items: list[dict[str, Any]], output_dir: str | Path, max_samples: int = 20) -> dict[str, Any]:
    root = Path(output_dir) / "detection_samples"
    root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for item in prediction_items:
        detections = item.get("detections", [])
        if not detections:
            continue
        frame = item["frame"]
        rel = Path("detection_samples") / f"{_safe(frame.topic)}_{frame.frame_seq:05d}.jpg"
        cv2.imwrite(str(Path(output_dir) / rel), draw_detections(frame.image, detections))
        samples.append(
            {
                "topic": frame.topic,
                "frame_seq": frame.frame_seq,
                "image_path": rel.as_posix(),
                "detections": [detection.to_dict() for detection in detections],
            }
        )
        if len(samples) >= max_samples:
            break
    return _write_index(root / "index.json", samples)


def _write_index(path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {"samples": samples}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _safe(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", topic).strip("_") or "topic"
