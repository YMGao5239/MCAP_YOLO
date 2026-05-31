"""JSON 报告 (FR-REPORT-001/002/003 + metrics.json).

 - mcap_summary.json   : files[], duration, topics, message_count...
 - quality_report.json : 按 Topic 质量汇总、评分、问题计数、Top-N 坏帧
 - yolo_predictions.json: 每帧检测结果 (bbox/class_id/label/confidence + 耗时)
 - metrics.json        : 平均/ P95 各阶段耗时、processed_frames 等
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def build_mcap_summary(file_summary: Any, detected_topics: list[Any]) -> dict[str, Any]:
    return {
        "files": [file_summary.to_dict() if hasattr(file_summary, "to_dict") else file_summary],
        "failed_files": [],
        "total_message_count": getattr(file_summary, "message_count", 0),
        "detected_topics": [
            topic.to_dict() if hasattr(topic, "to_dict") else dict(topic)
            for topic in detected_topics
        ],
    }
