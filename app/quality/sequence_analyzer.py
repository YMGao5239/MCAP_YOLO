"""按 Topic 汇总 + 视频时序质量分析 (FR-QUALITY-003 / FR-SEQ-001).

Topic 汇总: total_frames, processed_frames, decode_failed_frames,
  bad_quality_frames, avg_quality_score, p50/p95_quality_score,
  quality_issue_counts{...}。独立统计,异常不影响其他 Topic。
时序分析: estimated_fps, frame_interval_ms(avg/p95), timestamp_jump_count,
  long_gap_count(帧间隔/时间戳跳变)。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TopicQualitySummary:
    topic: str
    total_frames: int
    processed_frames: int
    decode_failed_frames: int
    bad_quality_frames: int
    avg_quality_score: float | None
    p50_quality_score: float | None
    p95_quality_score: float | None
    quality_issue_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceStats:
    estimated_fps: float
    frame_interval_ms_avg: float
    frame_interval_ms_p95: float
    timestamp_jump_count: int
    long_gap_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize_by_topic(records: Iterable[dict[str, Any]]) -> dict[str, TopicQualitySummary]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("topic", "<unknown>"))].append(record)

    summaries: dict[str, TopicQualitySummary] = {}
    for topic, items in grouped.items():
        scores = [
            float(item["quality_score"])
            for item in items
            if item.get("decoded", True) and item.get("quality_score") is not None
        ]
        issue_counts: Counter[str] = Counter()
        bad_quality_frames = 0
        decode_failed_frames = 0
        for item in items:
            if not item.get("decoded", True):
                decode_failed_frames += 1
            tags = list(item.get("quality_tags") or [])
            if "bad_quality" in tags:
                bad_quality_frames += 1
            for tag in tags:
                if tag not in {"bad_quality", "decode_failed"}:
                    issue_counts[tag] += 1

        summaries[topic] = TopicQualitySummary(
            topic=topic,
            total_frames=len(items),
            processed_frames=len(scores),
            decode_failed_frames=decode_failed_frames,
            bad_quality_frames=bad_quality_frames,
            avg_quality_score=round(mean(scores), 4) if scores else None,
            p50_quality_score=_percentile(scores, 50),
            p95_quality_score=_percentile(scores, 95),
            quality_issue_counts=dict(issue_counts),
        )
    return summaries


def analyze_sequence(
    timestamps_ns: Iterable[int],
    expected_interval_ms: float | None = None,
    long_gap_factor: float = 3.0,
) -> SequenceStats:
    timestamps = [int(ts) for ts in timestamps_ns]
    if len(timestamps) < 2:
        return SequenceStats(0.0, 0.0, 0.0, 0, 0)

    deltas_ms = [(b - a) / 1_000_000.0 for a, b in zip(timestamps, timestamps[1:])]
    positive = [delta for delta in deltas_ms if delta > 0]
    timestamp_jump_count = sum(1 for delta in deltas_ms if delta <= 0)
    if not positive:
        return SequenceStats(0.0, 0.0, 0.0, timestamp_jump_count, 0)

    avg_interval = float(mean(positive))
    p95_interval = float(np.percentile(positive, 95))
    reference = expected_interval_ms if expected_interval_ms is not None else float(np.median(positive))
    long_gap_count = sum(1 for delta in positive if delta > reference * long_gap_factor)
    estimated_fps = 1000.0 / reference if reference > 0 else 0.0
    return SequenceStats(
        estimated_fps=round(estimated_fps, 4),
        frame_interval_ms_avg=round(avg_interval, 4),
        frame_interval_ms_p95=round(p95_interval, 4),
        timestamp_jump_count=timestamp_jump_count,
        long_gap_count=long_gap_count,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, percentile)), 4)
