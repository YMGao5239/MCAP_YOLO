"""质量评分 (FR-QUALITY-002).

quality_score = 1.0
  - blur_penalty - exposure_penalty - contrast_penalty
  - resolution_penalty - corruption_penalty - timestamp_penalty
范围 [0,1];阈值可配置 (--quality-threshold 0.6);
低于阈值标记 quality_tags=["bad_quality"];输出每个扣分项 penalties。
README 中必须说明评分规则 (禁止事项 8: 评分规则不可解释)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from app.quality.analyzer import FrameQuality


@dataclass(frozen=True)
class QualityScore:
    score: float
    quality_tags: list[str]
    penalties: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QualityScorer:
    def __init__(self, quality_threshold: float = 0.6) -> None:
        self.quality_threshold = quality_threshold

    def score(self, quality: FrameQuality, timestamp_penalty: float = 0.0) -> QualityScore:
        penalties = {
            "blur_penalty": self._blur_penalty(quality),
            "exposure_penalty": self._exposure_penalty(quality),
            "contrast_penalty": self._contrast_penalty(quality),
            "resolution_penalty": self._resolution_penalty(quality),
            "corruption_penalty": 0.45 if quality.is_corrupted else 0.0,
            "timestamp_penalty": _clamp(timestamp_penalty, 0.0, 0.2),
        }
        raw_score = 1.0 - sum(penalties.values())
        score = round(_clamp(raw_score, 0.0, 1.0), 4)
        tags = self._tags(quality)
        if score < self.quality_threshold:
            tags.append("bad_quality")
        return QualityScore(score=score, quality_tags=tags, penalties={k: round(v, 4) for k, v in penalties.items()})

    def _blur_penalty(self, quality: FrameQuality) -> float:
        if quality.is_blurry:
            return 0.2
        return 0.0

    def _exposure_penalty(self, quality: FrameQuality) -> float:
        if quality.is_too_dark or quality.is_too_bright:
            return 0.2
        return 0.0

    def _contrast_penalty(self, quality: FrameQuality) -> float:
        if quality.is_low_contrast:
            return 0.15
        return 0.0

    def _resolution_penalty(self, quality: FrameQuality) -> float:
        if not quality.is_low_resolution:
            return 0.0
        pixels = quality.metrics.width * quality.metrics.height
        if pixels <= 0:
            return 0.25
        return 0.25 if pixels < 320 * 240 else 0.1

    def _tags(self, quality: FrameQuality) -> list[str]:
        tags: list[str] = []
        if quality.is_corrupted:
            tags.append("corrupted")
        if quality.is_too_dark:
            tags.append("too_dark")
        if quality.is_too_bright:
            tags.append("too_bright")
        if quality.is_blurry:
            tags.append("blurry")
        if quality.is_low_contrast:
            tags.append("low_contrast")
        if quality.is_low_resolution:
            tags.append("low_resolution")
        return tags


def score_frame(quality: FrameQuality, quality_threshold: float = 0.6, timestamp_penalty: float = 0.0) -> QualityScore:
    return QualityScorer(quality_threshold=quality_threshold).score(quality, timestamp_penalty=timestamp_penalty)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
