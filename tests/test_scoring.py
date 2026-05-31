"""评分规则单测: 各惩罚项叠加、阈值标记 bad_quality、score 落在 [0,1]."""
import cv2
import numpy as np

from app.quality.analyzer import QualityAnalyzer
from app.quality.scoring import QualityScorer


def textured_frame(width: int = 640, height: int = 480) -> np.ndarray:
    image = np.full((height, width, 3), 95, dtype=np.uint8)
    for y in range(0, height, 16):
        cv2.line(image, (0, y), (width - 1, y), (40 + y % 200, 120, 220 - y % 120), 2)
    cv2.putText(image, "quality", (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    return image


def test_good_frame_scores_high_and_has_explainable_zeroish_penalties():
    analyzer = QualityAnalyzer()
    scorer = QualityScorer(quality_threshold=0.6)

    result = scorer.score(analyzer.analyze(textured_frame()))

    assert 0.8 <= result.score <= 1.0
    assert result.quality_tags == []
    assert set(result.penalties) == {
        "blur_penalty",
        "exposure_penalty",
        "contrast_penalty",
        "resolution_penalty",
        "corruption_penalty",
        "timestamp_penalty",
    }


def test_bad_frame_penalties_accumulate_and_mark_bad_quality():
    analyzer = QualityAnalyzer()
    scorer = QualityScorer(quality_threshold=0.6)
    bad = np.zeros((90, 120, 3), dtype=np.uint8)

    result = scorer.score(analyzer.analyze(bad))

    assert 0.0 <= result.score <= 1.0
    assert result.score < 0.6
    assert "bad_quality" in result.quality_tags
    assert result.penalties["exposure_penalty"] > 0.0
    assert result.penalties["resolution_penalty"] > 0.0
    assert result.penalties["corruption_penalty"] > 0.0


def test_score_is_clamped_to_zero_for_corrupted_frames():
    analyzer = QualityAnalyzer()
    scorer = QualityScorer(quality_threshold=0.6)

    result = scorer.score(analyzer.analyze(np.asarray([], dtype=np.uint8)))

    assert result.score == 0.0
    assert "bad_quality" in result.quality_tags
    assert "corrupted" in result.quality_tags
