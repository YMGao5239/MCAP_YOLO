"""时序分析单测: FPS 估计、帧间隔、时间戳跳变计数."""
from app.quality.sequence_analyzer import analyze_sequence, summarize_by_topic


def test_analyze_sequence_estimates_fps_and_gap_counts():
    timestamps = [
        1_000_000_000,
        1_100_000_000,
        1_200_000_000,
        1_900_000_000,
        1_850_000_000,
    ]

    result = analyze_sequence(timestamps, expected_interval_ms=100.0, long_gap_factor=3.0)

    assert round(result.estimated_fps, 2) == 10.0
    assert result.frame_interval_ms_avg > 0
    assert result.frame_interval_ms_p95 >= result.frame_interval_ms_avg
    assert result.long_gap_count == 1
    assert result.timestamp_jump_count == 1


def test_summarize_by_topic_keeps_topics_independent():
    records = [
        {"topic": "/cam/a", "decoded": True, "quality_score": 0.9, "quality_tags": []},
        {"topic": "/cam/a", "decoded": True, "quality_score": 0.4, "quality_tags": ["bad_quality", "blurry"]},
        {"topic": "/cam/a", "decoded": False, "quality_score": None, "quality_tags": ["decode_failed"]},
        {"topic": "/cam/b", "decoded": True, "quality_score": 0.8, "quality_tags": []},
    ]

    summaries = summarize_by_topic(records)

    assert summaries["/cam/a"].total_frames == 3
    assert summaries["/cam/a"].processed_frames == 2
    assert summaries["/cam/a"].decode_failed_frames == 1
    assert summaries["/cam/a"].bad_quality_frames == 1
    assert summaries["/cam/a"].quality_issue_counts["blurry"] == 1
    assert summaries["/cam/b"].total_frames == 1
    assert summaries["/cam/b"].decode_failed_frames == 0
    assert summaries["/cam/b"].avg_quality_score == 0.8
