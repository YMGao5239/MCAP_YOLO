"""Markdown 报告 (FR-REPORT-002): quality_report.md."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def render_markdown_report(quality_report: dict[str, Any], yolo_predictions: dict[str, Any] | None, metrics: dict[str, Any]) -> str:
    lines = [
        "# MCAP Quality Report",
        "",
        "## Summary",
        "",
        f"- Total frames: {metrics.get('total_frames', 0)}",
        f"- Decoded frames: {metrics.get('decoded_frames', 0)}",
        f"- Bad quality frames: {metrics.get('bad_quality_frames', 0)}",
        f"- Inferred frames: {(yolo_predictions or {}).get('inferred_frames', metrics.get('inferred_frames', 0))}",
        f"- Skipped low quality frames: {(yolo_predictions or {}).get('skipped_low_quality_frames', metrics.get('skipped_low_quality_frames', 0))}",
        "",
        "## Topic Quality",
        "",
        "| Topic | Total | Processed | Bad | Avg | P50 | P95 | Issues |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for topic, summary in sorted((quality_report.get("topic_summaries") or {}).items()):
        issues = ", ".join(f"{k}:{v}" for k, v in (summary.get("quality_issue_counts") or {}).items()) or "-"
        lines.append(
            f"| {topic} | {summary.get('total_frames', 0)} | {summary.get('processed_frames', 0)} | "
            f"{summary.get('bad_quality_frames', 0)} | {summary.get('avg_quality_score')} | "
            f"{summary.get('p50_quality_score')} | {summary.get('p95_quality_score')} | {issues} |"
        )
    lines.extend(["", "## Sequence", ""])
    for topic, seq in sorted((quality_report.get("sequence") or {}).items()):
        lines.append(
            f"- `{topic}`: fps={seq.get('estimated_fps')}, avg_interval_ms={seq.get('frame_interval_ms_avg')}, "
            f"p95_interval_ms={seq.get('frame_interval_ms_p95')}, jumps={seq.get('timestamp_jump_count')}, gaps={seq.get('long_gap_count')}"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(path: str | Path, quality_report: dict[str, Any], yolo_predictions: dict[str, Any] | None, metrics: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(quality_report, yolo_predictions, metrics), encoding="utf-8")
    return output
