"""HTML 报告 (FR-REPORT-002,jinja2 模板渲染).

quality_report.html:含质量汇总图表、坏样本缩略图、检测样本。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template


TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MCAP Quality Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #17202a; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    th, td { border: 1px solid #d7dbdd; padding: 8px; text-align: left; }
    th { background: #f2f4f4; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
    .sample { border: 1px solid #d7dbdd; padding: 8px; border-radius: 4px; }
    .sample img { width: 100%; height: auto; display: block; }
    code { background: #f4f6f7; padding: 2px 4px; }
  </style>
</head>
<body>
  <h1>MCAP Quality Report</h1>
  <h2>Summary</h2>
  <ul>
    <li>Total frames: {{ metrics.get("total_frames", 0) }}</li>
    <li>Decoded frames: {{ metrics.get("decoded_frames", 0) }}</li>
    <li>Bad quality frames: {{ metrics.get("bad_quality_frames", 0) }}</li>
    <li>Inferred frames: {{ predictions.get("inferred_frames", 0) }}</li>
    <li>Skipped low quality frames: {{ predictions.get("skipped_low_quality_frames", 0) }}</li>
  </ul>
  <h2>Topic Quality</h2>
  <table>
    <tr><th>Topic</th><th>Total</th><th>Processed</th><th>Bad</th><th>Avg</th><th>P50</th><th>P95</th><th>Issues</th></tr>
    {% for topic, summary in quality.get("topic_summaries", {}).items() %}
    <tr>
      <td><code>{{ topic }}</code></td>
      <td>{{ summary.get("total_frames") }}</td>
      <td>{{ summary.get("processed_frames") }}</td>
      <td>{{ summary.get("bad_quality_frames") }}</td>
      <td>{{ summary.get("avg_quality_score") }}</td>
      <td>{{ summary.get("p50_quality_score") }}</td>
      <td>{{ summary.get("p95_quality_score") }}</td>
      <td>{{ summary.get("quality_issue_counts", {}) }}</td>
    </tr>
    {% endfor %}
  </table>
  <h2>Bad Samples</h2>
  <div class="grid">
    {% for sample in bad_samples.get("samples", []) %}
    <div class="sample"><img src="{{ sample.image_path }}"><p>{{ sample.topic }} #{{ sample.frame_seq }} score={{ sample.quality_score }} {{ sample.quality_tags }}</p></div>
    {% endfor %}
  </div>
  <h2>Detection Samples</h2>
  <div class="grid">
    {% for sample in detection_samples.get("samples", []) %}
    <div class="sample"><img src="{{ sample.image_path }}"><p>{{ sample.topic }} #{{ sample.frame_seq }} detections={{ sample.detections|length }}</p></div>
    {% endfor %}
  </div>
</body>
</html>"""
)


def render_html_report(
    quality_report: dict[str, Any],
    yolo_predictions: dict[str, Any] | None,
    metrics: dict[str, Any],
    bad_samples: dict[str, Any],
    detection_samples: dict[str, Any],
) -> str:
    return TEMPLATE.render(
        quality=quality_report,
        predictions=yolo_predictions or {},
        metrics=metrics,
        bad_samples=bad_samples,
        detection_samples=detection_samples,
    )


def write_html_report(
    path: str | Path,
    quality_report: dict[str, Any],
    yolo_predictions: dict[str, Any] | None,
    metrics: dict[str, Any],
    bad_samples: dict[str, Any],
    detection_samples: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html_report(quality_report, yolo_predictions, metrics, bad_samples, detection_samples), encoding="utf-8")
    return output
