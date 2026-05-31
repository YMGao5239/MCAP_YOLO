import json
from pathlib import Path

import numpy as np

from app.jobs.worker import YoloInferenceParams, run_yolo_inference
from app.report.html_report import render_html_report
from app.report.markdown_report import render_markdown_report
from app.yolo.onnx_runner import InferenceResult
from scripts.generate_test_mcap import generate_sample_mcap


class FakeRunner:
    def infer(self, tensor):
        raw = np.zeros((1, 84, 1), dtype=np.float32)
        raw[0, 0:4, 0] = [320, 320, 100, 100]
        raw[0, 4 + 2, 0] = 0.9
        return InferenceResult(outputs=[raw], inference_ms=1.0)


def test_yolo_pipeline_writes_reports_and_sample_indexes(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=10)
    output_dir = tmp_path / "outputs"

    run_yolo_inference(
        YoloInferenceParams(
            mcap=mcap,
            model_path=Path("models/yolov8n.onnx"),
            labels_path=Path("models/coco_classes.txt"),
            target_classes={"car"},
            output_dir=output_dir,
            max_bad_samples=4,
        ),
        runner=FakeRunner(),
    )

    expected = [
        "mcap_summary.json",
        "quality_report.json",
        "yolo_predictions.json",
        "metrics.json",
        "quality_report.html",
        "quality_report.md",
        "bad_samples/index.json",
        "detection_samples/index.json",
    ]
    for rel in expected:
        assert (output_dir / rel).exists(), rel

    bad_index = json.loads((output_dir / "bad_samples" / "index.json").read_text(encoding="utf-8"))
    detection_index = json.loads((output_dir / "detection_samples" / "index.json").read_text(encoding="utf-8"))
    assert len(bad_index["samples"]) <= 4
    assert len(bad_index["samples"]) > 0
    assert len(detection_index["samples"]) > 0
    assert (output_dir / bad_index["samples"][0]["image_path"]).exists()
    assert (output_dir / detection_index["samples"][0]["image_path"]).exists()


def test_markdown_and_html_render_basic_report_content():
    quality = {
        "topic_summaries": {
            "/camera": {
                "total_frames": 2,
                "processed_frames": 2,
                "bad_quality_frames": 1,
                "avg_quality_score": 0.7,
                "p50_quality_score": 0.7,
                "p95_quality_score": 0.9,
                "quality_issue_counts": {"blurry": 1},
            }
        }
    }
    predictions = {"inferred_frames": 1, "skipped_low_quality_frames": 1, "frames": []}
    metrics = {"total_frames": 2, "decoded_frames": 2}

    md = render_markdown_report(quality, predictions, metrics)
    html = render_html_report(quality, predictions, metrics, {"samples": []}, {"samples": []})

    assert "# MCAP Quality Report" in md
    assert "/camera" in md
    assert "<html" in html
    assert "MCAP Quality Report" in html
