import json
from pathlib import Path

import numpy as np

from app.jobs.manager import JobManager
from app.jobs.worker import YoloInferenceParams, run_yolo_inference
from app.yolo.onnx_runner import InferenceResult
from scripts.generate_test_mcap import generate_sample_mcap


class FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, tensor):
        self.calls += 1
        raw = np.zeros((1, 84, 1), dtype=np.float32)
        raw[0, 0:4, 0] = [320, 320, 100, 100]
        raw[0, 4 + 2, 0] = 0.9
        return InferenceResult(outputs=[raw], inference_ms=1.5)


def test_run_yolo_inference_applies_quality_gate_and_writes_outputs(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=10)
    output_dir = tmp_path / "outputs"
    runner = FakeRunner()

    result = run_yolo_inference(
        YoloInferenceParams(
            mcap=mcap,
            model_path=Path("models/yolov8n.onnx"),
            labels_path=Path("models/coco_classes.txt"),
            target_classes={"car"},
            output_dir=output_dir,
            quality_threshold=0.6,
            conf_threshold=0.25,
            nms_threshold=0.45,
        ),
        runner=runner,
    )

    assert result.status == "finished"
    assert result.total_frames == 20
    assert result.skipped_low_quality_frames > 0
    assert runner.calls == result.inferred_frames
    assert (output_dir / "quality_report.json").exists()
    assert (output_dir / "yolo_predictions.json").exists()
    assert (output_dir / "metrics.json").exists()

    predictions = json.loads((output_dir / "yolo_predictions.json").read_text(encoding="utf-8"))
    assert predictions["skipped_low_quality_frames"] == result.skipped_low_quality_frames
    assert predictions["inferred_frames"] == result.inferred_frames
    assert predictions["frames"][0]["detections"][0]["label"] == "car"


def test_run_yolo_inference_can_infer_low_quality_when_enabled(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=4)
    runner = FakeRunner()

    result = run_yolo_inference(
        YoloInferenceParams(
            mcap=mcap,
            model_path=Path("models/yolov8n.onnx"),
            labels_path=Path("models/coco_classes.txt"),
            target_classes={"car"},
            output_dir=tmp_path / "outputs",
            infer_low_quality=True,
        ),
        runner=runner,
    )

    assert result.total_frames == 8
    assert result.skipped_low_quality_frames == 0
    assert result.inferred_frames == 8


def test_job_manager_tracks_status_progress_and_failures(tmp_path):
    manager = JobManager()
    job = manager.create_job("quality_scan")

    manager.update(job.job_id, progress=0.5)
    running = manager.get(job.job_id)
    assert running.status == "running"
    assert running.progress == 0.5

    manager.finish(job.job_id, result_path=tmp_path / "result.json")
    finished = manager.get(job.job_id)
    assert finished.status == "finished"
    assert finished.progress == 1.0
    assert finished.result_path == str(tmp_path / "result.json")

    failed = manager.create_job("yolo")
    manager.fail(failed.job_id, RuntimeError("boom"))
    assert manager.get(failed.job_id).status == "failed"
    assert "boom" in manager.get(failed.job_id).error
