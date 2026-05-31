import json
from pathlib import Path

from app.api.health import health
from app.api.jobs import get_job
from app.api.mcap import InspectRequest, QualityScanRequest, inspect_mcap, quality_scan
from app.jobs.worker import QualityScanParams, run_quality_scan
from scripts.generate_test_mcap import generate_sample_mcap


def test_quality_scan_metrics_include_avg_p95_and_max_frames(tmp_path, capsys):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=10)
    output_dir = tmp_path / "outputs"

    result = run_quality_scan(QualityScanParams(mcap=mcap, output_dir=output_dir, max_frames=5))

    captured = capsys.readouterr()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert result.total_frames == 5
    assert metrics["total_frames"] == 5
    assert metrics["processed_frames_per_sec"] > 0
    for key in [
        "avg_decode_ms",
        "p95_decode_ms",
        "avg_preprocess_ms",
        "p95_preprocess_ms",
        "avg_inference_ms",
        "p95_inference_ms",
        "avg_postprocess_ms",
        "p95_postprocess_ms",
    ]:
        assert key in metrics
    assert "[metrics]" in captured.out
    assert "processed_frames=5" in captured.out


def test_fastapi_health_inspect_quality_scan_and_job_status(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=2)

    health_response = health()
    assert health_response["mcap_reader_available"] is True
    assert health_response["yolo_backend"] == "onnxruntime"

    inspect_response = inspect_mcap(InspectRequest(mcap_path=str(mcap)))
    assert inspect_response["message_count"] == 4

    scan_response = quality_scan(QualityScanRequest(mcap_path=str(mcap), output_dir=str(tmp_path / "api_outputs"), max_frames=3))
    job_id = scan_response["job_id"]
    status = get_job(job_id)
    assert status["status"] == "finished"
    assert Path(status["result_path"]).exists()

    try:
        inspect_mcap(InspectRequest(mcap_path=str(tmp_path / "missing.mcap")))
    except Exception as exc:
        assert "error" in str(exc)
    else:
        raise AssertionError("missing MCAP should raise a handled API error")
