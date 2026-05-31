from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_requirements_and_starts_uvicorn():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:" in text
    assert "COPY requirements.txt" in text
    assert "pip install --no-cache-dir -r requirements.txt" in text
    assert "uvicorn" in text
    assert "app.main:app" in text


def test_compose_mounts_runtime_dirs_and_has_smoke_profile():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "./test_data:/workspace/test_data" in text
    assert "./models:/workspace/models" in text
    assert "./outputs:/workspace/outputs" in text
    assert "profiles:" in text
    assert "test" in text
    assert "scripts/run_smoke_test.sh" in text


def test_smoke_script_runs_end_to_end_with_bounded_frames():
    text = (ROOT / "scripts" / "run_smoke_test.sh").read_text(encoding="utf-8")

    assert "scripts/run_mcap_yolo_inference.py" in text
    assert "--max-frames" in text
    assert "--output-dir" in text
