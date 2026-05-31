import importlib.util
from pathlib import Path

from rosbags.highlevel import AnyReader


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_test_mcap_writes_image_and_compressed_topics(tmp_path):
    script = load_script("generate_test_mcap.py")
    output = tmp_path / "sample.mcap"

    result = script.generate_sample_mcap(output, frame_count=6)

    assert result == output
    assert output.is_file()
    with AnyReader([output]) as reader:
        topics = {conn.topic: conn.msgtype for conn in reader.connections}

    assert topics["/camera/image"] == "sensor_msgs/msg/Image"
    assert topics["/camera/compressed"] == "sensor_msgs/msg/CompressedImage"


def test_download_yolo_model_exposes_onnx_verifier():
    script = load_script("download_yolo_model.py")

    assert callable(script.verify_onnx_model)
