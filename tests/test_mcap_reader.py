import json
import subprocess
import sys
from pathlib import Path

from app.mcap_io.reader import McapReader, discover_mcap_paths
from app.mcap_io.topic_scanner import TopicScanner, is_image_message_type
from scripts.generate_test_mcap import generate_sample_mcap


def test_reader_inspects_single_mcap_metadata(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=4, fps=2)

    summary = McapReader().inspect_file(mcap)

    assert summary.path == str(mcap)
    assert summary.message_count == 8
    assert summary.duration_sec > 0
    assert {topic.name: topic.message_count for topic in summary.topics} == {
        "/camera/image": 4,
        "/camera/compressed": 4,
    }


def test_topic_scanner_marks_ros_image_topics(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=2)
    summary = McapReader().inspect_file(mcap)

    detected = TopicScanner(auto_detect=True).scan(summary.topics)

    assert {topic.topic: topic.is_image_topic for topic in detected} == {
        "/camera/image": True,
        "/camera/compressed": True,
    }
    assert is_image_message_type("std_msgs/msg/String") is False


def test_reader_batches_directory_and_collects_failures(tmp_path):
    good = generate_sample_mcap(tmp_path / "nested" / "good.mcap", frame_count=1)
    bad = tmp_path / "broken.mcap"
    bad.write_text("not an mcap", encoding="utf-8")

    assert discover_mcap_paths(mcap_dir=tmp_path) == [bad, good]

    result = McapReader().inspect_many(discover_mcap_paths(mcap_dir=tmp_path))

    assert [item.path for item in result.files] == [str(good)]
    assert result.total_message_count == 2
    assert len(result.failed_files) == 1
    assert result.failed_files[0].path == str(bad)


def test_iter_messages_supports_sampling_and_time_window(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=6, fps=10)
    reader = McapReader()
    summary = reader.inspect_file(mcap)
    start_sec = 0.1
    end_sec = 0.5

    messages = list(
        reader.iter_messages(
            mcap,
            topics=["/camera/image"],
            sample_every_n=2,
            start_sec=start_sec,
            end_sec=end_sec,
        )
    )

    assert len(messages) == 2
    assert all(message.topic == "/camera/image" for message in messages)
    assert all(summary.start_time_ns + int(start_sec * 1e9) <= message.log_time_ns < summary.start_time_ns + int(end_sec * 1e9) for message in messages)


def test_quality_scan_cli_writes_mcap_summary(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=3)
    output_dir = tmp_path / "outputs"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_mcap_quality_scan.py",
            "--mcap",
            str(mcap),
            "--auto-detect-topics",
            "true",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "/camera/image" in completed.stdout
    summary = json.loads((output_dir / "mcap_summary.json").read_text(encoding="utf-8"))
    assert summary["total_message_count"] == 6
    assert {topic["topic"] for topic in summary["detected_topics"]} == {
        "/camera/image",
        "/camera/compressed",
    }
