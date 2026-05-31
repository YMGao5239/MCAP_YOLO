"""CLI: MCAP 质量扫描 (FR-CLI-001).

用法:
  python scripts/run_mcap_quality_scan.py \
    --mcap ./test_data/sample.mcap \
    --auto-detect-topics true \
    --quality-threshold 0.6 \
    --output-dir ./outputs
支持 --mcap-dir 目录批处理。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcap_io.reader import McapReader, discover_mcap_paths
from app.mcap_io.topic_scanner import TopicScanner


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MCAP files and detect ROS image topics.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mcap", type=Path, help="Single MCAP file path.")
    source.add_argument("--mcap-dir", type=Path, help="Directory to recursively scan for .mcap files.")
    parser.add_argument("--topics", nargs="*", default=None, help="Manually selected topic names.")
    parser.add_argument("--auto-detect-topics", type=parse_bool, default=True)
    parser.add_argument("--sample-every-n", type=int, default=1)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    paths = discover_mcap_paths(args.mcap, args.mcap_dir)
    reader = McapReader()
    batch = reader.inspect_many(paths)

    detected_topics = []
    for file_summary in batch.files:
        scanner = TopicScanner(auto_detect=args.auto_detect_topics, topics=args.topics)
        for topic in scanner.scan(file_summary.topics):
            item = topic.to_dict()
            item["mcap_file"] = file_summary.path
            detected_topics.append(item)

    return {
        "input_paths": [str(path) for path in paths],
        "files": [file.to_dict() for file in batch.files],
        "failed_files": [failure.to_dict() for failure in batch.failed_files],
        "total_message_count": batch.total_message_count,
        "detected_topics": detected_topics,
        "sampling": {
            "sample_every_n": max(1, args.sample_every_n),
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
        },
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mcap_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    output_path = write_summary(summary, args.output_dir)

    print(f"MCAP files: {len(summary['files'])}, failed: {len(summary['failed_files'])}")
    for topic in summary["detected_topics"]:
        marker = "image" if topic["is_image_topic"] else "other"
        print(f"- {topic['topic']} [{marker}] {topic['message_type']} frames={topic['message_count']}")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
