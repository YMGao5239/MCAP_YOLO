"""CLI: MCAP + YOLO 推理 (FR-CLI-002,推荐启动目标).

用法:
  python scripts/run_mcap_yolo_inference.py \
    --mcap ./test_data/sample.mcap \
    --auto-detect-topics true \
    --model ./models/yolov8n.onnx \
    --labels ./models/coco_classes.txt \
    --target-classes person,car,truck,bus \
    --sample-every-n 5 \
    --quality-threshold 0.6 \
    --conf-threshold 0.25 \
    --nms-threshold 0.45 \
    --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.jobs.worker import YoloInferenceParams, run_yolo_inference
from app.yolo.labels import parse_target_classes


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
    parser = argparse.ArgumentParser(description="Run MCAP quality gate and YOLO ONNX inference.")
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--auto-detect-topics", type=parse_bool, default=True)
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.onnx"))
    parser.add_argument("--labels", type=Path, default=Path("models/coco_classes.txt"))
    parser.add_argument("--target-classes", default=None)
    parser.add_argument("--sample-every-n", type=int, default=1)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--quality-threshold", type=float, default=0.6)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--infer-low-quality", type=parse_bool, default=False)
    parser.add_argument("--max-bad-samples", type=int, default=20)
    parser.add_argument("--max-detection-samples", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_yolo_inference(
        YoloInferenceParams(
            mcap=args.mcap,
            output_dir=args.output_dir,
            auto_detect_topics=args.auto_detect_topics,
            topics=args.topics,
            sample_every_n=args.sample_every_n,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            quality_threshold=args.quality_threshold,
            max_frames=args.max_frames,
            model_path=args.model,
            labels_path=args.labels,
            target_classes=parse_target_classes(args.target_classes),
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            infer_low_quality=args.infer_low_quality,
            max_bad_samples=args.max_bad_samples,
            max_detection_samples=args.max_detection_samples,
        )
    )
    print(f"status={result.status}")
    print(f"total_frames={result.total_frames} decoded_frames={result.decoded_frames}")
    print(f"inferred_frames={result.inferred_frames} skipped_low_quality_frames={result.skipped_low_quality_frames}")
    print(f"quality_report={result.quality_report_path}")
    print(f"yolo_predictions={result.yolo_predictions_path}")
    print(f"metrics={result.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
