#!/usr/bin/env bash
# 冒烟测试:端到端跑通 MCAP → 质量 → YOLO → 报告
set -euo pipefail

python scripts/run_mcap_yolo_inference.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --model ./models/yolov8n.onnx \
  --labels ./models/coco_classes.txt \
  --target-classes person,car,truck,bus \
  --sample-every-n 5 \
  --max-frames 12 \
  --quality-threshold 0.6 \
  --conf-threshold 0.25 \
  --nms-threshold 0.45 \
  --max-bad-samples 6 \
  --max-detection-samples 6 \
  --output-dir ./outputs

test -f ./outputs/quality_report.json
test -f ./outputs/yolo_predictions.json
test -f ./outputs/metrics.json
test -f ./outputs/quality_report.html

echo "smoke test done. see ./outputs"
