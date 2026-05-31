# 架构说明

MCAP File/Dir → MCAP Reader → Topic Scanner → ROS Image Decoder →
Per-Topic Frame Iterator → Image Quality Analyzer → Qualified Frames →
YOLO Detection + Classification → Detection Results + Latency →
JSON / HTML / Markdown Report
