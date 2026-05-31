# 架构说明

MCAP File/Dir → MCAP Reader → Topic Scanner → ROS Image Decoder →
Per-Topic Frame Iterator → Image Quality Analyzer → Qualified Frames →
YOLO Detection + Classification → Detection Results + Latency →
JSON / HTML / Markdown Report

TODO: 各模块职责、依赖关系、数据结构图。
