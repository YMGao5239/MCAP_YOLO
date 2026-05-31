# FastAPI 接口说明

| 接口 | 方法 | 说明 |
|---|---|---|
| /health | GET | 健康检查 |
| /mcap/inspect | POST | MCAP 信息解析 |
| /mcap/quality_scan | POST | 质量扫描任务 |
| /mcap/yolo_infer | POST | YOLO 推理任务 |
| /jobs/{job_id} | GET | 任务状态 |
| /mcap/frame | GET | 单帧预览 |
| /mcap/frame_yolo | GET | 单帧 YOLO 预览 |
