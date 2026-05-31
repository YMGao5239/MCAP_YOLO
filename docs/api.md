# FastAPI 接口说明

| 接口 | 方法 | 说明 |
|---|---|---|
| /health | GET | FR-API-002 健康检查 |
| /mcap/inspect | POST | FR-API-003 MCAP 信息解析 |
| /mcap/quality_scan | POST | FR-API-004 质量扫描任务 |
| /mcap/yolo_infer | POST | FR-API-005 YOLO 推理任务 |
| /jobs/{job_id} | GET | FR-API-006 任务状态 |
| /mcap/frame | GET | FR-API-007 单帧预览 (加分) |
| /mcap/frame_yolo | GET | FR-API-008 单帧 YOLO 预览 (加分) |

TODO: 请求/响应字段详述。
