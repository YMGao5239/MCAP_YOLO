"""健康检查接口: GET /health.

返回: {status, model_loaded, mcap_reader_available, yolo_backend}
"""
from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
from fastapi import APIRouter

from app.mcap_io.reader import McapReader


router = APIRouter()


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": Path("models/yolov8n.onnx").exists(),
        "mcap_reader_available": McapReader is not None,
        "yolo_backend": "onnxruntime",
        "onnxruntime_version": ort.__version__,
    }
