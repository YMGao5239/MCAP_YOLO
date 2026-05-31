"""MCAP 相关接口.

FR-API-003  POST /mcap/inspect        MCAP 信息解析 (上传或容器内路径)
FR-API-004  POST /mcap/quality_scan   MCAP 质量扫描任务
FR-API-007  GET  /mcap/frame          单帧预览 (base64, 加分)
FR-API-008  GET  /mcap/frame_yolo     单帧 YOLO 预览 (加分)
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.jobs import job_manager
from app.jobs.worker import QualityScanParams, run_quality_scan
from app.mcap_io.reader import McapReader
from app.mcap_io.ros_image_decoder import RosImageDecoder
from app.mcap_io.topic_scanner import TopicScanner


router = APIRouter()


class InspectRequest(BaseModel):
    mcap_path: str


class QualityScanRequest(BaseModel):
    mcap_path: str
    output_dir: str = "outputs"
    auto_detect_topics: bool = True
    topics: list[str] | None = None
    sample_every_n: int = 1
    start_sec: float | None = None
    end_sec: float | None = None
    quality_threshold: float = 0.6
    max_frames: int | None = None


@router.post("/inspect")
def inspect_mcap(request: InspectRequest) -> dict[str, object]:
    path = Path(request.mcap_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail={"error": f"MCAP path does not exist: {path}"})
    try:
        summary = McapReader().inspect_file(path)
        detected = TopicScanner(auto_detect=True).scan(summary.topics)
        payload = summary.to_dict()
        payload["detected_topics"] = [topic.to_dict() for topic in detected]
        return payload
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc


@router.post("/quality_scan")
def quality_scan(request: QualityScanRequest) -> dict[str, object]:
    job = job_manager.create_job("quality_scan")
    try:
        result = run_quality_scan(
            QualityScanParams(
                mcap=request.mcap_path,
                output_dir=request.output_dir,
                auto_detect_topics=request.auto_detect_topics,
                topics=request.topics,
                sample_every_n=request.sample_every_n,
                start_sec=request.start_sec,
                end_sec=request.end_sec,
                quality_threshold=request.quality_threshold,
                max_frames=request.max_frames,
            )
        )
        job_manager.finish(job.job_id, result_path=result.quality_report_path, report_path=result.quality_report_path)
    except Exception as exc:
        job_manager.fail(job.job_id, exc)
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    return job_manager.get(job.job_id).to_dict()


@router.get("/frame")
def frame_preview(mcap_path: str, topic: str | None = None, frame_seq: int = 0) -> dict[str, object]:
    if not Path(mcap_path).exists():
        raise HTTPException(status_code=400, detail={"error": f"MCAP path does not exist: {mcap_path}"})
    decoder = RosImageDecoder()
    reader = McapReader()
    topics = [topic] if topic else None
    for message in reader.iter_messages(mcap_path, topics=topics):
        frame = decoder.decode(message)
        if frame is None:
            continue
        if frame.frame_seq != frame_seq:
            continue
        ok, encoded = cv2.imencode(".jpg", frame.image)
        if not ok:
            raise HTTPException(status_code=500, detail={"error": "failed to encode preview"})
        return {
            "topic": frame.topic,
            "frame_seq": frame.frame_seq,
            "width": frame.width,
            "height": frame.height,
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    raise HTTPException(status_code=404, detail={"error": "frame not found"})
