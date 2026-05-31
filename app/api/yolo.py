"""FR-API-005 MCAP YOLO 推理任务: POST /mcap/yolo_infer.

请求字段: mcap_path, topics, model_path, labels_path, target_classes,
          sample_every_n, quality_threshold, nms_threshold, infer_low_quality
返回: {job_id, status}
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.jobs import job_manager
from app.jobs.worker import YoloInferenceParams, run_yolo_inference
from app.mcap_io.reader import McapReader
from app.mcap_io.ros_image_decoder import RosImageDecoder
from app.yolo.labels import load_labels, parse_target_classes
from app.yolo.onnx_runner import YoloOnnxRunner
from app.yolo.postprocess import postprocess
from app.yolo.preprocess import preprocess
from app.yolo.visualizer import draw_detections


router = APIRouter()


class YoloInferRequest(BaseModel):
    mcap_path: str
    output_dir: str = "outputs"
    topics: list[str] | None = None
    auto_detect_topics: bool = True
    model_path: str = "models/yolov8n.onnx"
    labels_path: str = "models/coco_classes.txt"
    target_classes: str | list[str] | None = None
    sample_every_n: int = 1
    quality_threshold: float = 0.6
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    infer_low_quality: bool = False
    max_frames: int | None = None


@router.post("/yolo_infer")
def yolo_infer(request: YoloInferRequest) -> dict[str, object]:
    job = job_manager.create_job("yolo_infer")
    try:
        result = run_yolo_inference(
            YoloInferenceParams(
                mcap=request.mcap_path,
                output_dir=request.output_dir,
                auto_detect_topics=request.auto_detect_topics,
                topics=request.topics,
                sample_every_n=request.sample_every_n,
                quality_threshold=request.quality_threshold,
                max_frames=request.max_frames,
                model_path=request.model_path,
                labels_path=request.labels_path,
                target_classes=parse_target_classes(request.target_classes),
                conf_threshold=request.conf_threshold,
                nms_threshold=request.nms_threshold,
                infer_low_quality=request.infer_low_quality,
            )
        )
        job_manager.finish(job.job_id, result_path=result.yolo_predictions_path, report_path=result.quality_report_path)
    except Exception as exc:
        job_manager.fail(job.job_id, exc)
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    return job_manager.get(job.job_id).to_dict()


@router.get("/frame_yolo")
def frame_yolo(
    mcap_path: str,
    topic: str | None = None,
    frame_seq: int = 0,
    model_path: str = "models/yolov8n.onnx",
    labels_path: str = "models/coco_classes.txt",
    target_classes: str | None = "person,car,truck,bus",
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> dict[str, object]:
    if not Path(mcap_path).exists():
        raise HTTPException(status_code=400, detail={"error": f"MCAP path does not exist: {mcap_path}"})
    decoder = RosImageDecoder()
    reader = McapReader()
    labels = load_labels(labels_path)
    runner = YoloOnnxRunner(model_path)
    topics = [topic] if topic else None
    for message in reader.iter_messages(mcap_path, topics=topics):
        frame = decoder.decode(message)
        if frame is None or frame.frame_seq != frame_seq:
            continue
        prep = preprocess(frame.image)
        inference = runner.infer(prep.tensor)
        detections = postprocess(
            inference.outputs,
            prep.letterbox,
            labels,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            target_classes=parse_target_classes(target_classes),
        )
        visualized = draw_detections(frame.image, detections)
        ok, encoded = cv2.imencode(".jpg", visualized)
        if not ok:
            raise HTTPException(status_code=500, detail={"error": "failed to encode YOLO preview"})
        return {
            "topic": frame.topic,
            "frame_seq": frame.frame_seq,
            "detections": [detection.to_dict() for detection in detections],
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    raise HTTPException(status_code=404, detail={"error": "frame not found"})
