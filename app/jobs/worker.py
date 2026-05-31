"""任务执行器: 串起 reader → decoder → quality → yolo → report 的完整管线."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.mcap_io.reader import McapReader
from app.mcap_io.ros_image_decoder import RosImageDecoder
from app.mcap_io.topic_scanner import TopicScanner
from app.core.logging import log_metrics_line, summarize_ms
from app.quality.analyzer import QualityAnalyzer
from app.quality.scoring import QualityScorer
from app.quality.sequence_analyzer import analyze_sequence, summarize_by_topic
from app.report.html_report import write_html_report
from app.report.json_report import build_mcap_summary, write_json_report
from app.report.markdown_report import write_markdown_report
from app.report.sample_exporter import export_bad_samples, export_detection_samples
from app.yolo.labels import load_labels, parse_target_classes
from app.yolo.onnx_runner import YoloOnnxRunner
from app.yolo.postprocess import postprocess
from app.yolo.preprocess import preprocess


@dataclass(frozen=True)
class QualityScanParams:
    mcap: str | Path
    output_dir: str | Path = Path("outputs")
    auto_detect_topics: bool = True
    topics: list[str] | None = None
    sample_every_n: int = 1
    start_sec: float | None = None
    end_sec: float | None = None
    quality_threshold: float = 0.6
    max_bad_samples: int = 20
    max_frames: int | None = None


@dataclass(frozen=True)
class YoloInferenceParams(QualityScanParams):
    model_path: str | Path = Path("models/yolov8n.onnx")
    labels_path: str | Path = Path("models/coco_classes.txt")
    target_classes: set[str] | list[str] | str | None = None
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    infer_low_quality: bool = False
    max_detection_samples: int = 20


@dataclass(frozen=True)
class PipelineResult:
    status: str
    output_dir: str
    quality_report_path: str
    yolo_predictions_path: str | None
    metrics_path: str
    total_frames: int
    decoded_frames: int
    bad_quality_frames: int
    skipped_low_quality_frames: int = 0
    inferred_frames: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_quality_scan(params: QualityScanParams) -> PipelineResult:
    context = _build_quality_records(params)
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mcap_summary_path = output_dir / "mcap_summary.json"
    quality_report_path = output_dir / "quality_report.json"
    metrics_path = output_dir / "metrics.json"
    markdown_path = output_dir / "quality_report.md"
    html_path = output_dir / "quality_report.html"

    bad_index = export_bad_samples(context["decoded_items"], output_dir, max_bad_samples=params.max_bad_samples)
    detection_index = {"samples": []}
    write_json_report(mcap_summary_path, context["mcap_summary"])
    write_json_report(quality_report_path, context["quality_report"])
    write_json_report(metrics_path, context["metrics"])
    write_markdown_report(markdown_path, context["quality_report"], None, context["metrics"])
    write_html_report(html_path, context["quality_report"], None, context["metrics"], bad_index, detection_index)
    return PipelineResult(
        status="finished",
        output_dir=str(output_dir),
        quality_report_path=str(quality_report_path),
        yolo_predictions_path=None,
        metrics_path=str(metrics_path),
        total_frames=context["metrics"]["total_frames"],
        decoded_frames=context["metrics"]["decoded_frames"],
        bad_quality_frames=context["metrics"]["bad_quality_frames"],
    )


def run_yolo_inference(params: YoloInferenceParams, runner: Any | None = None) -> PipelineResult:
    context = _build_quality_records(params)
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(params.labels_path)
    target_classes = parse_target_classes(params.target_classes)
    runner = runner or YoloOnnxRunner(params.model_path)

    frames: list[dict[str, Any]] = []
    prediction_items: list[dict[str, Any]] = []
    inference_ms_values: list[float] = []
    preprocess_ms_values: list[float] = []
    postprocess_ms_values: list[float] = []
    skipped_low_quality_frames = 0
    inferred_frames = 0

    for item in context["decoded_items"]:
        score = item["score"]["score"]
        if score < params.quality_threshold and not params.infer_low_quality:
            skipped_low_quality_frames += 1
            frames.append(_prediction_record(item, [], skipped=True, inference_ms=None))
            continue

        preprocess_started = time.perf_counter()
        prep = preprocess(item["frame"].image)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        inference = runner.infer(prep.tensor)
        postprocess_started = time.perf_counter()
        detections = postprocess(
            inference.outputs,
            prep.letterbox,
            labels,
            conf_threshold=params.conf_threshold,
            nms_threshold=params.nms_threshold,
            target_classes=target_classes,
        )
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        preprocess_ms_values.append(preprocess_ms)
        postprocess_ms_values.append(postprocess_ms)
        inference_ms_values.append(inference.inference_ms)
        inferred_frames += 1
        frames.append(_prediction_record(item, detections, skipped=False, inference_ms=inference.inference_ms))
        prediction_items.append({"frame": item["frame"], "detections": detections})

    mcap_summary_path = output_dir / "mcap_summary.json"
    quality_report_path = output_dir / "quality_report.json"
    predictions_path = output_dir / "yolo_predictions.json"
    metrics_path = output_dir / "metrics.json"
    markdown_path = output_dir / "quality_report.md"
    html_path = output_dir / "quality_report.html"
    yolo_predictions = {
        "mcap_file": str(params.mcap),
        "target_classes": sorted(target_classes) if target_classes else None,
        "conf_threshold": params.conf_threshold,
        "nms_threshold": params.nms_threshold,
        "infer_low_quality": params.infer_low_quality,
        "skipped_low_quality_frames": skipped_low_quality_frames,
        "inferred_frames": inferred_frames,
        "frames": frames,
    }
    avg_inf, p95_inf = summarize_ms(inference_ms_values)
    avg_pre, p95_pre = summarize_ms(preprocess_ms_values)
    avg_post, p95_post = summarize_ms(postprocess_ms_values)
    metrics = {
        **context["metrics"],
        "skipped_low_quality_frames": skipped_low_quality_frames,
        "inferred_frames": inferred_frames,
        "avg_preprocess_ms": avg_pre,
        "p95_preprocess_ms": p95_pre,
        "avg_inference_ms": avg_inf,
        "p95_inference_ms": p95_inf,
        "avg_postprocess_ms": avg_post,
        "p95_postprocess_ms": p95_post,
    }
    bad_index = export_bad_samples(context["decoded_items"], output_dir, max_bad_samples=params.max_bad_samples)
    detection_index = export_detection_samples(prediction_items, output_dir, max_samples=params.max_detection_samples)
    write_json_report(mcap_summary_path, context["mcap_summary"])
    write_json_report(quality_report_path, context["quality_report"])
    write_json_report(predictions_path, yolo_predictions)
    write_json_report(metrics_path, metrics)
    write_markdown_report(markdown_path, context["quality_report"], yolo_predictions, metrics)
    write_html_report(html_path, context["quality_report"], yolo_predictions, metrics, bad_index, detection_index)
    return PipelineResult(
        status="finished",
        output_dir=str(output_dir),
        quality_report_path=str(quality_report_path),
        yolo_predictions_path=str(predictions_path),
        metrics_path=str(metrics_path),
        total_frames=context["metrics"]["total_frames"],
        decoded_frames=context["metrics"]["decoded_frames"],
        bad_quality_frames=context["metrics"]["bad_quality_frames"],
        skipped_low_quality_frames=skipped_low_quality_frames,
        inferred_frames=inferred_frames,
    )


def _build_quality_records(params: QualityScanParams) -> dict[str, Any]:
    started_at = time.perf_counter()
    reader = McapReader()
    decoder = RosImageDecoder()
    analyzer = QualityAnalyzer()
    scorer = QualityScorer(quality_threshold=params.quality_threshold)
    summary = reader.inspect_file(params.mcap)
    scanner = TopicScanner(auto_detect=params.auto_detect_topics, topics=params.topics)
    detected_topics = scanner.scan(summary.topics)
    selected_topics = scanner.selected_topics(summary.topics)

    records: list[dict[str, Any]] = []
    decoded_items: list[dict[str, Any]] = []
    timestamps_by_topic: dict[str, list[int]] = {topic: [] for topic in selected_topics}
    for message in reader.iter_messages(
        params.mcap,
        topics=selected_topics,
        sample_every_n=params.sample_every_n,
        start_sec=params.start_sec,
        end_sec=params.end_sec,
    ):
        if params.max_frames is not None and len(records) >= params.max_frames:
            break
        frame = decoder.decode(message)
        if frame is None:
            records.append({"topic": message.topic, "decoded": False, "quality_score": None, "quality_tags": ["decode_failed"]})
            continue
        quality = analyzer.analyze(frame.image)
        score = scorer.score(quality)
        record = {
            "topic": frame.topic,
            "frame_seq": frame.frame_seq,
            "log_time_ns": frame.log_time_ns,
            "publish_time_ns": frame.publish_time_ns,
            "decoded": True,
            "width": frame.width,
            "height": frame.height,
            "encoding": frame.encoding,
            "quality_score": score.score,
            "quality_tags": score.quality_tags,
            "penalties": score.penalties,
            "quality_flags": {
                "is_too_dark": quality.is_too_dark,
                "is_too_bright": quality.is_too_bright,
                "is_blurry": quality.is_blurry,
                "is_low_contrast": quality.is_low_contrast,
                "is_low_resolution": quality.is_low_resolution,
                "is_corrupted": quality.is_corrupted,
            },
            "metrics": quality.metrics.to_dict(),
        }
        records.append(record)
        decoded_items.append({"frame": frame, "quality": quality, "score": score.to_dict(), "record": record})
        timestamps_by_topic.setdefault(frame.topic, []).append(frame.log_time_ns)

    summaries = summarize_by_topic(records)
    sequences = {topic: analyze_sequence(timestamps).to_dict() for topic, timestamps in timestamps_by_topic.items()}
    quality_report = {
        "mcap_file": str(params.mcap),
        "selected_topics": selected_topics,
        "topic_summaries": {topic: item.to_dict() for topic, item in summaries.items()},
        "sequence": sequences,
        "frames": records,
        "decoder_stats": decoder.stats.to_dict(),
    }
    metrics = {
        "total_frames": len(records),
        "processed_frames": sum(1 for item in records if item.get("decoded")),
        "decoded_frames": sum(1 for item in records if item.get("decoded")),
        "decode_failed_frames": sum(1 for item in records if not item.get("decoded")),
        "bad_quality_frames": sum(1 for item in records if "bad_quality" in (item.get("quality_tags") or [])),
    }
    decode_values = [item["frame"].decode_ms for item in decoded_items if item.get("frame") is not None]
    avg_decode, p95_decode = summarize_ms(decode_values)
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    metrics.update(
        {
            "avg_decode_ms": avg_decode,
            "p95_decode_ms": p95_decode,
            "avg_preprocess_ms": 0.0,
            "p95_preprocess_ms": 0.0,
            "avg_inference_ms": 0.0,
            "p95_inference_ms": 0.0,
            "avg_postprocess_ms": 0.0,
            "p95_postprocess_ms": 0.0,
            "processed_frames_per_sec": round(metrics["processed_frames"] / elapsed, 4),
        }
    )
    log_metrics_line(
        mcap_file=params.mcap,
        topic="*",
        message_type="*",
        frame_seq="-",
        decode_ms=avg_decode,
        preprocess_ms=metrics["avg_preprocess_ms"],
        inference_ms=metrics["avg_inference_ms"],
        postprocess_ms=metrics["avg_postprocess_ms"],
        quality_score="-",
        quality_tags="-",
        object_count="-",
        target_object_count="-",
        processed_frames=metrics["processed_frames"],
        decode_failed_frames=metrics["decode_failed_frames"],
        skipped_low_quality_frames=0,
        bad_quality_frames=metrics["bad_quality_frames"],
    )
    return {
        "mcap_summary": build_mcap_summary(summary, detected_topics),
        "quality_report": quality_report,
        "metrics": metrics,
        "decoded_items": decoded_items,
    }


def _prediction_record(item: dict[str, Any], detections: list[Any], skipped: bool, inference_ms: float | None) -> dict[str, Any]:
    frame = item["frame"]
    return {
        "topic": frame.topic,
        "frame_seq": frame.frame_seq,
        "log_time_ns": frame.log_time_ns,
        "quality_score": item["score"]["score"],
        "quality_tags": item["score"]["quality_tags"],
        "skipped_low_quality": skipped,
        "inference_ms": round(inference_ms, 4) if inference_ms is not None else None,
        "detections": [detection.to_dict() for detection in detections],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_report(path, payload)
