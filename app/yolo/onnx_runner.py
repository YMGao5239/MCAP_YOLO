"""YOLO ONNX Runtime 推理.

 - 加载 YOLO ONNX 模型
 - 支持 CPU 推理 (CPUExecutionProvider)
 - 输入来自 MCAP 解析出的图像帧
 - 输出 bbox, class_id, label, confidence
 - 统计推理性能 (inference_ms)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class InferenceResult:
    outputs: list[np.ndarray]
    inference_ms: float


class YoloOnnxRunner:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None) -> None:
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]

    def infer(self, tensor: np.ndarray) -> InferenceResult:
        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor.astype(np.float32, copy=False)})
        inference_ms = (time.perf_counter() - started) * 1000.0
        return InferenceResult(outputs=[np.asarray(item) for item in outputs], inference_ms=inference_ms)
