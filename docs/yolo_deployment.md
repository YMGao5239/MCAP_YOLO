# YOLO 部署说明

## 模型来源

默认模型为 Ultralytics YOLOv8n:

```text
models/yolov8n.pt   -> Ultralytics 预训练权重
models/yolov8n.onnx -> scripts/download_yolo_model.py 导出产物
```

导出命令由脚本封装:

```bash
python scripts/download_yolo_model.py --model yolov8n.pt --output models/yolov8n.onnx --opset 12 --imgsz 640
```

推理阶段只使用 ONNX Runtime,不调用 Ultralytics 高层预测 API。

## ONNX 输入输出

当前 `models/yolov8n.onnx` 经 ONNX Runtime 验证:

```text
input:
  images: [1, 3, 640, 640], tensor(float)

output:
  output0: [1, 84, 8400], tensor(float)
```

`84 = 4 + 80`,其中前 4 维为 `cx, cy, w, h`,后 80 维为 COCO 类别分数。

## 前处理

实现文件: `app/yolo/preprocess.py`

步骤:

1. letterbox resize 到 `640x640`,保持原始宽高比。
2. 记录 `ratio` 和 `pad=(pad_x, pad_y)`,供后处理还原坐标。
3. BGR 转 RGB。
4. 像素归一化到 `[0, 1]`。
5. HWC 转 CHW,增加 batch 维,输出 `float32 [1, 3, 640, 640]`。

## 后处理

实现文件: `app/yolo/postprocess.py`

步骤:

1. 解析 YOLOv8 ONNX 输出 `[1, 84, 8400]`。
2. 按类别分数取 `class_id` 和 `confidence`。
3. 使用 `conf_threshold` 过滤低置信度候选。
4. 将 `xywh` 转为 `xyxy`。
5. 反 letterbox:

```text
x = (x - pad_x) / ratio
y = (y - pad_y) / ratio
```

6. 将 bbox clamp 到原图范围,保证输出坐标属于原始图像。
7. 按类别执行自写 NMS,默认 `nms_threshold=0.45`。
8. 输出 `bbox_xyxy / class_id / label / confidence`。

## 类别过滤

类别文件:

```text
models/coco_classes.txt
```

`app/yolo/labels.py` 负责加载 label。`target_classes` 可过滤关键类别,例如:

```text
person,car,truck,bus
```

## CPU 推理

实现文件: `app/yolo/onnx_runner.py`

加载方式:

```python
ort.InferenceSession("models/yolov8n.onnx", providers=["CPUExecutionProvider"])
```

`YoloOnnxRunner.infer()` 返回原始 ONNX 输出和 `inference_ms`。

## 可视化

实现文件: `app/yolo/visualizer.py`

`draw_detections(image, detections)` 在原始 BGR 图上绘制 bbox、类别和置信度,用于后续
`detection_samples/` 和单帧预览接口。

## 质量门控

P7 管线串联时默认只对质量评分达到阈值的帧执行 YOLO。低质量帧会统计为
`skipped_low_quality_frames`,避免在全黑、模糊或低分辨率帧上浪费推理时间。
