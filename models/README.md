# models/

把 `yolov8n.onnx` 放在此处 (不入库,见 .gitignore)。

导出示例:
```bash
python scripts/download_yolo_model.py
# 或: yolo export model=yolov8n.pt format=onnx opset=12
```
`coco_classes.txt` 为 80 类标签 (class_id 顺序)。
