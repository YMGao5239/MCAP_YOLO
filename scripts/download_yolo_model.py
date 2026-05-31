"""下载 / 导出 YOLO ONNX 模型到 models/ (推荐脚本).

例: 用 ultralytics 导出 yolov8n.onnx (仅导出,推理仍走 ONNX Runtime)。
README 中需说明模型来源、输入输出、前后处理。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_OUTPUT = ROOT / "models" / "yolov8n.onnx"


def verify_onnx_model(path: Path) -> tuple[list[tuple[str, list[int | str | None], str]], list[tuple[str, list[int | str | None], str]]]:
    """Load an ONNX model with ONNX Runtime and return input/output metadata."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [(item.name, list(item.shape), item.type) for item in session.get_inputs()]
    outputs = [(item.name, list(item.shape), item.type) for item in session.get_outputs()]
    return inputs, outputs


def archive_downloaded_pt(model_name: str, models_dir: Path) -> None:
    downloaded_pt = Path(model_name)
    if downloaded_pt.exists() and downloaded_pt.is_file() and downloaded_pt.suffix == ".pt":
        target_pt = models_dir / downloaded_pt.name
        if downloaded_pt.resolve() != target_pt.resolve():
            shutil.move(str(downloaded_pt), target_pt)


def export_yolo_onnx(
    model_name: str = DEFAULT_MODEL,
    output_path: Path = DEFAULT_OUTPUT,
    opset: int = 12,
    image_size: int = 640,
    force: bool = False,
) -> Path:
    """Export a YOLO model to ONNX using Ultralytics, then verify it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        verify_onnx_model(output_path)
        archive_downloaded_pt(model_name, output_path.parent)
        return output_path

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required to export YOLO. Install it in your environment first: "
            "pip install ultralytics"
        ) from exc

    model = YOLO(model_name)
    exported = Path(
        model.export(
            format="onnx",
            opset=opset,
            imgsz=image_size,
            dynamic=False,
            simplify=False,
        )
    )

    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), output_path)

    archive_downloaded_pt(model_name, output_path.parent)
    verify_onnx_model(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX and verify it with ONNX Runtime.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics model name or local .pt path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--force", action="store_true", help="Re-export even if the ONNX file already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = export_yolo_onnx(
        model_name=args.model,
        output_path=args.output,
        opset=args.opset,
        image_size=args.imgsz,
        force=args.force,
    )
    inputs, outputs = verify_onnx_model(output)
    print(f"ONNX model ready: {output}")
    print("Inputs:")
    for name, shape, dtype in inputs:
        print(f"  - {name}: shape={shape}, type={dtype}")
    print("Outputs:")
    for name, shape, dtype in outputs:
        print(f"  - {name}: shape={shape}, type={dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
