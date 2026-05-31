"""生成测试用 MCAP (推荐脚本).

写入若干合成图像帧到 CompressedImage / Image Topic,便于无真实数据时复现。
README 中需说明测试 MCAP 如何生成或下载。
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "test_data" / "sample.mcap"
IMAGE_TOPIC = "/camera/image"
COMPRESSED_TOPIC = "/camera/compressed"
IMAGE_TYPE = "sensor_msgs/msg/Image"
COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"


def make_test_frame(seq: int) -> np.ndarray:
    """Create deterministic frames with a few quality problems mixed in."""
    if seq % 10 == 3:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if seq % 10 == 6:
        low_res = np.full((90, 120, 3), (35, 80, 160), dtype=np.uint8)
        cv2.putText(low_res, f"low {seq}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return low_res

    image = np.full((480, 640, 3), (45, 90, 135), dtype=np.uint8)
    cv2.rectangle(image, (60 + seq * 4, 80), (220 + seq * 4, 260), (40, 180, 70), -1)
    cv2.circle(image, (420, 220 + seq * 3), 60, (210, 80, 45), -1)
    cv2.line(image, (0, 360), (639, 120), (240, 240, 240), 3)
    cv2.putText(image, f"frame {seq:02d}", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if seq % 10 == 8:
        return cv2.GaussianBlur(image, (31, 31), 0)

    return image


def _header(typestore, seq: int, timestamp_ns: int):
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Header = typestore.types["std_msgs/msg/Header"]
    stamp = Time(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000)
    return Header(stamp=stamp, frame_id=f"camera_{seq:04d}")


def _image_message(typestore, frame: np.ndarray, seq: int, timestamp_ns: int):
    Image = typestore.types[IMAGE_TYPE]
    height, width = frame.shape[:2]
    return Image(
        header=_header(typestore, seq, timestamp_ns),
        height=height,
        width=width,
        encoding="bgr8",
        is_bigendian=0,
        step=width * 3,
        data=np.ascontiguousarray(frame).reshape(-1),
    )


def _compressed_message(typestore, frame: np.ndarray, seq: int, timestamp_ns: int):
    CompressedImage = typestore.types[COMPRESSED_TYPE]
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode frame {seq}")
    return CompressedImage(
        header=_header(typestore, seq, timestamp_ns),
        format="jpeg",
        data=np.asarray(encoded, dtype=np.uint8).reshape(-1),
    )


def generate_sample_mcap(output_path: Path = DEFAULT_OUTPUT, frame_count: int = 30, fps: float = 10.0) -> Path:
    """Generate a small MCAP with raw Image and CompressedImage topics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    start_ns = 1_700_000_000_000_000_000
    step_ns = int(1_000_000_000 / fps)

    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}-", dir=output_path.parent))
    bag_dir = temp_root / "rosbag"

    with Writer(bag_dir, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        raw_conn = writer.add_connection(IMAGE_TOPIC, IMAGE_TYPE, typestore=typestore)
        compressed_conn = writer.add_connection(COMPRESSED_TOPIC, COMPRESSED_TYPE, typestore=typestore)

        for seq in range(frame_count):
            timestamp_ns = start_ns + seq * step_ns
            frame = make_test_frame(seq)

            raw_msg = _image_message(typestore, frame, seq, timestamp_ns)
            raw_data = typestore.serialize_cdr(raw_msg, IMAGE_TYPE)
            writer.write(raw_conn, timestamp_ns, raw_data)

            compressed_msg = _compressed_message(typestore, frame, seq, timestamp_ns)
            compressed_data = typestore.serialize_cdr(compressed_msg, COMPRESSED_TYPE)
            writer.write(compressed_conn, timestamp_ns, compressed_data)

    mcap_files = sorted(bag_dir.glob("*.mcap"))
    if len(mcap_files) != 1:
        shutil.rmtree(temp_root)
        raise RuntimeError(f"Expected one MCAP storage file in {bag_dir}, found {len(mcap_files)}")

    shutil.move(str(mcap_files[0]), output_path)
    shutil.rmtree(temp_root)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic MCAP with ROS image topics.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .mcap path.")
    parser.add_argument("--frames", type=int, default=30, help="Frames per topic.")
    parser.add_argument("--fps", type=float, default=10.0, help="Synthetic frame rate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = generate_sample_mcap(args.output, frame_count=args.frames, fps=args.fps)
    print(f"Generated test MCAP: {output}")
    print(f"Topics: {IMAGE_TOPIC} ({IMAGE_TYPE}), {COMPRESSED_TOPIC} ({COMPRESSED_TYPE})")
    print("Mixed quality cases: normal, black, blurred, low-resolution frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
