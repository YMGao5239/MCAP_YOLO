"""ROS 图像消息解码 (FR-IMG-001/002/003).

CompressedImage (FR-IMG-001):
 - 字段 header / format / data;支持 jpeg / png;cv2.imdecode → BGR ndarray
 - 解码失败计数,不崩溃

Image 原始图像 (FR-IMG-002):
 - encoding: rgb8/bgr8/mono8/...;按 encoding 转 OpenCV BGR
 - 不支持的 encoding 记录并跳过

保留时间戳 (FR-IMG-003):
 - 返回 DecodedFrame(log_time, publish_time, frame_seq, width, height,
   encoding, decode_ms, image(ndarray))

设计要点 (NFR-001 稳定性):
 - 单帧解码失败/未知 encoding/损坏数据只计数并跳过,**绝不抛异常打断整条管线**。
 - reader 产出的是原始 CDR 字节,本模块负责反序列化 + 像素解码,
   保持 reader 与解码职责分离。
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore

from app.mcap_io.message_types import McapMessage

COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
IMAGE_TYPE = "sensor_msgs/msg/Image"

# 原始 Image.encoding → (通道数, numpy dtype, 转 BGR 的 cv2 颜色码或 None)
# None 表示已经是 BGR 排列,直接使用。
_RAW_ENCODINGS: dict[str, tuple[int, str, int | None]] = {
    "bgr8": (3, "uint8", None),
    "rgb8": (3, "uint8", cv2.COLOR_RGB2BGR),
    "bgra8": (4, "uint8", cv2.COLOR_BGRA2BGR),
    "rgba8": (4, "uint8", cv2.COLOR_RGBA2BGR),
    "mono8": (1, "uint8", cv2.COLOR_GRAY2BGR),
    "8uc1": (1, "uint8", cv2.COLOR_GRAY2BGR),
    "8uc3": (3, "uint8", None),
    "8uc4": (4, "uint8", cv2.COLOR_BGRA2BGR),
    "mono16": (1, "uint16", cv2.COLOR_GRAY2BGR),
    "16uc1": (1, "uint16", cv2.COLOR_GRAY2BGR),
}

# Bayer 马赛克 → BGR。
_BAYER_ENCODINGS: dict[str, int] = {
    "bayer_rggb8": cv2.COLOR_BayerBG2BGR,
    "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
    "bayer_gbrg8": cv2.COLOR_BayerGR2BGR,
    "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
}


@dataclass
class DecodedFrame:
    """一帧成功解码后的结果 (image 为 BGR ndarray)。"""

    topic: str
    message_type: str
    log_time_ns: int
    publish_time_ns: int | None
    frame_seq: int
    width: int
    height: int
    encoding: str
    decode_ms: float
    image: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """轻量元信息 (不含像素,便于写日志/报告)。"""
        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "log_time_ns": self.log_time_ns,
            "publish_time_ns": self.publish_time_ns,
            "frame_seq": self.frame_seq,
            "width": self.width,
            "height": self.height,
            "encoding": self.encoding,
            "decode_ms": round(self.decode_ms, 3),
        }


@dataclass
class DecodeStats:
    """整段解码过程的统计 (FR-IMG-003 / NFR-001)。"""

    total_frames: int = 0
    decoded_frames: int = 0
    decode_failed_frames: int = 0
    unsupported_encoding_frames: int = 0
    failures: list[str] = field(default_factory=list)
    unsupported_encodings: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_frames": self.total_frames,
            "decoded_frames": self.decoded_frames,
            "decode_failed_frames": self.decode_failed_frames,
            "unsupported_encoding_frames": self.unsupported_encoding_frames,
            "unsupported_encodings": dict(self.unsupported_encodings),
            "failures": list(self.failures),
        }


class DecodeError(Exception):
    """内部使用:单帧解码失败,被 decode() 捕获并计数,不向外传播。"""


class RosImageDecoder:
    """把 reader 产出的原始 ROS 图像消息解码为 BGR ndarray。

    用法::

        decoder = RosImageDecoder()
        for message in reader.iter_messages(path, topics=[...]):
            frame = decoder.decode(message)
            if frame is None:        # 坏帧/未知 encoding 已计数,跳过即可
                continue
            ...                      # frame.image 为 BGR ndarray
        print(decoder.stats.to_dict())
    """

    _MAX_RECORDED_FAILURES = 50

    def __init__(self, typestore=None) -> None:
        self.typestore = typestore or get_typestore(Stores.ROS2_HUMBLE)
        self.stats = DecodeStats()
        self._seq_by_topic: dict[str, int] = defaultdict(int)

    # ---- 公共入口 -----------------------------------------------------------
    def decode(self, message: McapMessage) -> DecodedFrame | None:
        """解码单条消息;失败返回 None 并累加统计,绝不抛异常。"""
        self.stats.total_frames += 1
        frame_seq = self._seq_by_topic[message.topic]
        self._seq_by_topic[message.topic] += 1

        started = time.perf_counter()
        try:
            ros_msg = self.typestore.deserialize_cdr(message.data, message.message_type)
            if message.message_type == COMPRESSED_TYPE:
                image, encoding = self._decode_compressed(ros_msg)
            elif message.message_type == IMAGE_TYPE:
                image, encoding = self._decode_raw(ros_msg)
            else:
                raise DecodeError(f"non-image message type {message.message_type!r}")
        except _UnsupportedEncoding as exc:
            self.stats.unsupported_encoding_frames += 1
            self.stats.unsupported_encodings[exc.encoding] = (
                self.stats.unsupported_encodings.get(exc.encoding, 0) + 1
            )
            self._record_failure(message, frame_seq, f"unsupported encoding {exc.encoding!r}")
            return None
        except Exception as exc:  # noqa: BLE001 - 任一坏帧都不能打断管线 (NFR-001)。
            self.stats.decode_failed_frames += 1
            self._record_failure(message, frame_seq, f"{type(exc).__name__}: {exc}")
            return None

        decode_ms = (time.perf_counter() - started) * 1000.0
        height, width = image.shape[:2]
        self.stats.decoded_frames += 1
        return DecodedFrame(
            topic=message.topic,
            message_type=message.message_type,
            log_time_ns=message.log_time_ns,
            publish_time_ns=_publish_time_ns(ros_msg),
            frame_seq=frame_seq,
            width=int(width),
            height=int(height),
            encoding=encoding,
            decode_ms=decode_ms,
            image=image,
        )

    # ---- CompressedImage (FR-IMG-001) --------------------------------------
    def _decode_compressed(self, ros_msg) -> tuple[np.ndarray, str]:
        fmt = (getattr(ros_msg, "format", "") or "").strip()
        buffer = np.asarray(ros_msg.data, dtype=np.uint8)
        if buffer.size == 0:
            raise DecodeError("empty CompressedImage payload")
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise DecodeError(f"cv2.imdecode failed (format={fmt!r})")
        encoding = fmt.split(";", 1)[0].strip() or "compressed"
        return image, encoding

    # ---- 原始 Image (FR-IMG-002) -------------------------------------------
    def _decode_raw(self, ros_msg) -> tuple[np.ndarray, str]:
        encoding = (getattr(ros_msg, "encoding", "") or "").strip()
        key = encoding.lower()
        height = int(ros_msg.height)
        width = int(ros_msg.width)
        if height <= 0 or width <= 0:
            raise DecodeError(f"invalid image size {width}x{height}")

        if key in _BAYER_ENCODINGS:
            channels, dtype, color_code = 1, "uint8", _BAYER_ENCODINGS[key]
        elif key in _RAW_ENCODINGS:
            channels, dtype, color_code = _RAW_ENCODINGS[key]
        else:
            raise _UnsupportedEncoding(encoding or "<empty>")

        itemsize = np.dtype(dtype).itemsize
        raw = np.asarray(ros_msg.data, dtype=np.uint8)
        if dtype != "uint8":
            if getattr(ros_msg, "is_bigendian", 0):
                raw = raw.reshape(-1, itemsize)[:, ::-1].reshape(-1)
            pixels = raw.view(dtype)
        else:
            pixels = raw

        row_pixels = width * channels
        step = int(getattr(ros_msg, "step", 0) or 0)
        step_pixels = step // itemsize if step else 0
        if step_pixels > row_pixels:  # 行有 padding,按 step 重排后裁掉。
            grid = pixels[: height * step_pixels].reshape(height, step_pixels)
            grid = grid[:, :row_pixels]
        else:
            if pixels.size < height * row_pixels:
                raise DecodeError(
                    f"payload too small: have {pixels.size}, need {height * row_pixels}"
                )
            grid = pixels[: height * row_pixels].reshape(height, row_pixels)

        if channels == 1:
            plane = grid.reshape(height, width)
        else:
            plane = grid.reshape(height, width, channels)

        if np.dtype(dtype).itemsize > 1:  # 16-bit → 归一化到 8-bit 灰度。
            plane = cv2.normalize(plane, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        image = plane if color_code is None else cv2.cvtColor(plane, color_code)
        return np.ascontiguousarray(image), encoding

    # ---- 内部工具 -----------------------------------------------------------
    def _record_failure(self, message: McapMessage, frame_seq: int, reason: str) -> None:
        if len(self.stats.failures) < self._MAX_RECORDED_FAILURES:
            self.stats.failures.append(f"{message.topic}#{frame_seq}: {reason}")


class _UnsupportedEncoding(DecodeError):
    def __init__(self, encoding: str) -> None:
        super().__init__(f"unsupported encoding {encoding!r}")
        self.encoding = encoding


def _publish_time_ns(ros_msg) -> int | None:
    """从 std_msgs/Header.stamp 提取发布时间戳 (纳秒)。"""
    header = getattr(ros_msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)
