"""图像解码单测: CompressedImage(jpeg/png) 与 raw Image(encoding)。

覆盖 FR-IMG-001/002/003 与 NFR-001:
 - jpeg / png 压缩图解码
 - bgr8 / rgb8 / mono8 / 带行 padding(step) 的原图解码
 - 未知 encoding、损坏数据只计数不崩溃
 - 时间戳保留 (log_time / publish_time / frame_seq)
 - 对 sample.mcap 逐帧解出 ndarray,统计正确
"""
from __future__ import annotations

import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore

from app.mcap_io.message_types import McapMessage
from app.mcap_io.reader import McapReader
from app.mcap_io.ros_image_decoder import RosImageDecoder
from scripts.generate_test_mcap import generate_sample_mcap

COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
IMAGE_TYPE = "sensor_msgs/msg/Image"


def _typestore():
    return get_typestore(Stores.ROS2_HUMBLE)


def _header(typestore, seq: int, stamp_ns: int):
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Header = typestore.types["std_msgs/msg/Header"]
    stamp = Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
    return Header(stamp=stamp, frame_id=f"cam_{seq}")


def _compressed_message(typestore, frame, fmt: str, seq: int = 0, stamp_ns: int = 0) -> McapMessage:
    ext = ".jpg" if fmt == "jpeg" else ".png"
    ok, encoded = cv2.imencode(ext, frame)
    assert ok
    CompressedImage = typestore.types[COMPRESSED_TYPE]
    msg = CompressedImage(
        header=_header(typestore, seq, stamp_ns),
        format=fmt,
        data=np.asarray(encoded, dtype=np.uint8).reshape(-1),
    )
    data = typestore.serialize_cdr(msg, COMPRESSED_TYPE)
    return McapMessage("mem", "/c", COMPRESSED_TYPE, log_time_ns=stamp_ns, data=bytes(data))


def _raw_message(typestore, frame, encoding: str, seq: int = 0, stamp_ns: int = 0, step: int | None = None) -> McapMessage:
    height, width = frame.shape[:2]
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    Image = typestore.types[IMAGE_TYPE]
    msg = Image(
        header=_header(typestore, seq, stamp_ns),
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=0,
        step=step if step is not None else width * channels,
        data=np.ascontiguousarray(frame).reshape(-1).astype(np.uint8),
    )
    data = typestore.serialize_cdr(msg, IMAGE_TYPE)
    return McapMessage("mem", "/r", IMAGE_TYPE, log_time_ns=stamp_ns, data=bytes(data))


def _color_frame() -> np.ndarray:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :16] = (200, 10, 10)   # BGR: blue block
    frame[:, 16:32] = (10, 200, 10)  # green block
    frame[:, 32:] = (10, 10, 200)    # red block
    return frame


def test_decode_compressed_jpeg_and_png():
    ts = _typestore()
    frame = _color_frame()
    decoder = RosImageDecoder(typestore=ts)

    jpeg = decoder.decode(_compressed_message(ts, frame, "jpeg"))
    png = decoder.decode(_compressed_message(ts, frame, "png"))

    assert jpeg is not None and png is not None
    assert jpeg.image.shape == (32, 48, 3)
    assert jpeg.encoding == "jpeg" and png.encoding == "png"
    # PNG 无损,应与原图逐像素一致 (BGR)。
    assert np.array_equal(png.image, frame)
    # JPEG 有损,允许误差但色块大致正确。
    assert abs(int(jpeg.image[10, 8, 0]) - 200) < 30
    assert decoder.stats.decoded_frames == 2
    assert decoder.stats.decode_failed_frames == 0


def test_decode_raw_bgr8_is_passthrough():
    ts = _typestore()
    frame = _color_frame()
    decoder = RosImageDecoder(typestore=ts)

    result = decoder.decode(_raw_message(ts, frame, "bgr8"))

    assert result is not None
    assert result.encoding == "bgr8"
    assert np.array_equal(result.image, frame)


def test_decode_raw_rgb8_is_converted_to_bgr():
    ts = _typestore()
    rgb = _color_frame()[:, :, ::-1].copy()  # 把 BGR 测试图当成 RGB 字节
    decoder = RosImageDecoder(typestore=ts)

    result = decoder.decode(_raw_message(ts, rgb, "rgb8"))

    assert result is not None
    # rgb8 解码后通道顺序应翻回 BGR,等于原始 _color_frame()。
    assert np.array_equal(result.image, _color_frame())


def test_decode_raw_mono8_expands_to_three_channels():
    ts = _typestore()
    gray = np.tile(np.arange(48, dtype=np.uint8), (32, 1))
    decoder = RosImageDecoder(typestore=ts)

    result = decoder.decode(_raw_message(ts, gray, "mono8"))

    assert result is not None
    assert result.image.shape == (32, 48, 3)
    assert np.array_equal(result.image[..., 0], result.image[..., 2])  # 灰度 → 三通道相同


def test_decode_raw_handles_row_padding_step():
    ts = _typestore()
    frame = _color_frame()
    height, width = frame.shape[:2]
    pad = 8
    padded = np.zeros((height, width * 3 + pad), dtype=np.uint8)
    padded[:, : width * 3] = frame.reshape(height, width * 3)
    # 手动构造带行 padding(step > width*3)的 Image 消息。
    Image = ts.types[IMAGE_TYPE]
    image_msg = Image(
        header=_header(ts, 0, 0),
        height=height,
        width=width,
        encoding="bgr8",
        is_bigendian=0,
        step=width * 3 + pad,
        data=padded.reshape(-1),
    )
    msg = McapMessage("mem", "/r", IMAGE_TYPE, 0, bytes(ts.serialize_cdr(image_msg, IMAGE_TYPE)))

    decoder = RosImageDecoder(typestore=ts)
    result = decoder.decode(msg)

    assert result is not None
    assert result.image.shape == (height, width, 3)
    assert np.array_equal(result.image, frame)


def test_unsupported_encoding_is_counted_not_raised():
    ts = _typestore()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    decoder = RosImageDecoder(typestore=ts)

    result = decoder.decode(_raw_message(ts, frame, "yuv422"))

    assert result is None
    assert decoder.stats.unsupported_encoding_frames == 1
    assert decoder.stats.decode_failed_frames == 0
    assert decoder.stats.unsupported_encodings.get("yuv422") == 1


def test_corrupted_compressed_payload_is_counted_not_raised():
    ts = _typestore()
    CompressedImage = ts.types[COMPRESSED_TYPE]
    msg = CompressedImage(
        header=_header(ts, 0, 0),
        format="jpeg",
        data=np.frombuffer(b"not-a-real-jpeg", dtype=np.uint8),
    )
    bad = McapMessage("mem", "/c", COMPRESSED_TYPE, 0, bytes(ts.serialize_cdr(msg, COMPRESSED_TYPE)))
    decoder = RosImageDecoder(typestore=ts)

    result = decoder.decode(bad)

    assert result is None
    assert decoder.stats.decode_failed_frames == 1
    assert decoder.stats.total_frames == 1


def test_non_image_message_type_is_skipped():
    ts = _typestore()
    decoder = RosImageDecoder(typestore=ts)
    msg = McapMessage("mem", "/s", "std_msgs/msg/String", 0, b"\x00\x01")

    assert decoder.decode(msg) is None
    assert decoder.stats.decode_failed_frames == 1


def test_timestamps_and_frame_seq_are_preserved():
    ts = _typestore()
    frame = _color_frame()
    decoder = RosImageDecoder(typestore=ts)

    first = decoder.decode(_compressed_message(ts, frame, "jpeg", seq=0, stamp_ns=1_700_000_000_111))
    second = decoder.decode(_compressed_message(ts, frame, "jpeg", seq=1, stamp_ns=1_700_000_000_222))

    assert first.log_time_ns == 1_700_000_000_111
    assert first.publish_time_ns == 1_700_000_000_111
    assert first.frame_seq == 0
    assert second.frame_seq == 1  # 同 topic 帧序递增
    assert first.decode_ms >= 0.0


def test_decodes_every_frame_of_sample_mcap(tmp_path):
    mcap = generate_sample_mcap(tmp_path / "sample.mcap", frame_count=12, fps=10)
    reader = McapReader()
    decoder = RosImageDecoder()

    frames = [decoder.decode(message) for message in reader.iter_messages(mcap)]
    decoded = [frame for frame in frames if frame is not None]

    # 两路 topic × 12 帧 = 24 条消息,全部能反序列化并解出 ndarray。
    assert decoder.stats.total_frames == 24
    assert decoder.stats.decoded_frames == 24
    assert decoder.stats.decode_failed_frames == 0
    assert decoder.stats.unsupported_encoding_frames == 0
    for frame in decoded:
        assert frame.image.ndim == 3 and frame.image.shape[2] == 3
        assert frame.image.dtype == np.uint8
        assert frame.width > 0 and frame.height > 0
