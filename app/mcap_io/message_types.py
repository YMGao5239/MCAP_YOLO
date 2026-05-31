"""消息类型常量与数据结构.

IMAGE_MESSAGE_TYPES = {
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/Image",
}
DecodedFrame 数据类等。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


IMAGE_MESSAGE_TYPES = {
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/Image",
}


@dataclass(frozen=True)
class TopicInfo:
    name: str
    message_type: str
    message_count: int
    serialization_format: str = "cdr"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectedTopic:
    topic: str
    message_type: str
    message_count: int
    is_image_topic: bool
    selected: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class McapFileSummary:
    path: str
    duration_sec: float
    message_count: int
    start_time_ns: int | None
    end_time_ns: int | None
    topics: list[TopicInfo]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["topics"] = [topic.to_dict() for topic in self.topics]
        return data


@dataclass(frozen=True)
class FailedMcapFile:
    path: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class McapBatchSummary:
    files: list[McapFileSummary]
    failed_files: list[FailedMcapFile]

    @property
    def total_message_count(self) -> int:
        return sum(file.message_count for file in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [file.to_dict() for file in self.files],
            "failed_files": [file.to_dict() for file in self.failed_files],
            "total_message_count": self.total_message_count,
        }


@dataclass(frozen=True)
class McapMessage:
    mcap_path: str
    topic: str
    message_type: str
    log_time_ns: int
    data: bytes
