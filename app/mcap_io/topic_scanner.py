"""Topic 扫描与图像 Topic 识别 (FR-MCAP-001 / 作业 7.2).

 - 手动指定: --topics /camera/front/image...
 - 自动发现: --auto-detect-topics true
 - 自动发现时至少识别 sensor_msgs/msg/CompressedImage 与 Image
 - 输出: detected_topics[{topic, message_type, message_count, is_image_topic}]
"""
from __future__ import annotations

from collections.abc import Iterable

from app.mcap_io.message_types import DetectedTopic, IMAGE_MESSAGE_TYPES, TopicInfo


def is_image_message_type(message_type: str) -> bool:
    return message_type in IMAGE_MESSAGE_TYPES


class TopicScanner:
    def __init__(self, auto_detect: bool = True, topics: Iterable[str] | None = None) -> None:
        self.auto_detect = auto_detect
        self.topics = set(topics or [])

    def scan(self, topics: Iterable[TopicInfo]) -> list[DetectedTopic]:
        detected: list[DetectedTopic] = []
        for topic in sorted(topics, key=lambda item: item.name):
            is_image = is_image_message_type(topic.message_type)
            selected = is_image if self.auto_detect else topic.name in self.topics
            detected.append(
                DetectedTopic(
                    topic=topic.name,
                    message_type=topic.message_type,
                    message_count=topic.message_count,
                    is_image_topic=is_image,
                    selected=selected,
                )
            )
        return detected

    def selected_topics(self, topics: Iterable[TopicInfo]) -> list[str]:
        return [topic.topic for topic in self.scan(topics) if topic.selected]
