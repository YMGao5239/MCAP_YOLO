"""MCAP 读取 (FR-MCAP-001/002/003/004).

推荐方案 A: rosbags (轻量、离线友好,不强依赖完整 ROS2)。
功能:
 - 读取单个 MCAP / 目录批处理 (递归扫描 .mcap)
 - 解析元信息: duration_sec, topics, message_count, start/end_time
 - 支持时间范围裁剪 (--start-sec / --end-sec)
 - 支持抽帧 (--sample-every-n / --target-fps)
 - 单文件读取失败不影响其他文件,输出失败文件列表
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from rosbags.highlevel import AnyReader

from app.mcap_io.message_types import (
    FailedMcapFile,
    McapBatchSummary,
    McapFileSummary,
    McapMessage,
    TopicInfo,
)


def discover_mcap_paths(mcap: str | Path | None = None, mcap_dir: str | Path | None = None) -> list[Path]:
    paths: list[Path] = []
    if mcap is not None:
        paths.append(Path(mcap))
    if mcap_dir is not None:
        paths.extend(sorted(Path(mcap_dir).rglob("*.mcap")))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


class McapReader:
    def inspect_file(self, path: str | Path) -> McapFileSummary:
        mcap_path = Path(path)
        with AnyReader([mcap_path]) as reader:
            topics = [
                TopicInfo(
                    name=connection.topic,
                    message_type=connection.msgtype,
                    message_count=connection.msgcount,
                    serialization_format=getattr(connection, "serialization_format", "cdr"),
                )
                for connection in sorted(reader.connections, key=lambda item: item.topic)
            ]
            return McapFileSummary(
                path=str(mcap_path),
                duration_sec=reader.duration / 1_000_000_000,
                message_count=reader.message_count,
                start_time_ns=reader.start_time,
                end_time_ns=reader.end_time,
                topics=topics,
            )

    def inspect_many(self, paths: Iterable[str | Path]) -> McapBatchSummary:
        files: list[McapFileSummary] = []
        failed_files: list[FailedMcapFile] = []
        for path in paths:
            try:
                files.append(self.inspect_file(path))
            except Exception as exc:  # noqa: BLE001 - batch mode must record and continue.
                failed_files.append(FailedMcapFile(path=str(path), error=f"{type(exc).__name__}: {exc}"))
        return McapBatchSummary(files=files, failed_files=failed_files)

    def iter_messages(
        self,
        path: str | Path,
        topics: Sequence[str] | None = None,
        sample_every_n: int = 1,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> Iterator[McapMessage]:
        mcap_path = Path(path)
        sample_every_n = max(1, sample_every_n)
        topic_filter = set(topics or [])

        with AnyReader([mcap_path]) as reader:
            start_ns = reader.start_time + int(start_sec * 1_000_000_000) if start_sec is not None else None
            stop_ns = reader.start_time + int(end_sec * 1_000_000_000) if end_sec is not None else None
            connections = [
                connection
                for connection in reader.connections
                if not topic_filter or connection.topic in topic_filter
            ]
            seen_by_topic: dict[str, int] = defaultdict(int)

            for connection, timestamp, data in reader.messages(connections=connections, start=start_ns, stop=stop_ns):
                seen_by_topic[connection.topic] += 1
                if (seen_by_topic[connection.topic] - 1) % sample_every_n != 0:
                    continue
                yield McapMessage(
                    mcap_path=str(mcap_path),
                    topic=connection.topic,
                    message_type=connection.msgtype,
                    log_time_ns=timestamp,
                    data=bytes(data),
                )
