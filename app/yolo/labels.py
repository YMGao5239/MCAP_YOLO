"""类别标签加载 (models/coco_classes.txt).

class_id -> label 映射;支持 --target-classes 关键类别过滤。
"""
from __future__ import annotations

from pathlib import Path


def load_labels(path: str | Path) -> list[str]:
    labels = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def parse_target_classes(value: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    targets = {item for item in items if item}
    return targets or None
