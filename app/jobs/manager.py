"""异步任务管理 (FR-API-004/005/006).

内存级任务表: job_id -> {status(running/finished/failed), progress,
result_path, report_path}。供 quality_scan / yolo_infer 异步执行与状态查询。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    job_type: str
    status: str
    progress: float
    result_path: str | None = None
    report_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = Lock()

    def create_job(self, job_type: str) -> JobStatus:
        job = JobStatus(job_id=uuid4().hex, job_type=job_type, status="running", progress=0.0)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobStatus:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job_id: {job_id}")
            return self._jobs[job_id]

    def update(self, job_id: str, progress: float | None = None, result_path: str | Path | None = None, report_path: str | Path | None = None) -> JobStatus:
        job = self.get(job_id)
        updated = JobStatus(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status,
            progress=max(0.0, min(1.0, progress if progress is not None else job.progress)),
            result_path=str(result_path) if result_path is not None else job.result_path,
            report_path=str(report_path) if report_path is not None else job.report_path,
            error=job.error,
        )
        with self._lock:
            self._jobs[job_id] = updated
        return updated

    def finish(self, job_id: str, result_path: str | Path | None = None, report_path: str | Path | None = None) -> JobStatus:
        job = self.get(job_id)
        finished = JobStatus(
            job_id=job.job_id,
            job_type=job.job_type,
            status="finished",
            progress=1.0,
            result_path=str(result_path) if result_path is not None else job.result_path,
            report_path=str(report_path) if report_path is not None else job.report_path,
            error=None,
        )
        with self._lock:
            self._jobs[job_id] = finished
        return finished

    def fail(self, job_id: str, error: BaseException | str) -> JobStatus:
        job = self.get(job_id)
        failed = JobStatus(
            job_id=job.job_id,
            job_type=job.job_type,
            status="failed",
            progress=job.progress,
            result_path=job.result_path,
            report_path=job.report_path,
            error=str(error),
        )
        with self._lock:
            self._jobs[job_id] = failed
        return failed
