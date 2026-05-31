"""查询任务状态接口: GET /jobs/{job_id}.

返回: {job_id, status, progress, result_path, report_path}
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.jobs.manager import JobManager


router = APIRouter()
job_manager = JobManager()


@router.get("/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    try:
        return job_manager.get(job_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"error": str(exc)}) from exc
