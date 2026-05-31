"""FastAPI 应用入口.

启动:  uvicorn app.main:app --host 0.0.0.0 --port 8000
访问:  http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI

from app.api import health, jobs, mcap, yolo

app = FastAPI(
    title="MCAP YOLO Image Quality Gateway",
    version="1.0.0",
    description="MCAP 视频图像数据质量评估与 YOLO 模型部署",
)

app.include_router(health.router)
app.include_router(mcap.router, prefix="/mcap")
app.include_router(jobs.router, prefix="/jobs")
app.include_router(yolo.router, prefix="/mcap")


@app.get("/")
def root():
    return {"service": "mcap_yolo_quality_gateway", "docs": "/docs"}
