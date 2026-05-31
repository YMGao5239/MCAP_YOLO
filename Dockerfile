# MCAP YOLO Image Quality Gateway - CPU-only runtime image.
FROM python:3.11-slim

# Keep Python logs visible in docker compose and avoid writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Runtime libraries used by OpenCV/Matplotlib in headless containers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first for better Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Default service command; docker-compose overrides this for smoke tests.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
