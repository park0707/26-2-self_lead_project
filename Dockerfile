# vidio-health-trainer Production Dockerfile
# Optimized for Render.com Free Tier (No Credit Card) & Standalone Cloud VPS

FROM python:3.10-slim

# 1. Install system runtime dependencies for OpenCV, MediaPipe (libGLESv2), & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libgles2-mesa \
    libegl1-mesa \
    libglib2.0-0 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup user and environment
ENV PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# 3. Cache & Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application source code, models, and configuration
COPY . /app

# 5. Expose application port
EXPOSE 10000

# 6. Launch FastAPI web application via uvicorn (dynamic port binding for Render)
CMD ["sh", "-c", "uvicorn src.web.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
