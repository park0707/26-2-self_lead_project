# vidio-health-trainer Production Dockerfile
# Optimized for Hugging Face Spaces (16GB RAM Free Tier) & Standalone Cloud VPS

FROM python:3.10-slim

# 1. Install system runtime dependencies for OpenCV & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup non-root user (Required for Hugging Face Spaces UID 1000 security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# 3. Cache & Install Python dependencies
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# 4. Copy application source code, models, and configuration
COPY --chown=user . /app

# 5. Expose application port (7860 for Hugging Face Spaces)
EXPOSE 7860

# 6. Launch FastAPI web application via uvicorn
CMD ["uvicorn", "src.web.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
