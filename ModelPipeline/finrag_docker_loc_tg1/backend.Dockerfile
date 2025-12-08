# ============================================================================
# Backend Dockerfile - FastAPI + ML Orchestrator
# Context: ModelPipeline/ (parent directory of finrag_docker_loc_tg1)
# Requirements: finrag_ml_tg1/environments/requirements_app_backend.txt
# ============================================================================

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy entire ModelPipeline context
COPY . .

# Install Python dependencies using uv for speed
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install --system --no-cache-dir \
        -r finrag_ml_tg1/environments/requirements_app_backend.txt

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# CRITICAL FIX: Add serving directory to Python path
ENV PYTHONPATH="/app/serving:${PYTHONPATH}"

# NEW: Tell config loader where ModelPipeline root is in Docker
ENV MODEL_PIPELINE_ROOT=/app

# Expose backend port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "backend.api_service:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]