# ============================================================================
# Frontend Dockerfile - Streamlit UI
# Context: ModelPipeline/ (parent directory of finrag_docker_loc_tg1)
# Requirements: serving/frontend/requirements.txt
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
        -r serving/frontend/requirements.txt

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# CRITICAL FIX: Add frontend directory to Python path
# This allows "from api_client import ..." to work
ENV PYTHONPATH="/app/serving/frontend:${PYTHONPATH}"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
# Note: Path is 'serving/frontend/app.py' relative to ModelPipeline/
CMD ["streamlit", "run", "serving/frontend/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.serverAddress", "localhost"]