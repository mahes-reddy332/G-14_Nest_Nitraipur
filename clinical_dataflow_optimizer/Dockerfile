
# ==========================================
# Backend Dockerfile (FastAPI)
# ==========================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (needed for compilation of some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for cache optimization
COPY api/requirements.txt /app/api/requirements.txt

# Install Python dependencies
# Note: We filter out win32 specific constraints if any, though Docker is Linux.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r api/requirements.txt

# Copy the entire workspace
# We exclude files via .dockerignore
COPY . /app

# Expose the API port
EXPOSE 8000

# Health check (requires curl or custom script)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# We use the main API entrypoint
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
