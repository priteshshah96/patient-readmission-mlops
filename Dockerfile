# Patient Readmission Prediction API - Azure Production Dockerfile
# Optimized for Azure Container Apps with multi-stage build

# Build stage - Install dependencies and build artifacts
FROM python:3.11-slim as builder

# Set build arguments for Azure
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Set build environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install UV package manager
RUN pip install uv==0.4.30

# Create application directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using UV (production only)
RUN uv sync --frozen --no-dev --no-editable

# Production stage - Minimal runtime environment
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    APP_ENV=production \
    PORT=8000

# Install runtime system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security (Azure best practice)
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/false appuser

# Create application directories with proper permissions
WORKDIR /app
RUN mkdir -p /app/logs /app/tmp \
    && chown -R appuser:appuser /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./

# Create optimized startup script for Azure
COPY --chown=appuser:appuser <<'SCRIPT' /app/start.sh
#!/bin/bash
set -e

echo "🚀 Starting Patient Readmission Prediction API"
echo "🔧 Environment: $APP_ENV"
echo "🌐 Port: $PORT"

# Azure Container Apps health check
echo "⚕️ Running pre-start health checks..."

# Validate environment variables
if [[ -z "$MLFLOW_TRACKING_URI" ]]; then
    echo "⚠️ MLFLOW_TRACKING_URI not set, using default"
    export MLFLOW_TRACKING_URI="http://localhost:5000"
fi

# Check Python environment
python -c "
import sys
print(f'🐍 Python version: {sys.version}')
print(f'🔍 Python path: {sys.path[0]}')

# Test critical imports
try:
    import mlflow
    import xgboost
    import fastapi
    import src.api.main
    print('✅ All critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Start the FastAPI application with Uvicorn
echo "🔄 Starting FastAPI server on port $PORT..."
exec uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --access-log \
    --log-level info \
    --timeout-keep-alive 30
SCRIPT

RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Expose the port (Azure Container Apps will map this)
EXPOSE $PORT

# Health check configuration optimized for Azure
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:$PORT/health/live || exit 1

# Azure Container Apps startup command
CMD ["/app/start.sh"]

# Metadata labels for Azure Container Registry
LABEL maintainer="Pritesh Shah <priteshshahwork@gmail.com>" \
      description="Patient Readmission Prediction API for Azure" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/priteshshah96/patient-readmission-mlops" \
      org.opencontainers.image.title="Patient Readmission MLOps API" \
      org.opencontainers.image.description="Production-ready API for predicting 30-day hospital readmissions on Azure" \
      azure.container.app=true \
      azure.region=eastus
