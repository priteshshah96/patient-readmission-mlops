#!/bin/bash
set -e

echo "🚀 Starting Patient Readmission Prediction API"
echo "🔧 Environment: $APP_ENV"
echo "🌐 Port: $PORT"
echo "📅 Started at: $(date)"

# Azure Container Apps health check
echo "⚕️ Running pre-start health checks..."

# Set default values for Azure Container Apps
export PORT=${PORT:-8000}
export APP_ENV=${APP_ENV:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Validate environment variables
if [[ -z "$MLFLOW_TRACKING_URI" ]]; then
    echo "⚠️ MLFLOW_TRACKING_URI not set, using default"
    export MLFLOW_TRACKING_URI="http://localhost:5000"
fi

echo "🔍 Environment Configuration:"
echo "  - MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo "  - MODEL_NAME: ${MODEL_NAME:-patient_readmission_xgboost}"
echo "  - MODEL_STAGE: ${MODEL_STAGE:-Production}"
echo "  - LOG_LEVEL: $LOG_LEVEL"

# Check Python environment and critical imports
echo "🐍 Validating Python environment..."
python -c "
import sys
import os
print(f'Python version: {sys.version}')
print(f'Python path: {sys.path[0]}')
print(f'Working directory: {os.getcwd()}')

# Test critical imports
try:
    import mlflow
    print(f'✅ MLflow version: {mlflow.__version__}')
    
    import xgboost
    print(f'✅ XGBoost version: {xgboost.__version__}')
    
    import fastapi
    print(f'✅ FastAPI version: {fastapi.__version__}')
    
    import uvicorn
    print(f'✅ Uvicorn version: {uvicorn.__version__}')
    
    # Test our application imports
    import src.api.main
    print('✅ Application imports successful')
    
    # Test model loading capability
    from src.api.prediction import get_prediction_service
    prediction_service = get_prediction_service()
    print('✅ Prediction service initialized')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️ Warning during initialization: {e}')
    print('⚠️ API will start but may need model loading at runtime')
"

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set up logging for Azure
export PYTHONUNBUFFERED=1

# Azure Container Apps optimized startup
echo "🔄 Starting FastAPI server on port $PORT..."
echo "🔄 Using single worker (Azure Container Apps best practice)"

# Start the FastAPI application with Uvicorn
exec uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --access-log \
    --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
    --timeout-keep-alive 30 \
    --timeout-graceful-shutdown 10 \
    --limit-concurrency 100 \
    --limit-max-requests 1000