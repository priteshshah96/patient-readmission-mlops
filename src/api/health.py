import time
import psutil
import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from typing import Dict, Any
from src.api.models import HealthResponse
from src.api.prediction import get_prediction_service, PredictionService
from src.api.config import get_settings, Settings
logger = logging.getLogger(__name__)

# Health check router
health_router = APIRouter(prefix="/health", tags=["health"])

# Track service start time
SERVICE_START_TIME = time.time()


@health_router.get("/", response_model=HealthResponse)
async def health_check(
    prediction_service: PredictionService = Depends(get_prediction_service),
    settings: Settings = Depends(get_settings)
) -> HealthResponse:
    """
    Basic health check endpoint.
    Returns service status and model loading status.
    """
    try:
        # Calculate uptime
        uptime_seconds = time.time() - SERVICE_START_TIME
        
        # Get memory usage
        memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Check model status
        model_loaded = prediction_service.is_model_loaded()
        model_version = prediction_service.model_version if model_loaded else None
        
        status = "healthy" if model_loaded else "degraded"
        
        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            model_version=model_version,
            uptime_seconds=uptime_seconds,
            memory_usage_mb=memory_usage_mb
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version=None,
            uptime_seconds=time.time() - SERVICE_START_TIME,
            memory_usage_mb=0.0
        )


@health_router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    prediction_service: PredictionService = Depends(get_prediction_service),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Detailed health check with comprehensive system information.
    """
    try:
        # Basic health info
        uptime_seconds = time.time() - SERVICE_START_TIME
        memory_info = psutil.Process().memory_info()
        
        # Model information
        model_info = prediction_service.get_model_info()
        
        # System information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        return {
            "service": {
                "status": "healthy" if prediction_service.is_model_loaded() else "degraded",
                "uptime_seconds": uptime_seconds,
                "uptime_human": f"{uptime_seconds // 3600:.0f}h {(uptime_seconds % 3600) // 60:.0f}m",
                "version": settings.app_version,
                "environment": "production" if not settings.debug else "development"
            },
            "model": {
                "loaded": prediction_service.is_model_loaded(),
                "info": model_info
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "percent": memory_percent
                },
                "disk_usage_percent": disk_usage
            },
            "configuration": {
                "model_name": settings.model_name,
                "model_stage": settings.model_stage,
                "batch_size_limit": settings.batch_size_limit,
                "prediction_threshold": settings.prediction_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "service": {
                "status": "unhealthy",
                "error": str(e),
                "uptime_seconds": time.time() - SERVICE_START_TIME
            },
            "timestamp": datetime.now().isoformat()
        }


@health_router.get("/ready")
async def readiness_check(
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> Dict[str, Any]:
    """
    Kubernetes-style readiness check.
    Returns 200 if service is ready to serve requests.
    """
    if prediction_service.is_model_loaded():
        return {
            "status": "ready",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Return 503 Service Unavailable if not ready
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "model_loaded": False,
                "message": "Model not loaded yet",
                "timestamp": datetime.now().isoformat()
            }
        )


@health_router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes-style liveness check.
    Returns 200 if service is alive (even if not ready).
    """
    return {
        "status": "alive",
        "uptime_seconds": time.time() - SERVICE_START_TIME,
        "timestamp": datetime.now().isoformat()
    }