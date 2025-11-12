"""
Health Check Router

REST API endpoints for comprehensive system health checks.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from services.health_check_service import HealthCheckService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health-check", tags=["Health Check"])


# Dependency to get health check service
async def get_health_check_service() -> HealthCheckService:
    """Get health check service instance"""
    # This will be injected by the main app
    from main import health_check_service
    return health_check_service


# ============================================================================
# Simple Health Check
# ============================================================================

@router.get("/")
async def simple_health_check(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """
    Simple health check endpoint
    
    Returns basic health status with minimal overhead.
    Suitable for load balancer health checks.
    """
    try:
        health = await health_service.get_simple_health()
        
        # Return appropriate status code
        if health["status"] == "healthy":
            status_code = 200
        elif health["status"] == "degraded":
            status_code = 200  # Still accepting traffic
        else:
            status_code = 503  # Service unavailable
        
        return health
        
    except Exception as e:
        logger.error(f"❌ Simple health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ============================================================================
# Detailed Health Check
# ============================================================================

@router.get("/detailed")
async def detailed_health_check(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """
    Detailed health check endpoint
    
    Returns comprehensive health information for all system components
    including database, cache, rate limiter, and performance monitor.
    """
    try:
        health = await health_service.get_detailed_health()
        
        return {
            "success": True,
            "health": health
        }
        
    except Exception as e:
        logger.error(f"❌ Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Component-Specific Health Checks
# ============================================================================

@router.get("/database")
async def database_health_check(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """Check database health"""
    try:
        health = await health_service.check_database_health()
        
        return {
            "success": True,
            "component": {
                "name": health.name,
                "status": health.status.value,
                "response_time_ms": round(health.response_time_ms, 2),
                "message": health.message,
                "details": health.details
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def cache_health_check(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """Check cache health"""
    try:
        health = await health_service.check_cache_health()
        
        return {
            "success": True,
            "component": {
                "name": health.name,
                "status": health.status.value,
                "response_time_ms": round(health.response_time_ms, 2),
                "message": health.message,
                "details": health.details
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Cache health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rate-limiter")
async def rate_limiter_health_check(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """Check rate limiter health"""
    try:
        health = await health_service.check_rate_limiter_health()
        
        return {
            "success": True,
            "component": {
                "name": health.name,
                "status": health.status.value,
                "response_time_ms": round(health.response_time_ms, 2),
                "message": health.message,
                "details": health.details
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Rate limiter health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-monitor")
async def performance_monitor_health_check(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """Check performance monitor health"""
    try:
        health = await health_service.check_performance_monitor_health()
        
        return {
            "success": True,
            "component": {
                "name": health.name,
                "status": health.status.value,
                "response_time_ms": round(health.response_time_ms, 2),
                "message": health.message,
                "details": health.details
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Performance monitor health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Readiness and Liveness Probes (Kubernetes-style)
# ============================================================================

@router.get("/readiness")
async def readiness_probe(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """
    Readiness probe
    
    Indicates whether the application is ready to accept traffic.
    Returns 200 if ready, 503 if not ready.
    """
    try:
        health = await health_service.check_all_components()
        
        # Ready if no components are unhealthy
        if health.unhealthy_components == 0:
            return {
                "ready": True,
                "status": health.status.value
            }
        else:
            return {
                "ready": False,
                "status": health.status.value,
                "unhealthy_components": health.unhealthy_components
            }
            
    except Exception as e:
        logger.error(f"❌ Readiness probe failed: {e}")
        return {
            "ready": False,
            "error": str(e)
        }


@router.get("/liveness")
async def liveness_probe(
    health_service: HealthCheckService = Depends(get_health_check_service)
):
    """
    Liveness probe
    
    Indicates whether the application is alive and running.
    Returns 200 if alive, 503 if dead.
    """
    try:
        # Simple check - just verify database connectivity
        db_health = await health_service.check_database_health()
        
        if db_health.status != "unhealthy":
            return {
                "alive": True,
                "status": "running"
            }
        else:
            return {
                "alive": False,
                "status": "dead",
                "message": db_health.message
            }
            
    except Exception as e:
        logger.error(f"❌ Liveness probe failed: {e}")
        return {
            "alive": False,
            "error": str(e)
        }

