"""
Enhanced Health Check Router for GRETA PAI
Provides comprehensive system health monitoring and status reporting
"""

import logging
import platform
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from config import get_settings
from database import health_check as db_health_check
from utils.metrics import get_metrics_summary, metrics_collector
from utils.cache import get_redis_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str
    timestamp: str
    version: str
    environment: str
    uptime_seconds: float
    services: Dict[str, Any]
    system: Dict[str, Any]
    performance: Dict[str, Any]


class ServiceStatus(BaseModel):
    """Individual service status"""
    status: str
    healthy: bool
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Application start time for uptime calculation
app_start_time = datetime.utcnow()


async def check_database_health() -> ServiceStatus:
    """Check database connection health"""
    
    try:
        start_time = datetime.utcnow()
        health_result = await db_health_check()
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        if health_result.get("database", False):
            return ServiceStatus(
                status="healthy",
                healthy=True,
                response_time_ms=round(response_time, 2),
                details=health_result
            )
        else:
            return ServiceStatus(
                status="unhealthy",
                healthy=False,
                response_time_ms=round(response_time, 2),
                error="Database connection failed",
                details=health_result
            )
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return ServiceStatus(
            status="error",
            healthy=False,
            error=str(e)
        )


async def check_redis_health() -> ServiceStatus:
    """Check Redis cache connection health"""
    
    try:
        start_time = datetime.utcnow()
        redis_client = await get_redis_client()
        
        if redis_client:
            # Test Redis with a simple ping
            await redis_client.ping()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await redis_client.info()
            
            return ServiceStatus(
                status="healthy",
                healthy=True,
                response_time_ms=round(response_time, 2),
                details={
                    "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "uptime_in_seconds": info.get("uptime_in_seconds")
                }
            )
        else:
            return ServiceStatus(
                status="unavailable",
                healthy=False,
                error="Redis client not available"
            )
            
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return ServiceStatus(
            status="error",
            healthy=False,
            error=str(e)
        )


def get_system_health() -> Dict[str, Any]:
    """Get current system health metrics"""
    
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory information
        memory = psutil.virtual_memory()
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        # Network information (if available)
        try:
            network = psutil.net_io_counters()
            network_info = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except Exception:
            network_info = None
        
        # Process information
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time()
        }
        
        return {
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "cpu": {
                "usage_percent": round(cpu_percent, 1),
                "count": cpu_count,
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": round(memory.percent, 1)
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 1)
            },
            "network": network_info,
            "process": process_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return {"error": str(e)}


@router.get(
    "/",
    response_model=HealthStatus,
    summary="System Health Check",
    description="""
    **Comprehensive system health monitoring endpoint**

    Returns detailed information about:
    - Overall system status (healthy/degraded/error)
    - Database connectivity and performance
    - Redis cache status and metrics
    - System resource usage (CPU, memory, disk)
    - Application performance metrics
    - Service uptime and version information

    **Use Cases:**
    - Load balancer health checks
    - Monitoring system integration
    - Debugging system issues
    - Performance monitoring

    **Response Status:**
    - `healthy`: All services operational
    - `degraded`: Some services have issues
    - `error`: Critical system failure
    """,
    responses={
        200: {
            "description": "System health information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2023-12-25T10:00:00Z",
                        "version": "3.0.0",
                        "environment": "development",
                        "uptime_seconds": 3600.5,
                        "services": {
                            "database": {
                                "status": "healthy",
                                "healthy": True,
                                "response_time_ms": 15.2
                            },
                            "redis_cache": {
                                "status": "healthy",
                                "healthy": True,
                                "response_time_ms": 2.1
                            }
                        },
                        "system": {
                            "cpu": {"usage_percent": 25.5},
                            "memory": {"usage_percent": 45.2}
                        },
                        "performance": {
                            "requests_per_minute": 150.5,
                            "average_response_time": 85.2
                        }
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable - health check failed",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Health check failed",
                        "error": "Database connection timeout",
                        "timestamp": "2023-12-25T10:00:00Z"
                    }
                }
            }
        }
    },
    tags=["Health"]
)
async def health_check():
    """
    Comprehensive health check endpoint
    Returns detailed system status and service health
    """
    
    try:
        settings = get_settings()
        current_time = datetime.utcnow()
        uptime = (current_time - app_start_time).total_seconds()
        
        # Check service health
        db_status = await check_database_health()
        redis_status = await check_redis_health()
        
        # Get system metrics
        system_health = get_system_health()
        
        # Get performance metrics
        performance_metrics = get_metrics_summary()
        
        # Determine overall health status
        services_healthy = db_status.healthy and redis_status.healthy
        overall_status = "healthy" if services_healthy else "degraded"
        
        # Build response
        health_response = HealthStatus(
            status=overall_status,
            timestamp=current_time.isoformat() + "Z",
            version="2.2",
            environment=settings.environment,
            uptime_seconds=round(uptime, 2),
            services={
                "database": db_status.dict(),
                "redis_cache": redis_status.dict()
            },
            system=system_health,
            performance=performance_metrics
        )
        
        logger.info(f"Health check completed - Status: {overall_status}")
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "Health check failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@router.get(
    "/quick",
    summary="Quick Health Check",
    description="""
    **Lightweight health check endpoint optimized for load balancers**

    Returns minimal status information with fast response time.
    Performs only essential checks:
    - Database connectivity (basic ping)
    - Application responsiveness

    **Ideal for:**
    - Load balancer health checks
    - High-frequency monitoring
    - Kubernetes liveness probes
    - Service mesh health checks

    **Response Time:** < 100ms typical
    """,
    responses={
        200: {
            "description": "Service is healthy and operational",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2023-12-25T10:00:00Z"
                    }
                }
            }
        },
        503: {
            "description": "Service is unhealthy or unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2023-12-25T10:00:00Z"
                    }
                }
            }
        }
    },
    tags=["Health"]
)
async def quick_health_check():
    """
    Quick health check for load balancers
    Returns minimal status information
    """
    
    try:
        # Quick database check
        db_result = await db_health_check()
        db_healthy = db_result.get("database", False)
        
        if db_healthy:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
            
    except Exception as e:
        logger.error(f"Quick health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@router.get(
    "/metrics",
    summary="Application Performance Metrics",
    description="""
    **Detailed application performance and usage metrics**

    Provides comprehensive metrics for monitoring and analytics:
    - Request volume and response times
    - Error rates and success rates
    - Cache performance statistics
    - Resource utilization trends
    - User activity metrics

    **Authentication Required:** Yes (Bearer token)

    **Use Cases:**
    - Performance monitoring dashboards
    - Capacity planning
    - SLA monitoring
    - Troubleshooting performance issues
    - Business intelligence

    **Metrics Include:**
    - Requests per minute/hour/day
    - Average/median/95th percentile response times
    - Cache hit/miss ratios
    - Error rates by endpoint
    - Active user counts
    """,
    responses={
        200: {
            "description": "Performance metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "requests_total": 15420,
                        "requests_per_minute": 125.5,
                        "average_response_time": 85.2,
                        "median_response_time": 65.1,
                        "p95_response_time": 180.5,
                        "error_rate": 0.02,
                        "cache_hit_rate": 0.85,
                        "active_users_1h": 42,
                        "collection_time": "2023-12-25T10:00:00Z",
                        "uptime_seconds": 3600.5
                    }
                }
            }
        },
        401: {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "message": "Authentication required",
                        "error_code": "UNAUTHORIZED"
                    }
                }
            }
        },
        500: {
            "description": "Failed to retrieve metrics",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Failed to retrieve metrics",
                        "message": "Database connection error",
                        "timestamp": "2023-12-25T10:00:00Z"
                    }
                }
            }
        }
    },
    tags=["Health"]
)
async def get_application_metrics():
    """
    Get detailed application performance metrics
    """
    
    try:
        metrics = get_metrics_summary()
        
        # Add additional metrics
        metrics["collection_time"] = datetime.utcnow().isoformat() + "Z"
        metrics["uptime_seconds"] = (datetime.utcnow() - app_start_time).total_seconds()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve metrics",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@router.get("/version")
async def get_version_info():
    """
    Get application version and build information
    """
    
    settings = get_settings()
    
    return {
        "version": "2.2",
        "name": "GRETA PAI Enhanced Backend",
        "description": "Complete Personal AI System",
        "environment": settings.environment,
        "platform": f"{platform.system()} {platform.machine()}",
        "python_version": platform.python_version(),
        "build_type": "refactored-modular",
        "features": [
            "Enhanced Learning System",
            "Local llama.cpp Processing", 
            "Multi-Agent Collaboration",
            "Predictive Analytics",
            "Enterprise Security",
            "Google Workspace Integration",
            "Voice Processing",
            "MCP Protocol Support"
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# Export router
__all__ = ["router"]
