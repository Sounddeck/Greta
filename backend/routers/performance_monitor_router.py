"""
Performance Monitor Router

REST API endpoints for performance monitoring and metrics (Phase 3, Task 3.4).
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from services.performance_monitor_service import PerformanceMonitorService
from database import get_database, Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance-monitor", tags=["Performance Monitoring"])


# Dependency to get performance monitor service
async def get_performance_monitor() -> PerformanceMonitorService:
    """Get performance monitor service instance"""
    # This will be injected by the main app
    from main import performance_monitor_service
    return performance_monitor_service


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def performance_health(
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Check performance monitoring service health"""
    try:
        metrics = performance_monitor.get_metrics()
        
        return {
            "success": True,
            "status": "healthy",
            "is_monitoring": metrics["is_monitoring"],
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"❌ Error checking performance health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Performance Summary
# ============================================================================

@router.get("/summary")
async def get_performance_summary(
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Get overall performance summary"""
    try:
        summary = await performance_monitor.get_performance_summary()
        
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"❌ Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoint Statistics
# ============================================================================

@router.get("/endpoints")
async def get_endpoint_stats(
    endpoint: Optional[str] = Query(None, description="Filter by endpoint path"),
    method: Optional[str] = Query(None, description="Filter by HTTP method"),
    sort_by: str = Query("avg_response_time_ms", description="Sort field"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Get endpoint performance statistics"""
    try:
        stats = await performance_monitor.get_endpoint_stats(
            endpoint=endpoint,
            method=method,
            sort_by=sort_by,
            limit=limit
        )
        
        return {
            "success": True,
            "count": len(stats),
            "endpoints": stats
        }
    except Exception as e:
        logger.error(f"❌ Error getting endpoint stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Slow Requests
# ============================================================================

@router.get("/slow-requests")
async def get_slow_requests(
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    threshold_ms: Optional[float] = Query(None, description="Response time threshold in ms"),
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Get slow requests"""
    try:
        slow_requests = await performance_monitor.get_slow_requests(
            limit=limit,
            threshold_ms=threshold_ms
        )
        
        return {
            "success": True,
            "count": len(slow_requests),
            "slow_requests": slow_requests
        }
    except Exception as e:
        logger.error(f"❌ Error getting slow requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error Requests
# ============================================================================

@router.get("/error-requests")
async def get_error_requests(
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    status_code: Optional[int] = Query(None, description="Filter by status code"),
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Get error requests"""
    try:
        error_requests = await performance_monitor.get_error_requests(
            limit=limit,
            status_code=status_code
        )
        
        return {
            "success": True,
            "count": len(error_requests),
            "error_requests": error_requests
        }
    except Exception as e:
        logger.error(f"❌ Error getting error requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# System Metrics
# ============================================================================

@router.get("/system")
async def get_system_metrics(
    hours: int = Query(1, ge=1, le=24, description="Hours of history"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Get system resource metrics"""
    try:
        metrics = await performance_monitor.get_system_metrics(
            hours=hours,
            limit=limit
        )
        
        return {
            "success": True,
            "count": len(metrics),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"❌ Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/current")
async def get_current_system_metrics(
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Get current system resource metrics"""
    try:
        metrics = await performance_monitor.collect_system_metrics()
        
        return {
            "success": True,
            "metrics": {
                "timestamp": metrics.timestamp,
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_used_mb": metrics.memory_used_mb,
                "memory_available_mb": metrics.memory_available_mb,
                "disk_usage_percent": metrics.disk_usage_percent,
                "disk_read_mb": metrics.disk_read_mb,
                "disk_write_mb": metrics.disk_write_mb,
                "network_sent_mb": metrics.network_sent_mb,
                "network_recv_mb": metrics.network_recv_mb,
                "active_connections": metrics.active_connections,
                "thread_count": metrics.thread_count
            }
        }
    except Exception as e:
        logger.error(f"❌ Error getting current system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Metrics Management
# ============================================================================

@router.post("/cleanup")
async def cleanup_old_metrics(
    hours: int = Query(24, ge=1, le=168, description="Delete metrics older than this many hours"),
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Clean up old performance metrics"""
    try:
        await performance_monitor.cleanup_old_metrics(hours=hours)
        
        return {
            "success": True,
            "message": f"Cleaned up metrics older than {hours} hours"
        }
    except Exception as e:
        logger.error(f"❌ Error cleaning up metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_service_metrics(
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Get performance monitoring service metrics"""
    try:
        metrics = performance_monitor.get_metrics()
        
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"❌ Error getting service metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Monitoring Control
# ============================================================================

@router.post("/monitoring/start")
async def start_monitoring(
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Start system monitoring"""
    try:
        await performance_monitor.start_system_monitoring()
        
        return {
            "success": True,
            "message": "System monitoring started"
        }
    except Exception as e:
        logger.error(f"❌ Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/stop")
async def stop_monitoring(
    performance_monitor: PerformanceMonitorService = Depends(get_performance_monitor)
):
    """Stop system monitoring"""
    try:
        await performance_monitor.stop_system_monitoring()
        
        return {
            "success": True,
            "message": "System monitoring stopped"
        }
    except Exception as e:
        logger.error(f"❌ Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

