"""
Performance Monitoring Router for GRETA PAI

This router provides API endpoints for performance monitoring, profiling,
and optimization management.

Endpoints:
    GET /performance/metrics - Get current performance metrics
    GET /performance/slow-queries - Get slow queries
    GET /performance/profiling - Get profiling data
    POST /performance/optimize-indexes - Optimize database indexes
    GET /performance/compression-stats - Get compression statistics
    POST /performance/clear-metrics - Clear performance metrics

Example:
    Get performance metrics:
        >>> import requests
        >>> response = requests.get("http://localhost:8000/api/v1/performance/metrics")
        >>> metrics = response.json()["data"]
        >>> print(f"Average response time: {metrics['average_response_time']}ms")

    Optimize indexes:
        >>> response = requests.post("http://localhost:8000/api/v1/performance/optimize-indexes")
        >>> print(response.json())
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from models.base_models import StandardResponse
from services.performance_service import PerformanceService, PerformanceMetrics, QueryMetrics
from utils.custom_exceptions import DatabaseError, ConfigurationError
from database import get_database
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/performance",
    tags=["Performance Monitoring"],
    responses={
        404: {"description": "Performance data not found"},
        500: {"description": "Performance operation failed"}
    }
)


# Service instance (will be initialized on startup)
_performance_service: Optional[PerformanceService] = None


async def get_performance_service(db: AsyncIOMotorDatabase = Depends(get_database)) -> PerformanceService:
    """Get or create performance service instance"""
    global _performance_service
    
    if _performance_service is None:
        _performance_service = PerformanceService(database=db)
        await _performance_service.initialize()
    
    return _performance_service


@router.get("/metrics", response_model=StandardResponse)
async def get_performance_metrics(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Get current performance metrics.
    
    Returns comprehensive performance metrics including response times,
    query performance, compression statistics, and resource usage.
    
    Returns:
        StandardResponse: Performance metrics data
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/performance/metrics"
        ```
        
        Response:
        ```json
        {
            "success": true,
            "message": "Performance metrics retrieved successfully",
            "data": {
                "total_requests": 1234,
                "average_response_time": 45.2,
                "p95_response_time": 120.5,
                "p99_response_time": 250.3,
                "slow_queries_count": 5,
                "cache_hit_ratio": 85.5,
                "compression_ratio": 3.2,
                "total_bytes_compressed": 1048576
            }
        }
        ```
    """
    try:
        metrics = await performance_service.get_performance_metrics()
        
        return StandardResponse(
            success=True,
            message="Performance metrics retrieved successfully",
            data={
                "total_requests": metrics.total_requests,
                "average_response_time": metrics.average_response_time,
                "p95_response_time": metrics.p95_response_time,
                "p99_response_time": metrics.p99_response_time,
                "slow_queries_count": metrics.slow_queries_count,
                "cache_hit_ratio": metrics.cache_hit_ratio,
                "compression_ratio": metrics.compression_ratio,
                "total_bytes_compressed": metrics.total_bytes_compressed,
                "active_connections": metrics.active_connections,
                "last_updated": metrics.last_updated.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/slow-queries", response_model=StandardResponse)
async def get_slow_queries(
    limit: int = Query(50, description="Maximum number of slow queries to return", ge=1, le=500),
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Get slow queries exceeding the performance threshold.
    
    Returns a list of queries that took longer than the configured
    slow query threshold, sorted by execution time.
    
    Args:
        limit: Maximum number of slow queries to return
        performance_service: Performance service instance
        
    Returns:
        StandardResponse: List of slow queries
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/performance/slow-queries?limit=10"
        ```
        
        Response:
        ```json
        {
            "success": true,
            "message": "Found 5 slow queries",
            "data": {
                "slow_queries": [
                    {
                        "query_name": "find_users",
                        "execution_time": 2.5,
                        "collection": "users",
                        "result_count": 1000,
                        "index_used": false,
                        "timestamp": "2025-10-16T10:30:00Z"
                    }
                ],
                "threshold": 1.0
            }
        }
        ```
    """
    try:
        slow_queries = await performance_service.get_slow_queries(limit)
        
        return StandardResponse(
            success=True,
            message=f"Found {len(slow_queries)} slow queries",
            data={
                "slow_queries": [
                    {
                        "query_name": qm.query_name,
                        "execution_time": qm.execution_time,
                        "collection": qm.collection,
                        "result_count": qm.result_count,
                        "index_used": qm.index_used,
                        "timestamp": qm.timestamp.isoformat()
                    }
                    for qm in slow_queries
                ],
                "threshold": performance_service.slow_query_threshold,
                "total_count": len(slow_queries)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get slow queries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve slow queries"
        )


@router.get("/profiling", response_model=StandardResponse)
async def get_profiling_data(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Get profiling data for all operations.
    
    Returns detailed profiling statistics for all profiled operations
    including average, min, max, and percentile execution times.
    
    Returns:
        StandardResponse: Profiling data
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/performance/profiling"
        ```
        
        Response:
        ```json
        {
            "success": true,
            "message": "Profiling data retrieved successfully",
            "data": {
                "user_query": {
                    "count": 100,
                    "average": 0.045,
                    "min": 0.012,
                    "max": 0.250,
                    "p95": 0.120,
                    "p99": 0.200
                }
            }
        }
        ```
    """
    try:
        profiling_data = await performance_service.get_profiling_data()
        
        return StandardResponse(
            success=True,
            message="Profiling data retrieved successfully",
            data=profiling_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get profiling data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profiling data"
        )


@router.post("/optimize-indexes", response_model=StandardResponse)
async def optimize_indexes(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Optimize database indexes based on query patterns.
    
    Analyzes query patterns and creates/optimizes database indexes
    for improved query performance.
    
    Returns:
        StandardResponse: Optimization results
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/performance/optimize-indexes"
        ```
        
        Response:
        ```json
        {
            "success": true,
            "message": "Database indexes optimized successfully",
            "data": {
                "indexes_optimized": true,
                "collections_analyzed": 5,
                "total_queries_analyzed": 1234
            }
        }
        ```
    """
    try:
        result = await performance_service.optimize_indexes()
        
        if result.get("success"):
            return StandardResponse(
                success=True,
                message="Database indexes optimized successfully",
                data=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Index optimization failed")
            )
        
    except DatabaseError as e:
        logger.error(f"Index optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during index optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Index optimization failed"
        )


@router.get("/compression-stats", response_model=StandardResponse)
async def get_compression_stats(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Get compression statistics.
    
    Returns statistics about response compression including
    compression ratios and bytes saved.
    
    Returns:
        StandardResponse: Compression statistics
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/performance/compression-stats"
        ```
        
        Response:
        ```json
        {
            "success": true,
            "message": "Compression statistics retrieved successfully",
            "data": {
                "compression_ratio": 3.2,
                "total_bytes_compressed": 1048576,
                "bytes_saved": 786432,
                "compression_enabled": true
            }
        }
        ```
    """
    try:
        metrics = await performance_service.get_performance_metrics()
        
        bytes_saved = metrics.total_bytes_compressed
        
        return StandardResponse(
            success=True,
            message="Compression statistics retrieved successfully",
            data={
                "compression_ratio": metrics.compression_ratio,
                "total_bytes_compressed": metrics.total_bytes_compressed,
                "bytes_saved": bytes_saved,
                "compression_enabled": performance_service.compression_enabled,
                "compression_threshold": performance_service.compression_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get compression stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compression statistics"
        )


@router.post("/clear-metrics", response_model=StandardResponse)
async def clear_metrics(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Clear performance metrics.
    
    Resets all performance metrics and profiling data.
    Useful for starting fresh measurements.
    
    Returns:
        StandardResponse: Clear operation result
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/performance/clear-metrics"
        ```
    """
    try:
        # Clear metrics
        performance_service.metrics = PerformanceMetrics()
        performance_service.query_metrics = []
        performance_service.response_times = []
        performance_service._profiling_data = {}
        
        return StandardResponse(
            success=True,
            message="Performance metrics cleared successfully",
            data={"cleared": True}
        )
        
    except Exception as e:
        logger.error(f"Failed to clear metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear performance metrics"
        )


@router.get("/health", response_model=StandardResponse)
async def performance_health_check(
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """
    Check performance service health.
    
    Returns health status of the performance monitoring service
    including current metrics and service status.
    
    Returns:
        StandardResponse: Health check results
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/performance/health"
        ```
    """
    try:
        metrics = await performance_service.get_performance_metrics()
        
        # Determine health status
        health_status = "healthy"
        warnings = []
        
        if metrics.slow_queries_count > 10:
            warnings.append(f"High number of slow queries: {metrics.slow_queries_count}")
            health_status = "degraded"
        
        if metrics.average_response_time > 500:  # 500ms
            warnings.append(f"High average response time: {metrics.average_response_time:.2f}ms")
            health_status = "degraded"
        
        if metrics.p99_response_time > 2000:  # 2 seconds
            warnings.append(f"High P99 response time: {metrics.p99_response_time:.2f}ms")
            health_status = "degraded"
        
        return StandardResponse(
            success=True,
            message=f"Performance service is {health_status}",
            data={
                "status": health_status,
                "warnings": warnings,
                "metrics_summary": {
                    "total_requests": metrics.total_requests,
                    "average_response_time": metrics.average_response_time,
                    "slow_queries_count": metrics.slow_queries_count
                },
                "profiling_enabled": performance_service.profiling_enabled,
                "compression_enabled": performance_service.compression_enabled
            }
        )
        
    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Performance health check failed"
        )
