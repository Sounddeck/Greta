"""
Cache Management Router for GRETA PAI

This router provides API endpoints for advanced cache management including
cache warming, invalidation, analytics, and monitoring.

Endpoints:
    POST /cache/warm - Trigger cache warming
    POST /cache/invalidate - Invalidate cache patterns
    GET /cache/analytics - Get cache analytics
    GET /cache/recommendations - Get warming recommendations
    GET /cache/stats - Get cache statistics
    POST /cache/invalidate/user/{user_id} - Invalidate user cache
    GET /cache/size - Get cache size information
    DELETE /cache/cleanup - Cleanup old analytics data

Example:
    Warm cache:
        >>> import requests
        >>> response = requests.post(
        ...     "http://localhost:8000/api/v1/cache/warm",
        ...     json={"strategy": "user_data", "limit": 100}
        ... )
        >>> print(response.json())

    Get cache analytics:
        >>> response = requests.get("http://localhost:8000/api/v1/cache/analytics")
        >>> analytics = response.json()["data"]
        >>> print(f"Hit ratio: {analytics['hit_ratio']}%")
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from models.base_models import StandardResponse
from services.cache_warming_service import CacheWarmingService, CacheAnalytics
from services.cache_invalidation_service import CacheInvalidationService, InvalidationStrategy
from utils.custom_exceptions import CacheError, InvalidationError
from database import get_database
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cache",
    tags=["Cache Management"],
    responses={
        404: {"description": "Cache operation not found"},
        500: {"description": "Cache operation failed"}
    }
)


# Request/Response Models
class CacheWarmRequest(BaseModel):
    """Cache warming request model"""
    strategy: str = Field(..., description="Warming strategy name")
    limit: Optional[int] = Field(100, description="Maximum items to warm", ge=1, le=1000)
    user_ids: Optional[List[str]] = Field(None, description="Specific user IDs to warm")
    endpoints: Optional[List[str]] = Field(None, description="Specific endpoints to warm")


class CacheInvalidateRequest(BaseModel):
    """Cache invalidation request model"""
    pattern: str = Field(..., description="Cache key pattern to invalidate")
    warm_after: bool = Field(False, description="Warm cache after invalidation")
    strategy: InvalidationStrategy = Field(
        InvalidationStrategy.IMMEDIATE,
        description="Invalidation strategy"
    )


class EventInvalidationRequest(BaseModel):
    """Event-based invalidation request model"""
    event: str = Field(..., description="Event name")
    context: Dict[str, Any] = Field(default_factory=dict, description="Event context")


# Service instances (will be initialized on startup)
_warming_service: Optional[CacheWarmingService] = None
_invalidation_service: Optional[CacheInvalidationService] = None


async def get_warming_service(db: AsyncIOMotorDatabase = Depends(get_database)) -> CacheWarmingService:
    """Get or create cache warming service instance"""
    global _warming_service
    
    if _warming_service is None:
        _warming_service = CacheWarmingService(database=db)
        await _warming_service.initialize()
    
    return _warming_service


async def get_invalidation_service() -> CacheInvalidationService:
    """Get or create cache invalidation service instance"""
    global _invalidation_service
    
    if _invalidation_service is None:
        _invalidation_service = CacheInvalidationService()
        await _invalidation_service.initialize()
    
    return _invalidation_service


@router.post("/warm", response_model=StandardResponse)
async def warm_cache(
    request: CacheWarmRequest,
    warming_service: CacheWarmingService = Depends(get_warming_service)
):
    """
    Trigger cache warming operation.
    
    Executes a cache warming strategy to preload frequently accessed data
    into Redis cache, improving response times.
    
    Args:
        request: Cache warming request with strategy and parameters
        warming_service: Cache warming service instance
        
    Returns:
        StandardResponse: Warming operation results
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/cache/warm" \\
             -H "Content-Type: application/json" \\
             -d '{"strategy": "user_data", "limit": 100}'
        ```
    """
    try:
        # Prepare strategy parameters
        kwargs = {"limit": request.limit}
        
        if request.user_ids:
            kwargs["user_ids"] = request.user_ids
        
        if request.endpoints:
            kwargs["endpoints"] = request.endpoints
        
        # Execute warming
        result = await warming_service.warm_cache(request.strategy, **kwargs)
        
        return StandardResponse(
            success=True,
            message=f"Cache warming '{request.strategy}' completed successfully",
            data=result
        )
        
    except CacheError as e:
        logger.error(f"Cache warming failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during cache warming: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache warming operation failed"
        )


@router.post("/warm/intelligent", response_model=StandardResponse)
async def start_intelligent_warming(
    warming_service: CacheWarmingService = Depends(get_warming_service)
):
    """
    Start intelligent cache warming based on analytics.
    
    Analyzes cache performance and automatically executes warming strategies
    based on recommendations and usage patterns.
    
    Returns:
        StandardResponse: Intelligent warming results
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/cache/warm/intelligent"
        ```
    """
    try:
        await warming_service.start_intelligent_warming()
        
        return StandardResponse(
            success=True,
            message="Intelligent cache warming completed successfully",
            data={"status": "completed"}
        )
        
    except CacheError as e:
        logger.error(f"Intelligent warming failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/invalidate", response_model=StandardResponse)
async def invalidate_cache(
    request: CacheInvalidateRequest,
    invalidation_service: CacheInvalidationService = Depends(get_invalidation_service)
):
    """
    Invalidate cache keys matching a pattern.
    
    Removes cache entries matching the specified pattern, optionally
    warming the cache afterward.
    
    Args:
        request: Cache invalidation request
        invalidation_service: Cache invalidation service instance
        
    Returns:
        StandardResponse: Invalidation operation results
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/cache/invalidate" \\
             -H "Content-Type: application/json" \\
             -d '{"pattern": "user:profile:*", "warm_after": true}'
        ```
    """
    try:
        result = await invalidation_service.invalidate_pattern(
            request.pattern,
            warm_after=request.warm_after
        )
        
        return StandardResponse(
            success=result.success,
            message=f"Invalidated {result.keys_invalidated} cache keys",
            data={
                "keys_invalidated": result.keys_invalidated,
                "patterns_processed": result.patterns_processed,
                "duration": result.duration,
                "strategy_used": result.strategy_used.value,
                "warm_operations": result.warm_operations
            }
        )
        
    except InvalidationError as e:
        logger.error(f"Cache invalidation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/invalidate/event", response_model=StandardResponse)
async def invalidate_by_event(
    request: EventInvalidationRequest,
    invalidation_service: CacheInvalidationService = Depends(get_invalidation_service)
):
    """
    Trigger cache invalidation based on an event.
    
    Executes registered invalidation rules for the specified event,
    invalidating relevant cache entries.
    
    Args:
        request: Event invalidation request
        invalidation_service: Cache invalidation service instance
        
    Returns:
        StandardResponse: Invalidation results
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/cache/invalidate/event" \\
             -H "Content-Type: application/json" \\
             -d '{"event": "user_updated", "context": {"user_id": "123"}}'
        ```
    """
    try:
        results = await invalidation_service.handle_event(request.event, request.context)
        
        total_invalidated = sum(r.keys_invalidated for r in results)
        successful = sum(1 for r in results if r.success)
        
        return StandardResponse(
            success=True,
            message=f"Event '{request.event}' processed: {total_invalidated} keys invalidated",
            data={
                "event": request.event,
                "total_invalidated": total_invalidated,
                "successful_operations": successful,
                "total_operations": len(results),
                "results": [
                    {
                        "success": r.success,
                        "keys_invalidated": r.keys_invalidated,
                        "strategy": r.strategy_used.value,
                        "error": r.error
                    }
                    for r in results
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Event-based invalidation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/invalidate/user/{user_id}", response_model=StandardResponse)
async def invalidate_user_cache(
    user_id: str,
    invalidation_service: CacheInvalidationService = Depends(get_invalidation_service)
):
    """
    Invalidate all cache entries for a specific user.
    
    Args:
        user_id: User identifier
        invalidation_service: Cache invalidation service instance
        
    Returns:
        StandardResponse: Invalidation results
        
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/cache/invalidate/user/user123"
        ```
    """
    try:
        results = await invalidation_service.invalidate_user_cache(user_id)
        
        return StandardResponse(
            success=True,
            message=f"User cache invalidated for user: {user_id}",
            data={"user_id": user_id, "results": results}
        )
        
    except Exception as e:
        logger.error(f"User cache invalidation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/analytics", response_model=StandardResponse)
async def get_cache_analytics(
    warming_service: CacheWarmingService = Depends(get_warming_service)
):
    """
    Get current cache analytics and performance metrics.
    
    Returns detailed cache performance information including hit ratios,
    response times, and usage statistics.
    
    Returns:
        StandardResponse: Cache analytics data
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/cache/analytics"
        ```
    """
    try:
        analytics = await warming_service.get_cache_analytics()
        
        return StandardResponse(
            success=True,
            message="Cache analytics retrieved successfully",
            data={
                "total_requests": analytics.total_requests,
                "cache_hits": analytics.cache_hits,
                "cache_misses": analytics.cache_misses,
                "hit_ratio": analytics.hit_ratio,
                "average_response_time": analytics.average_response_time,
                "total_size": analytics.total_size,
                "key_count": analytics.key_count,
                "warming_operations": analytics.warming_operations,
                "last_updated": analytics.last_updated.isoformat() if analytics.last_updated else None
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache analytics"
        )


@router.get("/recommendations", response_model=StandardResponse)
async def get_warming_recommendations(
    warming_service: CacheWarmingService = Depends(get_warming_service)
):
    """
    Get cache warming recommendations based on analytics.
    
    Analyzes cache performance and provides recommendations for
    improving cache hit ratios and performance.
    
    Returns:
        StandardResponse: Warming recommendations
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/cache/recommendations"
        ```
    """
    try:
        recommendations = await warming_service.get_warming_recommendations()
        
        return StandardResponse(
            success=True,
            message=f"Found {len(recommendations)} warming recommendations",
            data={
                "recommendations": [
                    {
                        "key_pattern": rec.key_pattern,
                        "priority": rec.priority,
                        "frequency": rec.frequency,
                        "reason": rec.reason,
                        "estimated_benefit": rec.estimated_benefit
                    }
                    for rec in recommendations
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve warming recommendations"
        )


@router.get("/stats", response_model=StandardResponse)
async def get_cache_stats(
    invalidation_service: CacheInvalidationService = Depends(get_invalidation_service)
):
    """
    Get cache invalidation statistics.
    
    Returns statistics about cache invalidation operations including
    success rates and performance metrics.
    
    Returns:
        StandardResponse: Invalidation statistics
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/cache/stats"
        ```
    """
    try:
        stats = await invalidation_service.get_invalidation_stats()
        
        return StandardResponse(
            success=True,
            message="Cache statistics retrieved successfully",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )


@router.get("/size", response_model=StandardResponse)
async def get_cache_size_info(
    warming_service: CacheWarmingService = Depends(get_warming_service)
):
    """
    Get detailed cache size information.
    
    Returns cache size breakdown by pattern and memory usage statistics.
    
    Returns:
        StandardResponse: Cache size information
        
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/cache/size"
        ```
    """
    try:
        size_info = await warming_service.get_cache_size_info()
        
        return StandardResponse(
            success=True,
            message="Cache size information retrieved successfully",
            data=size_info
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache size info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache size information"
        )


@router.delete("/cleanup", response_model=StandardResponse)
async def cleanup_old_analytics(
    days: int = Query(30, description="Number of days to retain analytics", ge=1, le=365),
    warming_service: CacheWarmingService = Depends(get_warming_service)
):
    """
    Clean up old analytics data.
    
    Removes analytics data older than the specified number of days.
    
    Args:
        days: Number of days to retain analytics data
        warming_service: Cache warming service instance
        
    Returns:
        StandardResponse: Cleanup results
        
    Example:
        ```bash
        curl -X DELETE "http://localhost:8000/api/v1/cache/cleanup?days=30"
        ```
    """
    try:
        cleaned_count = await warming_service.cleanup_expired_analytics(days)
        
        return StandardResponse(
            success=True,
            message=f"Cleaned up {cleaned_count} old analytics records",
            data={"cleaned_count": cleaned_count, "retention_days": days}
        )
        
    except Exception as e:
        logger.error(f"Failed to cleanup analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup analytics data"
        )
