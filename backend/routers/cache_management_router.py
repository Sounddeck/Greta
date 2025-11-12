"""
Cache Management Router for CPAS5

Phase 3, Task 3.2: Implement Caching Layer
REST API endpoints for cache management and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any
import logging
from services.cache_service import CacheService, CacheConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/cache-mgmt", tags=["Cache Management"])

# Global cache service instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """Get cache service instance"""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.connect()
    
    return _cache_service


# ========================================================================
# CACHE HEALTH AND STATUS
# ========================================================================

@router.get("/health")
async def cache_health(cache_service: CacheService = Depends(get_cache_service)):
    """
    Check cache health status.
    
    Phase 3, Task 3.2: Add cache health checks and monitoring
    
    Example:
        GET /api/v1/cache-mgmt/health
    """
    try:
        health = await cache_service.health_check()
        return health
        
    except Exception as e:
        logger.error(f"❌ Cache health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def cache_metrics(cache_service: CacheService = Depends(get_cache_service)):
    """
    Get cache performance metrics.
    
    Phase 3, Task 3.2: Implement cache hit/miss ratio tracking
    
    Returns:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_ratio: Cache hit ratio percentage
        - sets: Number of cache sets
        - deletes: Number of cache deletes
        - invalidations: Number of cache invalidations
        - errors: Number of cache errors
        - average_response_time_ms: Average cache response time
        - cache_size_bytes: Cache size in bytes
        - cache_size_mb: Cache size in megabytes
    
    Example:
        GET /api/v1/cache-mgmt/metrics
    """
    try:
        metrics = await cache_service.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Cache metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/reset")
async def reset_cache_metrics(cache_service: CacheService = Depends(get_cache_service)):
    """
    Reset cache metrics.
    
    Example:
        POST /api/v1/cache-mgmt/metrics/reset
    """
    try:
        cache_service.reset_metrics()
        return {
            "success": True,
            "message": "Cache metrics reset successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Cache metrics reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# CACHE OPERATIONS
# ========================================================================

@router.get("/get/{key}")
async def get_cache_value(
    key: str,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Get value from cache by key.
    
    Example:
        GET /api/v1/cache-mgmt/get/stats:crm
    """
    try:
        value = await cache_service.get(key)
        
        if value is None:
            raise HTTPException(status_code=404, detail=f"Key not found: {key}")
        
        return {
            "key": key,
            "value": value,
            "exists": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Cache get error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{key}")
async def delete_cache_value(
    key: str,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Delete value from cache.
    
    Example:
        DELETE /api/v1/cache-mgmt/delete/stats:crm
    """
    try:
        success = await cache_service.delete(key)
        
        return {
            "success": success,
            "key": key,
            "deleted": success
        }
        
    except Exception as e:
        logger.error(f"❌ Cache delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Pattern to match keys (e.g., 'crm:*')"),
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Clear cache keys matching pattern.
    
    Phase 3, Task 3.2: Implement cache key patterns for easy bulk invalidation
    
    Example:
        DELETE /api/v1/cache-mgmt/clear?pattern=crm:*
    """
    try:
        count = await cache_service.clear(pattern)
        
        return {
            "success": True,
            "pattern": pattern or "ALL",
            "keys_deleted": count
        }
        
    except Exception as e:
        logger.error(f"❌ Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exists/{key}")
async def check_cache_exists(
    key: str,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Check if key exists in cache.
    
    Example:
        GET /api/v1/cache-mgmt/exists/stats:crm
    """
    try:
        exists = await cache_service.exists(key)
        
        return {
            "key": key,
            "exists": exists
        }
        
    except Exception as e:
        logger.error(f"❌ Cache exists error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ttl/{key}")
async def get_cache_ttl(
    key: str,
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Get remaining TTL for key.
    
    Returns:
        - ttl: Remaining TTL in seconds (-1 = no expiry, -2 = doesn't exist)
    
    Example:
        GET /api/v1/cache-mgmt/ttl/stats:crm
    """
    try:
        ttl = await cache_service.ttl(key)
        
        return {
            "key": key,
            "ttl": ttl,
            "exists": ttl >= -1,
            "has_expiry": ttl > 0
        }
        
    except Exception as e:
        logger.error(f"❌ Cache TTL error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# CACHE INVALIDATION
# ========================================================================

@router.post("/invalidate/pattern")
async def invalidate_by_pattern(
    pattern: str = Query(..., description="Pattern to match keys (e.g., 'crm:*')"),
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Invalidate cache keys matching pattern.
    
    Phase 3, Task 3.2: Implement cache key patterns for easy bulk invalidation
    
    Example:
        POST /api/v1/cache-mgmt/invalidate/pattern?pattern=crm:stats:*
    """
    try:
        count = await cache_service.invalidate_by_pattern(pattern)
        
        return {
            "success": True,
            "pattern": pattern,
            "keys_invalidated": count
        }
        
    except Exception as e:
        logger.error(f"❌ Cache invalidation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invalidate/event")
async def invalidate_on_event(
    event_type: str = Query(..., description="Event type (create, update, delete)"),
    entity_id: str = Query(..., description="Entity ID"),
    cache_service: CacheService = Depends(get_cache_service)
):
    """
    Invalidate cache based on event.
    
    Phase 3, Task 3.2: Add event-based cache invalidation
    
    Example:
        POST /api/v1/cache-mgmt/invalidate/event?event_type=update&entity_id=contact_123
    """
    try:
        await cache_service.invalidate_on_event(event_type, entity_id)
        
        return {
            "success": True,
            "event_type": event_type,
            "entity_id": entity_id
        }
        
    except Exception as e:
        logger.error(f"❌ Cache invalidation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# CACHE STATISTICS
# ========================================================================

@router.get("/stats")
async def cache_statistics(cache_service: CacheService = Depends(get_cache_service)):
    """
    Get comprehensive cache statistics.
    
    Phase 3, Task 3.2: Add cache size and memory usage monitoring
    
    Example:
        GET /api/v1/cache-mgmt/stats
    """
    try:
        metrics = await cache_service.get_metrics()
        health = await cache_service.health_check()
        
        return {
            "metrics": metrics,
            "health": health,
            "performance": {
                "hit_ratio": metrics["hit_ratio"],
                "average_response_time_ms": metrics["average_response_time_ms"],
                "cache_efficiency": "excellent" if metrics["hit_ratio"] > 80 else "good" if metrics["hit_ratio"] > 60 else "needs_improvement"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Cache statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

