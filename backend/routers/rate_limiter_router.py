"""
Rate Limiter Router for CPAS5

Phase 3, Task 3.3: Add Rate Limiting
REST API endpoints for rate limit management and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import logging
from services.rate_limiter_service import RateLimiterService, RateLimitConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rate-limiter", tags=["Rate Limiter Management"])

# Global rate limiter service instance
_rate_limiter_service: Optional[RateLimiterService] = None


async def get_rate_limiter_service() -> RateLimiterService:
    """Get rate limiter service instance"""
    global _rate_limiter_service
    
    if _rate_limiter_service is None:
        _rate_limiter_service = RateLimiterService()
        await _rate_limiter_service.connect()
    
    return _rate_limiter_service


# ========================================================================
# REQUEST MODELS
# ========================================================================

class RateLimitCheckRequest(BaseModel):
    """Request model for checking rate limit"""
    identifier: str
    limit: int
    window: int = 60
    is_admin: bool = False
    ip: Optional[str] = None


class WhitelistRequest(BaseModel):
    """Request model for whitelist management"""
    ip: str
    reason: Optional[str] = None


# ========================================================================
# HEALTH AND STATUS
# ========================================================================

@router.get("/health")
async def rate_limiter_health(
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Check rate limiter health status.
    
    Phase 3, Task 3.3: Add rate limiter health checks
    
    Example:
        GET /api/v1/rate-limiter/health
    """
    try:
        health = await rate_limiter.health_check()
        return health
        
    except Exception as e:
        logger.error(f"❌ Rate limiter health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def rate_limiter_metrics(
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Get rate limiter metrics.
    
    Phase 3, Task 3.3: Track rate limit violations and usage
    
    Returns:
        - total_requests: Total requests processed
        - allowed_requests: Requests allowed
        - blocked_requests: Requests blocked
        - block_ratio: Percentage of blocked requests
        - bypass_count: Number of bypassed requests
        - violations_by_ip: Top 10 violators by IP
        - violations_by_user: Top 10 violators by user
    
    Example:
        GET /api/v1/rate-limiter/metrics
    """
    try:
        metrics = await rate_limiter.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Rate limiter metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/reset")
async def reset_rate_limiter_metrics(
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Reset rate limiter metrics.
    
    Example:
        POST /api/v1/rate-limiter/metrics/reset
    """
    try:
        rate_limiter.reset_metrics()
        return {
            "success": True,
            "message": "Rate limiter metrics reset successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Rate limiter metrics reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# RATE LIMIT OPERATIONS
# ========================================================================

@router.post("/check")
async def check_rate_limit(
    request: RateLimitCheckRequest,
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Check if request is within rate limit.
    
    Phase 3, Task 3.3: Manual rate limit checking
    
    Example:
        POST /api/v1/rate-limiter/check
        Body: {
            "identifier": "user_123",
            "limit": 100,
            "window": 60,
            "is_admin": false
        }
    """
    try:
        result = await rate_limiter.check_rate_limit(
            identifier=request.identifier,
            limit=request.limit,
            window=request.window,
            is_admin=request.is_admin,
            ip=request.ip
        )
        
        return {
            "allowed": result.allowed,
            "limit": result.limit,
            "remaining": result.remaining,
            "reset_at": result.reset_at,
            "retry_after": result.retry_after
        }
        
    except Exception as e:
        logger.error(f"❌ Rate limit check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# WHITELIST MANAGEMENT
# ========================================================================

@router.get("/whitelist")
async def get_whitelist(
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Get whitelisted IPs.
    
    Phase 3, Task 3.3: IP whitelist management
    
    Example:
        GET /api/v1/rate-limiter/whitelist
    """
    try:
        return {
            "whitelisted_ips": rate_limiter.config.whitelisted_ips
        }
        
    except Exception as e:
        logger.error(f"❌ Get whitelist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whitelist/add")
async def add_to_whitelist(
    request: WhitelistRequest,
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Add IP to whitelist.
    
    Phase 3, Task 3.3: Add IP to whitelist
    
    Example:
        POST /api/v1/rate-limiter/whitelist/add
        Body: {
            "ip": "192.168.1.100",
            "reason": "Internal server"
        }
    """
    try:
        if request.ip not in rate_limiter.config.whitelisted_ips:
            rate_limiter.config.whitelisted_ips.append(request.ip)
            logger.info(f"✅ Added {request.ip} to whitelist: {request.reason}")
        
        return {
            "success": True,
            "ip": request.ip,
            "reason": request.reason,
            "whitelisted_ips": rate_limiter.config.whitelisted_ips
        }
        
    except Exception as e:
        logger.error(f"❌ Add to whitelist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whitelist/remove")
async def remove_from_whitelist(
    ip: str = Query(..., description="IP address to remove"),
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Remove IP from whitelist.
    
    Phase 3, Task 3.3: Remove IP from whitelist
    
    Example:
        POST /api/v1/rate-limiter/whitelist/remove?ip=192.168.1.100
    """
    try:
        if ip in rate_limiter.config.whitelisted_ips:
            rate_limiter.config.whitelisted_ips.remove(ip)
            logger.info(f"✅ Removed {ip} from whitelist")
        
        return {
            "success": True,
            "ip": ip,
            "whitelisted_ips": rate_limiter.config.whitelisted_ips
        }
        
    except Exception as e:
        logger.error(f"❌ Remove from whitelist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# STATISTICS AND ANALYTICS
# ========================================================================

@router.get("/stats")
async def rate_limiter_statistics(
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Get comprehensive rate limiter statistics.
    
    Phase 3, Task 3.3: Rate limit analytics
    
    Example:
        GET /api/v1/rate-limiter/stats
    """
    try:
        metrics = await rate_limiter.get_metrics()
        health = await rate_limiter.health_check()
        
        return {
            "metrics": metrics,
            "health": health,
            "config": {
                "default_limits": {
                    "public": f"{rate_limiter.config.public_per_minute} req/min",
                    "authenticated": f"{rate_limiter.config.authenticated_per_minute} req/min",
                    "admin": f"{rate_limiter.config.admin_per_minute} req/min",
                    "heavy_ops": f"{rate_limiter.config.heavy_ops_per_minute} req/min",
                    "write_ops": f"{rate_limiter.config.write_ops_per_minute} req/min"
                },
                "whitelisted_ips": rate_limiter.config.whitelisted_ips
            },
            "performance": {
                "block_ratio": metrics["block_ratio"],
                "efficiency": "excellent" if metrics["block_ratio"] < 5 else "good" if metrics["block_ratio"] < 15 else "needs_attention"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Rate limiter statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violators")
async def get_top_violators(
    limit: int = Query(10, description="Number of top violators to return"),
    rate_limiter: RateLimiterService = Depends(get_rate_limiter_service)
):
    """
    Get top rate limit violators.
    
    Phase 3, Task 3.3: Track top violators
    
    Example:
        GET /api/v1/rate-limiter/violators?limit=10
    """
    try:
        metrics = await rate_limiter.get_metrics()
        
        # Combine IP and user violations
        all_violators = []
        
        for ip, count in metrics["violations_by_ip"].items():
            all_violators.append({
                "identifier": ip,
                "type": "ip",
                "violations": count
            })
        
        for user, count in metrics["violations_by_user"].items():
            all_violators.append({
                "identifier": user,
                "type": "user",
                "violations": count
            })
        
        # Sort by violations and limit
        all_violators.sort(key=lambda x: x["violations"], reverse=True)
        top_violators = all_violators[:limit]
        
        return {
            "top_violators": top_violators,
            "total_unique_violators": len(all_violators)
        }
        
    except Exception as e:
        logger.error(f"❌ Get top violators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

