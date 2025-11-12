"""
Rate Limiter Middleware for CPAS5

Phase 3, Task 3.3: Add Rate Limiting
Automatic rate limiting for API endpoints with configurable limits.
"""

import logging
import json
from typing import Optional, Callable, Set, Dict
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from services.rate_limiter_service import RateLimiterService, RateLimitResult
import time

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Rate Limiter Middleware for automatic API rate limiting.
    
    Features:
    - Per-user and per-IP rate limiting
    - Different limits for different endpoint types
    - Rate limit headers in responses
    - Admin bypass
    - Whitelist support
    """
    
    def __init__(
        self,
        app,
        rate_limiter: RateLimiterService,
        default_limit: int = 100,
        default_window: int = 60,
        admin_bypass: bool = True
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.default_limit = default_limit
        self.default_window = default_window
        self.admin_bypass = admin_bypass
        
        # Endpoint-specific rate limits (requests per minute)
        self.endpoint_limits = {
            # Public endpoints - 100 req/min
            "/api/v1/health": (100, 60),
            "/api/v1/status": (100, 60),
            
            # Heavy operations - 10 req/min
            "/api/v1/crm/stats": (10, 60),
            "/api/v1/ats/stats": (10, 60),
            "/api/v1/reports": (10, 60),
            "/api/v1/analytics": (10, 60),
            
            # Write operations - 100 req/min
            "/api/v1/crm/contacts": (100, 60),
            "/api/v1/crm/companies": (100, 60),
            "/api/v1/crm/deals": (100, 60),
            "/api/v1/ats/jobs": (100, 60),
            "/api/v1/ats/candidates": (100, 60),
            
            # Authenticated endpoints - 1000 req/min
            "/api/v1/user": (1000, 60),
            "/api/v1/profile": (1000, 60),
            
            # Admin endpoints - 10000 req/min (very high)
            "/api/v1/admin": (10000, 60),
        }
        
        # Exempt paths (no rate limiting)
        self.exempt_paths = {
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/static",
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded IP (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Get user ID from request headers or auth"""
        # Check for user ID in headers (set by auth middleware)
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey_{api_key[:8]}"
        
        return None
    
    def _is_admin(self, request: Request) -> bool:
        """Check if user is admin"""
        if not self.admin_bypass:
            return False
        
        # Check for admin role in headers
        user_role = request.headers.get("X-User-Role")
        return user_role == "admin"
    
    def _get_rate_limit(self, path: str) -> tuple[int, int]:
        """Get rate limit for path (limit, window)"""
        # Check exact match
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check prefix match
        for endpoint_path, (limit, window) in self.endpoint_limits.items():
            if path.startswith(endpoint_path):
                return limit, window
        
        # Default limit
        return self.default_limit, self.default_window
    
    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from rate limiting"""
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        return False
    
    def _add_rate_limit_headers(
        self,
        response: Response,
        result: RateLimitResult
    ):
        """Add rate limit headers to response"""
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_at)
        
        if result.retry_after:
            response.headers["Retry-After"] = str(result.retry_after)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        path = request.url.path
        
        # Skip rate limiting for exempt paths
        if self._is_exempt(path):
            return await call_next(request)
        
        # Get client information
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        is_admin = self._is_admin(request)
        
        # Determine identifier (prefer user ID over IP)
        identifier = user_id if user_id else f"ip_{client_ip}"
        
        # Get rate limit for this endpoint
        limit, window = self._get_rate_limit(path)
        
        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(
            identifier=identifier,
            limit=limit,
            window=window,
            is_admin=is_admin,
            ip=client_ip
        )
        
        # If rate limit exceeded, return 429
        if not result.allowed:
            logger.warning(
                f"🚫 Rate limit exceeded: {identifier} on {request.method} {path} "
                f"({result.limit} req/{window}s)"
            )
            
            error_response = {
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Please try again in {result.retry_after} seconds.",
                "limit": result.limit,
                "window": f"{window}s",
                "retry_after": result.retry_after,
                "reset_at": result.reset_at
            }
            
            response = Response(
                content=json.dumps(error_response),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json"
            )
            
            self._add_rate_limit_headers(response, result)
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        self._add_rate_limit_headers(response, result)
        
        return response


def rate_limited(
    rate_limiter: RateLimiterService,
    limit: int,
    window: int = 60,
    identifier_func: Optional[Callable] = None
):
    """
    Decorator for rate limiting specific functions.
    
    Args:
        rate_limiter: RateLimiterService instance
        limit: Maximum requests allowed
        window: Time window in seconds
        identifier_func: Function to extract identifier from request
    
    Example:
        @rate_limited(rate_limiter, limit=10, window=60)
        async def expensive_operation(request: Request):
            # This endpoint is limited to 10 requests per minute
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # If no request found, just call the function
                return await func(*args, **kwargs)
            
            # Get identifier
            if identifier_func:
                identifier = identifier_func(request)
            else:
                # Default: use IP address
                client_ip = request.client.host if request.client else "unknown"
                identifier = f"ip_{client_ip}"
            
            # Check rate limit
            result = await rate_limiter.check_rate_limit(
                identifier=identifier,
                limit=limit,
                window=window
            )
            
            if not result.allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Please try again in {result.retry_after} seconds.",
                        "limit": result.limit,
                        "retry_after": result.retry_after
                    }
                )
            
            # Call the function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

