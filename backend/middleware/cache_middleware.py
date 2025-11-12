"""
Cache Middleware for CPAS5

Phase 3, Task 3.2: Implement Caching Layer
Automatic API response caching with cache headers and bypass options.
"""

import logging
import hashlib
import json
from typing import Callable, Optional, Set
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers
from services.cache_service import CacheService

logger = logging.getLogger(__name__)


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Cache Middleware for automatic API response caching.
    
    Phase 3, Task 3.2: Implement cache middleware for automatic response caching
    
    Features:
    - Automatic caching of GET requests
    - Cache headers (Cache-Control, ETag)
    - Cache bypass for authenticated admin users
    - Configurable TTLs per endpoint
    """
    
    def __init__(
        self,
        app,
        cache_service: CacheService,
        cacheable_paths: Optional[Set[str]] = None,
        default_ttl: int = 300,
        admin_bypass: bool = True
    ):
        super().__init__(app)
        self.cache_service = cache_service
        self.cacheable_paths = cacheable_paths or {
            "/api/v1/crm/stats",
            "/api/v1/ats/jobs",
            "/api/v1/agent-builder/templates",
            "/api/v1/crm/contacts",
            "/api/v1/crm/companies",
            "/api/v1/crm/deals",
            "/api/v1/ats/candidates"
        }
        self.default_ttl = default_ttl
        self.admin_bypass = admin_bypass
        
        # TTL configuration per endpoint pattern
        self.ttl_config = {
            "/api/v1/crm/stats": 300,  # 5 minutes
            "/api/v1/ats/jobs": 120,  # 2 minutes
            "/api/v1/agent-builder/templates": 600,  # 10 minutes
            "/api/v1/crm/contacts": 120,  # 2 minutes
            "/api/v1/crm/companies": 120,  # 2 minutes
            "/api/v1/crm/deals": 120,  # 2 minutes
            "/api/v1/ats/candidates": 120,  # 2 minutes
        }
        
        logger.info(f"✅ Cache Middleware initialized (cacheable paths: {len(self.cacheable_paths)})")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with caching logic.
        
        Phase 3, Task 3.2: Cache read-heavy endpoints
        """
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Check if path is cacheable
        path = request.url.path
        if not self._is_cacheable(path):
            return await call_next(request)
        
        # Check for cache bypass
        if self._should_bypass_cache(request):
            logger.debug(f"⚠️ Cache bypass: {path}")
            response = await call_next(request)
            response.headers["X-Cache"] = "BYPASS"
            return response
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache
        cached_response = await self.cache_service.get(cache_key)
        
        if cached_response is not None:
            # Cache hit
            logger.debug(f"✅ Cache hit: {path}")
            
            # Create response from cached data
            response = JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=cached_response.get("headers", {})
            )
            
            # Add cache headers
            response.headers["X-Cache"] = "HIT"
            response.headers["Cache-Control"] = f"public, max-age={self._get_ttl(path)}"
            
            # Add ETag
            etag = self._generate_etag(cached_response["content"])
            response.headers["ETag"] = etag
            
            # Check If-None-Match header
            if request.headers.get("If-None-Match") == etag:
                return Response(status_code=304, headers=response.headers)
            
            return response
        
        # Cache miss - call the endpoint
        logger.debug(f"❌ Cache miss: {path}")
        response = await call_next(request)
        
        # Only cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            try:
                # Parse JSON response
                content = json.loads(body.decode())
                
                # Prepare cache data
                cache_data = {
                    "content": content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
                
                # Cache the response
                ttl = self._get_ttl(path)
                await self.cache_service.set(cache_key, cache_data, ttl=ttl)
                
                # Create new response
                response = JSONResponse(
                    content=content,
                    status_code=response.status_code,
                    headers=response.headers
                )
                
                # Add cache headers
                response.headers["X-Cache"] = "MISS"
                response.headers["Cache-Control"] = f"public, max-age={ttl}"
                
                # Add ETag
                etag = self._generate_etag(content)
                response.headers["ETag"] = etag
                
                logger.debug(f"✅ Response cached: {path} (TTL: {ttl}s)")
                
            except Exception as e:
                logger.error(f"❌ Error caching response: {e}")
                # Return original response on error
                response = Response(
                    content=body,
                    status_code=response.status_code,
                    headers=response.headers
                )
                response.headers["X-Cache"] = "ERROR"
        else:
            response.headers["X-Cache"] = "SKIP"
        
        return response
    
    def _is_cacheable(self, path: str) -> bool:
        """Check if path should be cached"""
        # Exact match
        if path in self.cacheable_paths:
            return True
        
        # Pattern match (e.g., /api/v1/crm/contacts/*)
        for cacheable_path in self.cacheable_paths:
            if path.startswith(cacheable_path):
                return True
        
        return False
    
    def _should_bypass_cache(self, request: Request) -> bool:
        """
        Check if cache should be bypassed.
        
        Phase 3, Task 3.2: Add cache bypass options for authenticated admin users
        """
        # Check for cache bypass header
        if request.headers.get("X-Cache-Bypass") == "true":
            return True
        
        # Check for no-cache in Cache-Control
        cache_control = request.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return True
        
        # Check for admin bypass
        if self.admin_bypass:
            # Check if user is admin (you'd implement this based on your auth system)
            user_role = request.headers.get("X-User-Role")
            if user_role == "admin":
                return True
        
        return False
    
    def _generate_cache_key(self, request: Request) -> str:
        """
        Generate cache key from request.
        
        Includes: path, query params, and relevant headers
        """
        # Base key from path
        path = request.url.path
        
        # Add query parameters
        query_params = str(sorted(request.query_params.items()))
        
        # Add relevant headers (e.g., Accept-Language for i18n)
        relevant_headers = {
            "Accept-Language": request.headers.get("Accept-Language", ""),
            "X-User-ID": request.headers.get("X-User-ID", "")  # For user-specific caching
        }
        headers_str = str(sorted(relevant_headers.items()))
        
        # Create hash
        key_data = f"{path}:{query_params}:{headers_str}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"response:{path}:{key_hash}"
    
    def _generate_etag(self, content: any) -> str:
        """
        Generate ETag from response content.
        
        Phase 3, Task 3.2: Add cache headers (ETag) to API responses
        """
        content_str = json.dumps(content, sort_keys=True)
        etag_hash = hashlib.md5(content_str.encode()).hexdigest()
        return f'"{etag_hash}"'
    
    def _get_ttl(self, path: str) -> int:
        """
        Get TTL for path.
        
        Phase 3, Task 3.2: Configure different TTLs for different endpoint types
        """
        # Check exact match
        if path in self.ttl_config:
            return self.ttl_config[path]
        
        # Check pattern match
        for pattern, ttl in self.ttl_config.items():
            if path.startswith(pattern):
                return ttl
        
        # Default TTL
        return self.default_ttl
    
    def add_cacheable_path(self, path: str, ttl: Optional[int] = None):
        """Add a path to cacheable paths"""
        self.cacheable_paths.add(path)
        if ttl is not None:
            self.ttl_config[path] = ttl
        logger.info(f"✅ Added cacheable path: {path} (TTL: {ttl or self.default_ttl}s)")
    
    def remove_cacheable_path(self, path: str):
        """Remove a path from cacheable paths"""
        self.cacheable_paths.discard(path)
        self.ttl_config.pop(path, None)
        logger.info(f"✅ Removed cacheable path: {path}")


# ========================================================================
# CACHE DECORATOR FOR FUNCTIONS
# ========================================================================

def cached(
    cache_service: CacheService,
    key_prefix: str,
    ttl: Optional[int] = None,
    key_builder: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Phase 3, Task 3.2: Cache frequently accessed data
    
    Example:
        @cached(cache_service, "crm_stats", ttl=300)
        async def get_crm_stats():
            # Expensive operation
            return stats
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key builder
                args_str = str(args) + str(sorted(kwargs.items()))
                key_hash = hashlib.md5(args_str.encode()).hexdigest()
                cache_key = f"{key_prefix}:{key_hash}"
            
            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug(f"✅ Function cache hit: {func.__name__}")
                return cached_result
            
            # Call function
            logger.debug(f"❌ Function cache miss: {func.__name__}")
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_service.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator

