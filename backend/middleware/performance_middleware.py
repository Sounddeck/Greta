"""
Performance Middleware

Automatically tracks performance metrics for all API requests including:
- Request/response time
- Endpoint statistics
- Slow request detection
- Error tracking
- Query performance
"""

import time
import uuid
import logging
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from services.performance_monitor_service import PerformanceMonitorService, RequestMetrics
from database import Database

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Performance Middleware
    
    Automatically tracks performance metrics for all API requests.
    """
    
    def __init__(
        self,
        app,
        performance_monitor: PerformanceMonitorService,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.performance_monitor = performance_monitor
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]
        logger.info("✅ PerformanceMiddleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track performance metrics"""
        
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Get client info
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        
        # Track start time
        start_time = time.time()
        
        # Initialize query tracking
        query_count = 0
        query_time_ms = 0.0
        cache_hits = 0
        cache_misses = 0
        
        # Process request
        error = None
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Get query metrics from request state if available
            if hasattr(request.state, "query_count"):
                query_count = request.state.query_count
            if hasattr(request.state, "query_time_ms"):
                query_time_ms = request.state.query_time_ms
            if hasattr(request.state, "cache_hits"):
                cache_hits = request.state.cache_hits
            if hasattr(request.state, "cache_misses"):
                cache_misses = request.state.cache_misses
            
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Error processing request: {e}")
            raise
        
        finally:
            # Calculate response time
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Create metrics
            metrics = RequestMetrics(
                request_id=request_id,
                endpoint=request.url.path,
                method=request.method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                timestamp=Database.utcnow(),
                user_id=user_id,
                ip_address=client_ip,
                error=error,
                query_count=query_count,
                query_time_ms=query_time_ms,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
            
            # Track metrics
            await self.performance_monitor.track_request(metrics)
            
            # Add performance headers to response
            if 'response' in locals():
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
                if query_count > 0:
                    response.headers["X-Query-Count"] = str(query_count)
                    response.headers["X-Query-Time"] = f"{query_time_ms:.2f}ms"
                if cache_hits > 0 or cache_misses > 0:
                    response.headers["X-Cache-Hits"] = str(cache_hits)
                    response.headers["X-Cache-Misses"] = str(cache_misses)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check X-Forwarded-For header first (for proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Get user ID from request"""
        # Try to get from request state (set by auth middleware)
        if hasattr(request.state, "user_id"):
            return request.state.user_id
        
        # Try to get from headers
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        return None


class QueryPerformanceTracker:
    """
    Query Performance Tracker
    
    Context manager for tracking database query performance.
    """
    
    def __init__(self, request: Request, query_name: str):
        self.request = request
        self.query_name = query_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            query_time_ms = (time.time() - self.start_time) * 1000
            
            # Update request state
            if not hasattr(self.request.state, "query_count"):
                self.request.state.query_count = 0
            if not hasattr(self.request.state, "query_time_ms"):
                self.request.state.query_time_ms = 0.0
            
            self.request.state.query_count += 1
            self.request.state.query_time_ms += query_time_ms
            
            # Log slow queries
            if query_time_ms >= 100.0:  # 100ms threshold
                logger.warning(
                    f"🐌 Slow query detected: {self.query_name} took {query_time_ms:.2f}ms"
                )


class CachePerformanceTracker:
    """
    Cache Performance Tracker
    
    Tracks cache hits and misses for requests.
    """
    
    @staticmethod
    def track_hit(request: Request):
        """Track a cache hit"""
        if not hasattr(request.state, "cache_hits"):
            request.state.cache_hits = 0
        request.state.cache_hits += 1
    
    @staticmethod
    def track_miss(request: Request):
        """Track a cache miss"""
        if not hasattr(request.state, "cache_misses"):
            request.state.cache_misses = 0
        request.state.cache_misses += 1

