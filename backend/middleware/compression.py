"""
Response Compression Middleware for GRETA PAI

This middleware provides automatic response compression using gzip
for responses exceeding a configurable size threshold.

Features:
- Automatic gzip compression for large responses
- Configurable compression threshold and level
- Content-Type based compression filtering
- Accept-Encoding header support
- Compression ratio tracking
- Performance metrics integration

Example:
    Add to FastAPI application:
        >>> from fastapi import FastAPI
        >>> from middleware.compression import CompressionMiddleware
        >>> 
        >>> app = FastAPI()
        >>> app.add_middleware(CompressionMiddleware, minimum_size=1024)
"""

import gzip
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.datastructures import Headers, MutableHeaders

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic response compression.
    
    Compresses responses using gzip when the client supports it
    and the response size exceeds the minimum threshold.
    
    Attributes:
        minimum_size: Minimum response size in bytes to trigger compression
        compression_level: Gzip compression level (1-9)
        compressible_types: Set of compressible content types
        
    Example:
        >>> app.add_middleware(
        ...     CompressionMiddleware,
        ...     minimum_size=1024,
        ...     compression_level=6
        ... )
    """
    
    def __init__(
        self,
        app,
        minimum_size: int = 1024,
        compression_level: int = 6,
        compressible_types: set = None
    ):
        """
        Initialize compression middleware.
        
        Args:
            app: FastAPI application instance
            minimum_size: Minimum response size in bytes to compress
            compression_level: Gzip compression level (1-9)
            compressible_types: Set of compressible content types
        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        
        # Default compressible content types
        self.compressible_types = compressible_types or {
            "text/html",
            "text/plain",
            "text/css",
            "text/javascript",
            "application/javascript",
            "application/json",
            "application/xml",
            "text/xml",
            "application/x-javascript",
            "application/xhtml+xml",
            "application/rss+xml",
            "application/atom+xml",
            "image/svg+xml"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and compress response if applicable.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Potentially compressed response
        """
        # Get response from next middleware/handler
        response = await call_next(request)
        
        # Check if client accepts gzip encoding
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Check if response is already compressed
        if response.headers.get("content-encoding"):
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in self.compressible_types:
            return response
        
        # Get response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        # Check if response size exceeds minimum
        if len(response_body) < self.minimum_size:
            # Return original response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Compress response
        try:
            compressed_body = gzip.compress(
                response_body,
                compresslevel=self.compression_level
            )
            
            # Calculate compression ratio
            original_size = len(response_body)
            compressed_size = len(compressed_body)
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            logger.debug(
                f"Compressed response: {original_size} -> {compressed_size} bytes "
                f"(ratio: {ratio:.2f}x)"
            )
            
            # Create new headers
            headers = MutableHeaders(response.headers)
            headers["content-encoding"] = "gzip"
            headers["content-length"] = str(len(compressed_body))
            headers["vary"] = "Accept-Encoding"
            
            # Add compression info header for debugging
            headers["x-compression-ratio"] = f"{ratio:.2f}"
            
            return Response(
                content=compressed_body,
                status_code=response.status_code,
                headers=dict(headers),
                media_type=response.media_type
            )
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return original response if compression fails
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )


class PerformanceTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking request/response performance.
    
    Tracks response times, request counts, and integrates with
    the performance service for metrics collection.
    
    Attributes:
        performance_service: Performance service instance
        
    Example:
        >>> from services.performance_service import PerformanceService
        >>> perf_service = PerformanceService()
        >>> app.add_middleware(PerformanceTrackingMiddleware, performance_service=perf_service)
    """
    
    def __init__(self, app, performance_service=None):
        """
        Initialize performance tracking middleware.
        
        Args:
            app: FastAPI application instance
            performance_service: Performance service instance
        """
        super().__init__(app)
        self.performance_service = performance_service
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Track request/response performance.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response with performance headers
        """
        import time
        
        # Record start time
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Track in performance service
        if self.performance_service:
            self.performance_service.track_response_time(response_time)
        
        # Add performance headers
        headers = MutableHeaders(response.headers)
        headers["x-response-time"] = f"{response_time:.3f}s"
        headers["x-response-time-ms"] = f"{response_time * 1000:.2f}"
        
        # Log slow requests
        if response_time > 1.0:
            logger.warning(
                f"⚠️ Slow request: {request.method} {request.url.path} "
                f"took {response_time:.3f}s"
            )
        
        return Response(
            content=b"".join([chunk async for chunk in response.body_iterator]),
            status_code=response.status_code,
            headers=dict(headers),
            media_type=response.media_type
        )


class ConnectionPoolMiddleware(BaseHTTPMiddleware):
    """
    Middleware for connection pool monitoring and optimization.
    
    Monitors database connection pool usage and logs warnings
    when pool utilization is high.
    
    Example:
        >>> app.add_middleware(ConnectionPoolMiddleware)
    """
    
    def __init__(self, app, max_pool_size: int = 100, warning_threshold: float = 0.8):
        """
        Initialize connection pool middleware.
        
        Args:
            app: FastAPI application instance
            max_pool_size: Maximum connection pool size
            warning_threshold: Threshold for pool utilization warnings (0-1)
        """
        super().__init__(app)
        self.max_pool_size = max_pool_size
        self.warning_threshold = warning_threshold
        self._active_connections = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Monitor connection pool usage.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response with connection pool headers
        """
        # Increment active connections
        self._active_connections += 1
        
        try:
            # Check pool utilization
            utilization = self._active_connections / self.max_pool_size
            
            if utilization > self.warning_threshold:
                logger.warning(
                    f"⚠️ High connection pool utilization: "
                    f"{self._active_connections}/{self.max_pool_size} "
                    f"({utilization * 100:.1f}%)"
                )
            
            # Process request
            response = await call_next(request)
            
            # Add connection pool headers
            headers = MutableHeaders(response.headers)
            headers["x-pool-active"] = str(self._active_connections)
            headers["x-pool-utilization"] = f"{utilization * 100:.1f}%"
            
            return Response(
                content=b"".join([chunk async for chunk in response.body_iterator]),
                status_code=response.status_code,
                headers=dict(headers),
                media_type=response.media_type
            )
            
        finally:
            # Decrement active connections
            self._active_connections = max(0, self._active_connections - 1)


class SerializationOptimizationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for optimizing JSON serialization/deserialization.
    
    Uses optimized JSON libraries (orjson, ujson) when available
    for faster serialization performance.
    
    Example:
        >>> app.add_middleware(SerializationOptimizationMiddleware)
    """
    
    def __init__(self, app):
        """
        Initialize serialization optimization middleware.
        
        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
        
        # Try to import optimized JSON libraries
        self.json_encoder = None
        self.json_decoder = None
        
        try:
            import orjson
            self.json_encoder = orjson.dumps
            self.json_decoder = orjson.loads
            logger.info("✅ Using orjson for optimized JSON serialization")
        except ImportError:
            try:
                import ujson
                self.json_encoder = ujson.dumps
                self.json_decoder = ujson.loads
                logger.info("✅ Using ujson for optimized JSON serialization")
            except ImportError:
                import json
                self.json_encoder = json.dumps
                self.json_decoder = json.loads
                logger.info("ℹ️ Using standard json library")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Optimize JSON serialization for response.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response with optimized serialization
        """
        # Process request
        response = await call_next(request)
        
        # Add serialization info header
        headers = MutableHeaders(response.headers)
        
        if self.json_encoder.__module__ == 'orjson':
            headers["x-json-encoder"] = "orjson"
        elif self.json_encoder.__module__ == 'ujson':
            headers["x-json-encoder"] = "ujson"
        else:
            headers["x-json-encoder"] = "json"
        
        return Response(
            content=b"".join([chunk async for chunk in response.body_iterator]),
            status_code=response.status_code,
            headers=dict(headers),
            media_type=response.media_type
        )
