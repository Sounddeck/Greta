"""
Logging Middleware for GRETA PAI
Provides correlation ID tracking and request/response logging
"""

import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger

from services.structured_logger import StructuredLogger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging with correlation ID tracking
    
    Features:
    - Generates unique correlation ID for each request
    - Tracks request/response timing
    - Logs request details (method, path, client IP)
    - Logs response details (status code, duration)
    - Attaches correlation ID to response headers
    - Enriches log context with user information
    """
    
    def __init__(
        self,
        app,
        exclude_paths: list[str] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024
    ):
        """
        Initialize logging middleware
        
        Args:
            app: ASGI application
            exclude_paths: List of paths to exclude from logging
            log_request_body: Whether to log request body
            log_response_body: Whether to log response body
            max_body_size: Maximum body size to log (bytes)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/favicon.ico",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and response with logging
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        
        # Set correlation ID in context
        StructuredLogger.set_correlation_id(correlation_id)
        StructuredLogger.set_request_path(request.url.path)
        
        # Extract user ID from request if available
        user_id = None
        if hasattr(request.state, "user_id"):
            user_id = request.state.user_id
            StructuredLogger.set_user_id(user_id)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Record start time
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "event": "request_started",
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        )
        
        # Log request body if enabled
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    logger.debug(
                        "Request body",
                        extra={
                            "event": "request_body",
                            "body": body.decode("utf-8", errors="ignore")[:self.max_body_size],
                            "body_size": len(body)
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to log request body: {e}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "event": "request_completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip,
                }
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-Duration-Ms"] = str(round(duration_ms, 2))
            
            # Clear context after request
            StructuredLogger.clear_context()
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "event": "request_failed",
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip,
                }
            )
            
            # Clear context after request
            StructuredLogger.clear_context()
            
            # Re-raise exception
            raise


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and slow request detection
    
    Features:
    - Tracks request duration
    - Logs slow requests (> threshold)
    - Provides performance metrics
    """
    
    def __init__(
        self,
        app,
        slow_request_threshold_ms: float = 1000.0,
        exclude_paths: list[str] = None
    ):
        """
        Initialize performance logging middleware
        
        Args:
            app: ASGI application
            slow_request_threshold_ms: Threshold for slow request warning (ms)
            exclude_paths: List of paths to exclude from logging
        """
        super().__init__(app)
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with performance monitoring
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        # Skip monitoring for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log slow requests
        if duration_ms > self.slow_request_threshold_ms:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path}",
                extra={
                    "event": "slow_request",
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "threshold_ms": self.slow_request_threshold_ms,
                    "status_code": response.status_code,
                }
            )
        
        return response

