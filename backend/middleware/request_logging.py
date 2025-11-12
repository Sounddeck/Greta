"""
Request/Response Logging Middleware for GRETA PAI
Provides comprehensive request tracking with performance monitoring
"""

import time
import uuid
import json
import logging
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
import contextvars

from utils.metrics import record_endpoint_request
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Context variables for request tracking
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id')
user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('user_id', default=None)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging and performance tracking
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        
        logger.info("✅ Request logging middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with comprehensive logging"""
        
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Extract user ID from request (if authenticated)
        user_id = await self._extract_user_id(request)
        if user_id:
            user_id_var.set(user_id)
        
        # Record request start time
        start_time = time.time()
        
        # Log request details
        await self._log_request(request, request_id, user_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response details
            await self._log_response(request, response, request_id, user_id, duration_ms)
            
            # Record metrics
            record_endpoint_request(
                endpoint=request.url.path,
                method=request.method,
                duration=duration_ms,
                status_code=response.status_code,
                user_id=user_id
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Record error metrics
            record_endpoint_request(
                endpoint=request.url.path,
                method=request.method,
                duration=duration_ms,
                status_code=500,
                user_id=user_id
            )
            
            raise
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request headers or JWT token"""
        
        try:
            # Try to get user ID from custom header
            user_id = request.headers.get("X-User-ID")
            if user_id:
                return user_id
            
            # Try to extract from Authorization header (JWT)
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                # Here you would decode the JWT and extract user ID
                # For now, we'll just return None
                # user_id = decode_jwt_user_id(token)
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract user ID: {e}")
            return None
    
    async def _log_request(self, request: Request, request_id: str, user_id: Optional[str]):
        """Log incoming request details"""
        
        # Basic request info
        log_data = {
            "request_id": request_id,
            "user_id": user_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("User-Agent"),
            "content_type": request.headers.get("Content-Type"),
            "content_length": request.headers.get("Content-Length")
        }
        
        # Log request body if enabled and appropriate
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await self._read_request_body(request)
                if body:
                    log_data["body"] = body
            except Exception as e:
                log_data["body_error"] = str(e)
        
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra=log_data
        )
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        user_id: Optional[str],
        duration_ms: float
    ):
        """Log outgoing response details"""
        
        # Basic response info
        log_data = {
            "request_id": request_id,
            "user_id": user_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "response_headers": dict(response.headers)
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = "error"
        elif response.status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        # Log response body if enabled and appropriate
        if self.log_response_body and hasattr(response, 'body'):
            try:
                body = await self._read_response_body(response)
                if body:
                    log_data["body"] = body
            except Exception as e:
                log_data["body_error"] = str(e)
        
        # Log with appropriate level
        log_message = f"Response: {request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms"
        
        if log_level == "error":
            logger.error(log_message, extra=log_data)
        elif log_level == "warning":
            logger.warning(log_message, extra=log_data)
        else:
            logger.info(log_message, extra=log_data)
    
    async def _read_request_body(self, request: Request) -> Optional[str]:
        """Safely read and truncate request body"""
        
        try:
            body = await request.body()
            if not body:
                return None
            
            # Truncate if too large
            if len(body) > self.max_body_size:
                body = body[:self.max_body_size]
                truncated = True
            else:
                truncated = False
            
            # Try to decode as text
            try:
                text = body.decode('utf-8')
                if truncated:
                    text += "... [truncated]"
                return text
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"
                
        except Exception as e:
            logger.debug(f"Failed to read request body: {e}")
            return None
    
    async def _read_response_body(self, response: Response) -> Optional[str]:
        """Safely read and truncate response body"""
        
        try:
            if not hasattr(response, 'body') or not response.body:
                return None
            
            body = response.body
            
            # Truncate if too large
            if len(body) > self.max_body_size:
                body = body[:self.max_body_size]
                truncated = True
            else:
                truncated = False
            
            # Try to decode as text
            try:
                text = body.decode('utf-8')
                if truncated:
                    text += "... [truncated]"
                return text
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"
                
        except Exception as e:
            logger.debug(f"Failed to read response body: {e}")
            return None


def get_current_request_id() -> Optional[str]:
    """Get the current request ID from context"""
    try:
        return request_id_var.get()
    except LookupError:
        return None


def get_current_user_id() -> Optional[str]:
    """Get the current user ID from context"""
    try:
        return user_id_var.get()
    except LookupError:
        return None


# Export middleware and utilities
__all__ = [
    "RequestLoggingMiddleware",
    "get_current_request_id",
    "get_current_user_id",
    "request_id_var",
    "user_id_var"
]
