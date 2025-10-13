"""
Security Middleware for GRETA PAI
Implements HTTPS enforcement, input validation, and security headers
"""

import os
import re
import json
from typing import Optional, List, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

logger = logging.getLogger(__name__)


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce HTTPS in production
    Redirects HTTP requests to HTTPS
    """
    
    def __init__(self, app: ASGIApp, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.environment = os.getenv("ENVIRONMENT", "development")
    
    async def dispatch(self, request: Request, call_next):
        # Only enforce HTTPS in production
        if self.enabled and self.environment == "production":
            # Check if request is HTTP
            if request.url.scheme == "http":
                # Redirect to HTTPS
                https_url = request.url.replace(scheme="https")
                logger.warning(f"Redirecting HTTP request to HTTPS: {request.url} -> {https_url}")
                return RedirectResponse(url=str(https_url), status_code=301)
        
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses
    Prevents clickjacking, XSS, MIME sniffing, etc.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' http://localhost:* ws://localhost:* wss://localhost:*; "
            "frame-ancestors 'none';"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate and sanitize input data
    Prevents SQL injection, XSS, command injection, etc.
    """
    
    # Dangerous patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
        r"(\bINSERT\b.*\bINTO\b.*\bVALUES\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\(.*\)",
        r"`.*`",
        r"\|\|",
        r"&&",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"\.\.\\",
    ]
    
    def __init__(self, app: ASGIApp, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.max_body_size = 10 * 1024 * 1024  # 10MB
        
        # Compile patterns for performance
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
        self.cmd_patterns = [re.compile(p) for p in self.COMMAND_INJECTION_PATTERNS]
        self.path_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS]
    
    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)
        
        # Skip validation for certain paths (health checks, static files)
        skip_paths = ["/health", "/docs", "/openapi.json", "/static"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        try:
            # Validate request body size
            if request.method in ["POST", "PUT", "PATCH"]:
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) > self.max_body_size:
                    logger.warning(f"Request body too large: {content_length} bytes from {request.client.host}")
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Request body too large"
                    )
            
            # Validate query parameters
            for key, value in request.query_params.items():
                self._validate_input(key, value, "query parameter")
            
            # Validate path parameters
            for key, value in request.path_params.items():
                self._validate_input(key, value, "path parameter")
            
            # Validate request body (if JSON)
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        body = await request.body()
                        if body:
                            data = json.loads(body)
                            self._validate_json_data(data)
                            
                            # Re-create request with validated body
                            async def receive():
                                return {"type": "http.request", "body": body}
                            
                            request._receive = receive
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in request body from {request.client.host}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid JSON in request body"
                        )
            
            response = await call_next(request)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input data"
            )
    
    def _validate_input(self, key: str, value: str, input_type: str):
        """Validate a single input value"""
        if not isinstance(value, str):
            return
        
        # Check for SQL injection
        for pattern in self.sql_patterns:
            if pattern.search(value):
                logger.warning(f"Potential SQL injection detected in {input_type} '{key}': {value[:100]}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid characters detected in {input_type}"
                )
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if pattern.search(value):
                logger.warning(f"Potential XSS detected in {input_type} '{key}': {value[:100]}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid characters detected in {input_type}"
                )
        
        # Check for command injection
        for pattern in self.cmd_patterns:
            if pattern.search(value):
                logger.warning(f"Potential command injection detected in {input_type} '{key}': {value[:100]}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid characters detected in {input_type}"
                )
        
        # Check for path traversal
        for pattern in self.path_patterns:
            if pattern.search(value):
                logger.warning(f"Potential path traversal detected in {input_type} '{key}': {value[:100]}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid characters detected in {input_type}"
                )
    
    def _validate_json_data(self, data: Any):
        """Recursively validate JSON data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    self._validate_input(key, value, "JSON field")
                elif isinstance(value, (dict, list)):
                    self._validate_json_data(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    self._validate_input("list_item", item, "JSON array item")
                elif isinstance(item, (dict, list)):
                    self._validate_json_data(item)


class CORSMiddleware:
    """
    CORS configuration with whitelist support
    Note: This is a configuration helper, actual CORS is handled by FastAPI's CORSMiddleware
    """
    
    @staticmethod
    def get_allowed_origins() -> List[str]:
        """Get allowed origins from environment"""
        environment = os.getenv("ENVIRONMENT", "development")
        
        if environment == "development":
            # Allow localhost in development
            return [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:5174",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:5174",
                "http://127.0.0.1:8000",
            ]
        else:
            # Production: Use whitelist from environment
            origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
            if origins_str:
                return [origin.strip() for origin in origins_str.split(",")]
            return []
    
    @staticmethod
    def get_cors_config() -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": CORSMiddleware.get_allowed_origins(),
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            "allow_headers": ["*"],
            "expose_headers": ["*"],
            "max_age": 3600,
        }

