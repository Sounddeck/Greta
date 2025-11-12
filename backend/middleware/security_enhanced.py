"""
Enhanced Security Middleware for GRETA PAI

This middleware provides comprehensive security features including
security headers, IP filtering, request signing verification, and
CORS configuration.

Features:
- Automatic security headers injection
- IP whitelist/blacklist enforcement
- Request signature verification
- CORS configuration
- Security event logging
- Rate limiting integration

Example:
    Add to FastAPI application:
        >>> from fastapi import FastAPI
        >>> from middleware.security_enhanced import SecurityHeadersMiddleware, IPFilterMiddleware
        >>> 
        >>> app = FastAPI()
        >>> app.add_middleware(SecurityHeadersMiddleware)
        >>> app.add_middleware(IPFilterMiddleware)
"""

import logging
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.
    
    Automatically adds security headers like HSTS, CSP, X-Frame-Options,
    etc. to all responses.
    
    Attributes:
        security_service: Security service instance
        
    Example:
        >>> from services.security_service import SecurityService
        >>> security_service = SecurityService()
        >>> app.add_middleware(SecurityHeadersMiddleware, security_service=security_service)
    """
    
    def __init__(self, app, security_service=None):
        """
        Initialize security headers middleware.
        
        Args:
            app: FastAPI application instance
            security_service: Security service instance
        """
        super().__init__(app)
        self.security_service = security_service
        
        # Default security headers if service not provided
        self.default_headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response with security headers
        """
        # Process request
        response = await call_next(request)
        
        # Get security headers
        if self.security_service:
            security_headers = self.security_service.get_security_headers()
        else:
            security_headers = self.default_headers
        
        # Add security headers to response
        headers = MutableHeaders(response.headers)
        for header_name, header_value in security_headers.items():
            headers[header_name] = header_value
        
        return Response(
            content=b"".join([chunk async for chunk in response.body_iterator]),
            status_code=response.status_code,
            headers=dict(headers),
            media_type=response.media_type
        )


class IPFilterMiddleware(BaseHTTPMiddleware):
    """
    Middleware for IP whitelist/blacklist filtering.
    
    Blocks requests from blacklisted IPs and enforces whitelist
    if configured.
    
    Attributes:
        security_service: Security service instance
        
    Example:
        >>> from services.security_service import SecurityService
        >>> security_service = SecurityService()
        >>> app.add_middleware(IPFilterMiddleware, security_service=security_service)
    """
    
    def __init__(self, app, security_service=None, enabled: bool = True):
        """
        Initialize IP filter middleware.
        
        Args:
            app: FastAPI application instance
            security_service: Security service instance
            enabled: Whether IP filtering is enabled
        """
        super().__init__(app)
        self.security_service = security_service
        self.enabled = enabled
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Filter requests based on IP address.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response or 403 Forbidden
        """
        if not self.enabled or not self.security_service:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else None
        
        # Check X-Forwarded-For header (for proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        if not client_ip:
            logger.warning("⚠️ Could not determine client IP")
            return await call_next(request)
        
        # Check if IP is allowed
        is_allowed = await self.security_service.is_ip_allowed(client_ip)
        
        if not is_allowed:
            # Log security event
            await self.security_service.log_security_event(
                "ip_blocked",
                {
                    "ip": client_ip,
                    "path": str(request.url.path),
                    "method": request.method
                }
            )
            
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "message": "Access denied",
                    "error": "Your IP address is not authorized to access this resource"
                }
            )
        
        return await call_next(request)


class RequestSigningMiddleware(BaseHTTPMiddleware):
    """
    Middleware for verifying request signatures on sensitive operations.
    
    Verifies HMAC signatures for requests to protected endpoints.
    
    Attributes:
        security_service: Security service instance
        protected_paths: List of paths requiring signature verification
        
    Example:
        >>> security_service = SecurityService()
        >>> app.add_middleware(
        ...     RequestSigningMiddleware,
        ...     security_service=security_service,
        ...     protected_paths=["/api/v1/admin", "/api/v1/users/delete"]
        ... )
    """
    
    def __init__(
        self,
        app,
        security_service=None,
        protected_paths: list = None,
        enabled: bool = False
    ):
        """
        Initialize request signing middleware.
        
        Args:
            app: FastAPI application instance
            security_service: Security service instance
            protected_paths: List of paths requiring signature verification
            enabled: Whether request signing is enabled
        """
        super().__init__(app)
        self.security_service = security_service
        self.protected_paths = protected_paths or []
        self.enabled = enabled
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Verify request signature for protected paths.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response or 401 Unauthorized
        """
        if not self.enabled or not self.security_service:
            return await call_next(request)
        
        # Check if path requires signature
        path = str(request.url.path)
        requires_signature = any(path.startswith(protected) for protected in self.protected_paths)
        
        if not requires_signature:
            return await call_next(request)
        
        # Get signature from header
        signature = request.headers.get("X-Request-Signature")
        
        if not signature:
            logger.warning(f"⚠️ Missing signature for protected path: {path}")
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "message": "Request signature required",
                    "error": "This endpoint requires a valid request signature"
                }
            )
        
        # Get request body for verification
        try:
            body = await request.body()
            if body:
                import json
                request_data = json.loads(body.decode())
            else:
                request_data = {}
            
            # Verify signature
            is_valid = self.security_service.verify_request_signature(request_data, signature)
            
            if not is_valid:
                logger.warning(f"⚠️ Invalid signature for path: {path}")
                
                # Log security event
                await self.security_service.log_security_event(
                    "invalid_signature",
                    {
                        "path": path,
                        "method": request.method,
                        "ip": request.client.host if request.client else None
                    }
                )
                
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "message": "Invalid request signature",
                        "error": "The request signature is invalid or expired"
                    }
                )
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Signature verification failed",
                    "error": str(e)
                }
            )
        
        return await call_next(request)


def configure_cors(app, security_service=None):
    """
    Configure CORS middleware for the application.
    
    Args:
        app: FastAPI application instance
        security_service: Security service instance
        
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> configure_cors(app, security_service)
    """
    if security_service and hasattr(security_service, 'cors_config'):
        cors_config = security_service.cors_config
    else:
        # Default CORS configuration
        from services.security_service import CORSConfig
        cors_config = CORSConfig()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allow_origins,
        allow_credentials=cors_config.allow_credentials,
        allow_methods=cors_config.allow_methods,
        allow_headers=cors_config.allow_headers,
        max_age=cors_config.max_age
    )
    
    logger.info("✅ CORS middleware configured")
    logger.info(f"   Allowed origins: {cors_config.allow_origins}")
    logger.info(f"   Allowed methods: {cors_config.allow_methods}")
