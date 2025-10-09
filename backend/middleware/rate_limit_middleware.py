"""
Rate Limiting Middleware for CPAS4
Middleware to enforce rate limits on API requests
"""

import logging
from typing import Callable
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from database import Database
from services.rate_limit_service import RateLimitService
from services.audit_service import AuditService
from models.audit_models import EventType, SeverityLevel

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits"""
    
    # Paths excluded from rate limiting
    EXCLUDED_PATHS = [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
        "/static/",
        "/api/v1/rate-limit/health"
    ]
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limit_service = None
        self.audit_service = None
    
    async def _ensure_services(self):
        """Ensure services are initialized"""
        if self.rate_limit_service is None:
            db = Database()
            await db.connect()
            self.rate_limit_service = RateLimitService(db)
            self.audit_service = AuditService(db)
    
    def _should_check_rate_limit(self, path: str) -> bool:
        """Check if path should be rate limited"""
        for excluded in self.EXCLUDED_PATHS:
            if path.startswith(excluded):
                return False
        return True
    
    def _extract_user_id(self, request: Request) -> str:
        """Extract user ID from request"""
        # Try to get from query params
        user_id = request.query_params.get("user_id")
        if user_id:
            return user_id
        
        # Try to get from headers
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        # Try to get from auth header (future)
        # auth_header = request.headers.get("Authorization")
        # if auth_header:
        #     # Parse JWT or API key to get user_id
        #     pass
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check X-Forwarded-For header (for proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Get from client
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _add_rate_limit_headers(
        self,
        response: Response,
        limit: int,
        remaining: int,
        reset_at: datetime
    ):
        """Add rate limit headers to response"""
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_at.timestamp()))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and enforce rate limits"""
        # Ensure services are initialized
        await self._ensure_services()
        
        path = request.url.path
        
        # Check if path should be rate limited
        if not self._should_check_rate_limit(path):
            return await call_next(request)
        
        # Extract user ID and IP
        user_id = self._extract_user_id(request)
        ip_address = self._get_client_ip(request)
        
        try:
            # Check rate limit
            is_allowed, violated_rule, status = await self.rate_limit_service.check_rate_limit(
                endpoint=path,
                user_id=user_id,
                ip_address=ip_address
            )
            
            if not is_allowed and violated_rule:
                # Rate limit exceeded
                logger.warning(
                    f"Rate limit exceeded: {path} - "
                    f"user={user_id}, ip={ip_address}, "
                    f"rule={violated_rule.name}"
                )
                
                # Log to audit system
                try:
                    await self.audit_service.log_event(
                        event_type=EventType.RATE_LIMIT_EXCEEDED,
                        user_id=user_id,
                        severity=SeverityLevel.WARNING,
                        resource_type="api",
                        action="rate_limit_exceeded",
                        ip_address=ip_address,
                        request_method=request.method,
                        request_path=path,
                        error_message=f"Rate limit exceeded: {violated_rule.name}",
                        metadata={
                            "rule_id": violated_rule.rule_id,
                            "rule_name": violated_rule.name,
                            "max_requests": violated_rule.max_requests,
                            "window_seconds": violated_rule.window_seconds,
                            "current_requests": status.current_requests if status else 0
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to log rate limit violation to audit: {e}")
                
                # Return 429 Too Many Requests
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Please try again later.",
                        "rule": violated_rule.name,
                        "limit": violated_rule.max_requests,
                        "window_seconds": violated_rule.window_seconds,
                        "retry_after": status.reset_at.isoformat() if status else None
                    }
                )
                
                # Add rate limit headers
                if status:
                    self._add_rate_limit_headers(
                        response,
                        limit=status.max_requests,
                        remaining=0,
                        reset_at=status.reset_at
                    )
                
                return response
            
            # Record request
            await self.rate_limit_service.record_request(
                endpoint=path,
                user_id=user_id,
                ip_address=ip_address
            )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to successful response
            statuses = await self.rate_limit_service.get_rate_limit_status(
                endpoint=path,
                user_id=user_id,
                ip_address=ip_address
            )
            
            if statuses:
                # Use the most restrictive limit
                most_restrictive = min(statuses, key=lambda s: s.remaining_requests)
                self._add_rate_limit_headers(
                    response,
                    limit=most_restrictive.max_requests,
                    remaining=most_restrictive.remaining_requests,
                    reset_at=most_restrictive.reset_at
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}")
            # Don't block request on error
            return await call_next(request)

