"""
Audit Middleware for CPAS4
Automatic logging of all API requests and responses
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from database import Database
from services.audit_service import AuditService
from models.audit_models import EventType, SeverityLevel, AuditResult

logger = logging.getLogger(__name__)


# Paths to exclude from audit logging
EXCLUDED_PATHS = [
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/static/",
    "/api/v1/audit/health",  # Don't log audit health checks
]


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log all API requests"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.db = None
        self.audit_service = None
    
    async def _ensure_service(self):
        """Ensure audit service is initialized"""
        if not self.audit_service:
            try:
                self.db = Database()
                await self.db.connect()
                self.audit_service = AuditService(self.db)
            except Exception as e:
                logger.error(f"Failed to initialize audit service: {e}")
    
    def _should_log(self, path: str) -> bool:
        """Check if path should be logged"""
        # Exclude health checks and static files
        for excluded in EXCLUDED_PATHS:
            if path.startswith(excluded):
                return False
        return True
    
    def _extract_user_id(self, request: Request) -> str:
        """Extract user ID from request"""
        # Try to get from headers
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        # Try to get from query params
        user_id = request.query_params.get("user_id")
        if user_id:
            return user_id
        
        # Try to get from path params
        if hasattr(request, "path_params"):
            user_id = request.path_params.get("user_id")
            if user_id:
                return user_id
        
        return "anonymous"
    
    def _get_event_type(self, method: str, path: str, status_code: int) -> EventType:
        """Determine event type from request"""
        # Authentication endpoints
        if "/login" in path:
            return EventType.LOGIN if status_code < 400 else EventType.LOGIN_FAILED
        if "/logout" in path:
            return EventType.LOGOUT
        if "/password" in path:
            return EventType.PASSWORD_CHANGE
        
        # Authorization endpoints
        if "/permission" in path or "/check" in path:
            return EventType.PERMISSION_CHECK
        if "/roles" in path and method == "POST":
            return EventType.ROLE_ASSIGNED
        if "/roles" in path and method == "DELETE":
            return EventType.ROLE_REVOKED
        
        # CRUD operations
        if method == "POST":
            return EventType.CREATE
        elif method == "GET":
            return EventType.READ
        elif method in ["PUT", "PATCH"]:
            return EventType.UPDATE
        elif method == "DELETE":
            return EventType.DELETE
        
        return EventType.CUSTOM
    
    def _get_severity(self, status_code: int) -> SeverityLevel:
        """Determine severity from status code"""
        if status_code >= 500:
            return SeverityLevel.CRITICAL
        elif status_code >= 400:
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.INFO
    
    def _get_result(self, status_code: int) -> AuditResult:
        """Determine result from status code"""
        if status_code < 400:
            return AuditResult.SUCCESS
        else:
            return AuditResult.FAILURE
    
    def _extract_resource_info(self, path: str) -> tuple:
        """Extract resource type and ID from path"""
        # Parse path to extract resource info
        parts = path.split("/")
        
        # Look for resource type and ID patterns
        resource_type = None
        resource_id = None
        
        # Common patterns: /api/v1/{resource_type}/{resource_id}
        if len(parts) >= 4:
            resource_type = parts[3]  # e.g., "agents", "tasks", "roles"
            if len(parts) >= 5 and not parts[4].startswith("?"):
                resource_id = parts[4]
        
        return resource_type, resource_id
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log audit event"""
        # Check if we should log this request
        if not self._should_log(request.url.path):
            return await call_next(request)
        
        # Ensure audit service is initialized
        await self._ensure_service()
        
        if not self.audit_service:
            # If audit service failed to initialize, just pass through
            return await call_next(request)
        
        # Capture request details
        start_time = time.time()
        user_id = self._extract_user_id(request)
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        method = request.method
        path = request.url.path
        
        # Extract resource info
        resource_type, resource_id = self._extract_resource_info(path)
        
        # Process request
        response = None
        error_message = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            error_message = str(e)
            status_code = 500
            # Re-raise the exception
            raise
        finally:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine event details
            event_type = self._get_event_type(method, path, status_code)
            severity = self._get_severity(status_code)
            result = self._get_result(status_code)
            
            # Log the event
            try:
                await self.audit_service.log_event(
                    event_type=event_type,
                    user_id=user_id if user_id != "anonymous" else None,
                    severity=severity,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    action=method,
                    result=result,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    request_method=method,
                    request_path=path,
                    response_status=status_code,
                    response_time_ms=round(response_time_ms, 2),
                    error_message=error_message,
                    metadata={
                        "query_params": dict(request.query_params),
                        "path_params": dict(request.path_params) if hasattr(request, "path_params") else {}
                    }
                )
            except Exception as e:
                logger.error(f"Failed to log audit event: {e}")
        
        return response


def get_audit_middleware() -> AuditMiddleware:
    """Get audit middleware instance"""
    return AuditMiddleware

