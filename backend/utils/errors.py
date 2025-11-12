"""
GRETA PAI Centralized Error Handling
Phase 2 Task 2.4 - Global error handling system
"""

from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class GretaError(Exception):
    """
    Base exception class for GRETA PAI
    All custom exceptions should inherit from this class
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.error_code = error_code or "GRETA_ERROR"
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON response"""
        return {
            "error": {
                "message": self.message,
                "code": self.error_code,
                "status_code": self.status_code,
                "details": self.details,
                "timestamp": datetime.utcnow().isoformat()
            }
        }


# Specific error classes

class AuthenticationError(GretaError):
    """Authentication failed"""
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
            error_code="AUTH_ERROR"
        )


class AuthorizationError(GretaError):
    """Authorization failed - insufficient permissions"""
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
            error_code="AUTHZ_ERROR"
        )


class ResourceNotFoundError(GretaError):
    """Resource not found"""
    def __init__(self, resource: str, resource_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"{resource} with ID '{resource_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details or {"resource": resource, "resource_id": resource_id},
            error_code="NOT_FOUND"
        )


class ValidationError(GretaError):
    """Data validation failed"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
            error_code="VALIDATION_ERROR"
        )


class DatabaseError(GretaError):
    """Database operation failed"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            error_code="DATABASE_ERROR"
        )


class ServiceUnavailableError(GretaError):
    """External service unavailable"""
    def __init__(self, service: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Service '{service}' is currently unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details or {"service": service},
            error_code="SERVICE_UNAVAILABLE"
        )


class RateLimitError(GretaError):
    """Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
            error_code="RATE_LIMIT"
        )


class ConfigurationError(GretaError):
    """Configuration error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            error_code="CONFIG_ERROR"
        )


# Global error handlers

async def greta_error_handler(request: Request, exc: GretaError) -> JSONResponse:
    """
    Handle GretaError exceptions
    Returns a standardized JSON error response
    """
    logger.error(
        f"GretaError: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions from Starlette/FastAPI
    Converts them to standardized error format
    """
    logger.warning(
        f"HTTP Exception: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "code": "HTTP_ERROR",
                "status_code": exc.status_code,
                "details": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors
    Returns detailed validation error information
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        f"Validation Error: {len(errors)} validation errors",
        extra={
            "errors": errors,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Request validation failed",
                "code": "VALIDATION_ERROR",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "details": {"validation_errors": errors},
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other unhandled exceptions
    Returns a generic error response and logs the full traceback
    """
    # Log the full traceback for debugging
    logger.error(
        f"Unhandled Exception: {str(exc)}",
        extra={
            "exception_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc()
        }
    )
    
    # In production, don't expose internal error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An internal server error occurred",
                "code": "INTERNAL_ERROR",
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "details": {
                    "exception_type": type(exc).__name__,
                    # Only include exception message in development
                    # "exception_message": str(exc)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


def register_error_handlers(app):
    """
    Register all error handlers with the FastAPI application
    Call this function in main.py after creating the app
    """
    # Custom GRETA errors
    app.add_exception_handler(GretaError, greta_error_handler)
    
    # HTTP exceptions
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Generic exception handler (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("✅ Error handlers registered successfully")


# Utility functions for raising errors

def raise_not_found(resource: str, resource_id: str):
    """Convenience function to raise ResourceNotFoundError"""
    raise ResourceNotFoundError(resource, resource_id)


def raise_validation_error(message: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function to raise ValidationError"""
    raise ValidationError(message, details)


def raise_auth_error(message: str = "Authentication failed"):
    """Convenience function to raise AuthenticationError"""
    raise AuthenticationError(message)


def raise_authz_error(message: str = "Insufficient permissions"):
    """Convenience function to raise AuthorizationError"""
    raise AuthorizationError(message)


def raise_service_unavailable(service: str):
    """Convenience function to raise ServiceUnavailableError"""
    raise ServiceUnavailableError(service)
