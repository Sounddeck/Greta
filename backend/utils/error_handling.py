"""
GRETA PAI - Advanced Error Handling & Logging
Phase 3 Code Quality Enhancements
Comprehensive error handling, logging standardization, and monitoring
"""
import os
import sys
import traceback
import inspect
from typing import Any, Dict, Optional, Union, Callable, Type, List
from functools import wraps
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncio
import hashlib
import json

from loguru import logger
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class GretaException(Exception):
    """Base exception class for GRETA PAI"""

    def __init__(self, message: str, error_code: str = "GRETA_ERROR", status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)


class ValidationError(GretaException):
    """Validation-related errors"""
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        super().__init__(message, "VALIDATION_ERROR", 400, {
            "field": field,
            "value": str(value)[:100] if value is not None else None,
            **kwargs
        })


class AuthenticationError(GretaException):
    """Authentication failures"""
    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(message, "AUTH_ERROR", 401, kwargs)


class AuthorizationError(GretaException):
    """Authorization failures"""
    def __init__(self, message: str = "Access denied", required_scopes: List[str] = None, **kwargs):
        super().__init__(message, "AUTHZ_ERROR", 403, {
            "required_scopes": required_scopes or [],
            **kwargs
        })


class ResourceNotFoundError(GretaException):
    """Resource not found errors"""
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            "NOT_FOUND",
            404,
            {"resource_type": resource_type, "resource_id": resource_id, **kwargs}
        )


class RateLimitError(GretaException):
    """Rate limiting errors"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60, **kwargs):
        super().__init__(message, "RATE_LIMIT", 429, {
            "retry_after": retry_after,
            **kwargs
        })


class ExternalServiceError(GretaException):
    """External service communication errors"""
    def __init__(self, service_name: str, original_error: str = None, **kwargs):
        super().__init__(
            f"External service error: {service_name}",
            "EXTERNAL_ERROR",
            502,
            {"service_name": service_name, "original_error": original_error, **kwargs}
        )


class ConfigurationError(GretaException):
    """Configuration-related errors"""
    def __init__(self, config_key: str, issue: str, **kwargs):
        super().__init__(
            f"Configuration error: {config_key} - {issue}",
            "CONFIG_ERROR",
            500,
            {"config_key": config_key, "issue": issue, **kwargs}
        )


class ErrorContext:
    """Context manager for error handling with automatic logging"""

    def __init__(self, operation: str, context: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.context = context or {}
        self.start_time = datetime.utcnow()

    async def __aenter__(self):
        logger.debug(f"Starting operation: {self.operation}", extra={
            "operation": self.operation,
            "context": self.context,
            "start_time": self.start_time.isoformat()
        })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()

        if exc_type:
            # Log the exception with full context
            error_details = {
                "operation": self.operation,
                "context": self.context,
                "duration_seconds": duration,
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_val),
                "traceback": traceback.format_exception(exc_type, exc_val, exc_tb)
            }
            logger.error(f"Operation failed: {self.operation}", extra=error_details)

            # Re-raise the exception
            raise exc_val
        else:
            logger.debug(f"Operation completed: {self.operation}", extra={
                "operation": self.operation,
                "context": self.context,
                "duration_seconds": duration,
                "success": True
            })


class ErrorHandler:
    """Centralized error handling and response formatting"""

    ERROR_CODE_MAP = {
        "VALIDATION_ERROR": {"status_code": 400, "description": "Input validation failed"},
        "AUTH_ERROR": {"status_code": 401, "description": "Authentication required"},
        "AUTHZ_ERROR": {"status_code": 403, "description": "Access denied"},
        "NOT_FOUND": {"status_code": 404, "description": "Resource not found"},
        "RATE_LIMIT": {"status_code": 429, "description": "Rate limit exceeded"},
        "EXTERNAL_ERROR": {"status_code": 502, "description": "External service error"},
        "CONFIG_ERROR": {"status_code": 500, "description": "Configuration error"},
        "INTERNAL_ERROR": {"status_code": 500, "description": "Internal server error"}
    }

    @staticmethod
    def format_error_response(error: Exception, request: Optional[Request] = None) -> JSONResponse:
        """Format any exception into a standardized JSON response"""

        if isinstance(error, GretaException):
            status_code = error.status_code
            error_response = {
                "success": False,
                "error": error.message,
                "error_code": error.error_code,
                "timestamp": error.timestamp.isoformat(),
                "details": error.details
            }
        else:
            # Handle non-Greta exceptions
            error_info = ErrorHandler.ERROR_CODE_MAP.get("INTERNAL_ERROR")
            status_code = error_info["status_code"]

            error_response = {
                "success": False,
                "error": "An unexpected error occurred",
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {}
            }

        # Add request context if available
        if request:
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            error_response["request_id"] = hashlib.md5(
                f"{client_ip}{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:8]
            error_response["path"] = str(request.url)

            # Log the full error for debugging
            logger.error(
                f"Unhandled exception on {request.method} {request.url}: {str(error)}",
                extra={
                    "method": request.method,
                    "path": str(request.url),
                    "client_ip": client_ip,
                    "user_agent": request.headers.get("user-agent", "unknown"),
                    "request_id": error_response["request_id"],
                    "traceback": traceback.format_exc() if not isinstance(error, GretaException) else None
                }
            )

        return JSONResponse(status_code=status_code, content=error_response)

    @staticmethod
    async def handle_generic_exception(request: Request, exc: Exception):
        """Global exception handler for FastAPI"""
        return ErrorHandler.format_error_response(exc, request)


class LoggingManager:
    """Advanced logging configuration and utilities"""

    LOG_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level> | "
        "{extra}"
    )

    @staticmethod
    def setup_logging(
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_file: str = "logs/greta.log",
        max_file_size: str = "10 MB",
        retention: str = "7 days"
    ):
        """Configure comprehensive logging"""

        # Remove default logger
        logger.remove()

        # Console logging
        logger.add(
            sys.stdout,
            format=LoggingManager.LOG_FORMAT,
            level=log_level,
            colorize=True,
            enqueue=True  # Thread-safe
        )

        # File logging if enabled
        if log_to_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logger.add(
                log_file,
                format=LoggingManager.LOG_FORMAT,
                level=log_level,
                rotation=max_file_size,
                retention=retention,
                encoding="utf-8",
                enqueue=True,
                serialize=True  # JSON format for file logs
            )

        # Add context-aware logging
        logger.configure(
            extra={"service": "greta-pai", "version": "2.0.0"}
        )

        logger.info("Logging system initialized", extra={
            "log_level": log_level,
            "log_to_file": log_to_file,
            "log_file": log_file
        })

    @staticmethod
    def create_operation_logger(operation_name: str):
        """Create a logger specifically for an operation"""

        def operation_logger(message: str, level: str = "INFO", **kwargs):
            extra = {
                "operation": operation_name,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
            getattr(logger, level.lower())(f"[{operation_name}] {message}", extra=extra)

        return operation_logger


class MetricsCollector:
    """Error and performance metrics collection"""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_samples: Dict[str, List[Dict]] = {}
        self.performance_metrics: Dict[str, List[float]] = {}

    def record_error(self, error_code: str, details: Dict[str, Any]):
        """Record an error occurrence"""
        if error_code not in self.error_counts:
            self.error_counts[error_code] = 0
            self.error_samples[error_code] = []

        self.error_counts[error_code] += 1

        # Keep only last 10 error samples
        if len(self.error_samples[error_code]) >= 10:
            self.error_samples[error_code].pop(0)

        self.error_samples[error_code].append({
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        })

    def record_performance(self, operation: str, duration: float):
        """Record operation performance"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []

        self.performance_metrics[operation].append(duration)

        # Keep only last 100 measurements
        if len(self.performance_metrics[operation]) > 100:
            self.performance_metrics[operation] = self.performance_metrics[operation][-100:]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "error_counts": self.error_counts,
            "recent_errors": {
                code: samples[-5:] for code, samples in self.error_samples.items()
            },
            "performance_stats": {
                op: {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations) if durations else 0,
                    "max_duration": max(durations) if durations else 0,
                    "min_duration": min(durations) if durations else 0
                }
                for op, durations in self.performance_metrics.items()
            }
        }


# Global instances
error_handler = ErrorHandler()
logging_manager = LoggingManager()
metrics_collector = MetricsCollector()

# Setup logging on import
logging_manager.setup_logging()


def handle_errors(operation_name: str = None):
    """Decorator for comprehensive error handling"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            async with ErrorContext(op_name, {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except GretaException as e:
                    metrics_collector.record_error(e.error_code, e.details)
                    raise
                except Exception as e:
                    metrics_collector.record_error("INTERNAL_ERROR", {"original_error": str(e)})
                    raise GretaException(f"Unexpected error in {op_name}: {str(e)}") from e

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            with ErrorContext(op_name, {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}):
                try:
                    result = func(*args, **kwargs)
                    return result
                except GretaException as e:
                    metrics_collector.record_error(e.error_code, e.details)
                    raise
                except Exception as e:
                    metrics_collector.record_error("INTERNAL_ERROR", {"original_error": str(e)})
                    raise GretaException(f"Unexpected error in {op_name}: {str(e)}") from e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


@asynccontextmanager
async def error_context(operation: str, **context):
    """Context manager for operation error handling"""
    async with ErrorContext(operation, context):
        yield


__all__ = [
    'GretaException',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'ResourceNotFoundError',
    'RateLimitError',
    'ExternalServiceError',
    'ConfigurationError',
    'ErrorContext',
    'ErrorHandler',
    'LoggingManager',
    'MetricsCollector',
    'error_handler',
    'logging_manager',
    'metrics_collector',
    'handle_errors',
    'error_context'
]
