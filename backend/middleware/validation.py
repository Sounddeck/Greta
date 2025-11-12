"""
Validation Middleware for GRETA PAI
Automatic request validation with detailed error responses
"""

import json
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from utils.validators import ValidationError as CustomValidationError
from models.base_models import ErrorResponse, ValidationErrorDetail

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic request validation and error handling
    """
    
    def __init__(self, app, enable_request_logging: bool = True):
        super().__init__(app)
        self.enable_request_logging = enable_request_logging
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through validation middleware
        """
        request_id = getattr(request.state, 'request_id', None)
        
        try:
            # Log incoming request if enabled
            if self.enable_request_logging:
                await self._log_request(request, request_id)
            
            # Process the request
            response = await call_next(request)
            
            # Log response if enabled
            if self.enable_request_logging:
                await self._log_response(request, response, request_id)
            
            return response
            
        except ValidationError as e:
            # Handle Pydantic validation errors
            return await self._handle_validation_error(e, request_id)
            
        except CustomValidationError as e:
            # Handle custom validation errors
            return await self._handle_custom_validation_error(e, request_id)
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return await self._handle_http_exception(e, request_id)
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(
                f"Unexpected error in validation middleware",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'path': request.url.path,
                    'method': request.method
                },
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    message="Internal server error",
                    error_code="INTERNAL_ERROR",
                    request_id=request_id
                ).dict()
            )
    
    async def _log_request(self, request: Request, request_id: Optional[str]):
        """Log incoming request details"""
        try:
            # Get request body for logging (be careful with large payloads)
            body = None
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    body_bytes = await request.body()
                    if len(body_bytes) < 10000:  # Only log small payloads
                        body = body_bytes.decode('utf-8')
                except Exception:
                    body = "<unable to read body>"
            
            logger.info(
                f"Incoming request: {request.method} {request.url.path}",
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'path': request.url.path,
                    'query_params': dict(request.query_params),
                    'headers': dict(request.headers),
                    'body': body,
                    'client_ip': request.client.host if request.client else None
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log request: {e}")
    
    async def _log_response(self, request: Request, response: Response, request_id: Optional[str]):
        """Log response details"""
        try:
            logger.info(
                f"Response: {response.status_code} for {request.method} {request.url.path}",
                extra={
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'method': request.method,
                    'path': request.url.path,
                    'response_headers': dict(response.headers)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log response: {e}")
    
    async def _handle_validation_error(self, error: ValidationError, request_id: Optional[str]) -> JSONResponse:
        """Handle Pydantic validation errors"""
        validation_errors = []
        
        for err in error.errors():
            field_path = '.'.join(str(loc) for loc in err['loc'])
            validation_errors.append(
                ValidationErrorDetail(
                    field=field_path,
                    message=err['msg'],
                    invalid_value=err.get('input'),
                    error_type=err['type']
                ).dict()
            )
        
        logger.warning(
            f"Validation error occurred",
            extra={
                'request_id': request_id,
                'validation_errors': validation_errors,
                'error_count': len(validation_errors)
            }
        )
        
        error_response = ErrorResponse(
            message="Validation failed",
            error_code="VALIDATION_ERROR",
            validation_errors=validation_errors,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.dict()
        )
    
    async def _handle_custom_validation_error(self, error: CustomValidationError, request_id: Optional[str]) -> JSONResponse:
        """Handle custom validation errors"""
        validation_error = ValidationErrorDetail(
            field=error.field,
            message=str(error),
            invalid_value=error.value,
            error_type=error.code
        )
        
        logger.warning(
            f"Custom validation error occurred",
            extra={
                'request_id': request_id,
                'field': error.field,
                'error_code': error.code,
                'error_message': str(error)
            }
        )
        
        error_response = ErrorResponse(
            message="Validation failed",
            error_code="VALIDATION_ERROR",
            validation_errors=[validation_error.dict()],
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.dict()
        )
    
    async def _handle_http_exception(self, error: HTTPException, request_id: Optional[str]) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""
        logger.warning(
            f"HTTP exception occurred",
            extra={
                'request_id': request_id,
                'status_code': error.status_code,
                'detail': error.detail
            }
        )
        
        error_response = ErrorResponse(
            message=str(error.detail),
            error_code=f"HTTP_{error.status_code}",
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=error.status_code,
            content=error_response.dict()
        )


def create_validation_middleware(enable_request_logging: bool = True) -> ValidationMiddleware:
    """
    Factory function to create validation middleware
    
    Args:
        enable_request_logging: Whether to enable detailed request/response logging
        
    Returns:
        Configured ValidationMiddleware instance
    """
    return ValidationMiddleware(
        app=None,  # Will be set by FastAPI
        enable_request_logging=enable_request_logging
    )


# Validation helper functions
def validate_request_data(data: Dict[str, Any], required_fields: list = None) -> Dict[str, Any]:
    """
    Validate request data against required fields
    
    Args:
        data: Request data to validate
        required_fields: List of required field names
        
    Returns:
        Validated data
        
    Raises:
        CustomValidationError: If validation fails
    """
    if required_fields:
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise CustomValidationError(
                field=', '.join(missing_fields),
                value=None,
                code="MISSING_REQUIRED_FIELDS",
                message=f"Missing required fields: {', '.join(missing_fields)}"
            )
    
    return data


def validate_pagination_params(page: int = 1, per_page: int = 20, max_per_page: int = 100) -> Dict[str, int]:
    """
    Validate pagination parameters
    
    Args:
        page: Page number (1-based)
        per_page: Items per page
        max_per_page: Maximum allowed items per page
        
    Returns:
        Validated pagination parameters
        
    Raises:
        CustomValidationError: If validation fails
    """
    if page < 1:
        raise CustomValidationError(
            field="page",
            value=page,
            code="INVALID_PAGE",
            message="Page number must be 1 or greater"
        )
    
    if per_page < 1:
        raise CustomValidationError(
            field="per_page",
            value=per_page,
            code="INVALID_PER_PAGE",
            message="Items per page must be 1 or greater"
        )
    
    if per_page > max_per_page:
        raise CustomValidationError(
            field="per_page",
            value=per_page,
            code="INVALID_PER_PAGE",
            message=f"Items per page cannot exceed {max_per_page}"
        )
    
    return {
        "page": page,
        "per_page": per_page,
        "offset": (page - 1) * per_page
    }


# Export validation middleware and utilities
__all__ = [
    "ValidationMiddleware",
    "create_validation_middleware",
    "validate_request_data",
    "validate_pagination_params"
]
