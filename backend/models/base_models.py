"""
Base Pydantic Models for GRETA PAI
Common schemas and base classes for all API models
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from pydantic import BaseModel, Field, validator
from enum import Enum

T = TypeVar('T')


class ResponseStatus(str, Enum):
    """Standard response status values"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class BaseResponse(BaseModel):
    """Base response model for all API responses"""
    
    status: ResponseStatus = Field(
        ...,
        description="Response status indicator"
    )
    message: str = Field(
        ...,
        description="Human-readable response message"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp in UTC"
    )
    request_id: Optional[str] = Field(
        None,
        description="Unique request identifier for tracking"
    )
    
    class Config:
        """Pydantic model configuration"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class SuccessResponse(BaseResponse, Generic[T]):
    """Success response with optional data payload"""
    
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Success status"
    )
    data: Optional[T] = Field(
        None,
        description="Response data payload"
    )


class ErrorResponse(BaseResponse):
    """Error response with detailed error information"""
    
    status: ResponseStatus = Field(
        default=ResponseStatus.ERROR,
        description="Error status"
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error context and details"
    )
    validation_errors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Field-specific validation errors"
    )


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information"""
    
    field: str = Field(
        ...,
        description="Field name that failed validation"
    )
    message: str = Field(
        ...,
        description="Validation error message"
    )
    code: Optional[str] = Field(
        None,
        description="Validation error code"
    )
    value: Optional[Any] = Field(
        None,
        description="Invalid value that caused the error"
    )


class PaginationMeta(BaseModel):
    """Pagination metadata for paginated responses"""
    
    page: int = Field(
        ...,
        ge=1,
        description="Current page number (1-based)"
    )
    per_page: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of items per page"
    )
    total_items: int = Field(
        ...,
        ge=0,
        description="Total number of items available"
    )
    total_pages: int = Field(
        ...,
        ge=0,
        description="Total number of pages"
    )
    has_next: bool = Field(
        ...,
        description="Whether there is a next page"
    )
    has_prev: bool = Field(
        ...,
        description="Whether there is a previous page"
    )
    
    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values):
        """Calculate total pages based on total items and per_page"""
        if 'total_items' in values and 'per_page' in values:
            total_items = values['total_items']
            per_page = values['per_page']
            return (total_items + per_page - 1) // per_page if total_items > 0 else 0
        return v
    
    @validator('has_next', always=True)
    def calculate_has_next(cls, v, values):
        """Calculate if there is a next page"""
        if 'page' in values and 'total_pages' in values:
            return values['page'] < values['total_pages']
        return False
    
    @validator('has_prev', always=True)
    def calculate_has_prev(cls, v, values):
        """Calculate if there is a previous page"""
        if 'page' in values:
            return values['page'] > 1
        return False


class PaginatedResponse(BaseResponse, Generic[T]):
    """Paginated response with metadata"""
    
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Response status"
    )
    data: List[T] = Field(
        ...,
        description="List of items for current page"
    )
    pagination: PaginationMeta = Field(
        ...,
        description="Pagination metadata"
    )


class HealthStatus(str, Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceHealth(BaseModel):
    """Individual service health status"""
    
    name: str = Field(
        ...,
        description="Service name"
    )
    status: HealthStatus = Field(
        ...,
        description="Service health status"
    )
    response_time_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Service response time in milliseconds"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if service is unhealthy"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional service-specific details"
    )
    last_check: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of last health check"
    )


class SystemMetrics(BaseModel):
    """System performance metrics"""
    
    cpu_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="CPU usage percentage"
    )
    memory_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Memory usage percentage"
    )
    disk_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Disk usage percentage"
    )
    uptime_seconds: float = Field(
        ...,
        ge=0,
        description="Application uptime in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metrics collection timestamp"
    )


class APIKeyScope(str, Enum):
    """API key permission scopes"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    VOICE = "voice"
    ANALYTICS = "analytics"


class RateLimitInfo(BaseModel):
    """Rate limiting information"""
    
    limit: int = Field(
        ...,
        ge=1,
        description="Rate limit (requests per window)"
    )
    remaining: int = Field(
        ...,
        ge=0,
        description="Remaining requests in current window"
    )
    reset_time: datetime = Field(
        ...,
        description="When the rate limit window resets"
    )
    window_seconds: int = Field(
        ...,
        ge=1,
        description="Rate limit window duration in seconds"
    )


class FileUploadInfo(BaseModel):
    """File upload information and constraints"""
    
    max_size_bytes: int = Field(
        ...,
        ge=1,
        description="Maximum file size in bytes"
    )
    allowed_extensions: List[str] = Field(
        ...,
        description="List of allowed file extensions"
    )
    allowed_mime_types: List[str] = Field(
        ...,
        description="List of allowed MIME types"
    )


# Export all models
__all__ = [
    "ResponseStatus",
    "BaseResponse",
    "SuccessResponse", 
    "ErrorResponse",
    "ValidationErrorDetail",
    "PaginationMeta",
    "PaginatedResponse",
    "HealthStatus",
    "ServiceHealth",
    "SystemMetrics",
    "APIKeyScope",
    "RateLimitInfo",
    "FileUploadInfo"
]
