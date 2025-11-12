"""
API-specific Pydantic Models for GRETA PAI
Health checks, metrics, system status, and general API response schemas
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from models.base_models import (
    ResponseStatus, BaseResponse, SuccessResponse, ErrorResponse,
    HealthStatus, ServiceHealth, SystemMetrics, RateLimitInfo
)


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    
    status: HealthStatus = Field(
        ...,
        description="Overall system health status"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(
        ...,
        description="Application version"
    )
    uptime: float = Field(
        ...,
        ge=0.0,
        description="System uptime in seconds"
    )
    services: Dict[str, ServiceHealth] = Field(
        ...,
        description="Individual service health status"
    )
    system_metrics: SystemMetrics = Field(
        ...,
        description="System performance metrics"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class MetricsResponse(BaseModel):
    """System metrics response"""
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metrics collection timestamp"
    )
    system: SystemMetrics = Field(
        ...,
        description="System performance metrics"
    )
    api_metrics: Dict[str, Any] = Field(
        ...,
        description="API endpoint metrics"
    )
    database_metrics: Dict[str, Any] = Field(
        ...,
        description="Database performance metrics"
    )
    cache_metrics: Dict[str, Any] = Field(
        ...,
        description="Cache performance metrics"
    )
    rate_limit_metrics: Dict[str, RateLimitInfo] = Field(
        ...,
        description="Rate limiting metrics by endpoint"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class VersionInfo(BaseModel):
    """Application version information"""
    
    version: str = Field(
        ...,
        description="Application version"
    )
    build_date: datetime = Field(
        ...,
        description="Build timestamp"
    )
    git_commit: Optional[str] = Field(
        None,
        description="Git commit hash"
    )
    git_branch: Optional[str] = Field(
        None,
        description="Git branch name"
    )
    environment: str = Field(
        ...,
        description="Deployment environment"
    )
    python_version: str = Field(
        ...,
        description="Python version"
    )
    dependencies: Dict[str, str] = Field(
        ...,
        description="Key dependency versions"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class APIEndpointInfo(BaseModel):
    """API endpoint information"""
    
    path: str = Field(
        ...,
        description="Endpoint path"
    )
    method: str = Field(
        ...,
        description="HTTP method"
    )
    tags: List[str] = Field(
        ...,
        description="Endpoint tags"
    )
    summary: Optional[str] = Field(
        None,
        description="Endpoint summary"
    )
    description: Optional[str] = Field(
        None,
        description="Endpoint description"
    )
    requires_auth: bool = Field(
        ...,
        description="Whether authentication is required"
    )
    rate_limit: Optional[RateLimitInfo] = Field(
        None,
        description="Rate limiting information"
    )


class APIDocumentation(BaseModel):
    """API documentation response"""
    
    title: str = Field(
        ...,
        description="API title"
    )
    description: str = Field(
        ...,
        description="API description"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    base_url: str = Field(
        ...,
        description="API base URL"
    )
    endpoints: List[APIEndpointInfo] = Field(
        ...,
        description="Available endpoints"
    )
    authentication: Dict[str, Any] = Field(
        ...,
        description="Authentication information"
    )
    rate_limits: Dict[str, RateLimitInfo] = Field(
        ...,
        description="Rate limiting information"
    )
    examples: Dict[str, Any] = Field(
        ...,
        description="API usage examples"
    )


class FileUploadResponse(BaseModel):
    """File upload response"""
    
    file_id: str = Field(
        ...,
        description="Unique file identifier"
    )
    filename: str = Field(
        ...,
        description="Original filename"
    )
    size: int = Field(
        ...,
        ge=0,
        description="File size in bytes"
    )
    content_type: str = Field(
        ...,
        description="MIME content type"
    )
    upload_url: Optional[str] = Field(
        None,
        description="URL for file access"
    )
    checksum: str = Field(
        ...,
        description="File checksum (SHA-256)"
    )
    uploaded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Upload timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="File expiration timestamp"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class SearchRequest(BaseModel):
    """Generic search request"""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Search filters"
    )
    sort_by: Optional[str] = Field(
        None,
        description="Sort field"
    )
    sort_order: str = Field(
        default="desc",
        pattern=r'^(asc|desc)$',
        description="Sort order"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number"
    )
    per_page: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate search query"""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()


class SearchResponse(BaseModel):
    """Generic search response"""
    
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Search results"
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of results"
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page"
    )
    per_page: int = Field(
        ...,
        ge=1,
        description="Items per page"
    )
    total_pages: int = Field(
        ...,
        ge=0,
        description="Total number of pages"
    )
    has_next: bool = Field(
        ...,
        description="Whether there are more pages"
    )
    has_prev: bool = Field(
        ...,
        description="Whether there are previous pages"
    )
    query: str = Field(
        ...,
        description="Original search query"
    )
    filters_applied: Optional[Dict[str, Any]] = Field(
        None,
        description="Filters that were applied"
    )
    search_time: float = Field(
        ...,
        ge=0.0,
        description="Search execution time in seconds"
    )


class BulkOperationRequest(BaseModel):
    """Bulk operation request"""
    
    operation: str = Field(
        ...,
        description="Operation type (create, update, delete)"
    )
    items: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="Items to process"
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Operation options"
    )
    
    @validator('operation')
    def validate_operation(cls, v):
        """Validate operation type"""
        allowed_operations = ['create', 'update', 'delete', 'upsert']
        if v not in allowed_operations:
            raise ValueError(f"Operation must be one of: {allowed_operations}")
        return v


class BulkOperationResponse(BaseModel):
    """Bulk operation response"""
    
    operation: str = Field(
        ...,
        description="Operation type"
    )
    total_items: int = Field(
        ...,
        ge=0,
        description="Total items processed"
    )
    successful_items: int = Field(
        ...,
        ge=0,
        description="Successfully processed items"
    )
    failed_items: int = Field(
        ...,
        ge=0,
        description="Failed items"
    )
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Operation results"
    )
    errors: List[Dict[str, Any]] = Field(
        ...,
        description="Error details for failed items"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds"
    )


# Export all API models
__all__ = [
    "HealthCheckResponse",
    "MetricsResponse",
    "VersionInfo",
    "APIEndpointInfo",
    "APIDocumentation",
    "FileUploadResponse",
    "SearchRequest",
    "SearchResponse",
    "BulkOperationRequest",
    "BulkOperationResponse"
]
