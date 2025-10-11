"""
Report Models for GRETA PAI
Pydantic models for report generation and management
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ReportType(str, Enum):
    """Report types"""
    TASK_SUMMARY = "task_summary"
    AGENT_PERFORMANCE = "agent_performance"
    KNOWLEDGE_STATS = "knowledge_stats"
    SYSTEM_OVERVIEW = "system_overview"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Report output formats"""
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    HTML = "html"


class ReportStatus(str, Enum):
    """Report generation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportGenerateRequest(BaseModel):
    """Request to generate a new report"""
    title: str = Field(..., description="Report title")
    report_type: ReportType = Field(..., description="Type of report to generate")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters to apply")
    format: ReportFormat = Field(default=ReportFormat.JSON, description="Output format")
    date_range: Optional[Dict[str, str]] = Field(default=None, description="Date range for report")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Weekly Task Summary",
                "report_type": "task_summary",
                "filters": {"agent_type": "general"},
                "format": "json",
                "date_range": {"start": "2025-10-01", "end": "2025-10-07"}
            }
        }


class ReportResponse(BaseModel):
    """Report response model"""
    id: str = Field(..., description="Report ID")
    title: str = Field(..., description="Report title")
    report_type: ReportType = Field(..., description="Report type")
    status: ReportStatus = Field(..., description="Generation status")
    format: ReportFormat = Field(..., description="Output format")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    download_url: Optional[str] = Field(default=None, description="Download URL if completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "report_123456",
                "title": "Weekly Task Summary",
                "report_type": "task_summary",
                "status": "completed",
                "format": "json",
                "created_at": "2025-10-09T16:00:00Z",
                "completed_at": "2025-10-09T16:00:05Z",
                "download_url": "/api/v1/reports/report_123456/download"
            }
        }


class ReportListResponse(BaseModel):
    """List of reports"""
    reports: List[ReportResponse] = Field(..., description="List of reports")
    total: int = Field(..., description="Total number of reports")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=10, description="Page size")


class ReportStatistics(BaseModel):
    """Report statistics"""
    total_reports: int = Field(..., description="Total number of reports")
    reports_by_type: Dict[str, int] = Field(..., description="Reports grouped by type")
    reports_by_status: Dict[str, int] = Field(..., description="Reports grouped by status")
    recent_reports: List[ReportResponse] = Field(..., description="Recent reports")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_reports": 42,
                "reports_by_type": {
                    "task_summary": 15,
                    "agent_performance": 10,
                    "knowledge_stats": 8,
                    "system_overview": 9
                },
                "reports_by_status": {
                    "completed": 38,
                    "failed": 2,
                    "processing": 2
                },
                "recent_reports": []
            }
        }
