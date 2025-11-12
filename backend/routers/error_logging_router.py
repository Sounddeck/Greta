"""
Error Logging Router for CPAS4
Dedicated endpoint for frontend error reporting
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from database import Database, get_database
from services.audit_service import AuditService
from models.audit_models import EventType, SeverityLevel, AuditResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/errors", tags=["Error Logging"])


class FrontendErrorReport(BaseModel):
    """Frontend error report model"""
    message: str = Field(..., description="Error message")
    stack: Optional[str] = Field(None, description="Error stack trace")
    componentStack: Optional[str] = Field(None, description="React component stack")
    timestamp: str = Field(..., description="Error timestamp (ISO format)")
    userAgent: str = Field(..., description="Browser user agent")
    url: str = Field(..., description="Page URL where error occurred")
    userId: Optional[str] = Field(None, description="User ID if authenticated")
    sessionId: Optional[str] = Field(None, description="Session ID")
    additionalContext: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ErrorLogResponse(BaseModel):
    """Error log response"""
    success: bool
    error_id: str
    message: str


@router.post("/log", response_model=ErrorLogResponse, status_code=201)
async def log_frontend_error(
    error_report: FrontendErrorReport,
    request: Request,
    db: Database = None
):
    """
    Log frontend error to backend
    
    This endpoint receives error reports from the frontend ErrorBoundary
    and stores them in the audit log for tracking and debugging.
    """
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Create audit service
        if db is None:
            from database import database as global_db
            db = global_db
        
        audit_service = AuditService(db)
        
        # Determine severity based on error type
        severity = SeverityLevel.ERROR
        if "TypeError" in error_report.message or "ReferenceError" in error_report.message:
            severity = SeverityLevel.CRITICAL
        elif "Warning" in error_report.message:
            severity = SeverityLevel.WARNING
        
        # Log to audit system
        audit_log = await audit_service.log_event(
            event_type=EventType.SYSTEM_ERROR,
            user_id=error_report.userId or "anonymous",
            severity=severity,
            resource_type="frontend",
            resource_id=error_report.url,
            action="error_occurred",
            result=AuditResult.FAILURE,
            ip_address=client_ip,
            user_agent=error_report.userAgent,
            request_method="CLIENT_ERROR",
            request_path=error_report.url,
            error_message=error_report.message,
            metadata={
                "stack": error_report.stack,
                "componentStack": error_report.componentStack,
                "timestamp": error_report.timestamp,
                "sessionId": error_report.sessionId,
                "additionalContext": error_report.additionalContext,
            }
        )
        
        # Also log to application logger
        logger.error(
            f"Frontend error: {error_report.message}",
            extra={
                "url": error_report.url,
                "user_id": error_report.userId,
                "user_agent": error_report.userAgent,
                "stack": error_report.stack,
                "component_stack": error_report.componentStack,
            }
        )
        
        return ErrorLogResponse(
            success=True,
            error_id=audit_log.log_id,
            message="Error logged successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to log frontend error: {e}")
        # Don't fail the request even if logging fails
        return ErrorLogResponse(
            success=False,
            error_id="",
            message=f"Failed to log error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "error_logging",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/stats")
async def get_error_stats(db: Database = None):
    """
    Get error statistics
    
    Returns counts of frontend errors by severity and time period
    """
    try:
        if db is None:
            from database import database as global_db
            db = global_db
        
        audit_service = AuditService(db)
        
        # Get errors from last 24 hours
        from datetime import timedelta
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        # Query frontend errors
        errors = await audit_service.query_audit_logs(
            event_type=EventType.SYSTEM_ERROR,
            resource_type="frontend",
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        # Calculate statistics
        total_errors = len(errors)
        by_severity = {}
        by_url = {}
        
        for error in errors:
            # Count by severity
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by URL
            url = error.resource_id or "unknown"
            by_url[url] = by_url.get(url, 0) + 1
        
        # Get top 5 URLs with most errors
        top_urls = sorted(by_url.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_errors_24h": total_errors,
            "by_severity": by_severity,
            "top_error_urls": [{"url": url, "count": count} for url, count in top_urls],
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get error stats: {e}")
        raise HTTPException(500, f"Failed to get error stats: {str(e)}")

