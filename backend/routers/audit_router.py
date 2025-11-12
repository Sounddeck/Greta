"""
Audit Logging Router for CPAS4
REST API endpoints for audit log retrieval and analysis
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from services.audit_service import AuditService
from models.audit_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/audit", tags=["Audit"])


async def get_service() -> AuditService:
    db = Database()
    await db.connect()
    return AuditService(db)


# Audit Log Query Endpoints
@router.get("/logs", response_model=List[AuditLogEntry])
async def query_audit_logs(
    user_id: Optional[str] = None,
    event_type: Optional[EventType] = None,
    severity: Optional[SeverityLevel] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    result: Optional[AuditResult] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    svc: AuditService = Depends(get_service)
):
    """Query audit logs with filters
    
    Requires RBAC permission: audit:read or audit:admin
    """
    # Parse dates
    start_datetime = datetime.fromisoformat(start_date) if start_date else None
    end_datetime = datetime.fromisoformat(end_date) if end_date else None
    
    return await svc.query_audit_logs(
        user_id=user_id,
        event_type=event_type,
        severity=severity,
        resource_type=resource_type,
        resource_id=resource_id,
        result=result,
        start_date=start_datetime,
        end_date=end_datetime,
        limit=limit,
        offset=offset
    )


@router.get("/logs/{log_id}", response_model=AuditLogEntry)
async def get_audit_log(
    log_id: str,
    svc: AuditService = Depends(get_service)
):
    """Get specific audit log entry
    
    Requires RBAC permission: audit:read or audit:admin
    """
    log = await svc.get_audit_log(log_id)
    if not log:
        raise HTTPException(404, "Audit log not found")
    return log


@router.get("/users/{user_id}/activity", response_model=UserActivity)
async def get_user_activity(
    user_id: str,
    limit: int = Query(100, le=1000),
    svc: AuditService = Depends(get_service)
):
    """Get user activity logs
    
    Requires RBAC permission: audit:read or audit:admin
    """
    return await svc.get_user_activity(user_id, limit)


@router.get("/events/recent", response_model=RecentEvents)
async def get_recent_events(
    limit: int = Query(100, le=1000),
    svc: AuditService = Depends(get_service)
):
    """Get recent audit events
    
    Requires RBAC permission: audit:read or audit:admin
    """
    return await svc.get_recent_events(limit)


@router.get("/stats", response_model=AuditStatistics)
async def get_audit_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    svc: AuditService = Depends(get_service)
):
    """Get audit statistics
    
    Requires RBAC permission: audit:read or audit:admin
    """
    # Parse dates
    start_datetime = datetime.fromisoformat(start_date) if start_date else None
    end_datetime = datetime.fromisoformat(end_date) if end_date else None
    
    return await svc.get_audit_statistics(start_datetime, end_datetime)


# Convenience Endpoints
@router.get("/events/authentication", response_model=List[AuditLogEntry])
async def get_authentication_events(
    user_id: Optional[str] = None,
    limit: int = Query(100, le=1000),
    svc: AuditService = Depends(get_service)
):
    """Get authentication events (login, logout, failed attempts)
    
    Requires RBAC permission: audit:read or audit:admin
    """
    # Get all authentication-related events
    auth_events = []
    
    for event_type in [EventType.LOGIN, EventType.LOGOUT, EventType.LOGIN_FAILED]:
        events = await svc.query_audit_logs(
            user_id=user_id,
            event_type=event_type,
            limit=limit
        )
        auth_events.extend(events)
    
    # Sort by timestamp descending
    auth_events.sort(key=lambda x: x.timestamp, reverse=True)
    
    return auth_events[:limit]


@router.get("/events/security", response_model=List[AuditLogEntry])
async def get_security_events(
    limit: int = Query(100, le=1000),
    svc: AuditService = Depends(get_service)
):
    """Get security events (alerts, rate limits, suspicious activity)
    
    Requires RBAC permission: audit:admin
    """
    # Get all security-related events
    security_events = []
    
    for event_type in [
        EventType.SECURITY_ALERT,
        EventType.RATE_LIMIT_EXCEEDED,
        EventType.SUSPICIOUS_ACTIVITY,
        EventType.ACCESS_DENIED
    ]:
        events = await svc.query_audit_logs(
            event_type=event_type,
            limit=limit
        )
        security_events.extend(events)
    
    # Sort by timestamp descending
    security_events.sort(key=lambda x: x.timestamp, reverse=True)
    
    return security_events[:limit]


@router.get("/events/failures", response_model=List[AuditLogEntry])
async def get_failure_events(
    limit: int = Query(100, le=1000),
    svc: AuditService = Depends(get_service)
):
    """Get failed operations
    
    Requires RBAC permission: audit:read or audit:admin
    """
    return await svc.query_audit_logs(
        result=AuditResult.FAILURE,
        limit=limit
    )


@router.get("/events/critical", response_model=List[AuditLogEntry])
async def get_critical_events(
    limit: int = Query(100, le=1000),
    svc: AuditService = Depends(get_service)
):
    """Get critical severity events
    
    Requires RBAC permission: audit:admin
    """
    return await svc.query_audit_logs(
        severity=SeverityLevel.CRITICAL,
        limit=limit
    )


# Manual Logging Endpoint (for testing)
@router.post("/logs", response_model=AuditLogEntry, status_code=201)
async def create_audit_log(
    log_create: AuditLogCreate,
    svc: AuditService = Depends(get_service)
):
    """Manually create audit log entry (for testing)
    
    Requires RBAC permission: audit:admin
    """
    return await svc.log_event(
        event_type=log_create.event_type,
        user_id=log_create.user_id,
        severity=log_create.severity,
        resource_type=log_create.resource_type,
        resource_id=log_create.resource_id,
        action=log_create.action,
        result=log_create.result,
        ip_address=log_create.ip_address,
        user_agent=log_create.user_agent,
        request_method=log_create.request_method,
        request_path=log_create.request_path,
        request_body=log_create.request_body,
        response_status=log_create.response_status,
        response_time_ms=log_create.response_time_ms,
        error_message=log_create.error_message,
        metadata=log_create.metadata
    )


# Health Check
@router.get("/health")
async def health_check(svc: AuditService = Depends(get_service)):
    """Audit module health check"""
    # Get recent logs count
    recent = await svc.get_recent_events(limit=10)
    
    return {
        "status": "healthy",
        "module": "audit",
        "recent_events": recent.count,
        "timestamp": datetime.utcnow().isoformat()
    }


# Module Statistics
@router.get("/module-stats")
async def get_module_stats(svc: AuditService = Depends(get_service)):
    """Get audit module statistics"""
    # Get stats for last 24 hours
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)
    
    stats = await svc.get_audit_statistics(start_date, end_date)
    
    return {
        "total_logs_24h": stats.total_logs,
        "unique_users_24h": stats.unique_users,
        "by_severity": {s.severity.value: s.count for s in stats.by_severity},
        "by_result": stats.by_result,
        "top_event_types": [
            {"event_type": s.event_type.value, "count": s.count}
            for s in sorted(stats.by_event_type, key=lambda x: x.count, reverse=True)[:5]
        ]
    }

