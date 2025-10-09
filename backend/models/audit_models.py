"""
Audit Logging Models for CPAS4
Pydantic models for audit logs, events, and activity tracking
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# Event Type Enums
class EventType(str, Enum):
    """Types of events that can be audited"""
    # Authentication Events
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    MFA_VERIFIED = "mfa_verified"
    MFA_FAILED = "mfa_failed"
    
    # Authorization Events
    PERMISSION_CHECK = "permission_check"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    
    # CRUD Operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGE = "configuration_change"
    
    # Security Events
    SECURITY_ALERT = "security_alert"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    
    # Integration Events
    INTEGRATION_CREATED = "integration_created"
    INTEGRATION_UPDATED = "integration_updated"
    INTEGRATION_DELETED = "integration_deleted"
    INTEGRATION_ERROR = "integration_error"
    
    # Agent Events
    AGENT_CREATED = "agent_created"
    AGENT_UPDATED = "agent_updated"
    AGENT_DELETED = "agent_deleted"
    AGENT_EXECUTED = "agent_executed"
    
    # Task Events
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_DELETED = "task_deleted"
    TASK_EXECUTED = "task_executed"
    
    # Workflow Events
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_DELETED = "workflow_deleted"
    WORKFLOW_EXECUTED = "workflow_executed"
    
    # Data Events
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_DELETION = "data_deletion"
    
    # Other
    CUSTOM = "custom"


class SeverityLevel(str, Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditResult(str, Enum):
    """Result of audited operation"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


# Audit Log Models
class AuditLogBase(BaseModel):
    """Audit log entry base"""
    user_id: Optional[str] = None
    event_type: EventType
    severity: SeverityLevel = SeverityLevel.INFO
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    result: AuditResult = AuditResult.SUCCESS
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_method: Optional[str] = None
    request_path: Optional[str] = None
    request_body: Optional[Dict[str, Any]] = None  # Sanitized
    response_status: Optional[int] = None
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AuditLogCreate(AuditLogBase):
    """Create audit log entry"""
    pass


class AuditLogEntry(AuditLogBase):
    """Audit log entry with ID and timestamp"""
    log_id: str
    timestamp: datetime
    created_at: datetime


# Query Models
class AuditLogQuery(BaseModel):
    """Query parameters for audit logs"""
    user_id: Optional[str] = None
    event_type: Optional[EventType] = None
    severity: Optional[SeverityLevel] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    result: Optional[AuditResult] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(100, le=1000)
    offset: int = 0


class AuditLogFilter(BaseModel):
    """Advanced filter for audit logs"""
    user_ids: Optional[List[str]] = None
    event_types: Optional[List[EventType]] = None
    severities: Optional[List[SeverityLevel]] = None
    resource_types: Optional[List[str]] = None
    results: Optional[List[AuditResult]] = None
    ip_addresses: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None
    search_text: Optional[str] = None


# Statistics Models
class EventTypeStats(BaseModel):
    """Statistics by event type"""
    event_type: EventType
    count: int
    success_count: int
    failure_count: int
    last_occurrence: Optional[datetime] = None


class SeverityStats(BaseModel):
    """Statistics by severity level"""
    severity: SeverityLevel
    count: int
    percentage: float


class UserActivityStats(BaseModel):
    """User activity statistics"""
    user_id: str
    total_events: int
    event_types: Dict[str, int]
    last_activity: Optional[datetime] = None
    most_common_action: Optional[str] = None


class AuditStatistics(BaseModel):
    """Overall audit statistics"""
    total_logs: int
    by_event_type: List[EventTypeStats]
    by_severity: List[SeverityStats]
    by_result: Dict[str, int]
    unique_users: int
    date_range: Dict[str, datetime]
    top_users: List[UserActivityStats]


# User Activity Models
class UserActivity(BaseModel):
    """User activity summary"""
    user_id: str
    logs: List[AuditLogEntry]
    total_events: int
    event_breakdown: Dict[str, int]
    first_activity: Optional[datetime] = None
    last_activity: Optional[datetime] = None


# Recent Events Model
class RecentEvents(BaseModel):
    """Recent audit events"""
    events: List[AuditLogEntry]
    count: int
    time_range: Dict[str, datetime]


# Sensitive Data Patterns
SENSITIVE_FIELDS = [
    "password",
    "token",
    "api_key",
    "secret",
    "private_key",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
    "session",
    "credential",
    "xoxp",  # Slack tokens
    "sk_",   # Stripe keys
    "whsec_" # Webhook secrets
]


def sanitize_data(data: Any) -> Any:
    """Sanitize sensitive data from logs"""
    if data is None:
        return None
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            # Check if key contains sensitive field name
            if any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_data(value)
        return sanitized
    
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    
    elif isinstance(data, str):
        # Check for token-like patterns
        if len(data) > 20 and any(sensitive in data.lower() for sensitive in ["token", "key", "secret"]):
            return "***REDACTED***"
        return data
    
    else:
        return data


# Audit Context Model
class AuditContext(BaseModel):
    """Context for audit logging"""
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None

