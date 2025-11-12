"""
Audit Logging Service for CPAS4
Service layer for capturing, storing, and retrieving audit logs
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from database import Database
from models.audit_models import *

logger = logging.getLogger(__name__)


class AuditService:
    """Audit service for logging and tracking system events"""
    
    def __init__(self, db: Database):
        self.db = db
        self.audit_logs_collection = "audit_logs"
    
    # Core Logging Methods
    async def log_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        result: AuditResult = AuditResult.SUCCESS,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_method: Optional[str] = None,
        request_path: Optional[str] = None,
        request_body: Optional[Dict[str, Any]] = None,
        response_status: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """Log an audit event"""
        log_id = f"audit_{uuid.uuid4().hex[:16]}"
        timestamp = datetime.utcnow()
        
        # Sanitize request body
        sanitized_body = sanitize_data(request_body) if request_body else None
        
        # Sanitize metadata
        sanitized_metadata = sanitize_data(metadata) if metadata else {}
        
        log_entry = AuditLogEntry(
            log_id=log_id,
            timestamp=timestamp,
            created_at=timestamp,
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            request_method=request_method,
            request_path=request_path,
            request_body=sanitized_body,
            response_status=response_status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=sanitized_metadata
        )
        
        await self.db.insert_one(self.audit_logs_collection, log_entry.model_dump())
        
        # Log to system logger based on severity
        log_msg = f"[{event_type.value}] User: {user_id or 'system'}, Resource: {resource_type}/{resource_id}, Result: {result.value}"
        if severity == SeverityLevel.CRITICAL:
            logger.critical(log_msg)
        elif severity == SeverityLevel.ERROR:
            logger.error(log_msg)
        elif severity == SeverityLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        return log_entry
    
    # Retrieval Methods
    async def get_audit_log(self, log_id: str) -> Optional[AuditLogEntry]:
        """Get single audit log entry"""
        doc = await self.db.find_one(self.audit_logs_collection, {"log_id": log_id})
        return AuditLogEntry(**doc) if doc else None
    
    async def query_audit_logs(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        severity: Optional[SeverityLevel] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        result: Optional[AuditResult] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLogEntry]:
        """Query audit logs with filters"""
        query = {}
        
        if user_id:
            query["user_id"] = user_id
        if event_type:
            query["event_type"] = event_type.value
        if severity:
            query["severity"] = severity.value
        if resource_type:
            query["resource_type"] = resource_type
        if resource_id:
            query["resource_id"] = resource_id
        if result:
            query["result"] = result.value
        
        # Get all docs first (date filtering done in Python due to ISO string comparison issues)
        docs = await self.db.find_many(
            self.audit_logs_collection,
            query,
            limit=limit * 10  # Get more to filter
        )

        # Convert to AuditLogEntry objects
        logs = [AuditLogEntry(**doc) for doc in docs]

        # Apply date range filter in Python
        if start_date or end_date:
            filtered_logs = []
            for log in logs:
                timestamp = log.timestamp
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)

                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue

                filtered_logs.append(log)

            logs = filtered_logs

        return logs[:limit]
    
    async def get_user_activity(
        self,
        user_id: str,
        limit: int = 100
    ) -> UserActivity:
        """Get all activity logs for a specific user"""
        logs = await self.query_audit_logs(user_id=user_id, limit=limit)
        
        # Calculate event breakdown
        event_breakdown = {}
        for log in logs:
            event_type = log.event_type.value
            event_breakdown[event_type] = event_breakdown.get(event_type, 0) + 1
        
        # Get first and last activity
        first_activity = min((log.timestamp for log in logs), default=None)
        last_activity = max((log.timestamp for log in logs), default=None)
        
        return UserActivity(
            user_id=user_id,
            logs=logs,
            total_events=len(logs),
            event_breakdown=event_breakdown,
            first_activity=first_activity,
            last_activity=last_activity
        )
    
    async def get_recent_events(self, limit: int = 100) -> RecentEvents:
        """Get recent audit events"""
        # Get all logs (limited)
        docs = await self.db.find_many(
            self.audit_logs_collection,
            {},
            limit=limit
        )
        
        events = [AuditLogEntry(**doc) for doc in docs]
        
        # Sort by timestamp descending
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Get time range
        time_range = {}
        if events:
            time_range["start"] = min(e.timestamp for e in events)
            time_range["end"] = max(e.timestamp for e in events)
        
        return RecentEvents(
            events=events,
            count=len(events),
            time_range=time_range
        )
    
    # Statistics Methods
    async def get_audit_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AuditStatistics:
        """Get audit statistics"""
        # Default to last 30 days if no date range specified
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get all logs in date range
        logs = await self.query_audit_logs(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        # Calculate statistics by event type
        event_type_stats = {}
        for log in logs:
            event_type = log.event_type
            if event_type not in event_type_stats:
                event_type_stats[event_type] = {
                    "count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "last_occurrence": None
                }
            
            stats = event_type_stats[event_type]
            stats["count"] += 1
            if log.result == AuditResult.SUCCESS:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
            
            if not stats["last_occurrence"] or log.timestamp > stats["last_occurrence"]:
                stats["last_occurrence"] = log.timestamp
        
        by_event_type = [
            EventTypeStats(
                event_type=event_type,
                count=stats["count"],
                success_count=stats["success_count"],
                failure_count=stats["failure_count"],
                last_occurrence=stats["last_occurrence"]
            )
            for event_type, stats in event_type_stats.items()
        ]
        
        # Calculate statistics by severity
        severity_counts = {}
        for log in logs:
            severity = log.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        total_logs = len(logs)
        by_severity = [
            SeverityStats(
                severity=severity,
                count=count,
                percentage=round((count / total_logs * 100), 2) if total_logs > 0 else 0
            )
            for severity, count in severity_counts.items()
        ]
        
        # Calculate statistics by result
        by_result = {}
        for log in logs:
            result = log.result.value
            by_result[result] = by_result.get(result, 0) + 1
        
        # Get unique users
        unique_users = len(set(log.user_id for log in logs if log.user_id))
        
        # Get top users
        user_event_counts = {}
        for log in logs:
            if log.user_id:
                user_event_counts[log.user_id] = user_event_counts.get(log.user_id, 0) + 1
        
        top_users = []
        for user_id, count in sorted(user_event_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            user_logs = [log for log in logs if log.user_id == user_id]
            event_types = {}
            for log in user_logs:
                event_types[log.event_type.value] = event_types.get(log.event_type.value, 0) + 1
            
            most_common = max(event_types.items(), key=lambda x: x[1])[0] if event_types else None
            last_activity = max((log.timestamp for log in user_logs), default=None)
            
            top_users.append(UserActivityStats(
                user_id=user_id,
                total_events=count,
                event_types=event_types,
                last_activity=last_activity,
                most_common_action=most_common
            ))
        
        return AuditStatistics(
            total_logs=total_logs,
            by_event_type=by_event_type,
            by_severity=by_severity,
            by_result=by_result,
            unique_users=unique_users,
            date_range={"start": start_date, "end": end_date},
            top_users=top_users
        )
    
    # Convenience Methods for Common Events
    async def log_authentication(
        self,
        user_id: str,
        event_type: EventType,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> AuditLogEntry:
        """Log authentication event"""
        return await self.log_event(
            event_type=event_type,
            user_id=user_id,
            severity=SeverityLevel.WARNING if not success else SeverityLevel.INFO,
            result=AuditResult.SUCCESS if success else AuditResult.FAILURE,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message
        )
    
    async def log_authorization(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        granted: bool
    ) -> AuditLogEntry:
        """Log authorization event"""
        return await self.log_event(
            event_type=EventType.ACCESS_GRANTED if granted else EventType.ACCESS_DENIED,
            user_id=user_id,
            severity=SeverityLevel.WARNING if not granted else SeverityLevel.INFO,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=AuditResult.SUCCESS if granted else AuditResult.FAILURE
        )

