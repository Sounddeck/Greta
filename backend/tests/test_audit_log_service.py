"""
Comprehensive tests for audit log service

Tests cover:
- Audit log creation with different severity levels
- Audit log querying with filters
- Data sanitization
- User activity tracking
- Error handling
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timedelta


class TestAuditLogCreation:
    """Test audit log creation"""
    
    @pytest.mark.asyncio
    async def test_create_audit_log_success(self):
        """Test creating an audit log entry"""
        from services.audit_service import AuditService
        from models.audit_models import EventType, SeverityLevel, AuditResult
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Create audit log
        log_entry = await audit_service.log_event(
            event_type=EventType.LOGIN,
            user_id="user_123",
            severity=SeverityLevel.INFO,
            result=AuditResult.SUCCESS,
            ip_address="192.168.1.1"
        )
        
        assert log_entry is not None
        assert log_entry.log_id.startswith("audit_")
        assert log_entry.event_type == EventType.LOGIN
        assert log_entry.user_id == "user_123"
        assert log_entry.severity == SeverityLevel.INFO
        assert log_entry.result == AuditResult.SUCCESS
    
    @pytest.mark.asyncio
    async def test_create_audit_log_with_severity_levels(self):
        """Test creating audit logs with different severity levels"""
        from services.audit_service import AuditService
        from models.audit_models import EventType, SeverityLevel, AuditResult
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Test each severity level
        for severity in [SeverityLevel.INFO, SeverityLevel.WARNING, SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
            log_entry = await audit_service.log_event(
                event_type=EventType.SYSTEM_ERROR,
                severity=severity,
                result=AuditResult.FAILURE
            )
            
            assert log_entry.severity == severity
    
    @pytest.mark.asyncio
    async def test_create_audit_log_with_metadata(self):
        """Test creating audit log with metadata"""
        from services.audit_service import AuditService
        from models.audit_models import EventType, SeverityLevel
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Create audit log with metadata
        metadata = {
            "action": "create_user",
            "details": "New user created",
            "count": 1
        }
        
        log_entry = await audit_service.log_event(
            event_type=EventType.CREATE,
            user_id="admin_123",
            severity=SeverityLevel.INFO,
            resource_type="user",
            resource_id="user_456",
            metadata=metadata
        )
        
        assert log_entry.metadata is not None
        assert log_entry.metadata["action"] == "create_user"
        assert log_entry.resource_type == "user"
        assert log_entry.resource_id == "user_456"
    
    @pytest.mark.asyncio
    async def test_create_audit_log_with_user_context(self):
        """Test creating audit log with user context"""
        from services.audit_service import AuditService
        from models.audit_models import EventType, SeverityLevel
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Create audit log with user context
        log_entry = await audit_service.log_event(
            event_type=EventType.UPDATE,
            user_id="user_123",
            severity=SeverityLevel.INFO,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
            request_method="PUT",
            request_path="/api/users/123"
        )
        
        assert log_entry.user_id == "user_123"
        assert log_entry.ip_address == "192.168.1.100"
        assert log_entry.user_agent == "Mozilla/5.0"
        assert log_entry.request_method == "PUT"
        assert log_entry.request_path == "/api/users/123"


class TestAuditLogQuerying:
    """Test audit log querying"""
    
    @pytest.mark.asyncio
    async def test_query_audit_logs_by_user_id(self):
        """Test querying audit logs by user ID"""
        from services.audit_service import AuditService
        from models.audit_models import EventType, SeverityLevel
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_many = AsyncMock(return_value=[
            {
                "log_id": "audit_123",
                "timestamp": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "user_id": "user_123",
                "event_type": "login",
                "severity": "info",
                "result": "success",
                "metadata": {}
            }
        ])
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Query logs by user ID
        logs = await audit_service.query_audit_logs(user_id="user_123")
        
        assert len(logs) == 1
        assert logs[0].user_id == "user_123"
    
    @pytest.mark.asyncio
    async def test_query_audit_logs_by_severity(self):
        """Test querying audit logs by severity level"""
        from services.audit_service import AuditService
        from models.audit_models import SeverityLevel
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_many = AsyncMock(return_value=[
            {
                "log_id": "audit_456",
                "timestamp": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "user_id": "user_123",
                "event_type": "system_error",
                "severity": "error",
                "result": "failure",
                "metadata": {}
            }
        ])
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Query logs by severity
        logs = await audit_service.query_audit_logs(severity=SeverityLevel.ERROR)
        
        assert len(logs) == 1
        assert logs[0].severity == SeverityLevel.ERROR
    
    @pytest.mark.asyncio
    async def test_query_audit_logs_by_date_range(self):
        """Test querying audit logs by date range"""
        from services.audit_service import AuditService
        
        # Create mock database
        now = datetime.utcnow()
        mock_db = AsyncMock()
        mock_db.find_many = AsyncMock(return_value=[
            {
                "log_id": "audit_789",
                "timestamp": now.isoformat(),
                "created_at": now.isoformat(),
                "user_id": "user_123",
                "event_type": "login",
                "severity": "info",
                "result": "success",
                "metadata": {}
            }
        ])
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Query logs by date range
        start_date = now - timedelta(days=1)
        end_date = now + timedelta(days=1)
        logs = await audit_service.query_audit_logs(
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(logs) == 1
    
    @pytest.mark.asyncio
    async def test_query_audit_logs_with_filters(self):
        """Test querying audit logs with multiple filters"""
        from services.audit_service import AuditService
        from models.audit_models import EventType, SeverityLevel, AuditResult
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_many = AsyncMock(return_value=[
            {
                "log_id": "audit_101",
                "timestamp": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "user_id": "user_123",
                "event_type": "login",
                "severity": "info",
                "resource_type": "auth",
                "result": "success",
                "metadata": {}
            }
        ])
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Query logs with multiple filters
        logs = await audit_service.query_audit_logs(
            user_id="user_123",
            event_type=EventType.LOGIN,
            severity=SeverityLevel.INFO,
            result=AuditResult.SUCCESS
        )
        
        assert len(logs) == 1
        assert logs[0].user_id == "user_123"
        assert logs[0].event_type == EventType.LOGIN


class TestDataSanitization:
    """Test data sanitization"""
    
    @pytest.mark.asyncio
    async def test_sanitize_sensitive_data(self):
        """Test sanitizing sensitive data in audit logs"""
        from services.audit_service import AuditService
        from models.audit_models import EventType
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Create audit log with sensitive data
        request_body = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "sk_test_123456"
        }
        
        log_entry = await audit_service.log_event(
            event_type=EventType.LOGIN,
            user_id="user_123",
            request_body=request_body
        )
        
        # Verify sensitive data is sanitized
        assert log_entry.request_body["username"] == "testuser"
        assert log_entry.request_body["password"] == "***REDACTED***"
        assert log_entry.request_body["api_key"] == "***REDACTED***"
    
    @pytest.mark.asyncio
    async def test_sanitize_pii(self):
        """Test sanitizing PII (personally identifiable information)"""
        from services.audit_service import AuditService
        from models.audit_models import EventType
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Create audit log with metadata containing tokens
        metadata = {
            "user_name": "John Doe",
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "refresh_token": "refresh_abc123"
        }
        
        log_entry = await audit_service.log_event(
            event_type=EventType.LOGIN,
            user_id="user_123",
            metadata=metadata
        )
        
        # Verify tokens are sanitized
        assert log_entry.metadata["user_name"] == "John Doe"
        assert log_entry.metadata["access_token"] == "***REDACTED***"
        assert log_entry.metadata["refresh_token"] == "***REDACTED***"
    
    @pytest.mark.asyncio
    async def test_sanitize_authentication_tokens(self):
        """Test sanitizing authentication tokens"""
        from services.audit_service import AuditService
        from models.audit_models import EventType
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Create audit log with authorization header
        request_body = {
            "authorization": "Bearer token123",
            "cookie": "session=abc123",
            "data": "public_data"
        }
        
        log_entry = await audit_service.log_event(
            event_type=EventType.READ,
            user_id="user_123",
            request_body=request_body
        )
        
        # Verify authorization data is sanitized
        assert log_entry.request_body["authorization"] == "***REDACTED***"
        assert log_entry.request_body["cookie"] == "***REDACTED***"
        assert log_entry.request_body["data"] == "public_data"


class TestUserActivityTracking:
    """Test user activity tracking"""
    
    @pytest.mark.asyncio
    async def test_get_user_activity(self):
        """Test getting user activity"""
        from services.audit_service import AuditService
        
        # Create mock database
        now = datetime.utcnow()
        mock_db = AsyncMock()
        mock_db.find_many = AsyncMock(return_value=[
            {
                "log_id": "audit_1",
                "timestamp": now.isoformat(),
                "created_at": now.isoformat(),
                "user_id": "user_123",
                "event_type": "login",
                "severity": "info",
                "result": "success",
                "metadata": {}
            },
            {
                "log_id": "audit_2",
                "timestamp": (now + timedelta(hours=1)).isoformat(),
                "created_at": (now + timedelta(hours=1)).isoformat(),
                "user_id": "user_123",
                "event_type": "create",
                "severity": "info",
                "result": "success",
                "metadata": {}
            }
        ])
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Get user activity
        activity = await audit_service.get_user_activity("user_123")
        
        assert activity.user_id == "user_123"
        assert activity.total_events == 2
        assert "login" in activity.event_breakdown
        assert "create" in activity.event_breakdown
        assert activity.first_activity is not None
        assert activity.last_activity is not None


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_audit_log(self):
        """Test getting nonexistent audit log"""
        from services.audit_service import AuditService
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_one = AsyncMock(return_value=None)
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Get nonexistent log
        log = await audit_service.get_audit_log("nonexistent")
        
        assert log is None
    
    @pytest.mark.asyncio
    async def test_query_with_no_results(self):
        """Test querying with no results"""
        from services.audit_service import AuditService
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_many = AsyncMock(return_value=[])
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Query logs
        logs = await audit_service.query_audit_logs(user_id="nonexistent")
        
        assert len(logs) == 0
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self):
        """Test database error handling"""
        from services.audit_service import AuditService
        from models.audit_models import EventType
        
        # Create mock database that raises error
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(side_effect=Exception("Database error"))
        
        # Create audit service
        audit_service = AuditService(mock_db)
        
        # Try to create log - should raise exception
        with pytest.raises(Exception):
            await audit_service.log_event(
                event_type=EventType.LOGIN,
                user_id="user_123"
            )
