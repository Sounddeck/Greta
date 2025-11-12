"""
Audit Module Test Suite for CPAS4
Comprehensive tests for Audit Logging
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database import Database
from services.audit_service import AuditService
from models.audit_models import *


class AuditModuleTests:
    """Test suite for Audit module"""
    
    def __init__(self):
        self.db = None
        self.service = None
        self.test_user_id = "test_user_audit_123"
        self.test_log_ids = []
    
    async def setup(self):
        """Setup test environment"""
        print("\n🔧 Setting up Audit Module tests...")
        self.db = Database()
        await self.db.connect()
        self.service = AuditService(self.db)
        print("✅ Test environment ready")
    
    async def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")
        
        # Delete test audit logs
        for log_id in self.test_log_ids:
            await self.db.delete_one("audit_logs", {"log_id": log_id})
        
        # Delete all test user logs
        test_logs = await self.db.find_many("audit_logs", {"user_id": self.test_user_id}, limit=1000)
        for log in test_logs:
            await self.db.delete_one("audit_logs", {"log_id": log.get("log_id")})
        
        print("✅ Cleanup complete")
    
    async def test_1_log_authentication_event(self):
        """Test 1: Log authentication event"""
        print("\n📝 Test 1: Log authentication event")
        
        log = await self.service.log_authentication(
            user_id=self.test_user_id,
            event_type=EventType.LOGIN,
            success=True,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0"
        )
        
        assert log is not None
        assert log.log_id.startswith("audit_")
        assert log.event_type == EventType.LOGIN
        assert log.user_id == self.test_user_id
        assert log.result == AuditResult.SUCCESS
        assert log.severity == SeverityLevel.INFO
        
        self.test_log_ids.append(log.log_id)
        print(f"✅ Logged authentication event: {log.log_id}")
    
    async def test_2_log_failed_login(self):
        """Test 2: Log failed login"""
        print("\n📝 Test 2: Log failed login")
        
        log = await self.service.log_authentication(
            user_id=self.test_user_id,
            event_type=EventType.LOGIN_FAILED,
            success=False,
            ip_address="192.168.1.100",
            error_message="Invalid credentials"
        )
        
        assert log is not None
        assert log.event_type == EventType.LOGIN_FAILED
        assert log.result == AuditResult.FAILURE
        assert log.severity == SeverityLevel.WARNING
        assert log.error_message == "Invalid credentials"
        
        self.test_log_ids.append(log.log_id)
        print(f"✅ Logged failed login: {log.log_id}")
    
    async def test_3_log_authorization_event(self):
        """Test 3: Log authorization event"""
        print("\n📝 Test 3: Log authorization event")
        
        log = await self.service.log_authorization(
            user_id=self.test_user_id,
            resource_type="agent",
            resource_id="agent_123",
            action="create",
            granted=True
        )
        
        assert log is not None
        assert log.event_type == EventType.ACCESS_GRANTED
        assert log.resource_type == "agent"
        assert log.resource_id == "agent_123"
        assert log.action == "create"
        assert log.result == AuditResult.SUCCESS
        
        self.test_log_ids.append(log.log_id)
        print(f"✅ Logged authorization event: {log.log_id}")
    
    async def test_4_log_access_denied(self):
        """Test 4: Log access denied"""
        print("\n📝 Test 4: Log access denied")
        
        log = await self.service.log_authorization(
            user_id=self.test_user_id,
            resource_type="system",
            resource_id="config",
            action="update",
            granted=False
        )
        
        assert log is not None
        assert log.event_type == EventType.ACCESS_DENIED
        assert log.result == AuditResult.FAILURE
        assert log.severity == SeverityLevel.WARNING
        
        self.test_log_ids.append(log.log_id)
        print(f"✅ Logged access denied: {log.log_id}")
    
    async def test_5_log_crud_operation(self):
        """Test 5: Log CRUD operation"""
        print("\n📝 Test 5: Log CRUD operation")
        
        log = await self.service.log_event(
            event_type=EventType.CREATE,
            user_id=self.test_user_id,
            resource_type="task",
            resource_id="task_456",
            action="POST",
            result=AuditResult.SUCCESS,
            request_method="POST",
            request_path="/api/v1/tasks",
            response_status=201,
            response_time_ms=45.67
        )
        
        assert log is not None
        assert log.event_type == EventType.CREATE
        assert log.resource_type == "task"
        assert log.response_status == 201
        assert log.response_time_ms == 45.67
        
        self.test_log_ids.append(log.log_id)
        print(f"✅ Logged CRUD operation: {log.log_id}")
    
    async def test_6_sanitize_sensitive_data(self):
        """Test 6: Sanitize sensitive data"""
        print("\n📝 Test 6: Sanitize sensitive data")
        
        # Log event with sensitive data
        log = await self.service.log_event(
            event_type=EventType.CREATE,
            user_id=self.test_user_id,
            request_body={
                "username": "testuser",
                "password": "super_secret_password",
                "api_key": "sk_test_1234567890",
                "token": "xoxp-secret-token"
            },
            metadata={
                "config": {
                    "database_password": "db_secret",
                    "normal_field": "visible_value"
                }
            }
        )
        
        assert log is not None
        assert log.request_body["username"] == "testuser"
        assert log.request_body["password"] == "***REDACTED***"
        assert log.request_body["api_key"] == "***REDACTED***"
        assert log.request_body["token"] == "***REDACTED***"
        assert log.metadata["config"]["database_password"] == "***REDACTED***"
        assert log.metadata["config"]["normal_field"] == "visible_value"
        
        self.test_log_ids.append(log.log_id)
        print(f"✅ Sensitive data sanitized correctly")
    
    async def test_7_get_audit_log(self):
        """Test 7: Get audit log by ID"""
        print("\n📝 Test 7: Get audit log by ID")
        
        if not self.test_log_ids:
            print("⚠️ Skipping - no test logs")
            return
        
        log = await self.service.get_audit_log(self.test_log_ids[0])
        
        assert log is not None
        assert log.log_id == self.test_log_ids[0]
        
        print(f"✅ Retrieved audit log: {log.log_id}")
    
    async def test_8_query_audit_logs(self):
        """Test 8: Query audit logs"""
        print("\n📝 Test 8: Query audit logs")
        
        logs = await self.service.query_audit_logs(
            user_id=self.test_user_id,
            limit=100
        )
        
        assert len(logs) > 0
        assert all(log.user_id == self.test_user_id for log in logs)
        
        print(f"✅ Queried {len(logs)} audit logs")
    
    async def test_9_query_by_event_type(self):
        """Test 9: Query by event type"""
        print("\n📝 Test 9: Query by event type")
        
        logs = await self.service.query_audit_logs(
            user_id=self.test_user_id,
            event_type=EventType.LOGIN,
            limit=100
        )
        
        assert all(log.event_type == EventType.LOGIN for log in logs)
        
        print(f"✅ Queried {len(logs)} LOGIN events")
    
    async def test_10_query_by_severity(self):
        """Test 10: Query by severity"""
        print("\n📝 Test 10: Query by severity")
        
        logs = await self.service.query_audit_logs(
            user_id=self.test_user_id,
            severity=SeverityLevel.WARNING,
            limit=100
        )
        
        assert all(log.severity == SeverityLevel.WARNING for log in logs)
        
        print(f"✅ Queried {len(logs)} WARNING events")
    
    async def test_11_get_user_activity(self):
        """Test 11: Get user activity"""
        print("\n📝 Test 11: Get user activity")
        
        activity = await self.service.get_user_activity(self.test_user_id, limit=100)
        
        assert activity is not None
        assert activity.user_id == self.test_user_id
        assert activity.total_events > 0
        assert len(activity.event_breakdown) > 0
        
        print(f"✅ Retrieved user activity: {activity.total_events} events")
    
    async def test_12_get_recent_events(self):
        """Test 12: Get recent events"""
        print("\n📝 Test 12: Get recent events")
        
        recent = await self.service.get_recent_events(limit=50)
        
        assert recent is not None
        assert recent.count > 0
        assert len(recent.events) > 0
        
        print(f"✅ Retrieved {recent.count} recent events")
    
    async def test_13_get_audit_statistics(self):
        """Test 13: Get audit statistics"""
        print("\n📝 Test 13: Get audit statistics")

        # Get all logs without date filter to verify statistics work
        all_logs = await self.service.query_audit_logs(limit=10000)

        print(f"  Total logs in database: {len(all_logs)}")

        # Use a very wide date range
        start_date = datetime.utcnow() - timedelta(days=3650)  # 10 years ago
        end_date = datetime.utcnow() + timedelta(days=365)     # 1 year in future

        stats = await self.service.get_audit_statistics(start_date, end_date)

        assert stats is not None
        # If we have logs in the database, stats should reflect them
        if len(all_logs) > 0:
            assert stats.total_logs > 0, f"Expected logs > 0, got {stats.total_logs}"
            assert len(stats.by_event_type) > 0
            assert len(stats.by_severity) > 0

        print(f"✅ Retrieved statistics: {stats.total_logs} total logs")
        print(f"  - Event types: {len(stats.by_event_type)}")
        print(f"  - Severity levels: {len(stats.by_severity)}")
        print(f"  - Unique users: {stats.unique_users}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 AUDIT MODULE TEST SUITE")
        print("="*80)
        
        try:
            await self.setup()
            
            # Run tests
            await self.test_1_log_authentication_event()
            await self.test_2_log_failed_login()
            await self.test_3_log_authorization_event()
            await self.test_4_log_access_denied()
            await self.test_5_log_crud_operation()
            await self.test_6_sanitize_sensitive_data()
            await self.test_7_get_audit_log()
            await self.test_8_query_audit_logs()
            await self.test_9_query_by_event_type()
            await self.test_10_query_by_severity()
            await self.test_11_get_user_activity()
            await self.test_12_get_recent_events()
            await self.test_13_get_audit_statistics()
            
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED (13/13)")
            print("="*80)
            
            return True
            
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()


async def main():
    """Main test runner"""
    tests = AuditModuleTests()
    success = await tests.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

