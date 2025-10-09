"""
Rate Limiting Module Test Suite for CPAS4
Comprehensive tests for rate limiting
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database import Database
from services.rate_limit_service import RateLimitService
from models.rate_limit_models import *


class RateLimitModuleTests:
    """Test suite for rate limiting module"""
    
    def __init__(self):
        self.db = None
        self.service = None
        self.test_user_id = "test_user_rate_limit"
        self.test_ip = "192.168.1.100"
        self.test_rule_id = None
    
    async def setup(self):
        """Setup test environment"""
        print("\n🔧 Setting up Rate Limiting Module tests...")
        self.db = Database()
        await self.db.connect()
        self.service = RateLimitService(self.db)
        print("✅ Test environment ready")
    
    async def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")
        
        # Delete rules
        rules = await self.db.find_many("rate_limit_rules", {}, limit=1000)
        for rule in rules:
            if rule.get("created_by") == "test":
                await self.db.delete_one("rate_limit_rules", {"rule_id": rule.get("rule_id")})
        
        # Delete violations
        violations = await self.db.find_many("rate_limit_violations", {"user_id": self.test_user_id}, limit=1000)
        for violation in violations:
            await self.db.delete_one("rate_limit_violations", {"violation_id": violation.get("violation_id")})
        
        print("✅ Cleanup complete")
    
    async def test_1_create_rule(self):
        """Test 1: Create rate limit rule"""
        print("\n📝 Test 1: Create rate limit rule")
        
        rule = RateLimitRuleCreate(
            name="Test API Rate Limit",
            endpoint_pattern="/api/test/*",
            max_requests=10,
            window_seconds=60,
            scope=RateLimitScope.USER,
            action=RateLimitAction.BLOCK,
            enabled=True
        )
        
        created_rule = await self.service.create_rule(rule, created_by="test")
        
        assert created_rule is not None
        assert created_rule.rule_id.startswith("rl_")
        assert created_rule.name == "Test API Rate Limit"
        assert created_rule.max_requests == 10
        assert created_rule.window_seconds == 60
        
        self.test_rule_id = created_rule.rule_id
        
        print(f"✅ Rate limit rule created: {created_rule.rule_id}")
    
    async def test_2_get_rule(self):
        """Test 2: Get rate limit rule"""
        print("\n📝 Test 2: Get rate limit rule")
        
        if not self.test_rule_id:
            print("⚠️ Skipping - no test rule")
            return
        
        rule = await self.service.get_rule(self.test_rule_id)
        
        assert rule is not None
        assert rule.rule_id == self.test_rule_id
        assert rule.name == "Test API Rate Limit"
        
        print(f"✅ Retrieved rule: {rule.rule_id}")
    
    async def test_3_list_rules(self):
        """Test 3: List rate limit rules"""
        print("\n📝 Test 3: List rate limit rules")
        
        rules = await self.service.list_rules()
        
        assert len(rules) > 0
        
        print(f"✅ Listed {len(rules)} rules")
    
    async def test_4_update_rule(self):
        """Test 4: Update rate limit rule"""
        print("\n📝 Test 4: Update rate limit rule")
        
        if not self.test_rule_id:
            print("⚠️ Skipping - no test rule")
            return
        
        update = RateLimitRuleUpdate(
            max_requests=20,
            enabled=False
        )
        
        updated_rule = await self.service.update_rule(self.test_rule_id, update)
        
        assert updated_rule is not None
        assert updated_rule.max_requests == 20
        assert updated_rule.enabled is False
        
        print(f"✅ Rule updated: max_requests=20, enabled=False")
    
    async def test_5_check_rate_limit_allowed(self):
        """Test 5: Check rate limit (allowed)"""
        print("\n📝 Test 5: Check rate limit (allowed)")
        
        # Create a strict rule for testing
        rule = RateLimitRuleCreate(
            name="Test Strict Limit",
            endpoint_pattern="/api/test/endpoint",
            max_requests=5,
            window_seconds=60,
            scope=RateLimitScope.USER,
            action=RateLimitAction.BLOCK,
            enabled=True
        )
        
        created_rule = await self.service.create_rule(rule, created_by="test")
        
        # First request should be allowed
        is_allowed, violated_rule, status = await self.service.check_rate_limit(
            endpoint="/api/test/endpoint",
            user_id=self.test_user_id,
            ip_address=self.test_ip
        )
        
        assert is_allowed is True
        assert violated_rule is None
        
        print(f"✅ Request allowed (no rate limit exceeded)")
    
    async def test_6_record_request(self):
        """Test 6: Record request"""
        print("\n📝 Test 6: Record request")
        
        # Record multiple requests
        for i in range(3):
            await self.service.record_request(
                endpoint="/api/test/endpoint",
                user_id=self.test_user_id,
                ip_address=self.test_ip
            )
        
        print(f"✅ Recorded 3 requests")
    
    async def test_7_check_rate_limit_exceeded(self):
        """Test 7: Check rate limit (exceeded)"""
        print("\n📝 Test 7: Check rate limit (exceeded)")
        
        # Record requests up to the limit
        for i in range(5):
            await self.service.record_request(
                endpoint="/api/test/endpoint",
                user_id=self.test_user_id,
                ip_address=self.test_ip
            )
        
        # Next check should exceed limit
        is_allowed, violated_rule, status = await self.service.check_rate_limit(
            endpoint="/api/test/endpoint",
            user_id=self.test_user_id,
            ip_address=self.test_ip
        )
        
        assert is_allowed is False
        assert violated_rule is not None
        assert status is not None
        assert status.is_limited is True
        
        print(f"✅ Rate limit exceeded detected")
        print(f"  - Rule: {violated_rule.name}")
        print(f"  - Current requests: {status.current_requests}/{status.max_requests}")
    
    async def test_8_get_rate_limit_status(self):
        """Test 8: Get rate limit status"""
        print("\n📝 Test 8: Get rate limit status")
        
        statuses = await self.service.get_rate_limit_status(
            endpoint="/api/test/endpoint",
            user_id=self.test_user_id,
            ip_address=self.test_ip
        )
        
        assert len(statuses) > 0
        
        for status in statuses:
            print(f"  - {status.identifier}: {status.current_requests}/{status.max_requests} (remaining: {status.remaining_requests})")
        
        print(f"✅ Retrieved {len(statuses)} rate limit statuses")
    
    async def test_9_get_violations(self):
        """Test 9: Get violations"""
        print("\n📝 Test 9: Get violations")
        
        violations = await self.service.get_violations(user_id=self.test_user_id)
        
        assert len(violations) > 0
        
        print(f"✅ Retrieved {len(violations)} violations")
    
    async def test_10_get_stats(self):
        """Test 10: Get statistics"""
        print("\n📝 Test 10: Get statistics")
        
        stats = await self.service.get_stats()
        
        assert stats is not None
        assert stats.total_rules > 0
        
        print(f"✅ Statistics retrieved:")
        print(f"  - Total rules: {stats.total_rules}")
        print(f"  - Enabled rules: {stats.enabled_rules}")
        print(f"  - Total violations: {stats.total_violations}")
    
    async def test_11_initialize_default_rules(self):
        """Test 11: Initialize default rules"""
        print("\n📝 Test 11: Initialize default rules")
        
        created_rules = await self.service.initialize_default_rules()
        
        print(f"✅ Initialized {len(created_rules)} default rules")
        
        for rule in created_rules:
            print(f"  - {rule.name}: {rule.max_requests} req/{rule.window_seconds}s")
    
    async def test_12_delete_rule(self):
        """Test 12: Delete rate limit rule"""
        print("\n📝 Test 12: Delete rate limit rule")
        
        if not self.test_rule_id:
            print("⚠️ Skipping - no test rule")
            return
        
        success = await self.service.delete_rule(self.test_rule_id)
        
        assert success is True
        
        print(f"✅ Rule deleted: {self.test_rule_id}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 RATE LIMITING MODULE TEST SUITE")
        print("="*80)
        
        try:
            await self.setup()
            
            # Run tests
            await self.test_1_create_rule()
            await self.test_2_get_rule()
            await self.test_3_list_rules()
            await self.test_4_update_rule()
            await self.test_5_check_rate_limit_allowed()
            await self.test_6_record_request()
            await self.test_7_check_rate_limit_exceeded()
            await self.test_8_get_rate_limit_status()
            await self.test_9_get_violations()
            await self.test_10_get_stats()
            await self.test_11_initialize_default_rules()
            await self.test_12_delete_rule()
            
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED (12/12)")
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
    tests = RateLimitModuleTests()
    success = await tests.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

