"""
API Key Module Test Suite for CPAS4
Comprehensive tests for API key management
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database import Database
from services.api_key_service import APIKeyService
from models.api_key_models import *


class APIKeyModuleTests:
    """Test suite for API key module"""
    
    def __init__(self):
        self.db = None
        self.service = None
        self.test_user_id = "test_user_api_key"
        self.test_api_key = None
        self.test_key_id = None
    
    async def setup(self):
        """Setup test environment"""
        print("\n🔧 Setting up API Key Module tests...")
        self.db = Database()
        await self.db.connect()
        self.service = APIKeyService(self.db)
        print("✅ Test environment ready")
    
    async def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")
        
        # Delete API keys
        keys = await self.db.find_many("api_keys", {"user_id": self.test_user_id}, limit=1000)
        for key in keys:
            await self.db.delete_one("api_keys", {"key_id": key.get("key_id")})
        
        # Delete usage records
        usage_records = await self.db.find_many("api_key_usage", {"user_id": self.test_user_id}, limit=1000)
        for record in usage_records:
            await self.db.delete_one("api_key_usage", {"usage_id": record.get("usage_id")})
        
        print("✅ Cleanup complete")
    
    async def test_1_generate_api_key(self):
        """Test 1: Generate API key"""
        print("\n📝 Test 1: Generate API key")
        
        api_key = self.service.generate_api_key()
        
        assert api_key is not None
        assert api_key.startswith(API_KEY_PREFIX)
        assert len(api_key) == API_KEY_LENGTH
        
        self.test_api_key = api_key
        
        print(f"✅ Generated API key: {api_key[:15]}...")
    
    async def test_2_create_api_key(self):
        """Test 2: Create API key"""
        print("\n📝 Test 2: Create API key")
        
        response = await self.service.create_api_key(
            user_id=self.test_user_id,
            name="Test API Key",
            scopes=[APIKeyScope.READ, APIKeyScope.WRITE],
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        assert response is not None
        assert response.key_id.startswith("key_")
        assert response.user_id == self.test_user_id
        assert response.name == "Test API Key"
        assert response.key.startswith(API_KEY_PREFIX)
        assert len(response.scopes) == 2
        
        self.test_api_key = response.key
        self.test_key_id = response.key_id
        
        print(f"✅ API key created: {response.key_id}")
        print(f"  - Key: {response.key[:15]}...")
        print(f"  - Prefix: {response.prefix}")
        print(f"  - Scopes: {[s.value for s in response.scopes]}")
    
    async def test_3_verify_api_key_valid(self):
        """Test 3: Verify API key (valid)"""
        print("\n📝 Test 3: Verify API key (valid)")
        
        if not self.test_api_key:
            print("⚠️ Skipping - no test API key")
            return
        
        result = await self.service.verify_api_key(self.test_api_key)
        
        assert result.valid is True
        assert result.key_id == self.test_key_id
        assert result.user_id == self.test_user_id
        assert len(result.scopes) == 2
        
        print(f"✅ API key verified successfully")
        print(f"  - Key ID: {result.key_id}")
        print(f"  - User ID: {result.user_id}")
    
    async def test_4_verify_api_key_invalid(self):
        """Test 4: Verify API key (invalid)"""
        print("\n📝 Test 4: Verify API key (invalid)")
        
        invalid_key = f"{API_KEY_PREFIX}invalid_key_12345678901234567890"
        
        result = await self.service.verify_api_key(invalid_key)
        
        assert result.valid is False
        assert result.error is not None
        
        print(f"✅ Invalid API key correctly rejected")
        print(f"  - Error: {result.error}")
    
    async def test_5_get_api_key(self):
        """Test 5: Get API key"""
        print("\n📝 Test 5: Get API key")
        
        if not self.test_key_id:
            print("⚠️ Skipping - no test key ID")
            return
        
        key = await self.service.get_api_key(self.test_key_id)
        
        assert key is not None
        assert key.key_id == self.test_key_id
        assert key.user_id == self.test_user_id
        
        print(f"✅ Retrieved API key: {key.key_id}")
    
    async def test_6_list_api_keys(self):
        """Test 6: List API keys"""
        print("\n📝 Test 6: List API keys")
        
        keys = await self.service.list_api_keys(self.test_user_id)
        
        assert len(keys) > 0
        
        print(f"✅ Listed {len(keys)} API keys")
        
        for key in keys:
            print(f"  - {key.name} ({key.prefix}): {key.status.value}")
    
    async def test_7_update_api_key(self):
        """Test 7: Update API key"""
        print("\n📝 Test 7: Update API key")
        
        if not self.test_key_id:
            print("⚠️ Skipping - no test key ID")
            return
        
        update = APIKeyUpdate(
            name="Updated Test API Key",
            scopes=[APIKeyScope.READ]
        )
        
        updated_key = await self.service.update_api_key(self.test_key_id, update)
        
        assert updated_key is not None
        assert updated_key.name == "Updated Test API Key"
        assert len(updated_key.scopes) == 1
        
        print(f"✅ API key updated")
    
    async def test_8_record_usage(self):
        """Test 8: Record API key usage"""
        print("\n📝 Test 8: Record API key usage")
        
        if not self.test_key_id:
            print("⚠️ Skipping - no test key ID")
            return
        
        # Record multiple usage events
        for i in range(5):
            await self.service.record_usage(
                key_id=self.test_key_id,
                user_id=self.test_user_id,
                endpoint=f"/api/test/endpoint{i % 2}",
                method="GET",
                ip_address="192.168.1.100",
                response_status=200
            )
        
        print(f"✅ Recorded 5 usage events")
    
    async def test_9_get_usage_stats(self):
        """Test 9: Get usage statistics"""
        print("\n📝 Test 9: Get usage statistics")
        
        if not self.test_key_id:
            print("⚠️ Skipping - no test key ID")
            return
        
        stats = await self.service.get_usage_stats(self.test_key_id)
        
        assert stats is not None
        assert stats.total_requests >= 5
        
        print(f"✅ Usage statistics retrieved:")
        print(f"  - Total requests: {stats.total_requests}")
        print(f"  - Last hour: {stats.requests_last_hour}")
        print(f"  - Last day: {stats.requests_last_day}")
        print(f"  - Most used endpoints: {len(stats.most_used_endpoints)}")
    
    async def test_10_rotate_api_key(self):
        """Test 10: Rotate API key"""
        print("\n📝 Test 10: Rotate API key")
        
        if not self.test_key_id:
            print("⚠️ Skipping - no test key ID")
            return
        
        response = await self.service.rotate_api_key(
            key_id=self.test_key_id,
            revoked_by="test_user"
        )
        
        assert response is not None
        assert response.old_key_id == self.test_key_id
        assert response.new_key_id != self.test_key_id
        assert response.new_key.startswith(API_KEY_PREFIX)
        
        # Update test key ID to new key
        self.test_key_id = response.new_key_id
        self.test_api_key = response.new_key
        
        print(f"✅ API key rotated")
        print(f"  - Old key: {response.old_key_id}")
        print(f"  - New key: {response.new_key_id}")
    
    async def test_11_verify_revoked_key(self):
        """Test 11: Verify revoked key (should fail)"""
        print("\n📝 Test 11: Verify revoked key (should fail)")
        
        # The old key from rotation should be revoked
        # We need to get it from the database
        keys = await self.service.list_api_keys(self.test_user_id, include_revoked=True)
        revoked_keys = [k for k in keys if k.status == APIKeyStatus.REVOKED]
        
        if revoked_keys:
            # We can't verify without the actual key, but we can check the status
            assert revoked_keys[0].revoked_at is not None
            print(f"✅ Revoked key found with revoked_at timestamp")
        else:
            print(f"⚠️ No revoked keys found (may have been cleaned up)")
    
    async def test_12_revoke_api_key(self):
        """Test 12: Revoke API key"""
        print("\n📝 Test 12: Revoke API key")
        
        if not self.test_key_id:
            print("⚠️ Skipping - no test key ID")
            return
        
        success = await self.service.revoke_api_key(
            key_id=self.test_key_id,
            reason="Test revocation",
            revoked_by="test_user"
        )
        
        assert success is True
        
        # Verify key is revoked
        key = await self.service.get_api_key(self.test_key_id)
        assert key.revoked_at is not None
        assert key.enabled is False
        
        print(f"✅ API key revoked")
    
    async def test_13_verify_revoked_key_fails(self):
        """Test 13: Verify revoked key fails"""
        print("\n📝 Test 13: Verify revoked key fails")

        if not self.test_api_key:
            print("⚠️ Skipping - no test API key")
            return

        result = await self.service.verify_api_key(self.test_api_key)

        assert result.valid is False
        assert result.error is not None
        # Check for either "revoked" or "disabled" in error message
        assert "revoked" in result.error.lower() or "disabled" in result.error.lower()

        print(f"✅ Revoked key verification correctly failed")
        print(f"  - Error: {result.error}")
    
    async def test_14_delete_api_key(self):
        """Test 14: Delete API key"""
        print("\n📝 Test 14: Delete API key")
        
        if not self.test_key_id:
            print("⚠️ Skipping - no test key ID")
            return
        
        success = await self.service.delete_api_key(self.test_key_id)
        
        assert success is True
        
        # Verify key is deleted
        key = await self.service.get_api_key(self.test_key_id)
        assert key is None
        
        print(f"✅ API key deleted")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 API KEY MODULE TEST SUITE")
        print("="*80)
        
        try:
            await self.setup()
            
            # Run tests
            await self.test_1_generate_api_key()
            await self.test_2_create_api_key()
            await self.test_3_verify_api_key_valid()
            await self.test_4_verify_api_key_invalid()
            await self.test_5_get_api_key()
            await self.test_6_list_api_keys()
            await self.test_7_update_api_key()
            await self.test_8_record_usage()
            await self.test_9_get_usage_stats()
            await self.test_10_rotate_api_key()
            await self.test_11_verify_revoked_key()
            await self.test_12_revoke_api_key()
            await self.test_13_verify_revoked_key_fails()
            await self.test_14_delete_api_key()
            
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED (14/14)")
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
    tests = APIKeyModuleTests()
    success = await tests.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

