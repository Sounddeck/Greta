"""
RBAC Module Test Suite for CPAS4
Comprehensive tests for Role-Based Access Control
"""

import asyncio
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database import Database
from services.rbac_service import RBACService
from models.rbac_models import *


class RBACModuleTests:
    """Test suite for RBAC module"""
    
    def __init__(self):
        self.db = None
        self.service = None
        self.test_role_id = None
        self.test_user_id = "test_user_123"
        self.test_admin_role_id = None
    
    async def setup(self):
        """Setup test environment"""
        print("\n🔧 Setting up RBAC Module tests...")
        self.db = Database()
        await self.db.connect()
        self.service = RBACService(self.db)
        print("✅ Test environment ready")
    
    async def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")

        # Delete test roles
        if self.test_role_id:
            await self.db.delete_one("roles", {"role_id": self.test_role_id})

        # Delete test user role assignments (find and delete one by one)
        user_roles = await self.db.find_many("user_roles", {"user_id": self.test_user_id}, limit=100)
        for ur in user_roles:
            await self.db.delete_one("user_roles", {"assignment_id": ur.get("assignment_id")})

        # Delete built-in roles created during tests (find and delete one by one)
        built_in_roles = await self.db.find_many("roles", {"created_by": "system"}, limit=100)
        for role in built_in_roles:
            await self.db.delete_one("roles", {"role_id": role.get("role_id")})

        print("✅ Cleanup complete")
    
    async def test_1_initialize_built_in_roles(self):
        """Test 1: Initialize built-in roles"""
        print("\n📝 Test 1: Initialize built-in roles")

        roles = await self.service.initialize_built_in_roles()

        print(f"  Roles created: {len(roles)}")
        for role in roles:
            print(f"  - {role.name} ({role.role_type.value}): {len(role.permissions)} permissions")

        assert len(roles) == 5, f"Expected 5 roles, got {len(roles)}"
        assert any(r.role_type == RoleType.ADMIN for r in roles), "No ADMIN role found"
        assert any(r.role_type == RoleType.USER for r in roles), "No USER role found"
        assert any(r.role_type == RoleType.VIEWER for r in roles), "No VIEWER role found"

        # Store admin role for later tests
        admin_role = next(r for r in roles if r.role_type == RoleType.ADMIN)
        self.test_admin_role_id = admin_role.role_id

        print(f"✅ Initialized {len(roles)} built-in roles")
    
    async def test_2_create_custom_role(self):
        """Test 2: Create custom role"""
        print("\n📝 Test 2: Create custom role")
        
        role = await self.service.create_role(
            RoleCreate(
                name="Test Custom Role",
                role_type=RoleType.CUSTOM,
                description="Custom role for testing",
                permissions=[
                    Permission(resource=ResourceType.AGENT, action=Action.READ),
                    Permission(resource=ResourceType.TASK, action=Action.CREATE)
                ],
                is_active=True
            ),
            created_by="test_system"
        )
        
        assert role is not None
        assert role.role_id.startswith("role_")
        assert role.name == "Test Custom Role"
        assert len(role.permissions) == 2
        
        self.test_role_id = role.role_id
        print(f"✅ Created custom role: {role.role_id}")
    
    async def test_3_get_role(self):
        """Test 3: Get role by ID"""
        print("\n📝 Test 3: Get role by ID")
        
        if not self.test_role_id:
            print("⚠️ Skipping - no test role")
            return
        
        role = await self.service.get_role(self.test_role_id)
        
        assert role is not None
        assert role.role_id == self.test_role_id
        assert role.name == "Test Custom Role"
        
        print(f"✅ Retrieved role: {role.role_id}")
    
    async def test_4_update_role(self):
        """Test 4: Update role"""
        print("\n📝 Test 4: Update role")
        
        if not self.test_role_id:
            print("⚠️ Skipping - no test role")
            return
        
        updated_role = await self.service.update_role(
            self.test_role_id,
            RoleUpdate(
                description="Updated description",
                permissions=[
                    Permission(resource=ResourceType.AGENT, action=Action.MANAGE)
                ]
            )
        )
        
        assert updated_role is not None
        assert updated_role.description == "Updated description"
        assert len(updated_role.permissions) == 1
        
        print(f"✅ Updated role: {self.test_role_id}")
    
    async def test_5_list_roles(self):
        """Test 5: List roles"""
        print("\n📝 Test 5: List roles")
        
        roles = await self.service.list_roles(active_only=True, limit=100)
        
        assert len(roles) > 0
        assert all(r.is_active for r in roles)
        
        print(f"✅ Listed {len(roles)} active roles")
    
    async def test_6_assign_role_to_user(self):
        """Test 6: Assign role to user"""
        print("\n📝 Test 6: Assign role to user")
        
        if not self.test_admin_role_id:
            print("⚠️ Skipping - no admin role")
            return
        
        assignment = await self.service.assign_role_to_user(
            user_id=self.test_user_id,
            role_id=self.test_admin_role_id,
            assigned_by="test_admin"
        )
        
        assert assignment is not None
        assert assignment.user_id == self.test_user_id
        assert assignment.role_id == self.test_admin_role_id
        assert assignment.is_active is True
        
        print(f"✅ Assigned role to user: {self.test_user_id}")
    
    async def test_7_get_user_roles(self):
        """Test 7: Get user roles"""
        print("\n📝 Test 7: Get user roles")
        
        roles = await self.service.get_user_roles(self.test_user_id)
        
        assert len(roles) > 0
        assert any(r.role_type == RoleType.ADMIN for r in roles)
        
        print(f"✅ Retrieved {len(roles)} roles for user")
    
    async def test_8_get_user_permissions(self):
        """Test 8: Get user permissions"""
        print("\n📝 Test 8: Get user permissions")
        
        perms = await self.service.get_user_permissions(self.test_user_id)
        
        assert perms is not None
        assert perms.user_id == self.test_user_id
        assert len(perms.roles) > 0
        assert len(perms.permissions) > 0
        
        print(f"✅ Retrieved {perms.permission_count} permissions for user")
    
    async def test_9_check_permission_allowed(self):
        """Test 9: Check permission (allowed)"""
        print("\n📝 Test 9: Check permission (allowed)")
        
        result = await self.service.check_permission(
            user_id=self.test_user_id,
            resource=ResourceType.AGENT,
            action=Action.CREATE
        )
        
        assert result is not None
        assert result.allowed is True
        assert len(result.matched_roles) > 0
        
        print(f"✅ Permission check passed: {result.allowed}")
    
    async def test_10_check_permission_denied(self):
        """Test 10: Check permission (denied)"""
        print("\n📝 Test 10: Check permission (denied)")
        
        # Create a user with no roles
        test_user_no_roles = "test_user_no_roles"
        
        result = await self.service.check_permission(
            user_id=test_user_no_roles,
            resource=ResourceType.AGENT,
            action=Action.CREATE
        )
        
        assert result is not None
        assert result.allowed is False
        assert result.reason is not None
        
        print(f"✅ Permission check denied: {result.reason}")
    
    async def test_11_revoke_role_from_user(self):
        """Test 11: Revoke role from user"""
        print("\n📝 Test 11: Revoke role from user")
        
        if not self.test_admin_role_id:
            print("⚠️ Skipping - no admin role")
            return
        
        success = await self.service.revoke_role_from_user(
            user_id=self.test_user_id,
            role_id=self.test_admin_role_id
        )
        
        assert success is True
        
        # Verify role was revoked
        roles = await self.service.get_user_roles(self.test_user_id)
        assert not any(r.role_id == self.test_admin_role_id for r in roles)
        
        print(f"✅ Revoked role from user: {self.test_user_id}")
    
    async def test_12_get_role_stats(self):
        """Test 12: Get role statistics"""
        print("\n📝 Test 12: Get role statistics")
        
        if not self.test_role_id:
            print("⚠️ Skipping - no test role")
            return
        
        stats = await self.service.get_role_stats(self.test_role_id)
        
        assert stats is not None
        assert stats.role_id == self.test_role_id
        assert stats.role_name == "Test Custom Role"
        
        print(f"✅ Retrieved role stats: {stats.user_count} users")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 RBAC MODULE TEST SUITE")
        print("="*80)
        
        try:
            await self.setup()
            
            # Run tests
            await self.test_1_initialize_built_in_roles()
            await self.test_2_create_custom_role()
            await self.test_3_get_role()
            await self.test_4_update_role()
            await self.test_5_list_roles()
            await self.test_6_assign_role_to_user()
            await self.test_7_get_user_roles()
            await self.test_8_get_user_permissions()
            await self.test_9_check_permission_allowed()
            await self.test_10_check_permission_denied()
            await self.test_11_revoke_role_from_user()
            await self.test_12_get_role_stats()
            
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED (12/12)")
            print("="*80)
            
            return True
            
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
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
    tests = RBACModuleTests()
    success = await tests.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

