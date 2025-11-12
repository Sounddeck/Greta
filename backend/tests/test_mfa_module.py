"""
MFA Module Test Suite for CPAS4
Comprehensive tests for Multi-Factor Authentication
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database import Database
from services.mfa_service import MFAService
from models.mfa_models import *


class MFAModuleTests:
    """Test suite for MFA module"""
    
    def __init__(self):
        self.db = None
        self.service = None
        self.test_user_id = "test_user_mfa_123"
        self.test_secret = None
        self.test_enrollment_id = None
        self.test_backup_codes = []
    
    async def setup(self):
        """Setup test environment"""
        print("\n🔧 Setting up MFA Module tests...")
        self.db = Database()
        await self.db.connect()
        self.service = MFAService(self.db)
        print("✅ Test environment ready")
    
    async def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")
        
        # Delete TOTP configurations
        totp_configs = await self.db.find_many("totp_configurations", {"user_id": self.test_user_id}, limit=100)
        for config in totp_configs:
            await self.db.delete_one("totp_configurations", {"config_id": config.get("config_id")})
        
        # Delete backup codes
        backup_codes = await self.db.find_many("backup_codes", {"user_id": self.test_user_id}, limit=100)
        for code in backup_codes:
            await self.db.delete_one("backup_codes", {"code_id": code.get("code_id")})
        
        # Delete MFA attempts
        attempts = await self.db.find_many("mfa_verification_attempts", {"user_id": self.test_user_id}, limit=100)
        for attempt in attempts:
            await self.db.delete_one("mfa_verification_attempts", {"attempt_id": attempt.get("attempt_id")})
        
        # Delete enrollments
        enrollments = await self.db.find_many("mfa_enrollments", {"user_id": self.test_user_id}, limit=100)
        for enrollment in enrollments:
            await self.db.delete_one("mfa_enrollments", {"enrollment_id": enrollment.get("enrollment_id")})
        
        print("✅ Cleanup complete")
    
    async def test_1_generate_totp_secret(self):
        """Test 1: Generate TOTP secret"""
        print("\n📝 Test 1: Generate TOTP secret")
        
        secret = self.service.generate_totp_secret()
        
        assert secret is not None
        assert len(secret) == 32  # Base32 encoded secret
        assert secret.isalnum()
        
        self.test_secret = secret
        print(f"✅ Generated TOTP secret: {secret[:8]}...")
    
    async def test_2_generate_qr_code_uri(self):
        """Test 2: Generate QR code URI"""
        print("\n📝 Test 2: Generate QR code URI")
        
        if not self.test_secret:
            print("⚠️ Skipping - no test secret")
            return
        
        uri = self.service.generate_qr_code_uri(self.test_user_id, self.test_secret)
        
        assert uri is not None
        assert uri.startswith("otpauth://totp/")
        assert self.test_user_id in uri
        assert TOTP_ISSUER in uri
        
        print(f"✅ Generated QR code URI: {uri[:50]}...")
    
    async def test_3_generate_qr_code_image(self):
        """Test 3: Generate QR code image"""
        print("\n📝 Test 3: Generate QR code image")
        
        if not self.test_secret:
            print("⚠️ Skipping - no test secret")
            return
        
        uri = self.service.generate_qr_code_uri(self.test_user_id, self.test_secret)
        image = self.service.generate_qr_code_image(uri)
        
        assert image is not None
        assert image.startswith("data:image/png;base64,")
        
        print(f"✅ Generated QR code image (base64): {len(image)} characters")
    
    async def test_4_verify_totp_valid(self):
        """Test 4: Verify TOTP with valid code"""
        print("\n📝 Test 4: Verify TOTP with valid code")
        
        if not self.test_secret:
            print("⚠️ Skipping - no test secret")
            return
        
        # Generate current TOTP code
        import pyotp
        totp = pyotp.TOTP(self.test_secret)
        current_code = totp.now()
        
        # Verify code
        is_valid = self.service.verify_totp(self.test_secret, current_code)
        
        assert is_valid is True
        
        print(f"✅ TOTP verification successful with code: {current_code}")
    
    async def test_5_verify_totp_invalid(self):
        """Test 5: Verify TOTP with invalid code"""
        print("\n📝 Test 5: Verify TOTP with invalid code")
        
        if not self.test_secret:
            print("⚠️ Skipping - no test secret")
            return
        
        # Use invalid code
        invalid_code = "000000"
        
        # Verify code
        is_valid = self.service.verify_totp(self.test_secret, invalid_code)
        
        assert is_valid is False
        
        print(f"✅ TOTP verification correctly rejected invalid code")
    
    async def test_6_enroll_totp(self):
        """Test 6: Enroll TOTP"""
        print("\n📝 Test 6: Enroll TOTP")
        
        response = await self.service.enroll_totp(self.test_user_id)
        
        assert response is not None
        assert response.enrollment_id.startswith("enroll_")
        assert response.user_id == self.test_user_id
        assert response.secret_key is not None
        assert response.qr_code_uri is not None
        assert response.qr_code_base64 is not None
        assert response.backup_codes is not None
        assert len(response.backup_codes) == BACKUP_CODE_COUNT
        
        self.test_enrollment_id = response.enrollment_id
        self.test_secret = response.secret_key
        self.test_backup_codes = response.backup_codes
        
        print(f"✅ TOTP enrollment started: {response.enrollment_id}")
        print(f"  - Secret: {response.secret_key[:8]}...")
        print(f"  - Backup codes: {len(response.backup_codes)}")
    
    async def test_7_verify_and_enable_totp(self):
        """Test 7: Verify and enable TOTP"""
        print("\n📝 Test 7: Verify and enable TOTP")
        
        if not self.test_secret:
            print("⚠️ Skipping - no test secret")
            return
        
        # Generate current TOTP code
        import pyotp
        totp = pyotp.TOTP(self.test_secret)
        current_code = totp.now()
        
        # Verify and enable
        config = await self.service.verify_and_enable_totp(
            user_id=self.test_user_id,
            code=current_code,
            ip_address="192.168.1.100"
        )
        
        assert config is not None
        assert config.config_id.startswith("totp_")
        assert config.user_id == self.test_user_id
        assert config.enabled is True
        assert config.verified is True
        
        print(f"✅ TOTP enabled: {config.config_id}")
    
    async def test_8_verify_totp_login(self):
        """Test 8: Verify TOTP during login"""
        print("\n📝 Test 8: Verify TOTP during login")
        
        if not self.test_secret:
            print("⚠️ Skipping - no test secret")
            return
        
        # Generate current TOTP code
        import pyotp
        totp = pyotp.TOTP(self.test_secret)
        current_code = totp.now()
        
        # Verify login
        is_valid = await self.service.verify_totp_login(
            user_id=self.test_user_id,
            code=current_code,
            ip_address="192.168.1.100"
        )
        
        assert is_valid is True
        
        print(f"✅ TOTP login verification successful")
    
    async def test_9_generate_backup_codes(self):
        """Test 9: Generate backup codes"""
        print("\n📝 Test 9: Generate backup codes")
        
        response = await self.service.generate_backup_codes(self.test_user_id)
        
        assert response is not None
        assert response.user_id == self.test_user_id
        assert len(response.codes) == BACKUP_CODE_COUNT
        assert response.count == BACKUP_CODE_COUNT
        
        # Verify code format (XXXX-XXXX-XXXX)
        for code in response.codes:
            assert len(code) == 14  # 12 chars + 2 dashes
            assert code[4] == "-"
            assert code[9] == "-"
        
        self.test_backup_codes = response.codes
        
        print(f"✅ Generated {len(response.codes)} backup codes")
    
    async def test_10_verify_backup_code(self):
        """Test 10: Verify backup code"""
        print("\n📝 Test 10: Verify backup code")
        
        if not self.test_backup_codes:
            print("⚠️ Skipping - no backup codes")
            return
        
        # Use first backup code
        code = self.test_backup_codes[0]
        
        # Verify code
        is_valid = await self.service.verify_backup_code(
            user_id=self.test_user_id,
            code=code,
            ip_address="192.168.1.100"
        )
        
        assert is_valid is True
        
        # Try to use same code again (should fail)
        is_valid_again = await self.service.verify_backup_code(
            user_id=self.test_user_id,
            code=code,
            ip_address="192.168.1.100"
        )
        
        assert is_valid_again is False
        
        print(f"✅ Backup code verified and consumed")
    
    async def test_11_get_backup_codes(self):
        """Test 11: Get backup codes (masked)"""
        print("\n📝 Test 11: Get backup codes (masked)")
        
        codes = await self.service.get_backup_codes(self.test_user_id)
        
        assert len(codes) > 0
        
        # Verify codes are masked
        for code in codes:
            assert code.code.startswith("****-****-")
        
        print(f"✅ Retrieved {len(codes)} backup codes (masked)")
    
    async def test_12_get_mfa_status(self):
        """Test 12: Get MFA status"""
        print("\n📝 Test 12: Get MFA status")
        
        status = await self.service.get_mfa_status(self.test_user_id)
        
        assert status is not None
        assert status.user_id == self.test_user_id
        assert status.totp_enabled is True
        assert status.totp_verified is True
        assert status.backup_codes_count > 0
        assert MFAMethod.TOTP in status.mfa_methods
        
        print(f"✅ MFA status retrieved:")
        print(f"  - TOTP enabled: {status.totp_enabled}")
        print(f"  - Backup codes: {status.backup_codes_remaining}/{status.backup_codes_count}")
    
    async def test_13_check_mfa_required(self):
        """Test 13: Check if MFA is required"""
        print("\n📝 Test 13: Check if MFA is required")
        
        is_required = await self.service.check_mfa_required(self.test_user_id)
        
        assert is_required is True
        
        print(f"✅ MFA required: {is_required}")
    
    async def test_14_disable_totp(self):
        """Test 14: Disable TOTP"""
        print("\n📝 Test 14: Disable TOTP")
        
        success = await self.service.disable_totp(self.test_user_id)
        
        assert success is True
        
        # Verify TOTP is disabled
        status = await self.service.get_mfa_status(self.test_user_id)
        assert status.totp_enabled is False
        
        print(f"✅ TOTP disabled successfully")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 MFA MODULE TEST SUITE")
        print("="*80)
        
        try:
            await self.setup()
            
            # Run tests
            await self.test_1_generate_totp_secret()
            await self.test_2_generate_qr_code_uri()
            await self.test_3_generate_qr_code_image()
            await self.test_4_verify_totp_valid()
            await self.test_5_verify_totp_invalid()
            await self.test_6_enroll_totp()
            await self.test_7_verify_and_enable_totp()
            await self.test_8_verify_totp_login()
            await self.test_9_generate_backup_codes()
            await self.test_10_verify_backup_code()
            await self.test_11_get_backup_codes()
            await self.test_12_get_mfa_status()
            await self.test_13_check_mfa_required()
            await self.test_14_disable_totp()
            
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
    tests = MFAModuleTests()
    success = await tests.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

