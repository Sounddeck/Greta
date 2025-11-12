"""
MFA (Multi-Factor Authentication) Service for CPAS4
Service layer for TOTP, backup codes, and MFA management
"""

import logging
import uuid
import secrets
import hashlib
import base64
import io
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import pyotp
import qrcode
from database import Database
from models.mfa_models import *

logger = logging.getLogger(__name__)


class MFAService:
    """MFA service for two-factor authentication"""
    
    def __init__(self, db: Database):
        self.db = db
        self.totp_configs_collection = "totp_configurations"
        self.backup_codes_collection = "backup_codes"
        self.mfa_attempts_collection = "mfa_verification_attempts"
        self.mfa_enrollments_collection = "mfa_enrollments"
    
    # TOTP Methods
    def generate_totp_secret(self) -> str:
        """Generate new TOTP secret"""
        return pyotp.random_base32()
    
    def generate_qr_code_uri(self, user_id: str, secret: str, issuer: str = TOTP_ISSUER) -> str:
        """Generate QR code URI for authenticator apps"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
    
    def generate_qr_code_image(self, uri: str) -> str:
        """Generate QR code image as base64 string"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code"""
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)  # Allow 1 step before/after for clock skew
    
    async def enroll_totp(self, user_id: str) -> TOTPEnrollmentResponse:
        """Start TOTP enrollment for user"""
        # Check if user already has TOTP enabled
        existing = await self.db.find_one(
            self.totp_configs_collection,
            {"user_id": user_id, "enabled": True}
        )
        if existing:
            raise ValueError("TOTP already enabled for this user")
        
        # Generate secret
        secret = self.generate_totp_secret()
        
        # Generate QR code URI
        qr_uri = self.generate_qr_code_uri(user_id, secret)
        
        # Generate QR code image
        qr_image = self.generate_qr_code_image(qr_uri)
        
        # Create enrollment record
        enrollment_id = f"enroll_{uuid.uuid4().hex[:12]}"
        enrollment = MFAEnrollment(
            enrollment_id=enrollment_id,
            user_id=user_id,
            method=MFAMethod.TOTP,
            status=MFAStatus.PENDING,
            created_at=datetime.utcnow(),
            qr_code_uri=qr_uri,
            secret_key=secret  # Temporary storage
        )
        
        await self.db.insert_one(self.mfa_enrollments_collection, enrollment.model_dump())
        
        # Generate backup codes
        backup_codes = await self._generate_backup_codes_internal(user_id)
        
        logger.info(f"TOTP enrollment started for user: {user_id}")
        
        return TOTPEnrollmentResponse(
            enrollment_id=enrollment_id,
            user_id=user_id,
            secret_key=secret,
            qr_code_uri=qr_uri,
            qr_code_base64=qr_image,
            backup_codes=backup_codes,
            instructions="Scan the QR code with your authenticator app and enter the 6-digit code to verify."
        )
    
    async def verify_and_enable_totp(
        self,
        user_id: str,
        code: str,
        ip_address: Optional[str] = None
    ) -> TOTPConfiguration:
        """Verify TOTP code and enable TOTP for user"""
        # Check rate limiting
        if await self._is_rate_limited(user_id):
            raise ValueError("Too many verification attempts. Please try again later.")
        
        # Get enrollment
        enrollment = await self.db.find_one(
            self.mfa_enrollments_collection,
            {"user_id": user_id, "method": MFAMethod.TOTP.value, "status": MFAStatus.PENDING.value}
        )
        
        if not enrollment:
            raise ValueError("No pending TOTP enrollment found")
        
        secret = enrollment.get("secret_key")
        
        # Verify code
        is_valid = self.verify_totp(secret, code)
        
        # Log attempt
        await self._log_verification_attempt(
            user_id=user_id,
            method=MFAMethod.TOTP,
            success=is_valid,
            ip_address=ip_address
        )
        
        if not is_valid:
            raise ValueError("Invalid TOTP code")
        
        # Create TOTP configuration
        config_id = f"totp_{uuid.uuid4().hex[:12]}"
        config = TOTPConfiguration(
            config_id=config_id,
            user_id=user_id,
            secret_key=secret,  # TODO: Encrypt in production
            enabled=True,
            verified=True,
            created_at=datetime.utcnow(),
            last_verified_at=datetime.utcnow(),
            verification_attempts=0
        )
        
        await self.db.insert_one(self.totp_configs_collection, config.model_dump())
        
        # Update enrollment status
        await self.db.update_one(
            self.mfa_enrollments_collection,
            {"enrollment_id": enrollment["enrollment_id"]},
            {"$set": {"status": MFAStatus.VERIFIED.value, "completed_at": datetime.utcnow().isoformat()}}
        )
        
        logger.info(f"TOTP enabled for user: {user_id}")
        
        return config
    
    async def verify_totp_login(
        self,
        user_id: str,
        code: str,
        ip_address: Optional[str] = None
    ) -> bool:
        """Verify TOTP code during login"""
        # Check rate limiting
        if await self._is_rate_limited(user_id):
            raise ValueError("Too many verification attempts. Please try again later.")
        
        # Get TOTP configuration
        config = await self.db.find_one(
            self.totp_configs_collection,
            {"user_id": user_id, "enabled": True}
        )
        
        if not config:
            raise ValueError("TOTP not enabled for this user")
        
        secret = config.get("secret_key")
        
        # Verify code
        is_valid = self.verify_totp(secret, code)
        
        # Log attempt
        await self._log_verification_attempt(
            user_id=user_id,
            method=MFAMethod.TOTP,
            success=is_valid,
            ip_address=ip_address
        )
        
        if is_valid:
            # Update last verified timestamp
            await self.db.update_one(
                self.totp_configs_collection,
                {"config_id": config["config_id"]},
                {"$set": {
                    "last_verified_at": datetime.utcnow().isoformat(),
                    "verification_attempts": 0
                }}
            )
        
        return is_valid
    
    async def disable_totp(self, user_id: str) -> bool:
        """Disable TOTP for user"""
        result = await self.db.update_one(
            self.totp_configs_collection,
            {"user_id": user_id},
            {"$set": {"enabled": False}}
        )
        
        if result:
            logger.info(f"TOTP disabled for user: {user_id}")
        
        return result
    
    # Backup Codes Methods
    def _generate_backup_code(self) -> str:
        """Generate a single backup code"""
        # Generate random code (12 characters)
        code = secrets.token_hex(6).upper()
        # Format as XXXX-XXXX-XXXX
        return f"{code[:4]}-{code[4:8]}-{code[8:12]}"
    
    def _hash_backup_code(self, code: str) -> str:
        """Hash backup code for storage"""
        return hashlib.sha256(code.encode()).hexdigest()
    
    async def _generate_backup_codes_internal(self, user_id: str) -> List[str]:
        """Generate backup codes (internal method)"""
        codes = []
        
        for _ in range(BACKUP_CODE_COUNT):
            code = self._generate_backup_code()
            code_hash = self._hash_backup_code(code)
            
            code_id = f"bc_{uuid.uuid4().hex[:12]}"
            backup_code = BackupCode(
                code_id=code_id,
                user_id=user_id,
                code_hash=code_hash,
                used=False,
                created_at=datetime.utcnow()
            )
            
            await self.db.insert_one(self.backup_codes_collection, backup_code.model_dump())
            codes.append(code)
        
        return codes
    
    async def generate_backup_codes(self, user_id: str) -> BackupCodesResponse:
        """Generate new backup codes for user"""
        # Delete old unused codes
        old_codes = await self.db.find_many(
            self.backup_codes_collection,
            {"user_id": user_id, "used": False},
            limit=100
        )
        for old_code in old_codes:
            await self.db.delete_one(self.backup_codes_collection, {"code_id": old_code["code_id"]})
        
        # Generate new codes
        codes = await self._generate_backup_codes_internal(user_id)
        
        logger.info(f"Generated {len(codes)} backup codes for user: {user_id}")
        
        return BackupCodesResponse(
            user_id=user_id,
            codes=codes,
            count=len(codes),
            created_at=datetime.utcnow()
        )
    
    async def verify_backup_code(
        self,
        user_id: str,
        code: str,
        ip_address: Optional[str] = None
    ) -> bool:
        """Verify and consume backup code"""
        code_hash = self._hash_backup_code(code)
        
        # Find matching unused code
        backup_code = await self.db.find_one(
            self.backup_codes_collection,
            {"user_id": user_id, "code_hash": code_hash, "used": False}
        )
        
        is_valid = backup_code is not None
        
        # Log attempt
        await self._log_verification_attempt(
            user_id=user_id,
            method=MFAMethod.BACKUP_CODES,
            success=is_valid,
            ip_address=ip_address
        )
        
        if is_valid:
            # Mark code as used
            await self.db.update_one(
                self.backup_codes_collection,
                {"code_id": backup_code["code_id"]},
                {"$set": {"used": True, "used_at": datetime.utcnow().isoformat()}}
            )
            logger.info(f"Backup code used for user: {user_id}")
        
        return is_valid

    async def get_backup_codes(self, user_id: str) -> List[BackupCodeDisplay]:
        """Get list of backup codes (masked)"""
        codes = await self.db.find_many(
            self.backup_codes_collection,
            {"user_id": user_id},
            limit=100
        )

        display_codes = []
        for code in codes:
            # Mask the code (show only last 4 characters)
            masked_code = f"****-****-{code['code_hash'][-4:]}"

            display_codes.append(BackupCodeDisplay(
                code_id=code["code_id"],
                code=masked_code,
                used=code["used"],
                created_at=datetime.fromisoformat(code["created_at"]) if isinstance(code["created_at"], str) else code["created_at"]
            ))

        return display_codes

    # MFA Status Methods
    async def get_mfa_status(self, user_id: str) -> MFAStatusResponse:
        """Get MFA status for user"""
        # Get TOTP configuration
        totp_config = await self.db.find_one(
            self.totp_configs_collection,
            {"user_id": user_id}
        )

        totp_enabled = totp_config is not None and totp_config.get("enabled", False)
        totp_verified = totp_config is not None and totp_config.get("verified", False)
        last_verified = None
        if totp_config and totp_config.get("last_verified_at"):
            last_verified_str = totp_config["last_verified_at"]
            last_verified = datetime.fromisoformat(last_verified_str) if isinstance(last_verified_str, str) else last_verified_str

        # Get backup codes
        backup_codes = await self.db.find_many(
            self.backup_codes_collection,
            {"user_id": user_id},
            limit=100
        )

        backup_codes_count = len(backup_codes)
        backup_codes_used = len([c for c in backup_codes if c.get("used", False)])
        backup_codes_remaining = backup_codes_count - backup_codes_used

        # Determine enabled methods
        mfa_methods = []
        if totp_enabled:
            mfa_methods.append(MFAMethod.TOTP)
        if backup_codes_remaining > 0:
            mfa_methods.append(MFAMethod.BACKUP_CODES)

        return MFAStatusResponse(
            user_id=user_id,
            totp_enabled=totp_enabled,
            totp_verified=totp_verified,
            backup_codes_count=backup_codes_count,
            backup_codes_used=backup_codes_used,
            backup_codes_remaining=backup_codes_remaining,
            mfa_methods=mfa_methods,
            last_verified_at=last_verified
        )

    async def check_mfa_required(self, user_id: str) -> bool:
        """Check if MFA is required for user"""
        status = await self.get_mfa_status(user_id)
        return status.totp_enabled

    # Helper Methods
    async def _log_verification_attempt(
        self,
        user_id: str,
        method: MFAMethod,
        success: bool,
        ip_address: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Log MFA verification attempt"""
        attempt_id = f"attempt_{uuid.uuid4().hex[:12]}"

        attempt = MFAVerificationAttempt(
            attempt_id=attempt_id,
            user_id=user_id,
            method=method,
            success=success,
            ip_address=ip_address,
            timestamp=datetime.utcnow(),
            error_message=error_message
        )

        await self.db.insert_one(self.mfa_attempts_collection, attempt.model_dump())

        # Update verification attempts counter (only for TOTP)
        if not success and method == MFAMethod.TOTP:
            config = await self.db.find_one(
                self.totp_configs_collection,
                {"user_id": user_id}
            )
            if config:
                new_attempts = config.get("verification_attempts", 0) + 1
                await self.db.update_one(
                    self.totp_configs_collection,
                    {"config_id": config["config_id"]},
                    {"$set": {
                        "verification_attempts": new_attempts,
                        "last_attempt_at": datetime.utcnow().isoformat()
                    }}
                )

    async def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited for MFA verification"""
        # Get recent attempts
        cutoff_time = datetime.utcnow() - timedelta(minutes=MFA_RATE_LIMIT_WINDOW_MINUTES)

        attempts = await self.db.find_many(
            self.mfa_attempts_collection,
            {"user_id": user_id},
            limit=100
        )

        # Filter to recent failed attempts
        recent_failed = [
            a for a in attempts
            if not a.get("success", False) and
            datetime.fromisoformat(a["timestamp"]) > cutoff_time
        ]

        return len(recent_failed) >= MFA_RATE_LIMIT_ATTEMPTS

    async def get_rate_limit_info(self, user_id: str) -> MFARateLimitInfo:
        """Get rate limit information for user"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=MFA_RATE_LIMIT_WINDOW_MINUTES)

        attempts = await self.db.find_many(
            self.mfa_attempts_collection,
            {"user_id": user_id},
            limit=100
        )

        recent_failed = [
            a for a in attempts
            if not a.get("success", False) and
            datetime.fromisoformat(a["timestamp"]) > cutoff_time
        ]

        is_locked = len(recent_failed) >= MFA_RATE_LIMIT_ATTEMPTS
        locked_until = None

        if is_locked and recent_failed:
            # Find the oldest failed attempt
            oldest_attempt = min(recent_failed, key=lambda x: x["timestamp"])
            locked_until = datetime.fromisoformat(oldest_attempt["timestamp"]) + timedelta(minutes=MFA_RATE_LIMIT_WINDOW_MINUTES)

        return MFARateLimitInfo(
            user_id=user_id,
            attempts=len(recent_failed),
            max_attempts=MFA_RATE_LIMIT_ATTEMPTS,
            window_minutes=MFA_RATE_LIMIT_WINDOW_MINUTES,
            locked_until=locked_until,
            is_locked=is_locked
        )

