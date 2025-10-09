"""
MFA (Multi-Factor Authentication) Models for CPAS4
Pydantic models for TOTP, backup codes, and MFA configuration
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# MFA Enums
class MFAMethod(str, Enum):
    """MFA methods supported"""
    TOTP = "totp"  # Time-based One-Time Password (Google Authenticator, Authy, etc.)
    SMS = "sms"    # SMS-based (future)
    EMAIL = "email"  # Email-based (future)
    BACKUP_CODES = "backup_codes"  # Backup codes


class MFAStatus(str, Enum):
    """MFA status for user"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    PENDING = "pending"  # Enrollment started but not verified
    VERIFIED = "verified"  # Recently verified


# TOTP Configuration Models
class TOTPConfigurationBase(BaseModel):
    """TOTP configuration base"""
    user_id: str
    enabled: bool = False
    verified: bool = False


class TOTPConfigurationCreate(TOTPConfigurationBase):
    """Create TOTP configuration"""
    secret_key: str  # Will be encrypted before storage


class TOTPConfiguration(TOTPConfigurationBase):
    """TOTP configuration with ID and timestamps"""
    config_id: str
    secret_key: str  # Encrypted
    created_at: datetime
    last_verified_at: Optional[datetime] = None
    verification_attempts: int = 0
    last_attempt_at: Optional[datetime] = None


# Backup Code Models
class BackupCodeBase(BaseModel):
    """Backup code base"""
    user_id: str
    used: bool = False


class BackupCodeCreate(BackupCodeBase):
    """Create backup code"""
    code: str  # Plain text (will be hashed before storage)


class BackupCode(BackupCodeBase):
    """Backup code with ID and timestamps"""
    code_id: str
    code_hash: str  # Hashed code
    used: bool = False
    used_at: Optional[datetime] = None
    created_at: datetime


class BackupCodeDisplay(BaseModel):
    """Backup code for display (masked or plain)"""
    code_id: str
    code: str  # Masked (e.g., "****-****-1234") or plain (only on generation)
    used: bool
    created_at: datetime


# MFA Verification Models
class MFAVerificationAttempt(BaseModel):
    """MFA verification attempt tracking"""
    attempt_id: str
    user_id: str
    method: MFAMethod
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime
    error_message: Optional[str] = None


# MFA Enrollment Models
class MFAEnrollmentBase(BaseModel):
    """MFA enrollment base"""
    user_id: str
    method: MFAMethod
    status: MFAStatus = MFAStatus.PENDING


class MFAEnrollment(MFAEnrollmentBase):
    """MFA enrollment with ID and timestamps"""
    enrollment_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    qr_code_uri: Optional[str] = None  # For TOTP
    secret_key: Optional[str] = None  # For TOTP (temporary, only during enrollment)


# TOTP Enrollment Response
class TOTPEnrollmentResponse(BaseModel):
    """Response for TOTP enrollment"""
    enrollment_id: str
    user_id: str
    secret_key: str  # Plain text secret (only shown once)
    qr_code_uri: str  # otpauth:// URI for QR code
    qr_code_base64: Optional[str] = None  # Base64 encoded QR code image
    backup_codes: Optional[List[str]] = None  # Generated backup codes
    instructions: str = "Scan the QR code with your authenticator app (Google Authenticator, Authy, Microsoft Authenticator, etc.) and enter the 6-digit code to verify."


# TOTP Verification Request
class TOTPVerificationRequest(BaseModel):
    """Request to verify TOTP code"""
    user_id: str
    code: str = Field(..., min_length=6, max_length=6, pattern="^[0-9]{6}$")


# Backup Code Verification Request
class BackupCodeVerificationRequest(BaseModel):
    """Request to verify backup code"""
    user_id: str
    code: str


# MFA Status Response
class MFAStatusResponse(BaseModel):
    """MFA status for user"""
    user_id: str
    totp_enabled: bool
    totp_verified: bool
    backup_codes_count: int
    backup_codes_used: int
    backup_codes_remaining: int
    mfa_methods: List[MFAMethod]
    last_verified_at: Optional[datetime] = None


# MFA Enable/Disable Request
class MFAToggleRequest(BaseModel):
    """Request to enable/disable MFA"""
    user_id: str
    verification_code: Optional[str] = None  # Required for disable


# Backup Codes Generation Response
class BackupCodesResponse(BaseModel):
    """Response for backup codes generation"""
    user_id: str
    codes: List[str]  # Plain text codes (only shown once)
    count: int
    created_at: datetime
    warning: str = "Save these codes in a secure location. They can only be used once and will not be shown again."


# Rate Limiting
class MFARateLimitInfo(BaseModel):
    """Rate limit information for MFA"""
    user_id: str
    attempts: int
    max_attempts: int = 5
    window_minutes: int = 15
    locked_until: Optional[datetime] = None
    is_locked: bool = False


# Constants
TOTP_ISSUER = "CPAS4/GRETA"
TOTP_DIGITS = 6
TOTP_INTERVAL = 30  # seconds
BACKUP_CODE_LENGTH = 12
BACKUP_CODE_COUNT = 10
MFA_RATE_LIMIT_ATTEMPTS = 5
MFA_RATE_LIMIT_WINDOW_MINUTES = 15

