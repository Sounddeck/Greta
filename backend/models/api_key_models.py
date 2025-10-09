"""
API Key Models for CPAS4
Pydantic models for API key management
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# API Key Enums
class APIKeyScope(str, Enum):
    """API key scopes/permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    ALL = "all"


class APIKeyStatus(str, Enum):
    """API key status"""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


# API Key Models
class APIKeyBase(BaseModel):
    """API key base"""
    name: str
    scopes: List[APIKeyScope] = [APIKeyScope.READ]
    expires_at: Optional[datetime] = None


class APIKeyCreate(APIKeyBase):
    """Create API key request"""
    user_id: str


class APIKey(APIKeyBase):
    """API key with ID and metadata"""
    key_id: str
    user_id: str
    key_hash: str  # SHA-256 hash of the key
    prefix: str  # First 8 characters for identification (e.g., "cpas_abc")
    enabled: bool = True
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None


class APIKeyResponse(BaseModel):
    """API key response (includes full key - only shown once)"""
    key_id: str
    user_id: str
    name: str
    key: str  # Full API key (only shown on creation)
    prefix: str
    scopes: List[APIKeyScope]
    expires_at: Optional[datetime] = None
    created_at: datetime
    warning: str = "Save this API key securely. It will not be shown again."


class APIKeyList(BaseModel):
    """API key list item (without full key)"""
    key_id: str
    user_id: str
    name: str
    prefix: str
    scopes: List[APIKeyScope]
    enabled: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    status: APIKeyStatus


class APIKeyUpdate(BaseModel):
    """Update API key"""
    name: Optional[str] = None
    scopes: Optional[List[APIKeyScope]] = None
    enabled: Optional[bool] = None
    expires_at: Optional[datetime] = None


class APIKeyRevoke(BaseModel):
    """Revoke API key request"""
    reason: Optional[str] = None
    revoked_by: Optional[str] = None


# API Key Usage Models
class APIKeyUsageRecord(BaseModel):
    """API key usage record"""
    usage_id: str
    key_id: str
    user_id: str
    endpoint: str
    method: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime
    response_status: Optional[int] = None


class APIKeyUsageStats(BaseModel):
    """API key usage statistics"""
    key_id: str
    total_requests: int
    requests_last_hour: int
    requests_last_day: int
    requests_last_week: int
    last_used_at: Optional[datetime] = None
    most_used_endpoints: List[Dict[str, Any]] = []


# API Key Rotation Models
class APIKeyRotateResponse(BaseModel):
    """API key rotation response"""
    old_key_id: str
    new_key_id: str
    new_key: str  # Full new API key
    prefix: str
    message: str = "Old key has been revoked. Use the new key for future requests."
    warning: str = "Save this API key securely. It will not be shown again."


# API Key Verification Models
class APIKeyVerificationResult(BaseModel):
    """API key verification result"""
    valid: bool
    key_id: Optional[str] = None
    user_id: Optional[str] = None
    scopes: Optional[List[APIKeyScope]] = None
    error: Optional[str] = None


# Constants
API_KEY_PREFIX = "cpas_"
API_KEY_LENGTH = 40  # Total length including prefix (cpas_ + 35 random chars)
API_KEY_RANDOM_LENGTH = 35  # Length of random part

