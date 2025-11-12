"""
Authentication-related Pydantic Models for GRETA PAI
JWT tokens, login/logout, API keys, and authentication schemas
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

from utils.validators import email_validator, validate_email_address


class TokenType(str, Enum):
    """Token type enumeration"""
    ACCESS = "access"
    REFRESH = "refresh"
    RESET = "reset"
    VERIFICATION = "verification"


class APIKeyScope(str, Enum):
    """API key scope enumeration"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    VOICE = "voice"
    ANALYTICS = "analytics"


class LoginRequest(BaseModel):
    """User login request"""
    
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Username or email address"
    )
    password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="User password"
    )
    remember_me: bool = Field(
        default=False,
        description="Whether to extend session duration"
    )
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if not v or not v.strip():
            raise ValueError("Username cannot be empty")
        return v.strip()


class LoginResponse(BaseModel):
    """User login response"""
    
    access_token: str = Field(
        ...,
        description="JWT access token"
    )
    refresh_token: str = Field(
        ...,
        description="JWT refresh token"
    )
    token_type: str = Field(
        default="bearer",
        description="Token type (always 'bearer')"
    )
    expires_in: int = Field(
        ...,
        description="Access token expiration time in seconds"
    )
    user_id: str = Field(
        ...,
        description="User identifier"
    )
    username: str = Field(
        ...,
        description="Username"
    )
    roles: List[str] = Field(
        ...,
        description="User roles"
    )


class TokenResponse(BaseModel):
    """Generic token response"""
    
    token: str = Field(
        ...,
        description="JWT token"
    )
    token_type: str = Field(
        default="bearer",
        description="Token type"
    )
    expires_in: int = Field(
        ...,
        description="Token expiration time in seconds"
    )
    expires_at: datetime = Field(
        ...,
        description="Token expiration timestamp"
    )


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    
    refresh_token: str = Field(
        ...,
        description="Valid refresh token"
    )


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    
    email: str = Field(
        ...,
        description="User email address"
    )
    
    # Use email validator from utils.validators
    _validate_email = validator('email', allow_reuse=True)(email_validator)


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    
    current_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Current password"
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password"
    )
    confirm_new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password confirmation"
    )
    
    @validator('confirm_new_password')
    def validate_password_match(cls, v, values):
        """Validate password confirmation matches"""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError("Password confirmation does not match")
        return v


class EmailVerificationRequest(BaseModel):
    """Email verification request"""
    
    token: str = Field(
        ...,
        min_length=1,
        description="Email verification token"
    )


class APIKeyCreate(BaseModel):
    """API key creation request"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="API key name/description"
    )
    scopes: List[APIKeyScope] = Field(
        ...,
        description="API key permissions"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Optional expiration date"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate API key name"""
        if not v or not v.strip():
            raise ValueError("API key name cannot be empty")
        return v.strip()
    
    @validator('scopes')
    def validate_scopes(cls, v):
        """Validate API key scopes"""
        if not v:
            raise ValueError("At least one scope is required")
        return v
    
    @validator('expires_at')
    def validate_expiration(cls, v):
        """Validate expiration date"""
        if v and v <= datetime.utcnow():
            raise ValueError("Expiration date must be in the future")
        return v


class APIKeyResponse(BaseModel):
    """API key response"""
    
    id: str = Field(
        ...,
        description="API key identifier"
    )
    name: str = Field(
        ...,
        description="API key name"
    )
    key: Optional[str] = Field(
        None,
        description="API key value (only shown on creation)"
    )
    scopes: List[APIKeyScope] = Field(
        ...,
        description="API key permissions"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Expiration timestamp"
    )
    last_used: Optional[datetime] = Field(
        None,
        description="Last usage timestamp"
    )
    is_active: bool = Field(
        ...,
        description="Whether API key is active"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class JWTPayload(BaseModel):
    """JWT token payload"""
    
    sub: str = Field(
        ...,
        description="Subject (user ID)"
    )
    username: str = Field(
        ...,
        description="Username"
    )
    roles: List[str] = Field(
        ...,
        description="User roles"
    )
    token_type: TokenType = Field(
        ...,
        description="Token type"
    )
    exp: int = Field(
        ...,
        description="Expiration timestamp"
    )
    iat: int = Field(
        ...,
        description="Issued at timestamp"
    )
    jti: Optional[str] = Field(
        None,
        description="JWT ID (for token revocation)"
    )


class LogoutRequest(BaseModel):
    """Logout request"""
    
    refresh_token: Optional[str] = Field(
        None,
        description="Refresh token to invalidate"
    )
    logout_all: bool = Field(
        default=False,
        description="Whether to logout from all devices"
    )


# Export all auth models
__all__ = [
    "TokenType",
    "APIKeyScope",
    "LoginRequest",
    "LoginResponse",
    "TokenResponse",
    "RefreshTokenRequest",
    "PasswordResetRequest",
    "PasswordChangeRequest",
    "EmailVerificationRequest",
    "APIKeyCreate",
    "APIKeyResponse",
    "JWTPayload",
    "LogoutRequest"
]
