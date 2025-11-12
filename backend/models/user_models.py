"""
User Models for GRETA PAI Authentication System
Defines user data structures and database schemas with comprehensive validation
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, validator
from bson import ObjectId
import re

from utils.validators import (
    email_validator,
    phone_validator,
    password_validator,
    validate_email_address,
    validate_phone_number,
    validate_password_strength
)


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ])
        ], serialization=core_schema.plain_serializer_function_ser_schema(str))

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


class UserBase(BaseModel):
    """Base user model with common fields"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, description="Phone number in international format")
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)
    roles: List[str] = Field(default=["user"])

    # Use new validators from utils.validators
    _validate_email = validator('email', allow_reuse=True)(email_validator)
    _validate_phone = validator('phone', allow_reuse=True)(phone_validator)

    @validator("username")
    def validate_username(cls, v):
        """Validate username format and restrictions"""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")

        # Check for reserved usernames
        reserved = ['admin', 'root', 'system', 'api', 'www', 'mail', 'ftp']
        if v.lower() in reserved:
            raise ValueError(f"Username '{v}' is reserved")

        return v.lower()

    @validator("full_name")
    def validate_full_name(cls, v):
        """Validate full name format"""
        if v and not v.strip():
            raise ValueError("Full name cannot be empty or whitespace only")
        return v.strip() if v else None

    @validator("roles")
    def validate_roles(cls, v):
        """Validate user roles"""
        valid_roles = ["user", "admin", "moderator", "developer"]
        for role in v:
            if role not in valid_roles:
                raise ValueError(f"Invalid role: {role}. Valid roles: {valid_roles}")
        return v


class UserCreate(UserBase):
    """User creation model with password"""
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)
    terms_accepted: bool = Field(..., description="Whether user has accepted terms of service")

    # Use new password validator from utils.validators
    _validate_password = validator('password', allow_reuse=True)(password_validator)

    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Validate that passwords match"""
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator('terms_accepted')
    def validate_terms_accepted(cls, v):
        """Validate terms acceptance"""
        if not v:
            raise ValueError("Terms of service must be accepted")
        return v


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, description="Updated phone number")
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    roles: Optional[List[str]] = None

    # Use new validators from utils.validators
    _validate_email = validator('email', allow_reuse=True)(email_validator)
    _validate_phone = validator('phone', allow_reuse=True)(phone_validator)

    @validator('full_name')
    def validate_full_name(cls, v):
        """Validate full name format"""
        if v is not None and not v.strip():
            raise ValueError("Full name cannot be empty or whitespace only")
        return v.strip() if v else None

    @validator("roles")
    def validate_roles(cls, v):
        """Validate user roles"""
        if v is not None:
            valid_roles = ["user", "admin", "moderator", "developer"]
            for role in v:
                if role not in valid_roles:
                    raise ValueError(f"Invalid role: {role}. Valid roles: {valid_roles}")
        return v


class UserInDB(UserBase):
    """User model as stored in database"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_count: int = Field(default=0)
    failed_login_attempts: int = Field(default=0)
    locked_until: Optional[datetime] = None
    email_verification_token: Optional[str] = None
    password_reset_token: Optional[str] = None
    password_reset_expires: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserResponse(UserBase):
    """User response model (excludes sensitive data)"""
    id: str = Field(alias="_id")
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    login_count: int
    
    class Config:
        allow_population_by_field_name = True


class UserLogin(BaseModel):
    """User login model"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)
    remember_me: bool = Field(default=False)


class PasswordChange(BaseModel):
    """Password change model"""
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)
    
    # Use new password validator from utils.validators
    _validate_new_password = validator('new_password', allow_reuse=True)(password_validator)
    
    @validator("confirm_new_password")
    def passwords_match(cls, v, values):
        """Validate that new passwords match"""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("New passwords do not match")
        return v


class PasswordReset(BaseModel):
    """Password reset model"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model"""
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)
    
    # Use new password validator from utils.validators
    _validate_new_password = validator('new_password', allow_reuse=True)(password_validator)
    
    @validator("confirm_new_password")
    def passwords_match(cls, v, values):
        """Validate that new passwords match"""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("New passwords do not match")
        return v


class EmailVerification(BaseModel):
    """Email verification model"""
    token: str = Field(..., min_length=1)


# Export all models
__all__ = [
    "PyObjectId",
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserInDB",
    "UserResponse",
    "UserLogin",
    "PasswordChange",
    "PasswordReset",
    "PasswordResetConfirm",
    "EmailVerification"
]
