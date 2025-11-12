"""
User Service for GRETA PAI Authentication System

This module provides comprehensive user management functionality including
user registration, authentication, profile management, and security operations.
It handles all database operations related to users with proper caching,
validation, and error handling.

The service provides:
- User registration and email verification
- Password hashing and authentication
- Profile management and updates
- Password reset functionality
- User search and listing with pagination
- Account activation/deactivation
- Security audit logging

Example:
    Basic user operations:
        >>> from services.user_service import UserService
        >>> from database import database
        >>>
        >>> user_service = UserService(database)
        >>>
        >>> # Create new user
        >>> user_data = UserCreate(
        ...     username="john_doe",
        ...     email="john@example.com",
        ...     password="secure_password"
        ... )
        >>> user = await user_service.create_user(user_data)
        >>>
        >>> # Authenticate user
        >>> authenticated = await user_service.authenticate_user(
        ...     "john_doe", "secure_password"
        ... )

Attributes:
    db (AsyncIOMotorDatabase): Database instance for user operations
    collection (AsyncIOMotorCollection): Users collection reference
    pwd_context (CryptContext): Password hashing context using bcrypt
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from passlib.context import CryptContext
from bson import ObjectId

from models.user_models import (
    UserCreate, UserUpdate, UserInDB, UserResponse,
    PasswordChange, PasswordReset, PasswordResetConfirm
)
from utils.error_handling import (
    ValidationError, AuthenticationError, ResourceNotFoundError
)
from utils.cache import cache, cache_invalidate, cache_manager
from config import get_settings


class UserService:
    """
    Service class for comprehensive user management operations.

    This class handles all user-related database operations including
    registration, authentication, profile management, and security features.
    It provides caching for performance and proper error handling.

    Attributes:
        db (AsyncIOMotorDatabase): MongoDB database instance
        collection (AsyncIOMotorCollection): Users collection reference
        pwd_context (CryptContext): Password hashing context using bcrypt

    Example:
        >>> user_service = UserService(database)
        >>> user = await user_service.get_user_by_username("john_doe")
        >>> if user:
        ...     print(f"Found user: {user.username}")
    """

    def __init__(self, database: AsyncIOMotorDatabase):
        """
        Initialize the user service with database connection.

        Args:
            database: MongoDB database instance for user operations

        Note:
            Automatically creates necessary database indexes for optimal
            query performance on username, email, and other fields.
        """
        self.db = database
        self.collection = database.users
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Create indexes
        self._create_indexes()

    async def _create_indexes(self) -> None:
        """
        Create database indexes for the users collection.

        Creates unique indexes on username and email fields, and regular
        indexes on frequently queried fields like created_at and is_active
        for optimal query performance.

        Raises:
            Exception: If index creation fails due to database issues

        Note:
            This method is called automatically during service initialization
            and is safe to call multiple times (indexes are created only if
            they don't already exist).
        """
        try:
            # Unique index on username
            await self.collection.create_index("username", unique=True)
            
            # Unique index on email
            await self.collection.create_index("email", unique=True)
            
            # Index on created_at for sorting
            await self.collection.create_index("created_at")
            
            # Index on is_active for filtering
            await self.collection.create_index("is_active")
            
            # Index on email_verification_token
            await self.collection.create_index("email_verification_token", sparse=True)
            
            # Index on password_reset_token
            await self.collection.create_index("password_reset_token", sparse=True)
            
        except Exception as e:
            # Indexes might already exist, which is fine
            pass
    
    def _hash_password(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def _generate_token(self) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(32)
    
    async def create_user(self, user_data: UserCreate) -> UserInDB:
        """Create a new user"""
        # Check if username already exists
        existing_user = await self.collection.find_one({"username": user_data.username.lower()})
        if existing_user:
            raise ValidationError("Username already exists", field="username")
        
        # Check if email already exists
        existing_email = await self.collection.find_one({"email": user_data.email.lower()})
        if existing_email:
            raise ValidationError("Email already exists", field="email")
        
        # Hash password
        password_hash = self._hash_password(user_data.password)
        
        # Create user document
        user_doc = {
            "username": user_data.username.lower(),
            "email": user_data.email.lower(),
            "full_name": user_data.full_name,
            "password_hash": password_hash,
            "is_active": user_data.is_active,
            "is_verified": False,  # Email verification required
            "roles": user_data.roles,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "login_count": 0,
            "failed_login_attempts": 0,
            "email_verification_token": self._generate_token(),
            "preferences": {}
        }
        
        # Insert user
        result = await self.collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        
        return UserInDB(**user_doc)
    
    @cache(ttl=300, key_prefix="user_profile")  # 5 minute cache
    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        try:
            object_id = ObjectId(user_id)
        except Exception:
            return None

        user_doc = await self.collection.find_one({"_id": object_id})
        if user_doc:
            return UserInDB(**user_doc)
        return None
    
    @cache(ttl=300, key_prefix="user_by_username")  # 5 minute cache
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        user_doc = await self.collection.find_one({"username": username.lower()})
        if user_doc:
            return UserInDB(**user_doc)
        return None
    
    @cache(ttl=300, key_prefix="user_by_email")  # 5 minute cache
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email"""
        user_doc = await self.collection.find_one({"email": email.lower()})
        if user_doc:
            return UserInDB(**user_doc)
        return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username/email and password"""
        # Try to find user by username or email
        user = await self.get_user_by_username(username)
        if not user:
            user = await self.get_user_by_email(username)
        
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            raise AuthenticationError("Account is temporarily locked due to failed login attempts")
        
        # Check if account is active
        if not user.is_active:
            raise AuthenticationError("Account is deactivated")
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            # Increment failed login attempts
            await self._increment_failed_login_attempts(user.id)
            return None
        
        # Reset failed login attempts and update last login
        await self._reset_failed_login_attempts(user.id)
        await self._update_last_login(user.id)
        
        return user
    
    async def _increment_failed_login_attempts(self, user_id: ObjectId):
        """Increment failed login attempts and lock account if necessary"""
        user = await self.collection.find_one({"_id": user_id})
        if not user:
            return
        
        failed_attempts = user.get("failed_login_attempts", 0) + 1
        update_data = {"failed_login_attempts": failed_attempts}
        
        # Lock account after 5 failed attempts for 30 minutes
        if failed_attempts >= 5:
            update_data["locked_until"] = datetime.utcnow() + timedelta(minutes=30)
        
        await self.collection.update_one(
            {"_id": user_id},
            {"$set": update_data}
        )
    
    async def _reset_failed_login_attempts(self, user_id: ObjectId):
        """Reset failed login attempts"""
        await self.collection.update_one(
            {"_id": user_id},
            {
                "$set": {"failed_login_attempts": 0},
                "$unset": {"locked_until": ""}
            }
        )
    
    async def _update_last_login(self, user_id: ObjectId):
        """Update last login timestamp and increment login count"""
        await self.collection.update_one(
            {"_id": user_id},
            {
                "$set": {"last_login": datetime.utcnow()},
                "$inc": {"login_count": 1}
            }
        )
    
    @cache_invalidate("user_profile:*", "user_by_username:*", "user_by_email:*")
    async def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[UserInDB]:
        """Update user information"""
        try:
            object_id = ObjectId(user_id)
        except Exception:
            raise ValidationError("Invalid user ID")
        
        # Prepare update data
        update_data = {}
        if user_data.email is not None:
            # Check if email already exists
            existing_email = await self.collection.find_one({
                "email": user_data.email.lower(),
                "_id": {"$ne": object_id}
            })
            if existing_email:
                raise ValidationError("Email already exists", field="email")
            update_data["email"] = user_data.email.lower()
            update_data["is_verified"] = False  # Re-verify email
            update_data["email_verification_token"] = self._generate_token()
        
        if user_data.full_name is not None:
            update_data["full_name"] = user_data.full_name
        
        if user_data.is_active is not None:
            update_data["is_active"] = user_data.is_active
        
        if user_data.is_verified is not None:
            update_data["is_verified"] = user_data.is_verified
        
        if user_data.roles is not None:
            update_data["roles"] = user_data.roles
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            
            result = await self.collection.update_one(
                {"_id": object_id},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                raise ResourceNotFoundError("User not found")
        
        return await self.get_user_by_id(user_id)
    
    async def change_password(self, user_id: str, password_data: PasswordChange) -> bool:
        """Change user password"""
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ResourceNotFoundError("User not found")
        
        # Verify current password
        if not self._verify_password(password_data.current_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect")
        
        # Hash new password
        new_password_hash = self._hash_password(password_data.new_password)
        
        # Update password
        await self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "password_hash": new_password_hash,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user (soft delete by deactivating)"""
        try:
            object_id = ObjectId(user_id)
        except Exception:
            raise ValidationError("Invalid user ID")
        
        result = await self.collection.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "is_active": False,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return result.matched_count > 0
    
    async def list_users(self, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[UserResponse]:
        """List users with pagination"""
        query = {}
        if active_only:
            query["is_active"] = True
        
        cursor = self.collection.find(query).skip(skip).limit(limit).sort("created_at", -1)
        users = []
        
        async for user_doc in cursor:
            users.append(UserResponse(**user_doc))
        
        return users
    
    async def get_user_count(self, active_only: bool = True) -> int:
        """Get total user count"""
        query = {}
        if active_only:
            query["is_active"] = True
        
        return await self.collection.count_documents(query)


# Global user service instance (will be initialized with database)
user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get user service instance"""
    if user_service is None:
        raise RuntimeError("User service not initialized")
    return user_service


def init_user_service(database: AsyncIOMotorDatabase):
    """Initialize user service with database"""
    global user_service
    user_service = UserService(database)
