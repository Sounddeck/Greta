"""
Single User Mode Utility

For single-user applications, this module provides a default system user
that owns all resources. This maintains the owner_id architecture for
data consistency while avoiding the complexity of multi-user authentication.
"""

import logging
from typing import Optional
from datetime import datetime

from database import Database

logger = logging.getLogger(__name__)

# Default system user configuration
DEFAULT_USER_ID = "system_user_default"
DEFAULT_USER_EMAIL = "system@greta.local"
DEFAULT_USER_USERNAME = "system"
DEFAULT_USER_NAME = "System User"


async def get_or_create_default_user(db: Database) -> str:
    """
    Get or create the default system user for single-user mode.
    
    This function ensures a default user exists in the database and returns
    their user ID. All business resources will be owned by this user.
    
    Args:
        db: Database instance
        
    Returns:
        str: The default user's ID
    """
    try:
        # Check if default user already exists
        existing_user = await db.find_one(
            "users",
            {"username": DEFAULT_USER_USERNAME}
        )
        
        if existing_user:
            user_id = existing_user.get("_id")
            logger.debug(f"Default system user already exists: {user_id}")
            return str(user_id)
        
        # Create default user
        default_user = {
            "_id": DEFAULT_USER_ID,
            "username": DEFAULT_USER_USERNAME,
            "email": DEFAULT_USER_EMAIL,
            "full_name": DEFAULT_USER_NAME,
            "password_hash": "N/A",  # No password needed for single-user mode
            "is_active": True,
            "is_verified": True,
            "role": "admin",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "login_count": 0,
            "failed_login_attempts": 0,
            "locked_until": None,
            "preferences": {
                "theme": "dark",
                "language": "en",
                "timezone": "UTC"
            },
            "metadata": {
                "single_user_mode": True,
                "description": "Default system user for single-user application"
            }
        }
        
        # Insert default user
        user_id = await db.insert_one("users", default_user)
        
        if user_id:
            logger.info(f"✅ Created default system user: {user_id}")
            return str(user_id)
        else:
            # If insert failed, return the default ID anyway
            logger.warning("⚠️ Could not insert default user, using default ID")
            return DEFAULT_USER_ID
            
    except Exception as e:
        logger.error(f"❌ Error getting/creating default user: {e}")
        # Return default ID as fallback
        return DEFAULT_USER_ID


async def get_default_user_id(db: Database) -> str:
    """
    Convenience function to get the default user ID.
    
    Args:
        db: Database instance
        
    Returns:
        str: The default user's ID
    """
    return await get_or_create_default_user(db)


class SingleUserDependency:
    """
    FastAPI dependency that provides the default user ID for single-user mode.
    
    Usage:
        @router.post("/endpoint")
        async def endpoint(user_id: str = Depends(get_single_user_id)):
            # user_id will be the default system user's ID
            pass
    """
    
    def __init__(self):
        self._cached_user_id: Optional[str] = None
    
    async def __call__(self, db: Database = None) -> str:
        """
        Get the default user ID, with caching for performance.
        
        Args:
            db: Database instance (optional, will create if not provided)
            
        Returns:
            str: The default user's ID
        """
        # Return cached value if available
        if self._cached_user_id:
            return self._cached_user_id
        
        # Get or create database connection
        if db is None:
            db = Database()
            await db.connect()
        
        # Get default user ID
        self._cached_user_id = await get_or_create_default_user(db)
        return self._cached_user_id


# Global instance for dependency injection
get_single_user_id = SingleUserDependency()


async def initialize_single_user_mode(db: Database) -> None:
    """
    Initialize single-user mode by ensuring the default user exists.
    
    This should be called during application startup.
    
    Args:
        db: Database instance
    """
    try:
        user_id = await get_or_create_default_user(db)
        logger.info(f"✅ Single-user mode initialized with user ID: {user_id}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize single-user mode: {e}")

