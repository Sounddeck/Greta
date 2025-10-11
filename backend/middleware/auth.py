"""
JWT Authentication Middleware for Greta PAI
Provides token generation, verification, and FastAPI dependencies
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    roles: list[str] = []


class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password for storage
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in the token (user_id, username, etc.)
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """
    Decode and verify a JWT access token
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData object with user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        
        if user_id is None:
            raise credentials_exception
        
        token_data = TokenData(
            user_id=user_id,
            username=payload.get("username"),
            email=payload.get("email"),
            roles=payload.get("roles", [])
        )
        
        return token_data
        
    except JWTError:
        raise credentials_exception


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """
    FastAPI dependency to verify JWT token from Authorization header
    
    Args:
        credentials: HTTP Bearer credentials from request header
        
    Returns:
        TokenData object with user information
        
    Raises:
        HTTPException: If token is invalid or expired
        
    Usage:
        @app.get("/protected")
        async def protected_route(token_data: TokenData = Depends(verify_token)):
            return {"user_id": token_data.user_id}
    """
    token = credentials.credentials
    return decode_access_token(token)


async def get_current_user(token_data: TokenData = Depends(verify_token)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user
    
    Args:
        token_data: Token data from verify_token dependency
        
    Returns:
        User information dictionary
        
    Usage:
        @app.get("/me")
        async def get_me(user: Dict = Depends(get_current_user)):
            return user
    """
    return {
        "user_id": token_data.user_id,
        "username": token_data.username,
        "email": token_data.email,
        "roles": token_data.roles
    }


def require_role(required_role: str):
    """
    Create a dependency that requires a specific role
    
    Args:
        required_role: Role name required to access the endpoint
        
    Returns:
        FastAPI dependency function
        
    Usage:
        @app.get("/admin")
        async def admin_route(user: Dict = Depends(require_role("admin"))):
            return {"message": "Admin access granted"}
    """
    async def role_checker(token_data: TokenData = Depends(verify_token)) -> Dict[str, Any]:
        if required_role not in token_data.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return {
            "user_id": token_data.user_id,
            "username": token_data.username,
            "email": token_data.email,
            "roles": token_data.roles
        }
    
    return role_checker


# Optional: Create a dependency for optional authentication
async def optional_verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[TokenData]:
    """
    FastAPI dependency for optional authentication
    Returns None if no token provided, validates if token exists
    
    Usage:
        @app.get("/public-or-private")
        async def route(token_data: Optional[TokenData] = Depends(optional_verify_token)):
            if token_data:
                return {"message": f"Hello {token_data.username}"}
            return {"message": "Hello guest"}
    """
    if credentials is None:
        return None
    
    return decode_access_token(credentials.credentials)


# Example user authentication function (to be implemented with your user database)
async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user with username and password
    
    This is a placeholder - implement with your actual user database
    
    Args:
        username: Username
        password: Plain text password
        
    Returns:
        User dict if authenticated, None otherwise
    """
    # TODO: Implement with actual user database
    # Example:
    # user = await db.users.find_one({"username": username})
    # if user and verify_password(password, user["password_hash"]):
    #     return user
    # return None
    
    # Mock implementation for testing
    if username == "admin" and password == "admin123":
        return {
            "user_id": "admin_001",
            "username": "admin",
            "email": "admin@greta.ai",
            "roles": ["admin", "user"]
        }
    
    return None

