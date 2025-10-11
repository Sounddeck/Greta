"""
Authentication Router for Greta PAI
Provides login, logout, token refresh, and user management endpoints
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

from middleware.auth import (
    create_access_token,
    authenticate_user,
    get_current_user,
    get_password_hash,
    Token,
    TokenData,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

from middleware.rate_limit import limiter, get_rate_limit


router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """User registration request model"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response model"""
    user_id: str
    username: str
    email: str
    roles: list[str]


@router.post("/login"
@limiter.limit(get_rate_limit("auth_login"))
@limiter.limit(get_rate_limit("auth_login")), response_model=Token)
async def login(request: LoginRequest):
    """
    Login endpoint - authenticate user and return JWT token
    
    Args:
        request: Login credentials (username, password)
        
    Returns:
        JWT access token
        
    Raises:
        HTTPException: If credentials are invalid
    """
    user = await authenticate_user(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "roles": user.get("roles", ["user"])
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    )


@router.post("/register"
@limiter.limit(get_rate_limit("auth_register"))
@limiter.limit(get_rate_limit("auth_register")), response_model=UserResponse)
async def register(request: RegisterRequest):
    """
    Register a new user
    
    Args:
        request: User registration data
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: If username/email already exists
    """
    # TODO: Implement with actual user database
    # Example:
    # existing_user = await db.users.find_one({
    #     "$or": [
    #         {"username": request.username},
    #         {"email": request.email}
    #     ]
    # })
    # if existing_user:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Username or email already registered"
    #     )
    
    # Hash password
    password_hash = get_password_hash(request.password)
    
    # Create user
    # user_id = str(uuid.uuid4())
    # new_user = {
    #     "user_id": user_id,
    #     "username": request.username,
    #     "email": request.email,
    #     "password_hash": password_hash,
    #     "full_name": request.full_name,
    #     "roles": ["user"],
    #     "created_at": datetime.utcnow(),
    #     "is_active": True
    # }
    # await db.users.insert_one(new_user)
    
    # Mock response for testing
    return UserResponse(
        user_id="user_001",
        username=request.username,
        email=request.email,
        roles=["user"]
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current authenticated user information
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        User information
    """
    return UserResponse(
        user_id=current_user["user_id"],
        username=current_user["username"],
        email=current_user["email"],
        roles=current_user["roles"]
    )


@router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Logout endpoint
    
    Note: JWT tokens are stateless, so logout is handled client-side
    by removing the token. This endpoint can be used for logging/analytics.
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        Success message
    """
    # TODO: Optionally add token to blacklist in Redis/database
    # await redis.setex(f"blacklist:{token}", ACCESS_TOKEN_EXPIRE_MINUTES * 60, "1")
    
    return {
        "success": True,
        "message": "Logged out successfully",
        "user_id": current_user["user_id"]
    }


@router.post("/refresh"
@limiter.limit(get_rate_limit("auth_refresh"))
@limiter.limit(get_rate_limit("auth_refresh")), response_model=Token)
async def refresh_token(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Refresh access token
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        New JWT access token
    """
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "user_id": current_user["user_id"],
            "username": current_user["username"],
            "email": current_user["email"],
            "roles": current_user["roles"]
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/verify")
async def verify_token_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Verify if token is valid
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        Token validity status
    """
    return {
        "valid": True,
        "user_id": current_user["user_id"],
        "username": current_user["username"]
    }

