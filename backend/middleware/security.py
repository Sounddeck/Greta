"""
GRETA PAI - Security Middleware
Security Hardening - Phase 1
Comprehensive security middleware for rate limiting, authentication, and error handling
"""
import time
from typing import Callable, Dict, Any, Optional
from functools import wraps
import hashlib
import secrets
from fastapi import Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import logging
from loguru import logger
from pydantic import BaseModel

# Security configuration
SECRET_KEY = "greta-pai-security-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security schemes
security_scheme = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """JWT token payload"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: list[str] = []


class SecurityMiddleware:
    """Centralized security middleware management"""

    def __init__(self):
        self.limiter = limiter
        self.failed_login_attempts: Dict[str, list] = {}
        self.blocked_ips: set = set()

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is temporarily blocked"""
        return ip in self.blocked_ips

    def record_failed_attempt(self, ip: str):
        """Record failed authentication attempt"""
        if ip not in self.failed_login_attempts:
            self.failed_login_attempts[ip] = []

        self.failed_login_attempts[ip].append(time.time())

        # Clean old attempts (> 1 hour)
        cutoff = time.time() - 3600
        self.failed_login_attempts[ip] = [
            t for t in self.failed_login_attempts[ip] if t > cutoff
        ]

        # Block IP if too many failed attempts
        if len(self.failed_login_attempts[ip]) >= 5:
            self.blocked_ips.add(ip)
            logger.warning(f"IP {ip} temporarily blocked due to too many failed attempts")

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            scopes: list = payload.get("scopes", [])
            if username is None:
                return None
            return TokenData(username=username, user_id=user_id, scopes=scopes)
        except JWTError:
            return None

    def hash_password(self, password: str) -> str:
        """Hash password for storage"""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> TokenData:
        """Dependency for getting current authenticated user"""
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_data = self.verify_token(credentials.credentials)
        if token_data is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalid or expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token_data


# Global security middleware instance
security_middleware = SecurityMiddleware()


def require_auth(scopes: Optional[list[str]] = None):
    """Decorator for protecting endpoints with authentication"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args or kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None and 'request' in kwargs:
                request = kwargs['request']

            if request:
                client_ip = get_remote_address(request)
                if security_middleware.is_ip_blocked(client_ip):
                    raise HTTPException(
                        status_code=429,
                        detail="Too many failed attempts. IP temporarily blocked."
                    )

            # Get current user
            current_user = None
            try:
                # Try to get from kwargs first, then check security scheme
                if 'current_user' in kwargs:
                    current_user = kwargs['current_user']
                else:
                    # This would normally use Depends, but for decorator we need manual auth
                    auth_header = request.headers.get('authorization')
                    if auth_header and auth_header.startswith('Bearer '):
                        token = auth_header.split(' ')[1]
                        current_user = security_middleware.verify_token(token)
            except Exception as e:
                logger.warning(f"Authentication failed: {e}")

            if current_user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            # Check scopes if required
            if scopes and not any(scope in current_user.scopes for scope in scopes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )

            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def rate_limit(limit: str):
    """Decorator for rate limiting"""
    def decorator(func: Callable):
        @limiter.limit(limit)
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global exception handlers
async def security_exception_handler(request: Request, exc: HTTPException):
    """Enhanced security exception handler"""
    client_ip = get_remote_address(request)

    # Log security events
    if exc.status_code in [401, 403]:
        logger.warning(f"Security violation from {client_ip}: {exc.detail} on {request.url}")
        security_middleware.record_failed_attempt(client_ip)
    elif exc.status_code == 429:
        logger.warning(f"Rate limit exceeded from {client_ip} on {request.url}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": f"SEC_{exc.status_code:03d}",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": hashlib.md5(f"{client_ip}{time.time()}".encode()).hexdigest()[:8]
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Generic exception handler with security logging"""
    client_ip = get_remote_address(request)
    request_id = hashlib.md5(f"{client_ip}{time.time()}".encode()).hexdigest()[:8]

    logger.error(f"Unhandled exception from {client_ip} on {request.url}: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "SEC_500",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
    )


# Rate limit exceeded handler
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler"""
    client_ip = get_remote_address(request)
    logger.warning(f"Rate limit exceeded from {client_ip} on {request.url}")

    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": "Rate limit exceeded. Please try again later.",
            "error_code": "SEC_429",
            "timestamp": datetime.utcnow().isoformat(),
            "retry_after": 60  # seconds
        },
        headers={"Retry-After": "60"}
    )


class SecurityHeadersMiddleware:
    """Middleware for adding security headers"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Add security headers
                security_headers = [
                    (b"X-Content-Type-Options", b"nosniff"),
                    (b"X-Frame-Options", b"DENY"),
                    (b"X-XSS-Protection", b"1; mode=block"),
                    (b"Strict-Transport-Security", b"max-age=31536000; includeSubDomains"),
                    (b"Content-Security-Policy", b"default-src 'self'"),
                    (b"Referrer-Policy", b"strict-origin-when-cross-origin"),
                ]
                headers.extend(security_headers)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


# Export security components
__all__ = [
    'SecurityMiddleware',
    'security_middleware',
    'require_auth',
    'rate_limit',
    'security_exception_handler',
    'generic_exception_handler',
    'custom_rate_limit_handler',
    'SecurityHeadersMiddleware',
]
