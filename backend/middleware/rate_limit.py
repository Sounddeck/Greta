"""
GRETA PAI Rate Limiting Middleware
Protects API endpoints from abuse with configurable rate limits
"""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from typing import Callable
import logging
from config import get_settings

logger = logging.getLogger(__name__)

# Initialize rate limiter with Redis storage
def get_storage_uri():
    """Get storage URI for rate limiter"""
    try:
        settings = get_settings()
        if settings.redis_password:
            return f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
        else:
            return f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
    except Exception as e:
        logger.warning(f"Failed to connect to Redis for rate limiting, falling back to memory: {e}")
        return "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute", "1000/hour"],
    storage_uri=get_storage_uri(),
    strategy="fixed-window"
)

# Custom rate limit configurations for different endpoint types
RATE_LIMITS = {
    # Authentication endpoints - stricter limits
    "auth_login": "5/minute",
    "auth_register": "3/minute",
    "auth_refresh": "10/minute",
    
    # Voice endpoints - moderate limits
    "voice_stt": "30/minute",
    "voice_tts": "30/minute",
    "voice_profiles": "60/minute",
    
    # MCP endpoints - moderate limits
    "mcp_servers": "60/minute",
    "mcp_tools": "60/minute",
    "mcp_execute": "30/minute",
    
    # Hook endpoints - moderate limits
    "hook_register": "30/minute",
    "hook_execute": "60/minute",
    "hook_list": "60/minute",
    
    # LLM endpoints - stricter limits (expensive operations)
    "llm_generate": "10/minute",
    "llm_chat": "20/minute",
    
    # File upload endpoints - very strict
    "file_upload": "5/minute",
    
    # OSINT Free Tools endpoints - moderate limits (no API key required)
    "osint_free": "20/minute",
    
    # Jina MCP endpoints - moderate limits
    "jina": "20/minute",
    
    # General API endpoints
    "api_read": "100/minute",
    "api_write": "50/minute",
}

def get_rate_limit(endpoint_type: str) -> str:
    """Get rate limit for specific endpoint type"""
    return RATE_LIMITS.get(endpoint_type, "100/minute")

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom handler for rate limit exceeded errors"""
    logger.warning(
        f"Rate limit exceeded for {request.client.host} on {request.url.path}",
        extra={
            "client_ip": request.client.host,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return Response(
        content={
            "error": {
                "message": "Rate limit exceeded. Please try again later.",
                "code": "RATE_LIMIT_EXCEEDED",
                "status_code": 429,
                "details": {
                    "retry_after": exc.detail
                },
                "timestamp": str(exc)
            }
        },
        status_code=429,
        headers={
            "Retry-After": str(exc.detail),
            "X-RateLimit-Limit": str(exc.detail),
        }
    )

# Export limiter and handler
__all__ = ["limiter", "rate_limit_exceeded_handler", "get_rate_limit", "RATE_LIMITS"]
