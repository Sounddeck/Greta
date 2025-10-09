"""
Rate Limiting Models for CPAS4
Pydantic models for rate limiting rules, violations, and status
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# Rate Limit Enums
class RateLimitScope(str, Enum):
    """Rate limit scope"""
    USER = "user"      # Per user
    IP = "ip"          # Per IP address
    GLOBAL = "global"  # Global (all requests)
    ENDPOINT = "endpoint"  # Per endpoint


class RateLimitAction(str, Enum):
    """Action to take when rate limit exceeded"""
    BLOCK = "block"    # Block request (429)
    LOG = "log"        # Log only (allow request)
    WARN = "warn"      # Warn user (allow request)


# Rate Limit Rule Models
class RateLimitRuleBase(BaseModel):
    """Rate limit rule base"""
    name: str
    endpoint_pattern: str = "*"  # Glob pattern for endpoints (* = all)
    max_requests: int = Field(..., gt=0)
    window_seconds: int = Field(..., gt=0)
    scope: RateLimitScope = RateLimitScope.USER
    action: RateLimitAction = RateLimitAction.BLOCK
    enabled: bool = True


class RateLimitRuleCreate(RateLimitRuleBase):
    """Create rate limit rule"""
    pass


class RateLimitRule(RateLimitRuleBase):
    """Rate limit rule with ID and timestamps"""
    rule_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None


class RateLimitRuleUpdate(BaseModel):
    """Update rate limit rule"""
    name: Optional[str] = None
    endpoint_pattern: Optional[str] = None
    max_requests: Optional[int] = Field(None, gt=0)
    window_seconds: Optional[int] = Field(None, gt=0)
    scope: Optional[RateLimitScope] = None
    action: Optional[RateLimitAction] = None
    enabled: Optional[bool] = None


# Rate Limit Violation Models
class RateLimitViolation(BaseModel):
    """Rate limit violation record"""
    violation_id: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    endpoint: str
    rule_id: str
    rule_name: str
    timestamp: datetime
    requests_count: int
    max_requests: int
    window_seconds: int


# Rate Limit Status Models
class RateLimitStatus(BaseModel):
    """Current rate limit status"""
    identifier: str  # user_id or ip_address
    scope: RateLimitScope
    endpoint: Optional[str] = None
    current_requests: int
    max_requests: int
    window_seconds: int
    reset_at: datetime
    is_limited: bool
    remaining_requests: int


class RateLimitHeaders(BaseModel):
    """Rate limit headers for HTTP responses"""
    limit: int  # X-RateLimit-Limit
    remaining: int  # X-RateLimit-Remaining
    reset: int  # X-RateLimit-Reset (Unix timestamp)


# Rate Limit Config Models
class RateLimitConfig(BaseModel):
    """Rate limit configuration"""
    enabled: bool = True
    default_max_requests: int = 100
    default_window_seconds: int = 60
    default_scope: RateLimitScope = RateLimitScope.USER
    default_action: RateLimitAction = RateLimitAction.BLOCK
    excluded_paths: List[str] = [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
        "/static/"
    ]


# Rate Limit Statistics Models
class RateLimitStats(BaseModel):
    """Rate limit statistics"""
    total_rules: int
    enabled_rules: int
    total_violations: int
    violations_last_hour: int
    violations_last_day: int
    top_violators: List[Dict[str, Any]] = []


# Request Tracking Models
class RequestRecord(BaseModel):
    """Request record for rate limiting"""
    identifier: str  # user_id or ip_address
    endpoint: str
    timestamp: datetime
    rule_id: str


# Constants
DEFAULT_RATE_LIMIT_RULES = [
    {
        "name": "Global API Rate Limit",
        "endpoint_pattern": "/api/*",
        "max_requests": 1000,
        "window_seconds": 60,
        "scope": RateLimitScope.USER,
        "action": RateLimitAction.BLOCK,
        "enabled": True
    },
    {
        "name": "Authentication Rate Limit",
        "endpoint_pattern": "/api/v1/auth/*",
        "max_requests": 10,
        "window_seconds": 60,
        "scope": RateLimitScope.IP,
        "action": RateLimitAction.BLOCK,
        "enabled": True
    },
    {
        "name": "MFA Verification Rate Limit",
        "endpoint_pattern": "/api/v1/mfa/*/verify*",
        "max_requests": 5,
        "window_seconds": 900,  # 15 minutes
        "scope": RateLimitScope.USER,
        "action": RateLimitAction.BLOCK,
        "enabled": True
    },
    {
        "name": "API Key Creation Rate Limit",
        "endpoint_pattern": "/api/v1/api-keys",
        "max_requests": 5,
        "window_seconds": 3600,  # 1 hour
        "scope": RateLimitScope.USER,
        "action": RateLimitAction.BLOCK,
        "enabled": True
    }
]

