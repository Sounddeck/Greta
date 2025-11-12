"""
Rate Limiter Service for CPAS5

Phase 3, Task 3.3: Add Rate Limiting
Implements distributed rate limiting using Redis with sliding window algorithm.
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import redis.asyncio as redis
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 1  # Use different DB than cache
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: int = 5
    
    # Rate limit windows (in seconds)
    minute_window: int = 60
    hour_window: int = 3600
    day_window: int = 86400
    
    # Default rate limits (requests per window)
    public_per_minute: int = 100
    authenticated_per_minute: int = 1000
    admin_per_minute: int = 10000  # Very high limit
    heavy_ops_per_minute: int = 10
    write_ops_per_minute: int = 100
    
    # Whitelist
    whitelisted_ips: list = field(default_factory=lambda: ["127.0.0.1", "::1", "localhost"])


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    limit: int
    remaining: int
    reset_at: int  # Unix timestamp
    retry_after: Optional[int] = None  # Seconds to wait


@dataclass
class RateLimitMetrics:
    """Rate limit metrics"""
    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    violations_by_ip: Dict[str, int] = field(default_factory=dict)
    violations_by_user: Dict[str, int] = field(default_factory=dict)
    bypass_count: int = 0
    
    @property
    def block_ratio(self) -> float:
        """Calculate block ratio percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.blocked_requests / self.total_requests) * 100


class RateLimiterService:
    """
    Redis-based Rate Limiter Service
    
    Implements sliding window algorithm for accurate rate limiting.
    Supports per-user and per-IP rate limiting.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = RateLimitMetrics()
        logger.info("🚦 Rate Limiter Service initialized")
    
    async def connect(self):
        """Connect to Redis with connection pooling"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info(f"✅ Rate Limiter connected to Redis at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis for rate limiting: {e}")
            logger.warning("⚠️ Rate limiting will be disabled (graceful degradation)")
            self.redis_client = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("✅ Rate Limiter disconnected from Redis")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health"""
        if not self.redis_client:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Redis client not initialized"
            }
        
        try:
            start_time = time.time()
            await self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "ping_time_ms": round(ping_time, 2),
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_seconds": info.get("uptime_in_seconds")
            }
            
        except Exception as e:
            logger.error(f"❌ Rate limiter health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    def _get_rate_limit_key(self, identifier: str, window: str) -> str:
        """Generate rate limit key for Redis"""
        return f"ratelimit:{identifier}:{window}"
    
    def _is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        return ip in self.config.whitelisted_ips
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int = 60,
        is_admin: bool = False,
        ip: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limit using sliding window algorithm.
        
        Args:
            identifier: User ID, IP address, or API key
            limit: Maximum requests allowed in window
            window: Time window in seconds (default: 60)
            is_admin: Whether user is admin (bypasses limits)
            ip: IP address for whitelist check
        
        Returns:
            RateLimitResult with allowed status and metadata
        """
        self.metrics.total_requests += 1
        
        # Check for bypass conditions
        if is_admin:
            self.metrics.bypass_count += 1
            logger.debug(f"✅ Rate limit bypassed for admin: {identifier}")
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit,
                reset_at=int(time.time() + window)
            )
        
        if ip and self._is_whitelisted(ip):
            self.metrics.bypass_count += 1
            logger.debug(f"✅ Rate limit bypassed for whitelisted IP: {ip}")
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit,
                reset_at=int(time.time() + window)
            )
        
        # If Redis is not available, allow request (graceful degradation)
        if not self.redis_client:
            logger.warning("⚠️ Redis unavailable, allowing request (graceful degradation)")
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit,
                reset_at=int(time.time() + window)
            )
        
        try:
            current_time = time.time()
            window_start = current_time - window
            key = self._get_rate_limit_key(identifier, f"{window}s")
            
            # Use Redis sorted set for sliding window
            # Score is timestamp, value is unique request ID
            pipe = self.redis_client.pipeline()
            
            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            pipe.zcard(key)
            
            # Add current request
            request_id = f"{current_time}:{hashlib.md5(str(current_time).encode()).hexdigest()[:8]}"
            pipe.zadd(key, {request_id: current_time})
            
            # Set expiration on key
            pipe.expire(key, window * 2)
            
            results = await pipe.execute()
            current_count = results[1]  # Count before adding current request
            
            # Calculate remaining and reset time
            remaining = max(0, limit - current_count - 1)
            reset_at = int(current_time + window)
            
            # Check if limit exceeded
            if current_count >= limit:
                self.metrics.blocked_requests += 1
                
                # Track violations
                if identifier.startswith("user_"):
                    self.metrics.violations_by_user[identifier] = \
                        self.metrics.violations_by_user.get(identifier, 0) + 1
                elif ip:
                    self.metrics.violations_by_ip[ip] = \
                        self.metrics.violations_by_ip.get(ip, 0) + 1
                
                logger.warning(
                    f"🚫 Rate limit exceeded for {identifier}: "
                    f"{current_count}/{limit} requests in {window}s window"
                )
                
                return RateLimitResult(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=window
                )
            
            self.metrics.allowed_requests += 1
            
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_at=reset_at
            )
            
        except Exception as e:
            logger.error(f"❌ Rate limit check error: {e}")
            # On error, allow request (fail open)
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=limit,
                reset_at=int(time.time() + window)
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get rate limit metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "allowed_requests": self.metrics.allowed_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "block_ratio": round(self.metrics.block_ratio, 2),
            "bypass_count": self.metrics.bypass_count,
            "violations_by_ip": dict(sorted(
                self.metrics.violations_by_ip.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # Top 10 violators
            "violations_by_user": dict(sorted(
                self.metrics.violations_by_user.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 violators
        }
    
    def reset_metrics(self):
        """Reset rate limit metrics"""
        self.metrics = RateLimitMetrics()
        logger.info("✅ Rate limit metrics reset")

