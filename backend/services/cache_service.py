"""
Cache Service for CPAS5

Phase 3, Task 3.2: Implement Caching Layer (12 hours)

This service provides:
- Redis-based caching infrastructure
- Cache invalidation strategies (TTL, event-based, pattern-based)
- Cache serialization/deserialization
- Cache health checks and monitoring
- Cache metrics (hit/miss ratio, size, performance)
"""

import logging
import json
import pickle
import time
from typing import Any, Optional, List, Dict, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration for different data types"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = False  # We'll handle serialization manually
    
    # Default TTLs for different data types (in seconds)
    default_ttl: int = 300  # 5 minutes
    stats_ttl: int = 300  # 5 minutes
    list_ttl: int = 120  # 2 minutes
    template_ttl: int = 600  # 10 minutes
    session_ttl: int = 3600  # 1 hour
    config_ttl: int = 1800  # 30 minutes


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    invalidations: int = 0
    errors: int = 0
    total_response_time_ms: float = 0.0
    cache_size_bytes: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def average_response_time_ms(self) -> float:
        """Calculate average cache response time"""
        total_ops = self.hits + self.misses + self.sets
        return (self.total_response_time_ms / total_ops) if total_ops > 0 else 0.0


class CacheService:
    """
    Redis-based Cache Service
    
    Phase 3, Task 3.2: Implement Caching Layer
    Provides caching infrastructure with invalidation strategies and monitoring.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.redis_client: Optional[Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self.metrics = CacheMetrics()
        self.connected = False
        self._invalidation_handlers: Dict[str, List[Callable]] = {}
        logger.info("✅ Cache Service initialized")
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    async def connect(self):
        """
        Connect to Redis with connection pooling.
        
        Phase 3, Task 3.2: Set up Redis connection and configuration
        """
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=self.config.decode_responses
            )
            
            # Create Redis client
            self.redis_client = Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            self.connected = True
            logger.info(
                f"✅ Connected to Redis at {self.config.host}:{self.config.port} "
                f"(pool size: {self.config.max_connections})"
            )
            
        except RedisConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.connected = False
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to Redis: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            self.connected = False
            logger.info("✅ Disconnected from Redis")
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting from Redis: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection health.
        
        Phase 3, Task 3.2: Add cache health checks and monitoring
        """
        try:
            if not self.redis_client:
                return {
                    "status": "disconnected",
                    "connected": False,
                    "error": "Redis client not initialized"
                }
            
            # Ping Redis
            start_time = time.time()
            await self.redis_client.ping()
            ping_time_ms = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "ping_time_ms": ping_time_ms,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_seconds": info.get("uptime_in_seconds")
            }
            
        except Exception as e:
            logger.error(f"❌ Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    # ========================================================================
    # SERIALIZATION
    # ========================================================================
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize Python object for Redis storage.
        
        Phase 3, Task 3.2: Configure cache serialization/deserialization
        """
        try:
            # Use pickle for complex Python objects
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"❌ Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis"""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"❌ Deserialization error: {e}")
            raise
    
    # ========================================================================
    # CORE CACHE OPERATIONS
    # ========================================================================
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Phase 3, Task 3.2: Implement CacheService with get operation
        """
        if not self.connected or not self.redis_client:
            logger.warning("⚠️ Cache not connected, skipping get")
            self.metrics.misses += 1
            return None
        
        start_time = time.time()
        
        try:
            data = await self.redis_client.get(key)
            
            response_time_ms = (time.time() - start_time) * 1000
            self.metrics.total_response_time_ms += response_time_ms
            
            if data is None:
                self.metrics.misses += 1
                logger.debug(f"❌ Cache miss: {key} ({response_time_ms:.2f}ms)")
                return None
            
            value = self._deserialize(data)
            self.metrics.hits += 1
            logger.debug(f"✅ Cache hit: {key} ({response_time_ms:.2f}ms)")
            return value
            
        except RedisError as e:
            logger.error(f"❌ Redis error getting key {key}: {e}")
            self.metrics.errors += 1
            self.metrics.misses += 1
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error getting key {key}: {e}")
            self.metrics.errors += 1
            self.metrics.misses += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache with optional TTL.
        
        Phase 3, Task 3.2: Implement CacheService with set operation
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
            nx: Only set if key doesn't exist
            xx: Only set if key exists
        """
        if not self.connected or not self.redis_client:
            logger.warning("⚠️ Cache not connected, skipping set")
            return False
        
        start_time = time.time()
        
        try:
            data = self._serialize(value)
            ttl_seconds = ttl if ttl is not None else self.config.default_ttl
            
            result = await self.redis_client.set(
                key,
                data,
                ex=ttl_seconds,
                nx=nx,
                xx=xx
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            self.metrics.total_response_time_ms += response_time_ms
            self.metrics.sets += 1
            
            if result:
                logger.debug(f"✅ Cache set: {key} (TTL: {ttl_seconds}s, {response_time_ms:.2f}ms)")
            else:
                logger.debug(f"⚠️ Cache set failed: {key} (nx={nx}, xx={xx})")
            
            return bool(result)
            
        except RedisError as e:
            logger.error(f"❌ Redis error setting key {key}: {e}")
            self.metrics.errors += 1
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error setting key {key}: {e}")
            self.metrics.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Phase 3, Task 3.2: Implement CacheService with delete operation
        """
        if not self.connected or not self.redis_client:
            logger.warning("⚠️ Cache not connected, skipping delete")
            return False
        
        try:
            result = await self.redis_client.delete(key)
            self.metrics.deletes += 1
            
            if result > 0:
                logger.debug(f"✅ Cache delete: {key}")
                return True
            else:
                logger.debug(f"⚠️ Cache delete failed (key not found): {key}")
                return False
            
        except RedisError as e:
            logger.error(f"❌ Redis error deleting key {key}: {e}")
            self.metrics.errors += 1
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error deleting key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache keys matching pattern.

        Phase 3, Task 3.2: Implement CacheService with clear operation

        Args:
            pattern: Redis key pattern (e.g., "crm:*", "stats:*")
                    If None, clears all keys (use with caution!)

        Returns:
            Number of keys deleted
        """
        if not self.connected or not self.redis_client:
            logger.warning("⚠️ Cache not connected, skipping clear")
            return 0

        try:
            if pattern:
                # Find keys matching pattern
                keys = []
                async for key in self.redis_client.scan_iter(match=pattern):
                    keys.append(key)

                if keys:
                    result = await self.redis_client.delete(*keys)
                    self.metrics.invalidations += result
                    logger.info(f"✅ Cache cleared: {result} keys matching '{pattern}'")
                    return result
                else:
                    logger.debug(f"⚠️ No keys found matching pattern: {pattern}")
                    return 0
            else:
                # Clear all keys (dangerous!)
                await self.redis_client.flushdb()
                logger.warning("⚠️ Cache cleared: ALL KEYS (flushdb)")
                return -1  # Unknown count

        except RedisError as e:
            logger.error(f"❌ Redis error clearing cache: {e}")
            self.metrics.errors += 1
            return 0
        except Exception as e:
            logger.error(f"❌ Unexpected error clearing cache: {e}")
            self.metrics.errors += 1
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.connected or not self.redis_client:
            return False

        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"❌ Error checking key existence: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key in seconds (-1 = no expiry, -2 = doesn't exist)"""
        if not self.connected or not self.redis_client:
            return -2

        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"❌ Error getting TTL: {e}")
            return -2

    # ========================================================================
    # CACHE INVALIDATION STRATEGIES
    # ========================================================================

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache keys matching pattern.

        Phase 3, Task 3.2: Implement cache key patterns for easy bulk invalidation

        Example patterns:
            "crm:contacts:*" - All contact caches
            "crm:stats:*" - All CRM stats caches
            "ats:jobs:*" - All job listing caches
        """
        count = await self.clear(pattern)
        if count > 0:
            logger.info(f"✅ Invalidated {count} cache entries matching '{pattern}'")
        return count

    async def invalidate_on_event(self, event_type: str, entity_id: str):
        """
        Invalidate cache based on events (create/update/delete).

        Phase 3, Task 3.2: Add event-based cache invalidation

        Args:
            event_type: Type of event (create, update, delete)
            entity_id: ID of the entity that changed
        """
        # Call registered invalidation handlers
        if event_type in self._invalidation_handlers:
            for handler in self._invalidation_handlers[event_type]:
                try:
                    await handler(entity_id)
                except Exception as e:
                    logger.error(f"❌ Error in invalidation handler: {e}")

        logger.debug(f"✅ Cache invalidation triggered: {event_type} for {entity_id}")

    def register_invalidation_handler(self, event_type: str, handler: Callable):
        """Register a handler for cache invalidation events"""
        if event_type not in self._invalidation_handlers:
            self._invalidation_handlers[event_type] = []
        self._invalidation_handlers[event_type].append(handler)
        logger.debug(f"✅ Registered invalidation handler for {event_type}")

    async def warm_cache(self, key: str, loader: Callable, ttl: Optional[int] = None) -> Any:
        """
        Warm cache with data from loader function.

        Phase 3, Task 3.2: Implement cache warming strategies for critical data

        Args:
            key: Cache key
            loader: Async function to load data if not cached
            ttl: Time-to-live in seconds

        Returns:
            Cached or loaded data
        """
        # Try to get from cache first
        value = await self.get(key)

        if value is not None:
            return value

        # Load data
        try:
            value = await loader()

            # Cache the loaded data
            await self.set(key, value, ttl=ttl)

            logger.debug(f"✅ Cache warmed: {key}")
            return value

        except Exception as e:
            logger.error(f"❌ Error warming cache for {key}: {e}")
            raise

    # ========================================================================
    # CACHE VERSIONING
    # ========================================================================

    def versioned_key(self, base_key: str, version: str = "v1") -> str:
        """
        Create versioned cache key.

        Phase 3, Task 3.2: Add cache versioning to handle schema changes

        Example:
            versioned_key("crm:stats", "v2") -> "crm:stats:v2"
        """
        return f"{base_key}:{version}"

    async def migrate_cache_version(self, base_key: str, old_version: str, new_version: str):
        """Migrate cache from old version to new version"""
        old_key = self.versioned_key(base_key, old_version)
        new_key = self.versioned_key(base_key, new_version)

        # Delete old version
        await self.delete(old_key)

        logger.info(f"✅ Cache version migrated: {old_key} -> {new_key}")

    # ========================================================================
    # CACHE METRICS AND MONITORING
    # ========================================================================

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.

        Phase 3, Task 3.2: Implement cache hit/miss ratio tracking
        """
        # Get cache size
        if self.connected and self.redis_client:
            try:
                info = await self.redis_client.info("memory")
                self.metrics.cache_size_bytes = info.get("used_memory", 0)
            except Exception as e:
                logger.error(f"❌ Error getting cache size: {e}")

        return {
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_ratio": self.metrics.hit_ratio,
            "sets": self.metrics.sets,
            "deletes": self.metrics.deletes,
            "invalidations": self.metrics.invalidations,
            "errors": self.metrics.errors,
            "average_response_time_ms": self.metrics.average_response_time_ms,
            "cache_size_bytes": self.metrics.cache_size_bytes,
            "cache_size_mb": self.metrics.cache_size_bytes / (1024 * 1024),
            "connected": self.connected
        }

    def reset_metrics(self):
        """Reset cache metrics"""
        self.metrics = CacheMetrics()
        logger.info("✅ Cache metrics reset")

    # ========================================================================
    # HELPER METHODS FOR SPECIFIC DATA TYPES
    # ========================================================================

    def get_stats_key(self, entity_type: str) -> str:
        """Get cache key for stats (5-minute TTL)"""
        return f"stats:{entity_type}"

    def get_list_key(self, entity_type: str, filters: str = "") -> str:
        """Get cache key for list operations (2-minute TTL)"""
        return f"list:{entity_type}:{filters}" if filters else f"list:{entity_type}"

    def get_template_key(self, template_id: str) -> str:
        """Get cache key for templates (10-minute TTL)"""
        return f"template:{template_id}"

    def get_session_key(self, session_id: str) -> str:
        """Get cache key for sessions (1-hour TTL)"""
        return f"session:{session_id}"

    def get_config_key(self, config_name: str) -> str:
        """Get cache key for configuration (30-minute TTL)"""
        return f"config:{config_name}"

    async def cache_stats(self, entity_type: str, data: Any) -> bool:
        """Cache stats with 5-minute TTL"""
        key = self.get_stats_key(entity_type)
        return await self.set(key, data, ttl=self.config.stats_ttl)

    async def cache_list(self, entity_type: str, data: Any, filters: str = "") -> bool:
        """Cache list data with 2-minute TTL"""
        key = self.get_list_key(entity_type, filters)
        return await self.set(key, data, ttl=self.config.list_ttl)

    async def cache_template(self, template_id: str, data: Any) -> bool:
        """Cache template with 10-minute TTL"""
        key = self.get_template_key(template_id)
        return await self.set(key, data, ttl=self.config.template_ttl)

    async def cache_session(self, session_id: str, data: Any) -> bool:
        """Cache session with 1-hour TTL"""
        key = self.get_session_key(session_id)
        return await self.set(key, data, ttl=self.config.session_ttl)

    async def cache_config(self, config_name: str, data: Any) -> bool:
        """Cache configuration with 30-minute TTL"""
        key = self.get_config_key(config_name)
        return await self.set(key, data, ttl=self.config.config_ttl)

