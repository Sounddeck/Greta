"""
Cache Utilities for GRETA PAI
Provides Redis client and caching functionality
"""

import logging
from typing import Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InMemoryCache:
    """
    Simple in-memory cache implementation
    Used as fallback when Redis is not available
    """
    
    def __init__(self):
        self.cache = {}
        self.expiry = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Check if key exists and hasn't expired
            if key in self.cache:
                if key in self.expiry:
                    if datetime.utcnow() > self.expiry[key]:
                        # Expired, remove it
                        del self.cache[key]
                        del self.expiry[key]
                        return None
                return self.cache[key]
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None):
        """Set value in cache"""
        try:
            self.cache[key] = value
            if expire_seconds:
                self.expiry[key] = datetime.utcnow() + timedelta(seconds=expire_seconds)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
            if key in self.expiry:
                del self.expiry[key]
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if key in self.cache:
                # Check expiry
                if key in self.expiry:
                    if datetime.utcnow() > self.expiry[key]:
                        # Expired
                        del self.cache[key]
                        del self.expiry[key]
                        return False
                return True
            return False
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def clear(self):
        """Clear all cache"""
        try:
            self.cache.clear()
            self.expiry.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    async def ping(self) -> bool:
        """Check if cache is available"""
        return True
    
    def info(self) -> dict:
        """Get cache info"""
        return {
            "type": "in_memory",
            "keys": len(self.cache),
            "available": True
        }


# Global cache instance (in-memory fallback)
_cache_client: Optional[InMemoryCache] = None


async def get_redis_client() -> InMemoryCache:
    """
    Get Redis client (or in-memory fallback)
    
    Returns:
        Cache client instance
    """
    global _cache_client
    
    if _cache_client is None:
        _cache_client = InMemoryCache()
        logger.info("Using in-memory cache (Redis not configured)")
    
    return _cache_client


async def get_cache() -> InMemoryCache:
    """Alias for get_redis_client"""
    return await get_redis_client()


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache"""
    client = await get_redis_client()
    return await client.get(key)


async def cache_set(key: str, value: Any, expire_seconds: Optional[int] = None):
    """Set value in cache"""
    client = await get_redis_client()
    await client.set(key, value, expire_seconds)


async def cache_delete(key: str):
    """Delete value from cache"""
    client = await get_redis_client()
    await client.delete(key)


async def cache_exists(key: str) -> bool:
    """Check if key exists in cache"""
    client = await get_redis_client()
    return await client.exists(key)


async def cache_clear():
    """Clear all cache"""
    client = await get_redis_client()
    await client.clear()

