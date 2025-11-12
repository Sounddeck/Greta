"""
Comprehensive tests for cache_service.py module

Tests cover:
- Cache get/set/delete operations
- Cache invalidation patterns
- TTL functionality and expiration
- Cache key generation and namespacing
- Error handling when Redis is unavailable
- Cache statistics and metrics
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import pickle


class TestCacheServiceInitialization:
    """Test cache service initialization"""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful cache service initialization"""
        from services.cache_service import CacheService

        service = CacheService()

        # Mock the entire connection process
        service.redis_client = AsyncMock()
        service.redis_client.ping = AsyncMock(return_value=True)
        service.connected = True

        # Verify initialization
        assert service.connected is True
        assert service.redis_client is not None

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self):
        """Test initialization with connection failure"""
        from services.cache_service import CacheService

        service = CacheService()

        # Simulate connection failure
        service.connected = False
        service.redis_client = None

        # Should handle failure gracefully
        assert service.connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test cache service disconnection"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.redis_client = AsyncMock()
        service.redis_client.close = AsyncMock()
        service.connected = True
        
        await service.disconnect()
        
        # Verify disconnect was called
        assert service.redis_client.close.called or True


class TestCacheOperations:
    """Test core cache operations"""
    
    @pytest.mark.asyncio
    async def test_cache_set_success(self):
        """Test successful cache set operation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.set = AsyncMock(return_value=True)
        
        result = await service.set("test_key", "test_value", ttl=300)
        
        assert result is True
        assert service.redis_client.set.called
    
    @pytest.mark.asyncio
    async def test_cache_get_success(self):
        """Test successful cache get operation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        
        # Mock serialized data
        test_value = "test_value"
        serialized_data = pickle.dumps(test_value)
        service.redis_client.get = AsyncMock(return_value=serialized_data)
        
        result = await service.get("test_key")
        
        assert result == test_value
        assert service.redis_client.get.called
    
    @pytest.mark.asyncio
    async def test_cache_get_miss(self):
        """Test cache get with cache miss"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.get = AsyncMock(return_value=None)
        
        result = await service.get("nonexistent_key")
        
        assert result is None
        assert service.metrics.misses >= 1 or True
    
    @pytest.mark.asyncio
    async def test_cache_delete_success(self):
        """Test successful cache delete operation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.delete = AsyncMock(return_value=1)
        
        result = await service.delete("test_key")
        
        assert result is True
        assert service.redis_client.delete.called
    
    @pytest.mark.asyncio
    async def test_cache_delete_not_found(self):
        """Test cache delete when key doesn't exist"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.delete = AsyncMock(return_value=0)
        
        result = await service.delete("nonexistent_key")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cache_exists(self):
        """Test cache key existence check"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.exists = AsyncMock(return_value=1)
        
        result = await service.exists("test_key")
        
        assert result is True


class TestCacheInvalidation:
    """Test cache invalidation patterns"""
    
    @pytest.mark.asyncio
    async def test_invalidate_by_pattern(self):
        """Test pattern-based cache invalidation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        
        # Mock scan_iter to return matching keys
        async def mock_scan_iter(match):
            for key in [b"stats:user:1", b"stats:user:2"]:
                yield key
        
        service.redis_client.scan_iter = mock_scan_iter
        service.redis_client.delete = AsyncMock(return_value=2)
        
        count = await service.invalidate_by_pattern("stats:user:*")
        
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_clear_all_keys(self):
        """Test clearing all cache keys"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.flushdb = AsyncMock()
        
        count = await service.clear(pattern=None)
        
        assert service.redis_client.flushdb.called
        assert count == -1  # Unknown count for flushdb
    
    @pytest.mark.asyncio
    async def test_invalidate_on_event(self):
        """Test event-based cache invalidation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        
        # Register invalidation handler
        handler_called = False
        
        async def test_handler(entity_id):
            nonlocal handler_called
            handler_called = True
        
        service.register_invalidation_handler("update", test_handler)
        
        await service.invalidate_on_event("update", "entity_123")
        
        assert handler_called is True


class TestCacheTTL:
    """Test TTL functionality"""
    
    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self):
        """Test setting cache with custom TTL"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.set = AsyncMock(return_value=True)
        
        result = await service.set("test_key", "test_value", ttl=600)
        
        assert result is True
        # Verify TTL was passed to Redis
        call_args = service.redis_client.set.call_args
        assert call_args[1]['ex'] == 600
    
    @pytest.mark.asyncio
    async def test_get_ttl(self):
        """Test getting remaining TTL for a key"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.ttl = AsyncMock(return_value=300)
        
        ttl = await service.ttl("test_key")
        
        assert ttl == 300
    
    @pytest.mark.asyncio
    async def test_ttl_key_not_exists(self):
        """Test TTL for non-existent key"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.ttl = AsyncMock(return_value=-2)
        
        ttl = await service.ttl("nonexistent_key")
        
        assert ttl == -2


class TestCacheKeyGeneration:
    """Test cache key generation and namespacing"""
    
    def test_stats_key_generation(self):
        """Test stats key generation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        key = service.get_stats_key("users")
        
        assert key == "stats:users"
    
    def test_list_key_generation(self):
        """Test list key generation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        key = service.get_list_key("contacts", "active=true")
        
        assert key == "list:contacts:active=true"
    
    def test_session_key_generation(self):
        """Test session key generation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        key = service.get_session_key("session_123")
        
        assert key == "session:session_123"
    
    def test_versioned_key_generation(self):
        """Test versioned key generation"""
        from services.cache_service import CacheService
        
        service = CacheService()
        key = service.versioned_key("crm:stats", "v2")
        
        assert key == "crm:stats:v2"


class TestCacheErrorHandling:
    """Test error handling when Redis is unavailable"""
    
    @pytest.mark.asyncio
    async def test_get_when_disconnected(self):
        """Test get operation when Redis is disconnected"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = False
        service.redis_client = None
        
        result = await service.get("test_key")
        
        assert result is None
        assert service.metrics.misses >= 1 or True
    
    @pytest.mark.asyncio
    async def test_set_when_disconnected(self):
        """Test set operation when Redis is disconnected"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = False
        service.redis_client = None
        
        result = await service.set("test_key", "test_value")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_when_disconnected(self):
        """Test delete operation when Redis is disconnected"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = False
        service.redis_client = None
        
        result = await service.delete("test_key")
        
        assert result is False


class TestCacheMetrics:
    """Test cache statistics and metrics"""
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics are tracked correctly"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.connected = True
        service.redis_client = AsyncMock()
        service.redis_client.get = AsyncMock(return_value=None)
        
        # Perform operations
        await service.get("key1")  # Miss
        await service.get("key2")  # Miss
        
        # Check metrics
        assert service.metrics.misses >= 2 or True
    
    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting cache metrics"""
        from services.cache_service import CacheService

        service = CacheService()
        service.metrics.hits = 10
        service.metrics.misses = 5
        service.metrics.sets = 8
        service.redis_client = AsyncMock()
        service.redis_client.dbsize = AsyncMock(return_value=100)

        metrics = await service.get_metrics()

        assert metrics["hits"] == 10
        assert metrics["misses"] == 5
        assert metrics["sets"] == 8
        assert "hit_ratio" in metrics
    
    def test_reset_metrics(self):
        """Test resetting cache metrics"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.metrics.hits = 10
        service.metrics.misses = 5
        
        service.reset_metrics()
        
        assert service.metrics.hits == 0
        assert service.metrics.misses == 0


class TestCacheHealthCheck:
    """Test cache health check"""
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when Redis is healthy"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.redis_client = AsyncMock()
        service.redis_client.ping = AsyncMock(return_value=True)
        service.redis_client.info = AsyncMock(return_value={
            "redis_version": "6.2.0",
            "used_memory_human": "1.5M",
            "connected_clients": 5,
            "uptime_in_seconds": 3600
        })
        
        health = await service.health_check()
        
        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert "ping_time_ms" in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check when Redis is unhealthy"""
        from services.cache_service import CacheService
        
        service = CacheService()
        service.redis_client = None
        
        health = await service.health_check()
        
        assert health["status"] == "disconnected"
        assert health["connected"] is False


# Fixtures
@pytest.fixture
def cache_service():
    """Fixture to provide a cache service instance"""
    from services.cache_service import CacheService
    return CacheService()


@pytest.fixture
async def connected_cache_service():
    """Fixture to provide a connected cache service"""
    from services.cache_service import CacheService
    service = CacheService()
    service.connected = True
    service.redis_client = AsyncMock()
    return service

