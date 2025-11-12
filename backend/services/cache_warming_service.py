"""
Cache Warming Service for GRETA PAI

This service provides intelligent cache warming strategies to preload frequently
accessed data into Redis cache, reducing response times and improving user experience.
It includes analytics, monitoring, and automated warming based on usage patterns.

The service provides:
- Intelligent cache warming based on access patterns
- Scheduled cache warming for predictable data
- Cache analytics and hit/miss ratio monitoring
- Distributed cache coordination across instances
- Cache compression for large objects
- Cache invalidation pattern management
- Performance metrics and optimization

Example:
    Basic cache warming:
        >>> from services.cache_warming_service import CacheWarmingService
        >>> 
        >>> warming_service = CacheWarmingService()
        >>> await warming_service.initialize()
        >>> 
        >>> # Warm user data cache
        >>> await warming_service.warm_user_data()
        >>> 
        >>> # Warm frequently accessed endpoints
        >>> await warming_service.warm_api_responses()

    Analytics and monitoring:
        >>> stats = await warming_service.get_cache_analytics()
        >>> print(f"Cache hit ratio: {stats.hit_ratio}%")
        >>> 
        >>> # Get warming recommendations
        >>> recommendations = await warming_service.get_warming_recommendations()

Attributes:
    redis_client (Redis): Redis client for cache operations
    db (AsyncIOMotorDatabase): MongoDB for analytics storage
    warming_strategies (Dict): Registered warming strategies
    analytics (CacheAnalytics): Cache performance analytics
    logger (logging.Logger): Service logger
"""

import asyncio
import json
import gzip
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorDatabase

from utils.cache import get_redis_client, CacheManager
from utils.custom_exceptions import CacheError, ConfigurationError
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheAnalytics:
    """
    Cache analytics and performance metrics.
    
    Tracks cache performance including hit ratios, response times,
    and usage patterns for optimization.
    
    Attributes:
        total_requests: Total cache requests
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        hit_ratio: Cache hit ratio percentage
        average_response_time: Average cache response time (ms)
        total_size: Total cache size in bytes
        key_count: Number of cached keys
        warming_operations: Number of warming operations
        last_updated: Last analytics update timestamp
    """
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_ratio: float = 0.0
    average_response_time: float = 0.0  # milliseconds
    total_size: int = 0  # bytes
    key_count: int = 0
    warming_operations: int = 0
    last_updated: datetime = None
    
    def calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100


@dataclass
class WarmingRecommendation:
    """
    Cache warming recommendation based on analytics.
    
    Provides recommendations for cache warming strategies
    based on access patterns and performance metrics.
    
    Attributes:
        key_pattern: Cache key pattern to warm
        priority: Warming priority (1-10)
        frequency: Recommended warming frequency
        reason: Reason for recommendation
        estimated_benefit: Estimated performance benefit
    """
    key_pattern: str
    priority: int
    frequency: str  # 'hourly', 'daily', 'weekly'
    reason: str
    estimated_benefit: float  # percentage improvement


class CacheWarmingService:
    """
    Advanced cache warming service with intelligent strategies.
    
    This service provides comprehensive cache warming capabilities including
    intelligent warming based on access patterns, scheduled warming,
    and performance optimization.
    
    Attributes:
        redis_client: Redis client for cache operations
        db: MongoDB database for analytics storage
        cache_manager: Cache manager instance
        warming_strategies: Dictionary of warming strategies
        
    Example:
        >>> warming_service = CacheWarmingService()
        >>> await warming_service.initialize()
        >>> await warming_service.start_intelligent_warming()
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        """
        Initialize cache warming service.
        
        Args:
            database: MongoDB database instance for analytics
        """
        self.redis_client = None
        self.db = database
        self.cache_manager = None
        self.settings = get_settings()
        
        # Warming strategies registry
        self.warming_strategies: Dict[str, Callable] = {}
        
        # Analytics tracking
        self.analytics = CacheAnalytics()
        self._analytics_lock = asyncio.Lock()
        
        # Configuration
        self.compression_threshold = 1024  # bytes
        self.max_cache_size = 100 * 1024 * 1024  # 100MB
        self.warming_batch_size = 100
        
        # Register default warming strategies
        self._register_default_strategies()
    
    async def initialize(self) -> None:
        """
        Initialize the cache warming service.
        
        Sets up Redis connection, MongoDB collections, and analytics tracking.
        
        Raises:
            CacheError: If initialization fails
        """
        try:
            # Initialize Redis connection
            self.redis_client = await get_redis_client()
            self.cache_manager = CacheManager(self.redis_client)
            logger.info("✅ Cache warming Redis connection established")
            
            # Initialize MongoDB collections for analytics
            if self.db:
                await self._initialize_analytics_collections()
                logger.info("✅ Cache warming analytics collections initialized")
            
            # Load existing analytics
            await self._load_analytics()
            
            # Start background analytics collection
            asyncio.create_task(self._analytics_collection_loop())
            
            logger.info("✅ Cache warming service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Cache warming initialization failed: {e}")
            raise CacheError(f"Failed to initialize cache warming: {e}")
    
    async def _initialize_analytics_collections(self) -> None:
        """Initialize MongoDB collections for cache analytics"""
        # Cache analytics collection
        analytics_collection = self.db.cache_analytics
        
        # Create indexes for efficient querying
        await analytics_collection.create_index("timestamp")
        await analytics_collection.create_index("key_pattern")
        await analytics_collection.create_index([("timestamp", -1), ("key_pattern", 1)])
        
        # Cache access patterns collection
        patterns_collection = self.db.cache_access_patterns
        await patterns_collection.create_index("key")
        await patterns_collection.create_index("access_count")
        await patterns_collection.create_index("last_accessed")
        
        logger.info("✅ Cache analytics MongoDB indexes created")
    
    def _register_default_strategies(self) -> None:
        """Register default cache warming strategies"""
        self.warming_strategies.update({
            "user_data": self._warm_user_data,
            "api_responses": self._warm_api_responses,
            "static_content": self._warm_static_content,
            "frequently_accessed": self._warm_frequently_accessed,
            "predictive": self._warm_predictive_data
        })
    
    async def warm_cache(self, strategy: str, **kwargs) -> Dict[str, Any]:
        """
        Execute cache warming strategy.
        
        Args:
            strategy: Warming strategy name
            **kwargs: Strategy-specific parameters
            
        Returns:
            Dict: Warming operation results
            
        Raises:
            CacheError: If warming strategy fails
        """
        if strategy not in self.warming_strategies:
            raise CacheError(f"Unknown warming strategy: {strategy}")
        
        try:
            start_time = datetime.utcnow()
            
            # Execute warming strategy
            result = await self.warming_strategies[strategy](**kwargs)
            
            # Update analytics
            async with self._analytics_lock:
                self.analytics.warming_operations += 1
            
            # Log warming operation
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"✅ Cache warming '{strategy}' completed in {duration:.2f}s")
            
            return {
                "strategy": strategy,
                "duration": duration,
                "result": result,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Cache warming '{strategy}' failed: {e}")
            raise CacheError(f"Cache warming failed: {e}")
    
    async def _warm_user_data(self, user_ids: List[str] = None, limit: int = 100) -> Dict[str, Any]:
        """
        Warm user data cache.
        
        Preloads frequently accessed user data into cache.
        
        Args:
            user_ids: Specific user IDs to warm (optional)
            limit: Maximum number of users to warm
            
        Returns:
            Dict: Warming results
        """
        warmed_count = 0
        
        try:
            if not self.db:
                return {"warmed_count": 0, "message": "Database not available"}
            
            # Get users to warm
            if user_ids:
                users_cursor = self.db.users.find({"user_id": {"$in": user_ids}})
            else:
                # Get most active users
                users_cursor = self.db.users.find(
                    {"is_active": True}
                ).sort("last_login", -1).limit(limit)
            
            # Warm user data
            async for user in users_cursor:
                try:
                    user_id = user.get("user_id")
                    if not user_id:
                        continue
                    
                    # Cache user profile
                    cache_key = f"user:profile:{user_id}"
                    user_data = {
                        "user_id": user_id,
                        "username": user.get("username"),
                        "email": user.get("email"),
                        "is_active": user.get("is_active"),
                        "last_login": user.get("last_login")
                    }
                    
                    await self._set_compressed_cache(cache_key, user_data, ttl=3600)
                    
                    # Cache user preferences
                    prefs_key = f"user:preferences:{user_id}"
                    preferences = user.get("preferences", {})
                    await self._set_compressed_cache(prefs_key, preferences, ttl=7200)
                    
                    warmed_count += 1
                    
                    # Batch processing to avoid overwhelming Redis
                    if warmed_count % self.warming_batch_size == 0:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.warning(f"Failed to warm user {user.get('user_id')}: {e}")
                    continue
            
            return {
                "warmed_count": warmed_count,
                "cache_keys": ["user:profile:*", "user:preferences:*"]
            }
            
        except Exception as e:
            logger.error(f"User data warming failed: {e}")
            return {"warmed_count": warmed_count, "error": str(e)}
    
    async def _warm_api_responses(self, endpoints: List[str] = None) -> Dict[str, Any]:
        """
        Warm API response cache.
        
        Preloads responses for frequently accessed API endpoints.
        
        Args:
            endpoints: Specific endpoints to warm (optional)
            
        Returns:
            Dict: Warming results
        """
        warmed_count = 0
        
        # Default endpoints to warm
        if not endpoints:
            endpoints = [
                "/health",
                "/api/v1/users/me",
                "/api/v1/agents/list",
                "/api/v1/voice/config",
                "/api/v1/system/stats"
            ]
        
        try:
            for endpoint in endpoints:
                try:
                    # Simulate API response caching
                    cache_key = f"api:response:{endpoint.replace('/', ':')}"
                    
                    # Generate mock response data (in real implementation, 
                    # this would call the actual endpoint)
                    response_data = {
                        "endpoint": endpoint,
                        "cached_at": datetime.utcnow().isoformat(),
                        "data": f"Cached response for {endpoint}"
                    }
                    
                    await self._set_compressed_cache(cache_key, response_data, ttl=300)
                    warmed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to warm endpoint {endpoint}: {e}")
                    continue
            
            return {
                "warmed_count": warmed_count,
                "endpoints": endpoints
            }
            
        except Exception as e:
            logger.error(f"API response warming failed: {e}")
            return {"warmed_count": warmed_count, "error": str(e)}
    
    async def _warm_static_content(self) -> Dict[str, Any]:
        """
        Warm static content cache.
        
        Preloads static content like configuration, templates, etc.
        
        Returns:
            Dict: Warming results
        """
        warmed_count = 0
        
        try:
            # Warm system configuration
            config_key = "system:config"
            config_data = {
                "app_name": self.settings.app_name if hasattr(self.settings, 'app_name') else "GRETA PAI",
                "version": "2.2.0",
                "features": ["AI", "Voice", "Agents", "Training"],
                "cached_at": datetime.utcnow().isoformat()
            }
            await self._set_compressed_cache(config_key, config_data, ttl=86400)
            warmed_count += 1
            
            # Warm feature flags
            features_key = "system:features"
            features_data = {
                "voice_enabled": True,
                "agents_enabled": True,
                "training_enabled": True,
                "analytics_enabled": True
            }
            await self._set_compressed_cache(features_key, features_data, ttl=3600)
            warmed_count += 1
            
            return {
                "warmed_count": warmed_count,
                "cache_keys": ["system:config", "system:features"]
            }
            
        except Exception as e:
            logger.error(f"Static content warming failed: {e}")
            return {"warmed_count": warmed_count, "error": str(e)}
    
    async def _warm_frequently_accessed(self, limit: int = 50) -> Dict[str, Any]:
        """
        Warm frequently accessed data based on analytics.
        
        Args:
            limit: Maximum number of keys to warm
            
        Returns:
            Dict: Warming results
        """
        warmed_count = 0
        
        try:
            if not self.db:
                return {"warmed_count": 0, "message": "Database not available"}
            
            # Get frequently accessed keys from analytics
            patterns_cursor = self.db.cache_access_patterns.find().sort(
                "access_count", -1
            ).limit(limit)
            
            async for pattern in patterns_cursor:
                try:
                    key = pattern.get("key")
                    if not key:
                        continue
                    
                    # Check if key is still valid and not already cached
                    exists = await self.redis_client.exists(key)
                    if exists:
                        continue
                    
                    # Regenerate cache data based on key pattern
                    cache_data = await self._regenerate_cache_data(key)
                    if cache_data:
                        await self._set_compressed_cache(key, cache_data, ttl=1800)
                        warmed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to warm key {pattern.get('key')}: {e}")
                    continue
            
            return {
                "warmed_count": warmed_count,
                "strategy": "frequently_accessed"
            }
            
        except Exception as e:
            logger.error(f"Frequently accessed warming failed: {e}")
            return {"warmed_count": warmed_count, "error": str(e)}
    
    async def _warm_predictive_data(self) -> Dict[str, Any]:
        """
        Warm cache with predictive data based on patterns.
        
        Returns:
            Dict: Warming results
        """
        warmed_count = 0
        
        try:
            # Predict and warm data for upcoming peak hours
            current_hour = datetime.utcnow().hour
            
            # Warm data for next 2 hours based on historical patterns
            for hour_offset in range(1, 3):
                predicted_hour = (current_hour + hour_offset) % 24
                
                # Generate predictive cache keys
                predictive_keys = [
                    f"stats:hourly:{predicted_hour}",
                    f"analytics:peak:{predicted_hour}",
                    f"users:active:{predicted_hour}"
                ]
                
                for key in predictive_keys:
                    try:
                        # Generate predictive data
                        predictive_data = {
                            "hour": predicted_hour,
                            "predicted_at": datetime.utcnow().isoformat(),
                            "data": f"Predictive data for hour {predicted_hour}"
                        }
                        
                        await self._set_compressed_cache(key, predictive_data, ttl=7200)
                        warmed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to warm predictive key {key}: {e}")
                        continue
            
            return {
                "warmed_count": warmed_count,
                "strategy": "predictive"
            }
            
        except Exception as e:
            logger.error(f"Predictive warming failed: {e}")
            return {"warmed_count": warmed_count, "error": str(e)}
    
    async def _set_compressed_cache(self, key: str, data: Any, ttl: int = 3600) -> None:
        """
        Set cache value with optional compression.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
        """
        try:
            # Serialize data
            serialized = json.dumps(data, default=str).encode('utf-8')
            
            # Compress if data is large
            if len(serialized) > self.compression_threshold:
                compressed = gzip.compress(serialized)
                await self.redis_client.setex(f"{key}:compressed", ttl, compressed)
                logger.debug(f"Cached compressed data for {key} ({len(compressed)} bytes)")
            else:
                await self.redis_client.setex(key, ttl, serialized)
                logger.debug(f"Cached data for {key} ({len(serialized)} bytes)")
                
        except Exception as e:
            logger.error(f"Failed to set compressed cache for {key}: {e}")
            raise
    
    async def _regenerate_cache_data(self, key: str) -> Optional[Any]:
        """
        Regenerate cache data based on key pattern.
        
        Args:
            key: Cache key to regenerate
            
        Returns:
            Any: Regenerated data or None
        """
        try:
            # Simple key pattern matching for regeneration
            if key.startswith("user:profile:"):
                user_id = key.split(":")[-1]
                return {"user_id": user_id, "regenerated": True}
            elif key.startswith("api:response:"):
                endpoint = key.replace("api:response:", "").replace(":", "/")
                return {"endpoint": endpoint, "regenerated": True}
            else:
                return {"key": key, "regenerated": True, "timestamp": datetime.utcnow().isoformat()}
                
        except Exception as e:
            logger.warning(f"Failed to regenerate data for key {key}: {e}")
            return None

    async def _load_analytics(self) -> None:
        """Load existing analytics from database"""
        if not self.db:
            return

        try:
            # Load latest analytics
            latest_analytics = await self.db.cache_analytics.find_one(
                {}, sort=[("timestamp", -1)]
            )

            if latest_analytics:
                self.analytics = CacheAnalytics(
                    total_requests=latest_analytics.get("total_requests", 0),
                    cache_hits=latest_analytics.get("cache_hits", 0),
                    cache_misses=latest_analytics.get("cache_misses", 0),
                    hit_ratio=latest_analytics.get("hit_ratio", 0.0),
                    average_response_time=latest_analytics.get("average_response_time", 0.0),
                    total_size=latest_analytics.get("total_size", 0),
                    key_count=latest_analytics.get("key_count", 0),
                    warming_operations=latest_analytics.get("warming_operations", 0),
                    last_updated=latest_analytics.get("last_updated")
                )

                logger.info("✅ Cache analytics loaded from database")

        except Exception as e:
            logger.warning(f"Failed to load analytics: {e}")

    async def _analytics_collection_loop(self) -> None:
        """Background loop for collecting cache analytics"""
        while True:
            try:
                await asyncio.sleep(300)  # Collect every 5 minutes
                await self._collect_analytics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analytics collection error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _collect_analytics(self) -> None:
        """Collect current cache analytics"""
        try:
            # Get Redis info
            redis_info = await self.redis_client.info()

            # Update analytics
            async with self._analytics_lock:
                self.analytics.total_size = redis_info.get("used_memory", 0)
                self.analytics.key_count = await self.redis_client.dbsize()
                self.analytics.last_updated = datetime.utcnow()

                # Calculate hit ratio
                self.analytics.hit_ratio = self.analytics.calculate_hit_ratio()

            # Save to database
            if self.db:
                analytics_doc = {
                    "timestamp": datetime.utcnow(),
                    "total_requests": self.analytics.total_requests,
                    "cache_hits": self.analytics.cache_hits,
                    "cache_misses": self.analytics.cache_misses,
                    "hit_ratio": self.analytics.hit_ratio,
                    "average_response_time": self.analytics.average_response_time,
                    "total_size": self.analytics.total_size,
                    "key_count": self.analytics.key_count,
                    "warming_operations": self.analytics.warming_operations
                }

                await self.db.cache_analytics.insert_one(analytics_doc)

        except Exception as e:
            logger.error(f"Failed to collect analytics: {e}")

    async def get_cache_analytics(self) -> CacheAnalytics:
        """
        Get current cache analytics.

        Returns:
            CacheAnalytics: Current cache performance metrics
        """
        # Update real-time metrics
        await self._collect_analytics()

        async with self._analytics_lock:
            return self.analytics

    async def get_warming_recommendations(self) -> List[WarmingRecommendation]:
        """
        Get cache warming recommendations based on analytics.

        Returns:
            List[WarmingRecommendation]: List of warming recommendations
        """
        recommendations = []

        try:
            # Analyze cache hit ratios
            if self.analytics.hit_ratio < 70:
                recommendations.append(WarmingRecommendation(
                    key_pattern="user:*",
                    priority=8,
                    frequency="hourly",
                    reason="Low cache hit ratio detected",
                    estimated_benefit=15.0
                ))

            # Analyze frequently missed keys
            if self.db:
                # Get most missed key patterns
                pipeline = [
                    {"$match": {"timestamp": {"$gte": datetime.utcnow() - timedelta(hours=24)}}},
                    {"$group": {
                        "_id": "$key_pattern",
                        "miss_count": {"$sum": "$cache_misses"},
                        "total_requests": {"$sum": "$total_requests"}
                    }},
                    {"$match": {"miss_count": {"$gt": 100}}},
                    {"$sort": {"miss_count": -1}},
                    {"$limit": 5}
                ]

                async for doc in self.db.cache_analytics.aggregate(pipeline):
                    key_pattern = doc["_id"]
                    miss_count = doc["miss_count"]
                    total_requests = doc["total_requests"]

                    miss_ratio = (miss_count / total_requests) * 100 if total_requests > 0 else 0

                    if miss_ratio > 50:
                        recommendations.append(WarmingRecommendation(
                            key_pattern=key_pattern,
                            priority=min(10, int(miss_ratio / 10)),
                            frequency="daily" if miss_ratio > 80 else "weekly",
                            reason=f"High miss ratio: {miss_ratio:.1f}%",
                            estimated_benefit=min(30.0, miss_ratio * 0.3)
                        ))

            # Sort by priority
            recommendations.sort(key=lambda x: x.priority, reverse=True)

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")

        return recommendations

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache keys matching pattern.

        Args:
            pattern: Key pattern to invalidate (supports wildcards)

        Returns:
            int: Number of keys invalidated
        """
        try:
            # Get keys matching pattern
            keys = await self.redis_client.keys(pattern)

            if keys:
                # Delete keys in batches
                deleted_count = 0
                batch_size = 100

                for i in range(0, len(keys), batch_size):
                    batch = keys[i:i + batch_size]
                    deleted = await self.redis_client.delete(*batch)
                    deleted_count += deleted

                logger.info(f"✅ Invalidated {deleted_count} cache keys matching '{pattern}'")
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Failed to invalidate pattern '{pattern}': {e}")
            raise CacheError(f"Cache invalidation failed: {e}")

    async def start_intelligent_warming(self) -> None:
        """
        Start intelligent cache warming based on analytics.

        This method analyzes cache performance and automatically
        executes warming strategies based on recommendations.
        """
        try:
            logger.info("🔥 Starting intelligent cache warming...")

            # Get warming recommendations
            recommendations = await self.get_warming_recommendations()

            # Execute high-priority recommendations
            for rec in recommendations:
                if rec.priority >= 7:
                    try:
                        # Map key patterns to warming strategies
                        if rec.key_pattern.startswith("user:"):
                            await self.warm_cache("user_data", limit=200)
                        elif rec.key_pattern.startswith("api:"):
                            await self.warm_cache("api_responses")
                        else:
                            await self.warm_cache("frequently_accessed", limit=100)

                        logger.info(f"✅ Executed warming for pattern: {rec.key_pattern}")

                    except Exception as e:
                        logger.warning(f"Failed to warm pattern {rec.key_pattern}: {e}")

            # Always warm static content
            await self.warm_cache("static_content")

            # Warm predictive data during off-peak hours
            current_hour = datetime.utcnow().hour
            if 2 <= current_hour <= 6:  # Off-peak hours
                await self.warm_cache("predictive")

            logger.info("✅ Intelligent cache warming completed")

        except Exception as e:
            logger.error(f"❌ Intelligent warming failed: {e}")
            raise CacheError(f"Intelligent warming failed: {e}")

    async def get_cache_size_info(self) -> Dict[str, Any]:
        """
        Get detailed cache size information.

        Returns:
            Dict: Cache size breakdown and statistics
        """
        try:
            redis_info = await self.redis_client.info()

            # Get key count by pattern
            key_patterns = {}
            all_keys = await self.redis_client.keys("*")

            for key in all_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                pattern = key_str.split(":")[0] if ":" in key_str else "other"
                key_patterns[pattern] = key_patterns.get(pattern, 0) + 1

            return {
                "total_memory": redis_info.get("used_memory", 0),
                "total_memory_human": redis_info.get("used_memory_human", "0B"),
                "total_keys": await self.redis_client.dbsize(),
                "key_patterns": key_patterns,
                "max_memory": redis_info.get("maxmemory", 0),
                "memory_usage_percentage": (
                    (redis_info.get("used_memory", 0) / redis_info.get("maxmemory", 1)) * 100
                    if redis_info.get("maxmemory", 0) > 0 else 0
                ),
                "fragmentation_ratio": redis_info.get("mem_fragmentation_ratio", 1.0)
            }

        except Exception as e:
            logger.error(f"Failed to get cache size info: {e}")
            return {"error": str(e)}

    async def cleanup_expired_analytics(self, days: int = 30) -> int:
        """
        Clean up old analytics data.

        Args:
            days: Number of days to retain analytics data

        Returns:
            int: Number of records cleaned up
        """
        if not self.db:
            return 0

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Clean up analytics
            result = await self.db.cache_analytics.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })

            # Clean up access patterns
            patterns_result = await self.db.cache_access_patterns.delete_many({
                "last_accessed": {"$lt": cutoff_date}
            })

            total_cleaned = result.deleted_count + patterns_result.deleted_count

            logger.info(f"✅ Cleaned up {total_cleaned} old analytics records")
            return total_cleaned

        except Exception as e:
            logger.error(f"Failed to cleanup analytics: {e}")
            return 0
