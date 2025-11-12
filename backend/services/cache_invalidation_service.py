"""
Cache Invalidation Service for GRETA PAI

This service provides intelligent cache invalidation patterns and strategies
to ensure data consistency while maintaining optimal cache performance.
It includes event-driven invalidation, pattern-based invalidation, and
distributed cache coordination.

The service provides:
- Event-driven cache invalidation
- Pattern-based invalidation strategies
- Distributed cache coordination across instances
- Cache dependency tracking and management
- Intelligent invalidation timing
- Cache warming after invalidation
- Performance impact monitoring

Example:
    Basic cache invalidation:
        >>> from services.cache_invalidation_service import CacheInvalidationService
        >>> 
        >>> invalidation_service = CacheInvalidationService()
        >>> await invalidation_service.initialize()
        >>> 
        >>> # Invalidate user-related caches
        >>> await invalidation_service.invalidate_user_cache("user123")
        >>> 
        >>> # Invalidate by pattern
        >>> await invalidation_service.invalidate_pattern("api:response:*")

    Event-driven invalidation:
        >>> # Register invalidation rules
        >>> await invalidation_service.register_invalidation_rule(
        ...     event="user_updated",
        ...     patterns=["user:profile:{user_id}", "user:preferences:{user_id}"]
        ... )
        >>> 
        >>> # Trigger invalidation
        >>> await invalidation_service.handle_event("user_updated", {"user_id": "123"})

Attributes:
    redis_client (Redis): Redis client for cache operations
    invalidation_rules (Dict): Event-based invalidation rules
    dependency_graph (Dict): Cache dependency tracking
    logger (logging.Logger): Service logger
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

from utils.cache import get_redis_client
from utils.custom_exceptions import CacheError, InvalidationError
from config import get_settings

logger = logging.getLogger(__name__)


class InvalidationStrategy(str, Enum):
    """Cache invalidation strategy enumeration"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    LAZY = "lazy"
    BATCH = "batch"
    SMART = "smart"


@dataclass
class InvalidationRule:
    """
    Cache invalidation rule configuration.
    
    Defines how cache should be invalidated when specific events occur.
    
    Attributes:
        event: Event name that triggers invalidation
        patterns: List of cache key patterns to invalidate
        strategy: Invalidation strategy to use
        delay: Delay before invalidation (for delayed strategy)
        dependencies: Cache keys that depend on this rule
        warm_after: Whether to warm cache after invalidation
        priority: Rule priority (higher = more important)
    """
    event: str
    patterns: List[str]
    strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE
    delay: int = 0  # seconds
    dependencies: List[str] = field(default_factory=list)
    warm_after: bool = False
    priority: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InvalidationResult:
    """
    Cache invalidation operation result.
    
    Contains information about the invalidation operation including
    performance metrics and affected keys.
    
    Attributes:
        success: Whether invalidation was successful
        keys_invalidated: Number of keys invalidated
        patterns_processed: List of patterns processed
        duration: Operation duration in seconds
        strategy_used: Invalidation strategy used
        error: Error message if failed
        warm_operations: Number of warming operations triggered
    """
    success: bool
    keys_invalidated: int = 0
    patterns_processed: List[str] = field(default_factory=list)
    duration: float = 0.0
    strategy_used: InvalidationStrategy = InvalidationStrategy.IMMEDIATE
    error: Optional[str] = None
    warm_operations: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CacheInvalidationService:
    """
    Intelligent cache invalidation service with pattern-based strategies.
    
    This service manages cache invalidation across the application with
    support for event-driven invalidation, dependency tracking, and
    intelligent invalidation strategies.
    
    Attributes:
        redis_client: Redis client for cache operations
        invalidation_rules: Dictionary of registered invalidation rules
        dependency_graph: Cache dependency tracking
        pending_invalidations: Queue of pending invalidations
        
    Example:
        >>> invalidation_service = CacheInvalidationService()
        >>> await invalidation_service.initialize()
        >>> await invalidation_service.invalidate_user_cache("user123")
    """
    
    def __init__(self):
        """Initialize cache invalidation service"""
        self.redis_client = None
        self.settings = get_settings()
        
        # Invalidation rules and dependencies
        self.invalidation_rules: Dict[str, List[InvalidationRule]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.pending_invalidations: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.invalidation_stats = {
            "total_invalidations": 0,
            "successful_invalidations": 0,
            "failed_invalidations": 0,
            "keys_invalidated": 0,
            "average_duration": 0.0
        }
        
        # Configuration
        self.batch_size = 100
        self.max_delay = 300  # 5 minutes
        self.enable_warming = True
        
        # Background tasks
        self._processor_task = None
        self._running = False
    
    async def initialize(self) -> None:
        """
        Initialize the cache invalidation service.
        
        Sets up Redis connection and starts background processing.
        
        Raises:
            CacheError: If initialization fails
        """
        try:
            # Initialize Redis connection
            self.redis_client = await get_redis_client()
            logger.info("✅ Cache invalidation Redis connection established")
            
            # Register default invalidation rules
            await self._register_default_rules()
            
            # Start background processor
            await self._start_background_processor()
            
            self._running = True
            logger.info("✅ Cache invalidation service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Cache invalidation initialization failed: {e}")
            raise CacheError(f"Failed to initialize cache invalidation: {e}")
    
    async def _register_default_rules(self) -> None:
        """Register default invalidation rules"""
        default_rules = [
            InvalidationRule(
                event="user_updated",
                patterns=["user:profile:{user_id}", "user:preferences:{user_id}", "user:cache:{user_id}"],
                strategy=InvalidationStrategy.IMMEDIATE,
                warm_after=True,
                priority=8
            ),
            InvalidationRule(
                event="user_deleted",
                patterns=["user:*:{user_id}"],
                strategy=InvalidationStrategy.IMMEDIATE,
                priority=10
            ),
            InvalidationRule(
                event="agent_updated",
                patterns=["agent:config:{agent_id}", "agent:cache:{agent_id}"],
                strategy=InvalidationStrategy.DELAYED,
                delay=30,
                warm_after=True,
                priority=7
            ),
            InvalidationRule(
                event="system_config_updated",
                patterns=["system:*", "config:*", "api:response:*"],
                strategy=InvalidationStrategy.BATCH,
                warm_after=True,
                priority=9
            ),
            InvalidationRule(
                event="voice_config_updated",
                patterns=["voice:config:*", "voice:cache:*"],
                strategy=InvalidationStrategy.IMMEDIATE,
                warm_after=True,
                priority=6
            )
        ]
        
        for rule in default_rules:
            await self.register_invalidation_rule(rule)
        
        logger.info(f"✅ Registered {len(default_rules)} default invalidation rules")
    
    async def _start_background_processor(self) -> None:
        """Start background invalidation processor"""
        self._processor_task = asyncio.create_task(self._invalidation_processor())
        logger.info("✅ Background invalidation processor started")
    
    async def _invalidation_processor(self) -> None:
        """Background processor for handling invalidation queue"""
        while self._running:
            try:
                # Process pending invalidations
                try:
                    invalidation_data = await asyncio.wait_for(
                        self.pending_invalidations.get(), timeout=1.0
                    )
                    await self._process_invalidation(invalidation_data)
                except asyncio.TimeoutError:
                    continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Invalidation processor error: {e}")
                await asyncio.sleep(1)
    
    async def register_invalidation_rule(self, rule: InvalidationRule) -> None:
        """
        Register a cache invalidation rule.
        
        Args:
            rule: Invalidation rule to register
        """
        if rule.event not in self.invalidation_rules:
            self.invalidation_rules[rule.event] = []
        
        self.invalidation_rules[rule.event].append(rule)
        
        # Sort by priority (higher priority first)
        self.invalidation_rules[rule.event].sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"✅ Registered invalidation rule for event: {rule.event}")
    
    async def handle_event(self, event: str, context: Dict[str, Any] = None) -> List[InvalidationResult]:
        """
        Handle an event that may trigger cache invalidation.
        
        Args:
            event: Event name
            context: Event context data for pattern substitution
            
        Returns:
            List[InvalidationResult]: Results of invalidation operations
        """
        results = []
        context = context or {}
        
        if event not in self.invalidation_rules:
            logger.debug(f"No invalidation rules for event: {event}")
            return results
        
        try:
            for rule in self.invalidation_rules[event]:
                try:
                    # Substitute context variables in patterns
                    resolved_patterns = []
                    for pattern in rule.patterns:
                        resolved_pattern = pattern.format(**context)
                        resolved_patterns.append(resolved_pattern)
                    
                    # Create invalidation operation
                    invalidation_data = {
                        "patterns": resolved_patterns,
                        "strategy": rule.strategy,
                        "delay": rule.delay,
                        "warm_after": rule.warm_after,
                        "event": event,
                        "context": context,
                        "rule_priority": rule.priority
                    }
                    
                    # Queue or execute immediately based on strategy
                    if rule.strategy == InvalidationStrategy.IMMEDIATE:
                        result = await self._execute_invalidation(invalidation_data)
                        results.append(result)
                    else:
                        await self.pending_invalidations.put(invalidation_data)
                        logger.debug(f"Queued invalidation for event: {event}")
                
                except Exception as e:
                    logger.error(f"Failed to process rule for event {event}: {e}")
                    results.append(InvalidationResult(
                        success=False,
                        error=str(e),
                        strategy_used=rule.strategy
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to handle event {event}: {e}")
            return [InvalidationResult(success=False, error=str(e))]
    
    async def _process_invalidation(self, invalidation_data: Dict[str, Any]) -> None:
        """Process a queued invalidation operation"""
        try:
            strategy = InvalidationStrategy(invalidation_data.get("strategy", "immediate"))
            delay = invalidation_data.get("delay", 0)
            
            # Apply delay if specified
            if delay > 0 and strategy == InvalidationStrategy.DELAYED:
                await asyncio.sleep(min(delay, self.max_delay))
            
            # Execute invalidation
            await self._execute_invalidation(invalidation_data)
            
        except Exception as e:
            logger.error(f"Failed to process invalidation: {e}")
    
    async def _execute_invalidation(self, invalidation_data: Dict[str, Any]) -> InvalidationResult:
        """Execute cache invalidation operation"""
        start_time = datetime.utcnow()
        patterns = invalidation_data.get("patterns", [])
        strategy = InvalidationStrategy(invalidation_data.get("strategy", "immediate"))
        warm_after = invalidation_data.get("warm_after", False)
        
        result = InvalidationResult(
            success=True,
            patterns_processed=patterns,
            strategy_used=strategy
        )
        
        try:
            total_invalidated = 0
            
            if strategy == InvalidationStrategy.BATCH:
                # Batch invalidation
                total_invalidated = await self._batch_invalidate(patterns)
            else:
                # Individual pattern invalidation
                for pattern in patterns:
                    invalidated = await self._invalidate_pattern(pattern)
                    total_invalidated += invalidated
            
            result.keys_invalidated = total_invalidated
            
            # Trigger cache warming if requested
            if warm_after and self.enable_warming:
                warm_count = await self._trigger_warming(patterns)
                result.warm_operations = warm_count
            
            # Update statistics
            self.invalidation_stats["total_invalidations"] += 1
            self.invalidation_stats["successful_invalidations"] += 1
            self.invalidation_stats["keys_invalidated"] += total_invalidated
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration = duration
            
            # Update average duration
            total_ops = self.invalidation_stats["total_invalidations"]
            current_avg = self.invalidation_stats["average_duration"]
            self.invalidation_stats["average_duration"] = (
                (current_avg * (total_ops - 1) + duration) / total_ops
            )
            
            logger.info(f"✅ Invalidated {total_invalidated} keys using {strategy.value} strategy")
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            self.invalidation_stats["failed_invalidations"] += 1
            logger.error(f"❌ Invalidation failed: {e}")
        
        return result
    
    async def _invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern"""
        try:
            # Get keys matching pattern
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Delete keys in batches
                deleted_count = 0
                for i in range(0, len(keys), self.batch_size):
                    batch = keys[i:i + self.batch_size]
                    deleted = await self.redis_client.delete(*batch)
                    deleted_count += deleted
                
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to invalidate pattern '{pattern}': {e}")
            raise InvalidationError(f"Pattern invalidation failed: {e}")
    
    async def _batch_invalidate(self, patterns: List[str]) -> int:
        """Batch invalidate multiple patterns efficiently"""
        try:
            all_keys = set()
            
            # Collect all keys from all patterns
            for pattern in patterns:
                keys = await self.redis_client.keys(pattern)
                all_keys.update(keys)
            
            if all_keys:
                # Delete all keys in batches
                keys_list = list(all_keys)
                deleted_count = 0
                
                for i in range(0, len(keys_list), self.batch_size):
                    batch = keys_list[i:i + self.batch_size]
                    deleted = await self.redis_client.delete(*batch)
                    deleted_count += deleted
                
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Batch invalidation failed: {e}")
            raise InvalidationError(f"Batch invalidation failed: {e}")
    
    async def _trigger_warming(self, patterns: List[str]) -> int:
        """Trigger cache warming for invalidated patterns"""
        try:
            warm_count = 0
            
            # Import here to avoid circular imports
            from services.cache_warming_service import CacheWarmingService
            
            # Create warming service instance
            warming_service = CacheWarmingService()
            await warming_service.initialize()
            
            # Determine warming strategies based on patterns
            for pattern in patterns:
                try:
                    if pattern.startswith("user:"):
                        await warming_service.warm_cache("user_data", limit=50)
                        warm_count += 1
                    elif pattern.startswith("api:"):
                        await warming_service.warm_cache("api_responses")
                        warm_count += 1
                    elif pattern.startswith("system:") or pattern.startswith("config:"):
                        await warming_service.warm_cache("static_content")
                        warm_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to warm pattern {pattern}: {e}")
            
            return warm_count
            
        except Exception as e:
            logger.warning(f"Cache warming after invalidation failed: {e}")
            return 0
    
    async def invalidate_user_cache(self, user_id: str) -> InvalidationResult:
        """
        Invalidate all cache entries for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            InvalidationResult: Invalidation operation result
        """
        return await self.handle_event("user_updated", {"user_id": user_id})
    
    async def invalidate_pattern(self, pattern: str, warm_after: bool = False) -> InvalidationResult:
        """
        Invalidate cache keys matching a specific pattern.
        
        Args:
            pattern: Cache key pattern to invalidate
            warm_after: Whether to warm cache after invalidation
            
        Returns:
            InvalidationResult: Invalidation operation result
        """
        invalidation_data = {
            "patterns": [pattern],
            "strategy": InvalidationStrategy.IMMEDIATE,
            "warm_after": warm_after
        }
        
        return await self._execute_invalidation(invalidation_data)
    
    async def get_invalidation_stats(self) -> Dict[str, Any]:
        """
        Get cache invalidation statistics.
        
        Returns:
            Dict: Invalidation performance statistics
        """
        return {
            **self.invalidation_stats,
            "registered_rules": sum(len(rules) for rules in self.invalidation_rules.values()),
            "pending_invalidations": self.pending_invalidations.qsize(),
            "service_running": self._running
        }
    
    async def shutdown(self) -> None:
        """Shutdown the cache invalidation service"""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Cache invalidation service shutdown complete")
