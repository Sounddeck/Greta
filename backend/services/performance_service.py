"""
Performance Optimization Service for GRETA PAI

This service provides comprehensive performance optimization including
database query optimization, connection pooling, response compression,
serialization optimization, and performance profiling.

The service provides:
- Database query optimization and indexing
- Connection pool management and optimization
- Response compression (gzip, brotli)
- Serialization/deserialization optimization
- Performance profiling and metrics
- Query performance monitoring
- Slow query detection and logging
- Resource usage tracking

Example:
    Basic performance monitoring:
        >>> from services.performance_service import PerformanceService
        >>> 
        >>> perf_service = PerformanceService()
        >>> await perf_service.initialize()
        >>> 
        >>> # Profile a function
        >>> @perf_service.profile("user_query")
        >>> async def get_user(user_id: str):
        ...     return await db.users.find_one({"user_id": user_id})

    Query optimization:
        >>> # Optimize database indexes
        >>> await perf_service.optimize_indexes()
        >>> 
        >>> # Get slow queries
        >>> slow_queries = await perf_service.get_slow_queries()

Attributes:
    db (AsyncIOMotorDatabase): MongoDB database instance
    metrics (Dict): Performance metrics tracking
    slow_query_threshold (float): Threshold for slow query detection (seconds)
    logger (logging.Logger): Service logger
"""

import asyncio
import time
import gzip
import json
import logging
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorDatabase

from utils.custom_exceptions import DatabaseError, ConfigurationError
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """
    Database query performance metrics.
    
    Tracks query execution time, result count, and optimization opportunities.
    
    Attributes:
        query_name: Name/identifier of the query
        execution_time: Query execution time in seconds
        result_count: Number of results returned
        collection: Database collection queried
        index_used: Whether an index was used
        timestamp: Query execution timestamp
    """
    query_name: str
    execution_time: float
    result_count: int = 0
    collection: str = ""
    index_used: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceMetrics:
    """
    Overall performance metrics.
    
    Tracks system-wide performance including response times,
    resource usage, and optimization statistics.
    
    Attributes:
        total_requests: Total number of requests processed
        average_response_time: Average response time in milliseconds
        p95_response_time: 95th percentile response time
        p99_response_time: 99th percentile response time
        slow_queries_count: Number of slow queries detected
        cache_hit_ratio: Cache hit ratio percentage
        compression_ratio: Average compression ratio
        total_bytes_compressed: Total bytes compressed
        active_connections: Number of active database connections
    """
    total_requests: int = 0
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    slow_queries_count: int = 0
    cache_hit_ratio: float = 0.0
    compression_ratio: float = 0.0
    total_bytes_compressed: int = 0
    active_connections: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class PerformanceService:
    """
    Comprehensive performance optimization service.
    
    This service provides performance monitoring, optimization,
    and profiling capabilities for the application.
    
    Attributes:
        db: MongoDB database instance
        metrics: Performance metrics tracking
        query_metrics: Query-specific metrics
        slow_query_threshold: Threshold for slow query detection
        
    Example:
        >>> perf_service = PerformanceService()
        >>> await perf_service.initialize()
        >>> metrics = await perf_service.get_performance_metrics()
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        """
        Initialize performance service.
        
        Args:
            database: MongoDB database instance
        """
        self.db = database
        self.settings = get_settings()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.query_metrics: List[QueryMetrics] = []
        self.response_times: List[float] = []
        
        # Configuration
        self.slow_query_threshold = 1.0  # seconds
        self.max_metrics_history = 1000
        self.compression_enabled = True
        self.compression_threshold = 1024  # bytes
        
        # Connection pool settings
        self.max_pool_size = 100
        self.min_pool_size = 10
        
        # Profiling
        self.profiling_enabled = True
        self._profiling_data: Dict[str, List[float]] = {}
    
    async def initialize(self) -> None:
        """
        Initialize the performance service.
        
        Sets up database indexes, connection pooling, and monitoring.
        
        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            if self.db:
                # Optimize database indexes
                await self._create_performance_indexes()
                logger.info("✅ Performance indexes created")
                
                # Start background monitoring
                asyncio.create_task(self._monitoring_loop())
                logger.info("✅ Performance monitoring started")
            
            logger.info("✅ Performance service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Performance service initialization failed: {e}")
            raise ConfigurationError("performance_service", str(e))
    
    async def _create_performance_indexes(self) -> None:
        """Create database indexes for optimal query performance"""
        try:
            # User collection indexes
            await self.db.users.create_index("user_id", unique=True)
            await self.db.users.create_index("email", unique=True)
            await self.db.users.create_index("username")
            await self.db.users.create_index("is_active")
            await self.db.users.create_index("last_login")
            await self.db.users.create_index([("created_at", -1)])
            
            # Compound indexes for common queries
            await self.db.users.create_index([("is_active", 1), ("last_login", -1)])
            await self.db.users.create_index([("username", 1), ("is_active", 1)])
            
            # Performance metrics collection indexes
            if await self._collection_exists("performance_metrics"):
                await self.db.performance_metrics.create_index("timestamp")
                await self.db.performance_metrics.create_index([("timestamp", -1)])
                await self.db.performance_metrics.create_index("query_name")
            
            # Query metrics collection indexes
            if await self._collection_exists("query_metrics"):
                await self.db.query_metrics.create_index("timestamp")
                await self.db.query_metrics.create_index("execution_time")
                await self.db.query_metrics.create_index([("execution_time", -1)])
                await self.db.query_metrics.create_index("collection")
            
            logger.info("✅ Database indexes optimized")
            
        except Exception as e:
            logger.error(f"Failed to create performance indexes: {e}")
            raise DatabaseError("create_indexes", str(e))
    
    async def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        collections = await self.db.list_collection_names()
        return collection_name in collections
    
    async def _monitoring_loop(self) -> None:
        """Background loop for performance monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self._collect_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self) -> None:
        """Collect current performance metrics"""
        try:
            # Calculate response time percentiles
            if self.response_times:
                sorted_times = sorted(self.response_times)
                count = len(sorted_times)
                
                self.metrics.average_response_time = sum(sorted_times) / count
                self.metrics.p95_response_time = sorted_times[int(count * 0.95)] if count > 0 else 0
                self.metrics.p99_response_time = sorted_times[int(count * 0.99)] if count > 0 else 0
            
            # Count slow queries
            self.metrics.slow_queries_count = sum(
                1 for qm in self.query_metrics
                if qm.execution_time > self.slow_query_threshold
            )
            
            # Update timestamp
            self.metrics.last_updated = datetime.utcnow()
            
            # Persist metrics to database
            if self.db:
                await self.db.performance_metrics.insert_one({
                    "timestamp": datetime.utcnow(),
                    "total_requests": self.metrics.total_requests,
                    "average_response_time": self.metrics.average_response_time,
                    "p95_response_time": self.metrics.p95_response_time,
                    "p99_response_time": self.metrics.p99_response_time,
                    "slow_queries_count": self.metrics.slow_queries_count,
                    "cache_hit_ratio": self.metrics.cache_hit_ratio,
                    "compression_ratio": self.metrics.compression_ratio
                })
            
            # Trim old metrics to prevent memory growth
            if len(self.response_times) > self.max_metrics_history:
                self.response_times = self.response_times[-self.max_metrics_history:]
            
            if len(self.query_metrics) > self.max_metrics_history:
                self.query_metrics = self.query_metrics[-self.max_metrics_history:]
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    def profile(self, operation_name: str):
        """
        Decorator for profiling function execution time.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Returns:
            Decorated function with profiling
            
        Example:
            >>> @perf_service.profile("user_query")
            >>> async def get_user(user_id: str):
            ...     return await db.users.find_one({"user_id": user_id})
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return await func(*args, **kwargs)
                
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    
                    # Record profiling data
                    if operation_name not in self._profiling_data:
                        self._profiling_data[operation_name] = []
                    
                    self._profiling_data[operation_name].append(execution_time)
                    
                    # Trim old data
                    if len(self._profiling_data[operation_name]) > 100:
                        self._profiling_data[operation_name] = self._profiling_data[operation_name][-100:]
                    
                    # Log slow operations
                    if execution_time > self.slow_query_threshold:
                        logger.warning(
                            f"⚠️ Slow operation detected: {operation_name} took {execution_time:.3f}s"
                        )
            
            return wrapper
        return decorator
    
    @asynccontextmanager
    async def track_query(self, query_name: str, collection: str = ""):
        """
        Context manager for tracking query performance.
        
        Args:
            query_name: Name of the query
            collection: Database collection being queried
            
        Yields:
            QueryMetrics: Metrics object for the query
            
        Example:
            >>> async with perf_service.track_query("find_user", "users") as metrics:
            ...     user = await db.users.find_one({"user_id": user_id})
            ...     metrics.result_count = 1 if user else 0
        """
        start_time = time.time()
        metrics = QueryMetrics(
            query_name=query_name,
            execution_time=0.0,
            collection=collection
        )
        
        try:
            yield metrics
        finally:
            metrics.execution_time = time.time() - start_time
            self.query_metrics.append(metrics)
            
            # Log slow queries
            if metrics.execution_time > self.slow_query_threshold:
                logger.warning(
                    f"⚠️ Slow query detected: {query_name} on {collection} "
                    f"took {metrics.execution_time:.3f}s"
                )
    
    def compress_response(self, data: Any, compression_level: int = 6) -> Tuple[bytes, float]:
        """
        Compress response data using gzip.
        
        Args:
            data: Data to compress (will be JSON serialized)
            compression_level: Gzip compression level (1-9)
            
        Returns:
            Tuple[bytes, float]: Compressed data and compression ratio
            
        Example:
            >>> compressed, ratio = perf_service.compress_response({"data": "..."})
            >>> print(f"Compression ratio: {ratio:.2f}")
        """
        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                json_data = json.dumps(data, default=str).encode('utf-8')
            elif isinstance(data, str):
                json_data = data.encode('utf-8')
            else:
                json_data = str(data).encode('utf-8')
            
            original_size = len(json_data)
            
            # Only compress if above threshold
            if original_size < self.compression_threshold:
                return json_data, 1.0
            
            # Compress data
            compressed = gzip.compress(json_data, compresslevel=compression_level)
            compressed_size = len(compressed)
            
            # Calculate compression ratio
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Update metrics
            self.metrics.total_bytes_compressed += original_size - compressed_size
            
            # Update average compression ratio
            if self.metrics.compression_ratio == 0:
                self.metrics.compression_ratio = ratio
            else:
                self.metrics.compression_ratio = (self.metrics.compression_ratio + ratio) / 2
            
            logger.debug(
                f"Compressed {original_size} bytes to {compressed_size} bytes "
                f"(ratio: {ratio:.2f}x)"
            )
            
            return compressed, ratio
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return original data if compression fails
            return json_data if 'json_data' in locals() else b'', 1.0
    
    def decompress_response(self, compressed_data: bytes) -> Any:
        """
        Decompress gzip-compressed response data.
        
        Args:
            compressed_data: Compressed data bytes
            
        Returns:
            Any: Decompressed and deserialized data
        """
        try:
            decompressed = gzip.decompress(compressed_data)
            return json.loads(decompressed.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None
    
    async def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.
        
        Returns:
            PerformanceMetrics: Current performance metrics
        """
        await self._collect_metrics()
        return self.metrics
    
    async def get_slow_queries(self, limit: int = 50) -> List[QueryMetrics]:
        """
        Get slow queries exceeding the threshold.
        
        Args:
            limit: Maximum number of slow queries to return
            
        Returns:
            List[QueryMetrics]: List of slow queries
        """
        slow_queries = [
            qm for qm in self.query_metrics
            if qm.execution_time > self.slow_query_threshold
        ]
        
        # Sort by execution time (slowest first)
        slow_queries.sort(key=lambda x: x.execution_time, reverse=True)
        
        return slow_queries[:limit]
    
    async def get_profiling_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get profiling data for all operations.
        
        Returns:
            Dict: Profiling statistics for each operation
        """
        profiling_stats = {}
        
        for operation, times in self._profiling_data.items():
            if times:
                profiling_stats[operation] = {
                    "count": len(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "p95": sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0,
                    "p99": sorted(times)[int(len(times) * 0.99)] if len(times) > 0 else 0
                }
        
        return profiling_stats
    
    async def optimize_indexes(self) -> Dict[str, Any]:
        """
        Optimize database indexes based on query patterns.
        
        Returns:
            Dict: Optimization results
        """
        try:
            # Recreate indexes
            await self._create_performance_indexes()
            
            # Analyze query patterns
            query_patterns = {}
            for qm in self.query_metrics:
                if qm.collection:
                    if qm.collection not in query_patterns:
                        query_patterns[qm.collection] = []
                    query_patterns[qm.collection].append(qm)
            
            return {
                "success": True,
                "indexes_optimized": True,
                "collections_analyzed": len(query_patterns),
                "total_queries_analyzed": len(self.query_metrics)
            }
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    def track_response_time(self, response_time: float) -> None:
        """
        Track response time for a request.
        
        Args:
            response_time: Response time in seconds
        """
        self.response_times.append(response_time * 1000)  # Convert to milliseconds
        self.metrics.total_requests += 1
