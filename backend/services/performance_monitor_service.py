"""
Performance Monitor Service

Provides comprehensive application performance monitoring (APM) including:
- Request/response time tracking
- Endpoint performance metrics
- Slow request detection
- Performance profiling
- Error rate tracking
"""

import time
import asyncio
import psutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from motor.motor_asyncio import AsyncIOMotorClient

from database import Database

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: datetime
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    error: Optional[str] = None
    query_count: int = 0
    query_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class EndpointStats:
    """Aggregated statistics for an endpoint"""
    endpoint: str
    method: str
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    error_rate: float = 0.0
    requests_per_minute: float = 0.0
    slow_requests: int = 0
    last_updated: datetime = field(default_factory=lambda: Database.utcnow())


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int = 0
    thread_count: int = 0


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration"""
    slow_request_threshold_ms: float = 1000.0  # 1 second
    slow_query_threshold_ms: float = 100.0     # 100ms
    metrics_retention_hours: int = 24
    metrics_buffer_size: int = 10000
    enable_profiling: bool = True
    enable_query_tracking: bool = True
    enable_system_monitoring: bool = True
    system_metrics_interval_seconds: int = 60


class PerformanceMonitorService:
    """
    Performance Monitor Service
    
    Tracks application performance metrics including:
    - Request/response times
    - Endpoint statistics
    - Slow request detection
    - System resource usage
    - Database query performance
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.db: Optional[Database] = None
        
        # In-memory metrics buffers
        self.request_metrics: deque = deque(maxlen=self.config.metrics_buffer_size)
        self.endpoint_stats: Dict[str, EndpointStats] = {}
        self.system_metrics: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.slow_requests: deque = deque(maxlen=1000)
        self.error_requests: deque = deque(maxlen=1000)
        
        # System monitoring
        self.system_monitor_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        logger.info("✅ PerformanceMonitorService initialized")
    
    async def connect(self, db: Database):
        """Connect to database and start monitoring"""
        self.db = db
        
        # Create indexes for performance metrics
        await self._create_indexes()
        
        # Start system monitoring
        if self.config.enable_system_monitoring:
            await self.start_system_monitoring()
        
        logger.info("✅ PerformanceMonitorService connected to database")
    
    async def _create_indexes(self):
        """Create database indexes for performance metrics"""
        try:
            # Request metrics indexes
            await self.db.db.performance_requests.create_index([("request_id", 1)], unique=True)
            await self.db.db.performance_requests.create_index([("endpoint", 1)])
            await self.db.db.performance_requests.create_index([("timestamp", 1)])
            await self.db.db.performance_requests.create_index([("response_time_ms", 1)])
            await self.db.db.performance_requests.create_index([("status_code", 1)])
            
            # Endpoint stats indexes
            await self.db.db.performance_endpoints.create_index([("endpoint", 1), ("method", 1)], unique=True)
            await self.db.db.performance_endpoints.create_index([("avg_response_time_ms", 1)])
            await self.db.db.performance_endpoints.create_index([("error_rate", 1)])
            
            # System metrics indexes
            await self.db.db.performance_system.create_index([("timestamp", 1)])
            
            logger.info("✅ Created performance monitoring indexes")
        except Exception as e:
            logger.error(f"❌ Error creating performance indexes: {e}")
    
    async def track_request(self, metrics: RequestMetrics):
        """Track a request's performance metrics"""
        try:
            # Add to in-memory buffer
            self.request_metrics.append(metrics)
            
            # Check if slow request
            if metrics.response_time_ms >= self.config.slow_request_threshold_ms:
                self.slow_requests.append(metrics)
                logger.warning(
                    f"🐌 Slow request detected: {metrics.method} {metrics.endpoint} "
                    f"took {metrics.response_time_ms:.2f}ms"
                )
            
            # Check if error
            if metrics.status_code >= 400:
                self.error_requests.append(metrics)
            
            # Update endpoint statistics
            await self._update_endpoint_stats(metrics)
            
            # Persist to database (async, non-blocking)
            if self.db:
                asyncio.create_task(self._persist_request_metrics(metrics))
            
        except Exception as e:
            logger.error(f"❌ Error tracking request: {e}")
    
    async def _update_endpoint_stats(self, metrics: RequestMetrics):
        """Update aggregated endpoint statistics"""
        key = f"{metrics.method}:{metrics.endpoint}"
        
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = EndpointStats(
                endpoint=metrics.endpoint,
                method=metrics.method
            )
        
        stats = self.endpoint_stats[key]
        stats.total_requests += 1
        
        if metrics.status_code >= 400:
            stats.total_errors += 1
        
        if metrics.response_time_ms >= self.config.slow_request_threshold_ms:
            stats.slow_requests += 1
        
        # Update response time statistics
        stats.min_response_time_ms = min(stats.min_response_time_ms, metrics.response_time_ms)
        stats.max_response_time_ms = max(stats.max_response_time_ms, metrics.response_time_ms)
        
        # Calculate moving average
        old_avg = stats.avg_response_time_ms
        stats.avg_response_time_ms = (
            (old_avg * (stats.total_requests - 1) + metrics.response_time_ms) / stats.total_requests
        )
        
        # Calculate error rate
        stats.error_rate = stats.total_errors / stats.total_requests if stats.total_requests > 0 else 0.0
        
        stats.last_updated = Database.utcnow()
    
    async def _persist_request_metrics(self, metrics: RequestMetrics):
        """Persist request metrics to database"""
        try:
            if not self.db:
                return
            
            doc = {
                "request_id": metrics.request_id,
                "endpoint": metrics.endpoint,
                "method": metrics.method,
                "status_code": metrics.status_code,
                "response_time_ms": metrics.response_time_ms,
                "timestamp": Database.to_iso(metrics.timestamp),
                "user_id": metrics.user_id,
                "ip_address": metrics.ip_address,
                "error": metrics.error,
                "query_count": metrics.query_count,
                "query_time_ms": metrics.query_time_ms,
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
            }
            
            await self.db.db.performance_requests.insert_one(doc)
            
        except Exception as e:
            logger.error(f"❌ Error persisting request metrics: {e}")

    async def get_endpoint_stats(
        self,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        sort_by: str = "avg_response_time_ms",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get endpoint statistics"""
        try:
            stats_list = list(self.endpoint_stats.values())

            # Filter by endpoint/method if specified
            if endpoint:
                stats_list = [s for s in stats_list if s.endpoint == endpoint]
            if method:
                stats_list = [s for s in stats_list if s.method == method]

            # Sort
            reverse = sort_by not in ["endpoint", "method"]
            stats_list.sort(key=lambda x: getattr(x, sort_by, 0), reverse=reverse)

            # Limit
            stats_list = stats_list[:limit]

            # Convert to dict
            return [asdict(s) for s in stats_list]

        except Exception as e:
            logger.error(f"❌ Error getting endpoint stats: {e}")
            return []

    async def get_slow_requests(
        self,
        limit: int = 100,
        threshold_ms: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get slow requests"""
        try:
            threshold = threshold_ms or self.config.slow_request_threshold_ms

            slow = [
                asdict(m) for m in self.slow_requests
                if m.response_time_ms >= threshold
            ]

            # Sort by response time (slowest first)
            slow.sort(key=lambda x: x["response_time_ms"], reverse=True)

            return slow[:limit]

        except Exception as e:
            logger.error(f"❌ Error getting slow requests: {e}")
            return []

    async def get_error_requests(
        self,
        limit: int = 100,
        status_code: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get error requests"""
        try:
            errors = list(self.error_requests)

            # Filter by status code if specified
            if status_code:
                errors = [e for e in errors if e.status_code == status_code]

            # Convert to dict
            errors = [asdict(e) for e in errors]

            # Sort by timestamp (most recent first)
            errors.sort(key=lambda x: x["timestamp"], reverse=True)

            return errors[:limit]

        except Exception as e:
            logger.error(f"❌ Error getting error requests: {e}")
            return []

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0.0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0.0

            # Network metrics
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0.0
            network_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0.0

            # Process metrics
            process = psutil.Process()
            active_connections = len(process.connections())
            thread_count = process.num_threads()

            metrics = SystemMetrics(
                timestamp=Database.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_connections=active_connections,
                thread_count=thread_count
            )

            # Add to buffer
            self.system_metrics.append(metrics)

            # Persist to database
            if self.db:
                asyncio.create_task(self._persist_system_metrics(metrics))

            return metrics

        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=Database.utcnow(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0
            )

    async def _persist_system_metrics(self, metrics: SystemMetrics):
        """Persist system metrics to database"""
        try:
            if not self.db:
                return

            doc = asdict(metrics)
            doc["timestamp"] = Database.to_iso(metrics.timestamp)

            await self.db.db.performance_system.insert_one(doc)

        except Exception as e:
            logger.error(f"❌ Error persisting system metrics: {e}")

    async def start_system_monitoring(self):
        """Start background system monitoring"""
        if self.is_monitoring:
            logger.warning("⚠️ System monitoring already running")
            return

        self.is_monitoring = True
        self.system_monitor_task = asyncio.create_task(self._system_monitor_loop())
        logger.info("✅ Started system monitoring")

    async def stop_system_monitoring(self):
        """Stop background system monitoring"""
        self.is_monitoring = False

        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("✅ Stopped system monitoring")

    async def _system_monitor_loop(self):
        """Background loop for system monitoring"""
        while self.is_monitoring:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.config.system_metrics_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in system monitor loop: {e}")
                await asyncio.sleep(self.config.system_metrics_interval_seconds)

    async def get_system_metrics(
        self,
        hours: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent system metrics"""
        try:
            cutoff = Database.utcnow() - timedelta(hours=hours)

            metrics = [
                asdict(m) for m in self.system_metrics
                if m.timestamp >= cutoff
            ]

            # Sort by timestamp (most recent first)
            metrics.sort(key=lambda x: x["timestamp"], reverse=True)

            return metrics[:limit]

        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return []

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        try:
            # Calculate summary statistics
            total_requests = len(self.request_metrics)
            total_slow = len(self.slow_requests)
            total_errors = len(self.error_requests)

            # Calculate average response time
            if total_requests > 0:
                avg_response_time = sum(m.response_time_ms for m in self.request_metrics) / total_requests
            else:
                avg_response_time = 0.0

            # Get current system metrics
            current_system = await self.collect_system_metrics()

            # Get top slow endpoints
            endpoint_stats = await self.get_endpoint_stats(sort_by="avg_response_time_ms", limit=5)

            return {
                "summary": {
                    "total_requests": total_requests,
                    "total_slow_requests": total_slow,
                    "total_errors": total_errors,
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "slow_request_rate": round(total_slow / total_requests * 100, 2) if total_requests > 0 else 0.0,
                    "error_rate": round(total_errors / total_requests * 100, 2) if total_requests > 0 else 0.0,
                },
                "system": asdict(current_system),
                "top_slow_endpoints": endpoint_stats,
                "timestamp": Database.to_iso(Database.utcnow())
            }

        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {
                "summary": {},
                "system": {},
                "top_slow_endpoints": [],
                "timestamp": Database.to_iso(Database.utcnow())
            }

    async def cleanup_old_metrics(self, hours: int = 24):
        """Clean up old metrics from database"""
        try:
            if not self.db:
                return

            cutoff = Database.utcnow() - timedelta(hours=hours)
            cutoff_iso = Database.to_iso(cutoff)

            # Delete old request metrics
            result1 = await self.db.db.performance_requests.delete_many({
                "timestamp": {"$lt": cutoff_iso}
            })

            # Delete old system metrics
            result2 = await self.db.db.performance_system.delete_many({
                "timestamp": {"$lt": cutoff_iso}
            })

            logger.info(
                f"✅ Cleaned up old metrics: {result1.deleted_count} requests, "
                f"{result2.deleted_count} system metrics"
            )

        except Exception as e:
            logger.error(f"❌ Error cleaning up old metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            "total_requests_tracked": len(self.request_metrics),
            "total_slow_requests": len(self.slow_requests),
            "total_error_requests": len(self.error_requests),
            "total_endpoints": len(self.endpoint_stats),
            "system_metrics_count": len(self.system_metrics),
            "is_monitoring": self.is_monitoring,
            "config": asdict(self.config)
        }

