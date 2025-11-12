"""
Health Check Service

Provides comprehensive health checks for all system components including:
- Database connectivity and performance
- Cache (Redis) connectivity and performance
- Rate limiter functionality
- System resources
- Overall system health
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from database import Database

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    last_checked: datetime = field(default_factory=lambda: Database.utcnow())


@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus
    components: List[ComponentHealth]
    total_response_time_ms: float
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    timestamp: datetime = field(default_factory=lambda: Database.utcnow())


class HealthCheckService:
    """
    Health Check Service
    
    Monitors health of all system components and provides
    comprehensive health status information.
    """
    
    def __init__(self):
        self.db: Optional[Database] = None
        self.cache_service = None
        self.rate_limiter_service = None
        self.performance_monitor = None
        
        logger.info("✅ HealthCheckService initialized")
    
    async def connect(
        self,
        db: Database,
        cache_service=None,
        rate_limiter_service=None,
        performance_monitor=None
    ):
        """Connect to services"""
        self.db = db
        self.cache_service = cache_service
        self.rate_limiter_service = rate_limiter_service
        self.performance_monitor = performance_monitor
        
        logger.info("✅ HealthCheckService connected to services")
    
    async def check_database_health(self) -> ComponentHealth:
        """Check database health"""
        start_time = time.time()
        
        try:
            if not self.db:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    message="Database not connected"
                )
            
            # Test database connection with a simple query
            result = await self.db.db.command("ping")
            
            # Get database stats
            stats = await self.db.db.command("dbStats")
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check if response time is acceptable
            if response_time_ms > 1000:  # 1 second
                status = HealthStatus.DEGRADED
                message = f"Database responding slowly ({response_time_ms:.2f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = "Database is healthy"
            
            return ComponentHealth(
                name="database",
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details={
                    "ping": result.get("ok") == 1,
                    "collections": stats.get("collections", 0),
                    "data_size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2),
                    "storage_size_mb": round(stats.get("storageSize", 0) / (1024 * 1024), 2),
                    "indexes": stats.get("indexes", 0)
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Database health check failed: {e}")
            
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message=f"Database health check failed: {str(e)}"
            )
    
    async def check_cache_health(self) -> ComponentHealth:
        """Check cache (Redis) health"""
        start_time = time.time()
        
        try:
            if not self.cache_service:
                return ComponentHealth(
                    name="cache",
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0.0,
                    message="Cache service not configured"
                )
            
            # Test cache connection
            health = await self.cache_service.health_check()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if not health.get("connected"):
                return ComponentHealth(
                    name="cache",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=response_time_ms,
                    message="Cache not connected"
                )
            
            # Check response time
            if response_time_ms > 500:  # 500ms
                status = HealthStatus.DEGRADED
                message = f"Cache responding slowly ({response_time_ms:.2f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = "Cache is healthy"
            
            # Get cache metrics
            metrics = self.cache_service.get_metrics()
            
            return ComponentHealth(
                name="cache",
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details={
                    "connected": health.get("connected"),
                    "hit_ratio": round(metrics.get("hit_ratio", 0.0), 2),
                    "total_hits": metrics.get("total_hits", 0),
                    "total_misses": metrics.get("total_misses", 0),
                    "total_sets": metrics.get("total_sets", 0),
                    "total_deletes": metrics.get("total_deletes", 0)
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Cache health check failed: {e}")
            
            return ComponentHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message=f"Cache health check failed: {str(e)}"
            )
    
    async def check_rate_limiter_health(self) -> ComponentHealth:
        """Check rate limiter health"""
        start_time = time.time()
        
        try:
            if not self.rate_limiter_service:
                return ComponentHealth(
                    name="rate_limiter",
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0.0,
                    message="Rate limiter service not configured"
                )
            
            # Test rate limiter connection
            health = await self.rate_limiter_service.health_check()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if not health.get("connected"):
                return ComponentHealth(
                    name="rate_limiter",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=response_time_ms,
                    message="Rate limiter not connected"
                )
            
            # Check response time
            if response_time_ms > 500:  # 500ms
                status = HealthStatus.DEGRADED
                message = f"Rate limiter responding slowly ({response_time_ms:.2f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = "Rate limiter is healthy"
            
            # Get rate limiter metrics
            metrics = self.rate_limiter_service.get_metrics()
            
            return ComponentHealth(
                name="rate_limiter",
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details={
                    "connected": health.get("connected"),
                    "total_requests": metrics.get("total_requests", 0),
                    "total_allowed": metrics.get("total_allowed", 0),
                    "total_blocked": metrics.get("total_blocked", 0),
                    "block_ratio": round(metrics.get("block_ratio", 0.0), 2)
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Rate limiter health check failed: {e}")
            
            return ComponentHealth(
                name="rate_limiter",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message=f"Rate limiter health check failed: {str(e)}"
            )
    
    async def check_performance_monitor_health(self) -> ComponentHealth:
        """Check performance monitor health"""
        start_time = time.time()
        
        try:
            if not self.performance_monitor:
                return ComponentHealth(
                    name="performance_monitor",
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0.0,
                    message="Performance monitor not configured"
                )
            
            # Get performance monitor metrics
            metrics = self.performance_monitor.get_metrics()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check if monitoring is active
            if not metrics.get("is_monitoring"):
                status = HealthStatus.DEGRADED
                message = "Performance monitoring is not active"
            else:
                status = HealthStatus.HEALTHY
                message = "Performance monitor is healthy"
            
            return ComponentHealth(
                name="performance_monitor",
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details={
                    "is_monitoring": metrics.get("is_monitoring", False),
                    "total_requests_tracked": metrics.get("total_requests_tracked", 0),
                    "total_slow_requests": metrics.get("total_slow_requests", 0),
                    "total_error_requests": metrics.get("total_error_requests", 0),
                    "total_endpoints": metrics.get("total_endpoints", 0)
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Performance monitor health check failed: {e}")

            return ComponentHealth(
                name="performance_monitor",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message=f"Performance monitor health check failed: {str(e)}"
            )

    async def check_all_components(self) -> SystemHealth:
        """Check health of all system components"""
        start_time = time.time()

        try:
            # Run all health checks in parallel
            results = await asyncio.gather(
                self.check_database_health(),
                self.check_cache_health(),
                self.check_rate_limiter_health(),
                self.check_performance_monitor_health(),
                return_exceptions=True
            )

            # Filter out exceptions and convert to ComponentHealth
            components = []
            for result in results:
                if isinstance(result, ComponentHealth):
                    components.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"❌ Component health check failed: {result}")

            # Calculate overall health
            healthy_count = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
            degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
            unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)

            # Determine overall status
            if unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif degraded_count > 0:
                overall_status = HealthStatus.DEGRADED
            elif healthy_count == len(components):
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN

            total_response_time_ms = (time.time() - start_time) * 1000

            return SystemHealth(
                status=overall_status,
                components=components,
                total_response_time_ms=total_response_time_ms,
                healthy_components=healthy_count,
                degraded_components=degraded_count,
                unhealthy_components=unhealthy_count
            )

        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")

            return SystemHealth(
                status=HealthStatus.UNHEALTHY,
                components=[],
                total_response_time_ms=(time.time() - start_time) * 1000,
                healthy_components=0,
                degraded_components=0,
                unhealthy_components=0
            )

    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information for all components"""
        try:
            system_health = await self.check_all_components()

            return {
                "status": system_health.status.value,
                "timestamp": Database.to_iso(system_health.timestamp),
                "total_response_time_ms": round(system_health.total_response_time_ms, 2),
                "summary": {
                    "healthy": system_health.healthy_components,
                    "degraded": system_health.degraded_components,
                    "unhealthy": system_health.unhealthy_components,
                    "total": len(system_health.components)
                },
                "components": [
                    {
                        "name": c.name,
                        "status": c.status.value,
                        "response_time_ms": round(c.response_time_ms, 2),
                        "message": c.message,
                        "details": c.details,
                        "last_checked": Database.to_iso(c.last_checked)
                    }
                    for c in system_health.components
                ]
            }

        except Exception as e:
            logger.error(f"❌ Error getting detailed health: {e}")
            return {
                "status": "unhealthy",
                "timestamp": Database.to_iso(Database.utcnow()),
                "error": str(e)
            }

    async def get_simple_health(self) -> Dict[str, Any]:
        """Get simple health status (fast check)"""
        try:
            # Just check database connectivity
            db_health = await self.check_database_health()

            if db_health.status == HealthStatus.HEALTHY:
                return {
                    "status": "healthy",
                    "timestamp": Database.to_iso(Database.utcnow())
                }
            elif db_health.status == HealthStatus.DEGRADED:
                return {
                    "status": "degraded",
                    "message": db_health.message,
                    "timestamp": Database.to_iso(Database.utcnow())
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": db_health.message,
                    "timestamp": Database.to_iso(Database.utcnow())
                }

        except Exception as e:
            logger.error(f"❌ Error getting simple health: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": Database.to_iso(Database.utcnow())
            }

