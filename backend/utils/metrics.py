"""
Metrics Collection and Reporting for GRETA PAI
Provides performance metrics and monitoring capabilities
"""

import logging
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.recent_requests = deque(maxlen=1000)
        self.start_time = datetime.utcnow()
    
    def record_request(self, endpoint: str, method: str = "GET"):
        """Record an API request"""
        self.request_counts[f"{method} {endpoint}"] += 1
        self.recent_requests.append({
            "endpoint": endpoint,
            "method": method,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def record_response_time(self, endpoint: str, response_time_ms: float):
        """Record response time for an endpoint"""
        self.response_times[endpoint].append(response_time_ms)
        # Keep only last 100 response times per endpoint
        if len(self.response_times[endpoint]) > 100:
            self.response_times[endpoint] = self.response_times[endpoint][-100:]
    
    def record_error(self, endpoint: str, error_type: str = "unknown"):
        """Record an error"""
        self.error_counts[f"{endpoint}:{error_type}"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        # Calculate average response times
        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "uptime_seconds": uptime_seconds,
            "request_counts": dict(self.request_counts),
            "average_response_times_ms": avg_response_times,
            "recent_requests_count": len(self.recent_requests),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.request_counts.clear()
        self.response_times.clear()
        self.error_counts.clear()
        self.recent_requests.clear()
        self.start_time = datetime.utcnow()
        logger.info("Metrics reset")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get current metrics summary
    
    Returns:
        Dict with metrics data
    """
    try:
        return metrics_collector.get_summary()
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        return {
            "error": str(e),
            "total_requests": 0,
            "total_errors": 0,
            "timestamp": datetime.utcnow().isoformat()
        }


def record_request(endpoint: str, method: str = "GET"):
    """Record an API request"""
    try:
        metrics_collector.record_request(endpoint, method)
    except Exception as e:
        logger.error(f"Error recording request: {e}")


def record_response_time(endpoint: str, response_time_ms: float):
    """Record response time for an endpoint"""
    try:
        metrics_collector.record_response_time(endpoint, response_time_ms)
    except Exception as e:
        logger.error(f"Error recording response time: {e}")


def record_error(endpoint: str, error_type: str = "unknown"):
    """Record an error"""
    try:
        metrics_collector.record_error(endpoint, error_type)
    except Exception as e:
        logger.error(f"Error recording error: {e}")

