"""
Metrics Router

Exposes application metrics for monitoring systems
"""

from fastapi import APIRouter, Response
from backend.services.prometheus_metrics import get_prometheus_metrics

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get(
    "",
    summary="Prometheus Metrics",
    description="""
    **Prometheus-compatible metrics endpoint**
    
    Exposes application metrics in Prometheus text format for scraping by monitoring systems.
    
    **Metrics Categories:**
    - HTTP request metrics (count, duration, in-progress)
    - Database operation metrics (queries, connections)
    - Cache metrics (hits, misses, size)
    - WebSocket metrics (connections, messages)
    - Workflow execution metrics
    - LLM/AI request metrics (tokens, duration)
    - Lead scoring metrics
    - System resource metrics (CPU, memory, disk)
    - Error metrics
    - Business metrics (active users, leads, campaigns)
    
    **Use Cases:**
    - Prometheus scraping
    - Grafana dashboards
    - Alerting rules
    - Performance monitoring
    - Capacity planning
    
    **Format:** Prometheus text format
    **Content-Type:** text/plain; version=0.0.4
    """,
    responses={
        200: {
            "description": "Metrics in Prometheus format",
            "content": {
                "text/plain": {
                    "example": """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/api/v1/health",status="200"} 1523.0

# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",endpoint="/api/v1/health",le="0.005"} 1200.0
http_request_duration_seconds_bucket{method="GET",endpoint="/api/v1/health",le="0.01"} 1450.0
http_request_duration_seconds_sum{method="GET",endpoint="/api/v1/health"} 125.5
http_request_duration_seconds_count{method="GET",endpoint="/api/v1/health"} 1523.0

# HELP system_cpu_usage_percent System CPU usage percentage
# TYPE system_cpu_usage_percent gauge
system_cpu_usage_percent 25.5

# HELP system_memory_usage_percent System memory usage percentage
# TYPE system_memory_usage_percent gauge
system_memory_usage_percent 45.2
"""
                }
            }
        }
    }
)
async def prometheus_metrics():
    """
    Get Prometheus metrics
    
    Returns metrics in Prometheus text format for scraping
    """
    return get_prometheus_metrics()


@router.get(
    "/health",
    summary="Metrics Health Check",
    description="Quick health check for metrics endpoint"
)
async def metrics_health():
    """Health check for metrics endpoint"""
    return {
        "status": "healthy",
        "endpoint": "/metrics",
        "format": "prometheus"
    }

