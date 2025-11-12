"""
Prometheus Metrics Service

Provides Prometheus-compatible metrics for monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from typing import Dict, Any
import time
import psutil
from datetime import datetime

# Application Info
app_info = Info('greta_pai_app', 'GRETA PAI Application Information')
app_info.info({
    'version': '2.2',
    'environment': 'production',
    'name': 'GRETA PAI Enhanced Backend'
})

# Request Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint']
)

# Database Metrics
db_connections_active = Gauge(
    'db_connections_active',
    'Number of active database connections'
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'collection']
)

db_operations_total = Counter(
    'db_operations_total',
    'Total database operations',
    ['operation', 'collection', 'status']
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

cache_size_bytes = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type']
)

# WebSocket Metrics
websocket_connections_active = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections'
)

websocket_messages_sent_total = Counter(
    'websocket_messages_sent_total',
    'Total WebSocket messages sent',
    ['event_type']
)

websocket_messages_received_total = Counter(
    'websocket_messages_received_total',
    'Total WebSocket messages received',
    ['event_type']
)

# Workflow Metrics
workflow_executions_total = Counter(
    'workflow_executions_total',
    'Total workflow executions',
    ['status']
)

workflow_execution_duration_seconds = Histogram(
    'workflow_execution_duration_seconds',
    'Workflow execution duration in seconds',
    ['workflow_id']
)

workflow_node_executions_total = Counter(
    'workflow_node_executions_total',
    'Total workflow node executions',
    ['node_type', 'status']
)

# AI/LLM Metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['provider', 'model', 'status']
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['provider', 'model']
)

llm_tokens_used_total = Counter(
    'llm_tokens_used_total',
    'Total tokens used',
    ['provider', 'model', 'type']  # type: prompt, completion
)

# Lead Scoring Metrics
lead_scores_calculated_total = Counter(
    'lead_scores_calculated_total',
    'Total lead scores calculated'
)

lead_score_distribution = Histogram(
    'lead_score_distribution',
    'Distribution of lead scores',
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

# System Metrics
system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage_percent = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

system_disk_usage_percent = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage'
)

process_cpu_usage_percent = Gauge(
    'process_cpu_usage_percent',
    'Process CPU usage percentage'
)

process_memory_usage_bytes = Gauge(
    'process_memory_usage_bytes',
    'Process memory usage in bytes'
)

process_threads_count = Gauge(
    'process_threads_count',
    'Number of process threads'
)

# Error Metrics
errors_total = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'endpoint']
)

# Business Metrics
users_active_total = Gauge(
    'users_active_total',
    'Number of active users',
    ['time_window']  # 1h, 24h, 7d
)

leads_generated_total = Counter(
    'leads_generated_total',
    'Total leads generated',
    ['source']
)

campaigns_active_total = Gauge(
    'campaigns_active_total',
    'Number of active campaigns'
)


class MetricsCollector:
    """Collects and updates system metrics"""
    
    def __init__(self):
        self.last_update = time.time()
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_usage_percent.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            system_memory_usage_percent.set(memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            system_disk_usage_percent.set(disk_percent)
            
            # Process metrics
            process = psutil.Process()
            process_cpu_usage_percent.set(process.cpu_percent())
            process_memory_usage_bytes.set(process.memory_info().rss)
            process_threads_count.set(process.num_threads())
            
            self.last_update = time.time()
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")
    
    def should_update(self, interval: int = 15) -> bool:
        """Check if metrics should be updated (default: every 15 seconds)"""
        return (time.time() - self.last_update) >= interval


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_prometheus_metrics() -> Response:
    """
    Generate Prometheus metrics in text format
    
    Returns:
        Response with Prometheus metrics
    """
    # Update system metrics before generating output
    if metrics_collector.should_update():
        metrics_collector.update_system_metrics()
    
    # Generate metrics in Prometheus format
    metrics_output = generate_latest()
    
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


# Middleware helper functions
def track_request_start(method: str, endpoint: str):
    """Track request start"""
    http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()


def track_request_end(method: str, endpoint: str, status: int, duration: float):
    """Track request completion"""
    http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()
    http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


def track_db_operation(operation: str, collection: str, status: str, duration: float):
    """Track database operation"""
    db_operations_total.labels(operation=operation, collection=collection, status=status).inc()
    db_query_duration_seconds.labels(operation=operation, collection=collection).observe(duration)


def track_cache_hit(cache_type: str = "redis"):
    """Track cache hit"""
    cache_hits_total.labels(cache_type=cache_type).inc()


def track_cache_miss(cache_type: str = "redis"):
    """Track cache miss"""
    cache_misses_total.labels(cache_type=cache_type).inc()


def track_websocket_connection(delta: int):
    """Track WebSocket connection change"""
    if delta > 0:
        websocket_connections_active.inc(delta)
    else:
        websocket_connections_active.dec(abs(delta))


def track_websocket_message(event_type: str, direction: str = "sent"):
    """Track WebSocket message"""
    if direction == "sent":
        websocket_messages_sent_total.labels(event_type=event_type).inc()
    else:
        websocket_messages_received_total.labels(event_type=event_type).inc()


def track_workflow_execution(status: str, duration: float, workflow_id: str):
    """Track workflow execution"""
    workflow_executions_total.labels(status=status).inc()
    workflow_execution_duration_seconds.labels(workflow_id=workflow_id).observe(duration)


def track_workflow_node(node_type: str, status: str):
    """Track workflow node execution"""
    workflow_node_executions_total.labels(node_type=node_type, status=status).inc()


def track_llm_request(provider: str, model: str, status: str, duration: float, tokens: Dict[str, int]):
    """Track LLM request"""
    llm_requests_total.labels(provider=provider, model=model, status=status).inc()
    llm_request_duration_seconds.labels(provider=provider, model=model).observe(duration)
    
    if tokens:
        if 'prompt' in tokens:
            llm_tokens_used_total.labels(provider=provider, model=model, type='prompt').inc(tokens['prompt'])
        if 'completion' in tokens:
            llm_tokens_used_total.labels(provider=provider, model=model, type='completion').inc(tokens['completion'])


def track_lead_score(score: float):
    """Track lead score calculation"""
    lead_scores_calculated_total.inc()
    lead_score_distribution.observe(score)


def track_error(error_type: str, endpoint: str):
    """Track error occurrence"""
    errors_total.labels(error_type=error_type, endpoint=endpoint).inc()

