"""
Performance Logging Utilities for CPAS5
Provides decorators and utilities for tracking execution time, token usage, and database performance
"""

import time
import functools
from typing import Callable, Any, Optional
from loguru import logger


def track_execution_time(operation_name: Optional[str] = None):
    """
    Decorator to track execution time of functions
    
    Args:
        operation_name: Optional name for the operation (defaults to function name)
    
    Usage:
        @track_execution_time("user_authentication")
        async def authenticate_user(username: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Operation completed: {op_name}",
                    extra={
                        "event": "operation_completed",
                        "operation": op_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    f"Operation failed: {op_name}",
                    extra={
                        "event": "operation_failed",
                        "operation": op_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Operation completed: {op_name}",
                    extra={
                        "event": "operation_completed",
                        "operation": op_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    f"Operation failed: {op_name}",
                    extra={
                        "event": "operation_failed",
                        "operation": op_name,
                        "duration_ms": round(duration_ms, 2),
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_llm_call(model_name: Optional[str] = None):
    """
    Decorator to track LLM API calls with token usage
    
    Args:
        model_name: Optional model name (can be extracted from result if not provided)
    
    Usage:
        @track_llm_call("gpt-4")
        async def call_llm(prompt: str):
            ...
            return {"tokens": 150, "response": "..."}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract token usage from result if available
                tokens = None
                if isinstance(result, dict):
                    tokens = result.get("tokens") or result.get("token_count")
                
                logger.info(
                    f"LLM call completed: {model_name or 'unknown'}",
                    extra={
                        "event": "llm_call_completed",
                        "model": model_name or result.get("model", "unknown") if isinstance(result, dict) else "unknown",
                        "duration_ms": round(duration_ms, 2),
                        "tokens": tokens,
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    f"LLM call failed: {model_name or 'unknown'}",
                    extra={
                        "event": "llm_call_failed",
                        "model": model_name or "unknown",
                        "duration_ms": round(duration_ms, 2),
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                raise
        
        return async_wrapper
    
    return decorator


def track_database_query(collection_name: Optional[str] = None, operation: Optional[str] = None):
    """
    Decorator to track database query performance
    
    Args:
        collection_name: Name of the database collection
        operation: Type of operation (find, insert, update, delete)
    
    Usage:
        @track_database_query("users", "find")
        async def find_user(user_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Count results if it's a list
                result_count = len(result) if isinstance(result, list) else 1 if result else 0
                
                logger.info(
                    f"Database query completed: {collection_name or 'unknown'}.{operation or 'query'}",
                    extra={
                        "event": "database_query_completed",
                        "collection": collection_name or "unknown",
                        "operation": operation or "query",
                        "duration_ms": round(duration_ms, 2),
                        "result_count": result_count,
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    f"Database query failed: {collection_name or 'unknown'}.{operation or 'query'}",
                    extra={
                        "event": "database_query_failed",
                        "collection": collection_name or "unknown",
                        "operation": operation or "query",
                        "duration_ms": round(duration_ms, 2),
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                
                raise
        
        return async_wrapper
    
    return decorator


# Import asyncio for coroutine check
import asyncio


class PerformanceMetrics:
    """
    Context manager for tracking performance metrics
    
    Usage:
        async with PerformanceMetrics("complex_operation") as metrics:
            # Do work
            metrics.add_metadata("items_processed", 100)
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.metadata = {}
    
    async def __aenter__(self):
        self.start_time = time.time()
        logger.debug(
            f"Operation started: {self.operation_name}",
            extra={
                "event": "operation_started",
                "operation": self.operation_name
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            logger.info(
                f"Operation completed: {self.operation_name}",
                extra={
                    "event": "operation_completed",
                    "operation": self.operation_name,
                    "duration_ms": round(duration_ms, 2),
                    "status": "success",
                    **self.metadata
                }
            )
        else:
            logger.error(
                f"Operation failed: {self.operation_name}",
                extra={
                    "event": "operation_failed",
                    "operation": self.operation_name,
                    "duration_ms": round(duration_ms, 2),
                    "status": "error",
                    "error": str(exc_val),
                    "error_type": exc_type.__name__,
                    **self.metadata
                }
            )
        
        return False  # Don't suppress exceptions
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to be logged with the operation"""
        self.metadata[key] = value

