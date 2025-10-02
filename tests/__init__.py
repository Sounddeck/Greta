"""
GRETA PAI - Test Framework
Phase 3 Code Quality - Comprehensive Testing Suite
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock
import os
import sys

# Add backend to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from utils.error_handling import GretaException, ValidationError, metrics_collector
from utils.performance import ContextWindowManager, MemoryStats


@pytest.fixture(autouse=True)
async def cleanup_test_state():
    """Clean up test state between tests"""
    # Reset metrics collector
    metrics_collector.error_counts.clear()
    metrics_collector.error_samples.clear()
    metrics_collector.performance_metrics.clear()

    yield

    # Post-test cleanup
    # Add any additional cleanup here


@pytest.fixture
def test_context_manager():
    """Provide a fresh context manager for testing"""
    return ContextWindowManager()


@pytest.fixture
def test_memory_stats():
    """Mock memory stats for testing"""
    stats = MemoryStats()
    stats.process_memory_mb = 50.0
    stats.virtual_memory_mb = 1024.0
    stats.available_memory_mb = 2048.0
    stats.memory_usage_percent = 25.0
    return stats


@pytest.fixture
def mock_greta_exception():
    """Create a mock Greta exception"""
    return GretaException(
        message="Test error",
        error_code="TEST_ERROR",
        status_code=400,
        details={"test_field": "test_value"}
    )


@pytest.fixture
def validation_exception():
    """Create a validation error"""
    return ValidationError(
        message="Invalid input provided",
        field="test_field",
        value="invalid_value"
    )


# Test configuration
TEST_CONFIG = {
    "test_database_url": "mongodb://localhost:27017/test_greta_db",
    "test_redis_url": "redis://localhost:6379/1",
    "test_log_level": "WARNING",  # Reduce log noise in tests
    "test_timeout": 30
}


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "security: marks tests as security-related")
    config.addinivalue_line("markers", "performance: marks tests as performance-related")


def async_test(coro):
    """Helper to run async tests"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


class MockedService:
    """Mock service for testing dependencies"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class APIClient:
    """Test client for API endpoints"""

    def __init__(self, app):
        self.app = app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def get(self, path: str, **kwargs):
        return self.client.get(path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.client.post(path, **kwargs)

    def put(self, path: str, **kwargs):
        return self.client.put(path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.client.delete(path, **kwargs)


class AsyncAPIClient:
    """Async test client for API endpoints"""

    def __init__(self, app):
        self.app = app
        from httpx import AsyncClient
        self.client = AsyncClient(app=app, base_url="http://testserver")

    async def get(self, path: str, **kwargs):
        async with self.client as client:
            return await client.get(path, **kwargs)

    async def post(self, path: str, **kwargs):
        async with self.client as client:
            return await client.post(path, **kwargs)

    async def put(self, path: str, **kwargs):
        async with self.client as client:
            return await client.put(path, **kwargs)

    async def delete(self, path: str, **kwargs):
        async with self.client as client:
            return await client.delete(path, **kwargs)
