"""
GRETA PAI - Error Handling Tests
Phase 3 Code Quality Testing
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from . import APIClient, async_test, validation_exception, mock_greta_exception
from utils.error_handling import (
    GretaException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    RateLimitError,
    ExternalServiceError,
    ConfigurationError,
    ErrorHandler,
    ErrorContext,
    handle_errors,
    error_handler,
    metrics_collector
)


class TestGretaExceptions:
    """Test custom exception classes"""

    def test_greta_exception_basic(self):
        """Test basic Greta exception creation"""
        exc = GretaException("Test error", "TEST_CODE", 400, {"key": "value"})

        assert exc.message == "Test error"
        assert exc.error_code == "TEST_CODE"
        assert exc.status_code == 400
        assert exc.details == {"key": "value"}
        assert isinstance(exc.timestamp, datetime)

    def test_validation_error(self):
        """Test validation error with field details"""
        exc = ValidationError("Invalid value", field="email", value="invalid@")

        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.status_code == 400
        assert exc.details["field"] == "email"
        assert exc.details["value"] == "invalid@"

    def test_authentication_error(self):
        """Test authentication error"""
        exc = AuthenticationError("Token expired")

        assert exc.error_code == "AUTH_ERROR"
        assert exc.status_code == 401

    def test_authorization_error(self):
        """Test authorization error with scopes"""
        exc = AuthorizationError("Access denied", required_scopes=["admin", "write"])

        assert exc.error_code == "AUTHZ_ERROR"
        assert exc.status_code == 403
        assert "admin" in exc.details["required_scopes"]

    def test_resource_not_found_error(self):
        """Test resource not found error"""
        exc = ResourceNotFoundError("user", "12345")

        assert exc.error_code == "NOT_FOUND"
        assert exc.status_code == 404
        assert "user not found: 12345" in str(exc)

    def test_rate_limit_error(self):
        """Test rate limiting error"""
        exc = RateLimitError("Too many requests", retry_after=30)

        assert exc.error_code == "RATE_LIMIT"
        assert exc.status_code == 429
        assert exc.details["retry_after"] == 30


class TestErrorHandler:
    """Test error handling and response formatting"""

    def test_format_greta_exception_response(self, mock_greta_exception):
        """Test formatting Greta exception into JSON response"""
        response = ErrorHandler.format_error_response(mock_greta_exception)

        assert response.status_code == 400
        data = json.loads(response.body)

        assert data["success"] is False
        assert data["error"] == "Test error"
        assert data["error_code"] == "TEST_ERROR"
        assert "timestamp" in data
        assert data["details"]["test_field"] == "test_value"

    def test_format_generic_exception_response(self):
        """Test formatting generic exception"""
        exc = ValueError("Generic error")
        response = ErrorHandler.format_error_response(exc)

        assert response.status_code == 500
        data = json.loads(response.body)

        assert data["success"] is False
        assert data["error"] == "An unexpected error occurred"
        assert data["error_code"] == "INTERNAL_ERROR"

    def test_format_with_request_context(self, mock_greta_exception):
        """Test error formatting with request context"""
        from fastapi import Request
        from unittest.mock import Mock

        # Mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = "http://localhost:8000/api/test"
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "test-agent"}

        response = ErrorHandler.format_error_response(mock_greta_exception, request)

        assert response.status_code == 400
        data = json.loads(response.body)

        assert "request_id" in data
        assert data["path"] == "http://localhost:8000/api/test"


class TestErrorContext:
    """Test error context manager"""

    @pytest.mark.asyncio
    async def test_error_context_success(self):
        """Test successful operation in error context"""
        logs = []

        # Mock logger to capture logs
        import utils.error_handling
        original_debug = utils.error_handling.logger.debug
        original_debug = utils.error_handling.logger.debug
        utils.error_handling.logger.debug = Mock(side_effect=lambda *args, **kwargs: logs.append(args))

        try:
            async with ErrorContext("test_operation", {"key": "value"}):
                assert True  # Operation succeeds

            # Should have logged start and completion
            assert len(logs) >= 2
        finally:
            utils.error_handling.logger.debug = original_debug

    @pytest.mark.asyncio
    async def test_error_context_exception(self):
        """Test exception handling in error context"""
        logs = []

        # Mock logger to capture logs
        import utils.error_handling
        original_error = utils.error_handling.logger.error
        utils.error_handling.logger.error = Mock(side_effect=lambda *args, **kwargs: logs.append(args))

        try:
            with pytest.raises(ValueError):
                async with ErrorContext("failing_operation"):
                    raise ValueError("Test exception")

            # Should have logged the error
            assert len(logs) > 0
        finally:
            utils.error_handling.logger.error = original_error


class TestMetricsCollector:
    """Test metrics collection functionality"""

    def test_record_error(self):
        """Test error recording"""
        metrics_collector.record_error("TEST_ERROR", {"severity": "high"})

        assert metrics_collector.error_counts["TEST_ERROR"] == 1
        assert len(metrics_collector.error_samples["TEST_ERROR"]) == 1

        sample = metrics_collector.error_samples["TEST_ERROR"][0]
        assert sample["details"]["severity"] == "high"
        assert "timestamp" in sample

    def test_record_performance(self):
        """Test performance metric recording"""
        metrics_collector.record_performance("test_operation", 1.5)

        assert len(metrics_collector.performance_metrics["test_operation"]) == 1
        assert metrics_collector.performance_metrics["test_operation"][0] == 1.5

    def test_get_stats(self):
        """Test statistics retrieval"""
        # Add some test data
        metrics_collector.record_error("TEST_ERROR", {"count": 1})
        metrics_collector.record_performance("test_op", 2.0)
        metrics_collector.record_performance("test_op", 4.0)

        stats = metrics_collector.get_stats()

        assert "error_counts" in stats
        assert stats["error_counts"]["TEST_ERROR"] == 1

        assert "performance_stats" in stats
        test_op_stats = stats["performance_stats"]["test_op"]
        assert test_op_stats["count"] == 2
        assert test_op_stats["avg_duration"] == 3.0
        assert test_op_stats["min_duration"] == 2.0
        assert test_op_stats["max_duration"] == 4.0


class TestHandleErrorsDecorator:
    """Test error handling decorator"""

    def test_sync_function_success(self):
        """Test decorator on successful sync function"""
        @handle_errors("test_sync")
        def successful_function(x, y):
            return x + y

        result = successful_function(2, 3)
        assert result == 5

    def test_sync_function_exception(self):
        """Test decorator on sync function that raises exception"""
        @handle_errors("test_sync_fail")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(GretaException) as exc_info:
            failing_function()

        assert "Unexpected error in" in str(exc_info.value)
        assert exc_info.value.error_code == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """Test decorator on successful async function"""
        @handle_errors("test_async")
        async def successful_async_function(x, y):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y

        result = await successful_async_function(3, 4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_async_function_exception(self):
        """Test decorator on async function that raises exception"""
        @handle_errors("test_async_fail")
        async def failing_async_function():
            await asyncio.sleep(0.01)  # Simulate async work
            raise ConnectionError("Network error")

        with pytest.raises(GretaException) as exc_info:
            await failing_async_function()

        assert "Network error" in str(exc_info.value)
        assert exc_info.value.error_code == "INTERNAL_ERROR"


class TestIntegrationWithFastAPI:
    """Integration tests with FastAPI endpoints"""

    @pytest.mark.asyncio
    async def test_error_in_endpoint(self):
        """Test error handling in actual API endpoint"""
        from backend.main import app

        client = APIClient(app)

        # Test non-existent endpoint to trigger error
        response = client.get("/api/nonexistent")

        # Should get proper error response
        assert response.status_code == 404
        data = response.json()
        assert "success" in data
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_successful_endpoint(self):
        """Test successful endpoint response"""
        from backend.main import app

        client = APIClient(app)

        # Test root endpoint
        response = client.get("/")

        # Should work without errors
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
