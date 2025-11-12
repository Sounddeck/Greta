"""
Comprehensive tests for logging middleware

Tests cover:
- Request/response logging
- Performance metrics
- Error logging
- Log formatting
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request, Response
import time


class TestRequestResponseLogging:
    """Test request and response logging"""
    
    @pytest.mark.asyncio
    async def test_request_logging(self):
        """Test request logging"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            assert response.status_code == 200
            assert mock_logger.info.called
    
    @pytest.mark.asyncio
    async def test_response_logging(self):
        """Test response logging"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "POST"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock response
        mock_response = Response(content="Created", status_code=201)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            assert response.status_code == 201
            # Verify both request started and completed logs
            assert mock_logger.info.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_correlation_id_added(self):
        """Test correlation ID is added to response"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert "X-Correlation-ID" in response.headers
        assert "X-Request-Duration-Ms" in response.headers
    
    @pytest.mark.asyncio
    async def test_excluded_paths_not_logged(self):
        """Test excluded paths are not logged"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware with excluded paths
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(
            app=mock_app,
            exclude_paths=["/health", "/metrics"]
        )
        
        # Create mock request for excluded path
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/health"
        mock_request.method = "GET"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            assert response.status_code == 200
            # Should not log excluded paths
            assert not mock_logger.info.called


class TestPerformanceMetrics:
    """Test performance metrics"""
    
    @pytest.mark.asyncio
    async def test_request_duration_tracking(self):
        """Test request duration tracking"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock response with delay
        async def delayed_response(request):
            await AsyncMock()()
            return Response(content="OK", status_code=200)

        call_next = delayed_response
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert "X-Request-Duration-Ms" in response.headers
        duration = float(response.headers["X-Request-Duration-Ms"])
        assert duration >= 0
    
    @pytest.mark.asyncio
    async def test_slow_request_detection(self):
        """Test slow request detection"""
        from middleware.logging_middleware import PerformanceLoggingMiddleware
        
        # Create middleware with low threshold
        mock_app = AsyncMock()
        middleware = PerformanceLoggingMiddleware(
            app=mock_app,
            slow_request_threshold_ms=10.0
        )
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/slow"
        mock_request.method = "GET"
        
        # Mock slow response
        async def slow_response(request):
            import asyncio
            await asyncio.sleep(0.02)  # 20ms delay
            return Response(content="OK", status_code=200)

        call_next = slow_response
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            assert response.status_code == 200
            # Should log warning for slow request
            assert mock_logger.warning.called
    
    @pytest.mark.asyncio
    async def test_performance_metrics_in_logs(self):
        """Test performance metrics are included in logs"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            # Check that duration_ms is in log extra data
            call_args = mock_logger.info.call_args_list
            assert len(call_args) >= 2
            # Check completed log has duration_ms
            completed_log = call_args[-1]
            assert 'extra' in completed_log.kwargs
            assert 'duration_ms' in completed_log.kwargs['extra']


class TestErrorLogging:
    """Test error logging"""
    
    @pytest.mark.asyncio
    async def test_exception_logging(self):
        """Test exception logging"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/error"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock call_next to raise exception
        call_next = AsyncMock(side_effect=Exception("Test error"))
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            with pytest.raises(Exception):
                await middleware.dispatch(mock_request, call_next)
            
            # Should log the error
            assert mock_logger.info.called  # Request started
    
    @pytest.mark.asyncio
    async def test_error_response_logging(self):
        """Test error response logging"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock error response
        mock_response = Response(content="Error", status_code=500)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            assert response.status_code == 500
            assert mock_logger.info.called


class TestLogFormatting:
    """Test log formatting"""
    
    @pytest.mark.asyncio
    async def test_log_includes_method_and_path(self):
        """Test log includes method and path"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/users/123"
        mock_request.method = "PUT"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            # Check log message includes method and path
            call_args = mock_logger.info.call_args_list
            assert len(call_args) >= 2
            first_log_message = call_args[0].args[0]
            assert "PUT" in first_log_message
            assert "/api/users/123" in first_log_message
    
    @pytest.mark.asyncio
    async def test_log_includes_client_ip(self):
        """Test log includes client IP"""
        from middleware.logging_middleware import LoggingMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = LoggingMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "203.0.113.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)
            
            # Check log extra data includes client_ip
            call_args = mock_logger.info.call_args_list
            assert len(call_args) >= 1
            first_log = call_args[0]
            assert 'extra' in first_log.kwargs
            assert 'client_ip' in first_log.kwargs['extra']
            assert first_log.kwargs['extra']['client_ip'] == "203.0.113.1"

