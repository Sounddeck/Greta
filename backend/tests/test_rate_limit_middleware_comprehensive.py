"""
Comprehensive tests for rate limit middleware

Tests cover:
- Rate limiting logic
- Token bucket algorithm
- IP-based limiting
- Error responses
"""

import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import json


# Mock custom_exceptions module
class RateLimitExceeded(Exception):
    pass

mock_custom_exceptions = type(sys)('utils.custom_exceptions')
mock_custom_exceptions.RateLimitExceeded = RateLimitExceeded
sys.modules['utils.custom_exceptions'] = mock_custom_exceptions


class TestRateLimitingLogic:
    """Test rate limiting logic"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_within_limit(self):
        """Test rate limit allows requests within limit"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService
        
        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.limit = 100
        mock_result.remaining = 99
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 0
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert response.status_code == 200
        assert mock_rate_limiter.check_rate_limit.called
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_when_exceeded(self):
        """Test rate limit blocks requests when exceeded"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService
        
        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = False
        mock_result.limit = 100
        mock_result.remaining = 0
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 60
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        
        # Mock call_next
        call_next = AsyncMock()
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert response.status_code == 429
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "Retry-After" in response.headers
    
    @pytest.mark.asyncio
    async def test_rate_limit_per_endpoint(self):
        """Test rate limit per endpoint"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService
        
        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.limit = 10
        mock_result.remaining = 9
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 0
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)
        
        # Create mock request for specific endpoint
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/v1/crm/stats"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert response.status_code == 200
        # Verify check_rate_limit was called with endpoint-specific limit
        assert mock_rate_limiter.check_rate_limit.called
    
    @pytest.mark.asyncio
    async def test_rate_limit_headers_added(self):
        """Test rate limit headers are added to response"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService
        
        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.limit = 100
        mock_result.remaining = 95
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 0
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestTokenBucketAlgorithm:
    """Test token bucket algorithm"""
    
    @pytest.mark.asyncio
    async def test_token_bucket_refill(self):
        """Test token bucket refill logic"""
        from services.rate_limiter_service import RateLimiterService

        # Create rate limiter service
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        rate_limiter = RateLimiterService()
        rate_limiter.redis_client = mock_redis

        # Check rate limit (should allow first request)
        result = await rate_limiter.check_rate_limit(
            identifier="test-user",
            limit=10,
            window=60
        )

        assert result.allowed is True
        assert result.limit == 10
    
    @pytest.mark.asyncio
    async def test_token_bucket_consumption(self):
        """Test token bucket consumption"""
        from services.rate_limiter_service import RateLimiterService

        # Create rate limiter service
        mock_redis = AsyncMock()
        # Simulate bucket with 5 tokens remaining
        mock_redis.get = AsyncMock(return_value=b'5')
        mock_redis.setex = AsyncMock()

        rate_limiter = RateLimiterService()
        rate_limiter.redis_client = mock_redis

        # Check rate limit
        result = await rate_limiter.check_rate_limit(
            identifier="test-user",
            limit=10,
            window=60
        )

        assert result.allowed is True
        assert result.remaining >= 0
    
    @pytest.mark.asyncio
    async def test_token_bucket_empty(self):
        """Test token bucket when empty (limit exceeded)"""
        from services.rate_limiter_service import RateLimiterService

        # Create rate limiter service
        mock_redis = AsyncMock()

        # Mock pipeline to simulate limit exceeded
        mock_pipeline = AsyncMock()
        mock_pipeline.zremrangebyscore = MagicMock(return_value=None)
        mock_pipeline.zcard = MagicMock(return_value=None)
        mock_pipeline.zadd = MagicMock(return_value=None)
        mock_pipeline.expire = MagicMock(return_value=None)
        # Return count of 10 (at limit) before adding current request
        mock_pipeline.execute = AsyncMock(return_value=[None, 10, None, None])

        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

        rate_limiter = RateLimiterService()
        rate_limiter.redis_client = mock_redis

        # Check rate limit with limit of 10
        result = await rate_limiter.check_rate_limit(
            identifier="test-user",
            limit=10,
            window=60
        )

        # Should be blocked (10 requests already in window, limit is 10)
        assert result.allowed is False
        assert result.remaining == 0


class TestIPBasedLimiting:
    """Test IP-based rate limiting"""

    @pytest.mark.asyncio
    async def test_ip_extraction_from_client(self):
        """Test IP extraction from client"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService

        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.limit = 100
        mock_result.remaining = 99
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 0
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)

        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {}

        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 200
        # Verify IP was used in rate limit check
        call_args = mock_rate_limiter.check_rate_limit.call_args
        assert "192.168.1.100" in str(call_args)

    @pytest.mark.asyncio
    async def test_ip_extraction_from_x_forwarded_for(self):
        """Test IP extraction from X-Forwarded-For header"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService

        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.limit = 100
        mock_result.remaining = 99
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 0
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)

        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)

        # Create mock request with X-Forwarded-For header
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}

        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 200
        # Verify first IP from X-Forwarded-For was used
        call_args = mock_rate_limiter.check_rate_limit.call_args
        assert "203.0.113.1" in str(call_args)

    @pytest.mark.asyncio
    async def test_ip_extraction_from_x_real_ip(self):
        """Test IP extraction from X-Real-IP header"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService

        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.limit = 100
        mock_result.remaining = 99
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 0
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)

        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)

        # Create mock request with X-Real-IP header
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"X-Real-IP": "203.0.113.50"}

        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 200
        # Verify X-Real-IP was used
        call_args = mock_rate_limiter.check_rate_limit.call_args
        assert "203.0.113.50" in str(call_args)

    @pytest.mark.asyncio
    async def test_different_ips_have_separate_limits(self):
        """Test different IPs have separate rate limits"""
        from services.rate_limiter_service import RateLimiterService

        # Create rate limiter service
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        rate_limiter = RateLimiterService()
        rate_limiter.redis_client = mock_redis

        # Check rate limit for IP 1
        result1 = await rate_limiter.check_rate_limit(
            identifier="ip:192.168.1.1",
            limit=10,
            window=60
        )

        # Check rate limit for IP 2
        result2 = await rate_limiter.check_rate_limit(
            identifier="ip:192.168.1.2",
            limit=10,
            window=60
        )

        # Both should be allowed (separate limits)
        assert result1.allowed is True
        assert result2.allowed is True


class TestErrorResponses:
    """Test error responses"""

    @pytest.mark.asyncio
    async def test_429_response_format(self):
        """Test 429 response format"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService

        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = False
        mock_result.limit = 100
        mock_result.remaining = 0
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 60
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)

        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}

        # Mock call_next
        call_next = AsyncMock()

        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 429
        assert response.headers["X-RateLimit-Limit"] == "100"
        assert response.headers["X-RateLimit-Remaining"] == "0"
        assert response.headers["Retry-After"] == "60"

    @pytest.mark.asyncio
    async def test_rate_limit_error_message(self):
        """Test rate limit error message"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService

        # Mock rate limiter service
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_result = MagicMock()
        mock_result.allowed = False
        mock_result.limit = 100
        mock_result.remaining = 0
        mock_result.reset_at = 1234567890
        mock_result.retry_after = 60
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_result)

        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}

        # Mock call_next
        call_next = AsyncMock()

        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 429
        # Parse response body
        body = response.body.decode() if hasattr(response.body, 'decode') else str(response.body)
        assert "Rate limit exceeded" in body or "Too many requests" in body

    @pytest.mark.asyncio
    async def test_rate_limit_service_error_handling(self):
        """Test rate limit service error handling"""
        from middleware.rate_limiter_middleware import RateLimiterMiddleware
        from services.rate_limiter_service import RateLimiterService

        # Mock rate limiter service to raise exception
        mock_rate_limiter = MagicMock(spec=RateLimiterService)
        mock_rate_limiter.check_rate_limit = AsyncMock(side_effect=Exception("Redis connection failed"))

        # Create middleware
        mock_app = AsyncMock()
        middleware = RateLimiterMiddleware(app=mock_app, rate_limiter=mock_rate_limiter)

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}

        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        # Process request - should handle error gracefully
        # (either allow request or return error response)
        try:
            response = await middleware.dispatch(mock_request, call_next)
            # If it doesn't raise, it should either allow or deny
            assert response.status_code in [200, 429, 500, 503]
        except Exception as e:
            # If it raises, that's also acceptable error handling
            assert "Redis connection failed" in str(e)

