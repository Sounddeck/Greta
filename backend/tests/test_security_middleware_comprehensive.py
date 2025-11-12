"""
Comprehensive tests for security middleware

Tests cover:
- Authentication and authorization
- Security headers
- Input validation
- Error handling
"""

import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta

# Mock jose module before importing middleware.auth
class MockJWTError(Exception):
    pass

class MockJWT:
    @staticmethod
    def encode(payload, key, algorithm):
        return "mock.jwt.token"

    @staticmethod
    def decode(token, key, algorithms):
        if token == "invalid_token" or token == "expired.token.here":
            raise MockJWTError("Invalid token")
        return {
            "user_id": "test-123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"]
        }

mock_jose = type(sys)('jose')
mock_jose.jwt = MockJWT
mock_jose.JWTError = MockJWTError
sys.modules['jose'] = mock_jose

# Mock custom_exceptions module
class ValidationException(Exception):
    pass

class InvalidInputError(Exception):
    pass

mock_custom_exceptions = type(sys)('utils.custom_exceptions')
mock_custom_exceptions.ValidationException = ValidationException
mock_custom_exceptions.InvalidInputError = InvalidInputError
sys.modules['utils.custom_exceptions'] = mock_custom_exceptions

# Mock passlib module
class MockCryptContext:
    def __init__(self, schemes, deprecated):
        pass

    def hash(self, password):
        return f"hashed_{password}"

    def verify(self, password, hashed):
        return hashed == f"hashed_{password}"

mock_passlib_context = type(sys)('passlib.context')
mock_passlib_context.CryptContext = MockCryptContext
mock_passlib = type(sys)('passlib')
mock_passlib.context = mock_passlib_context
sys.modules['passlib'] = mock_passlib
sys.modules['passlib.context'] = mock_passlib_context


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization"""
    
    @pytest.mark.asyncio
    async def test_jwt_token_validation_success(self):
        """Test successful JWT token validation"""
        from middleware.auth import decode_access_token, create_access_token
        
        # Create a valid token
        user_data = {
            "user_id": "test-123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"]
        }
        
        token = create_access_token(user_data)
        
        # Decode and validate
        token_data = decode_access_token(token)
        
        assert token_data.user_id == "test-123"
        assert token_data.username == "testuser"
        assert token_data.email == "test@example.com"
        assert "user" in token_data.roles
    
    @pytest.mark.asyncio
    async def test_authentication_header_parsing(self):
        """Test parsing authentication header"""
        from middleware.auth import verify_token
        from fastapi.security import HTTPAuthorizationCredentials
        
        # Create a valid token
        from middleware.auth import create_access_token
        user_data = {"user_id": "test-123", "username": "testuser"}
        token = create_access_token(user_data)
        
        # Create credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        # Verify token
        token_data = await verify_token(credentials)
        
        assert token_data.user_id == "test-123"
        assert token_data.username == "testuser"
    
    @pytest.mark.asyncio
    async def test_authorization_checks_for_protected_routes(self):
        """Test authorization checks for protected routes"""
        from middleware.security_enhanced import IPFilterMiddleware

        # Mock security service
        mock_security_service = MagicMock()
        mock_security_service.is_ip_allowed = AsyncMock(return_value=True)

        # Create middleware
        mock_app = AsyncMock()
        middleware = IPFilterMiddleware(
            app=mock_app,
            security_service=mock_security_service
        )

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/protected"
        mock_request.method = "GET"

        # Mock call_next
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 200
        assert mock_security_service.is_ip_allowed.called
    
    @pytest.mark.asyncio
    async def test_handle_missing_invalid_tokens(self):
        """Test handling missing or invalid tokens"""
        from middleware.auth import decode_access_token
        
        # Test invalid token
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("invalid_token")
        
        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in exc_info.value.detail


class TestSecurityHeaders:
    """Test security headers"""
    
    @pytest.mark.asyncio
    async def test_security_headers_added_to_responses(self):
        """Test security headers are added to responses"""
        from middleware.security_middleware import SecurityHeadersMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = SecurityHeadersMiddleware(app=mock_app)
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        
        # Create mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        # Verify security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers
    
    @pytest.mark.asyncio
    async def test_cors_headers_configuration(self):
        """Test CORS headers configuration"""
        from middleware.security_middleware import CORSMiddleware
        
        # Get CORS config
        cors_config = CORSMiddleware.get_cors_config()
        
        assert "allow_origins" in cors_config
        assert "allow_credentials" in cors_config
        assert cors_config["allow_credentials"] is True
        assert "allow_methods" in cors_config
        assert "GET" in cors_config["allow_methods"]
        assert "POST" in cors_config["allow_methods"]
    
    @pytest.mark.asyncio
    async def test_content_security_policy_headers(self):
        """Test Content-Security-Policy headers"""
        from middleware.security_middleware import SecurityHeadersMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = SecurityHeadersMiddleware(app=mock_app)
        
        # Create mock request and response
        mock_request = MagicMock(spec=Request)
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        # Verify CSP header
        assert "Content-Security-Policy" in response.headers
        csp = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
    
    @pytest.mark.asyncio
    async def test_x_frame_options_and_other_security_headers(self):
        """Test X-Frame-Options and other security headers"""
        from middleware.security_middleware import SecurityHeadersMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = SecurityHeadersMiddleware(app=mock_app)
        
        # Create mock request and response
        mock_request = MagicMock(spec=Request)
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        # Verify headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "1; mode=block" in response.headers["X-XSS-Protection"]


class TestInputValidation:
    """Test input validation"""
    
    @pytest.mark.asyncio
    async def test_request_body_validation(self):
        """Test request body validation"""
        from middleware.security_middleware import InputValidationMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = InputValidationMiddleware(app=mock_app, enabled=True)
        
        # Create mock request with valid JSON
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.body = AsyncMock(return_value=b'{"name": "test"}')
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_query_parameter_sanitization(self):
        """Test query parameter sanitization"""
        from middleware.security_middleware import InputValidationMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = InputValidationMiddleware(app=mock_app, enabled=True)
        
        # Create mock request with query params
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.query_params = {"search": "test query"}
        mock_request.headers = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_path_parameter_validation(self):
        """Test path parameter validation"""
        from middleware.security_middleware import InputValidationMiddleware
        
        # Create middleware
        mock_app = AsyncMock()
        middleware = InputValidationMiddleware(app=mock_app, enabled=True)
        
        # Create mock request with path params
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/users/123"
        mock_request.path_params = {"user_id": "123"}
        mock_request.headers = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        
        # Mock response
        mock_response = Response(content="OK", status_code=200)
        call_next = AsyncMock(return_value=mock_response)
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_malicious_input(self):
        """Test handling malicious input"""
        from middleware.security_middleware import InputValidationMiddleware

        # Create middleware
        mock_app = AsyncMock()
        middleware = InputValidationMiddleware(app=mock_app, enabled=True)

        # Create mock request with malicious JSON
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.body = AsyncMock(return_value=b'invalid json')
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        # Process request - should raise HTTPException
        call_next = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(mock_request, call_next)

        assert exc_info.value.status_code == 400


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_authentication_failure_responses(self):
        """Test authentication failure responses"""
        from middleware.auth import decode_access_token

        # Test expired token
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("expired.token.here")

        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_authorization_failure_responses(self):
        """Test authorization failure responses"""
        from middleware.security_enhanced import IPFilterMiddleware

        # Mock security service to deny access
        mock_security_service = MagicMock()
        mock_security_service.is_ip_allowed = AsyncMock(return_value=False)
        mock_security_service.log_security_event = AsyncMock()

        # Create middleware
        mock_app = AsyncMock()
        middleware = IPFilterMiddleware(
            app=mock_app,
            security_service=mock_security_service
        )

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/protected"
        mock_request.method = "GET"

        # Mock call_next
        call_next = AsyncMock()

        # Process request
        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 403
        assert mock_security_service.log_security_event.called

    @pytest.mark.asyncio
    async def test_validation_error_responses(self):
        """Test validation error responses"""
        from middleware.security_middleware import InputValidationMiddleware

        # Create middleware
        mock_app = AsyncMock()
        middleware = InputValidationMiddleware(app=mock_app, enabled=True)

        # Create mock request with invalid JSON
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.body = AsyncMock(return_value=b'{"invalid": json}')
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        # Mock call_next
        call_next = AsyncMock()

        # Process request - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(mock_request, call_next)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_security_exception_handling(self):
        """Test security exception handling"""
        from middleware.security_enhanced import IPFilterMiddleware

        # Mock security service to raise exception
        mock_security_service = MagicMock()
        mock_security_service.is_ip_allowed = AsyncMock(side_effect=Exception("Security check failed"))
        mock_security_service.log_security_event = AsyncMock()

        # Create middleware
        mock_app = AsyncMock()
        middleware = IPFilterMiddleware(
            app=mock_app,
            security_service=mock_security_service
        )

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/api/protected"
        mock_request.method = "GET"

        # Mock call_next
        call_next = AsyncMock()

        # Process request - should handle exception gracefully
        with pytest.raises(Exception) as exc_info:
            await middleware.dispatch(mock_request, call_next)

        assert "Security check failed" in str(exc_info.value)

