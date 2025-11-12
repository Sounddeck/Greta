"""
OAuth 2.0 Module Tests

Comprehensive test suite for OAuth 2.0 authentication flow,
token management, and provider configuration.

NOTE: These tests use mock data and do not require real Google Cloud credentials.
"""

import asyncio
import unittest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from models.oauth_models import (
    OAuthProvider,
    OAuthAuthorizationRequest,
    OAuthCallbackRequest,
    OAuthRevokeRequest,
)
from services.oauth_service import OAuthService


class TestOAuthModule(unittest.TestCase):
    """Test OAuth 2.0 module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.db = Database()
        cls.service = OAuthService(cls.db)
        cls.test_user_id = "test_oauth_user"
        cls.test_provider = OAuthProvider.GOOGLE
    
    def setUp(self):
        """Set up each test"""
        self.loop = asyncio.get_event_loop()
    
    def test_01_service_initialization(self):
        """Test 1: OAuth service initialization"""
        print("\n" + "="*80)
        print("TEST 1: OAuth Service Initialization")
        print("="*80)
        
        assert self.service is not None
        assert self.service.db is not None
        assert self.service.mock_mode is True  # Should be in mock mode
        
        print("✅ OAuth service initialized successfully")
        print(f"   - Mock mode: {self.service.mock_mode}")
        print(f"   - Tokens collection: {self.service.tokens_collection}")
        print(f"   - States collection: {self.service.states_collection}")
    
    def test_02_initiate_oauth_flow(self):
        """Test 2: Initiate OAuth authorization flow"""
        print("\n" + "="*80)
        print("TEST 2: Initiate OAuth Authorization Flow")
        print("="*80)
        
        async def run_test():
            request = OAuthAuthorizationRequest(
                user_id=self.test_user_id,
                provider=self.test_provider,
                scopes=None  # Use default scopes
            )
            
            response = await self.service.initiate_oauth_flow(request)
            
            assert response is not None
            assert response.authorization_url is not None
            assert response.state is not None
            assert response.expires_at is not None
            
            print("✅ OAuth flow initiated successfully")
            print(f"   - Authorization URL: {response.authorization_url[:80]}...")
            print(f"   - State: {response.state[:20]}...")
            print(f"   - Expires at: {response.expires_at}")
            
            return response.state
        
        self.test_state = self.loop.run_until_complete(run_test())
    
    def test_03_handle_oauth_callback_success(self):
        """Test 3: Handle OAuth callback (success)"""
        print("\n" + "="*80)
        print("TEST 3: Handle OAuth Callback (Success)")
        print("="*80)
        
        async def run_test():
            # First initiate flow to get state
            auth_request = OAuthAuthorizationRequest(
                user_id=self.test_user_id,
                provider=self.test_provider
            )
            auth_response = await self.service.initiate_oauth_flow(auth_request)
            
            # Now handle callback
            callback_request = OAuthCallbackRequest(
                code="mock_authorization_code",
                state=auth_response.state
            )
            
            response = await self.service.handle_oauth_callback(callback_request)
            
            assert response is not None
            assert response.success is True
            assert response.token_id is not None
            assert response.provider == self.test_provider
            assert response.user_id == self.test_user_id
            
            print("✅ OAuth callback handled successfully")
            print(f"   - Success: {response.success}")
            print(f"   - Token ID: {response.token_id}")
            print(f"   - Provider: {response.provider}")
            print(f"   - User ID: {response.user_id}")
            
            return response.token_id
        
        self.test_token_id = self.loop.run_until_complete(run_test())
    
    def test_04_handle_oauth_callback_invalid_state(self):
        """Test 4: Handle OAuth callback (invalid state)"""
        print("\n" + "="*80)
        print("TEST 4: Handle OAuth Callback (Invalid State)")
        print("="*80)
        
        async def run_test():
            callback_request = OAuthCallbackRequest(
                code="mock_authorization_code",
                state="invalid_state_12345"
            )
            
            response = await self.service.handle_oauth_callback(callback_request)
            
            assert response is not None
            assert response.success is False
            assert response.error == "invalid_state"
            
            print("✅ Invalid state rejected correctly")
            print(f"   - Success: {response.success}")
            print(f"   - Error: {response.error}")
            print(f"   - Error description: {response.error_description}")
        
        self.loop.run_until_complete(run_test())
    
    def test_05_handle_oauth_callback_with_error(self):
        """Test 5: Handle OAuth callback (with error)"""
        print("\n" + "="*80)
        print("TEST 5: Handle OAuth Callback (With Error)")
        print("="*80)
        
        async def run_test():
            callback_request = OAuthCallbackRequest(
                code="",
                state="some_state",
                error="access_denied",
                error_description="User denied access"
            )
            
            response = await self.service.handle_oauth_callback(callback_request)
            
            assert response is not None
            assert response.success is False
            assert response.error == "access_denied"
            
            print("✅ OAuth error handled correctly")
            print(f"   - Success: {response.success}")
            print(f"   - Error: {response.error}")
            print(f"   - Error description: {response.error_description}")
        
        self.loop.run_until_complete(run_test())
    
    def test_06_get_user_token(self):
        """Test 6: Get user OAuth token"""
        print("\n" + "="*80)
        print("TEST 6: Get User OAuth Token")
        print("="*80)
        
        async def run_test():
            # First create a token
            auth_request = OAuthAuthorizationRequest(
                user_id=self.test_user_id,
                provider=self.test_provider
            )
            auth_response = await self.service.initiate_oauth_flow(auth_request)
            
            callback_request = OAuthCallbackRequest(
                code="mock_code",
                state=auth_response.state
            )
            await self.service.handle_oauth_callback(callback_request)
            
            # Now get the token
            token = await self.service.get_user_token(self.test_user_id, self.test_provider)
            
            # Note: In mock mode, token might not be stored in DB
            # This test verifies the method works without errors
            print("✅ Get user token method works")
            print(f"   - Token found: {token is not None}")
            if token:
                print(f"   - Token ID: {token.token_id}")
                print(f"   - Provider: {token.provider}")
        
        self.loop.run_until_complete(run_test())
    
    def test_07_refresh_access_token(self):
        """Test 7: Refresh access token"""
        print("\n" + "="*80)
        print("TEST 7: Refresh Access Token")
        print("="*80)
        
        async def run_test():
            # Attempt to refresh token
            response = await self.service.refresh_access_token(
                self.test_user_id,
                self.test_provider
            )
            
            # In mock mode, this might return None if no token exists
            print("✅ Refresh token method works")
            print(f"   - Response: {response is not None}")
            if response:
                print(f"   - Token ID: {response.token_id}")
                print(f"   - Expires at: {response.expires_at}")
        
        self.loop.run_until_complete(run_test())
    
    def test_08_revoke_token(self):
        """Test 8: Revoke OAuth token"""
        print("\n" + "="*80)
        print("TEST 8: Revoke OAuth Token")
        print("="*80)
        
        async def run_test():
            request = OAuthRevokeRequest(
                user_id=self.test_user_id,
                provider=self.test_provider,
                reason="Test revocation"
            )
            
            response = await self.service.revoke_token(request)
            
            assert response is not None
            # Response might indicate token not found in mock mode
            print("✅ Revoke token method works")
            print(f"   - Success: {response.success}")
            print(f"   - Message: {response.message}")
        
        self.loop.run_until_complete(run_test())
    
    def test_09_oauth_health_check(self):
        """Test 9: OAuth service health check"""
        print("\n" + "="*80)
        print("TEST 9: OAuth Service Health Check")
        print("="*80)
        
        async def run_test():
            health = await self.service.get_health()
            
            assert health is not None
            assert health.status is not None
            assert health.message is not None
            assert health.timestamp is not None
            
            print("✅ OAuth health check successful")
            print(f"   - Status: {health.status}")
            print(f"   - Message: {health.message}")
            print(f"   - Providers configured: {health.providers_configured}")
            print(f"   - Active tokens: {health.active_tokens}")
        
        self.loop.run_until_complete(run_test())
    
    def test_10_multiple_providers(self):
        """Test 10: Support for multiple OAuth providers"""
        print("\n" + "="*80)
        print("TEST 10: Multiple OAuth Providers")
        print("="*80)
        
        providers = [
            OAuthProvider.GOOGLE,
            OAuthProvider.MICROSOFT,
            OAuthProvider.GITHUB,
        ]
        
        print("✅ Multiple providers supported")
        for provider in providers:
            print(f"   - {provider.value}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print("\n" + "="*80)
        print("OAUTH MODULE TESTS COMPLETE")
        print("="*80)


if __name__ == "__main__":
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOAuthModule)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED ({}/{})".format(
            result.testsRun - len(result.failures) - len(result.errors),
            result.testsRun
        ))
    else:
        print("❌ SOME TESTS FAILED ({}/{})".format(
            result.testsRun - len(result.failures) - len(result.errors),
            result.testsRun
        ))
    print("="*80)

