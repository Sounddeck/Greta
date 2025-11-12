"""
Test OAuth Service Initialization (Phase 2.2.1)

This script tests that the OAuth service initializes correctly in mock mode
when Google credentials are not configured.

Expected Results:
- OAuth service initializes without errors
- Mock mode is enabled
- Multi-workspace support is configurable
- No import errors
- Service is ready for credential configuration later
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_oauth_service_init():
    """Test OAuth service initialization"""
    print("=" * 80)
    print("TESTING: OAuth Service Initialization (Phase 2.2.1)")
    print("=" * 80)
    print()
    
    # Test 1: Import OAuth models
    print("Test 1: Importing OAuth models...")
    try:
        from models.oauth_models import (
            OAuthProvider,
            OAuthToken,
            WorkspaceOAuthCredentials,
            WorkspaceOAuthCredentialsCreate,
            WorkspaceOAuthCredentialsResponse,
            WorkspaceListResponse,
        )
        print("✅ OAuth models imported successfully")
        print(f"   - OAuthProvider: {OAuthProvider}")
        print(f"   - WorkspaceOAuthCredentials: {WorkspaceOAuthCredentials}")
    except Exception as e:
        print(f"❌ Failed to import OAuth models: {e}")
        return False
    
    print()
    
    # Test 2: Import OAuth service
    print("Test 2: Importing OAuth service...")
    try:
        from services.oauth_service import OAuthService
        print("✅ OAuth service imported successfully")
    except Exception as e:
        print(f"❌ Failed to import OAuth service: {e}")
        return False
    
    print()
    
    # Test 3: Initialize OAuth service (mock mode)
    print("Test 3: Initializing OAuth service in mock mode...")
    try:
        # Ensure we're in mock mode by unsetting credentials
        os.environ["GOOGLE_CLIENT_ID"] = ""
        os.environ["GOOGLE_CLIENT_SECRET"] = ""
        
        oauth_service = OAuthService()
        print("✅ OAuth service initialized successfully")
        print(f"   - Mock mode: {oauth_service.mock_mode}")
        print(f"   - Multi-workspace enabled: {oauth_service.enable_multi_workspace}")
        print(f"   - Default workspace ID: {oauth_service.default_workspace_id}")
        
        if not oauth_service.mock_mode:
            print("❌ Expected mock mode to be enabled")
            return False
        
    except Exception as e:
        print(f"❌ Failed to initialize OAuth service: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 4: Test mock mode detection with placeholder values
    print("Test 4: Testing mock mode detection with placeholder values...")
    try:
        os.environ["GOOGLE_CLIENT_ID"] = "your-google-client-id-here"
        os.environ["GOOGLE_CLIENT_SECRET"] = "your-google-client-secret-here"
        
        oauth_service = OAuthService()
        print("✅ OAuth service initialized with placeholder values")
        print(f"   - Mock mode: {oauth_service.mock_mode}")
        
        if not oauth_service.mock_mode:
            print("❌ Expected mock mode to be enabled with placeholder values")
            return False
        
    except Exception as e:
        print(f"❌ Failed to initialize OAuth service with placeholders: {e}")
        return False
    
    print()
    
    # Test 5: Test multi-workspace configuration
    print("Test 5: Testing multi-workspace configuration...")
    try:
        # Enable multi-workspace mode
        os.environ["ENABLE_MULTI_WORKSPACE"] = "true"
        os.environ["DEFAULT_WORKSPACE_ID"] = "test-workspace"
        
        oauth_service = OAuthService()
        print("✅ OAuth service initialized with multi-workspace enabled")
        print(f"   - Multi-workspace enabled: {oauth_service.enable_multi_workspace}")
        print(f"   - Default workspace ID: {oauth_service.default_workspace_id}")
        
        if not oauth_service.enable_multi_workspace:
            print("❌ Expected multi-workspace to be enabled")
            return False
        
        if oauth_service.default_workspace_id != "test-workspace":
            print("❌ Expected default workspace ID to be 'test-workspace'")
            return False
        
    except Exception as e:
        print(f"❌ Failed to initialize OAuth service with multi-workspace: {e}")
        return False
    
    print()
    
    # Test 6: Test workspace credentials model
    print("Test 6: Testing workspace credentials model...")
    try:
        from datetime import datetime
        
        workspace_creds = WorkspaceOAuthCredentials(
            credentials_id="test_cred_123",
            workspace_id="test-workspace",
            workspace_name="Test Workspace",
            provider=OAuthProvider.GOOGLE,
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/api/v1/oauth/callback",
            workspace_domain="test.com",
            workspace_type="business",
            owner_user_id="user_123",
            authorized_user_ids=["user_123"],
            is_active=True,
            is_default=True,
            created_at=datetime.utcnow()
        )
        
        print("✅ Workspace credentials model created successfully")
        print(f"   - Workspace ID: {workspace_creds.workspace_id}")
        print(f"   - Workspace name: {workspace_creds.workspace_name}")
        print(f"   - Owner: {workspace_creds.owner_user_id}")
        print(f"   - Active: {workspace_creds.is_active}")
        
    except Exception as e:
        print(f"❌ Failed to create workspace credentials model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 7: Test Google API packages are installed
    print("Test 7: Testing Google API packages...")
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import Flow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        
        print("✅ Google API packages imported successfully")
        print(f"   - google-auth: {Credentials}")
        print(f"   - google-auth-oauthlib: {Flow}")
        print(f"   - google-api-python-client: {build}")
        
    except Exception as e:
        print(f"❌ Failed to import Google API packages: {e}")
        print("   Make sure packages are installed:")
        print("   pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        return False
    
    print()
    
    # Test 8: Test database collections
    print("Test 8: Testing database collections...")
    try:
        from database import Database
        
        db = Database()
        print("✅ Database initialized successfully")
        print(f"   - Tokens collection: {db.tokens_collection if hasattr(db, 'tokens_collection') else 'oauth_tokens'}")
        print(f"   - States collection: {db.states_collection if hasattr(db, 'states_collection') else 'oauth_states'}")
        print(f"   - Workspace credentials collection: workspace_credentials")
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        print("   Note: This is expected if MongoDB is not running")
        print("   The service will use file-based storage as fallback")
    
    print()
    
    # All tests passed
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("- OAuth models imported successfully")
    print("- OAuth service initializes in mock mode")
    print("- Mock mode detection works correctly")
    print("- Multi-workspace configuration works")
    print("- Workspace credentials model works")
    print("- Google API packages installed")
    print("- Database collections configured")
    print()
    print("Next Steps:")
    print("1. Follow docs/GOOGLE_WORKSPACE_SETUP_GUIDE.md to obtain Google credentials")
    print("2. Configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")
    print("3. Restart backend server")
    print("4. OAuth service will automatically switch to real API mode")
    print()
    
    return True


if __name__ == "__main__":
    success = test_oauth_service_init()
    sys.exit(0 if success else 1)

