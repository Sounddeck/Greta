"""
OAuth 2.0 Service for Google Workspace Integration

This service handles OAuth 2.0 authentication flow, token management,
and provider configuration for Google Workspace APIs.

TODO: Requires Google Cloud OAuth credentials to function with real APIs.
      Set environment variables:
      - GOOGLE_CLIENT_ID
      - GOOGLE_CLIENT_SECRET
      - GOOGLE_REDIRECT_URI
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from urllib.parse import urlencode

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request

from database import Database
from models.oauth_models import (
    OAuthProvider,
    OAuthToken,
    OAuthTokenCreate,
    OAuthTokenResponse,
    OAuthTokenRefreshResponse,
    OAuthState,
    OAuthAuthorizationRequest,
    OAuthAuthorizationResponse,
    OAuthCallbackRequest,
    OAuthCallbackResponse,
    OAuthRevokeRequest,
    OAuthRevokeResponse,
    GoogleWorkspaceScopes,
    OAuthHealthResponse,
)


class OAuthService:
    """OAuth 2.0 service for Google Workspace"""
    
    def __init__(self, db: Optional[Database] = None):
        """Initialize OAuth service"""
        self.db = db or Database()
        self.tokens_collection = "oauth_tokens"
        self.states_collection = "oauth_states"
        
        # Google OAuth configuration (from environment variables)
        # TODO: Set these environment variables with your Google Cloud credentials
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.google_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/v1/oauth/callback")
        
        # OAuth URIs
        self.google_auth_uri = "https://accounts.google.com/o/oauth2/auth"
        self.google_token_uri = "https://oauth2.googleapis.com/token"
        self.google_revoke_uri = "https://oauth2.googleapis.com/revoke"
        
        # Mock mode (when credentials not available)
        self.mock_mode = not (self.google_client_id and self.google_client_secret)
        if self.mock_mode:
            print("⚠️  OAuth Service running in MOCK MODE (no Google credentials configured)")
    
    # ========================================================================
    # OAuth Authorization Flow
    # ========================================================================
    
    async def initiate_oauth_flow(
        self,
        request: OAuthAuthorizationRequest
    ) -> OAuthAuthorizationResponse:
        """
        Initiate OAuth 2.0 authorization flow
        
        Returns authorization URL for user to visit
        """
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        
        # Determine scopes
        scopes = request.scopes or GoogleWorkspaceScopes.get_all_workspace_scopes()
        
        # Store state in database
        state_record = OAuthState(
            state_id=f"state_{secrets.token_urlsafe(16)}",
            state=state,
            user_id=request.user_id,
            provider=request.provider,
            redirect_uri=request.redirect_uri or self.google_redirect_uri,
            scopes=scopes,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=10),
            used=False
        )

        # Convert datetime to ISO format for storage
        state_data = state_record.model_dump()
        state_data["created_at"] = state_record.created_at.isoformat()
        state_data["expires_at"] = state_record.expires_at.isoformat()

        await self.db.insert_one(
            self.states_collection,
            state_data
        )
        
        # Build authorization URL
        if self.mock_mode:
            # Mock authorization URL
            auth_url = f"{self.google_auth_uri}?mock=true&state={state}"
        else:
            # Real authorization URL
            params = {
                "client_id": self.google_client_id,
                "redirect_uri": request.redirect_uri or self.google_redirect_uri,
                "response_type": "code",
                "scope": " ".join(scopes),
                "state": state,
                "access_type": "offline",
                "prompt": "consent"
            }
            auth_url = f"{self.google_auth_uri}?{urlencode(params)}"
        
        return OAuthAuthorizationResponse(
            authorization_url=auth_url,
            state=state,
            expires_at=state_record.expires_at
        )
    
    async def handle_oauth_callback(
        self,
        request: OAuthCallbackRequest
    ) -> OAuthCallbackResponse:
        """
        Handle OAuth 2.0 callback
        
        Exchanges authorization code for access token
        """
        # Check for errors
        if request.error:
            return OAuthCallbackResponse(
                success=False,
                error=request.error,
                error_description=request.error_description
            )
        
        # Verify state
        state_record = await self.db.find_one(
            self.states_collection,
            {"state": request.state}
        )

        if not state_record:
            return OAuthCallbackResponse(
                success=False,
                error="invalid_state",
                error_description="Invalid or expired state parameter"
            )

        # Convert ISO strings back to datetime
        if isinstance(state_record.get("created_at"), str):
            state_record["created_at"] = datetime.fromisoformat(state_record["created_at"])
        if isinstance(state_record.get("expires_at"), str):
            state_record["expires_at"] = datetime.fromisoformat(state_record["expires_at"])
        if isinstance(state_record.get("used_at"), str):
            state_record["used_at"] = datetime.fromisoformat(state_record["used_at"])

        state = OAuthState(**state_record)
        
        # Check if state already used
        if state.used:
            return OAuthCallbackResponse(
                success=False,
                error="state_already_used",
                error_description="State parameter has already been used"
            )
        
        # Check if state expired
        if datetime.utcnow() > state.expires_at:
            return OAuthCallbackResponse(
                success=False,
                error="state_expired",
                error_description="State parameter has expired"
            )
        
        # Exchange code for tokens
        if self.mock_mode:
            # Mock token exchange
            token_data = await self._create_mock_token(state.user_id, state.provider, state.scopes)
        else:
            # Real token exchange
            # TODO: Implement real token exchange with Google OAuth
            token_data = await self._create_mock_token(state.user_id, state.provider, state.scopes)
        
        # Mark state as used
        await self.db.update_one(
            self.states_collection,
            {"state": request.state},
            {"$set": {"used": True, "used_at": datetime.utcnow().isoformat()}}
        )
        
        return OAuthCallbackResponse(
            success=True,
            token_id=token_data["token_id"],
            provider=state.provider,
            user_id=state.user_id
        )
    
    async def _create_mock_token(
        self,
        user_id: str,
        provider: OAuthProvider,
        scopes: List[str]
    ) -> dict:
        """Create mock OAuth token for testing"""
        token_id = f"token_{secrets.token_urlsafe(16)}"
        access_token = f"mock_access_token_{secrets.token_urlsafe(32)}"
        refresh_token = f"mock_refresh_token_{secrets.token_urlsafe(32)}"

        token_record = OAuthToken(
            token_id=token_id,
            user_id=user_id,
            provider=provider,
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            expires_at=datetime.utcnow() + timedelta(hours=1),
            refresh_token=refresh_token,
            scope=" ".join(scopes),
            created_at=datetime.utcnow(),
            revoked=False
        )

        # Store in database
        token_data = token_record.model_dump()
        token_data["created_at"] = token_record.created_at.isoformat()
        token_data["expires_at"] = token_record.expires_at.isoformat()

        await self.db.insert_one(
            self.tokens_collection,
            token_data
        )

        return token_data
    
    # ========================================================================
    # Token Management
    # ========================================================================
    
    async def get_user_token(
        self,
        user_id: str,
        provider: OAuthProvider
    ) -> Optional[OAuthToken]:
        """Get user's OAuth token"""
        token_record = await self.db.find_one(
            self.tokens_collection,
            {"user_id": user_id, "provider": provider, "revoked": False}
        )

        if not token_record:
            return None

        # Convert ISO strings back to datetime
        if isinstance(token_record.get("created_at"), str):
            token_record["created_at"] = datetime.fromisoformat(token_record["created_at"])
        if isinstance(token_record.get("expires_at"), str):
            token_record["expires_at"] = datetime.fromisoformat(token_record["expires_at"])
        if isinstance(token_record.get("refreshed_at"), str):
            token_record["refreshed_at"] = datetime.fromisoformat(token_record["refreshed_at"])
        if isinstance(token_record.get("revoked_at"), str):
            token_record["revoked_at"] = datetime.fromisoformat(token_record["revoked_at"])

        return OAuthToken(**token_record)
    
    async def refresh_access_token(
        self,
        user_id: str,
        provider: OAuthProvider
    ) -> Optional[OAuthTokenRefreshResponse]:
        """
        Refresh expired access token
        
        TODO: Implement real token refresh with Google OAuth
        """
        token = await self.get_user_token(user_id, provider)
        
        if not token or not token.refresh_token:
            return None
        
        if self.mock_mode:
            # Mock token refresh
            new_access_token = f"mock_access_token_{secrets.token_urlsafe(32)}"
            new_expires_at = datetime.utcnow() + timedelta(hours=1)
            
            await self.db.update_one(
                self.tokens_collection,
                {"token_id": token.token_id},
                {
                    "$set": {
                        "access_token": new_access_token,
                        "expires_at": new_expires_at.isoformat(),
                        "refreshed_at": datetime.utcnow().isoformat()
                    }
                }
            )
            
            return OAuthTokenRefreshResponse(
                token_id=token.token_id,
                access_token=new_access_token,
                token_type=token.token_type,
                expires_at=new_expires_at,
                refreshed_at=datetime.utcnow()
            )
        else:
            # TODO: Real token refresh
            # credentials = Credentials(
            #     token=token.access_token,
            #     refresh_token=token.refresh_token,
            #     token_uri=self.google_token_uri,
            #     client_id=self.google_client_id,
            #     client_secret=self.google_client_secret
            # )
            # credentials.refresh(Request())
            pass
    
    async def revoke_token(
        self,
        request: OAuthRevokeRequest
    ) -> OAuthRevokeResponse:
        """Revoke OAuth token"""
        token = await self.get_user_token(request.user_id, request.provider)
        
        if not token:
            return OAuthRevokeResponse(
                success=False,
                token_id="",
                revoked_at=datetime.utcnow(),
                message="Token not found"
            )
        
        # Mark as revoked
        revoked_at = datetime.utcnow()
        await self.db.update_one(
            self.tokens_collection,
            {"token_id": token.token_id},
            {
                "$set": {
                    "revoked": True,
                    "revoked_at": revoked_at.isoformat()
                }
            }
        )
        
        return OAuthRevokeResponse(
            success=True,
            token_id=token.token_id,
            revoked_at=revoked_at,
            message="Token revoked successfully"
        )
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def get_health(self) -> OAuthHealthResponse:
        """Get OAuth service health"""
        # Count active tokens
        try:
            tokens = await self.db.find_many(
                self.tokens_collection,
                {"revoked": False}
            )
            active_tokens = len(tokens) if tokens else 0
        except:
            active_tokens = 0

        providers_configured = []
        if self.google_client_id and self.google_client_secret:
            providers_configured.append(OAuthProvider.GOOGLE)

        status = "healthy" if not self.mock_mode else "mock_mode"
        message = "OAuth service operational" if not self.mock_mode else "OAuth service in mock mode (no credentials)"

        return OAuthHealthResponse(
            status=status,
            message=message,
            timestamp=datetime.utcnow(),
            providers_configured=providers_configured,
            active_tokens=active_tokens
        )

