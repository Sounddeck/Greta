"""
OAuth 2.0 Router for Google Workspace Integration

This router provides endpoints for OAuth 2.0 authentication flow,
token management, and provider configuration.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional

from models.oauth_models import (
    OAuthProvider,
    OAuthAuthorizationRequest,
    OAuthAuthorizationResponse,
    OAuthCallbackRequest,
    OAuthCallbackResponse,
    OAuthTokenRefreshRequest,
    OAuthTokenRefreshResponse,
    OAuthRevokeRequest,
    OAuthRevokeResponse,
    OAuthHealthResponse,
    GoogleAccountListResponse,
    GoogleAccountSetPrimaryRequest,
    GoogleAccountSetPrimaryResponse,
    GoogleAccountRemoveRequest,
    GoogleAccountRemoveResponse,
)
from services.oauth_service import OAuthService

router = APIRouter(prefix="/api/v1/oauth", tags=["OAuth 2.0"])


# ============================================================================
# OAuth Authorization Flow
# ============================================================================

@router.post("/authorize", response_model=OAuthAuthorizationResponse)
async def initiate_oauth_flow(
    user_id: str = Query(..., description="User ID"),
    provider: OAuthProvider = Query(OAuthProvider.GOOGLE, description="OAuth provider"),
    scopes: Optional[str] = Query(None, description="Comma-separated scopes (optional)")
):
    """
    Initiate OAuth 2.0 authorization flow
    
    Returns authorization URL for user to visit in browser.
    
    **Steps**:
    1. Call this endpoint to get authorization URL
    2. Redirect user to authorization URL
    3. User authorizes app in browser
    4. Google redirects to callback URL with code
    5. Call /oauth/callback with code and state
    
    **Example**:
    ```
    POST /api/v1/oauth/authorize?user_id=user123&provider=google
    ```
    
    **Response**:
    ```json
    {
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?...",
        "state": "random_state_string",
        "expires_at": "2025-10-09T12:00:00Z"
    }
    ```
    """
    service = OAuthService()
    
    scope_list = scopes.split(",") if scopes else None
    
    request = OAuthAuthorizationRequest(
        user_id=user_id,
        provider=provider,
        scopes=scope_list
    )
    
    return await service.initiate_oauth_flow(request)


@router.get("/callback", response_model=OAuthCallbackResponse)
async def handle_oauth_callback(
    code: Optional[str] = Query(None, description="Authorization code"),
    state: str = Query(..., description="State parameter"),
    error: Optional[str] = Query(None, description="Error code"),
    error_description: Optional[str] = Query(None, description="Error description")
):
    """
    Handle OAuth 2.0 callback
    
    This endpoint is called by Google after user authorizes the app.
    Exchanges authorization code for access token.
    
    **Note**: This endpoint is typically called automatically by Google's redirect.
    You don't need to call it manually.
    
    **Example**:
    ```
    GET /api/v1/oauth/callback?code=auth_code&state=random_state
    ```
    
    **Response**:
    ```json
    {
        "success": true,
        "token_id": "token_abc123",
        "provider": "google",
        "user_id": "user123"
    }
    ```
    """
    service = OAuthService()
    
    request = OAuthCallbackRequest(
        code=code or "",
        state=state,
        error=error,
        error_description=error_description
    )
    
    response = await service.handle_oauth_callback(request)
    
    if not response.success:
        raise HTTPException(
            status_code=400,
            detail={
                "error": response.error,
                "error_description": response.error_description
            }
        )
    
    return response


# ============================================================================
# Token Management
# ============================================================================

@router.post("/refresh", response_model=OAuthTokenRefreshResponse)
async def refresh_access_token(
    user_id: str = Query(..., description="User ID"),
    provider: OAuthProvider = Query(OAuthProvider.GOOGLE, description="OAuth provider")
):
    """
    Refresh expired access token
    
    Use this endpoint when access token expires (typically after 1 hour).
    Requires valid refresh token.
    
    **Example**:
    ```
    POST /api/v1/oauth/refresh?user_id=user123&provider=google
    ```
    
    **Response**:
    ```json
    {
        "token_id": "token_abc123",
        "access_token": "new_access_token",
        "token_type": "Bearer",
        "expires_at": "2025-10-09T13:00:00Z",
        "refreshed_at": "2025-10-09T12:00:00Z"
    }
    ```
    """
    service = OAuthService()
    
    response = await service.refresh_access_token(user_id, provider)
    
    if not response:
        raise HTTPException(
            status_code=404,
            detail="No refresh token found for user"
        )
    
    return response


@router.post("/revoke", response_model=OAuthRevokeResponse)
async def revoke_token(
    user_id: str = Query(..., description="User ID"),
    provider: OAuthProvider = Query(OAuthProvider.GOOGLE, description="OAuth provider"),
    reason: Optional[str] = Query(None, description="Revocation reason")
):
    """
    Revoke OAuth token
    
    Revokes user's OAuth token, requiring re-authorization.
    
    **Example**:
    ```
    POST /api/v1/oauth/revoke?user_id=user123&provider=google&reason=user_request
    ```
    
    **Response**:
    ```json
    {
        "success": true,
        "token_id": "token_abc123",
        "revoked_at": "2025-10-09T12:00:00Z",
        "message": "Token revoked successfully"
    }
    ```
    """
    service = OAuthService()
    
    request = OAuthRevokeRequest(
        user_id=user_id,
        provider=provider,
        reason=reason
    )
    
    return await service.revoke_token(request)


@router.get("/token/status")
async def get_token_status(
    user_id: str = Query(..., description="User ID"),
    provider: OAuthProvider = Query(OAuthProvider.GOOGLE, description="OAuth provider")
):
    """
    Get OAuth token status
    
    Check if user has valid OAuth token.
    
    **Example**:
    ```
    GET /api/v1/oauth/token/status?user_id=user123&provider=google
    ```
    
    **Response**:
    ```json
    {
        "has_token": true,
        "is_valid": true,
        "expires_at": "2025-10-09T13:00:00Z",
        "has_refresh_token": true
    }
    ```
    """
    service = OAuthService()
    
    token = await service.get_user_token(user_id, provider)
    
    if not token:
        return {
            "has_token": False,
            "is_valid": False,
            "expires_at": None,
            "has_refresh_token": False
        }
    
    from datetime import datetime
    is_valid = datetime.utcnow() < token.expires_at and not token.revoked
    
    return {
        "has_token": True,
        "is_valid": is_valid,
        "expires_at": token.expires_at.isoformat(),
        "has_refresh_token": token.refresh_token is not None
    }


# ============================================================================
# Multi-Account Management
# ============================================================================

@router.get("/accounts/google", response_model=GoogleAccountListResponse)
async def list_google_accounts(
    user_id: str = Query(..., description="User ID")
):
    """
    List all connected Google accounts for a user

    Returns list of all Google accounts connected to the user's profile,
    including account email, name, primary status, and token information.

    **Example**:
    ```
    GET /api/v1/oauth/accounts/google?user_id=user_123
    ```

    **Response**:
    ```json
    {
        "accounts": [
            {
                "account_email": "user@gmail.com",
                "account_name": "John Doe",
                "is_primary": true,
                "token_id": "tok_abc123",
                "expires_at": "2025-10-14T10:00:00Z",
                "scopes": ["gmail.readonly", "calendar"],
                "created_at": "2025-10-13T10:00:00Z"
            }
        ],
        "total_accounts": 1,
        "primary_account": "user@gmail.com"
    }
    ```
    """
    service = OAuthService()
    return await service.list_user_accounts(user_id, OAuthProvider.GOOGLE)


@router.post("/accounts/google/primary", response_model=GoogleAccountSetPrimaryResponse)
async def set_primary_google_account(
    request: GoogleAccountSetPrimaryRequest
):
    """
    Set primary Google account for a user

    Marks the specified account as the primary account. The primary account
    is used by default when no specific account is specified in API calls.

    **Example**:
    ```
    POST /api/v1/oauth/accounts/google/primary
    {
        "user_id": "user_123",
        "account_email": "work@company.com"
    }
    ```

    **Response**:
    ```json
    {
        "success": true,
        "message": "Primary account set to work@company.com",
        "primary_account": "work@company.com"
    }
    ```
    """
    service = OAuthService()
    return await service.set_primary_account(request)


@router.delete("/accounts/google/{account_email}")
async def remove_google_account(
    account_email: str,
    user_id: str = Query(..., description="User ID")
):
    """
    Remove a connected Google account

    Revokes the OAuth token and removes the account from the user's profile.
    If the removed account was the primary account, another account will be
    automatically set as primary.

    **Example**:
    ```
    DELETE /api/v1/oauth/accounts/google/user@gmail.com?user_id=user_123
    ```

    **Response**:
    ```json
    {
        "success": true,
        "message": "Account user@gmail.com removed successfully",
        "removed_account": "user@gmail.com"
    }
    ```
    """
    service = OAuthService()
    request = GoogleAccountRemoveRequest(
        user_id=user_id,
        account_email=account_email
    )
    return await service.remove_account(request)


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health", response_model=OAuthHealthResponse)
async def get_oauth_health():
    """
    Get OAuth service health
    
    Check OAuth service status and configuration.
    
    **Example**:
    ```
    GET /api/v1/oauth/health
    ```
    
    **Response**:
    ```json
    {
        "status": "mock_mode",
        "message": "OAuth service in mock mode (no credentials)",
        "timestamp": "2025-10-09T12:00:00Z",
        "providers_configured": [],
        "active_tokens": 0
    }
    ```
    """
    service = OAuthService()
    return await service.get_health()

