"""
OAuth 2.0 Models for Google Workspace Integration

This module defines Pydantic models for OAuth 2.0 authentication flow,
token management, and provider configuration.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class OAuthProvider(str, Enum):
    """OAuth providers supported"""
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"
    SLACK = "slack"


class OAuthTokenType(str, Enum):
    """OAuth token types"""
    BEARER = "Bearer"
    MAC = "MAC"


class OAuthGrantType(str, Enum):
    """OAuth grant types"""
    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"
    CLIENT_CREDENTIALS = "client_credentials"


# ============================================================================
# OAuth Configuration Models
# ============================================================================

class OAuthConfigBase(BaseModel):
    """Base OAuth configuration"""
    provider: OAuthProvider
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str]
    auth_uri: str = "https://accounts.google.com/o/oauth2/auth"
    token_uri: str = "https://oauth2.googleapis.com/token"
    revoke_uri: Optional[str] = "https://oauth2.googleapis.com/revoke"


class OAuthConfig(OAuthConfigBase):
    """OAuth configuration with metadata"""
    config_id: str
    user_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class OAuthConfigCreate(OAuthConfigBase):
    """Create OAuth configuration"""
    pass


class OAuthConfigResponse(BaseModel):
    """OAuth configuration response (without secrets)"""
    config_id: str
    provider: OAuthProvider
    client_id: str
    redirect_uri: str
    scopes: List[str]
    created_at: datetime


# ============================================================================
# OAuth Token Models
# ============================================================================

class OAuthTokenBase(BaseModel):
    """Base OAuth token"""
    access_token: str
    token_type: OAuthTokenType = OAuthTokenType.BEARER
    expires_in: int  # Seconds until expiration
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class OAuthToken(OAuthTokenBase):
    """OAuth token with metadata"""
    token_id: str
    user_id: str
    provider: OAuthProvider
    expires_at: datetime
    created_at: datetime
    refreshed_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revoked: bool = False


class OAuthTokenCreate(BaseModel):
    """Create OAuth token"""
    user_id: str
    provider: OAuthProvider
    access_token: str
    token_type: OAuthTokenType = OAuthTokenType.BEARER
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class OAuthTokenResponse(BaseModel):
    """OAuth token response"""
    token_id: str
    provider: OAuthProvider
    token_type: OAuthTokenType
    expires_at: datetime
    scope: Optional[str] = None
    has_refresh_token: bool


class OAuthTokenRefreshRequest(BaseModel):
    """Refresh OAuth token request"""
    user_id: str
    provider: OAuthProvider


class OAuthTokenRefreshResponse(BaseModel):
    """Refresh OAuth token response"""
    token_id: str
    access_token: str
    token_type: OAuthTokenType
    expires_at: datetime
    refreshed_at: datetime


# ============================================================================
# OAuth Flow Models
# ============================================================================

class OAuthStateBase(BaseModel):
    """Base OAuth state for CSRF protection"""
    state: str
    user_id: str
    provider: OAuthProvider
    redirect_uri: str
    scopes: List[str]


class OAuthState(OAuthStateBase):
    """OAuth state with metadata"""
    state_id: str
    created_at: datetime
    expires_at: datetime
    used: bool = False
    used_at: Optional[datetime] = None


class OAuthAuthorizationRequest(BaseModel):
    """OAuth authorization request"""
    user_id: str
    provider: OAuthProvider
    scopes: Optional[List[str]] = None
    redirect_uri: Optional[str] = None


class OAuthAuthorizationResponse(BaseModel):
    """OAuth authorization response"""
    authorization_url: str
    state: str
    expires_at: datetime


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request"""
    code: str
    state: str
    error: Optional[str] = None
    error_description: Optional[str] = None


class OAuthCallbackResponse(BaseModel):
    """OAuth callback response"""
    success: bool
    token_id: Optional[str] = None
    provider: Optional[OAuthProvider] = None
    user_id: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None


class OAuthRevokeRequest(BaseModel):
    """Revoke OAuth token request"""
    user_id: str
    provider: OAuthProvider
    reason: Optional[str] = None


class OAuthRevokeResponse(BaseModel):
    """Revoke OAuth token response"""
    success: bool
    token_id: str
    revoked_at: datetime
    message: str


# ============================================================================
# Google Workspace Scopes
# ============================================================================

class GoogleWorkspaceScopes:
    """Google Workspace OAuth scopes"""
    
    # Gmail scopes
    GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
    GMAIL_MODIFY = "https://www.googleapis.com/auth/gmail.modify"
    GMAIL_SEND = "https://www.googleapis.com/auth/gmail.send"
    GMAIL_COMPOSE = "https://www.googleapis.com/auth/gmail.compose"
    GMAIL_LABELS = "https://www.googleapis.com/auth/gmail.labels"
    
    # Calendar scopes
    CALENDAR = "https://www.googleapis.com/auth/calendar"
    CALENDAR_READONLY = "https://www.googleapis.com/auth/calendar.readonly"
    CALENDAR_EVENTS = "https://www.googleapis.com/auth/calendar.events"
    
    # Drive scopes
    DRIVE = "https://www.googleapis.com/auth/drive"
    DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
    DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
    DRIVE_METADATA = "https://www.googleapis.com/auth/drive.metadata"
    
    # Sheets scopes
    SPREADSHEETS = "https://www.googleapis.com/auth/spreadsheets"
    SPREADSHEETS_READONLY = "https://www.googleapis.com/auth/spreadsheets.readonly"
    
    # Docs scopes
    DOCUMENTS = "https://www.googleapis.com/auth/documents"
    DOCUMENTS_READONLY = "https://www.googleapis.com/auth/documents.readonly"
    
    # User info scopes
    USERINFO_EMAIL = "https://www.googleapis.com/auth/userinfo.email"
    USERINFO_PROFILE = "https://www.googleapis.com/auth/userinfo.profile"
    
    @classmethod
    def get_all_workspace_scopes(cls) -> List[str]:
        """Get all Google Workspace scopes"""
        return [
            cls.GMAIL_MODIFY,
            cls.GMAIL_SEND,
            cls.CALENDAR,
            cls.DRIVE,
            cls.SPREADSHEETS,
            cls.DOCUMENTS,
            cls.USERINFO_EMAIL,
            cls.USERINFO_PROFILE,
        ]
    
    @classmethod
    def get_gmail_scopes(cls) -> List[str]:
        """Get Gmail scopes"""
        return [cls.GMAIL_MODIFY, cls.GMAIL_SEND, cls.GMAIL_LABELS]
    
    @classmethod
    def get_calendar_scopes(cls) -> List[str]:
        """Get Calendar scopes"""
        return [cls.CALENDAR, cls.CALENDAR_EVENTS]
    
    @classmethod
    def get_drive_scopes(cls) -> List[str]:
        """Get Drive scopes"""
        return [cls.DRIVE, cls.DRIVE_FILE]
    
    @classmethod
    def get_sheets_scopes(cls) -> List[str]:
        """Get Sheets scopes"""
        return [cls.SPREADSHEETS]
    
    @classmethod
    def get_docs_scopes(cls) -> List[str]:
        """Get Docs scopes"""
        return [cls.DOCUMENTS]


# ============================================================================
# Health Check Models
# ============================================================================

class OAuthHealthResponse(BaseModel):
    """OAuth service health response"""
    status: str
    message: str
    timestamp: datetime
    providers_configured: List[OAuthProvider]
    active_tokens: int

