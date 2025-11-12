# Google Workspace Multi-Tenant Integration - Current State Analysis

**Date:** 2025-11-10  
**Phase:** Phase 1, Item 2 - Google Cloud OAuth Setup  
**Status:** 🔄 **ANALYSIS COMPLETE - READY FOR IMPLEMENTATION**

---

## Executive Summary

CPAS5 has a **comprehensive multi-tenant Google Workspace architecture already designed and partially implemented**, but is currently running in **MOCK MODE** due to missing Google Cloud OAuth credentials. The codebase is production-ready for multi-workspace scenarios and only requires:

1. ✅ Google Cloud Project setup
2. ✅ OAuth 2.0 credentials configuration
3. ✅ Real API integration (replacing mock implementations)
4. ✅ Secrets manager integration (for secure credential storage)

**Key Finding:** The multi-tenant architecture is **already built** - we just need to configure it and enable real API calls.

---

## 1. Current Implementation Status

### ✅ **IMPLEMENTED (Mock Mode)**

#### A. Multi-Workspace Data Models
**File:** `backend/models/oauth_models.py` (lines 358-505)

**Workspace Credentials Model:**
```python
class WorkspaceOAuthCredentials(BaseModel):
    credentials_id: str          # Unique credential ID
    workspace_id: str            # Workspace identifier (e.g., "acme-corp")
    workspace_name: str          # Human-readable name
    client_id: str               # Google OAuth Client ID
    client_secret: str           # Google OAuth Client Secret (encrypted)
    redirect_uri: str            # OAuth redirect URI
    workspace_domain: str        # Google Workspace domain (e.g., "acme.com")
    workspace_type: str          # "business", "personal", "client"
    owner_user_id: str           # User who owns this workspace
    authorized_user_ids: List[str]  # Users authorized to access
    is_active: bool              # Active status
    is_default: bool             # Default workspace for user
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime
```

**Multi-Account Token Model:**
```python
class OAuthToken(BaseModel):
    token_id: str
    user_id: str
    provider: OAuthProvider
    access_token: str
    refresh_token: str
    account_email: str           # For multi-account support
    account_name: str            # Display name
    is_primary: bool             # Primary account flag
    expires_at: datetime
    scope: str
```

#### B. Database Collections
**File:** `backend/database.py` (lines 158-162, 430-463)

**Collections:**
1. **`oauth_tokens`** - User OAuth tokens with multi-account support
   - Indexes: `user_id`, `provider`, `account_email`, `is_primary`
   - Compound indexes for efficient queries

2. **`oauth_states`** - OAuth authorization states (CSRF protection)
   - Indexes: `state`, `user_id`, `provider`, `expires_at`

3. **`workspace_credentials`** - Multi-workspace OAuth credentials
   - Indexes: `credentials_id`, `workspace_id`, `owner_user_id`, `workspace_domain`
   - Compound indexes: `(owner_user_id, is_active)`, `(owner_user_id, is_default)`

#### C. OAuth Service with Multi-Workspace Support
**File:** `backend/services/oauth_service.py` (862 lines)

**Key Features:**
- ✅ Multi-workspace mode toggle (`ENABLE_MULTI_WORKSPACE`)
- ✅ Workspace credential management (CRUD operations)
- ✅ Per-workspace OAuth flow
- ✅ Multi-account token management
- ✅ Primary account selection
- ✅ Account switching
- ✅ Token refresh mechanism
- ✅ Mock mode for testing

**Methods Implemented:**
```python
# Workspace Management
async def get_workspace_credentials(workspace_id, user_id)
async def create_workspace_credentials(request)
async def update_workspace_credentials(credentials_id, user_id, update)
async def delete_workspace_credentials(credentials_id, user_id)
async def list_user_workspaces(user_id)
async def set_default_workspace(user_id, workspace_id)
async def authorize_user_to_workspace(workspace_id, user_id, authorized_by)
async def revoke_user_from_workspace(workspace_id, user_id, revoked_by)

# Multi-Account Management
async def list_user_accounts(user_id, provider)
async def set_primary_account(user_id, provider, account_email)
async def remove_account(user_id, provider, account_email)
async def get_user_token(user_id, provider, account_email)
```

#### D. Google Workspace Services (Mock Mode)
**Files:**
- `backend/services/gmail_service.py` (621 lines)
- `backend/services/calendar_service.py` (407 lines)
- `backend/services/drive_service.py` (396 lines)

**All services support:**
- ✅ Multi-account operations via `account_email` parameter
- ✅ Token retrieval from OAuth service
- ✅ Mock implementations for testing
- ✅ Real API placeholders (commented out)

**Example:**
```python
async def send_email(
    user_id: str,
    request: GmailSendRequest,
    account_email: Optional[str] = None  # ← Multi-account support
) -> GmailSendResponse:
    # Get token for specific account or primary
    access_token = await self._get_user_access_token(user_id, account_email)
    
    if self.mock_mode:
        # Mock implementation
        return mock_response
    else:
        # Real Gmail API call (TODO)
        pass
```

#### E. API Routers
**Files:**
- `backend/routers/oauth_router.py` - OAuth flow endpoints
- `backend/routers/gmail_router.py` - Gmail API endpoints
- `backend/routers/calendar_router.py` - Calendar API endpoints
- `backend/routers/drive_router.py` - Drive API endpoints

**All routers support:**
- ✅ Multi-account operations
- ✅ Workspace selection
- ✅ Account switching

#### F. Frontend Integration
**File:** `cpas4-ui/src/stores/googleAccountStore.ts`

**Features:**
- ✅ Account management UI
- ✅ OAuth flow initiation
- ✅ Account switching
- ✅ Primary account selection

---

## 2. What's Missing (Real API Integration)

### ❌ **NOT IMPLEMENTED**

#### A. Google Cloud Project Setup
**Status:** Not configured  
**Required:**
1. Create Google Cloud Project
2. Enable APIs:
   - Gmail API
   - Google Calendar API
   - Google Drive API
   - Google People API (for profile info)
3. Configure OAuth consent screen
4. Create OAuth 2.0 credentials (Client ID + Secret)

#### B. Real OAuth Token Exchange
**File:** `backend/services/oauth_service.py` (line 255)

**Current:**
```python
# TODO: Implement real token exchange with Google OAuth
token_data = await self._create_mock_token(user_id, provider, scopes)
```

**Needed:**
```python
# Real token exchange using google-auth-oauthlib
flow = Flow.from_client_config(
    client_config={
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }
    },
    scopes=scopes
)
flow.fetch_token(code=authorization_code)
credentials = flow.credentials
```

#### C. Real Gmail API Calls
**File:** `backend/services/gmail_service.py` (lines 58-71, 312, 354, etc.)

**Current:**
```python
# TODO: Implement real Gmail API service with credentials
if self.mock_mode:
    return None  # Mock mode

# Real Gmail API service (commented out)
# credentials = Credentials(token=access_token)
# service = build('gmail', 'v1', credentials=credentials)
# return service
```

**Needed:**
```python
credentials = Credentials(token=access_token)
service = build('gmail', 'v1', credentials=credentials)

# Send email
message = service.users().messages().send(
    userId='me',
    body={'raw': base64_encoded_message}
).execute()
```

#### D. Real Calendar API Calls
**File:** `backend/services/calendar_service.py` (lines 129, 293, 371)

**Current:**
```python
# TODO: Implement real Google Calendar API call
raise NotImplementedError("Real Google Calendar API not yet implemented")
```

**Needed:**
```python
credentials = Credentials(token=access_token)
service = build('calendar', 'v3', credentials=credentials)

# Create event
event = service.events().insert(
    calendarId='primary',
    body=event_data
).execute()
```

#### E. Real Drive API Calls
**File:** `backend/services/drive_service.py` (similar pattern)

**Needed:**
```python
credentials = Credentials(token=access_token)
service = build('drive', 'v3', credentials=credentials)

# Upload file
file_metadata = {'name': filename}
media = MediaFileUpload(file_path, mimetype=mime_type)
file = service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id'
).execute()
```

#### F. Secrets Manager Integration
**Status:** Secrets manager implemented (Phase 1.1) but not integrated with OAuth

**Needed:**
1. Store workspace OAuth credentials in Vault/AWS Secrets Manager
2. Encrypt `client_secret` before database storage
3. Decrypt when retrieving credentials
4. Rotate credentials periodically

---

## 3. Multi-Tenant Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         User (Agency Owner)                      │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ├─── Workspace 1: "Acme Corp"
                │    ├── workspace_id: "acme-corp"
                │    ├── workspace_domain: "acme.com"
                │    ├── client_id: "123...apps.googleusercontent.com"
                │    ├── client_secret: "GOCSPX-..." (encrypted)
                │    ├── authorized_users: ["user_123", "user_456"]
                │    └── OAuth Tokens:
                │        ├── admin@acme.com (primary)
                │        └── sales@acme.com
                │
                ├─── Workspace 2: "Beta Inc"
                │    ├── workspace_id: "beta-inc"
                │    ├── workspace_domain: "beta.com"
                │    ├── client_id: "456...apps.googleusercontent.com"
                │    ├── client_secret: "GOCSPX-..." (encrypted)
                │    └── OAuth Tokens:
                │        └── ceo@beta.com (primary)
                │
                └─── Workspace 3: "Personal"
                     ├── workspace_id: "personal"
                     ├── workspace_type: "personal"
                     └── OAuth Tokens:
                         └── user@gmail.com (primary)
```

### Data Isolation

**Workspace Level:**
- Each workspace has its own Google OAuth credentials
- Each workspace connects to a different Google Workspace instance
- Credentials stored in `workspace_credentials` collection

**Account Level:**
- Each workspace can have multiple Google accounts
- Each account has its own OAuth tokens
- Tokens stored in `oauth_tokens` collection with `account_email`

**User Level:**
- Users can be authorized to access multiple workspaces
- Users can switch between workspaces
- Users can switch between accounts within a workspace

---

## 4. Recommended Implementation Plan

### Phase 1.2: Google Cloud OAuth Setup (Current)

#### Step 1: Create Google Cloud Project (1 hour)
1. Go to https://console.cloud.google.com
2. Create new project: "CPAS5-Production"
3. Enable billing (required for OAuth)

#### Step 2: Enable APIs (30 minutes)
```bash
gcloud services enable gmail.googleapis.com
gcloud services enable calendar-json.googleapis.com
gcloud services enable drive.googleapis.com
gcloud services enable people.googleapis.com
```

#### Step 3: Configure OAuth Consent Screen (1 hour)
1. Navigate to "APIs & Services" > "OAuth consent screen"
2. Choose "External" (for multi-tenant support)
3. Fill in application details:
   - App name: "CPAS5 / GRETA PAI"
   - User support email
   - Developer contact email
4. Add scopes:
   - `https://www.googleapis.com/auth/gmail.modify`
   - `https://www.googleapis.com/auth/calendar`
   - `https://www.googleapis.com/auth/drive.file`
   - `https://www.googleapis.com/auth/userinfo.email`
   - `https://www.googleapis.com/auth/userinfo.profile`

#### Step 4: Create OAuth 2.0 Credentials (30 minutes)
1. Navigate to "Credentials" > "Create Credentials" > "OAuth client ID"
2. Application type: "Web application"
3. Name: "CPAS5 Web Client"
4. Authorized redirect URIs:
   - `http://localhost:8000/api/v1/oauth/callback` (development)
   - `https://app.greta.ai/api/v1/oauth/callback` (production)
5. Download JSON credentials

#### Step 5: Store Credentials in Secrets Manager (1 hour)
```python
from services.secrets_manager import get_secrets_manager

async def store_google_credentials():
    manager = get_secrets_manager()
    
    # Store default workspace credentials
    await manager.set_secret(
        "cpas5/google/default/client_id",
        "123456789012-abc...apps.googleusercontent.com"
    )
    await manager.set_secret(
        "cpas5/google/default/client_secret",
        "GOCSPX-abc123..."
    )
```

#### Step 6: Implement Real OAuth Token Exchange (4 hours)
- Replace mock token exchange in `oauth_service.py`
- Implement real Google OAuth flow
- Add token refresh mechanism
- Add error handling

#### Step 7: Implement Real API Calls (8 hours)
- Gmail API integration (3 hours)
- Calendar API integration (3 hours)
- Drive API integration (2 hours)

#### Step 8: Testing (4 hours)
- Test OAuth flow
- Test multi-account support
- Test multi-workspace support
- Test API calls

**Total Effort:** ~20 hours (2.5 days)

---

## 5. Multi-Workspace Credential Management

### Storing Workspace Credentials

**Option 1: Database + Secrets Manager (Recommended)**
```python
# Store in database (without secret)
workspace_creds = {
    "credentials_id": "cred_abc123",
    "workspace_id": "acme-corp",
    "workspace_name": "Acme Corporation",
    "client_id": "123...apps.googleusercontent.com",
    "client_secret_ref": "cpas5/workspaces/acme-corp/client_secret",  # Reference
    "redirect_uri": "https://app.greta.ai/api/v1/oauth/callback",
    "workspace_domain": "acme.com"
}

# Store secret in Vault
await secrets_manager.set_secret(
    "cpas5/workspaces/acme-corp/client_secret",
    "GOCSPX-actual-secret-here"
)
```

**Option 2: All in Secrets Manager**
```python
# Store entire workspace config in Vault
await secrets_manager.set_secret(
    "cpas5/workspaces/acme-corp/credentials",
    json.dumps({
        "client_id": "123...apps.googleusercontent.com",
        "client_secret": "GOCSPX-...",
        "redirect_uri": "https://app.greta.ai/api/v1/oauth/callback"
    })
)
```

### Retrieving Workspace Credentials

```python
async def get_workspace_oauth_config(workspace_id: str):
    # Get workspace metadata from database
    workspace = await db.find_one(
        "workspace_credentials",
        {"workspace_id": workspace_id}
    )
    
    # Get client_secret from secrets manager
    client_secret = await secrets_manager.get_secret(
        f"cpas5/workspaces/{workspace_id}/client_secret"
    )
    
    return {
        "client_id": workspace["client_id"],
        "client_secret": client_secret,
        "redirect_uri": workspace["redirect_uri"]
    }
```

---

## 6. Next Steps

### Immediate (This Week)
1. ✅ **Create Google Cloud Project** (1 hour)
2. ✅ **Enable APIs** (30 min)
3. ✅ **Configure OAuth consent screen** (1 hour)
4. ✅ **Create OAuth credentials** (30 min)
5. ✅ **Store credentials in Vault** (1 hour)

### Short-term (Next Week)
6. ✅ **Implement real OAuth token exchange** (4 hours)
7. ✅ **Implement real Gmail API** (3 hours)
8. ✅ **Implement real Calendar API** (3 hours)
9. ✅ **Implement real Drive API** (2 hours)
10. ✅ **Testing and validation** (4 hours)

### Medium-term (Next 2 Weeks)
11. ✅ **Multi-workspace credential encryption** (2 hours)
12. ✅ **Workspace switching UI** (4 hours)
13. ✅ **Account switching UI** (4 hours)
14. ✅ **Error handling and retry logic** (4 hours)
15. ✅ **Rate limiting for API calls** (2 hours)

---

## 7. Summary

### ✅ What We Have
- **Complete multi-tenant architecture** designed and implemented
- **Database schema** for workspaces and multi-account support
- **OAuth service** with workspace and account management
- **Mock implementations** for all Google Workspace services
- **Frontend integration** for account management
- **Secrets manager** ready for credential storage

### ❌ What We Need
- **Google Cloud Project** with OAuth credentials
- **Real OAuth token exchange** implementation
- **Real API calls** for Gmail, Calendar, Drive
- **Secrets manager integration** for workspace credentials
- **Testing** with real Google APIs

### 🎯 Effort Required
- **Setup:** 3-4 hours
- **Implementation:** 16-20 hours
- **Testing:** 4-6 hours
- **Total:** 23-30 hours (3-4 days)

---

**Status:** ✅ **READY FOR IMPLEMENTATION**  
**Next Action:** Create Google Cloud Project and configure OAuth credentials

Would you like me to proceed with Step 1 (Google Cloud Project setup)?

