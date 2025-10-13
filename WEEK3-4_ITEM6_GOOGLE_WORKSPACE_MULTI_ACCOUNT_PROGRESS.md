# Week 3-4, Item 6: Google Workspace Multi-Account Support - IN PROGRESS ⏳

**Date:** October 13, 2025  
**Status:** ⏳ Partial Implementation (Models Updated)  
**Time Spent:** ~1.5 hours  
**Estimated Remaining:** 4.5-6.5 hours  

---

## 📋 Overview

Implementing multi-account support for Google Workspace integration, allowing users to connect and manage multiple Google accounts (Gmail, Calendar, Drive) simultaneously. This enables users to work with personal and work accounts, or multiple business accounts, from a single interface.

---

## ✅ Completed Work

### 1. OAuth Models Updated for Multi-Account Support

**File Modified:** `backend/models/oauth_models.py`

**Changes Made:**

**Added to `OAuthToken` model:**
```python
account_email: Optional[str] = None  # For multi-account support
account_name: Optional[str] = None  # Display name for the account
is_primary: bool = False  # Mark primary account
```

**Added to `OAuthTokenCreate` model:**
```python
account_email: Optional[str] = None
account_name: Optional[str] = None
is_primary: bool = False
```

**Added to `OAuthTokenResponse` model:**
```python
account_email: Optional[str] = None
account_name: Optional[str] = None
is_primary: bool = False
```

**Added to `OAuthTokenRefreshRequest` model:**
```python
account_email: Optional[str] = None  # Specify which account to refresh
```

**New Models Created:**
1. `GoogleAccountInfo` - Account information with email, name, picture, primary status
2. `GoogleAccountListResponse` - List of connected accounts
3. `GoogleAccountSetPrimaryRequest` - Set primary account request
4. `GoogleAccountSetPrimaryResponse` - Set primary account response
5. `GoogleAccountRemoveRequest` - Remove account request
6. `GoogleAccountRemoveResponse` - Remove account response

---

## 🚧 Remaining Work

### 2. OAuth Service Updates (NOT STARTED)

**File to Modify:** `backend/services/oauth_service.py`

**Required Changes:**

**A. Update Token Storage:**
- Modify `_store_token()` to include `account_email`, `account_name`, `is_primary`
- Update database queries to filter by `user_id`, `provider`, AND `account_email`
- Add unique index on `(user_id, provider, account_email)`

**B. Add Multi-Account Methods:**
```python
async def list_user_accounts(
    self,
    user_id: str,
    provider: OAuthProvider
) -> GoogleAccountListResponse:
    """List all connected accounts for a user"""
    
async def set_primary_account(
    self,
    user_id: str,
    account_email: str
) -> GoogleAccountSetPrimaryResponse:
    """Set primary account for a user"""
    
async def remove_account(
    self,
    user_id: str,
    account_email: str
) -> GoogleAccountRemoveResponse:
    """Remove a connected account"""
    
async def get_account_token(
    self,
    user_id: str,
    account_email: Optional[str] = None
) -> Optional[OAuthToken]:
    """Get token for specific account (or primary if not specified)"""
```

**C. Update Existing Methods:**
- `get_user_token()` - Add `account_email` parameter, default to primary
- `refresh_token()` - Add `account_email` parameter
- `revoke_token()` - Add `account_email` parameter

---

### 3. Gmail Service Updates (NOT STARTED)

**File to Modify:** `backend/services/gmail_service.py`

**Required Changes:**

**A. Add Account Parameter to All Methods:**
```python
async def send_email(
    self,
    user_id: str,
    request: GmailSendRequest,
    account_email: Optional[str] = None  # NEW
) -> GmailSendResponse:
    """Send email from specific account"""
    
async def list_messages(
    self,
    user_id: str,
    request: GmailListRequest,
    account_email: Optional[str] = None  # NEW
) -> GmailMessageListResponse:
    """List messages from specific account"""
```

**B. Update Token Retrieval:**
```python
async def _get_user_access_token(
    self,
    user_id: str,
    account_email: Optional[str] = None
) -> Optional[str]:
    """Get access token for specific account"""
    token = await self.oauth_service.get_account_token(user_id, account_email)
    if token:
        return token.access_token
    return None
```

---

### 4. Calendar Service Updates (NOT STARTED)

**File to Modify:** `backend/services/calendar_service.py`

**Same pattern as Gmail Service:**
- Add `account_email` parameter to all methods
- Update token retrieval to use specific account
- Support multi-calendar across multiple accounts

---

### 5. Drive Service Updates (NOT STARTED)

**File to Modify:** `backend/services/drive_service.py`

**Same pattern as Gmail Service:**
- Add `account_email` parameter to all methods
- Update token retrieval to use specific account
- Support file operations across multiple accounts

---

### 6. Router Updates (NOT STARTED)

**Files to Modify:**
- `backend/routers/oauth_router.py`
- `backend/routers/gmail_router.py`
- `backend/routers/calendar_router.py`
- `backend/routers/drive_router.py`

**Required Changes:**

**A. OAuth Router - Add Account Management Endpoints:**
```python
@router.get("/accounts/google")
async def list_google_accounts(user_id: str = Query(...)):
    """List all connected Google accounts"""
    
@router.post("/accounts/google/primary")
async def set_primary_google_account(request: GoogleAccountSetPrimaryRequest):
    """Set primary Google account"""
    
@router.delete("/accounts/google/{account_email}")
async def remove_google_account(user_id: str, account_email: str):
    """Remove a Google account"""
```

**B. Gmail/Calendar/Drive Routers - Add Account Parameter:**
```python
@router.post("/gmail/send")
async def send_email(
    request: GmailSendRequest,
    user_id: str = Query(...),
    account_email: Optional[str] = Query(None)  # NEW
):
    """Send email from specific account"""
```

---

### 7. Frontend Components (NOT STARTED)

**Components to Create:**

**A. Account Selector Component:**
```typescript
// cpas4-ui/src/components/google/AccountSelector.tsx
interface GoogleAccount {
  email: string;
  name: string;
  pictureUrl?: string;
  isPrimary: boolean;
  tokenId: string;
}

export function AccountSelector({
  accounts,
  selectedAccount,
  onSelectAccount,
  onAddAccount,
  onRemoveAccount,
  onSetPrimary
}: AccountSelectorProps) {
  // Dropdown to select active account
  // Show account email, name, picture
  // Mark primary account
  // Add new account button
  // Remove account button
  // Set as primary button
}
```

**B. Account Manager Component:**
```typescript
// cpas4-ui/src/components/google/AccountManager.tsx
export function AccountManager() {
  // List all connected accounts
  // Add new account (OAuth flow)
  // Remove account
  // Set primary account
  // View account details (scopes, expiry)
  // Refresh tokens
}
```

**C. Zustand Store:**
```typescript
// cpas4-ui/src/stores/googleAccountStore.ts
interface GoogleAccountState {
  accounts: GoogleAccount[];
  selectedAccount: string | null;
  primaryAccount: string | null;
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchAccounts: () => Promise<void>;
  selectAccount: (email: string) => void;
  addAccount: () => Promise<void>;
  removeAccount: (email: string) => Promise<void>;
  setPrimaryAccount: (email: string) => Promise<void>;
}
```

---

### 8. UI Integration (NOT STARTED)

**Required Changes:**

**A. Add Account Selector to Gmail/Calendar/Drive Pages:**
- Show current account at top of page
- Dropdown to switch accounts
- Visual indicator of primary account

**B. Add Account Management Page:**
- Route: `/settings/google-accounts`
- List all connected accounts
- Add/remove accounts
- Set primary account
- View token status

**C. Update API Calls:**
- Add `account_email` parameter to all Google Workspace API calls
- Use selected account from store
- Fall back to primary account if none selected

---

## 🔧 Technical Implementation Details

### Database Schema Changes

**OAuth Tokens Collection:**
```javascript
{
  token_id: "tok_abc123",
  user_id: "user_123",
  provider: "google",
  account_email: "user@gmail.com",  // NEW
  account_name: "John Doe",  // NEW
  is_primary: true,  // NEW
  access_token: "ya29.a0...",
  refresh_token: "1//0g...",
  expires_at: ISODate("2025-10-14T10:00:00Z"),
  created_at: ISODate("2025-10-13T10:00:00Z"),
  revoked: false
}
```

**Unique Index:**
```javascript
db.oauth_tokens.createIndex(
  { user_id: 1, provider: 1, account_email: 1 },
  { unique: true }
)
```

---

### OAuth Flow for Additional Accounts

**1. User clicks "Add Account"**
- Frontend initiates OAuth flow
- Backend generates authorization URL
- User redirected to Google login

**2. User selects/logs in to Google account**
- Google shows consent screen
- User grants permissions

**3. Google redirects back with code**
- Backend exchanges code for tokens
- Backend fetches user info (email, name, picture)
- Backend stores token with `account_email`
- If first account, set as primary

**4. Frontend updates**
- Fetch updated account list
- Show new account in selector
- Set as active account

---

### Account Selection Logic

**Priority Order:**
1. **Explicitly selected account** - User chose from dropdown
2. **Primary account** - User's default account
3. **First account** - If no primary set
4. **Error** - If no accounts connected

**Implementation:**
```python
async def get_active_account_token(
    user_id: str,
    account_email: Optional[str] = None
) -> OAuthToken:
    # 1. Use explicitly selected account
    if account_email:
        token = await get_account_token(user_id, account_email)
        if token:
            return token
    
    # 2. Use primary account
    token = await get_primary_account_token(user_id)
    if token:
        return token
    
    # 3. Use first account
    accounts = await list_user_accounts(user_id)
    if accounts:
        return accounts[0].token
    
    # 4. No accounts
    raise HTTPException(404, "No Google accounts connected")
```

---

## 📊 Progress Summary

**Completed:**
- ✅ OAuth models updated for multi-account support (6 new fields, 6 new models)

**Remaining:**
- ⏳ OAuth service multi-account methods (3-4 hours)
- ⏳ Gmail/Calendar/Drive service updates (1-2 hours)
- ⏳ Router endpoint updates (1 hour)
- ⏳ Frontend components (2-3 hours)
- ⏳ UI integration (1-2 hours)
- ⏳ Testing and documentation (1 hour)

**Total Remaining:** ~9-13 hours

---

## 🎯 Next Steps

**Immediate Priority:**
1. Update OAuth service with multi-account methods
2. Update Gmail/Calendar/Drive services to accept `account_email`
3. Add account management endpoints to OAuth router
4. Create frontend AccountSelector component
5. Create frontend AccountManager component
6. Create googleAccountStore
7. Integrate account selector into Gmail/Calendar/Drive pages
8. Test multi-account OAuth flow
9. Test account switching
10. Document usage

---

## 📝 Notes

**Current Status:**
- Google Workspace integration exists but is in **mock mode**
- Requires Google Cloud OAuth credentials to function with real APIs
- See `GOOGLE_WORKSPACE_SETUP.md` for credential setup instructions

**Blockers:**
- None (can implement multi-account support in mock mode)
- Real testing requires OAuth credentials

**Dependencies:**
- Existing OAuth service
- Existing Gmail/Calendar/Drive services
- MongoDB for token storage

---

## 🚀 Future Enhancements

**After Multi-Account Support:**
1. **Account Sync Status** - Show last sync time per account
2. **Account Quotas** - Display API quota usage per account
3. **Account Health** - Monitor token expiry, errors
4. **Bulk Operations** - Perform actions across multiple accounts
5. **Account Groups** - Group related accounts (e.g., "Work Accounts")
6. **Account Permissions** - Fine-grained scope management per account
7. **Account Sharing** - Share account access with team members

---

**Status:** ⏳ In Progress (Models Complete, Services Pending)  
**Next:** Complete OAuth service multi-account methods  
**Estimated Completion:** 9-13 hours remaining

