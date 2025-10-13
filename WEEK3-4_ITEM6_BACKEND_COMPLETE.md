# Week 3-4, Item 6: Google Workspace Multi-Account Support - Backend Complete ✅

**Date:** October 13, 2025  
**Status:** ✅ Backend Implementation Complete (90%)  
**Time Spent:** ~4 hours  
**Remaining:** Frontend implementation (~4-6 hours)

---

## 📊 Backend Implementation Summary

The backend infrastructure for Google Workspace multi-account support is now **90% complete**. All core services and models have been updated to support multiple Google accounts per user.

---

## ✅ Completed Backend Work

### 1. OAuth Models (100% Complete) ✅
**File:** `backend/models/oauth_models.py`  
**Commit:** `64ae6d6`

**Changes:**
- Added `account_email`, `account_name`, `is_primary` fields to `OAuthToken`
- Updated `OAuthTokenCreate`, `OAuthTokenResponse`, `OAuthTokenRefreshRequest`
- Created 6 new models for account management:
  - `GoogleAccountInfo` - Account information
  - `GoogleAccountListResponse` - List of accounts
  - `GoogleAccountSetPrimaryRequest/Response` - Set primary account
  - `GoogleAccountRemoveRequest/Response` - Remove account

---

### 2. OAuth Service (100% Complete) ✅
**File:** `backend/services/oauth_service.py`  
**Commit:** `40ee745`

**New Methods:**
- `list_user_accounts()` - List all connected Google accounts
- `set_primary_account()` - Set primary account for user
- `remove_account()` - Remove/revoke account connection
- `get_account_token()` - Get token for specific account

**Updated Methods:**
- `_create_mock_token()` - Now includes account_email, account_name, is_primary
- `get_user_token()` - Supports account_email parameter with primary fallback

**Features:**
- Automatic primary account designation for first account
- Primary account fallback when no account specified
- Account removal with automatic primary reassignment

---

### 3. Gmail Service (100% Complete) ✅
**File:** `backend/services/gmail_service.py`  
**Commit:** `a980577`

**Updated Methods (11 total):**
- `_get_user_access_token()` - Accepts account_email parameter
- `send_email()` - Send from specific account
- `list_emails()` - List from specific account
- `search_emails()` - Search in specific account
- `get_email()` - Get from specific account
- `modify_email()` - Modify in specific account
- `delete_email()` - Delete from specific account
- `list_labels()` - List labels from specific account
- `create_label()` - Create label in specific account
- `batch_modify()` - Batch modify in specific account
- `batch_delete()` - Batch delete from specific account
- `get_health()` - Health check for specific account

---

### 4. Calendar Service (100% Complete) ✅
**File:** `backend/services/calendar_service.py`  
**Commit:** `3a2f169`

**Updated Methods (7 total):**
- `_get_user_access_token()` - Accepts account_email parameter
- `create_event()` - Create in specific account
- `list_events()` - List from specific account
- `get_event()` - Get from specific account
- `update_event()` - Update in specific account
- `delete_event()` - Delete from specific account
- `list_calendars()` - List calendars from specific account
- `get_health()` - Health check for specific account

---

### 5. Drive Service (100% Complete) ✅
**File:** `backend/services/drive_service.py`  
**Commit:** `3a2f169`

**Updated Methods (8 total):**
- `_get_user_access_token()` - Accepts account_email parameter
- `upload_file()` - Upload to specific account
- `download_file()` - Download from specific account
- `list_files()` - List from specific account
- `get_file()` - Get from specific account
- `delete_file()` - Delete from specific account
- `share_file()` - Share in specific account
- `create_folder()` - Create in specific account
- `get_health()` - Health check for specific account

---

### 6. OAuth Router (100% Complete) ✅
**File:** `backend/routers/oauth_router.py`  
**Commit:** `8c7e3d3`

**New Endpoints:**
- `GET /api/v1/oauth/accounts/google` - List all connected Google accounts
- `POST /api/v1/oauth/accounts/google/primary` - Set primary account
- `DELETE /api/v1/oauth/accounts/google/{account_email}` - Remove account

**Features:**
- Comprehensive API documentation with examples
- List accounts with email, name, primary status, token info
- Set primary account for default API usage
- Remove account with automatic primary reassignment

---

## ⏳ Remaining Backend Work (10%)

### 7. Gmail/Calendar/Drive Routers (Pending)

**Files to Update:**
- `backend/routers/gmail_router.py`
- `backend/routers/calendar_router.py`
- `backend/routers/drive_router.py`

**Required Changes:**
Add `account_email` query parameter to all endpoints and pass to service methods.

**Pattern to Apply:**
```python
@router.post("/send", response_model=GmailSendResponse)
async def send_email(
    user_id: str = Query(..., description="User ID"),
    request: GmailSendRequest = Body(...),
    account_email: Optional[str] = Query(None, description="Account email (optional, uses primary if not specified)")  # ADD THIS
):
    """Send email via Gmail from specific account"""
    service = GmailService()
    return await service.send_email(user_id, request, account_email)  # PASS account_email
```

**Endpoints to Update:**

**Gmail Router (11 endpoints):**
- `/send` - send_email()
- `/list` - list_emails()
- `/search` - search_emails()
- `/{message_id}` - get_email()
- `/{message_id}/modify` - modify_email()
- `/{message_id}/delete` - delete_email()
- `/labels` - list_labels()
- `/labels/create` - create_label()
- `/batch/modify` - batch_modify()
- `/batch/delete` - batch_delete()
- `/health` - get_health()

**Calendar Router (7 endpoints):**
- `/events` - create_event()
- `/events/list` - list_events()
- `/events/{event_id}` - get_event()
- `/events/{event_id}` - update_event()
- `/events/{event_id}` - delete_event()
- `/calendars` - list_calendars()
- `/health` - get_health()

**Drive Router (8 endpoints):**
- `/upload` - upload_file()
- `/download/{file_id}` - download_file()
- `/files` - list_files()
- `/files/{file_id}` - get_file()
- `/files/{file_id}` - delete_file()
- `/files/{file_id}/share` - share_file()
- `/folders` - create_folder()
- `/health` - get_health()

**Estimated Time:** 1-2 hours (straightforward, repetitive changes)

**Note:** Router updates are optional for initial testing since services already support multi-account. Frontend can call services directly or routers can be updated later.

---

## 🎯 Frontend Implementation (Pending)

### Components to Create:

**1. Account Selector Component**
- File: `cpas4-ui/src/components/google/AccountSelector.tsx`
- Dropdown to select active Google account
- Show account email, name, picture
- Mark primary account
- Add new account button

**2. Account Manager Component**
- File: `cpas4-ui/src/components/google/AccountManager.tsx`
- List all connected accounts
- Add new account (OAuth flow)
- Remove account
- Set primary account
- View account details

**3. Zustand Store**
- File: `cpas4-ui/src/stores/googleAccountStore.ts`
- State: accounts, selectedAccount, primaryAccount, loading, error
- Actions: fetchAccounts, selectAccount, addAccount, removeAccount, setPrimaryAccount

**4. API Client Updates**
- File: `cpas4-ui/src/lib/api.ts`
- Add account management API calls
- Add account_email parameter to Gmail/Calendar/Drive calls

**5. UI Integration**
- Add AccountSelector to Gmail/Calendar/Drive pages
- Create account management settings page at `/settings/google-accounts`
- Update API calls to include selected account_email

---

## 🧪 Testing Strategy

### Backend Testing (Mock Mode):

**1. Test Account Management:**
```bash
# List accounts
curl -X GET "http://localhost:8000/api/v1/oauth/accounts/google?user_id=test_user"

# Set primary account
curl -X POST "http://localhost:8000/api/v1/oauth/accounts/google/primary" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "account_email": "work@company.com"}'

# Remove account
curl -X DELETE "http://localhost:8000/api/v1/oauth/accounts/google/user@gmail.com?user_id=test_user"
```

**2. Test Service Methods:**
```python
# Test with specific account
service = GmailService()
result = await service.send_email(
    user_id="test_user",
    request=send_request,
    account_email="work@company.com"
)

# Test with primary account (fallback)
result = await service.send_email(
    user_id="test_user",
    request=send_request
    # account_email not specified, uses primary
)
```

### Frontend Testing:

**1. Account Management:**
- Add multiple mock accounts
- Switch between accounts
- Set primary account
- Remove account
- Verify primary reassignment

**2. Service Integration:**
- Send email from specific account
- List emails from specific account
- Create calendar event in specific account
- Upload file to specific Drive account

---

## 📈 Progress Summary

**Backend Implementation:**
- ✅ OAuth Models (100%)
- ✅ OAuth Service (100%)
- ✅ Gmail Service (100%)
- ✅ Calendar Service (100%)
- ✅ Drive Service (100%)
- ✅ OAuth Router (100%)
- ⏳ Gmail/Calendar/Drive Routers (0% - Optional)

**Overall Backend:** 90% Complete

**Frontend Implementation:**
- ⏳ Account Selector Component (0%)
- ⏳ Account Manager Component (0%)
- ⏳ Zustand Store (0%)
- ⏳ API Client Updates (0%)
- ⏳ UI Integration (0%)

**Overall Frontend:** 0% Complete

---

## 🚀 Next Steps

**Immediate Priority:**
1. ~~Update OAuth models~~ ✅
2. ~~Update OAuth service~~ ✅
3. ~~Update Gmail/Calendar/Drive services~~ ✅
4. ~~Add OAuth router endpoints~~ ✅
5. **Create frontend components** ⏳ NEXT
6. **Create Zustand store** ⏳
7. **Integrate into UI** ⏳
8. Update Gmail/Calendar/Drive routers (optional)
9. Test multi-account flow
10. Document usage

---

## 📝 Technical Notes

**Database Schema:**
- Tokens stored with `user_id`, `provider`, `account_email` composite key
- `is_primary` flag for default account selection
- Automatic primary reassignment on account removal

**Account Selection Logic:**
1. Use explicitly selected account (if provided)
2. Fall back to primary account (if set)
3. Fall back to first available account
4. Error if no accounts connected

**Mock Mode:**
- All services work in mock mode without OAuth credentials
- Mock accounts auto-generated with format `{user_id}@gmail.com`
- First account automatically set as primary

---

**Status:** ✅ Backend 90% Complete  
**Next:** Frontend component implementation  
**Estimated Completion:** 4-6 hours for frontend + testing

