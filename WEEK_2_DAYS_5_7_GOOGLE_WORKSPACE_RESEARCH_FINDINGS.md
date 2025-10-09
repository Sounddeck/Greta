# Week 2, Days 5-7: Google Workspace Integration - Research Findings

## 📊 Research Status

**Date**: October 9, 2025  
**Phase**: Week 2 - Advanced Security + Google Workspace  
**Module**: Google Workspace Integration (Research Phase)  
**Status**: ✅ **RESEARCH COMPLETE**

---

## 🔍 Research Findings

### **Existing Google Workspace MCP Servers**

#### **1. epaproditus/google-workspace-mcp-server**

**Repository**: https://github.com/epaproditus/google-workspace-mcp-server  
**Stars**: 25  
**License**: MIT  
**Language**: TypeScript/JavaScript (Node.js)  
**Last Updated**: December 31, 2024

**Supported Features**:
- ✅ **Gmail API**:
  - `list_emails` - List recent emails with filtering
  - `search_emails` - Advanced email search with Gmail query syntax
  - `send_email` - Send emails with CC/BCC support
  - `modify_email` - Modify email labels (archive, trash, mark read/unread)

- ✅ **Google Calendar API**:
  - `list_events` - List upcoming events with date range filtering
  - `create_event` - Create calendar events with attendees
  - `update_event` - Update existing events
  - `delete_event` - Delete events

**Missing Features**:
- ❌ **Google Drive API** - Not supported
- ❌ **Google Sheets API** - Not supported
- ❌ **Google Docs API** - Not supported

**OAuth Implementation**:
- Uses OAuth 2.0 authorization code flow
- Requires Google Cloud Project with OAuth credentials
- Stores refresh token for persistent access
- Required scopes:
  - `https://www.googleapis.com/auth/gmail.modify`
  - `https://www.googleapis.com/auth/gmail.send`
  - `https://www.googleapis.com/auth/calendar`

**Setup Requirements**:
1. Google Cloud Console project
2. Enable Gmail API and Calendar API
3. Create OAuth 2.0 credentials (Web application)
4. Set redirect URI: `http://localhost:4100/code`
5. Run `get-refresh-token.js` to obtain refresh token
6. Configure MCP settings with client ID, secret, and refresh token

#### **2. Other MCP Servers Found**

**Google Apps Script MCP Server** (Medium article):
- Uses Google Apps Script to host MCP server
- Leverages GAS integration with Google Workspace
- Experimental approach, not production-ready

**Data Commons MCP Server** (Google):
- Official Google MCP server for Data Commons
- Not related to Google Workspace APIs
- Focus on public datasets, not productivity tools

---

## 🎯 Recommendation: Build Custom Python Implementation

### **Rationale**

1. **Missing Features** ⚠️
   - Existing MCP server only supports Gmail and Calendar
   - We need Drive, Sheets, and Docs support
   - Would require building 60% of functionality anyway

2. **Language Mismatch** ⚠️
   - Existing server is Node.js/TypeScript
   - Our CPAS4/GRETA backend is Python/FastAPI
   - Integration would require running Node.js process alongside Python

3. **Integration Complexity** ⚠️
   - MCP server runs as separate process
   - Would need IPC (Inter-Process Communication) between Python and Node.js
   - Additional infrastructure and complexity

4. **Full Control** ✅
   - Custom implementation gives complete control over features
   - Can integrate directly with FastAPI backend
   - No dependency on external MCP server maintenance

5. **Python Libraries Available** ✅
   - Google provides excellent Python client libraries
   - `google-api-python-client` - Official Google API client
   - `google-auth` - OAuth 2.0 authentication
   - `google-auth-oauthlib` - OAuth flow helpers
   - `google-auth-httplib2` - HTTP transport

6. **Seamless Integration** ✅
   - Direct integration with existing FastAPI routers
   - Consistent with other CPAS4/GRETA modules
   - Same database, same audit logging, same RBAC

---

## 📋 Implementation Plan

### **Phase 1: OAuth 2.0 Foundation** (Day 5)

**Models** (`backend/models/oauth_models.py`):
- `OAuthProvider` enum (Google, Microsoft, etc.)
- `OAuthToken` model (access_token, refresh_token, expires_at)
- `OAuthConfig` model (client_id, client_secret, scopes)
- `OAuthState` model (state parameter for CSRF protection)

**Service** (`backend/services/oauth_service.py`):
- `initiate_oauth_flow()` - Generate authorization URL
- `handle_oauth_callback()` - Exchange code for tokens
- `refresh_access_token()` - Refresh expired tokens
- `revoke_token()` - Revoke OAuth tokens
- `get_user_tokens()` - Retrieve stored tokens

**Router** (`backend/routers/oauth_router.py`):
- `GET /api/v1/oauth/authorize` - Initiate OAuth flow
- `GET /api/v1/oauth/callback` - Handle OAuth callback
- `POST /api/v1/oauth/refresh` - Refresh tokens
- `POST /api/v1/oauth/revoke` - Revoke tokens

**Required Google OAuth Scopes**:
```python
GOOGLE_WORKSPACE_SCOPES = [
    # Gmail
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly',
    
    # Calendar
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
    
    # Drive
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    
    # Sheets
    'https://www.googleapis.com/auth/spreadsheets',
    
    # Docs
    'https://www.googleapis.com/auth/documents',
]
```

### **Phase 2: Gmail Integration** (Day 5)

**Models** (`backend/models/gmail_models.py`):
- `GmailMessage` model
- `GmailThread` model
- `GmailLabel` model
- `GmailAttachment` model

**Service** (`backend/services/gmail_service.py`):
- `send_email()` - Send emails with attachments
- `read_email()` - Read email by ID
- `list_emails()` - List emails with filters
- `search_emails()` - Search emails by query
- `create_label()` - Create custom label
- `modify_labels()` - Add/remove labels
- `delete_email()` - Delete email

**Router** (`backend/routers/gmail_router.py`):
- `POST /api/v1/gmail/send` - Send email
- `GET /api/v1/gmail/messages/{message_id}` - Get email
- `GET /api/v1/gmail/messages` - List emails
- `GET /api/v1/gmail/search` - Search emails
- `POST /api/v1/gmail/labels` - Create label
- `PUT /api/v1/gmail/messages/{message_id}/labels` - Modify labels

### **Phase 3: Google Calendar Integration** (Day 6)

**Models** (`backend/models/calendar_models.py`):
- `CalendarEvent` model
- `CalendarAttendee` model
- `CalendarReminder` model
- `CalendarRecurrence` model

**Service** (`backend/services/calendar_service.py`):
- `create_event()` - Create calendar event
- `get_event()` - Get event by ID
- `list_events()` - List events with date range
- `update_event()` - Update event
- `delete_event()` - Delete event
- `add_attendees()` - Add attendees to event

**Router** (`backend/routers/calendar_router.py`):
- `POST /api/v1/calendar/events` - Create event
- `GET /api/v1/calendar/events/{event_id}` - Get event
- `GET /api/v1/calendar/events` - List events
- `PUT /api/v1/calendar/events/{event_id}` - Update event
- `DELETE /api/v1/calendar/events/{event_id}` - Delete event

### **Phase 4: Google Drive Integration** (Day 6)

**Models** (`backend/models/drive_models.py`):
- `DriveFile` model
- `DriveFolder` model
- `DrivePermission` model
- `DriveFileMetadata` model

**Service** (`backend/services/drive_service.py`):
- `upload_file()` - Upload file to Drive
- `download_file()` - Download file from Drive
- `list_files()` - List files with filters
- `create_folder()` - Create folder
- `share_file()` - Share file with permissions
- `delete_file()` - Delete file
- `move_file()` - Move file to folder

**Router** (`backend/routers/drive_router.py`):
- `POST /api/v1/drive/files` - Upload file
- `GET /api/v1/drive/files/{file_id}` - Download file
- `GET /api/v1/drive/files` - List files
- `POST /api/v1/drive/folders` - Create folder
- `POST /api/v1/drive/files/{file_id}/permissions` - Share file
- `DELETE /api/v1/drive/files/{file_id}` - Delete file

### **Phase 5: Google Sheets Integration** (Day 7)

**Models** (`backend/models/sheets_models.py`):
- `Spreadsheet` model
- `Sheet` model
- `CellData` model
- `Range` model

**Service** (`backend/services/sheets_service.py`):
- `create_spreadsheet()` - Create new spreadsheet
- `read_spreadsheet()` - Read spreadsheet data
- `write_spreadsheet()` - Write data to spreadsheet
- `update_cells()` - Update specific cells
- `append_rows()` - Append rows to sheet
- `create_sheet()` - Create new sheet in spreadsheet

**Router** (`backend/routers/sheets_router.py`):
- `POST /api/v1/sheets/spreadsheets` - Create spreadsheet
- `GET /api/v1/sheets/spreadsheets/{spreadsheet_id}` - Read spreadsheet
- `PUT /api/v1/sheets/spreadsheets/{spreadsheet_id}` - Write to spreadsheet
- `PUT /api/v1/sheets/spreadsheets/{spreadsheet_id}/cells` - Update cells
- `POST /api/v1/sheets/spreadsheets/{spreadsheet_id}/append` - Append rows

### **Phase 6: Google Docs Integration** (Day 7)

**Models** (`backend/models/docs_models.py`):
- `Document` model
- `DocumentContent` model
- `TextRun` model
- `Paragraph` model

**Service** (`backend/services/docs_service.py`):
- `create_document()` - Create new document
- `read_document()` - Read document content
- `update_document()` - Update document content
- `append_text()` - Append text to document
- `insert_text()` - Insert text at position
- `format_text()` - Apply text formatting

**Router** (`backend/routers/docs_router.py`):
- `POST /api/v1/docs/documents` - Create document
- `GET /api/v1/docs/documents/{document_id}` - Read document
- `PUT /api/v1/docs/documents/{document_id}` - Update document
- `POST /api/v1/docs/documents/{document_id}/append` - Append text

---

## 🔧 Technical Requirements

### **Dependencies**

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

**Packages**:
- `google-auth` (2.x) - Google authentication library
- `google-auth-oauthlib` (1.x) - OAuth 2.0 flow helpers
- `google-auth-httplib2` (0.2.x) - HTTP transport for Google APIs
- `google-api-python-client` (2.x) - Google API client library

### **Google Cloud Setup**

1. **Create Google Cloud Project**:
   - Go to https://console.cloud.google.com/
   - Create new project or select existing
   - Note the Project ID

2. **Enable APIs**:
   - Gmail API
   - Google Calendar API
   - Google Drive API
   - Google Sheets API
   - Google Docs API

3. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Application type: "Web application"
   - Authorized redirect URIs:
     - `http://localhost:8000/api/v1/oauth/callback`
     - `https://your-domain.com/api/v1/oauth/callback` (production)
   - Download credentials JSON

4. **Configure OAuth Consent Screen**:
   - User type: External (for testing) or Internal (for organization)
   - Add required scopes
   - Add test users (for external apps)

### **Environment Variables**

```bash
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/oauth/callback

# Optional: Service Account (for server-to-server)
GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/service-account-key.json
```

---

## ⚠️ Implementation Constraints

### **Current Limitations**

1. **No Google Cloud Credentials Available**:
   - Cannot test OAuth flow without real credentials
   - Cannot make actual API calls to Google services
   - Implementation will be **code-complete but untested**

2. **OAuth Flow Requires User Interaction**:
   - Users must authorize app in browser
   - Cannot fully automate testing without credentials

3. **Rate Limits**:
   - Gmail API: 250 quota units/user/second
   - Calendar API: 1,000,000 queries/day
   - Drive API: 1,000 queries/100 seconds/user
   - Sheets API: 100 requests/100 seconds/user
   - Docs API: 300 requests/minute/user

### **Recommended Approach**

Given the constraints, I recommend:

1. **Create Complete Code Structure**: Implement all models, services, and routers
2. **Mock Testing**: Create comprehensive tests with mocked Google API responses
3. **Documentation**: Provide detailed setup instructions for users with Google Cloud access
4. **Placeholder Implementation**: Mark sections requiring real credentials with TODO comments
5. **Integration Guide**: Document how to obtain and configure Google Cloud credentials

---

## 🎯 Next Steps

**Option 1: Full Implementation (Requires Google Cloud Credentials)**
- Implement all 6 phases (OAuth, Gmail, Calendar, Drive, Sheets, Docs)
- Create comprehensive tests with real API calls
- Verify all functionality end-to-end
- **Timeline**: 3 days (as planned)
- **Blocker**: Requires Google Cloud Project with OAuth credentials

**Option 2: Code-Complete Implementation (No Credentials Required)**
- Implement all models, services, and routers
- Create mock tests simulating Google API responses
- Document setup process for users with credentials
- **Timeline**: 2 days
- **Status**: Can proceed immediately

**Option 3: Partial Implementation (Gmail + Calendar Only)**
- Implement OAuth + Gmail + Calendar (matching existing MCP server)
- Skip Drive, Sheets, Docs for now
- **Timeline**: 1.5 days
- **Status**: Can proceed immediately

---

## 💡 Recommendation

I recommend **Option 2: Code-Complete Implementation** because:

1. ✅ Provides complete codebase for all Google Workspace services
2. ✅ Users can test with their own Google Cloud credentials
3. ✅ Demonstrates full architecture and integration patterns
4. ✅ Can be tested with mocked responses
5. ✅ Production-ready code structure
6. ✅ Comprehensive documentation for setup

**Would you like me to proceed with Option 2 (Code-Complete Implementation with Mock Testing)?**

This will include:
- Complete models for all services
- Complete services with Google API integration
- Complete routers with all endpoints
- Mock tests simulating Google API responses
- Comprehensive setup documentation
- Integration with existing CPAS4/GRETA modules

**Estimated Timeline**: 2 days (faster than original 3 days due to no real API testing)

