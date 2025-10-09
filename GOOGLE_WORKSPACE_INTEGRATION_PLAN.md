# Google Workspace Integration Plan for CPAS4

## Current Status: ❌ Not Implemented

## Implementation Requirements

### 1. **Google OAuth 2.0 Setup**

**Prerequisites:**
- Google Cloud Project
- OAuth 2.0 credentials (Client ID, Client Secret)
- Redirect URI configured

**Scopes Needed:**
```python
GOOGLE_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/contacts.readonly',
]
```

### 2. **Required Python Packages**

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2
pip install google-api-python-client
```

### 3. **Implementation Architecture**

#### **A. Models** (`backend/models/integration_models.py`)

```python
class GoogleWorkspaceConnection(BaseModel):
    connection_id: str
    user_id: str
    access_token: str  # Encrypted
    refresh_token: str  # Encrypted
    token_expiry: datetime
    scopes: List[str]
    connected_services: List[str]  # ['gmail', 'calendar', 'drive']
    created_at: datetime
    last_synced: datetime

class EmailMessage(BaseModel):
    message_id: str
    thread_id: str
    from_email: str
    to_emails: List[str]
    subject: str
    body: str
    received_at: datetime
    labels: List[str]
    attachments: List[Dict[str, Any]]

class CalendarEvent(BaseModel):
    event_id: str
    calendar_id: str
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: datetime
    attendees: List[str]
    location: Optional[str]
    meeting_link: Optional[str]
```

#### **B. Service** (`backend/services/google_workspace_service.py`)

```python
class GoogleWorkspaceService:
    async def authenticate(self, authorization_code: str) -> GoogleWorkspaceConnection
    async def refresh_token(self, connection_id: str) -> bool
    
    # Gmail
    async def sync_emails(self, connection_id: str, since: datetime) -> List[EmailMessage]
    async def send_email(self, connection_id: str, to: str, subject: str, body: str)
    async def search_emails(self, connection_id: str, query: str) -> List[EmailMessage]
    
    # Calendar
    async def sync_calendar(self, connection_id: str) -> List[CalendarEvent]
    async def create_event(self, connection_id: str, event: CalendarEventCreate) -> CalendarEvent
    async def update_event(self, connection_id: str, event_id: str, updates: Dict)
    
    # Contacts
    async def sync_contacts(self, connection_id: str) -> List[Contact]
    async def import_to_crm(self, connection_id: str) -> int  # Returns count imported
```

#### **C. Router** (`backend/routers/google_workspace_router.py`)

```python
@router.get("/auth/google/authorize")
async def initiate_google_auth()
    # Returns authorization URL

@router.post("/auth/google/callback")
async def google_auth_callback(code: str)
    # Exchanges code for tokens

@router.post("/gmail/sync")
async def sync_gmail(connection_id: str)

@router.post("/calendar/sync")
async def sync_calendar(connection_id: str)

@router.post("/contacts/import")
async def import_contacts_to_crm(connection_id: str)
```

### 4. **Security Considerations**

**Token Storage:**
- Store tokens encrypted in MongoDB
- Use `cryptography.fernet` for encryption
- Rotate encryption keys regularly

**Token Refresh:**
- Implement automatic token refresh before expiry
- Background task to check token validity every hour

**Webhook Security:**
- Verify Google webhook signatures
- Use HTTPS only
- Implement rate limiting

### 5. **Integration with Existing Modules**

**CRM Integration:**
- Auto-import Gmail contacts to CRM
- Link email threads to CRM contacts
- Track email interactions as activities

**Calendar Integration:**
- Sync calendar events to Projects module (meetings)
- Create tasks from calendar events
- Send meeting reminders via CPAS4

**Agent Builder Integration:**
- Agents can read emails and respond
- Agents can schedule meetings
- Agents can search email history

### 6. **Implementation Timeline**

**Phase 1: OAuth & Gmail (2 days)**
- OAuth 2.0 authentication flow
- Gmail sync and search
- Email sending

**Phase 2: Calendar (1 day)**
- Calendar sync
- Event creation/updates
- Meeting scheduling

**Phase 3: Contacts & Drive (1 day)**
- Contact import to CRM
- Drive file access (read-only)

**Phase 4: Agent Integration (1 day)**
- Connect agents to Gmail
- Connect agents to Calendar
- Testing and refinement

**Total: 5 days**

### 7. **Example Usage**

```python
# User authenticates
auth_url = await google_service.get_auth_url()
# User visits URL, grants permissions, returns with code

# Exchange code for tokens
connection = await google_service.authenticate(code)

# Sync emails
emails = await google_service.sync_emails(connection.connection_id, since=datetime.now() - timedelta(days=7))

# Import contacts to CRM
count = await google_service.import_to_crm(connection.connection_id)
print(f"Imported {count} contacts")

# Agent sends email
await agent_service.execute_tool(
    agent_id="agent_001",
    tool="send_email",
    params={
        "to": "client@example.com",
        "subject": "Follow-up",
        "body": "Thank you for the meeting..."
    }
)
```

### 8. **Recommended Approach**

**Option A: Full Implementation (5 days)**
- Complete Google Workspace integration
- All services (Gmail, Calendar, Contacts, Drive)
- Agent integration

**Option B: Gmail-Only (2 days)**
- Focus on email integration first
- Most valuable for business communication
- Expand to other services later

**Option C: Use Existing MCP Servers**
- Check if Google Workspace MCP servers exist
- Integrate via MCP Hub instead of direct API
- Faster implementation, less maintenance

**Recommendation: Option C → Option B → Option A**
1. First check MCP Hub for existing Google integrations
2. If not available, implement Gmail-only
3. Expand to full Workspace integration based on usage

### 9. **Privacy & Data Handling**

**Data Storage:**
- Store only metadata locally (subject, sender, date)
- Don't store full email bodies unless explicitly requested
- Implement data retention policies

**User Control:**
- Users can disconnect at any time
- Users can choose which services to connect
- Users can set sync frequency

**Compliance:**
- GDPR-compliant data handling
- User consent for data access
- Right to delete all synced data

