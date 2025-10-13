# Greta/CPAS4 Enhancement Analysis - Part 2

**Continued from ENHANCEMENT_ANALYSIS_COMPREHENSIVE.md**

---

### 6. Google Workspace Multi-Account Support 🔴 SIGNIFICANT DEVELOPMENT

**Current Status:** Single Account OAuth Implemented (Mock Mode)

**What Exists:**
- ✅ OAuth 2.0 service: `backend/services/oauth_service.py`
- ✅ Gmail service: `backend/services/gmail_service.py`
- ✅ Calendar service: `backend/services/calendar_service.py`
- ✅ Drive service: `backend/services/drive_service.py`
- ✅ OAuth flow endpoints: `/api/v1/oauth/*`
- ⚠️ Currently in MOCK MODE (no real Google credentials)
- ❌ No multi-account support
- ❌ No account switching UI

**Current OAuth Flow:**
```python
# Single account per user
token = await oauth_service.get_user_token(user_id, OAuthProvider.GOOGLE)
```

**What's Needed for Multi-Account:**

**Backend Changes (4-5 hours):**

1. **Update Token Model:**
```python
class OAuthToken(BaseModel):
    token_id: str
    user_id: str
    provider: OAuthProvider
    account_id: str  # NEW - unique identifier for this account
    account_email: str  # NEW - email address
    account_name: str  # NEW - display name
    is_primary: bool = False  # NEW - primary account flag
    # ... existing fields ...
```

2. **Update OAuth Service Methods:**
```python
# OLD: Single account
async def get_user_token(user_id: str, provider: OAuthProvider)

# NEW: Multi-account
async def get_user_tokens(user_id: str, provider: OAuthProvider) -> List[OAuthToken]
async def get_account_token(user_id: str, provider: OAuthProvider, account_id: str) -> OAuthToken
async def set_primary_account(user_id: str, provider: OAuthProvider, account_id: str)
async def add_account(user_id: str, provider: OAuthProvider, auth_code: str) -> OAuthToken
async def remove_account(user_id: str, provider: OAuthProvider, account_id: str)
```

3. **Update Service Methods to Accept Account ID:**
```python
# Gmail Service
async def send_email(user_id: str, account_id: str, request: EmailSendRequest)
async def list_emails(user_id: str, account_id: str, request: EmailListRequest)

# Calendar Service
async def create_event(user_id: str, account_id: str, event: CalendarEventCreate)
async def list_events(user_id: str, account_id: str, request: CalendarEventListRequest)
```

4. **Add Account Management Endpoints:**
```python
# backend/routers/oauth_router.py
@router.get("/accounts")
async def list_accounts(user_id: str, provider: OAuthProvider)

@router.post("/accounts/add")
async def add_account(user_id: str, provider: OAuthProvider, code: str)

@router.delete("/accounts/{account_id}")
async def remove_account(user_id: str, account_id: str)

@router.put("/accounts/{account_id}/primary")
async def set_primary(user_id: str, account_id: str)
```

**Frontend Changes (2-3 hours):**

1. **Create Account Selector Component:**
```typescript
// cpas4-ui/src/components/google/AccountSelector.tsx
interface GoogleAccount {
  account_id: string;
  email: string;
  name: string;
  is_primary: boolean;
  avatar_url?: string;
}

export function AccountSelector({
  accounts,
  selectedAccountId,
  onAccountChange,
  onAddAccount,
}: AccountSelectorProps) {
  return (
    <Select value={selectedAccountId} onValueChange={onAccountChange}>
      {accounts.map(account => (
        <SelectItem key={account.account_id} value={account.account_id}>
          <div className="flex items-center gap-2">
            <Avatar src={account.avatar_url} />
            <div>
              <div>{account.name}</div>
              <div className="text-xs text-muted-foreground">{account.email}</div>
            </div>
            {account.is_primary && <Badge>Primary</Badge>}
          </div>
        </SelectItem>
      ))}
      <SelectItem value="add_new">
        <Plus className="mr-2" /> Add Account
      </SelectItem>
    </Select>
  );
}
```

2. **Add Account Management UI:**
```typescript
// cpas4-ui/src/components/google/AccountManager.tsx
// - List all connected accounts
// - Add new account button (triggers OAuth flow)
// - Remove account button
// - Set primary account
// - Account status indicators
```

3. **Update Existing Google Components:**
```typescript
// Add account selector to:
// - Gmail compose dialog
// - Calendar event creation
// - Drive file upload
```

**Database Migration:**
```javascript
// Add account_id and account_email to existing tokens
db.oauth_tokens.updateMany(
  { account_id: { $exists: false } },
  { 
    $set: { 
      account_id: "$token_id",  // Use existing token_id as account_id
      account_email: "unknown@example.com",
      account_name: "Primary Account",
      is_primary: true
    }
  }
);
```

**Files to Create/Modify:**
```
Backend:
- backend/models/oauth_models.py (update OAuthToken model)
- backend/services/oauth_service.py (add multi-account methods)
- backend/services/gmail_service.py (add account_id parameter)
- backend/services/calendar_service.py (add account_id parameter)
- backend/services/drive_service.py (add account_id parameter)
- backend/routers/oauth_router.py (add account management endpoints)

Frontend:
- cpas4-ui/src/components/google/AccountSelector.tsx (NEW)
- cpas4-ui/src/components/google/AccountManager.tsx (NEW)
- cpas4-ui/src/stores/googleAccountStore.ts (NEW - Zustand store)
- cpas4-ui/src/api/googleApi.ts (NEW - account API calls)
```

**Testing Plan:**
1. Test OAuth flow with multiple accounts
2. Test account switching in Gmail/Calendar/Drive
3. Test primary account functionality
4. Test account removal
5. Test token refresh for multiple accounts

**Estimated Effort:** 6-8 hours  
**Priority:** ⭐⭐⭐ HIGH (Business value)  
**Dependencies:** Google Cloud OAuth credentials required  

**Note:** Currently in MOCK MODE. To enable real Google integration:
1. Create Google Cloud Project
2. Enable Gmail, Calendar, Drive APIs
3. Create OAuth 2.0 credentials
4. Set environment variables:
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
   - `GOOGLE_REDIRECT_URI`

---

### 7. Lead Generation Research Module 🔴 SIGNIFICANT DEVELOPMENT

**Current Status:** OSINT Tools Available, No Automated Lead Gen

**What Exists:**
- ✅ Comprehensive OSINT toolkit: `backend/services/osint_service.py`
- ✅ Research automation: `backend/services/research_automation.py`
- ✅ OSINT tools: 50+ tools across 8 categories
- ✅ Web search, social monitoring, domain intelligence
- ❌ No lead generation workflow
- ❌ No continuous research tracking
- ❌ No lead scoring/qualification

**OSINT Tools Available:**
```python
# Marketing Research
- Google Trends, SEMrush, Ahrefs, SimilarWeb, BuzzSumo

# Social Monitoring
- LinkedIn, Twitter, Facebook, Instagram, GitHub

# Domain Intelligence
- WHOIS, DNS lookup, SSL certificates, subdomain enumeration

# Competitive Analysis
- Market intelligence, competitor tracking, brand monitoring

# Threat Intelligence
- Shodan, Censys, VirusTotal, AbuseIPDB
```

**Implementation Plan (8-12 hours):**

**Phase 1: Lead Generation Service (4-5 hours)**

Create `backend/services/lead_generation_service.py`:

```python
class LeadGenerationService:
    """Automated lead generation and research"""
    
    async def create_lead_campaign(self, config: LeadCampaignConfig) -> str:
        """
        Create automated lead generation campaign
        
        Args:
            config: Campaign configuration
                - target_industry: str
                - target_location: str
                - company_size: str
                - keywords: List[str]
                - research_depth: str (shallow, medium, deep)
                - frequency: str (daily, weekly, monthly)
        
        Returns:
            campaign_id
        """
        
    async def research_lead(self, lead_id: str) -> LeadResearchResult:
        """
        Perform deep research on a lead
        
        Research includes:
        - Company information (website, size, revenue, funding)
        - Key personnel (executives, decision makers)
        - Recent news and press releases
        - Social media presence and activity
        - Technology stack (if applicable)
        - Competitors and market position
        - Contact information
        - Engagement opportunities
        """
        
    async def score_lead(self, lead_id: str) -> LeadScore:
        """
        Score lead based on research findings
        
        Scoring criteria:
        - Company fit (industry, size, location)
        - Engagement potential (social activity, recent news)
        - Decision maker accessibility
        - Budget indicators
        - Timing signals (hiring, funding, expansion)
        """
        
    async def track_lead_changes(self, lead_id: str) -> List[LeadChange]:
        """
        Track changes in lead status over time
        
        Monitors:
        - Company news and announcements
        - Personnel changes
        - Funding rounds
        - Product launches
        - Social media activity
        - Website changes
        """
```

**Phase 2: Research Workflow (2-3 hours)**

```python
class LeadResearchWorkflow:
    """Automated research workflow for leads"""
    
    async def execute_research_workflow(self, lead: Lead) -> ResearchReport:
        """
        Execute multi-stage research workflow
        
        Stages:
        1. Company Discovery (OSINT tools)
        2. Personnel Identification (LinkedIn, company website)
        3. News & Events Monitoring (RSS feeds, news APIs)
        4. Social Media Analysis (Twitter, LinkedIn activity)
        5. Technology Stack Detection (BuiltWith, Wappalyzer)
        6. Competitive Landscape (market research)
        7. Contact Information Enrichment
        8. Engagement Opportunity Identification
        """
        
        # Stage 1: Company Discovery
        company_data = await self._discover_company(lead.company_name)
        
        # Stage 2: Personnel Identification
        key_people = await self._identify_decision_makers(company_data)
        
        # Stage 3: News Monitoring
        recent_news = await self._monitor_news(lead.company_name)
        
        # Stage 4: Social Analysis
        social_activity = await self._analyze_social_media(company_data)
        
        # Stage 5: Tech Stack
        tech_stack = await self._detect_technology(company_data.website)
        
        # Stage 6: Competitive Analysis
        competitors = await self._analyze_competitors(company_data)
        
        # Stage 7: Contact Enrichment
        contacts = await self._enrich_contacts(key_people)
        
        # Stage 8: Opportunities
        opportunities = await self._identify_opportunities(
            company_data, recent_news, social_activity
        )
        
        return ResearchReport(
            lead_id=lead.lead_id,
            company_data=company_data,
            key_people=key_people,
            recent_news=recent_news,
            social_activity=social_activity,
            tech_stack=tech_stack,
            competitors=competitors,
            contacts=contacts,
            opportunities=opportunities,
            lead_score=self._calculate_score(...)
        )
```

**Phase 3: Continuous Monitoring (2-3 hours)**

```python
class LeadMonitoringService:
    """Continuous monitoring of leads"""
    
    async def start_monitoring(self, lead_id: str, frequency: str):
        """Start continuous monitoring of a lead"""
        
    async def check_for_updates(self, lead_id: str) -> List[LeadUpdate]:
        """Check for updates on a lead"""
        
    async def send_alerts(self, lead_id: str, updates: List[LeadUpdate]):
        """Send alerts for important lead updates"""
```

**Phase 4: Frontend UI (2-3 hours)**

```typescript
// cpas4-ui/src/components/leads/LeadGenerationDashboard.tsx
interface Lead {
  lead_id: string;
  company_name: string;
  industry: string;
  location: string;
  lead_score: number;
  status: 'new' | 'researching' | 'qualified' | 'contacted' | 'converted';
  last_research_date: string;
  next_research_date: string;
  key_insights: string[];
  opportunities: string[];
}

export function LeadGenerationDashboard() {
  return (
    <div className="space-y-6">
      {/* Campaign Overview */}
      <CampaignStats />
      
      {/* Lead List */}
      <LeadList
        leads={leads}
        onResearch={handleResearch}
        onViewDetails={handleViewDetails}
      />
      
      {/* Research Timeline */}
      <ResearchTimeline />
      
      {/* Insights & Opportunities */}
      <InsightsPanel />
    </div>
  );
}
```

**Database Schema:**

```javascript
// leads collection
{
  lead_id: "lead_123",
  campaign_id: "campaign_456",
  company_name: "Acme Corp",
  industry: "Technology",
  location: "San Francisco, CA",
  company_size: "50-200",
  website: "https://acme.com",
  lead_score: 85,
  status: "qualified",
  research_history: [
    {
      research_id: "research_789",
      date: "2025-10-12T10:00:00Z",
      findings: {...},
      score: 85
    }
  ],
  key_people: [
    {
      name: "John Doe",
      title: "CEO",
      linkedin: "https://linkedin.com/in/johndoe",
      email: "john@acme.com"
    }
  ],
  recent_news: [...],
  opportunities: [...],
  monitoring_active: true,
  next_research_date: "2025-10-19T10:00:00Z",
  created_at: "2025-10-01T00:00:00Z",
  updated_at: "2025-10-12T10:00:00Z"
}
```

**Files to Create:**
```
Backend:
- backend/services/lead_generation_service.py (NEW)
- backend/services/lead_research_workflow.py (NEW)
- backend/services/lead_monitoring_service.py (NEW)
- backend/models/lead_models.py (NEW)
- backend/routers/lead_generation_router.py (NEW)

Frontend:
- cpas4-ui/src/components/leads/LeadGenerationDashboard.tsx (NEW)
- cpas4-ui/src/components/leads/LeadList.tsx (NEW)
- cpas4-ui/src/components/leads/LeadDetails.tsx (NEW)
- cpas4-ui/src/components/leads/CampaignManager.tsx (NEW)
- cpas4-ui/src/stores/leadStore.ts (NEW)
- cpas4-ui/src/api/leadApi.ts (NEW)
```

**Estimated Effort:** 8-12 hours  
**Priority:** ⭐⭐ MEDIUM (High value but not critical)  
**Dependencies:** OSINT tools (already available)  

---

### 8. Continuous Learning & Training ✅ FULLY IMPLEMENTED

**Current Status:** ACTIVE AND OPERATIONAL

**What Exists:**
- ✅ Learning Service: `backend/services/learning_service.py`
- ✅ Meta Learning Engine: `backend/services/meta_learning_engine.py`
- ✅ Training Platform: `backend/services/training_platform_service.py`
- ✅ Interactive Training: `backend/services/interactive_training.py`
- ✅ Continual Learning System in intelligent agents
- ✅ HRM (Hierarchical Reasoning Model) coordination
- ✅ PAI interaction recording and analysis
- ✅ Llama3 model training integration

**Key Features:**

1. **PAI Interaction Recording:**
```python
await learning_service.record_pai_interaction({
    "user_id": "user_123",
    "type": "conversation",
    "category": "general",
    "strategic_reasoning": "...",
    "tactical_reasoning": "...",
    "operational_reasoning": "...",
    "response_quality": 0.9,
    "user_satisfaction": 0.85,
    "context_relevance": 0.92
})
```

2. **Hierarchical Reasoning Model (HRM):**
- Strategic Layer: Long-term patterns and goals
- Tactical Layer: Medium-term adaptations
- Operational Layer: Immediate response refinements

3. **Training Data Generation:**
```python
training_samples = await learning_service.generate_training_samples()
# Generates samples from all three HRM layers
```

4. **Model Training Trigger:**
```python
success = await learning_service.trigger_llama3_training(training_data_path)
```

5. **Continual Learning:**
- Reinforcement learning
- Supervised learning
- Unsupervised learning
- Meta learning

**Privacy Controls:**
- User data anonymization
- Opt-in/opt-out mechanisms
- Data retention policies
- GDPR compliance ready

**Verification:**
```bash
# Check learning service logs
tail -f backend/logs/greta.log | grep "PAI interaction recorded"

# Check training cycles
# Look for: "Learning cycle completed"
```

**Status:** ✅ **FULLY OPERATIONAL**  
**Action Required:** None - system is learning from interactions  
**Estimated Effort:** 0 hours (already complete)  
**Priority:** N/A (Active)  

---

**Document continues in ENHANCEMENT_ANALYSIS_PART3.md...**

