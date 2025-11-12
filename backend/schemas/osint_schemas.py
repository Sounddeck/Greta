"""
OSINT Database Schemas for CPAS5
MongoDB collection schemas for OSINT results, contacts, and leads
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ContactSource(str, Enum):
    """Source of contact information"""
    HUNTER_IO = "hunter_io"
    WEB_SCRAPING = "web_scraping"
    PHONE_LOOKUP = "phone_lookup"
    MANUAL_ENTRY = "manual_entry"
    API_IMPORT = "api_import"
    OTHER = "other"


class ContactStatus(str, Enum):
    """Status of contact"""
    NEW = "new"
    VERIFIED = "verified"
    INVALID = "invalid"
    BOUNCED = "bounced"
    OPTED_OUT = "opted_out"


class LeadStatus(str, Enum):
    """Status of lead"""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    CONVERTED = "converted"
    LOST = "lost"


# ==================== OSINT RESULTS ====================

class OSINTResult(BaseModel):
    """OSINT tool execution result"""
    result_id: str = Field(..., description="Unique result ID")
    tool_id: str = Field(..., description="Tool that generated this result")
    agent_id: str = Field(..., description="Agent that executed the tool")
    execution_id: str = Field(..., description="Execution ID from tool registry")
    
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters used")
    result: Dict[str, Any] = Field(default_factory=dict, description="Raw result data")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Expiration date for data retention")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result_id": "osint_res_abc123",
                "tool_id": "osint_hunter_domain_search",
                "agent_id": "research_agent_001",
                "execution_id": "exec_xyz789",
                "parameters": {
                    "domain": "example.com",
                    "limit": 10
                },
                "result": {
                    "emails": [
                        {
                            "email": "john.doe@example.com",
                            "first_name": "John",
                            "last_name": "Doe",
                            "position": "CEO",
                            "confidence": 95
                        }
                    ],
                    "total_found": 1
                },
                "metadata": {
                    "execution_time": 2.5,
                    "api_calls_used": 1,
                    "cost": 0.01
                },
                "created_at": "2025-01-15T10:30:00Z",
                "expires_at": "2025-02-15T10:30:00Z"
            }
        }


# ==================== CONTACTS ====================

class SocialProfile(BaseModel):
    """Social media profile"""
    platform: str = Field(..., description="Platform name (linkedin, twitter, etc.)")
    url: str = Field(..., description="Profile URL")
    username: Optional[str] = Field(None, description="Username/handle")
    verified: bool = Field(False, description="Whether profile is verified")


class Contact(BaseModel):
    """Contact information"""
    contact_id: str = Field(..., description="Unique contact ID")
    
    # Basic information
    email: Optional[EmailStr] = Field(None, description="Email address")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    full_name: Optional[str] = Field(None, description="Full name")
    
    # Phone information
    phone: Optional[str] = Field(None, description="Phone number")
    phone_country_code: Optional[int] = Field(None, description="Phone country code")
    phone_type: Optional[str] = Field(None, description="Phone type (mobile, fixed_line, etc.)")
    
    # Company information
    company: Optional[str] = Field(None, description="Company name")
    domain: Optional[str] = Field(None, description="Company domain")
    position: Optional[str] = Field(None, description="Job position/title")
    department: Optional[str] = Field(None, description="Department")
    
    # Social profiles
    social_profiles: List[SocialProfile] = Field(default_factory=list, description="Social media profiles")
    
    # Location
    location: Optional[str] = Field(None, description="Location/city")
    country: Optional[str] = Field(None, description="Country")
    timezone: Optional[str] = Field(None, description="Timezone")
    
    # Metadata
    source: ContactSource = Field(ContactSource.OTHER, description="Source of contact")
    source_tool: Optional[str] = Field(None, description="Tool ID that found this contact")
    source_url: Optional[str] = Field(None, description="URL where contact was found")
    
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0-1)")
    verified: bool = Field(False, description="Whether contact is verified")
    status: ContactStatus = Field(ContactStatus.NEW, description="Contact status")
    
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified_at: Optional[datetime] = Field(None, description="Last verification date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "contact_id": "contact_abc123",
                "email": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "full_name": "John Doe",
                "phone": "+1-555-0123",
                "phone_country_code": 1,
                "phone_type": "mobile",
                "company": "Example Corp",
                "domain": "example.com",
                "position": "CEO",
                "department": "Executive",
                "social_profiles": [
                    {
                        "platform": "linkedin",
                        "url": "https://linkedin.com/in/johndoe",
                        "username": "johndoe",
                        "verified": True
                    }
                ],
                "location": "San Francisco, CA",
                "country": "United States",
                "timezone": "America/Los_Angeles",
                "source": "hunter_io",
                "source_tool": "osint_hunter_domain_search",
                "confidence_score": 0.95,
                "verified": True,
                "status": "verified",
                "tags": ["lead", "decision_maker", "tech"],
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-15T10:30:00Z"
            }
        }


# ==================== LEADS ====================

class Lead(BaseModel):
    """Lead information for sales/marketing"""
    lead_id: str = Field(..., description="Unique lead ID")
    contact_id: str = Field(..., description="Associated contact ID")
    
    # Lead scoring
    lead_score: int = Field(0, ge=0, le=100, description="Lead score (0-100)")
    lead_status: LeadStatus = Field(LeadStatus.NEW, description="Lead status")
    
    # Qualification
    qualified: bool = Field(False, description="Whether lead is qualified")
    qualification_criteria: Dict[str, bool] = Field(default_factory=dict, description="Qualification criteria met")
    
    # Engagement
    contacted: bool = Field(False, description="Whether lead has been contacted")
    contact_attempts: int = Field(0, description="Number of contact attempts")
    last_contact_date: Optional[datetime] = Field(None, description="Last contact date")
    next_follow_up: Optional[datetime] = Field(None, description="Next follow-up date")
    
    # Source
    source_campaign: Optional[str] = Field(None, description="Source campaign")
    source_channel: Optional[str] = Field(None, description="Source channel")
    
    # Assignment
    assigned_to: Optional[str] = Field(None, description="Agent/user assigned to this lead")
    
    # Notes and tags
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    converted_at: Optional[datetime] = Field(None, description="Conversion date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lead_id": "lead_abc123",
                "contact_id": "contact_abc123",
                "lead_score": 85,
                "lead_status": "qualified",
                "qualified": True,
                "qualification_criteria": {
                    "has_email": True,
                    "has_phone": True,
                    "decision_maker": True,
                    "target_industry": True
                },
                "contacted": True,
                "contact_attempts": 2,
                "last_contact_date": "2025-01-14T15:00:00Z",
                "next_follow_up": "2025-01-17T10:00:00Z",
                "source_campaign": "Q1_2025_Outreach",
                "source_channel": "osint_research",
                "assigned_to": "sales_agent_001",
                "tags": ["high_priority", "tech_industry", "decision_maker"],
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-01-15T14:30:00Z"
            }
        }


# ==================== API KEYS ====================

class OSINTAPIKey(BaseModel):
    """OSINT API key storage (encrypted)"""
    key_id: str = Field(..., description="Unique key ID")
    service: str = Field(..., description="Service name (hunter_io, etc.)")
    api_key_encrypted: str = Field(..., description="Encrypted API key")
    
    # Metadata
    owner_id: Optional[str] = Field(None, description="Owner user/agent ID")
    description: Optional[str] = Field(None, description="Key description")
    
    # Usage limits
    rate_limit: Optional[int] = Field(None, description="Rate limit (calls per minute)")
    daily_quota: Optional[int] = Field(None, description="Daily quota")
    monthly_quota: Optional[int] = Field(None, description="Monthly quota")
    
    # Usage tracking
    calls_today: int = Field(0, description="Calls made today")
    calls_this_month: int = Field(0, description="Calls made this month")
    total_calls: int = Field(0, description="Total calls made")
    
    # Status
    active: bool = Field(True, description="Whether key is active")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = Field(None, description="Last usage date")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")


# ==================== USAGE LOGS ====================

class OSINTUsageLog(BaseModel):
    """OSINT tool usage log"""
    log_id: str = Field(..., description="Unique log ID")
    tool_id: str = Field(..., description="Tool ID")
    agent_id: str = Field(..., description="Agent ID")
    execution_id: str = Field(..., description="Execution ID")
    
    # Request details
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    
    # Response details
    success: bool = Field(..., description="Whether execution was successful")
    execution_time: float = Field(..., description="Execution time in seconds")
    
    # API usage (if applicable)
    api_calls_used: int = Field(0, description="Number of API calls used")
    cost: float = Field(0.0, description="Cost in USD")
    
    # Results summary
    results_count: int = Field(0, description="Number of results returned")
    contacts_found: int = Field(0, description="Number of contacts found")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)

