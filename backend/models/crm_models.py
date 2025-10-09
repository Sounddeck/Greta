"""
CRM Models for CPAS4
Comprehensive data models for Customer Relationship Management
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ContactStatus(str, Enum):
    """Contact status"""
    LEAD = "lead"
    PROSPECT = "prospect"
    CUSTOMER = "customer"
    INACTIVE = "inactive"
    LOST = "lost"


class CompanySize(str, Enum):
    """Company size categories"""
    STARTUP = "1-10"
    SMALL = "11-50"
    MEDIUM = "51-200"
    LARGE = "201-1000"
    ENTERPRISE = "1000+"


class DealStage(str, Enum):
    """Deal pipeline stages"""
    LEAD = "lead"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class ActivityType(str, Enum):
    """Activity types"""
    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    NOTE = "note"
    TASK = "task"
    DEMO = "demo"


class Priority(str, Enum):
    """Priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# ============================================================================
# CONTACT MODELS
# ============================================================================

class ContactBase(BaseModel):
    """Base contact information"""
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    title: Optional[str] = None
    company_id: Optional[str] = None
    status: ContactStatus = ContactStatus.LEAD
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class ContactCreate(ContactBase):
    """Contact creation request"""
    pass


class ContactUpdate(BaseModel):
    """Contact update request"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    title: Optional[str] = None
    company_id: Optional[str] = None
    status: Optional[ContactStatus] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    lead_score: Optional[float] = None


class Contact(ContactBase):
    """Complete contact model"""
    contact_id: str
    lead_score: float = 0.0
    last_contacted: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"
    
    class Config:
        json_schema_extra = {
            "example": {
                "contact_id": "contact_abc123",
                "first_name": "John",
                "last_name": "Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123",
                "title": "CEO",
                "company_id": "company_xyz789",
                "status": "customer",
                "lead_score": 85.5,
                "tags": ["vip", "tech"]
            }
        }


# ============================================================================
# COMPANY MODELS
# ============================================================================

class CompanyBase(BaseModel):
    """Base company information"""
    name: str
    industry: Optional[str] = None
    size: Optional[CompanySize] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class CompanyCreate(CompanyBase):
    """Company creation request"""
    pass


class CompanyUpdate(BaseModel):
    """Company update request"""
    name: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[CompanySize] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class Company(CompanyBase):
    """Complete company model"""
    company_id: str
    contact_count: int = 0
    deal_count: int = 0
    total_revenue: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"


# ============================================================================
# DEAL MODELS
# ============================================================================

class DealBase(BaseModel):
    """Base deal information"""
    title: str
    description: Optional[str] = None
    value: float
    stage: DealStage = DealStage.LEAD
    probability: float = 0.0
    expected_close_date: Optional[datetime] = None
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class DealCreate(DealBase):
    """Deal creation request"""
    pass


class DealUpdate(BaseModel):
    """Deal update request"""
    title: Optional[str] = None
    description: Optional[str] = None
    value: Optional[float] = None
    stage: Optional[DealStage] = None
    probability: Optional[float] = None
    expected_close_date: Optional[datetime] = None
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    actual_close_date: Optional[datetime] = None


class Deal(DealBase):
    """Complete deal model"""
    deal_id: str
    actual_close_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"
    owner: str = "system"


# ============================================================================
# ACTIVITY MODELS
# ============================================================================

class ActivityBase(BaseModel):
    """Base activity information"""
    activity_type: ActivityType
    subject: str
    description: Optional[str] = None
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    deal_id: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: bool = False
    priority: Priority = Priority.MEDIUM
    tags: List[str] = Field(default_factory=list)


class ActivityCreate(ActivityBase):
    """Activity creation request"""
    pass


class ActivityUpdate(BaseModel):
    """Activity update request"""
    activity_type: Optional[ActivityType] = None
    subject: Optional[str] = None
    description: Optional[str] = None
    contact_id: Optional[str] = None
    company_id: Optional[str] = None
    deal_id: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: Optional[bool] = None
    priority: Optional[Priority] = None
    tags: Optional[List[str]] = None
    completed_at: Optional[datetime] = None


class Activity(ActivityBase):
    """Complete activity model"""
    activity_id: str
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"


# ============================================================================
# AI-POWERED FEATURES
# ============================================================================

class LeadScore(BaseModel):
    """AI-generated lead score"""
    contact_id: str
    score: float  # 0-100
    factors: Dict[str, float]  # Contributing factors
    confidence: float  # 0-1
    recommendations: List[str]
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class NextAction(BaseModel):
    """AI-suggested next action"""
    entity_type: str  # contact, company, deal
    entity_id: str
    action_type: ActivityType
    subject: str
    description: str
    priority: Priority
    reasoning: str
    confidence: float  # 0-1


# ============================================================================
# QUERY AND STATS MODELS
# ============================================================================

class CRMStats(BaseModel):
    """CRM statistics"""
    total_contacts: int
    total_companies: int
    total_deals: int
    total_activities: int
    contacts_by_status: Dict[str, int]
    deals_by_stage: Dict[str, int]
    total_pipeline_value: float
    total_won_value: float
    average_deal_size: float
    win_rate: float
    activities_by_type: Dict[str, int]

