"""
Lead Generation Models

Data models for automated lead generation and research functionality.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class LeadStatus(str, Enum):
    """Lead status"""
    NEW = "new"
    RESEARCHING = "researching"
    QUALIFIED = "qualified"
    CONTACTED = "contacted"
    CONVERTED = "converted"
    DISQUALIFIED = "disqualified"


class ResearchDepth(str, Enum):
    """Research depth level"""
    SHALLOW = "shallow"  # Basic company info
    MEDIUM = "medium"  # Company + key people + news
    DEEP = "deep"  # Comprehensive research


class MonitoringFrequency(str, Enum):
    """Monitoring frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class CompanySize(str, Enum):
    """Company size categories"""
    STARTUP = "1-10"
    SMALL = "11-50"
    MEDIUM = "51-200"
    LARGE = "201-1000"
    ENTERPRISE = "1000+"


# ============================================================================
# Campaign Models
# ============================================================================

class LeadCampaignConfig(BaseModel):
    """Lead generation campaign configuration"""
    campaign_name: str
    target_industry: str
    target_location: str
    company_size: CompanySize
    keywords: List[str]
    research_depth: ResearchDepth = ResearchDepth.MEDIUM
    monitoring_frequency: MonitoringFrequency = MonitoringFrequency.WEEKLY
    max_leads: int = 100
    auto_research: bool = True


class LeadCampaign(BaseModel):
    """Lead generation campaign"""
    campaign_id: str
    user_id: str
    config: LeadCampaignConfig
    status: str = "active"  # active, paused, completed
    leads_generated: int = 0
    leads_qualified: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Lead Models
# ============================================================================

class KeyPerson(BaseModel):
    """Key person at a company"""
    name: str
    title: str
    linkedin: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None


class CompanyData(BaseModel):
    """Company information"""
    company_name: str
    website: Optional[str] = None
    industry: str
    location: str
    company_size: Optional[str] = None
    revenue: Optional[str] = None
    funding: Optional[str] = None
    description: Optional[str] = None
    founded_year: Optional[int] = None


class NewsItem(BaseModel):
    """News item about a company"""
    title: str
    url: str
    source: str
    published_date: str
    summary: Optional[str] = None


class SocialActivity(BaseModel):
    """Social media activity"""
    platform: str  # linkedin, twitter, facebook
    followers: Optional[int] = None
    engagement_rate: Optional[float] = None
    recent_posts: List[str] = []
    activity_level: str = "low"  # low, medium, high


class TechnologyStack(BaseModel):
    """Technology stack information"""
    technologies: List[str] = []
    frameworks: List[str] = []
    platforms: List[str] = []
    tools: List[str] = []


class Opportunity(BaseModel):
    """Engagement opportunity"""
    opportunity_type: str  # hiring, funding, expansion, product_launch
    description: str
    confidence: float  # 0-1
    source: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class LeadScore(BaseModel):
    """Lead scoring"""
    score: float  # 0-100
    factors: Dict[str, float]  # Contributing factors
    confidence: float  # 0-1
    recommendations: List[str]
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchReport(BaseModel):
    """Comprehensive research report"""
    research_id: str
    lead_id: str
    company_data: CompanyData
    key_people: List[KeyPerson]
    recent_news: List[NewsItem]
    social_activity: List[SocialActivity]
    tech_stack: Optional[TechnologyStack] = None
    competitors: List[str] = []
    opportunities: List[Opportunity]
    lead_score: LeadScore
    research_depth: ResearchDepth
    researched_at: datetime = Field(default_factory=datetime.utcnow)


class Lead(BaseModel):
    """Lead entity"""
    lead_id: str
    campaign_id: str
    user_id: str
    company_name: str
    industry: str
    location: str
    company_size: Optional[str] = None
    website: Optional[str] = None
    lead_score: float = 0.0
    status: LeadStatus = LeadStatus.NEW
    research_history: List[str] = []  # List of research_ids
    key_people: List[KeyPerson] = []
    recent_news: List[NewsItem] = []
    opportunities: List[Opportunity] = []
    monitoring_active: bool = False
    next_research_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateCampaignRequest(BaseModel):
    """Create campaign request"""
    user_id: str
    config: LeadCampaignConfig


class CreateCampaignResponse(BaseModel):
    """Create campaign response"""
    campaign_id: str
    message: str


class ResearchLeadRequest(BaseModel):
    """Research lead request"""
    lead_id: str
    research_depth: ResearchDepth = ResearchDepth.MEDIUM


class ResearchLeadResponse(BaseModel):
    """Research lead response"""
    research_id: str
    lead_id: str
    status: str
    message: str


class LeadListResponse(BaseModel):
    """Lead list response"""
    leads: List[Lead]
    total: int
    page: int
    page_size: int


class CampaignStatsResponse(BaseModel):
    """Campaign statistics"""
    campaign_id: str
    total_leads: int
    qualified_leads: int
    contacted_leads: int
    converted_leads: int
    avg_lead_score: float
    top_opportunities: List[Opportunity]


class LeadUpdate(BaseModel):
    """Lead update/change"""
    update_type: str  # news, personnel, funding, product, social
    description: str
    source: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class MonitoringStatusResponse(BaseModel):
    """Monitoring status"""
    lead_id: str
    monitoring_active: bool
    frequency: MonitoringFrequency
    last_check: Optional[datetime] = None
    next_check: Optional[datetime] = None
    updates_found: int = 0


# ============================================================================
# Health Check
# ============================================================================

class LeadGenerationHealthResponse(BaseModel):
    """Lead generation service health"""
    status: str
    message: str
    timestamp: datetime
    active_campaigns: int
    total_leads: int
    researches_today: int

