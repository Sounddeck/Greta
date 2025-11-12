"""
Business Models for Multi-Business Management
Pydantic models for business entities, executive teams, and decision-making
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class DecisionAuthority(str, Enum):
    """Decision authority levels for executives"""
    ADVISORY_ONLY = "advisory_only"  # Can only provide advice, no decision-making power
    LOW = "low"  # Can make minor decisions independently
    MEDIUM = "medium"  # Can make moderate decisions independently
    HIGH = "high"  # Can make major decisions independently
    REQUIRES_APPROVAL = "requires_approval"  # All decisions require approval


class BusinessStatus(str, Enum):
    """Business operational status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    SUSPENDED = "suspended"


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

class AgentConfig(BaseModel):
    """Configuration for an executive agent"""
    name: str
    role: str
    capabilities: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    personality: str = ""
    decision_authority: DecisionAuthority = DecisionAuthority.MEDIUM
    requires_approval_for: List[str] = Field(default_factory=list)
    agent_id: str


# ============================================================================
# EXECUTIVE TEAM
# ============================================================================

class ExecutiveTeam(BaseModel):
    """Executive team configuration for a business"""
    business_advisor: Optional[AgentConfig] = None
    ceo: Optional[AgentConfig] = None
    cfo: Optional[AgentConfig] = None
    coo: Optional[AgentConfig] = None
    cto: Optional[AgentConfig] = None
    sales: Optional[AgentConfig] = None
    marketing: Optional[AgentConfig] = None
    accounting: Optional[AgentConfig] = None
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True


# ============================================================================
# BUSINESS INFRASTRUCTURE
# ============================================================================

class GoogleWorkspaceConfig(BaseModel):
    """Google Workspace configuration"""
    domain: str = ""
    admin_email: str = ""
    org_unit: str = ""


class NotebookLMConfig(BaseModel):
    """NotebookLM configuration"""
    project_id: str = ""
    notebook_id: str = ""


class CRMIntegration(BaseModel):
    """CRM integration configuration"""
    enabled: bool = False
    sync_interval: int = 3600


class ATSIntegration(BaseModel):
    """ATS integration configuration"""
    enabled: bool = False
    sync_interval: int = 3600


class BusinessInfrastructure(BaseModel):
    """Business infrastructure configuration"""
    google_workspace: Optional[GoogleWorkspaceConfig] = Field(default_factory=GoogleWorkspaceConfig)
    notebooklm: Optional[NotebookLMConfig] = Field(default_factory=NotebookLMConfig)
    crm_integration: Optional[CRMIntegration] = Field(default_factory=CRMIntegration)
    ats_integration: Optional[ATSIntegration] = Field(default_factory=ATSIntegration)


# ============================================================================
# APPROVAL WORKFLOW
# ============================================================================

class ApprovalWorkflow(BaseModel):
    """Approval workflow configuration"""
    enabled: bool = True
    approval_threshold: float = 10000.0
    auto_approve_below: float = 1000.0
    require_approval_for: List[str] = Field(default_factory=list)
    autonomous_mode: bool = False


# ============================================================================
# BUSINESS MODELS
# ============================================================================

class BusinessCreate(BaseModel):
    """Model for creating a new business"""
    name: str
    description: str
    industry: str
    tags: List[str] = Field(default_factory=list)
    workspace_path: Optional[str] = None
    isolated_data: bool = True
    enable_executive_team: bool = True
    autonomous_mode: bool = False


class BusinessUpdate(BaseModel):
    """Model for updating a business"""
    name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    status: Optional[BusinessStatus] = None
    tags: Optional[List[str]] = None
    infrastructure: Optional[BusinessInfrastructure] = None
    approval_workflow: Optional[ApprovalWorkflow] = None
    executive_team: Optional[ExecutiveTeam] = None


class Business(BaseModel):
    """Business entity model"""
    id: Optional[str] = Field(None, alias="_id")
    name: str
    description: str
    industry: str
    owner_id: str
    workspace_path: str
    isolated_data: bool = True
    executive_team: ExecutiveTeam = Field(default_factory=ExecutiveTeam)
    infrastructure: BusinessInfrastructure = Field(default_factory=BusinessInfrastructure)
    approval_workflow: ApprovalWorkflow = Field(default_factory=ApprovalWorkflow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: BusinessStatus = BusinessStatus.ACTIVE
    tags: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic config"""
        populate_by_name = True
        arbitrary_types_allowed = True


class BusinessStats(BaseModel):
    """Business statistics"""
    total_decisions: int = 0
    pending_approvals: int = 0
    approved_decisions: int = 0
    rejected_decisions: int = 0
    total_revenue: float = 0.0
    total_expenses: float = 0.0
    active_projects: int = 0
    team_size: int = 0


# ============================================================================
# EXECUTIVE DECISION
# ============================================================================

class ExecutiveDecision(BaseModel):
    """Record of an executive decision"""
    decision_id: str
    business_id: str
    executive_role: str
    decision_topic: str
    decision_type: str
    recommendation: Dict[str, Any]
    requires_approval: bool
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

