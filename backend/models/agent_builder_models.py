"""
Agent Builder Models for CPAS4
Comprehensive data models for custom agent creation and management
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class AgentStatus(str, Enum):
    """Agent status"""
    DRAFT = "draft"
    TESTING = "testing"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStatus(str, Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ToolCategory(str, Enum):
    """Tool category"""
    DATA_PROCESSING = "data_processing"
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    FILE_OPERATIONS = "file_operations"
    DATABASE = "database"
    AI_ML = "ai_ml"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    ANALYTICS = "analytics"
    CUSTOM = "custom"


class AgentType(str, Enum):
    """Agent type"""
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_CREATION = "content_creation"
    TASK_AUTOMATION = "task_automation"
    CUSTOMER_SUPPORT = "customer_support"
    SALES = "sales"
    MARKETING = "marketing"
    FINANCE = "finance"
    GENERAL = "general"
    CUSTOM = "custom"


class ExecutionStatus(str, Enum):
    """Execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# TOOL MODELS
# ============================================================================

class ToolDefinition(BaseModel):
    """Tool definition"""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_permissions: List[str] = Field(default_factory=list)
    is_enabled: bool = True


class ToolAssignment(BaseModel):
    """Tool assignment to agent"""
    tool_id: str
    permissions: List[str] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True


# ============================================================================
# AGENT TEMPLATE MODELS
# ============================================================================

class AgentTemplateBase(BaseModel):
    """Base agent template information"""
    name: str
    description: str
    agent_type: AgentType
    capabilities: List[str] = Field(default_factory=list)
    default_tools: List[str] = Field(default_factory=list)
    default_prompt: str
    configuration: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class AgentTemplateCreate(AgentTemplateBase):
    """Agent template creation request"""
    pass


class AgentTemplateUpdate(BaseModel):
    """Agent template update request"""
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    default_tools: Optional[List[str]] = None
    default_prompt: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class AgentTemplate(AgentTemplateBase):
    """Complete agent template model"""
    template_id: str
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"


# ============================================================================
# CUSTOM AGENT MODELS
# ============================================================================

class CustomAgentBase(BaseModel):
    """Base custom agent information"""
    name: str
    description: str
    agent_type: AgentType
    capabilities: List[str] = Field(default_factory=list)
    system_prompt: str
    tools: List[ToolAssignment] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CustomAgentCreate(CustomAgentBase):
    """Custom agent creation request"""
    template_id: Optional[str] = None  # Create from template
    status: AgentStatus = AgentStatus.DRAFT


class CustomAgentUpdate(BaseModel):
    """Custom agent update request"""
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[ToolAssignment]] = None
    configuration: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatus] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class CustomAgent(CustomAgentBase):
    """Complete custom agent model"""
    agent_id: str
    status: AgentStatus = AgentStatus.DRAFT
    version: int = 1
    current_version_id: Optional[str] = None
    template_id: Optional[str] = None
    generated_code: Optional[str] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_execution_time: float = 0.0
    last_executed: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    created_by: str = "system"


# ============================================================================
# AGENT VERSION MODELS
# ============================================================================

class AgentVersionBase(BaseModel):
    """Base agent version information"""
    agent_id: str
    version_number: int
    changes: str
    configuration_snapshot: Dict[str, Any]
    code_snapshot: Optional[str] = None


class AgentVersionCreate(AgentVersionBase):
    """Agent version creation request"""
    pass


class AgentVersion(AgentVersionBase):
    """Complete agent version model"""
    version_id: str
    is_active: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"


# ============================================================================
# DEPLOYMENT MODELS
# ============================================================================

class DeploymentBase(BaseModel):
    """Base deployment information"""
    agent_id: str
    version_id: str
    environment: str = "production"
    configuration: Dict[str, Any] = Field(default_factory=dict)


class DeploymentCreate(DeploymentBase):
    """Deployment creation request"""
    pass


class DeploymentUpdate(BaseModel):
    """Deployment update request"""
    status: Optional[DeploymentStatus] = None
    error_message: Optional[str] = None


class Deployment(DeploymentBase):
    """Complete deployment model"""
    deployment_id: str
    status: DeploymentStatus = DeploymentStatus.PENDING
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    deployed_by: str = "system"


# ============================================================================
# EXECUTION MODELS
# ============================================================================

class AgentExecutionBase(BaseModel):
    """Base agent execution information"""
    agent_id: str
    input_data: Dict[str, Any]
    configuration: Dict[str, Any] = Field(default_factory=dict)


class AgentExecutionCreate(AgentExecutionBase):
    """Agent execution creation request"""
    pass


class AgentExecutionUpdate(BaseModel):
    """Agent execution update request"""
    status: Optional[ExecutionStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    logs: Optional[List[str]] = None


class AgentExecution(AgentExecutionBase):
    """Complete agent execution model"""
    execution_id: str
    status: ExecutionStatus = ExecutionStatus.QUEUED
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    execution_time: float = 0.0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    executed_by: str = "system"


# ============================================================================
# CODE GENERATION MODELS
# ============================================================================

class CodeGenerationRequest(BaseModel):
    """Code generation request"""
    agent_id: str
    include_tools: bool = True
    include_tests: bool = False
    output_format: str = "python"  # python, json, yaml


class CodeGenerationResult(BaseModel):
    """Code generation result"""
    agent_id: str
    generated_code: str
    file_path: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    is_valid: bool = True
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationResult(BaseModel):
    """Agent validation result"""
    agent_id: str
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# STATISTICS MODELS
# ============================================================================

class AgentBuilderStats(BaseModel):
    """Agent Builder statistics"""
    total_agents: int
    total_templates: int
    total_executions: int
    agents_by_status: Dict[str, int]
    agents_by_type: Dict[str, int]
    executions_by_status: Dict[str, int]
    most_used_tools: List[Dict[str, Any]]
    most_used_templates: List[Dict[str, Any]]
    average_execution_time: float
    success_rate: float
    total_deployments: int
    active_deployments: int

