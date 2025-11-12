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
# MCP TOOL MODELS
# ============================================================================

class MCPToolAssignment(BaseModel):
    """MCP tool assignment to agent"""
    tool_id: str  # MCP tool ID from mcp_tools collection
    server_id: str  # MCP server ID
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True


# ============================================================================
# SKILLS MODELS
# ============================================================================

class SkillAssignment(BaseModel):
    """Skill assignment to agent"""
    skill_name: str  # Skill name from skills service
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True
    priority: int = 0  # Higher priority skills are tried first


# ============================================================================
# MEMORY CONFIGURATION MODELS
# ============================================================================

class MemoryConfiguration(BaseModel):
    """Memory system configuration for agent"""
    # Vector Search configuration
    enable_vector_search: bool = False
    vector_search_namespace: Optional[str] = None  # Namespace for this agent's memories
    vector_search_limit: int = 10  # Max memories to retrieve

    # ReasoningBank configuration
    enable_reasoning_bank: bool = False
    reasoning_bank_pattern_types: List[str] = Field(default_factory=list)  # Pattern types to use
    reasoning_bank_limit: int = 5  # Max patterns to retrieve

    # Functional Memory configuration
    enable_functional_memory: bool = False
    memory_types: List[str] = Field(default_factory=lambda: ["episodic", "semantic"])  # Memory types to use

    # Memory Orchestrator configuration
    enable_memory_orchestrator: bool = False

    # General settings
    auto_store_patterns: bool = True  # Automatically store successful executions as patterns
    memory_retention_days: int = 90  # How long to keep memories


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
    mcp_tools: List[MCPToolAssignment] = Field(default_factory=list)  # NEW: MCP tools
    skills: List[SkillAssignment] = Field(default_factory=list)  # NEW: Skills
    memory_config: Optional[MemoryConfiguration] = None  # NEW: Memory configuration
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
    mcp_tools: Optional[List[MCPToolAssignment]] = None  # NEW: MCP tools
    skills: Optional[List[SkillAssignment]] = None  # NEW: Skills
    memory_config: Optional[MemoryConfiguration] = None  # NEW: Memory configuration
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
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class AgentVersionCreate(AgentVersionBase):
    """Agent version creation request"""
    pass


class AgentVersion(AgentVersionBase):
    """Complete agent version model"""
    version_id: str
    is_active: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"


class VersionDiff(BaseModel):
    """Version comparison/diff result"""
    version_a_id: str
    version_a_number: int
    version_b_id: str
    version_b_number: int
    changes: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "version_a_id": "version_abc123",
                "version_a_number": 1,
                "version_b_id": "version_def456",
                "version_b_number": 2,
                "changes": {
                    "llm.model_id": {"old": "llama_cpp:llama3-8b", "new": "llama_cpp:llama3-70b"},
                    "tools": {"added": ["web_search"], "removed": []}
                },
                "summary": "Changed LLM model, added web_search tool"
            }
        }


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


# ============================================================================
# DRAFT MODELS (for autosave functionality)
# ============================================================================

class AgentDraftBase(BaseModel):
    """Base agent draft information"""
    name: str = ""
    description: str = ""
    agent_type: Optional[AgentType] = None
    capabilities: List[str] = Field(default_factory=list)
    system_prompt: str = ""
    tools: List[ToolAssignment] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    template_id: Optional[str] = None
    current_step: int = 0  # Track wizard progress


class AgentDraftCreate(AgentDraftBase):
    """Agent draft creation request"""
    pass


class AgentDraftUpdate(BaseModel):
    """Agent draft update request"""
    name: Optional[str] = None
    description: Optional[str] = None
    agent_type: Optional[AgentType] = None
    capabilities: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[ToolAssignment]] = None
    configuration: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    template_id: Optional[str] = None
    current_step: Optional[int] = None


class AgentDraft(AgentDraftBase):
    """Complete agent draft model"""
    draft_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"
    expires_at: Optional[datetime] = None  # Auto-delete old drafts


# ============================================================================
# LLM CONFIGURATION MODELS
# ============================================================================

class LLMProvider(str, Enum):
    """LLM provider"""
    LLAMA_CPP = "llama_cpp"  # Local llama.cpp (GGUF models) - DEFAULT
    OLLAMA = "ollama"        # Local Ollama
    OPENAI = "openai"        # OpenAI API
    ANTHROPIC = "anthropic"  # Anthropic API
    GOOGLE = "google"        # Google AI API
    CUSTOM = "custom"        # Custom endpoint


class LLMModelConfig(BaseModel):
    """LLM model configuration"""
    provider: LLMProvider = LLMProvider.LLAMA_CPP
    model_id: str = "llama_cpp:llama3-8b"
    model_name: str = "Llama 3 8B (llama.cpp)"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=100000)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    stop_sequences: List[str] = Field(default_factory=list)
    context_window: int = Field(8192, ge=1, le=200000)
    api_key: Optional[str] = None  # For cloud providers
    api_base: Optional[str] = None  # For custom endpoints


# ============================================================================
# ENHANCED VALIDATION MODELS
# ============================================================================

class ValidationRule(BaseModel):
    """Validation rule"""
    field: str
    rule_type: str  # required, min_length, max_length, pattern, custom
    value: Any
    error_message: str
    severity: str = "error"  # error, warning, info


class EnhancedValidationResult(BaseModel):
    """Enhanced agent validation result"""
    agent_id: Optional[str] = None
    draft_id: Optional[str] = None
    is_valid: bool
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    field_validations: Dict[str, bool] = Field(default_factory=dict)
    completion_percentage: float = 0.0
    validated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# PHASE 2, TASK 2.4: COMPREHENSIVE VALIDATION MODELS
# ============================================================================

class DependencyStatus(str, Enum):
    """Dependency status"""
    AVAILABLE = "available"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    DEPRECATED = "deprecated"


class DependencyType(str, Enum):
    """Dependency type"""
    TOOL = "tool"
    MODEL = "model"
    API_KEY = "api_key"
    INTEGRATION = "integration"
    LIBRARY = "library"
    SERVICE = "service"


class DependencyCheck(BaseModel):
    """
    Dependency check result.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Represents the status of a single dependency.
    """
    dependency_id: str
    dependency_name: str
    dependency_type: DependencyType
    status: DependencyStatus
    required: bool
    version_required: Optional[str] = None
    version_available: Optional[str] = None
    error_message: Optional[str] = None
    resolution_suggestion: Optional[str] = None


class DependencyValidationResult(BaseModel):
    """
    Comprehensive dependency validation result.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Contains all dependency checks for an agent configuration.
    """
    agent_id: Optional[str] = None
    draft_id: Optional[str] = None
    all_dependencies_met: bool
    missing_dependencies: List[DependencyCheck] = Field(default_factory=list)
    incompatible_dependencies: List[DependencyCheck] = Field(default_factory=list)
    available_dependencies: List[DependencyCheck] = Field(default_factory=list)
    circular_dependencies: List[str] = Field(default_factory=list)
    resolution_steps: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class CompatibilityIssue(BaseModel):
    """
    Compatibility issue between components.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Represents an incompatibility between agent components.
    """
    component_a: str
    component_b: str
    issue_type: str  # e.g., "tool_agent_type", "model_capability", "integration_version"
    severity: str  # "error", "warning", "info"
    description: str
    recommendation: str


class CompatibilityValidationResult(BaseModel):
    """
    Comprehensive compatibility validation result.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Contains all compatibility checks for an agent configuration.
    """
    agent_id: Optional[str] = None
    draft_id: Optional[str] = None
    is_compatible: bool
    compatibility_score: float  # 0-100
    critical_issues: List[CompatibilityIssue] = Field(default_factory=list)
    warnings: List[CompatibilityIssue] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceEstimate(BaseModel):
    """
    Performance estimation for agent configuration.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Provides performance predictions for an agent.
    """
    agent_id: Optional[str] = None
    draft_id: Optional[str] = None

    # Token usage estimates
    estimated_tokens_per_request: int
    estimated_context_tokens: int
    estimated_completion_tokens: int
    max_tokens_per_request: int

    # Response time estimates
    estimated_response_time_ms: int
    min_response_time_ms: int
    max_response_time_ms: int

    # Cost estimates (in USD)
    estimated_cost_per_request: float
    estimated_cost_per_1k_requests: float
    estimated_monthly_cost: float  # Based on usage patterns

    # Resource requirements
    memory_requirement_mb: int
    cpu_requirement: str  # "low", "medium", "high"
    gpu_required: bool

    # Performance factors
    complexity_score: float  # 0-100
    optimization_suggestions: List[str] = Field(default_factory=list)
    bottlenecks: List[str] = Field(default_factory=list)

    estimated_at: datetime = Field(default_factory=datetime.utcnow)


class ComprehensiveValidationResult(BaseModel):
    """
    Complete validation result combining all validation types.

    Phase 2, Task 2.4: Add Agent Builder Validation
    Aggregates configuration, dependency, compatibility, and performance validation.
    """
    agent_id: Optional[str] = None
    draft_id: Optional[str] = None

    # Overall status
    is_valid: bool
    is_deployable: bool
    overall_score: float  # 0-100

    # Individual validation results
    configuration_validation: EnhancedValidationResult
    dependency_validation: DependencyValidationResult
    compatibility_validation: CompatibilityValidationResult
    performance_estimate: PerformanceEstimate

    # Summary
    critical_issues_count: int
    warnings_count: int
    suggestions_count: int

    # Recommendations
    deployment_readiness: str  # "ready", "needs_fixes", "not_ready"
    next_steps: List[str] = Field(default_factory=list)

    validated_at: datetime = Field(default_factory=datetime.utcnow)

