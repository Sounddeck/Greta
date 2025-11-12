"""
Team Models for CPAS5 Agent Builder
Multi-agent collaboration patterns similar to AutoGen Studio
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class WorkflowPattern(str, Enum):
    """Workflow pattern types for team collaboration"""
    SEQUENTIAL = "sequential"        # Agents execute in predefined order
    AUTONOMOUS = "autonomous"        # LLM decides next speaker (GroupChat)
    ROUND_ROBIN = "round_robin"      # Agents take turns in rotation
    SELECTOR = "selector"            # Model-based speaker selection


class TerminationType(str, Enum):
    """Termination condition types"""
    MAX_TURNS = "max_turns"              # Maximum number of conversation turns
    KEYWORD = "keyword"                  # Specific keyword detected in message
    MAX_CONSECUTIVE = "max_consecutive"  # Max consecutive auto-replies from same agent
    FUNCTION = "function"                # Custom termination function


class TeamStatus(str, Enum):
    """Team status"""
    ACTIVE = "active"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """Team execution status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


# ============================================================================
# TEAM AGENT MODELS
# ============================================================================

class TeamAgent(BaseModel):
    """Agent role assignment in a team"""
    role: str = Field(..., description="Agent role (e.g., researcher, writer, reviewer)")
    agent_id: str = Field(..., description="Reference to CustomAgent ID")
    order: Optional[int] = Field(None, description="Execution order for sequential pattern")
    description: Optional[str] = Field(None, description="Role description")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "researcher",
                "agent_id": "agent-abc123",
                "order": 1,
                "description": "Conducts research and gathers information"
            }
        }


# ============================================================================
# TERMINATION CONDITION MODELS
# ============================================================================

class TerminationCondition(BaseModel):
    """Termination condition configuration"""
    type: TerminationType
    value: str | int = Field(..., description="Condition value (number or string)")
    description: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "type": "max_turns",
                "value": 10,
                "description": "Stop after 10 conversation turns"
            }
        }


# ============================================================================
# TEAM BASE MODELS
# ============================================================================

class TeamBase(BaseModel):
    """Base team fields"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=1000)
    agents: List[TeamAgent] = Field(..., min_items=1)
    workflow_pattern: WorkflowPattern
    termination: TerminationCondition
    metadata: Optional[Dict[str, Any]] = None


class TeamCreate(TeamBase):
    """Team creation payload"""
    pass


class TeamUpdate(BaseModel):
    """Team update payload"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    agents: Optional[List[TeamAgent]] = None
    workflow_pattern: Optional[WorkflowPattern] = None
    termination: Optional[TerminationCondition] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[TeamStatus] = None


class Team(TeamBase):
    """Complete team model"""
    team_id: str
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    status: TeamStatus = TeamStatus.ACTIVE

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "team_id": "team-abc123",
                "name": "Research Team",
                "description": "Multi-agent research and analysis team",
                "agents": [
                    {
                        "role": "researcher",
                        "agent_id": "agent-001",
                        "order": 1,
                        "description": "Conducts research"
                    },
                    {
                        "role": "analyst",
                        "agent_id": "agent-002",
                        "order": 2,
                        "description": "Analyzes findings"
                    }
                ],
                "workflow_pattern": "sequential",
                "termination": {
                    "type": "max_turns",
                    "value": 10
                },
                "status": "active",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:00:00Z"
            }
        }


# ============================================================================
# TEAM EXECUTION MODELS
# ============================================================================

class TeamExecutionRequest(BaseModel):
    """Team execution request"""
    team_id: str
    task: str = Field(..., min_length=1, description="The task/prompt for the team")
    context: Optional[Dict[str, Any]] = None
    max_iterations: Optional[int] = Field(None, gt=0, le=100, description="Override termination max_turns")

    class Config:
        json_schema_extra = {
            "example": {
                "team_id": "team-abc123",
                "task": "Research the latest trends in AI and create a comprehensive report",
                "context": {
                    "focus_areas": ["machine learning", "natural language processing"],
                    "output_format": "markdown"
                },
                "max_iterations": 15
            }
        }


class TeamMessage(BaseModel):
    """Message in team conversation"""
    message_id: str
    agent_id: str
    agent_role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg-abc123",
                "agent_id": "agent-001",
                "agent_role": "researcher",
                "content": "I have completed the research on AI trends...",
                "timestamp": "2025-01-15T10:05:00Z",
                "metadata": {
                    "tokens": 150,
                    "model": "ollama:llama3.2"
                }
            }
        }


class TeamExecutionResult(BaseModel):
    """Team execution result"""
    execution_id: str
    team_id: str
    status: ExecutionStatus
    messages: List[TeamMessage]
    final_output: Optional[str] = None
    metadata: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "execution_id": "exec-abc123",
                "team_id": "team-abc123",
                "status": "completed",
                "messages": [],
                "final_output": "Comprehensive AI trends report...",
                "metadata": {
                    "total_turns": 8,
                    "total_tokens": 1500,
                    "duration_ms": 45000,
                    "termination_reason": "max_turns_reached"
                },
                "created_at": "2025-01-15T10:00:00Z",
                "completed_at": "2025-01-15T10:05:00Z"
            }
        }


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class TeamValidationResult(BaseModel):
    """Team validation result"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "valid": True,
                "errors": [],
                "warnings": ["Agent agent-002 is not currently active"]
            }
        }


# ============================================================================
# LIST RESPONSE MODELS
# ============================================================================

class TeamListResponse(BaseModel):
    """Team list response"""
    teams: List[Team]
    total: int
    limit: int
    offset: int


class ExecutionListResponse(BaseModel):
    """Execution list response"""
    executions: List[TeamExecutionResult]
    total: int
    limit: int

