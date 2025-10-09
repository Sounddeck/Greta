"""
Task Models for CPAS4 Task Persistence System
Comprehensive task management with lifecycle tracking
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(str, Enum):
    """Task priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskType(str, Enum):
    """Task type enumeration"""
    AGENT_EXECUTION = "agent_execution"
    WORKFLOW = "workflow"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    AUTOMATED = "automated"


class TaskMetadata(BaseModel):
    """Task metadata"""
    agent_type: Optional[str] = None
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Task execution result"""
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None


class Task(BaseModel):
    """Complete task model"""
    task_id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    task_type: TaskType = TaskType.MANUAL
    
    # Input/Output
    input_data: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[TaskResult] = None
    
    # Metadata
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)
    
    # Lifecycle timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dependencies and relationships
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    
    # Retry and error handling
    retry_count: int = 0
    max_retries: int = 3
    error_history: List[str] = Field(default_factory=list)
    
    # Progress tracking
    progress_percentage: float = 0.0
    progress_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_abc123",
                "title": "Generate Marketing Campaign",
                "description": "Create a comprehensive marketing campaign for Q4",
                "status": "pending",
                "priority": "high",
                "task_type": "agent_execution",
                "input_data": {
                    "agent_type": "marketing",
                    "campaign_type": "social_media"
                },
                "metadata": {
                    "agent_type": "marketing",
                    "user_id": "user_123",
                    "tags": ["marketing", "q4", "campaign"]
                }
            }
        }


class TaskCreate(BaseModel):
    """Task creation request"""
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    task_type: TaskType = TaskType.MANUAL
    input_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[TaskMetadata] = None
    parent_task_id: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)


class TaskUpdate(BaseModel):
    """Task update request"""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    input_data: Optional[Dict[str, Any]] = None
    result: Optional[TaskResult] = None
    progress_percentage: Optional[float] = None
    progress_message: Optional[str] = None


class TaskQuery(BaseModel):
    """Task query parameters"""
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    task_type: Optional[TaskType] = None
    agent_type: Optional[str] = None
    user_id: Optional[str] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = 100
    skip: int = 0


class TaskStats(BaseModel):
    """Task statistics"""
    total_tasks: int
    pending: int
    in_progress: int
    completed: int
    failed: int
    cancelled: int
    paused: int
    by_priority: Dict[str, int]
    by_type: Dict[str, int]
    by_agent: Dict[str, int]
    average_execution_time: Optional[float] = None
    success_rate: Optional[float] = None


class TaskBatch(BaseModel):
    """Batch task operations"""
    task_ids: List[str]
    operation: str  # "cancel", "pause", "resume", "delete"
    reason: Optional[str] = None


class TaskDependencyGraph(BaseModel):
    """Task dependency graph"""
    task_id: str
    dependencies: List[str]
    dependents: List[str]
    depth: int
    can_execute: bool
    blocking_tasks: List[str]

