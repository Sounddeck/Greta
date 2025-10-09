"""
Project Management Module Data Models for CPAS4
Complete project management with projects, tasks, milestones, and time tracking
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class ProjectStatus(str, Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class MilestoneStatus(str, Enum):
    UPCOMING = "upcoming"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    MISSED = "missed"


class TimeEntryType(str, Enum):
    WORK = "work"
    MEETING = "meeting"
    RESEARCH = "research"
    REVIEW = "review"
    BREAK = "break"
    OTHER = "other"


# Project Models
class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    client_id: Optional[str] = None
    client_name: Optional[str] = None
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    budget: Optional[float] = Field(None, ge=0)
    currency: str = Field(default="USD", max_length=3)
    team_members: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    end_date: Optional[datetime] = None
    budget: Optional[float] = Field(None, ge=0)
    team_members: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Project(ProjectBase):
    project_id: str
    status: ProjectStatus = ProjectStatus.PLANNING
    progress: float = Field(default=0.0, ge=0, le=100)
    total_tasks: int = 0
    completed_tasks: int = 0
    total_hours: float = 0.0
    budget_spent: float = 0.0
    created_at: datetime
    updated_at: datetime
    created_by: str = "system"


# Task Models
class TaskBase(BaseModel):
    project_id: str
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    assignee_id: Optional[str] = None
    assignee_name: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_hours: Optional[float] = Field(None, ge=0)
    due_date: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee_id: Optional[str] = None
    assignee_name: Optional[str] = None
    estimated_hours: Optional[float] = Field(None, ge=0)
    actual_hours: Optional[float] = Field(None, ge=0)
    due_date: Optional[datetime] = None
    dependencies: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Task(TaskBase):
    task_id: str
    status: TaskStatus = TaskStatus.TODO
    actual_hours: float = 0.0
    progress: float = Field(default=0.0, ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "system"


# Milestone Models
class MilestoneBase(BaseModel):
    project_id: str
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: datetime
    deliverables: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MilestoneCreate(MilestoneBase):
    pass


class MilestoneUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[MilestoneStatus] = None
    due_date: Optional[datetime] = None
    deliverables: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Milestone(MilestoneBase):
    milestone_id: str
    status: MilestoneStatus = MilestoneStatus.UPCOMING
    progress: float = Field(default=0.0, ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


# Time Entry Models
class TimeEntryBase(BaseModel):
    project_id: str
    task_id: Optional[str] = None
    user_id: str
    user_name: str
    entry_type: TimeEntryType = TimeEntryType.WORK
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_hours: Optional[float] = Field(None, ge=0)
    description: Optional[str] = None
    billable: bool = True
    hourly_rate: Optional[float] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeEntryCreate(TimeEntryBase):
    pass


class TimeEntryUpdate(BaseModel):
    end_time: Optional[datetime] = None
    duration_hours: Optional[float] = Field(None, ge=0)
    description: Optional[str] = None
    billable: Optional[bool] = None
    hourly_rate: Optional[float] = Field(None, ge=0)
    metadata: Optional[Dict[str, Any]] = None


class TimeEntry(TimeEntryBase):
    entry_id: str
    total_cost: float = 0.0
    created_at: datetime
    updated_at: datetime


# Resource Models
class ResourceBase(BaseModel):
    project_id: str
    name: str = Field(..., min_length=1, max_length=200)
    resource_type: str  # e.g., "human", "equipment", "material"
    description: Optional[str] = None
    cost_per_unit: Optional[float] = Field(None, ge=0)
    quantity: float = Field(default=1.0, gt=0)
    unit: str = Field(default="unit")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResourceCreate(ResourceBase):
    pass


class ResourceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    cost_per_unit: Optional[float] = Field(None, ge=0)
    quantity: Optional[float] = Field(None, gt=0)
    metadata: Optional[Dict[str, Any]] = None


class Resource(ResourceBase):
    resource_id: str
    total_cost: float = 0.0
    created_at: datetime
    updated_at: datetime


# Statistics Models
class ProjectStats(BaseModel):
    total_projects: int
    active_projects: int
    completed_projects: int
    total_tasks: int
    completed_tasks: int
    total_milestones: int
    completed_milestones: int
    total_time_entries: int
    total_hours_logged: float
    total_billable_hours: float
    total_revenue: float
    projects_by_status: Dict[str, int]
    tasks_by_status: Dict[str, int]
    tasks_by_priority: Dict[str, int]
    milestones_by_status: Dict[str, int]
    time_by_type: Dict[str, float]
    average_project_completion: float
    average_task_completion: float


# Report Models
class ProjectReport(BaseModel):
    report_type: str
    project_id: Optional[str] = None
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    data: Dict[str, Any]
    summary: Dict[str, Any]

