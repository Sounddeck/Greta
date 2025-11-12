"""
Reasoning Pattern Models - ClaudeFlow-inspired Pattern Storage

This module defines the data models for storing and retrieving reasoning patterns,
inspired by ClaudeFlow's approach to capturing and reusing successful problem-solving strategies.

A reasoning pattern captures:
- The problem context and type
- The solution approach and steps taken
- Success metrics and effectiveness
- Metadata for categorization and discovery
- Relationships to other patterns
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import uuid4


class SolutionStep(BaseModel):
    """A single step in a solution approach"""
    step_number: int = Field(..., description="Order of this step in the solution")
    description: str = Field(..., description="What was done in this step")
    reasoning: Optional[str] = Field(None, description="Why this step was taken")
    tools_used: List[str] = Field(default_factory=list, description="Tools/methods used")
    duration_ms: Optional[float] = Field(None, description="Time taken for this step")
    success: bool = Field(True, description="Whether this step succeeded")


class SuccessMetrics(BaseModel):
    """Metrics for evaluating pattern effectiveness"""
    success_rate: float = Field(1.0, ge=0.0, le=1.0, description="Success rate (0-1)")
    execution_time_ms: Optional[float] = Field(None, description="Average execution time")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality rating (0-1)")
    usage_count: int = Field(0, ge=0, description="Number of times pattern was used")
    last_used_at: Optional[datetime] = Field(None, description="When pattern was last used")
    feedback_score: Optional[float] = Field(None, ge=0.0, le=5.0, description="User feedback (0-5)")


class PatternContext(BaseModel):
    """Contextual information about when/where pattern applies"""
    domain: Optional[str] = Field(None, description="Domain/area (e.g., 'web development', 'data analysis')")
    complexity: Optional[str] = Field(None, description="Complexity level: 'simple', 'moderate', 'complex'")
    prerequisites: List[str] = Field(default_factory=list, description="Required conditions/tools")
    constraints: List[str] = Field(default_factory=list, description="Limitations or constraints")
    environment: Optional[Dict[str, Any]] = Field(None, description="Environment details")


class ReasoningPattern(BaseModel):
    """
    A reasoning pattern captures a successful problem-solving approach
    that can be reused for similar problems in the future.
    
    Inspired by ClaudeFlow's pattern storage system.
    """
    
    # Identity
    pattern_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique pattern identifier")
    version: int = Field(1, ge=1, description="Pattern version number")
    parent_pattern_id: Optional[str] = Field(None, description="ID of parent pattern if this is a refinement")
    
    # Classification
    pattern_type: str = Field(..., description="Category: development, analysis, optimization, orchestration, etc.")
    title: str = Field(..., description="Short, descriptive title")
    tags: List[str] = Field(default_factory=list, description="Keywords for discovery")
    
    # Problem Definition
    problem_description: str = Field(..., description="Detailed description of the problem this pattern solves")
    problem_signature: Optional[str] = Field(None, description="Normalized problem signature for matching")
    
    # Solution Approach
    solution_steps: List[SolutionStep] = Field(..., description="Steps taken to solve the problem")
    solution_summary: str = Field(..., description="High-level summary of the solution approach")
    key_insights: List[str] = Field(default_factory=list, description="Important learnings from this pattern")
    
    # Context and Applicability
    context: PatternContext = Field(default_factory=PatternContext, description="When/where this pattern applies")
    
    # Effectiveness
    success_metrics: SuccessMetrics = Field(default_factory=SuccessMetrics, description="Pattern effectiveness metrics")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When pattern was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When pattern was last updated")
    created_by: str = Field(..., description="Agent ID that created this pattern")
    
    # Relationships
    related_patterns: List[str] = Field(default_factory=list, description="IDs of similar/related patterns")
    supersedes: Optional[str] = Field(None, description="ID of pattern this one replaces")
    
    # Additional Data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PatternSearchQuery(BaseModel):
    """Query for searching reasoning patterns"""
    query: str = Field(..., description="Search query (problem description)")
    pattern_type: Optional[str] = Field(None, description="Filter by pattern type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    min_success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum success rate")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    include_related: bool = Field(False, description="Include related patterns in results")


class PatternSearchResult(BaseModel):
    """Result from pattern search"""
    pattern: ReasoningPattern = Field(..., description="The matching pattern")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="How relevant this pattern is (0-1)")
    match_type: str = Field(..., description="Type of match: 'semantic', 'keyword', 'hybrid'")
    match_explanation: Optional[str] = Field(None, description="Why this pattern matched")


class PatternCreateRequest(BaseModel):
    """Request to create a new reasoning pattern"""
    pattern_type: str
    title: str
    problem_description: str
    solution_steps: List[SolutionStep]
    solution_summary: str
    tags: Optional[List[str]] = None
    key_insights: Optional[List[str]] = None
    context: Optional[PatternContext] = None
    created_by: str


class PatternUpdateRequest(BaseModel):
    """Request to update an existing pattern"""
    title: Optional[str] = None
    problem_description: Optional[str] = None
    solution_steps: Optional[List[SolutionStep]] = None
    solution_summary: Optional[str] = None
    tags: Optional[List[str]] = None
    key_insights: Optional[List[str]] = None
    context: Optional[PatternContext] = None
    success_metrics: Optional[SuccessMetrics] = None


class PatternUsageRecord(BaseModel):
    """Record of pattern usage"""
    pattern_id: str
    used_at: datetime = Field(default_factory=datetime.utcnow)
    used_by: str
    problem_context: str
    success: bool
    execution_time_ms: Optional[float] = None
    feedback: Optional[str] = None
    feedback_score: Optional[float] = Field(None, ge=0.0, le=5.0)


class PatternStats(BaseModel):
    """Statistics about pattern usage"""
    total_patterns: int
    patterns_by_type: Dict[str, int]
    most_used_patterns: List[Dict[str, Any]]
    highest_rated_patterns: List[Dict[str, Any]]
    recent_patterns: List[Dict[str, Any]]
    average_success_rate: float
    total_usage_count: int


class PatternEvolutionRequest(BaseModel):
    """Request to create an evolved version of a pattern"""
    parent_pattern_id: str
    improvements: str
    new_solution_steps: Optional[List[SolutionStep]] = None
    new_insights: Optional[List[str]] = None
    created_by: str

