"""
🎯 Skill Models - Data models for Skills System
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class SkillBase(BaseModel):
    """Base skill information"""
    name: str
    display_name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    category: str
    activation_keywords: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class SkillResponse(SkillBase):
    """Skill response with full details"""
    activation_patterns: List[str] = Field(default_factory=list)
    handler_function: str
    context_requirements: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class SkillListResponse(BaseModel):
    """Response for listing skills"""
    skills: List[SkillResponse]
    total: int
    categories: List[str]


class SkillMatchRequest(BaseModel):
    """Request to match user input to a skill"""
    user_input: str = Field(..., description="Natural language input from user")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context for matching")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")


class SkillMatchResponse(BaseModel):
    """Response from skill matching"""
    matched: bool
    skill: Optional[SkillResponse] = None
    confidence: Optional[float] = None
    matched_keywords: List[str] = Field(default_factory=list)
    matched_patterns: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class SkillExecutionRequest(BaseModel):
    """Request to execute a skill"""
    skill_name: str = Field(..., description="Name of the skill to execute")
    user_input: str = Field(..., description="User's input")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Execution context")


class SkillExecutionResponse(BaseModel):
    """Response from skill execution"""
    success: bool
    skill: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

