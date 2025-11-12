"""
Tool Discovery Models for CPAS5 - Phase 2 Feature 1

This module defines the data models for the Progressive Tool Discovery feature,
which automatically discovers and suggests relevant tools based on task context.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TaskIntent(str, Enum):
    """Classification of task intent"""
    DATA_EXTRACTION = "data_extraction"
    WEB_SCRAPING = "web_scraping"
    RESEARCH = "research"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_CREATION = "content_creation"
    AUTOMATION = "automation"
    COMMUNICATION = "communication"
    FILE_OPERATION = "file_operation"
    API_INTERACTION = "api_interaction"
    DATABASE_OPERATION = "database_operation"
    GENERAL = "general"


class TaskComplexity(str, Enum):
    """Estimated task complexity"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskAnalysis(BaseModel):
    """
    Analysis of a task to extract requirements and characteristics
    
    This model represents the analyzed components of a user task,
    including keywords, entities, intent, and complexity.
    """
    description: str = Field(..., description="Original task description")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords from task")
    entities: List[str] = Field(default_factory=list, description="Extracted entities (URLs, file paths, etc.)")
    intent: TaskIntent = Field(default=TaskIntent.GENERAL, description="Classified task intent")
    complexity: TaskComplexity = Field(default=TaskComplexity.MEDIUM, description="Estimated complexity")
    requires_tools: bool = Field(default=True, description="Whether task likely requires tools")
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "Scrape product prices from competitor websites",
                "keywords": ["scrape", "product", "prices", "websites"],
                "entities": ["competitor websites"],
                "intent": "web_scraping",
                "complexity": "medium",
                "requires_tools": True
            }
        }


class ToolSuggestion(BaseModel):
    """
    A suggested tool with relevance score and reasoning
    
    This model represents a tool suggestion for a given task,
    including the relevance score and explanation.
    """
    tool_id: str = Field(..., description="Unique tool identifier")
    tool_name: str = Field(..., description="Human-readable tool name")
    tool_category: str = Field(..., description="Tool category (e.g., OSINT, CODE, DATA)")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)")
    final_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Final score after ranking adjustments")
    reason: str = Field(..., description="Explanation of why this tool is relevant")
    
    # Scoring breakdown
    keyword_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Keyword matching score")
    semantic_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Semantic similarity score")
    historical_success_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Historical success rate")
    
    # Additional metadata
    usage_count: int = Field(default=0, description="Number of times tool has been used")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall success rate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_id": "osint_playwright_scraper",
                "tool_name": "Playwright Web Scraper",
                "tool_category": "OSINT",
                "relevance_score": 0.85,
                "final_score": 0.92,
                "reason": "High relevance for web scraping tasks with automatic data extraction",
                "keyword_match_score": 0.75,
                "semantic_similarity_score": 0.90,
                "historical_success_score": 0.88,
                "usage_count": 150,
                "success_rate": 0.88
            }
        }


class ToolUsagePattern(BaseModel):
    """
    Pattern of tool usage for learning and improvement
    
    This model stores historical tool usage patterns to improve
    future tool suggestions through machine learning.
    """
    pattern_id: str = Field(default="", description="Unique pattern identifier")
    task_description: str = Field(..., description="Original task description")
    task_intent: TaskIntent = Field(default=TaskIntent.GENERAL, description="Classified task intent")
    tools_used: List[str] = Field(..., description="List of tool IDs used")
    tools_suggested: List[str] = Field(default_factory=list, description="List of tool IDs that were suggested")
    success: bool = Field(..., description="Whether the execution was successful")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    
    # Metadata
    agent_id: Optional[str] = Field(default=None, description="Agent that executed the task")
    agent_type: Optional[str] = Field(default=None, description="Type of agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When pattern was created")
    
    # Learning metrics
    suggestion_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="How accurate were the suggestions")
    tools_used_count: int = Field(default=0, description="Number of tools actually used")
    tools_suggested_count: int = Field(default=0, description="Number of tools suggested")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pattern_id": "pattern_abc123",
                "task_description": "Scrape product data from e-commerce site",
                "task_intent": "web_scraping",
                "tools_used": ["osint_playwright_scraper"],
                "tools_suggested": ["osint_playwright_scraper", "web_search"],
                "success": True,
                "execution_time": 5.2,
                "agent_id": "agent_xyz789",
                "agent_type": "research",
                "suggestion_accuracy": 0.5,
                "tools_used_count": 1,
                "tools_suggested_count": 2
            }
        }


class ToolDiscoveryRequest(BaseModel):
    """
    Request for tool discovery
    
    This model represents a request to discover relevant tools
    for a given task.
    """
    task_description: str = Field(..., description="Task description to analyze")
    agent_id: Optional[str] = Field(default=None, description="Agent requesting tools")
    agent_type: Optional[str] = Field(default=None, description="Type of agent")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of suggestions")
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum relevance threshold")
    
    # Context for better suggestions
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_description": "Extract contact information from company websites",
                "agent_id": "agent_xyz789",
                "agent_type": "research",
                "limit": 5,
                "min_relevance": 0.3,
                "context": {
                    "previous_tools": ["web_search"],
                    "domain": "business_intelligence"
                }
            }
        }


class ToolDiscoveryResponse(BaseModel):
    """
    Response from tool discovery
    
    This model represents the response from a tool discovery request,
    including suggested tools and analysis.
    """
    task_analysis: TaskAnalysis = Field(..., description="Analysis of the task")
    suggestions: List[ToolSuggestion] = Field(default_factory=list, description="Suggested tools")
    discovery_time: float = Field(default=0.0, description="Time taken for discovery in seconds")
    total_tools_analyzed: int = Field(default=0, description="Total number of tools analyzed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_analysis": {
                    "description": "Scrape product prices",
                    "keywords": ["scrape", "product", "prices"],
                    "entities": [],
                    "intent": "web_scraping",
                    "complexity": "medium",
                    "requires_tools": True
                },
                "suggestions": [
                    {
                        "tool_id": "osint_playwright_scraper",
                        "tool_name": "Playwright Web Scraper",
                        "tool_category": "OSINT",
                        "relevance_score": 0.85,
                        "final_score": 0.92,
                        "reason": "High relevance for web scraping",
                        "keyword_match_score": 0.75,
                        "semantic_similarity_score": 0.90,
                        "historical_success_score": 0.88,
                        "usage_count": 150,
                        "success_rate": 0.88
                    }
                ],
                "discovery_time": 0.45,
                "total_tools_analyzed": 25
            }
        }


class ToolLearningData(BaseModel):
    """
    Learning data for tool discovery improvement
    
    This model stores aggregated learning data to improve
    tool discovery over time.
    """
    tool_id: str = Field(..., description="Tool identifier")
    task_intent: TaskIntent = Field(..., description="Task intent category")
    
    # Success metrics
    total_uses: int = Field(default=0, description="Total number of uses")
    successful_uses: int = Field(default=0, description="Number of successful uses")
    failed_uses: int = Field(default=0, description="Number of failed uses")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    
    # Performance metrics
    avg_execution_time: float = Field(default=0.0, description="Average execution time")
    min_execution_time: float = Field(default=0.0, description="Minimum execution time")
    max_execution_time: float = Field(default=0.0, description="Maximum execution time")
    
    # Temporal data
    last_used: Optional[datetime] = Field(default=None, description="Last time tool was used")
    first_used: Optional[datetime] = Field(default=None, description="First time tool was used")
    
    # Trend data
    recent_success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate in last 7 days")
    trend: str = Field(default="stable", description="Usage trend (increasing, decreasing, stable)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_id": "osint_playwright_scraper",
                "task_intent": "web_scraping",
                "total_uses": 150,
                "successful_uses": 132,
                "failed_uses": 18,
                "success_rate": 0.88,
                "avg_execution_time": 5.2,
                "min_execution_time": 1.5,
                "max_execution_time": 30.0,
                "recent_success_rate": 0.92,
                "trend": "increasing"
            }
        }


class ToolDiscoveryStats(BaseModel):
    """
    Statistics for tool discovery system
    
    This model provides overall statistics about the tool
    discovery system's performance.
    """
    total_discoveries: int = Field(default=0, description="Total number of discovery requests")
    total_suggestions: int = Field(default=0, description="Total number of suggestions made")
    avg_suggestions_per_request: float = Field(default=0.0, description="Average suggestions per request")
    avg_discovery_time: float = Field(default=0.0, description="Average discovery time in seconds")
    
    # Accuracy metrics
    suggestion_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall suggestion accuracy")
    top_1_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Top-1 suggestion accuracy")
    top_3_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Top-3 suggestion accuracy")
    
    # Learning metrics
    total_patterns_learned: int = Field(default=0, description="Total patterns learned")
    improvement_rate: float = Field(default=0.0, description="Rate of improvement over time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_discoveries": 500,
                "total_suggestions": 2500,
                "avg_suggestions_per_request": 5.0,
                "avg_discovery_time": 0.45,
                "suggestion_accuracy": 0.82,
                "top_1_accuracy": 0.65,
                "top_3_accuracy": 0.85,
                "total_patterns_learned": 450,
                "improvement_rate": 0.15
            }
        }

