"""
Prompt Template Models
Data models for the prompt template library system
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class PromptCategory(str, Enum):
    """Prompt template categories"""
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    WRITING = "writing"
    BUSINESS = "business"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    GENERAL = "general"


class PromptTemplate(BaseModel):
    """Prompt template model"""
    id: Optional[str] = Field(None, description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: PromptCategory = Field(..., description="Template category")
    template: str = Field(..., description="Template text with placeholders")
    variables: List[str] = Field(default_factory=list, description="Template variables")
    tags: List[str] = Field(default_factory=list, description="Search tags")
    version: str = Field(default="1.0.0", description="Template version")
    is_active: bool = Field(default=True, description="Whether template is active")
    is_system: bool = Field(default=False, description="Whether template is system-provided")
    usage_count: int = Field(default=0, description="Number of times used")
    rating: float = Field(default=0.0, description="Average user rating")
    created_by: Optional[str] = Field(None, description="Creator user ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Research Paper Analysis",
                "description": "Analyze a research paper and extract key insights",
                "category": "research",
                "template": "Analyze the following research paper: {paper_title}\n\nFocus on:\n1. {focus_area_1}\n2. {focus_area_2}\n\nProvide a detailed analysis with citations.",
                "variables": ["paper_title", "focus_area_1", "focus_area_2"],
                "tags": ["research", "analysis", "academic"],
                "version": "1.0.0"
            }
        }


class PromptTemplateCreate(BaseModel):
    """Create prompt template request"""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: PromptCategory = Field(..., description="Template category")
    template: str = Field(..., description="Template text with placeholders")
    variables: List[str] = Field(default_factory=list, description="Template variables")
    tags: List[str] = Field(default_factory=list, description="Search tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PromptTemplateUpdate(BaseModel):
    """Update prompt template request"""
    name: Optional[str] = Field(None, description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: Optional[PromptCategory] = Field(None, description="Template category")
    template: Optional[str] = Field(None, description="Template text")
    variables: Optional[List[str]] = Field(None, description="Template variables")
    tags: Optional[List[str]] = Field(None, description="Search tags")
    is_active: Optional[bool] = Field(None, description="Whether template is active")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PromptTemplateResponse(BaseModel):
    """Prompt template response"""
    success: bool
    template: Optional[PromptTemplate] = None
    message: str


class PromptTemplateListResponse(BaseModel):
    """List of prompt templates response"""
    success: bool
    templates: List[PromptTemplate]
    total: int
    page: int
    page_size: int
    message: str


class PromptTemplateRenderRequest(BaseModel):
    """Render prompt template request"""
    template_id: str = Field(..., description="Template ID")
    variables: Dict[str, str] = Field(..., description="Variable values")


class PromptTemplateRenderResponse(BaseModel):
    """Rendered prompt response"""
    success: bool
    template_id: str
    rendered_prompt: str
    variables_used: Dict[str, str]
    message: str


class PromptTemplateRatingRequest(BaseModel):
    """Rate prompt template request"""
    template_id: str = Field(..., description="Template ID")
    rating: float = Field(..., ge=0.0, le=5.0, description="Rating (0-5)")
    feedback: Optional[str] = Field(None, description="Optional feedback")


class PromptTemplateSearchRequest(BaseModel):
    """Search prompt templates request"""
    query: Optional[str] = Field(None, description="Search query")
    category: Optional[PromptCategory] = Field(None, description="Filter by category")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    is_active: bool = Field(default=True, description="Filter by active status")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")


# Default system templates
DEFAULT_TEMPLATES = [
    {
        "name": "Research Paper Analysis",
        "description": "Analyze a research paper and extract key insights",
        "category": "research",
        "template": """Analyze the following research paper: {paper_title}

Author(s): {authors}

Focus Areas:
1. {focus_area_1}
2. {focus_area_2}
3. {focus_area_3}

Please provide:
- Executive summary
- Key findings and contributions
- Methodology analysis
- Strengths and limitations
- Practical applications
- Citations and references""",
        "variables": ["paper_title", "authors", "focus_area_1", "focus_area_2", "focus_area_3"],
        "tags": ["research", "analysis", "academic", "paper"],
        "is_system": True
    },
    {
        "name": "Code Review",
        "description": "Comprehensive code review template",
        "category": "coding",
        "template": """Review the following {language} code:

```{language}
{code}
```

Review Criteria:
- Code quality and readability
- Performance optimization
- Security vulnerabilities
- Best practices adherence
- Error handling
- Documentation
- Test coverage

Provide specific recommendations for improvement.""",
        "variables": ["language", "code"],
        "tags": ["coding", "review", "quality", "best-practices"],
        "is_system": True
    },
    {
        "name": "Data Analysis Report",
        "description": "Generate a comprehensive data analysis report",
        "category": "data_analysis",
        "template": """Analyze the following dataset: {dataset_name}

Dataset Description: {description}

Analysis Goals:
1. {goal_1}
2. {goal_2}
3. {goal_3}

Please provide:
- Data overview and statistics
- Key patterns and trends
- Anomalies and outliers
- Correlations and relationships
- Visualizations (describe what to create)
- Actionable insights
- Recommendations""",
        "variables": ["dataset_name", "description", "goal_1", "goal_2", "goal_3"],
        "tags": ["data", "analysis", "statistics", "insights"],
        "is_system": True
    },
    {
        "name": "Business Proposal",
        "description": "Create a professional business proposal",
        "category": "business",
        "template": """Create a business proposal for: {project_name}

Client: {client_name}
Industry: {industry}
Budget Range: {budget}

Proposal Sections:
1. Executive Summary
2. Problem Statement: {problem}
3. Proposed Solution: {solution}
4. Implementation Timeline
5. Deliverables
6. Pricing and ROI
7. Team and Qualifications
8. Next Steps

Make it professional, persuasive, and client-focused.""",
        "variables": ["project_name", "client_name", "industry", "budget", "problem", "solution"],
        "tags": ["business", "proposal", "sales", "professional"],
        "is_system": True
    },
    {
        "name": "Creative Story Writing",
        "description": "Generate creative story content",
        "category": "creative",
        "template": """Write a {genre} story with the following elements:

Title: {title}
Setting: {setting}
Main Character: {protagonist}
Conflict: {conflict}
Theme: {theme}

Story Requirements:
- Engaging opening hook
- Well-developed characters
- Clear plot progression
- Vivid descriptions
- Satisfying resolution
- Word count: {word_count}

Make it compelling and original.""",
        "variables": ["genre", "title", "setting", "protagonist", "conflict", "theme", "word_count"],
        "tags": ["creative", "writing", "story", "fiction"],
        "is_system": True
    },
    {
        "name": "Technical Documentation",
        "description": "Create comprehensive technical documentation",
        "category": "technical",
        "template": """Create technical documentation for: {product_name}

Type: {doc_type}
Audience: {target_audience}
Technical Level: {technical_level}

Documentation Sections:
1. Overview and Purpose
2. System Requirements
3. Installation/Setup: {setup_details}
4. Configuration
5. Usage Examples
6. API Reference (if applicable)
7. Troubleshooting
8. FAQ
9. Support and Resources

Make it clear, accurate, and user-friendly.""",
        "variables": ["product_name", "doc_type", "target_audience", "technical_level", "setup_details"],
        "tags": ["technical", "documentation", "guide", "reference"],
        "is_system": True
    },
    {
        "name": "General Task Breakdown",
        "description": "Break down complex tasks into actionable steps",
        "category": "general",
        "template": """Break down the following task into actionable steps:

Task: {task_description}

Context: {context}
Deadline: {deadline}
Resources Available: {resources}

Please provide:
1. Task analysis and scope
2. Step-by-step action plan
3. Time estimates for each step
4. Required resources and dependencies
5. Potential challenges and mitigation strategies
6. Success criteria
7. Milestones and checkpoints

Make it practical and achievable.""",
        "variables": ["task_description", "context", "deadline", "resources"],
        "tags": ["general", "planning", "productivity", "organization"],
        "is_system": True
    }
]

