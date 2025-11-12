"""
Master Agent Knowledge Base Models
Defines the schema for system knowledge entries stored in MongoDB
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class KnowledgeCategory(str, Enum):
    """Knowledge entry categories"""
    ARCHITECTURE = "architecture"
    FEATURES = "features"
    API_ENDPOINTS = "api_endpoints"
    WORKFLOWS = "workflows"
    CONFIGURATION = "configuration"
    CAPABILITIES = "capabilities"
    INTEGRATIONS = "integrations"
    TROUBLESHOOTING = "troubleshooting"


class SystemKnowledgeEntry(BaseModel):
    """System knowledge base entry"""
    category: KnowledgeCategory = Field(..., description="Knowledge category")
    title: str = Field(..., description="Descriptive title")
    content: str = Field(..., description="Detailed markdown-formatted content")
    tags: List[str] = Field(default_factory=list, description="Search tags")
    priority: int = Field(default=5, ge=1, le=10, description="Priority (1-10)")
    version: str = Field(default="1.0.0", description="Documentation version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "category": "architecture",
                "title": "CPAS5 System Architecture",
                "content": "# CPAS5 Architecture\n\nThe system is built on...",
                "tags": ["architecture", "system-design", "overview"],
                "priority": 10,
                "version": "1.0.0"
            }
        }


class SystemKnowledgeResponse(SystemKnowledgeEntry):
    """Response model with MongoDB ID"""
    id: str = Field(..., description="MongoDB document ID")


class KnowledgeSearchRequest(BaseModel):
    """Search request for knowledge base"""
    query: str = Field(..., description="Search query")
    category: Optional[KnowledgeCategory] = Field(None, description="Filter by category")
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class KnowledgeSearchResponse(BaseModel):
    """Search response"""
    results: List[SystemKnowledgeResponse]
    total: int
    query: str
    category: Optional[str] = None


class KnowledgeListResponse(BaseModel):
    """List response with pagination"""
    entries: List[SystemKnowledgeResponse]
    total: int
    category: Optional[str] = None
    page: int
    page_size: int
    total_pages: int


class MasterAgentContextRequest(BaseModel):
    """Request for Master Agent context initialization"""
    include_categories: Optional[List[KnowledgeCategory]] = Field(
        default=None,
        description="Specific categories to include"
    )
    max_entries: int = Field(default=20, description="Maximum entries to retrieve")
    min_priority: int = Field(default=5, description="Minimum priority threshold")


class MasterAgentContextResponse(BaseModel):
    """Master Agent context response"""
    system_overview: str
    available_endpoints: List[Dict[str, str]]
    capabilities: List[str]
    workflows: List[Dict[str, str]]
    configuration_notes: str
    knowledge_entries: List[SystemKnowledgeResponse]
    generated_at: datetime = Field(default_factory=datetime.utcnow)

