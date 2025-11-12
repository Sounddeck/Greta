"""
Figma Models for CPAS5
Data models for Figma-to-Code functionality
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Framework(str, Enum):
    """Supported frameworks for code generation"""
    REACT = "react"
    VUE = "vue"
    SVELTE = "svelte"
    HTML = "html"
    ANGULAR = "angular"
    NEXT = "next"


class FigmaCodeGenerateRequest(BaseModel):
    """Request to generate code from Figma design"""
    figma_url: str = Field(..., description="Figma file or frame URL")
    framework: Framework = Field(default=Framework.REACT, description="Target framework")
    component_name: str = Field(..., description="Name for the generated component")
    use_cache: bool = Field(default=True, description="Use cached results if available")
    include_styles: bool = Field(default=True, description="Include CSS/styles")
    responsive: bool = Field(default=True, description="Generate responsive code")


class FigmaCodeGenerateResponse(BaseModel):
    """Response with generated code"""
    code: str = Field(..., description="Generated component code")
    styles: Optional[str] = Field(None, description="Generated CSS/styles")
    framework: str = Field(..., description="Framework used")
    component_name: str = Field(..., description="Component name")
    file_id: str = Field(..., description="Figma file ID")
    node_id: Optional[str] = Field(None, description="Figma node ID if specific frame")
    generated_at: str = Field(..., description="Generation timestamp")
    dependencies: List[str] = Field(default_factory=list, description="Required npm packages")
    instructions: Optional[str] = Field(None, description="Setup/usage instructions")


class FigmaFileInfo(BaseModel):
    """Figma file information"""
    file_id: str
    file_name: str
    last_modified: datetime
    thumbnail_url: Optional[str] = None
    version: str


class FigmaNode(BaseModel):
    """Figma node/frame information"""
    node_id: str
    name: str
    type: str
    width: float
    height: float
    children_count: int

