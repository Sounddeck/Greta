"""
Tool Discovery API Router - Phase 2 Feature 1

This router provides REST API endpoints for the Progressive Tool Discovery feature:
- POST /api/v1/tool-discovery/discover - Discover relevant tools for a task
- POST /api/v1/tool-discovery/learn - Learn from execution results
- GET /api/v1/tool-discovery/stats - Get discovery statistics
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
import logging

from database import get_database, Database
from services.tool_discovery_service import get_tool_discovery_service, ToolDiscoveryService
from models.tool_discovery_models import (
    ToolDiscoveryRequest,
    ToolDiscoveryResponse,
    ToolDiscoveryStats
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tool-discovery", tags=["Tool Discovery"])


# ============================================================================
# Request/Response Models
# ============================================================================

class LearnFromExecutionRequest(BaseModel):
    """Request model for learning from execution"""
    task_description: str = Field(..., description="Original task description")
    tools_used: List[str] = Field(..., description="List of tool IDs actually used")
    tools_suggested: List[str] = Field(default_factory=list, description="List of tool IDs suggested")
    success: bool = Field(..., description="Whether execution was successful")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    agent_id: Optional[str] = Field(default=None, description="Agent ID")
    agent_type: Optional[str] = Field(default=None, description="Agent type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_description": "Scrape product prices from competitor websites",
                "tools_used": ["osint_playwright_scraper"],
                "tools_suggested": ["osint_playwright_scraper", "web_search"],
                "success": True,
                "execution_time": 5.2,
                "agent_id": "agent_abc123",
                "agent_type": "research"
            }
        }


class LearnFromExecutionResponse(BaseModel):
    """Response model for learning from execution"""
    success: bool = Field(..., description="Whether learning was successful")
    message: str = Field(..., description="Status message")
    suggestion_accuracy: float = Field(default=0.0, description="Accuracy of suggestions")
    pattern_id: Optional[str] = Field(default=None, description="Stored pattern ID")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/discover", response_model=ToolDiscoveryResponse)
async def discover_tools(
    request: ToolDiscoveryRequest,
    db: Database = Depends(get_database)
):
    """
    Discover relevant tools for a task
    
    This endpoint analyzes a task description and suggests relevant tools
    based on keyword matching, semantic similarity, and historical success rates.
    
    **Scoring Algorithm**:
    - Keyword Match: 30% weight
    - Semantic Similarity: 40% weight
    - Historical Success: 30% weight
    
    **Parameters**:
    - task_description: The task to analyze
    - agent_id: Optional agent ID for personalization
    - agent_type: Optional agent type
    - limit: Maximum number of suggestions (default: 5)
    - min_relevance: Minimum relevance threshold (default: 0.3)
    
    **Returns**:
    - task_analysis: Analysis of the task (keywords, intent, complexity)
    - suggestions: List of suggested tools with relevance scores
    - discovery_time: Time taken for discovery
    - total_tools_analyzed: Number of tools analyzed
    
    **Example**:
    ```json
    {
      "task_description": "Scrape product prices from competitor websites",
      "limit": 5,
      "min_relevance": 0.3
    }
    ```
    """
    try:
        logger.info(f"🔍 Tool discovery request: {request.task_description[:50]}...")
        
        # Get tool discovery service
        service = get_tool_discovery_service(db)
        
        # Discover tools
        response = await service.discover_tools(request)
        
        logger.info(f"✅ Discovered {len(response.suggestions)} tools in {response.discovery_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Tool discovery failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool discovery failed: {str(e)}"
        )


@router.post("/learn", response_model=LearnFromExecutionResponse)
async def learn_from_execution(
    request: LearnFromExecutionRequest,
    db: Database = Depends(get_database)
):
    """
    Learn from execution results
    
    This endpoint stores execution patterns and updates learning data to
    improve future tool suggestions. Call this after each agent execution
    to enable continuous learning.
    
    **Parameters**:
    - task_description: Original task description
    - tools_used: List of tool IDs actually used
    - tools_suggested: List of tool IDs that were suggested
    - success: Whether execution was successful
    - execution_time: Execution time in seconds
    - agent_id: Optional agent ID
    - agent_type: Optional agent type
    
    **Returns**:
    - success: Whether learning was successful
    - message: Status message
    - suggestion_accuracy: Accuracy of suggestions (0.0-1.0)
    - pattern_id: ID of stored pattern
    
    **Example**:
    ```json
    {
      "task_description": "Scrape product prices",
      "tools_used": ["osint_playwright_scraper"],
      "tools_suggested": ["osint_playwright_scraper", "web_search"],
      "success": true,
      "execution_time": 5.2
    }
    ```
    """
    try:
        logger.info(f"📚 Learning from execution: {len(request.tools_used)} tools used")
        
        # Get tool discovery service
        service = get_tool_discovery_service(db)
        
        # Learn from execution
        await service.learn_from_execution(
            task_description=request.task_description,
            tools_used=request.tools_used,
            tools_suggested=request.tools_suggested,
            success=request.success,
            execution_time=request.execution_time,
            agent_id=request.agent_id,
            agent_type=request.agent_type
        )
        
        # Calculate suggestion accuracy
        if request.tools_suggested:
            tools_used_set = set(request.tools_used)
            tools_suggested_set = set(request.tools_suggested)
            correct = tools_used_set.intersection(tools_suggested_set)
            accuracy = len(correct) / len(tools_suggested_set)
        else:
            accuracy = 0.0
        
        logger.info(f"✅ Learning complete: {accuracy:.0%} suggestion accuracy")
        
        return LearnFromExecutionResponse(
            success=True,
            message=f"Learned from execution with {accuracy:.0%} suggestion accuracy",
            suggestion_accuracy=accuracy,
            pattern_id=None  # Pattern ID is generated internally
        )
        
    except Exception as e:
        logger.error(f"❌ Learning failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Learning failed: {str(e)}"
        )


@router.get("/stats", response_model=ToolDiscoveryStats)
async def get_discovery_stats(
    db: Database = Depends(get_database)
):
    """
    Get tool discovery statistics
    
    This endpoint returns overall statistics about the tool discovery system,
    including total discoveries, suggestion accuracy, and performance metrics.
    
    **Returns**:
    - total_discoveries: Total number of discovery requests
    - total_suggestions: Total number of suggestions made
    - avg_suggestions_per_request: Average suggestions per request
    - avg_discovery_time: Average discovery time in seconds
    - suggestion_accuracy: Overall suggestion accuracy
    - top_1_accuracy: Top-1 suggestion accuracy
    - top_3_accuracy: Top-3 suggestion accuracy
    - total_patterns_learned: Total patterns learned
    - improvement_rate: Rate of improvement over time
    
    **Example Response**:
    ```json
    {
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
    ```
    """
    try:
        logger.info("📊 Fetching tool discovery statistics")
        
        # Get tool discovery service
        service = get_tool_discovery_service(db)
        
        # Get stats
        stats = await service.get_stats()
        
        logger.info(f"✅ Stats retrieved: {stats.total_discoveries} discoveries")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint for tool discovery service
    
    Returns basic health status of the tool discovery service.
    """
    return {
        "status": "healthy",
        "service": "tool_discovery",
        "version": "1.0.0"
    }

