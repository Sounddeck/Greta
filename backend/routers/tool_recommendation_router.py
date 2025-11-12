"""
Tool Recommendation Router

REST API endpoints for intelligent tool recommendations and auto-selection
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from services.tool_recommendation_service import (
    ToolRecommendationService,
    ToolRecommendation,
    ToolChainRecommendation,
    ToolPerformanceMetrics
)
from services.vector_search_service import VectorSearchService
from database import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tool-recommendations", tags=["Tool Recommendations"])


# ==================== REQUEST/RESPONSE MODELS ====================

class ToolRecommendationRequest(BaseModel):
    """Request for tool recommendations"""
    task_description: str = Field(..., description="Natural language task description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of recommendations")
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum relevance threshold")


class ToolChainRequest(BaseModel):
    """Request for tool chain recommendations"""
    task_description: str = Field(..., description="Natural language task description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of chain recommendations")


class AutoSelectRequest(BaseModel):
    """Request for automatic tool selection"""
    task_description: str = Field(..., description="Natural language task description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for auto-selection")


class ToolExecutionRecord(BaseModel):
    """Record of tool execution for performance tracking"""
    tool_id: str = Field(..., description="Tool identifier")
    success: bool = Field(..., description="Whether execution was successful")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    user_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="User rating (0-5)")


class ToolRecommendationResponse(BaseModel):
    """Response with tool recommendations"""
    recommendations: List[ToolRecommendation]
    total_count: int
    task_description: str


class ToolChainResponse(BaseModel):
    """Response with tool chain recommendations"""
    chains: List[ToolChainRecommendation]
    total_count: int
    task_description: str


class AutoSelectResponse(BaseModel):
    """Response from automatic tool selection"""
    selected_tools: List[Dict[str, Any]]
    confidence: float
    auto_selected: bool
    reason: str


# ==================== DEPENDENCY INJECTION ====================

_recommendation_service_instance = None


async def get_recommendation_service() -> ToolRecommendationService:
    """Get or create Tool Recommendation Service instance"""
    global _recommendation_service_instance
    
    if _recommendation_service_instance is None:
        try:
            # Get dependencies
            db = await get_database()
            
            # Initialize Vector Search Service
            try:
                vector_search = VectorSearchService()
                await vector_search.initialize()
            except Exception as e:
                logger.warning(f"Vector Search not available: {e}")
                vector_search = None
            
            # Create service
            _recommendation_service_instance = ToolRecommendationService(db, vector_search)
            await _recommendation_service_instance.initialize()
            
            logger.info("✅ Tool Recommendation Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tool Recommendation Service: {e}")
            raise HTTPException(status_code=500, detail=f"Service initialization failed: {e}")
    
    return _recommendation_service_instance


# ==================== API ENDPOINTS ====================

@router.post("/recommend", response_model=ToolRecommendationResponse)
async def recommend_tools(
    request: ToolRecommendationRequest,
    service: ToolRecommendationService = Depends(get_recommendation_service)
):
    """
    Get intelligent tool recommendations for a task
    
    This endpoint analyzes the task description and returns ranked tool recommendations
    based on semantic similarity and historical performance data.
    """
    try:
        recommendations = await service.recommend_tools(
            task_description=request.task_description,
            context=request.context,
            limit=request.limit,
            min_relevance=request.min_relevance
        )
        
        return ToolRecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            task_description=request.task_description
        )
        
    except Exception as e:
        logger.error(f"Tool recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-chains", response_model=ToolChainResponse)
async def recommend_tool_chains(
    request: ToolChainRequest,
    service: ToolRecommendationService = Depends(get_recommendation_service)
):
    """
    Get intelligent tool chain recommendations for complex tasks
    
    This endpoint suggests sequences of tools that work well together
    based on historical usage patterns and task analysis.
    """
    try:
        chains = await service.recommend_tool_chains(
            task_description=request.task_description,
            context=request.context,
            limit=request.limit
        )
        
        return ToolChainResponse(
            chains=chains,
            total_count=len(chains),
            task_description=request.task_description
        )
        
    except Exception as e:
        logger.error(f"Tool chain recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-select", response_model=AutoSelectResponse)
async def auto_select_tools(
    request: AutoSelectRequest,
    service: ToolRecommendationService = Depends(get_recommendation_service)
):
    """
    Automatically select the best tools for a task
    
    This endpoint uses AI to automatically select tools when confidence is high enough.
    If confidence is low, it returns recommendations for manual selection.
    """
    try:
        selected_tools, confidence = await service.auto_select_tools(
            task_description=request.task_description,
            context=request.context,
            confidence_threshold=request.confidence_threshold
        )
        
        auto_selected = len(selected_tools) > 0 and confidence >= request.confidence_threshold
        
        return AutoSelectResponse(
            selected_tools=[tool.dict() for tool in selected_tools],
            confidence=confidence,
            auto_selected=auto_selected,
            reason=(
                f"Auto-selected {len(selected_tools)} tool(s) with {confidence:.1%} confidence"
                if auto_selected
                else f"Confidence ({confidence:.1%}) below threshold ({request.confidence_threshold:.1%}) - manual selection recommended"
            )
        )
        
    except Exception as e:
        logger.error(f"Auto-selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-execution")
async def record_tool_execution(
    record: ToolExecutionRecord,
    service: ToolRecommendationService = Depends(get_recommendation_service)
):
    """
    Record tool execution for performance tracking
    
    This endpoint allows the system to learn from tool usage and improve
    future recommendations based on actual performance data.
    """
    try:
        await service.record_tool_execution(
            tool_id=record.tool_id,
            success=record.success,
            execution_time=record.execution_time,
            user_rating=record.user_rating
        )
        
        return {
            "success": True,
            "message": f"Execution recorded for tool {record.tool_id}",
            "tool_id": record.tool_id
        }
        
    except Exception as e:
        logger.error(f"Failed to record execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{tool_id}", response_model=ToolPerformanceMetrics)
async def get_tool_performance(
    tool_id: str,
    service: ToolRecommendationService = Depends(get_recommendation_service)
):
    """
    Get performance metrics for a specific tool
    
    Returns historical performance data including success rate, execution times,
    and user ratings.
    """
    try:
        if tool_id not in service.performance_cache:
            raise HTTPException(status_code=404, detail=f"No performance data for tool {tool_id}")
        
        return service.performance_cache[tool_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_all_performance_metrics(
    service: ToolRecommendationService = Depends(get_recommendation_service)
):
    """
    Get performance metrics for all tools
    
    Returns a summary of performance data for all tools in the registry.
    """
    try:
        metrics = list(service.performance_cache.values())
        
        return {
            "total_tools": len(metrics),
            "metrics": [m.dict() for m in metrics],
            "summary": {
                "total_executions": sum(m.total_executions for m in metrics),
                "average_success_rate": (
                    sum(m.success_rate for m in metrics) / len(metrics)
                    if metrics else 0.0
                ),
                "tools_with_ratings": sum(1 for m in metrics if m.user_ratings)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    service: ToolRecommendationService = Depends(get_recommendation_service)
):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tool_recommendation",
        "cached_metrics": len(service.performance_cache),
        "cached_patterns": len(service.chain_patterns_cache),
        "components": {
            "database": service.db is not None,
            "vector_search": service.vector_search is not None
        }
    }

