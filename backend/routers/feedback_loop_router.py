"""
Feedback Loop Router

REST API endpoints for continuous learning feedback loop
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from services.feedback_loop_service import (
    FeedbackLoopService,
    FeedbackEntry,
    LearningMetrics,
    SkillConfidenceAdjustment,
    FeedbackType
)
from services.reasoning_bank_service import ReasoningBankService
from services.tool_recommendation_service import ToolRecommendationService
from services.vector_search_service import VectorSearchService
from database import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/feedback", tags=["Feedback Loop"])


# ==================== REQUEST/RESPONSE MODELS ====================

class FeedbackRequest(BaseModel):
    """Request to record feedback"""
    entity_type: str = Field(..., description="Type of entity (workflow, tool, skill, pattern)")
    entity_id: str = Field(..., description="ID of the entity")
    feedback_type: str = Field(..., description="Type of feedback (success, failure, user_rating, etc.)")
    success: Optional[bool] = Field(default=None, description="Whether execution was successful")
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="User rating (0-5)")
    execution_time: Optional[float] = Field(default=None, ge=0.0, description="Execution time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    user_comment: Optional[str] = Field(default=None, description="User's comment")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    user_id: str = Field(default="system", description="User who provided feedback")


class FeedbackResponse(BaseModel):
    """Response after recording feedback"""
    success: bool
    message: str
    feedback_id: str
    learning_triggered: bool


class LearningMetricsResponse(BaseModel):
    """Response with learning metrics"""
    metrics: LearningMetrics
    recent_feedback_count: int


# ==================== DEPENDENCY INJECTION ====================

_feedback_service_instance = None


async def get_feedback_service() -> FeedbackLoopService:
    """Get or create Feedback Loop Service instance"""
    global _feedback_service_instance
    
    if _feedback_service_instance is None:
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
            
            # Initialize ReasoningBank Service
            try:
                reasoning_bank = ReasoningBankService(db, vector_search)
            except Exception as e:
                logger.warning(f"ReasoningBank not available: {e}")
                reasoning_bank = None
            
            # Initialize Tool Recommendation Service
            try:
                tool_recommendation = ToolRecommendationService(db, vector_search)
                await tool_recommendation.initialize()
            except Exception as e:
                logger.warning(f"Tool Recommendation not available: {e}")
                tool_recommendation = None
            
            # Create service
            _feedback_service_instance = FeedbackLoopService(db, reasoning_bank, tool_recommendation)
            await _feedback_service_instance.initialize()
            
            logger.info("✅ Feedback Loop Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Feedback Loop Service: {e}")
            raise HTTPException(status_code=500, detail=f"Service initialization failed: {e}")
    
    return _feedback_service_instance


# ==================== API ENDPOINTS ====================

@router.post("/record", response_model=FeedbackResponse)
async def record_feedback(
    request: FeedbackRequest,
    service: FeedbackLoopService = Depends(get_feedback_service)
):
    """
    Record feedback for continuous learning
    
    This endpoint records feedback about workflows, tools, skills, or patterns
    to enable the system to learn and improve over time.
    """
    try:
        feedback = await service.record_feedback(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            feedback_type=request.feedback_type,
            success=request.success,
            rating=request.rating,
            execution_time=request.execution_time,
            error_message=request.error_message,
            user_comment=request.user_comment,
            context=request.context,
            user_id=request.user_id
        )
        
        # Check if learning was triggered
        learning_triggered = len(service.recent_feedback) >= service.min_feedback_for_learning
        
        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded for {request.entity_type} {request.entity_id}",
            feedback_id=feedback.feedback_id,
            learning_triggered=learning_triggered
        )
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=LearningMetricsResponse)
async def get_learning_metrics(
    service: FeedbackLoopService = Depends(get_feedback_service)
):
    """
    Get current learning metrics
    
    Returns metrics about the continuous learning system including
    success rates, patterns refined, skills adjusted, etc.
    """
    try:
        metrics = await service.get_learning_metrics()
        recent_feedback_count = len(service.recent_feedback)
        
        return LearningMetricsResponse(
            metrics=metrics,
            recent_feedback_count=recent_feedback_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get learning metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def get_recent_feedback(
    limit: int = 20,
    service: FeedbackLoopService = Depends(get_feedback_service)
):
    """
    Get recent feedback entries
    
    Returns the most recent feedback entries for monitoring and debugging.
    """
    try:
        feedback_entries = await service.get_recent_feedback(limit=limit)
        
        return {
            "total_count": len(feedback_entries),
            "feedback": [f.dict() for f in feedback_entries]
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skill-adjustments")
async def get_skill_adjustments(
    skill_id: Optional[str] = None,
    limit: int = 20,
    service: FeedbackLoopService = Depends(get_feedback_service)
):
    """
    Get skill confidence adjustments
    
    Returns history of skill confidence adjustments made by the learning system.
    """
    try:
        adjustments = await service.get_skill_adjustments(skill_id=skill_id, limit=limit)
        
        return {
            "total_count": len(adjustments),
            "adjustments": [a.dict() for a in adjustments]
        }
        
    except Exception as e:
        logger.error(f"Failed to get skill adjustments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-learning")
async def trigger_learning_cycle(
    service: FeedbackLoopService = Depends(get_feedback_service)
):
    """
    Manually trigger a learning cycle
    
    Forces the system to run a learning cycle immediately, refining patterns
    and adjusting skill confidence based on accumulated feedback.
    """
    try:
        await service._trigger_learning_cycle()
        
        return {
            "success": True,
            "message": "Learning cycle triggered successfully",
            "metrics": service.learning_metrics.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger learning cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    service: FeedbackLoopService = Depends(get_feedback_service)
):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "feedback_loop",
        "recent_feedback_count": len(service.recent_feedback),
        "total_feedback": service.learning_metrics.total_feedback_entries,
        "success_rate": service.learning_metrics.success_rate,
        "components": {
            "database": service.db is not None,
            "reasoning_bank": service.reasoning_bank is not None,
            "tool_recommendation": service.tool_recommendation is not None
        }
    }

