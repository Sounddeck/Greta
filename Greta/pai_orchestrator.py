"""
PAI Orchestrator Router - API for hierarchical reasoning and intelligence
Provides endpoints for PAI's system intelligence beyond LLM capabilities
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from services.llm_enhanced import pai_intelligence_orchestrator
from services.memory_orchestrator import pai_memory_orchestrator
from database import pai_database

router = APIRouter()
logger = logging.getLogger(__name__)

# Request/Response Models

class IntelligenceRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    task_type: str = "general"
    context_depth: int = 3
    reasoning_priority: str = "balanced"  # fast, balanced, thorough
    include_metadata: bool = True

class ReasoningAnalysisRequest(BaseModel):
    input_text: str
    analysis_type: str = "full"  # perception, analysis, synthesis, decision, full
    include_trace: bool = True

class SystemIntelligenceQuery(BaseModel):
    query_type: str = "performance"  # performance, learning, capabilities, statistics
    time_range_days: int = 30

class AdaptLearningRequest(BaseModel):
    user_feedback: str
    interaction_quality: float = 0.8
    preferred_adaptations: List[str] = []

@router.post("/process")
async def process_with_pai_intelligence(request: IntelligenceRequest) -> Dict[str, Any]:
    """Process request using PAI's system intelligence (system smarter than LLM)"""
    try:
        # Prepare input context
        context = {
            "message": request.message,
            "user_id": request.user_id,
            "task": request.message.split()[0] if request.message.split() else "unknown",
            "request_metadata": {
                "task_type": request.task_type,
                "context_depth": request.context_depth,
                "reasoning_priority": request.reasoning_priority
            }
        }
        
        # Process with PAI intelligence orchestrator
        result = await pai_intelligence_orchestrator.process_intelligently(context)
        
        return {
            "response": result["response"],
            "pai_managed": True,
            "intelligence_trace": result["intelligence_trace"] if request.include_metadata else None,
            "processing_note": "This response was orchestrated by PAI system intelligence - smarter than individual LLM",
            "metadata": {
                "system_confidence": result["intelligence_trace"].get("system_reasoning_confidence", 0),
                "capability_used": "hierarchical_reasoning + context_synthesis + llm_augmentation",
                "pai_signature": result.get("pai_signature", "")
            }
        }
    except Exception as e:
        logger.error(f"PAI processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"PAI processing error: {str(e)}")

@router.post("/reason")
async def analyze_with_hierarchical_reasoning(request: ReasoningAnalysisRequest) -> Dict[str, Any]:
    """Analyze input using PAI's 4-layer hierarchical reasoning"""
    try:
        input_context = {
            "message": request.input_text,
            "task_type": "analysis",
            "reasoning_scope": request.analysis_type
        }
        
        # Process through hierarchical reasoner
        reasoning_result = await pai_intelligence_orchestrator.hierarchical_reasoner.reason_hierarchically(
            input_context, input_context.get("task_type", "")
        )
        
        response = {
            "reasoning_analysis": {
                "perception_layer": reasoning_result.get("perception", {}),
                "analysis_layer": reasoning_result.get("analysis", {}),
                "synthesis_layer": reasoning_result.get("synthesis", {}),
                "decision_layer": reasoning_result.get("decision", {})
            },
            "overall_confidence": reasoning_result.get("confidence", 0),
            "processing_time": reasoning_result.get("processing_time", 0),
            "pai_methodology": "4-layer hierarchical reasoning system"
        }
        
        if request.include_trace:
            response["reasoning_trace"] = {
                "trace_available": reasoning_result.get("reasoning_trace", False),
                "confidence_evolution": [
                    reasoning_result.get("perception", {}).get("confidence", 0),
                    reasoning_result.get("analysis", {}).get("estimated_complexity_confidence", 0.5),
                    reasoning_result.get("synthesis", {}).get("strategy_confidence", 0.5),
                    reasoning_result.get("decision", {}).get("final_confidence", 0)]
            }
        
        return response
    except Exception as e:
        logger.error(f"Hierarchical reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hierarchical reasoning error: {str(e)}")

@router.get("/status")
async def get_pai_system_status() -> Dict[str, Any]:
    """Get comprehensive PAI system status"""
    try:
        # Get PAI orchestrator stats
        intel_stats = {
            "operations_processed": pai_intelligence_orchestrator.operation_count,
            "average_confidence": pai_intelligence_orchestrator.average_confidence,
            "learning_enabled": pai_intelligence_orchestrator.learning_enabled
        }
        
        # Get memory system stats
        memory_stats = await pai_memory_orchestrator.get_memory_statistics()
        
        # Get database stats
        db_stats = await pai_database.get_pai

