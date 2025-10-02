"""
PAI Memory Router - Real memory operations and orchestration
Provides API endpoints for the intelligent memory system
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from database import pai_database
from services.memory_orchestrator import pai_memory_orchestrator

router = APIRouter()
logger = logging.getLogger(__name__)

# Request/Response Models

class StoreMemoryRequest(BaseModel):
    user_id: str
    content: str
    tags: List[str] = []
    importance_score: float = 0.5
    metadata: Dict[str, Any] = {}

class RetrieveMemoriesRequest(BaseModel):
    user_id: str
    tags: Optional[List[str]] = None
    limit: int = 20
    min_importance: float = 0.0

class MemoryAnalysisRequest(BaseModel):
    user_id: str
    analysis_type: str = "general"  # general, patterns, learning_progress
    time_range_days: int = 30

class ConversationContextRequest(BaseModel):
    conversation_id: str
    context_depth: int = 5

@router.post("/store")
async def store_memory(request: StoreMemoryRequest) -> Dict[str, str]:
    """Store memory in PAI system"""
    try:
        memory_result = await pai_memory_orchestrator.intelligent_store({
            "user_id": request.user_id,
            "content": request.content,
            "tags": request.tags,
            "importance_score": request.importance_score,
            "metadata": request.metadata,
            "stored_at": datetime.utcnow().isoformat()
        }, {})
        
        return {
            "status": "stored",
            "memory_id": memory_result.get("store_id", "unknown"),
            "intelligence_score": memory_result.get("intelligence_score", 0.0)
        }
    except Exception as e:
        logger.error(f"Memory storage failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")

@router.post("/retrieve")
async def retrieve_memories(request: RetrieveMemoriesRequest) -> Dict[str, Any]:
    """Retrieve contextual memories for user"""
    try:
        query = {"user_id": request.user_id}
        if request.tags:
            query["tags"] = {"$in": request.tags}
        
        memories = await pai_database.retrieve_memories(query, limit=request.limit)
        
        # Filter by importance
        filtered_memories = [m for m in memories if m.get("importance_score", 0) >= request.min_importance]
        
        return {
            "memories": filtered_memories,
            "total_count": len(filtered_memories),
            "context_synthesis": True
        }
    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

@router.get("/context/{conversation_id}")
async def get_conversation_context(conversation_id: str, context_depth: int = 5) -> Dict[str, Any]:
    """Get full PAI context for a conversation"""
    try:
        context = await pai_database.get_conversation_context(conversation_id, context_depth)
        
        if not context:
            return {"error": "Conversation context not found"}
        
        return {
            "conversation": context.get("conversation", {}),
            "related_memories": context.get("related_memories", []),
            "relevant_insights": context.get("relevant_insights", []),
            "pai_analysis": context.get("pai_analysis", {}),
            "context_readiness": context.get("pai_analysis", {}).get("pai_context_readiness", False)
        }
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation context: {str(e)}")

@router.get("/statistics/{user_id}")
async def get_memory_statistics(user_id: str) -> Dict[str, Any]:
    """Get comprehensive memory statistics for user"""
    try:
        # Query user's memories
        memory_query = {"user_id": user_id}
        user_memories = await pai_database.retrieve_memories(memory_query, limit=1000)
        
        # Calculate statistics
        total_memories = len(user_memories)
        avg_importance = sum(m.get("importance_score", 0.5) for m in user_memories) / total_memories if total_memories > 0 else 0.5
        tags = []
        for memory in user_memories:
            tags.extend(memory.get("tags", []))
        
        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_memories": total_memories,
            "average_importance_score": avg_importance,
            "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "memory_distribution": {
                "high_importance": len([m for m in user_memories if m.get("importance_score", 0) > 0.7]),
                "medium_importance": len([m for m in user_memories if 0.4 <= m.get("importance_score", 0) <= 0.7]),
                "low_importance": len([m for m in user_memories if m.get("importance_score", 0) < 0.4])
            },
            "pai_memory_intelligence": {
                "pattern_recognition_active": True,
                "context_synthesis_enabled": True,
                "adaptive_learning": True
            }
        }
    except Exception as e:
        logger.error(f"Memory statistics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory statistics: {str(e)}")

@router.post("/analyze/{user_id}")
async def analyze_memory_patterns(user_id: str, request: MemoryAnalysisRequest) -> Dict[str, Any]:
    """Analyze memory patterns and learning progress"""
    try:
        analysis_result = await pai_memory_orchestrator.pattern_analysis(
            user_id, request.analysis_type, request.time_range_days
        )
        
        return {
            "analysis_type": request.analysis_type,
            "time_range": f"{request.time_range_days} days",
            "patterns_detected": analysis_result.get("patterns", []),
            "learning_progress": analysis_result.get("learning_progress", {}),
            "insights_generated": analysis_result.get("insights", []),
            "pai_analysis": {
                "intelligence_growth": analysis_result.get("intelligence_growth", 0.0),
                "pattern_complexity": analysis_result.get("pattern_complexity", 0.0),
                "adaptation_rate": analysis_result.get("adaptation_rate", 0.0)
            }
        }
    except Exception as e:
        logger.error(f"Memory analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze memory patterns: {str(e)}")

@router.post("/consolidate/{user_id}")
async def consolidate_memories(user_id: str) -> Dict[str, Any]:
    """Consolidate and optimize user's memory space"""
    try:
        maintenance_result = await pai_memory_orchestrator.memory_maintenance()
        
        return {
            "status": "maintenance_complete",
            "actions_performed": maintenance_result.get("actions_performed", []),
            "memory_stats": maintenance_result.get("memory_stats", {}),
            "improvements_made": maintenance_result.get("performance_metrics", {}),
            "next_consolidation": "Scheduled for automated daily execution"
        }
    except Exception as e:
        logger.error(f"Memory consolidation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to consolidate memories: {str(e)}")

@router.delete("/cleanup/{user_id}")
async def cleanup_old_memories(user_id: str, older_than_days: int = 180) -> Dict[str, Any]:
    """Clean up old memories based on age and importance"""
    try:
        cleanup_query = {
            "user_id": user_id,
            "stored_at": {"$lt": datetime.utcnow().timestamp() - (older_than_days * 24 * 60 * 60)},
            "importance_score": {"$lt": 0.3}  # Only delete low-importance old memories
        }
        
        result = await pai_database.db.memory_store.delete_many(cleanup_query)
        
        return {
            "cleanup_executed": True,
            "memories_removed": result.deleted_count,
            "cleanup_criteria": {
                "older_than_days": older_than_days,
                "importance_threshold": 0.3
            },
            "pai_note": "PAI system recommends regular cleanup to maintain optimal performance"
        }
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup memories: {str(e)}")

@router.get("/health")
async def memory_health_check() -> Dict[str, Any]:
    """Check memory system health"""
    try:
        # Test database connectivity
        db_health = await pai_database.health_check()
        
        # Test memory orchestrator
        memory_stats = await pai_memory_orchestrator.get_memory_statistics()
        
        return {
            "database": db_health,
            "memory_orchestrator": {
                "status": "healthy" if memory_stats else "degraded",
                "capabilities": [
                    "pattern_recognition",
                    "context_synthesis",
                    "learning_optimization"
                ]
            },
            "overall_status": "healthy" if db_health.get("status") == "healthy" and memory_stats else "degraded"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

