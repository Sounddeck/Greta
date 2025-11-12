"""
GRETA Functional Memory Router
Complete API endpoints for memory management
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import database
from services.llm_service import get_llm_service
from services.learning_service import LearningService
from services.functional_memory_service import FunctionalMemoryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/functional-memory", tags=["Functional Memory"])

# Global instance
memory_instance: Optional[FunctionalMemoryService] = None

# Pydantic models
class MemoryStoreRequest(BaseModel):
    """Memory storage request"""
    content: str = Field(..., description="Memory content")
    type: str = Field(..., description="Memory type (episodic, semantic, procedural, working)")
    importance: float = Field(default=0.5, description="Memory importance (0-1)")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SemanticSearchRequest(BaseModel):
    """Semantic search request"""
    query: str = Field(..., description="Search query")
    memory_type: Optional[str] = Field(None, description="Filter by memory type")
    limit: int = Field(default=10, description="Maximum results")

class ContextAddRequest(BaseModel):
    """Add to context request"""
    interaction: Dict[str, Any] = Field(..., description="Interaction data")

async def get_memory() -> FunctionalMemoryService:
    """Get or initialize Functional Memory instance"""
    global memory_instance
    
    if memory_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()
        
        memory_instance = FunctionalMemoryService(db, llm_service, learning_service)
        await memory_instance.initialize()
    
    return memory_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "functional_memory",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(service: FunctionalMemoryService = Depends(get_memory)):
    """Get service metrics"""
    return await service.get_metrics()

# ==================== MEMORY STORAGE & RETRIEVAL ====================

@router.post("/memories")
async def store_memory(
    request: MemoryStoreRequest,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Store a new memory"""
    try:
        result = await service.store_memory({
            "content": request.content,
            "type": request.type,
            "importance": request.importance,
            "tags": request.tags,
            "metadata": request.metadata
        })

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memories")
async def get_all_memories(
    memory_type: Optional[str] = None,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Get all memories"""
    try:
        result = await service.get_all_memories(memory_type)
        return result
    except Exception as e:
        logger.error(f"Error getting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memories/{memory_id}")
async def get_memory_by_id(
    memory_id: str,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Get a specific memory"""
    try:
        result = await service.retrieve_memory(memory_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Delete a memory"""
    try:
        result = await service.delete_memory(memory_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SEMANTIC SEARCH ====================

@router.post("/search/semantic")
async def semantic_search(
    request: SemanticSearchRequest,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Semantic search using vector similarity"""
    try:
        result = await service.semantic_search(
            request.query,
            request.limit,
            request.memory_type
        )
        return result
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MEMORY CONSOLIDATION ====================

@router.post("/consolidate")
async def consolidate_memories(service: FunctionalMemoryService = Depends(get_memory)):
    """Consolidate memories"""
    try:
        result = await service.consolidate_memories()
        return result
    except Exception as e:
        logger.error(f"Error consolidating memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== KNOWLEDGE GRAPH ====================

@router.get("/knowledge-graph")
async def get_knowledge_graph(service: FunctionalMemoryService = Depends(get_memory)):
    """Get knowledge graph"""
    try:
        result = await service.get_knowledge_graph()
        return result
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CONTEXT MANAGEMENT ====================

@router.post("/context")
async def add_to_context(
    request: ContextAddRequest,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Add interaction to context window"""
    try:
        result = await service.add_to_context(request.interaction)
        return result
    except Exception as e:
        logger.error(f"Error adding to context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context")
async def get_context(
    limit: int = 10,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Get recent context"""
    try:
        result = await service.get_context(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== METADATA ====================

@router.get("/memory-types")
async def get_memory_types(service: FunctionalMemoryService = Depends(get_memory)):
    """Get available memory types"""
    try:
        result = await service.get_memory_types()
        return result
    except Exception as e:
        logger.error(f"Error getting memory types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    service: FunctionalMemoryService = Depends(get_memory)
):
    """Get activity log"""
    try:
        result = await service.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

