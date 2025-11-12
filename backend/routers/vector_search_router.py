"""
🔍 Vector Search API Router
Provides endpoints for semantic memory storage and retrieval
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import vector search service
try:
    from services.vector_search_service import vector_search_service, CHROMADB_AVAILABLE
except ImportError:
    logger.error("Failed to import vector_search_service")
    vector_search_service = None
    CHROMADB_AVAILABLE = False

# Create router
router = APIRouter(prefix="/api/v1/vector-search", tags=["Vector Search"])


# ============================================================================
# Request/Response Models
# ============================================================================

class StoreMemoryRequest(BaseModel):
    """Request to store a memory"""
    content: str = Field(..., description="Text content to store")
    namespace: str = Field(default="default", description="Namespace for organization")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    memory_id: Optional[str] = Field(default=None, description="Optional custom ID")


class StoreMemoryResponse(BaseModel):
    """Response from storing a memory"""
    success: bool
    memory_id: str
    namespace: str


class SearchMemoriesRequest(BaseModel):
    """Request to search memories"""
    query: str = Field(..., description="Search query")
    namespace: str = Field(default="default", description="Namespace to search in")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


class MemoryResult(BaseModel):
    """A single memory search result"""
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class SearchMemoriesResponse(BaseModel):
    """Response from searching memories"""
    success: bool
    query: str
    namespace: str
    results: List[MemoryResult]
    count: int


class GetMemoryResponse(BaseModel):
    """Response from getting a specific memory"""
    success: bool
    memory: Optional[MemoryResult] = None


class DeleteMemoryResponse(BaseModel):
    """Response from deleting a memory"""
    success: bool
    deleted: bool


class CollectionStatsResponse(BaseModel):
    """Collection statistics"""
    namespace: str
    count: int
    embedding_function: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/store", response_model=StoreMemoryResponse)
async def store_memory(request: StoreMemoryRequest):
    """
    Store a memory in the vector database
    
    Stores text content with optional metadata for later semantic search.
    """
    if not CHROMADB_AVAILABLE or not vector_search_service:
        raise HTTPException(
            status_code=503,
            detail="Vector Search Service not available - install ChromaDB"
        )
    
    try:
        memory_id = await vector_search_service.store_memory(
            content=request.content,
            namespace=request.namespace,
            metadata=request.metadata,
            memory_id=request.memory_id
        )
        
        return StoreMemoryResponse(
            success=True,
            memory_id=memory_id,
            namespace=request.namespace
        )
        
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchMemoriesResponse)
async def search_memories(request: SearchMemoriesRequest):
    """
    Search for similar memories using semantic search
    
    Uses vector similarity to find relevant memories based on the query.
    Target latency: <100ms
    """
    if not CHROMADB_AVAILABLE or not vector_search_service:
        raise HTTPException(
            status_code=503,
            detail="Vector Search Service not available - install ChromaDB"
        )
    
    try:
        memories = await vector_search_service.search_memories(
            query=request.query,
            namespace=request.namespace,
            limit=request.limit,
            metadata_filter=request.metadata_filter
        )
        
        results = [
            MemoryResult(
                id=m["id"],
                content=m["content"],
                metadata=m["metadata"],
                distance=m.get("distance")
            )
            for m in memories
        ]
        
        return SearchMemoriesResponse(
            success=True,
            query=request.query,
            namespace=request.namespace,
            results=results,
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{memory_id}", response_model=GetMemoryResponse)
async def get_memory(
    memory_id: str,
    namespace: str = Query(default="default", description="Namespace")
):
    """
    Retrieve a specific memory by ID
    """
    if not CHROMADB_AVAILABLE or not vector_search_service:
        raise HTTPException(
            status_code=503,
            detail="Vector Search Service not available - install ChromaDB"
        )
    
    try:
        memory = await vector_search_service.get_memory(memory_id, namespace)
        
        if memory:
            return GetMemoryResponse(
                success=True,
                memory=MemoryResult(
                    id=memory["id"],
                    content=memory["content"],
                    metadata=memory["metadata"]
                )
            )
        else:
            return GetMemoryResponse(success=False, memory=None)
        
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/{memory_id}", response_model=DeleteMemoryResponse)
async def delete_memory(
    memory_id: str,
    namespace: str = Query(default="default", description="Namespace")
):
    """
    Delete a memory by ID
    """
    if not CHROMADB_AVAILABLE or not vector_search_service:
        raise HTTPException(
            status_code=503,
            detail="Vector Search Service not available - install ChromaDB"
        )
    
    try:
        deleted = await vector_search_service.delete_memory(memory_id, namespace)
        
        return DeleteMemoryResponse(
            success=True,
            deleted=deleted
        )
        
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{namespace}", response_model=CollectionStatsResponse)
async def get_collection_stats(namespace: str):
    """
    Get statistics for a collection/namespace
    """
    if not CHROMADB_AVAILABLE or not vector_search_service:
        raise HTTPException(
            status_code=503,
            detail="Vector Search Service not available - install ChromaDB"
        )
    
    try:
        stats = await vector_search_service.get_collection_stats(namespace)
        
        return CollectionStatsResponse(
            namespace=stats["namespace"],
            count=stats["count"],
            embedding_function=stats["embedding_function"]
        )
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Check if Vector Search Service is available
    """
    return {
        "service": "Vector Search",
        "available": CHROMADB_AVAILABLE and vector_search_service is not None,
        "chromadb_installed": CHROMADB_AVAILABLE
    }

