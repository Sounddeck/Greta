"""
GRETA Knowledge Base Router
Complete API endpoints for knowledge management
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
from services.knowledge_base_service import KnowledgeBaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base"])

# Global instance
knowledge_instance: Optional[KnowledgeBaseService] = None

# Pydantic models
class DocumentAddRequest(BaseModel):
    """Document addition request"""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: Optional[str] = Field(None, description="Document category (auto-categorized if not provided)")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DocumentUpdateRequest(BaseModel):
    """Document update request"""
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    """Keyword search request"""
    search_term: str = Field(..., description="Search term")

class ImageGenerationRequest(BaseModel):
    """Image generation request"""
    prompt: str = Field(..., description="Text description of the image to generate")
    provider: Optional[str] = Field(None, description="Image generation provider (stable_diffusion, dalle, midjourney)")
    negative_prompt: Optional[str] = Field(None, description="What to avoid in the image")
    width: int = Field(default=512, description="Image width in pixels")
    height: int = Field(default=512, description="Image height in pixels")
    num_inference_steps: int = Field(default=20, description="Number of denoising steps (more = better quality)")
    guidance_scale: float = Field(default=7.5, description="How closely to follow the prompt (7-15 recommended)")
    save_to_knowledge_base: bool = Field(default=False, description="Save generated image to knowledge base")
    category: Optional[str] = Field(None, description="Filter by category")
    tags: List[str] = Field(default_factory=list, description="Filter by tags")

class SemanticSearchRequest(BaseModel):
    """Semantic search request"""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum results")
    category: Optional[str] = Field(None, description="Filter by category")

async def get_knowledge() -> KnowledgeBaseService:
    """Get or initialize Knowledge Base instance"""
    global knowledge_instance
    
    if knowledge_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()
        
        knowledge_instance = KnowledgeBaseService(db, llm_service, learning_service)
        await knowledge_instance.initialize()
    
    return knowledge_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "knowledge_base",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(service: KnowledgeBaseService = Depends(get_knowledge)):
    """Get service metrics"""
    return await service.get_metrics()

# ==================== DOCUMENTS ====================

@router.post("/documents")
async def add_document(
    request: DocumentAddRequest,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Add a new document"""
    try:
        result = await service.add_document(request.dict())

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def get_all_documents(
    category: Optional[str] = None,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Get all documents"""
    try:
        result = await service.get_all_documents(category)
        return result
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Get a specific document"""
    try:
        result = await service.get_document(document_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/documents/{document_id}")
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Update a document"""
    try:
        result = await service.update_document(document_id, request.dict(exclude_unset=True))

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Delete a document"""
    try:
        result = await service.delete_document(document_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SEARCH ====================

@router.post("/search")
async def search_documents(
    request: SearchRequest,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Search documents with keywords"""
    try:
        result = await service.search_documents(request.dict())
        return result
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/semantic")
async def semantic_search(
    request: SemanticSearchRequest,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Semantic search using embeddings"""
    try:
        result = await service.semantic_search(
            request.query,
            request.limit,
            request.category
        )
        return result
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CATEGORIES ====================

@router.get("/categories")
async def get_categories(service: KnowledgeBaseService = Depends(get_knowledge)):
    """Get all categories"""
    try:
        result = await service.get_categories()
        return result
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== KNOWLEDGE GRAPH ====================

@router.get("/knowledge-graph")
async def get_knowledge_graph(service: KnowledgeBaseService = Depends(get_knowledge)):
    """Get knowledge graph"""
    try:
        result = await service.get_knowledge_graph()
        return result
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """Get activity log"""
    try:
        result = await service.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== IMAGE GENERATION ====================

@router.post("/generate-image")
async def generate_image(
    request: ImageGenerationRequest,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """
    Generate image from text prompt using AI

    Supports multiple providers:
    - stable_diffusion: Free, local generation (default)
    - dalle: OpenAI DALL-E (requires API key)
    - midjourney: Midjourney (requires API key)

    Returns base64-encoded image data
    """
    try:
        result = await service.generate_image(
            prompt=request.prompt,
            provider=request.provider,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            save_to_knowledge_base=request.save_to_knowledge_base
        )
        return result
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/image-providers")
async def list_image_providers(
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """
    List all available image generation providers

    Returns information about each provider including:
    - Name and description
    - Cost (free/paid)
    - Availability status
    - Supported features
    """
    try:
        providers = service.list_image_providers()
        return {
            "success": True,
            "providers": providers,
            "default_provider": "stable_diffusion"
        }
    except Exception as e:
        logger.error(f"Error listing image providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/image-providers/{provider_name}")
async def get_image_provider_info(
    provider_name: str,
    service: KnowledgeBaseService = Depends(get_knowledge)
):
    """
    Get detailed information about a specific image generation provider

    Args:
        provider_name: Name of the provider (stable_diffusion, dalle, midjourney)
    """
    try:
        info = service.get_image_provider_info(provider_name)
        return {
            "success": True,
            "provider": info
        }
    except Exception as e:
        logger.error(f"Error getting provider info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

