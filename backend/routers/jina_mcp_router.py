"""
Jina AI MCP Router
API endpoints for Jina AI search and embedding capabilities

Endpoints:
- POST /api/v1/jina/search - AI-powered web search
- POST /api/v1/jina/embed - Generate text embeddings
- POST /api/v1/jina/analyze - Analyze content from URL
- GET /api/v1/jina/metrics - Service metrics
- GET /api/v1/jina/health - Health check
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from services.jina_mcp_service import get_jina_service
from middleware.rate_limit import limiter, get_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jina", tags=["Jina MCP"])


# ==================== Request/Response Models ====================

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    limit: int = Field(10, description="Maximum number of results", ge=1, le=50)
    include_content: bool = Field(True, description="Include full content in results")


class EmbedRequest(BaseModel):
    """Embedding request model"""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)
    model: Optional[str] = Field(None, description="Model to use (default: jina-embeddings-v2-base-en)")


class AnalyzeRequest(BaseModel):
    """Content analysis request model"""
    url: str = Field(..., description="URL to analyze", min_length=1, max_length=2000)
    extract_images: bool = Field(False, description="Extract images from content")


class JinaResponse(BaseModel):
    """Generic Jina response model"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== API Endpoints ====================

@router.post("/search", response_model=JinaResponse)
@limiter.limit(get_rate_limit("jina"))
async def search(request: SearchRequest):
    """
    Perform AI-powered web search using Jina AI
    
    **Features:**
    - Semantic search with AI understanding
    - Ranked results by relevance
    - Optional full content extraction
    - Mock mode when API key not configured
    
    **Rate Limit:** 20 requests/minute
    """
    try:
        service = get_jina_service()
        result = await service.search(
            query=request.query,
            limit=request.limit,
            include_content=request.include_content
        )
        
        if result.get("success"):
            return JinaResponse(success=True, data=result)
        else:
            return JinaResponse(
                success=False,
                error=result.get("error", "Search failed")
            )
    
    except Exception as e:
        logger.error(f"Jina search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed", response_model=JinaResponse)
@limiter.limit(get_rate_limit("jina"))
async def embed(request: EmbedRequest):
    """
    Generate text embeddings using Jina AI
    
    **Features:**
    - High-quality text embeddings
    - Support for multiple texts in one request
    - Configurable embedding model
    - Mock mode when API key not configured
    
    **Rate Limit:** 20 requests/minute
    """
    try:
        service = get_jina_service()
        result = await service.embed(
            texts=request.texts,
            model=request.model
        )
        
        if result.get("success"):
            return JinaResponse(success=True, data=result)
        else:
            return JinaResponse(
                success=False,
                error=result.get("error", "Embedding generation failed")
            )
    
    except Exception as e:
        logger.error(f"Jina embed endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=JinaResponse)
@limiter.limit(get_rate_limit("jina"))
async def analyze_content(request: AnalyzeRequest):
    """
    Analyze and extract content from a URL using Jina AI
    
    **Features:**
    - Clean content extraction
    - Metadata extraction (title, author, date)
    - Optional image extraction
    - Mock mode when API key not configured
    
    **Rate Limit:** 20 requests/minute
    """
    try:
        service = get_jina_service()
        result = await service.analyze_content(
            url=request.url,
            extract_images=request.extract_images
        )
        
        if result.get("success"):
            return JinaResponse(success=True, data=result)
        else:
            return JinaResponse(
                success=False,
                error=result.get("error", "Content analysis failed")
            )
    
    except Exception as e:
        logger.error(f"Jina analyze endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """
    Get Jina MCP service metrics
    
    Returns usage statistics and configuration
    """
    try:
        service = get_jina_service()
        return service.get_metrics()
    
    except Exception as e:
        logger.error(f"Jina metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Jina MCP service
    
    Returns service status and configuration
    """
    try:
        service = get_jina_service()
        return service.get_health()
    
    except Exception as e:
        logger.error(f"Jina health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== System-Wide MCP Configuration ====================

def get_jina_mcp_config() -> Dict[str, Any]:
    """
    Get Jina MCP server configuration for system-wide registration
    
    This configuration makes Jina available to all agents by default
    """
    return {
        "name": "jina",
        "display_name": "Jina AI",
        "description": "AI-powered search, embeddings, and content analysis",
        "enabled": True,
        "system_wide": True,  # Available to all agents
        "capabilities": [
            "search",
            "embeddings",
            "content_analysis",
        ],
        "endpoints": {
            "search": "/api/v1/jina/search",
            "embed": "/api/v1/jina/embed",
            "analyze": "/api/v1/jina/analyze",
            "metrics": "/api/v1/jina/metrics",
            "health": "/api/v1/jina/health",
        },
        "rate_limits": {
            "search": "20/minute",
            "embed": "20/minute",
            "analyze": "20/minute",
        },
        "requires_api_key": True,
        "api_key_env_var": "JINA_API_KEY",
        "mock_mode_available": True,
        "documentation": "https://jina.ai/docs",
    }

