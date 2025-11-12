"""
Figma Router for CPAS5
REST API endpoints for Figma-to-Code functionality
"""

from fastapi import APIRouter, HTTPException, Depends
import logging

from database import Database
from services.figma_service import FigmaService
from models.figma_models import (
    FigmaCodeGenerateRequest,
    FigmaCodeGenerateResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/figma", tags=["Figma"])


async def get_figma_service() -> FigmaService:
    """Dependency to get Figma service instance"""
    db = Database()
    await db.connect()
    return FigmaService(db)


@router.post("/generate-code", response_model=FigmaCodeGenerateResponse)
async def generate_code_from_figma(
    request: FigmaCodeGenerateRequest,
    figma_service: FigmaService = Depends(get_figma_service)
):
    """
    Generate production-ready code from Figma design
    
    **Parameters:**
    - `figma_url`: Figma file or frame URL
    - `framework`: Target framework (react, vue, svelte, html, angular, next)
    - `component_name`: Name for the generated component
    - `use_cache`: Use cached results if available (default: true)
    - `include_styles`: Include CSS/styles (default: true)
    - `responsive`: Generate responsive code (default: true)
    
    **Returns:**
    - Generated component code
    - CSS/styles (if requested)
    - Required dependencies
    - Setup instructions
    
    **Example:**
    ```json
    {
      "figma_url": "https://www.figma.com/file/ABC123/Design-Name",
      "framework": "react",
      "component_name": "HeroSection",
      "use_cache": true,
      "include_styles": true,
      "responsive": true
    }
    ```
    """
    try:
        result = await figma_service.generate_code(request)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate code: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    import os
    
    figma_api_key_set = bool(os.getenv("FIGMA_API_KEY"))
    
    return {
        "status": "healthy" if figma_api_key_set else "degraded",
        "module": "Figma-to-Code",
        "figma_api_configured": figma_api_key_set,
        "features": [
            "Code Generation from Figma Designs",
            "Multi-Framework Support (React, Vue, Svelte, HTML)",
            "Responsive Code Generation",
            "CSS/Styles Extraction",
            "Dependency Management",
            "Result Caching"
        ],
        "supported_frameworks": ["react", "vue", "svelte", "html", "angular", "next"],
        "endpoints": 2,
        "database": "MongoDB/File Storage"
    }


@router.delete("/cache")
async def clear_cache(figma_service: FigmaService = Depends(get_figma_service)):
    """Clear the Figma code generation cache"""
    try:
        result = await figma_service.db.delete_many(
            figma_service.cache_collection,
            {}
        )
        return {
            "success": True,
            "message": f"Cache cleared successfully",
            "deleted_count": result.get("deleted_count", 0)
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

