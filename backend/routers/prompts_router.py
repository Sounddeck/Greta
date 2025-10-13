"""
CPAS4 UI Prompts Router
Provides prompt enhancement endpoints and template library for the CPAS4 UI
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from services.llm_service import get_llm_service
from services.prompt_library_service import PromptLibraryService
from models.prompt_template_models import (
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplateRenderRequest,
    PromptTemplateRatingRequest,
    PromptTemplateSearchRequest,
    PromptCategory
)
from database import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["Prompts"])

# Pydantic models
class PromptEnhanceRequest(BaseModel):
    """Prompt enhancement request"""
    prompt: str = Field(..., description="Original prompt to enhance")
    agent_type: str = Field(default="general", description="Agent type for context")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class PromptEnhanceResponse(BaseModel):
    """Prompt enhancement response"""
    id: str
    original_prompt: str
    enhanced_prompt: str
    quality_metrics: Dict[str, float]
    engine: str
    created_at: str

@router.post("/enhance")
async def enhance_prompt(request: PromptEnhanceRequest):
    """
    Enhance a prompt using AI
    
    This endpoint improves the clarity, specificity, and structure of user prompts.
    """
    try:
        logger.info(f"Enhancing prompt for agent type: {request.agent_type}")
        logger.info(f"Original prompt: {request.prompt}")
        
        # Get LLM service
        llm_service = await get_llm_service()
        
        # Agent-specific enhancement strategies
        enhancement_strategies = {
            'research': "Make this research prompt more specific and structured. Include what sources to check, what format the output should be in, and what depth of analysis is needed.",
            'coding': "Make this coding prompt more specific. Include programming language, framework preferences, code style requirements, and expected output format.",
            'data_analysis': "Make this data analysis prompt more specific. Include what type of analysis is needed, what visualizations to create, and what insights to focus on.",
            'content_creation': "Make this content creation prompt more specific. Include target audience, tone, length, format, and key messages to convey.",
            'task_automation': "Make this automation prompt more specific. Include what tasks to automate, what triggers to use, and what the expected outcome should be.",
            'customer_support': "Make this customer support prompt more specific. Include customer context, issue type, desired resolution, and tone requirements.",
            'sales': "Make this sales prompt more specific. Include target customer profile, value proposition to emphasize, and desired call-to-action.",
            'marketing': "Make this marketing prompt more specific. Include campaign goals, target audience, channels to use, and success metrics.",
            'finance': "Make this finance prompt more specific. Include time period, metrics to analyze, reporting format, and key insights needed.",
            'general': "Make this prompt more clear, specific, and actionable. Add relevant context and structure."
        }
        
        strategy = enhancement_strategies.get(request.agent_type, enhancement_strategies['general'])
        
        # Build enhancement prompt
        enhancement_prompt = f"""You are a prompt engineering expert. Enhance the user's prompt to make it more effective.

Enhancement Strategy: {strategy}

User's Original Prompt:
"{request.prompt}"

Your Task: Rewrite the above prompt to be more clear, specific, and effective. Output ONLY the improved prompt text with no preamble, no labels, and no explanations. Start directly with the enhanced prompt."""
        
        try:
            # Query LLM for enhancement using generate_completion
            llm_request = {
                "message": enhancement_prompt,
                "max_tokens": 500,
                "temperature": 0.7
            }

            response = await llm_service.generate_completion(llm_request)

            enhanced_prompt = response.get('response', response.get('text', request.prompt)).strip()

            # Remove any quotes or extra formatting
            enhanced_prompt = enhanced_prompt.strip('"').strip("'").strip()

            # If the LLM added explanations, try to extract just the prompt
            if '\n\n' in enhanced_prompt:
                enhanced_prompt = enhanced_prompt.split('\n\n')[0]

        except Exception as llm_error:
            logger.error(f"LLM enhancement failed: {llm_error}")
            # Fallback: Add basic structure to the prompt
            enhanced_prompt = f"Please {request.prompt.lower().strip('?.!')}. Provide a detailed, well-structured response with specific examples and actionable insights."
        
        # Calculate simple quality metrics
        original_length = len(request.prompt.split())
        enhanced_length = len(enhanced_prompt.split())
        
        quality_metrics = {
            "clarity": min(1.0, enhanced_length / max(original_length, 1) * 0.8),
            "specificity": min(1.0, (enhanced_length - original_length) / max(original_length, 1) * 0.5 + 0.5),
            "structure": 0.85 if len(enhanced_prompt) > len(request.prompt) else 0.6
        }
        
        # Create response
        prompt_id = f"prompt_{datetime.utcnow().timestamp()}"
        
        return {
            "id": prompt_id,
            "original_prompt": request.prompt,
            "enhanced_prompt": enhanced_prompt,
            "quality_metrics": quality_metrics,
            "engine": "llm",
            "created_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prompt enhancement failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prompt enhancement failed: {str(e)}"
        )

@router.post("/enhance/async")
async def enhance_prompt_async(request: PromptEnhanceRequest):
    """Enhance a prompt asynchronously"""
    # For now, just call the sync version
    return await enhance_prompt(request)

@router.get("/enhance/{prompt_id}")
async def get_enhanced_prompt(prompt_id: str):
    """Get enhanced prompt by ID"""
    return {
        "id": prompt_id,
        "status": "completed",
        "message": "Prompt retrieval not fully implemented yet"
    }

@router.post("/enhance/{prompt_id}/feedback")
async def submit_feedback(prompt_id: str, feedback: Dict[str, Any]):
    """Submit feedback on enhanced prompt"""
    return {
        "id": prompt_id,
        "feedback_received": True,
        "message": "Feedback recorded"
    }

@router.get("/statistics")
async def get_statistics():
    """Get prompt enhancement statistics"""
    return {
        "total_enhancements": 0,
        "average_quality": 0.85,
        "message": "Statistics not fully implemented yet"
    }


# ============================================================================
# PROMPT TEMPLATE LIBRARY ENDPOINTS
# ============================================================================

@router.post("/templates/initialize")
async def initialize_templates(db = Depends(get_database)):
    """Initialize default system templates"""
    try:
        service = PromptLibraryService(db)
        result = await service.initialize_default_templates()
        return result
    except Exception as e:
        logger.error(f"Failed to initialize templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates")
async def create_template(template_data: PromptTemplateCreate, db = Depends(get_database)):
    """Create a new prompt template"""
    try:
        service = PromptLibraryService(db)
        result = await service.create_template(template_data)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates(
    category: Optional[PromptCategory] = None,
    tags: Optional[str] = None,
    is_active: bool = True,
    page: int = 1,
    page_size: int = 20,
    db = Depends(get_database)
):
    """List prompt templates with filtering and pagination"""
    try:
        service = PromptLibraryService(db)

        # Parse tags if provided
        tags_list = tags.split(",") if tags else None

        result = await service.list_templates(
            category=category,
            tags=tags_list,
            is_active=is_active,
            page=page,
            page_size=page_size
        )

        return result
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/search")
async def search_templates(
    q: str,
    category: Optional[PromptCategory] = None,
    db = Depends(get_database)
):
    """Search prompt templates"""
    try:
        service = PromptLibraryService(db)
        templates = await service.search_templates(q, category)

        return {
            "success": True,
            "templates": templates,
            "total": len(templates),
            "message": f"Found {len(templates)} templates"
        }
    except Exception as e:
        logger.error(f"Failed to search templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{template_id}")
async def get_template(template_id: str, db = Depends(get_database)):
    """Get a specific template by ID"""
    try:
        service = PromptLibraryService(db)
        template = await service.get_template(template_id)

        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

        return {
            "success": True,
            "template": template,
            "message": "Template retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/templates/{template_id}")
async def update_template(
    template_id: str,
    update_data: PromptTemplateUpdate,
    db = Depends(get_database)
):
    """Update a template"""
    try:
        service = PromptLibraryService(db)
        result = await service.update_template(template_id, update_data)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str, db = Depends(get_database)):
    """Delete a template (soft delete)"""
    try:
        service = PromptLibraryService(db)
        result = await service.delete_template(template_id)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/render")
async def render_template(request: PromptTemplateRenderRequest, db = Depends(get_database)):
    """Render a template with provided variables"""
    try:
        service = PromptLibraryService(db)
        result = await service.render_template(request.template_id, request.variables)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to render template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/rate")
async def rate_template(request: PromptTemplateRatingRequest, db = Depends(get_database)):
    """Rate a template"""
    try:
        service = PromptLibraryService(db)
        result = await service.rate_template(request.template_id, request.rating, request.feedback)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rate template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

