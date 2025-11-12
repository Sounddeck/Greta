"""
Prompts Router

API endpoints for prompt enhancement, orchestration, and Chain-of-Thought integration.
"""

from fastapi import APIRouter, HTTPException, status, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

from services.cot_prompt_enhancer import CoTPromptEnhancer, TaskType
from services.prompt_orchestrator import PromptOrchestrator
from services.prompt_library_service import PromptLibraryService
from models.prompt_template_models import (
    PromptTemplate,
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplateResponse,
    PromptTemplateListResponse,
    PromptTemplateRenderRequest,
    PromptCategory
)
from database import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["Prompts"])

# Initialize services
cot_enhancer = CoTPromptEnhancer()
prompt_orchestrator = PromptOrchestrator()

# Dependency for prompt library service
async def get_prompt_library_service():
    """Get prompt library service instance"""
    db = await get_database()
    return PromptLibraryService(db)


# Request/Response Models

class CoTEnhanceRequest(BaseModel):
    """Request model for CoT enhancement"""
    prompt: str = Field(..., description="The prompt to enhance with Chain-of-Thought reasoning")
    task_type: Optional[str] = Field(None, description="Explicit task type (auto-detected if not provided)")
    custom_template: Optional[str] = Field(None, description="Custom CoT template to use")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "How can I optimize the performance of my Python application?",
                "task_type": "problem_solving"
            }
        }


class CoTEnhanceResponse(BaseModel):
    """Response model for CoT enhancement"""
    success: bool
    enhanced_prompt: str
    original_prompt: str
    task_type: str
    cot_template_used: str
    enhancement_applied: bool
    metadata: Dict[str, Any]
    error: Optional[str] = None


class PromptGenerateRequest(BaseModel):
    """Request model for full prompt generation with orchestration"""
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context information")
    task_type: str = Field("general", description="Type of task being performed")
    enable_cot: bool = Field(True, description="Whether to apply CoT enhancement")

    class Config:
        json_schema_extra = {
            "example": {
                "user_context": {
                    "user_id": "user123",
                    "tags": ["technical", "python"]
                },
                "task_type": "coding",
                "enable_cot": True
            }
        }


class PromptGenerateResponse(BaseModel):
    """Response model for prompt generation"""
    prompt: str
    metadata: Dict[str, Any]
    components: Dict[str, Any]


class TaskTypesResponse(BaseModel):
    """Response model for available task types"""
    task_types: List[str]
    count: int


# API Endpoints

@router.post("/enhance", response_model=CoTEnhanceResponse)
async def enhance_prompt(request: CoTEnhanceRequest):
    """
    Simple prompt enhancement without Chain-of-Thought

    This endpoint enhances prompts without adding CoT templates,
    useful for users who want basic prompt improvement.
    """
    try:
        # Return the prompt as-is with minimal enhancement
        # In a real implementation, this could apply grammar fixes, clarity improvements, etc.
        result = {
            "success": True,
            "enhanced_prompt": request.prompt,  # Return original prompt
            "original_prompt": request.prompt,
            "task_type": request.task_type or "general",
            "cot_template_used": "none",
            "enhancement_applied": False,
            "metadata": {
                "original_length": len(request.prompt),
                "enhanced_length": len(request.prompt),
                "enhancement_type": "simple"
            }
        }
        return CoTEnhanceResponse(**result)
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/enhance-cot", response_model=CoTEnhanceResponse)
async def enhance_prompt_with_cot(request: CoTEnhanceRequest):
    """
    Enhance a prompt with Chain-of-Thought reasoning patterns
    
    This endpoint adds structured step-by-step reasoning to prompts,
    improving LLM performance on complex tasks.
    
    **Task Types:**
    - reasoning: Logical reasoning and deduction
    - analysis: Analytical and evaluative tasks
    - problem_solving: Problem-solving and troubleshooting
    - creative: Creative and ideation tasks
    - research: Research and investigation
    - coding: Programming and development
    - planning: Planning and strategy
    - decision_making: Decision-making tasks
    - general: General-purpose tasks
    
    **Example:**
    ```json
    {
        "prompt": "How can I improve database query performance?",
        "task_type": "problem_solving"
    }
    ```
    """
    try:
        # Convert task_type string to TaskType enum if provided
        task_type_enum = None
        if request.task_type:
            try:
                task_type_enum = TaskType(request.task_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task_type. Must be one of: {', '.join([t.value for t in TaskType])}"
                )
        
        # Enhance the prompt
        result = cot_enhancer.enhance_with_cot(
            prompt=request.prompt,
            task_type=task_type_enum,
            custom_cot_template=request.custom_template
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to enhance prompt")
            )
        
        return CoTEnhanceResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing prompt with CoT: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/generate", response_model=PromptGenerateResponse)
async def generate_pai_prompt(request: PromptGenerateRequest):
    """
    Generate a complete PAI prompt with orchestration and CoT enhancement
    
    This endpoint combines:
    - Master prompt selection
    - Context prompt selection
    - Dynamic adaptations
    - Chain-of-Thought enhancement (optional)
    
    **Example:**
    ```json
    {
        "user_context": {
            "user_id": "user123",
            "tags": ["technical", "python"]
        },
        "task_type": "coding",
        "enable_cot": true
    }
    ```
    """
    try:
        result = await prompt_orchestrator.generate_pai_prompt(
            user_context=request.user_context,
            task_type=request.task_type,
            enable_cot=request.enable_cot
        )
        
        return PromptGenerateResponse(**result)
        
    except Exception as e:
        logger.error(f"Error generating PAI prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/task-types", response_model=TaskTypesResponse)
async def get_available_task_types():
    """
    Get list of available task types for CoT enhancement
    
    Returns all supported task types that can be used with the
    enhance-cot endpoint.
    """
    try:
        task_types = cot_enhancer.get_available_task_types()
        return TaskTypesResponse(
            task_types=task_types,
            count=len(task_types)
        )
    except Exception as e:
        logger.error(f"Error getting task types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/cot-template/{task_type}")
async def get_cot_template(task_type: str):
    """
    Get the CoT template for a specific task type
    
    **Parameters:**
    - task_type: One of the available task types
    
    **Example:**
    ```
    GET /api/v1/prompts/cot-template/problem_solving
    ```
    """
    try:
        # Convert to TaskType enum
        try:
            task_type_enum = TaskType(task_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid task_type. Must be one of: {', '.join([t.value for t in TaskType])}"
            )
        
        template = cot_enhancer.get_cot_template(task_type_enum)
        
        return {
            "task_type": task_type,
            "template": template,
            "length": len(template)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting CoT template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/cot-stats")
async def get_cot_stats():
    """
    Get statistics about CoT enhancement usage
    
    Returns information about CoT configuration and usage.
    """
    try:
        stats = prompt_orchestrator.get_cot_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting CoT stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/cot-enable")
async def enable_cot(enabled: bool = True):
    """
    Enable or disable CoT enhancement globally
    
    **Parameters:**
    - enabled: True to enable, False to disable
    
    **Example:**
    ```
    POST /api/v1/prompts/cot-enable?enabled=true
    ```
    """
    try:
        prompt_orchestrator.enable_cot_enhancement(enabled)
        return {
            "success": True,
            "cot_enabled": enabled,
            "message": f"CoT enhancement {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        logger.error(f"Error toggling CoT: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for prompts service"""
    from datetime import datetime
    return {
        "status": "healthy",
        "service": "prompts",
        "cot_enabled": prompt_orchestrator.cot_enabled,
        "available_task_types": len(cot_enhancer.get_available_task_types())
    }


# ============================================================================
# PROMPT TEMPLATE LIBRARY ENDPOINTS
# ============================================================================

@router.post("/templates/initialize")
async def initialize_templates(
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Initialize prompt template library with default system templates

    This endpoint seeds the database with pre-built prompt templates
    for common use cases. Safe to call multiple times - won't duplicate templates.
    """
    try:
        result = await service.initialize_default_templates()
        return result
    except Exception as e:
        logger.error(f"Error initializing templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize templates: {str(e)}"
        )


@router.get("/templates", response_model=PromptTemplateListResponse)
async def list_templates(
    category: Optional[PromptCategory] = Query(None, description="Filter by category"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    List all prompt templates with optional filtering and pagination

    **Parameters:**
    - category: Filter templates by category (optional)
    - page: Page number (default: 1)
    - page_size: Number of items per page (default: 20, max: 100)
    """
    try:
        result = await service.list_templates(
            category=category,
            page=page,
            page_size=page_size
        )
        return result
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list templates: {str(e)}"
        )


@router.get("/templates/search", response_model=PromptTemplateListResponse)
async def search_templates(
    q: str = Query(..., min_length=1, description="Search query"),
    category: Optional[PromptCategory] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Search prompt templates by keywords

    Searches in template name, description, and tags.

    **Parameters:**
    - q: Search query string
    - category: Filter by category (optional)
    - limit: Maximum number of results (default: 20, max: 100)
    """
    try:
        result = await service.search_templates(
            query=q,
            category=category,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Error searching templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search templates: {str(e)}"
        )


@router.get("/templates/{template_id}", response_model=PromptTemplateResponse)
async def get_template(
    template_id: str,
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Get a specific prompt template by ID

    **Parameters:**
    - template_id: Unique template identifier
    """
    try:
        result = await service.get_template(template_id)
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("message", "Template not found")
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}"
        )


@router.post("/templates", response_model=PromptTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    template_data: PromptTemplateCreate,
    user_id: Optional[str] = Query(None, description="User ID"),
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Create a new prompt template

    **Request Body:**
    - name: Template name
    - description: Template description
    - category: Template category
    - template: Template text with {variable} placeholders
    - variables: List of variable names
    - tags: Search tags
    - metadata: Additional metadata (optional)
    """
    try:
        result = await service.create_template(template_data, user_id)
        return result
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )


@router.put("/templates/{template_id}", response_model=PromptTemplateResponse)
async def update_template(
    template_id: str,
    template_data: PromptTemplateUpdate,
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Update an existing prompt template

    **Parameters:**
    - template_id: Template ID to update

    **Request Body:**
    - Any fields from PromptTemplateUpdate (all optional)
    """
    try:
        result = await service.update_template(template_id, template_data)
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("message", "Template not found")
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update template: {str(e)}"
        )


@router.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Delete a prompt template

    **Parameters:**
    - template_id: Template ID to delete

    **Note:** System templates cannot be deleted
    """
    try:
        result = await service.delete_template(template_id)
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("message", "Template not found")
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete template: {str(e)}"
        )


@router.post("/templates/render")
async def render_template(
    request: PromptTemplateRenderRequest,
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Render a template with variable substitution

    **Request Body:**
    - template_id: Template ID to render
    - variables: Dictionary of variable name -> value mappings

    **Example:**
    ```json
    {
        "template_id": "template_123",
        "variables": {
            "paper_title": "Deep Learning for NLP",
            "focus_area_1": "Model Architecture",
            "focus_area_2": "Training Techniques"
        }
    }
    ```
    """
    try:
        result = await service.render_template(
            request.template_id,
            request.variables
        )
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("message", "Template not found")
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render template: {str(e)}"
        )


@router.post("/templates/rate")
async def rate_template(
    template_id: str = Query(..., description="Template ID"),
    rating: int = Query(..., ge=1, le=5, description="Rating (1-5)"),
    feedback: Optional[str] = Query(None, description="Optional feedback"),
    service: PromptLibraryService = Depends(get_prompt_library_service)
):
    """
    Rate a prompt template

    **Parameters:**
    - template_id: Template ID to rate
    - rating: Rating from 1 to 5 stars
    - feedback: Optional feedback text
    """
    try:
        result = await service.rate_template(template_id, rating, feedback)
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("message", "Template not found")
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rate template: {str(e)}"
        )

