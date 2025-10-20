"""
Prompts Router

API endpoints for prompt enhancement, orchestration, and Chain-of-Thought integration.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

from services.cot_prompt_enhancer import CoTPromptEnhancer, TaskType
from services.prompt_orchestrator import PromptOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/prompts", tags=["Prompts"])

# Initialize services
cot_enhancer = CoTPromptEnhancer()
prompt_orchestrator = PromptOrchestrator()


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
    return {
        "status": "healthy",
        "service": "prompts",
        "cot_enabled": prompt_orchestrator.cot_enabled,
        "available_task_types": len(cot_enhancer.get_available_task_types())
    }

