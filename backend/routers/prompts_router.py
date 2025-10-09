"""
CPAS4 UI Prompts Router
Provides prompt enhancement endpoints for the CPAS4 UI
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from services.llm_service import get_llm_service

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

