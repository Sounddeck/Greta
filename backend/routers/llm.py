"""LLM Router for GRETA PAI System
Large Language Model processing and completion endpoints with enhanced AI integration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import asyncio
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])

class CompletionRequest(BaseModel):
    """Request model for text completion"""
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: Optional[int] = Field(100, ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    model: str = Field("mock-model", description="Model to use - check /models endpoint", examples=["mock-model", "llama-7b", "gpt-3.5-turbo"])
    system_message: Optional[str] = Field(None, description="System message for context")

class CompletionResponse(BaseModel):
    """Response model for completion"""
    completion: str
    model: str
    provider: str
    tokens_used: int
    finish_reason: str
    processing_time: float
    timestamp: datetime

try:
    from backend.services.llm_integration import llm_integration_service, CompletionRequest as IntegrationRequest, ModelConfig, ModelProvider
    _integration_available = True
    logger.info("LLM integration service loaded successfully")
except ImportError as e:
    logger.warning(f"LLM integration service not available: {e}")
    _integration_available = False

@router.post("/completion", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    """Generate text completion using configured LLM"""
    try:
        start_time = asyncio.get_event_loop().time()

        if not _integration_available:
            return CompletionResponse(
                completion=f"[PLACEHOLDER] LLM processing not available - would process: '{request.prompt[:100]}...' with model {request.model}",
                model=request.model,
                provider="unavailable",
                tokens_used=len(request.prompt.split()),
                finish_reason="service_unavailable",
                processing_time=round(asyncio.get_event_loop().time() - start_time, 2),
                timestamp=datetime.now()
            )

        # Get model configuration
        model_config = llm_integration_service.get_model_config(request.model)
        if not model_config:
            available_models = [m["name"] for m in llm_integration_service.list_available_models()]
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found. Available: {available_models}"
            )

        # Create integration request
        integration_request = IntegrationRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            model_config=model_config,
            system_message=request.system_message
        )

        # Generate completion
        response = await llm_integration_service.generate_completion(integration_request)

        processing_time = asyncio.get_event_loop().time() - start_time

        return CompletionResponse(
            completion=response.text,
            model=response.model_used,
            provider=response.provider,
            tokens_used=response.tokens_used,
            finish_reason=response.finish_reason,
            processing_time=response.processing_time,
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"LLM completion error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

@router.get("/models")
async def list_available_models():
    """List available LLM models with their configurations"""
    try:
        if not _integration_available:
            return {"models": [
                {"name": "mock-model", "provider": "local_mock", "model_name": "mock-model", "max_tokens": 2048, "context_window": 4096, "temperature": 0.7}
            ], "message": "Integration service not available - using mock"}

        return {"models": llm_integration_service.list_available_models()}

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model list")

@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        if not _integration_available:
            if model_name == "mock-model":
                return {
                    "name": "mock-model",
                    "provider": "local_mock",
                    "model_name": "mock-model",
                    "max_tokens": 2048,
                    "context_window": 4096,
                    "temperature": 0.7,
                    "capabilities": ["text_generation", "analysis", "code_generation"]
                }
            else:
                raise HTTPException(status_code=404, detail="Model not found and integration unavailable")

        models = llm_integration_service.list_available_models()
        for model in models:
            if model["name"] == model_name:
                return model

        raise HTTPException(status_code=404, detail="Model not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.get("/status")
async def llm_service_status():
    """Get LLM service status and capabilities"""
    try:
        status_info = {
            "integration_available": _integration_available,
            "service_status": "available" if _integration_available else "mock_mode"
        }

        if _integration_available:
            health = await llm_integration_service.health_check()
            status_info.update({
                "health_status": health,
                "model_count": len(llm_integration_service.models),
                "providers_loaded": list(health.get("providers", {}).keys())
            })

        return status_info

    except Exception as e:
        logger.error(f"Error checking LLM status: {e}")
        raise HTTPException(status_code=500, detail="Failed to check LLM status")

@router.post("/analyze/{model_name}")
async def analyze_text(model_name: str, request: CompletionRequest):
    """Analyze text using specific LLM model with specialized prompting"""
    try:
        if not _integration_available:
            return {
                "analysis": f"[MOCK ANALYSIS] Would analyze: {request.prompt[:100]}... using model {model_name}",
                "model_used": model_name,
                "analysis_type": "text_analysis",
                "confidence": 0.0
            }

        # Add analytical system prompt
        analysis_prompt = f"Please analyze the following text comprehensively. Provide insights, patterns, sentiment, key themes, and actionable recommendations:\n\n{request.prompt}"

        enhanced_request = CompletionRequest(
            prompt=analysis_prompt,
            max_tokens=max(request.max_tokens, 500),  # Ensure adequate space for analysis
            temperature=min(request.temperature or 0.7, 0.8),  # Prefer deterministic analysis
            model=model_name,
            system_message="You are an expert analytical AI. Provide structured, insightful analysis with clear reasoning and evidence-based conclusions."
        )

        response = await generate_completion(enhanced_request)

        return {
            "analysis": response.completion,
            "model_used": response.model,
            "analysis_type": "comprehensive_text_analysis",
            "processing_time": response.processing_time,
            "confidence": 0.85 if response.model != "mock-model" else 0.0
        }

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Text analysis failed")

@router.post("/generate/{model_name}/code")
async def generate_code(model_name: str, request: Dict[str, Any]):
    """Generate code using LLM with programming-specific prompting"""
    try:
        code_prompt = request.get("prompt", "")
        language = request.get("language", "python")
        context = request.get("context", "")

        if not code_prompt:
            raise HTTPException(status_code=400, detail="Code generation prompt required")

        full_prompt = f"""Generate {language} code based on this requirement. Include proper error handling, comments, and best practices:

Context: {context}
Requirement: {code_prompt}

Generate clean, efficient, and well-documented {language} code:"""

        enhanced_request = CompletionRequest(
            prompt=full_prompt,
            max_tokens=1000,  # Code can be longer
            temperature=0.3 if language.lower() in ["python", "javascript", "sql"] else 0.7,  # More deterministic for well-structured languages
            model=model_name,
            system_message=f"You are an expert {language} developer. Generate clean, efficient, well-documented code with proper error handling and best practices."
        )

        response = await generate_completion(enhanced_request)

        return {
            "code": response.completion,
            "language": language,
            "model_used": response.model,
            "processing_time": response.processing_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail="Code generation failed")

@router.post("/chat/{model_name}")
async def interactive_chat(model_name: str, messages: List[Dict[str, str]]):
    """Interactive chat using LLM with conversation context"""
    try:
        if not messages:
            raise HTTPException(status_code=400, detail="Messages required for chat")

        # Build conversation context
        conversation_context = ""
        for msg in messages[-10:]:  # Limit to last 10 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_context += f"{role.title()}: {content}\n"

        full_prompt = f"""Continue this conversation naturally. Maintain context and provide helpful, engaging responses.

{conversation_context}
Assistant: """

        request = CompletionRequest(
            prompt=full_prompt,
            max_tokens=300,  # Interactive responses
            temperature=0.8,  # More creative/conversational
            model=model_name,
            system_message="You are a helpful, friendly AI assistant engaged in natural conversation. Respond naturally and contextually."
        )

        response = await generate_completion(request)

        # Extract just the assistant's response (after "Assistant: ")
        assistant_response = response.completion
        if "Assistant:" in assistant_response:
            assistant_response = assistant_response.split("Assistant:")[-1].strip()

        return {
            "response": assistant_response,
            "model_used": response.model,
            "processing_time": response.processing_time,
            "conversation_id": str(uuid.uuid4())[:8]  # Simple session ID
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")

@router.get("/health")
async def llm_health():
    """LLM service health check"""
    try:
        health_info = {
            "status": "healthy" if _integration_available else "degraded",
            "component": "llm_router",
            "integration_available": _integration_available,
            "capabilities": [
                "text_completion",
                "text_analysis",
                "code_generation",
                "interactive_chat",
                "model_management"
            ]
        }

        if _integration_available:
            health_info["service_health"] = await llm_integration_service.health_check()

        return health_info

    except Exception as e:
        logger.error(f"LLM health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
