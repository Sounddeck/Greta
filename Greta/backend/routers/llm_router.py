"""
LLM Router - OpenAI-Compatible API
Week 1 Implementation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from loguru import logger

from backend.services.llm_service import llm_service


router = APIRouter(prefix="/v1", tags=["LLM"])


# Request/Response Models (OpenAI-compatible)

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: int = Field(2000, ge=1, le=32000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(40, ge=1, le=100, description="Top-k sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming (not yet implemented)")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model name to use")
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(2000, ge=1, le=32000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=100)
    stop: Optional[List[str]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "cpas4"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Helper Functions

def format_messages_as_prompt(messages: List[ChatMessage]) -> str:
    """
    Convert chat messages to a single prompt string
    
    Args:
        messages: List of chat messages
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    
    # Add final assistant prompt
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)


# API Endpoints

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint
    
    This endpoint is compatible with AutoGen and other OpenAI-compatible clients.
    """
    try:
        logger.info(f"Chat completion request for model: {request.model}")
        
        # Format messages as prompt
        prompt = format_messages_as_prompt(request.messages)
        
        # Generate response
        response_text = await llm_service.generate(
            prompt=prompt,
            model_name=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop or []
        )
        
        # Format as OpenAI-compatible response
        import time
        import uuid
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text.strip()
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()),  # Rough estimate
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        )
        
    except ValueError as e:
        logger.error(f"Model not loaded: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions")
async def completion(request: CompletionRequest):
    """
    OpenAI-compatible completion endpoint
    """
    try:
        logger.info(f"Completion request for model: {request.model}")
        
        # Generate response
        response_text = await llm_service.generate(
            prompt=request.prompt,
            model_name=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop or []
        )
        
        import time
        import uuid
        
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(request.prompt.split()) + len(response_text.split())
            }
        }
        
    except ValueError as e:
        logger.error(f"Model not loaded: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """
    List available models (OpenAI-compatible)
    """
    try:
        models = llm_service.list_models()
        
        model_list = [
            ModelInfo(
                id=model,
                object="model",
                created=0,
                owned_by="cpas4"
            )
            for model in models
        ]
        
        return ModelListResponse(
            object="list",
            data=model_list
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}")
async def get_model(model_name: str):
    """
    Get model information
    """
    try:
        model_info = llm_service.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Management Endpoints (CPAS4-specific)

@router.post("/models/{model_name}/load")
async def load_model(model_name: str, n_ctx: int = 4096, n_gpu_layers: int = 35):
    """
    Load a model into memory
    """
    try:
        success = await llm_service.load_model(
            model_name=model_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
        
        if success:
            return {"status": "loaded", "model": model_name}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """
    Unload a model from memory
    """
    try:
        success = await llm_service.unload_model(model_name)
        
        if success:
            return {"status": "unloaded", "model": model_name}
        else:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not loaded")
            
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/loaded")
async def list_loaded_models():
    """
    List currently loaded models
    """
    try:
        loaded = llm_service.list_loaded_models()
        return {"loaded_models": loaded}
        
    except Exception as e:
        logger.error(f"Failed to list loaded models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
