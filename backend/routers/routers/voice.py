from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from services.piper_voice_service import voice_service, voice_queue
import asyncio

router = APIRouter()

class VoiceSynthesisRequest(BaseModel):
    text: str
    voice_model: str = "de_DE-karl-medium"
    speed: float = 1.0
    caching: bool = True

class VoiceAnnounceRequest(BaseModel):
    message: str
    priority: str = "normal"
    queue: bool = False

@router.post("/voice/synthesize")
async def synthesize_text(request: VoiceSynthesisRequest):
    """Core text-to-speech synthesis endpoint"""
    try:
        audio_data = await voice_service.generate_audio(
            request.text,
            request.voice_model, 
            request.speed
        )
        return Response(audio_data, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(500, f"Voice synthesis failed: {e}")

@router.post("/voice/announce")
async def announce_message(
    request: VoiceAnnounceRequest, 
    background_tasks: BackgroundTasks
):
    """Priority-based voice announcements"""
    try:
        if request.priority == "urgent":
            audio_data = await voice_service.generate_audio(
                request.message, 
                "de_DE-karl-medium",
                1.0
            )
            return Response(audio_data, 
                          media_type="audio/wav", 
                          headers={"X-Urgent": "true"})
        else:
            # Handle background processing here instead of using the queue initially
            background_tasks.add_task(
                voice_service.generate_audio,
                request.message,
                "de_DE-karl-medium",
                1.0
            )
            return {"status": "queued", "message": request.message}
    except Exception as e:
        raise HTTPException(500, f"Voice announcement failed: {e}")

@router.get("/voice/models")
async def list_voice_models():
    """Get available voice models"""
    try:
        models = voice_service.list_available_models()
        model_data = []
        
        for model_name in models:
            info = voice_service.get_model_info(model_name)
            if info:
                model_data.append({
                    "name": model_name,
                    **info
                })
        
        return {"models": model_data}
    except Exception as e:
        raise HTTPException(500, f"Failed to list models: {e}")

@router.get("/voice/stats")
async def voice_statistics():
    """Get voice service performance statistics"""
    try:
        stats = voice_service.get_performance_stats()
        return {
            "service": "Piper TTS Voice Service",
            "status": "active",
            **stats
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get stats: {e}")

@router.post("/voice/preload")
async def preload_models(models: List[str]):
    """Preload specified voice models"""
    try:
        voice_service.preload_models(models)
        return {
            "status": "preload_started", 
            "models": models,
            "message": f"Started preloading {len(models)} voice models"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to preload models: {e}")

@router.get("/voice/status")
async def voice_system_status():
    """Enhanced system status with comprehensive information"""
    try:
        stats = voice_service.get_performance_stats()
        queue_status = getattr(voice_queue, '_get_queue_status', lambda: {})()
        
        return {
            "service": "GRETA PAI Piper Voice System",
            "version": "1.0.0",
            "engine": "Piper TTS (Open Source)",
            "status": "operational",
            "performance": stats,
            "queue": queue_status,
            "configuration": {
                "supported_formats": ["wav"],
                "default_voice": "de_DE-karl-medium",
                "german_support": True,
                "thread_pool_size": 2
            },
            "capabilities": [
                "Real-time synthesis",
                "German accent voices",
                "Background queue processing",
                " Automatic model downloading",
                "Performance monitoring"
            ]
        }
    except Exception as e:
        return {
            "service": "GRETA PAI Piper Voice System",
            "status": "error",
            "error": str(e)
        }

# Startup tasks for voice service
@router.on_event("startup")
async def startup_voice_service():
    """Initialize voice service on startup"""
    try:
        await voice_queue.start()
        # Preload default German voice model
        voice_service.preload_models(["de_DE-karl-medium", "en_US-lessac-medium"])
    except Exception as e:
        print(f"Voice service startup failed: {e}")

@router.on_event("shutdown")
async def shutdown_voice_service():
    """Cleanup voice service on shutdown"""
    try:
        await voice_queue.stop()
        await voice_service.close()
    except Exception as e:
        print(f"Voice service shutdown failed: {e}")
