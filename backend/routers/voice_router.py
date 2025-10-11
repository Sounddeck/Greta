"""
Voice Integration Router for CPAS4/Greta
REST API endpoints for Voice Service (Week 21-23)
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import logging
import io
from pydantic import BaseModel

# Import voice service
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Greta" / "backend"))

try:
    from services.voice_service import (
        voice_service,
        AudioData,
        AudioFormat,
        STTEngine,
        TTSEngine,
        Language,
        VoiceCommand,
        STTResult,
        TTSResult,
        VoiceProfile,
        VoiceCommandResult,
        VoiceStats,
    )
    VOICE_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Voice service not available: {e}")
    VOICE_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])

# Request/Response Models
class STTRequest(BaseModel):
    language: Optional[str] = None
    engine: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    language: Optional[str] = None
    engine: Optional[str] = None

class VoiceProfileCreate(BaseModel):
    profile_id: str
    user_id: str
    name: str
    preferred_language: str = "en"
    preferred_voice: str = "default"

class VoiceProfileUpdate(BaseModel):
    preferred_language: Optional[str] = None
    preferred_voice: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

class ConversationStart(BaseModel):
    conversation_id: str
    profile_id: Optional[str] = None

# ============================================================================
# Speech-to-Text Endpoints
# ============================================================================

@router.post("/stt", response_model=Dict[str, Any])
async def speech_to_text(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    engine: Optional[str] = Form(None)
):
    """
    Convert speech to text
    
    Args:
        audio: Audio file (WAV, MP3, OGG, FLAC, RAW)
        language: Language code (en, es, fr, de, zh, ja, auto)
        engine: STT engine (whisper, google, azure, mock)
    
    Returns:
        STTResult with transcription, confidence, language, duration
    """
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        # Read audio data
        audio_bytes = await audio.read()
        
        # Determine format from filename
        format_str = audio.filename.split('.')[-1].lower() if audio.filename else 'wav'
        try:
            audio_format = AudioFormat(format_str)
        except ValueError:
            audio_format = AudioFormat.WAV
        
        # Create AudioData
        audio_data = AudioData(
            data=audio_bytes,
            format=audio_format,
            duration=len(audio_bytes) / 16000.0,  # Rough estimate
        )
        
        # Convert to Language enum
        lang = Language(language) if language else None
        eng = STTEngine(engine) if engine else None
        
        # Perform STT
        result = await voice_service.speech_to_text(audio_data, language=lang, engine=eng)
        
        return {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language.value,
            "duration": result.duration,
            "engine": result.engine.value,
            "timestamp": result.timestamp,
        }
    
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(500, f"STT failed: {str(e)}")

# ============================================================================
# Text-to-Speech Endpoints
# ============================================================================

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech
    
    Args:
        text: Text to synthesize
        voice: Voice type (default, male, female, child)
        language: Language code (en, es, fr, de, zh, ja)
        engine: TTS engine (coqui, google, azure, mock)
    
    Returns:
        Audio file (WAV format)
    """
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        # Convert to Language enum
        lang = Language(request.language) if request.language else None
        eng = TTSEngine(request.engine) if request.engine else None
        
        # Perform TTS
        result = await voice_service.text_to_speech(
            request.text,
            voice=request.voice,
            language=lang,
            engine=eng
        )
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(result.audio_data.data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech.{result.audio_data.format.value}"
            }
        )
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(500, f"TTS failed: {str(e)}")

# ============================================================================
# Voice Activity Detection Endpoints
# ============================================================================

@router.post("/vad", response_model=Dict[str, Any])
async def voice_activity_detection(
    audio: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    """
    Detect voice activity in audio
    
    Args:
        audio: Audio file
        threshold: Detection threshold (0.0-1.0)
    
    Returns:
        has_voice: Boolean indicating if voice was detected
        segments: List of speech segments (start, end) in seconds
    """
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        # Read audio data
        audio_bytes = await audio.read()
        
        # Create AudioData
        audio_data = AudioData(
            data=audio_bytes,
            format=AudioFormat.WAV,
            duration=len(audio_bytes) / 16000.0,
        )
        
        # Detect voice activity
        has_voice = await voice_service.detect_voice_activity(audio_data, threshold)
        segments = await voice_service.detect_speech_segments(audio_data)
        
        return {
            "has_voice": has_voice,
            "segments": segments,
            "threshold": threshold,
        }
    
    except Exception as e:
        logger.error(f"VAD error: {e}")
        raise HTTPException(500, f"VAD failed: {str(e)}")

# ============================================================================
# Voice Command Endpoints
# ============================================================================

@router.post("/commands/detect-wake-word", response_model=Dict[str, bool])
async def detect_wake_word(text: str):
    """Detect wake word in text"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        has_wake_word = await voice_service.detect_wake_word(text)
        return {"has_wake_word": has_wake_word}
    except Exception as e:
        logger.error(f"Wake word detection error: {e}")
        raise HTTPException(500, f"Wake word detection failed: {str(e)}")

@router.post("/commands/parse", response_model=Dict[str, Any])
async def parse_voice_command(text: str):
    """Parse voice command from text"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        result = await voice_service.parse_voice_command(text)
        if not result:
            return {"command": None}
        
        return {
            "command": result.command.value,
            "parameters": result.parameters,
            "confidence": result.confidence,
            "text": result.text,
            "timestamp": result.timestamp,
        }
    except Exception as e:
        logger.error(f"Command parsing error: {e}")
        raise HTTPException(500, f"Command parsing failed: {str(e)}")

@router.post("/commands/execute", response_model=Dict[str, Any])
async def execute_voice_command(text: str):
    """Parse and execute voice command"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        # Parse command
        command_result = await voice_service.parse_voice_command(text)
        if not command_result:
            raise HTTPException(400, "No valid command found")
        
        # Execute command
        result = await voice_service.execute_voice_command(command_result)
        return result
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        raise HTTPException(500, f"Command execution failed: {str(e)}")

# ============================================================================
# Voice Profile Endpoints
# ============================================================================

@router.post("/profiles", response_model=Dict[str, Any], status_code=201)
async def create_voice_profile(profile: VoiceProfileCreate):
    """Create a voice profile"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        result = await voice_service.create_voice_profile(
            profile.profile_id,
            profile.user_id,
            profile.name,
            Language(profile.preferred_language),
            profile.preferred_voice
        )
        
        return {
            "profile_id": result.profile_id,
            "user_id": result.user_id,
            "name": result.name,
            "preferred_language": result.preferred_language.value,
            "preferred_voice": result.preferred_voice,
            "created_at": result.created_at,
        }
    except Exception as e:
        logger.error(f"Profile creation error: {e}")
        raise HTTPException(500, f"Profile creation failed: {str(e)}")

@router.get("/profiles/{profile_id}", response_model=Dict[str, Any])
async def get_voice_profile(profile_id: str):
    """Get voice profile by ID"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        result = await voice_service.get_voice_profile(profile_id)
        if not result:
            raise HTTPException(404, "Profile not found")
        
        return {
            "profile_id": result.profile_id,
            "user_id": result.user_id,
            "name": result.name,
            "preferred_language": result.preferred_language.value,
            "preferred_voice": result.preferred_voice,
            "settings": result.settings,
            "created_at": result.created_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(500, f"Profile retrieval failed: {str(e)}")

@router.get("/profiles", response_model=List[Dict[str, Any]])
async def list_voice_profiles():
    """List all voice profiles"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")
    
    try:
        profiles = await voice_service.list_voice_profiles()
        return [
            {
                "profile_id": p.profile_id,
                "user_id": p.user_id,
                "name": p.name,
                "preferred_language": p.preferred_language.value,
                "preferred_voice": p.preferred_voice,
                "created_at": p.created_at,
            }
            for p in profiles
        ]
    except Exception as e:
        logger.error(f"Profile listing error: {e}")
        raise HTTPException(500, f"Profile listing failed: {str(e)}")

@router.patch("/profiles/{profile_id}", response_model=Dict[str, Any])
async def update_voice_profile(profile_id: str, updates: VoiceProfileUpdate):
    """Update voice profile"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        update_dict = {}
        if updates.preferred_language:
            update_dict["preferred_language"] = updates.preferred_language
        if updates.preferred_voice:
            update_dict["preferred_voice"] = updates.preferred_voice
        if updates.settings:
            update_dict["settings"] = updates.settings

        result = await voice_service.update_voice_profile(profile_id, update_dict)
        if not result:
            raise HTTPException(404, "Profile not found")

        return {
            "profile_id": result.profile_id,
            "user_id": result.user_id,
            "name": result.name,
            "preferred_language": result.preferred_language.value,
            "preferred_voice": result.preferred_voice,
            "settings": result.settings,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(500, f"Profile update failed: {str(e)}")

@router.delete("/profiles/{profile_id}", status_code=204)
async def delete_voice_profile(profile_id: str):
    """Delete voice profile"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        result = await voice_service.delete_voice_profile(profile_id)
        if not result:
            raise HTTPException(404, "Profile not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile deletion error: {e}")
        raise HTTPException(500, f"Profile deletion failed: {str(e)}")

# ============================================================================
# Conversation Mode Endpoints
# ============================================================================

@router.post("/conversation/start", response_model=Dict[str, bool])
async def start_conversation(request: ConversationStart):
    """Start voice conversation mode"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        result = await voice_service.start_conversation(
            request.conversation_id,
            request.profile_id
        )
        return {"started": result}
    except Exception as e:
        logger.error(f"Conversation start error: {e}")
        raise HTTPException(500, f"Conversation start failed: {str(e)}")

@router.post("/conversation/stop", response_model=Dict[str, Any])
async def stop_conversation():
    """Stop voice conversation mode"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        context = await voice_service.stop_conversation()
        return context
    except Exception as e:
        logger.error(f"Conversation stop error: {e}")
        raise HTTPException(500, f"Conversation stop failed: {str(e)}")

@router.post("/conversation/process", response_model=Dict[str, Any])
async def process_voice_input(
    audio: UploadFile = File(...),
    profile_id: Optional[str] = Form(None)
):
    """Process voice input in conversation mode"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        # Read audio data
        audio_bytes = await audio.read()

        # Create AudioData
        audio_data = AudioData(
            data=audio_bytes,
            format=AudioFormat.WAV,
            duration=len(audio_bytes) / 16000.0,
        )

        # Process voice input
        result = await voice_service.process_voice_input(audio_data, profile_id)

        # Convert audio response to base64 if present
        if result.get("audio_response"):
            import base64
            audio_resp = result["audio_response"]
            result["audio_response"] = {
                "data": base64.b64encode(audio_resp.data).decode(),
                "format": audio_resp.format.value,
                "duration": audio_resp.duration,
            }

        return result
    except Exception as e:
        logger.error(f"Voice input processing error: {e}")
        raise HTTPException(500, f"Voice input processing failed: {str(e)}")

# ============================================================================
# Statistics & Configuration Endpoints
# ============================================================================

@router.get("/stats", response_model=Dict[str, Any])
async def get_voice_stats():
    """Get voice service statistics"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        stats = await voice_service.get_stats()
        return {
            "total_stt_requests": stats.total_stt_requests,
            "total_tts_requests": stats.total_tts_requests,
            "total_commands": stats.total_commands,
            "total_conversations": stats.total_conversations,
            "avg_stt_time": stats.avg_stt_time,
            "avg_tts_time": stats.avg_tts_time,
            "languages_used": stats.languages_used,
            "commands_by_type": stats.commands_by_type,
        }
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(500, f"Stats retrieval failed: {str(e)}")

@router.post("/stats/reset", status_code=204)
async def reset_voice_stats():
    """Reset voice service statistics"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        await voice_service.reset_stats()
    except Exception as e:
        logger.error(f"Stats reset error: {e}")
        raise HTTPException(500, f"Stats reset failed: {str(e)}")

@router.get("/languages", response_model=List[str])
async def get_supported_languages():
    """Get list of supported languages"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        return await voice_service.get_supported_languages()
    except Exception as e:
        logger.error(f"Language listing error: {e}")
        raise HTTPException(500, f"Language listing failed: {str(e)}")

@router.get("/voices", response_model=List[str])
async def get_supported_voices(engine: Optional[str] = None):
    """Get list of supported voices"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        eng = TTSEngine(engine) if engine else None
        return await voice_service.get_supported_voices(eng)
    except Exception as e:
        logger.error(f"Voice listing error: {e}")
        raise HTTPException(500, f"Voice listing failed: {str(e)}")

@router.get("/formats", response_model=List[str])
async def get_audio_formats():
    """Get list of supported audio formats"""
    if not VOICE_SERVICE_AVAILABLE:
        raise HTTPException(503, "Voice service not available")

    try:
        return await voice_service.get_audio_formats()
    except Exception as e:
        logger.error(f"Format listing error: {e}")
        raise HTTPException(500, f"Format listing failed: {str(e)}")

