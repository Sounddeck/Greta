"""
Voice Processing Pydantic Models for GRETA PAI
Text-to-Speech, Speech-to-Text, and voice configuration schemas
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from utils.validators import validate_url, url_validator


class VoiceProvider(str, Enum):
    """Voice service provider enumeration"""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"
    GOOGLE = "google"
    AWS = "aws"
    LOCAL = "local"


class VoiceGender(str, Enum):
    """Voice gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class AudioFormat(str, Enum):
    """Audio format enumeration"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"


class VoiceQuality(str, Enum):
    """Voice quality enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


class TTSRequest(BaseModel):
    """Text-to-Speech request"""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to convert to speech"
    )
    voice_id: Optional[str] = Field(
        None,
        description="Specific voice identifier"
    )
    provider: VoiceProvider = Field(
        default=VoiceProvider.OPENAI,
        description="Voice service provider"
    )
    language: str = Field(
        default="en-US",
        description="Language code (BCP 47 format)"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speech speed multiplier (0.25-4.0)"
    )
    pitch: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Voice pitch multiplier (0.5-2.0)"
    )
    volume: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Audio volume multiplier (0.0-2.0)"
    )
    format: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Output audio format"
    )
    quality: VoiceQuality = Field(
        default=VoiceQuality.MEDIUM,
        description="Audio quality level"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text content"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code format"""
        if not v or len(v) < 2:
            raise ValueError("Language code must be at least 2 characters")
        return v.lower()


class STTRequest(BaseModel):
    """Speech-to-Text request"""
    
    audio_url: Optional[str] = Field(
        None,
        description="URL to audio file"
    )
    audio_data: Optional[str] = Field(
        None,
        description="Base64 encoded audio data"
    )
    provider: VoiceProvider = Field(
        default=VoiceProvider.OPENAI,
        description="Speech recognition provider"
    )
    language: Optional[str] = Field(
        None,
        description="Expected language (auto-detect if None)"
    )
    format: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Audio format"
    )
    enable_punctuation: bool = Field(
        default=True,
        description="Whether to add punctuation"
    )
    enable_timestamps: bool = Field(
        default=False,
        description="Whether to include word timestamps"
    )
    
    # Use URL validator from utils.validators
    _validate_audio_url = validator('audio_url', allow_reuse=True)(url_validator)
    
    @validator('audio_data')
    def validate_audio_data(cls, v, values):
        """Validate audio data"""
        if not values.get('audio_url') and not v:
            raise ValueError("Either audio_url or audio_data must be provided")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code format"""
        if v and len(v) < 2:
            raise ValueError("Language code must be at least 2 characters")
        return v.lower() if v else None


class VoiceConfig(BaseModel):
    """Voice configuration settings"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Voice configuration name"
    )
    provider: VoiceProvider = Field(
        ...,
        description="Voice service provider"
    )
    voice_id: str = Field(
        ...,
        description="Provider-specific voice identifier"
    )
    language: str = Field(
        ...,
        description="Voice language"
    )
    gender: VoiceGender = Field(
        ...,
        description="Voice gender"
    )
    age_range: Optional[str] = Field(
        None,
        description="Voice age range (e.g., 'young', 'middle-aged', 'elderly')"
    )
    accent: Optional[str] = Field(
        None,
        description="Voice accent (e.g., 'american', 'british', 'australian')"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Voice description"
    )
    is_premium: bool = Field(
        default=False,
        description="Whether this is a premium voice"
    )
    is_active: bool = Field(
        default=True,
        description="Whether this voice is available"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate voice name"""
        if not v or not v.strip():
            raise ValueError("Voice name cannot be empty")
        return v.strip()


class VoiceResponse(BaseModel):
    """Voice processing response"""
    
    success: bool = Field(
        ...,
        description="Whether the request was successful"
    )
    audio_url: Optional[str] = Field(
        None,
        description="URL to generated audio file"
    )
    audio_data: Optional[str] = Field(
        None,
        description="Base64 encoded audio data"
    )
    text: Optional[str] = Field(
        None,
        description="Transcribed text (for STT)"
    )
    duration: Optional[float] = Field(
        None,
        description="Audio duration in seconds"
    )
    format: Optional[AudioFormat] = Field(
        None,
        description="Audio format"
    )
    provider: Optional[VoiceProvider] = Field(
        None,
        description="Provider used"
    )
    voice_id: Optional[str] = Field(
        None,
        description="Voice ID used"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Processing time in seconds"
    )
    cost: Optional[float] = Field(
        None,
        description="Processing cost in USD"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


class VoiceListResponse(BaseModel):
    """Voice list response"""
    
    voices: List[VoiceConfig] = Field(
        ...,
        description="Available voices"
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of voices"
    )
    providers: List[VoiceProvider] = Field(
        ...,
        description="Available providers"
    )
    languages: List[str] = Field(
        ...,
        description="Available languages"
    )


class VoiceUsageStats(BaseModel):
    """Voice usage statistics"""
    
    total_requests: int = Field(
        default=0,
        ge=0,
        description="Total number of requests"
    )
    total_characters: int = Field(
        default=0,
        ge=0,
        description="Total characters processed"
    )
    total_duration: float = Field(
        default=0.0,
        ge=0.0,
        description="Total audio duration in seconds"
    )
    total_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total cost in USD"
    )
    provider_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Usage breakdown by provider"
    )
    language_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Usage breakdown by language"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }


# Export all voice models
__all__ = [
    "VoiceProvider",
    "VoiceGender",
    "AudioFormat",
    "VoiceQuality",
    "TTSRequest",
    "STTRequest",
    "VoiceConfig",
    "VoiceResponse",
    "VoiceListResponse",
    "VoiceUsageStats"
]
