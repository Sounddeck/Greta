"""
Voice Service Configuration with Caching
Provides cached configuration for voice service endpoints
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import timedelta

from utils.cache import cache, cache_manager
from config import get_settings

logger = logging.getLogger(__name__)


class VoiceConfigService:
    """Service for managing voice service configuration with caching"""
    
    def __init__(self):
        self.settings = get_settings()
    
    @cache(ttl=timedelta(minutes=10), key_prefix="voice_config_languages")
    async def get_supported_languages(self) -> List[str]:
        """
        Get supported languages for voice services
        Cached for 10 minutes
        """
        # Default supported languages
        languages = [
            "en-US", "en-GB", "es-ES", "fr-FR", "de-DE", 
            "it-IT", "pt-BR", "ru-RU", "ja-JP", "ko-KR",
            "zh-CN", "zh-TW", "ar-SA", "hi-IN", "nl-NL"
        ]
        
        logger.debug(f"Retrieved {len(languages)} supported languages")
        return languages
    
    @cache(ttl=timedelta(minutes=10), key_prefix="voice_config_voices")
    async def get_supported_voices(self, engine: Optional[str] = None) -> List[str]:
        """
        Get supported voices for TTS engines
        Cached for 10 minutes
        """
        voices = {
            "coqui": [
                "jenny", "ljspeech", "vctk", "mailabs",
                "thorsten", "sam", "blizzard2013"
            ],
            "google": [
                "en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-C",
                "en-GB-Wavenet-A", "es-ES-Wavenet-A", "fr-FR-Wavenet-A"
            ],
            "azure": [
                "en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural",
                "en-GB-SoniaNeural", "es-ES-ElviraNeural", "fr-FR-DeniseNeural"
            ],
            "mock": ["mock-voice-1", "mock-voice-2"]
        }
        
        if engine and engine in voices:
            result = voices[engine]
        else:
            # Return all voices if no engine specified
            result = []
            for engine_voices in voices.values():
                result.extend(engine_voices)
        
        logger.debug(f"Retrieved {len(result)} voices for engine: {engine or 'all'}")
        return result
    
    @cache(ttl=timedelta(minutes=10), key_prefix="voice_config_formats")
    async def get_supported_audio_formats(self) -> List[str]:
        """
        Get supported audio formats
        Cached for 10 minutes
        """
        formats = [
            "audio/wav", "audio/mp3", "audio/ogg", "audio/flac",
            "audio/aac", "audio/m4a", "audio/webm"
        ]
        
        logger.debug(f"Retrieved {len(formats)} supported audio formats")
        return formats
    
    @cache(ttl=timedelta(minutes=10), key_prefix="voice_config_engines")
    async def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available voice engines and their capabilities
        Cached for 10 minutes
        """
        engines = {
            "stt_engines": {
                "whisper": {
                    "name": "OpenAI Whisper",
                    "description": "High-quality speech recognition",
                    "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                    "available": True
                },
                "google": {
                    "name": "Google Speech-to-Text",
                    "description": "Google Cloud STT service",
                    "languages": ["en-US", "es-ES", "fr-FR", "de-DE"],
                    "available": False  # Requires API key
                },
                "azure": {
                    "name": "Azure Speech Services",
                    "description": "Microsoft Azure STT service",
                    "languages": ["en-US", "es-ES", "fr-FR", "de-DE"],
                    "available": False  # Requires API key
                },
                "mock": {
                    "name": "Mock STT Engine",
                    "description": "Testing and development engine",
                    "languages": ["en-US"],
                    "available": True
                }
            },
            "tts_engines": {
                "coqui": {
                    "name": "Coqui TTS",
                    "description": "Open-source neural TTS",
                    "voices": await self.get_supported_voices("coqui"),
                    "available": True
                },
                "google": {
                    "name": "Google Text-to-Speech",
                    "description": "Google Cloud TTS service",
                    "voices": await self.get_supported_voices("google"),
                    "available": False  # Requires API key
                },
                "azure": {
                    "name": "Azure Speech Services",
                    "description": "Microsoft Azure TTS service",
                    "voices": await self.get_supported_voices("azure"),
                    "available": False  # Requires API key
                },
                "mock": {
                    "name": "Mock TTS Engine",
                    "description": "Testing and development engine",
                    "voices": await self.get_supported_voices("mock"),
                    "available": True
                }
            }
        }
        
        logger.debug("Retrieved voice engine configuration")
        return engines
    
    @cache(ttl=timedelta(minutes=5), key_prefix="voice_config_settings")
    async def get_voice_settings(self) -> Dict[str, Any]:
        """
        Get voice service settings
        Cached for 5 minutes
        """
        settings = {
            "max_audio_duration": 300,  # 5 minutes
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "default_language": "en-US",
            "default_stt_engine": "whisper",
            "default_tts_engine": "coqui",
            "default_voice": "jenny",
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "vad_threshold": 0.5,
            "silence_timeout": 2.0,
            "allowed_formats": await self.get_supported_audio_formats(),
            "rate_limits": {
                "stt_requests_per_minute": 30,
                "tts_requests_per_minute": 30,
                "max_concurrent_requests": 5
            }
        }
        
        logger.debug("Retrieved voice service settings")
        return settings
    
    @cache(ttl=timedelta(minutes=15), key_prefix="voice_config_capabilities")
    async def get_service_capabilities(self) -> Dict[str, Any]:
        """
        Get complete voice service capabilities
        Cached for 15 minutes
        """
        capabilities = {
            "languages": await self.get_supported_languages(),
            "audio_formats": await self.get_supported_audio_formats(),
            "engines": await self.get_available_engines(),
            "settings": await self.get_voice_settings(),
            "features": {
                "speech_to_text": True,
                "text_to_speech": True,
                "voice_activity_detection": True,
                "speech_segmentation": True,
                "wake_word_detection": True,
                "voice_commands": True,
                "voice_profiles": True,
                "conversation_mode": True,
                "real_time_processing": False,
                "batch_processing": True
            },
            "version": "1.0.0",
            "last_updated": "2024-10-16"
        }
        
        logger.debug("Retrieved complete voice service capabilities")
        return capabilities
    
    async def clear_cache(self):
        """Clear all voice configuration cache"""
        patterns = [
            "voice_config_languages:*",
            "voice_config_voices:*", 
            "voice_config_formats:*",
            "voice_config_engines:*",
            "voice_config_settings:*",
            "voice_config_capabilities:*"
        ]
        
        cleared_count = 0
        for pattern in patterns:
            count = await cache_manager.clear_pattern(pattern)
            cleared_count += count
        
        logger.info(f"Cleared {cleared_count} voice configuration cache entries")
        return cleared_count


# Global voice config service instance
voice_config_service = VoiceConfigService()


# Convenience functions
async def get_voice_languages() -> List[str]:
    """Get supported languages"""
    return await voice_config_service.get_supported_languages()


async def get_voice_voices(engine: Optional[str] = None) -> List[str]:
    """Get supported voices"""
    return await voice_config_service.get_supported_voices(engine)


async def get_voice_formats() -> List[str]:
    """Get supported audio formats"""
    return await voice_config_service.get_supported_audio_formats()


async def get_voice_engines() -> Dict[str, Dict[str, Any]]:
    """Get available engines"""
    return await voice_config_service.get_available_engines()


async def get_voice_settings() -> Dict[str, Any]:
    """Get voice service settings"""
    return await voice_config_service.get_voice_settings()


async def get_voice_capabilities() -> Dict[str, Any]:
    """Get complete voice service capabilities"""
    return await voice_config_service.get_service_capabilities()


# Export all functions
__all__ = [
    "VoiceConfigService",
    "voice_config_service",
    "get_voice_languages",
    "get_voice_voices", 
    "get_voice_formats",
    "get_voice_engines",
    "get_voice_settings",
    "get_voice_capabilities"
]
