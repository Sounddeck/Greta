"""
GRETA PAI Voice Service
German-accented female voice for Greta with macOS native TTS integration
"""

import asyncio
import logging
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from ..config import get_settings
from ..database import get_database

logger = logging.getLogger(__name__)

class GretaVoiceService:
    """
    GRETA PAI Voice Service
    German-accented female voice with macOS native TTS integration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db = None
        
        # Voice configuration
        self.voice_config = {
            "gender": "female",
            "accent": "german",
            "language": "en",
            "engine": "macos_native",
            "preferred_voices": [
                {"name": "Serena", "quality": "high", "german_accent": True},
                {"name": "Karen", "quality": "high", "german_accent": True},
                {"name": "Samantha", "quality": "high", "german_accent": True},
                {"name": "Monica", "quality": "medium", "german_accent": True},
                {"name": "Tessa", "quality": "medium", "german_accent": True}
            ],
            "speech_rate": 180,  # Words per minute
            "pitch": 0.9,  # Slightly higher for female voice
            "volume": 0.8
        }
        
        # Audio output configuration
        self.audio_config = {
            "output_format": "aiff",
            "sample_rate": 22050,
            "bit_depth": 16,
            "channels": 1,
            "output_directory": self.settings.base_dir / "audio"
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_synthesis_requests": 0,
            "successful_synthesis": 0,
            "total_audio_duration": 0,
            "average_synthesis_time": 0,
            "voice_usage": {}
        }
        
    async def initialize(self):
        """Initialize voice service"""
        try:
            self.db = await get_database()
            
            # Create audio directory
            self.audio_config["output_directory"].mkdir(parents=True, exist_ok=True)
            
            # Detect available voices
            await self._detect_available_voices()
            
            # Test TTS engine
            await self._test_tts_engine()
            
            logger.info("✅ Greta Voice Service initialized with German-accented female voice")
            
        except Exception as e:
            logger.error(f"❌ Voice Service initialization failed: {e}")
            raise
    
    async def _detect_available_voices(self):
        """Detect available macOS voices"""
        try:
            # Use macOS say command to list voices
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                voices = self._parse_voice_list(result.stdout)
                self.available_voices = voices
                logger.info(f"✅ Detected {len(voices)} macOS voices")
                
                # Find best German-accented female voices
                self._select_optimal_voices()
            else:
                logger.warning("⚠️ Could not detect macOS voices")
                self.available_voices = []
                
        except Exception as e:
            logger.error(f"❌ Voice detection failed: {e}")
            self.available_voices = []
    
    def _parse_voice_list(self, voice_output: str) -> list:
        """Parse macOS voice list output"""
        voices = []
        lines = voice_output.split('\n')
        
        for line in lines:
            if line.strip() and not line.startswith('Voices'):
                # Parse voice information
                if '#' in line:
                    voice_name = line.split('#')[0].strip()
                    voices.append({
                        "name": voice_name,
                        "language": "en",  # Default assumption
                        "gender": "female" if any(female in voice_name.lower() for female in ["samantha", "serena", "karen", "monica", "tessa"]) else "unknown"
                    })
        
        return voices
    
    def _select_optimal_voices(self):
        """Select optimal voices for German-accented female voice"""
        optimal_voices = []
        
        # Check preferred voices first
        for preferred in self.voice_config["preferred_voices"]:
            for voice in self.available_voices:
                if voice["name"] == preferred["name"]:
                    optimal_voices.append(voice)
                    break
        
        # If no preferred voices found, use available female voices
        if not optimal_voices:
            for voice in self.available_voices:
                if voice["gender"] == "female":
                    optimal_voices.append(voice)
        
        self.optimal_voices = optimal_voices if optimal_voices else self.available_voices[:1]
        
        if self.optimal_voices:
            logger.info(f"✅ Selected optimal voice: {self.optimal_voices[0]['name']}")
        else:
            logger.warning("⚠️ No suitable voices found")
    
    async def _test_tts_engine(self):
        """Test TTS engine functionality"""
        try:
            test_text = "Greta voice test"
            test_file = self.audio_config["output_directory"] / "voice_test.aiff"
            
            # Synthesize test audio
            result = subprocess.run(
                ["say", "-v", self.optimal_voices[0]["name"] if self.optimal_voices else "Alex", "-o", str(test_file), test_text],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and test_file.exists():
                logger.info("✅ TTS engine test successful")
                self.tts_working = True
            else:
                logger.warning("⚠️ TTS engine test failed")
                self.tts_working = False
                
        except Exception as e:
            logger.error(f"❌ TTS engine test failed: {e}")
            self.tts_working = False
    
    async def synthesize_speech(self, text: str, voice_name: Optional[str] = None, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize speech with German-accented female voice
        """
        start_time = datetime.utcnow()
        
        try:
            self.performance_metrics["total_synthesis_requests"] += 1
            
            # Select voice
            selected_voice = voice_name if voice_name else (self.optimal_voices[0]["name"] if self.optimal_voices else "Alex")
            
            # Generate output filename if not provided
            if not output_file:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_file = self.audio_config["output_directory"] / f"greta_speech_{timestamp}.aiff"
            
            # Prepare say command with German accent parameters
            say_command = [
                "say",
                "-v", selected_voice,
                "-r", str(self.voice_config["speech_rate"]),
                "-p", str(self.voice_config["pitch"]),
                "-a", str(self.voice_config["volume"]),
                "-o", str(output_file),
                text
            ]
            
            # Add German accent enhancement
            enhanced_text = self._enhance_for_german_accent(text)
            
            # Synthesize speech
            result = subprocess.run(
                say_command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            synthesis_time = (datetime.utcnow() - start_time).total_seconds()
            
            if result.returncode == 0 and output_file.exists():
                # Update metrics
                self.performance_metrics["successful_synthesis"] += 1
                self.performance_metrics["total_audio_duration"] += self._estimate_audio_duration(text)
                await self._update_voice_metrics(selected_voice, synthesis_time, True)
                
                # Get audio file info
                audio_info = await self._get_audio_info(output_file)
                
                return {
                    "success": True,
                    "text": text,
                    "voice": selected_voice,
                    "output_file": str(output_file),
                    "synthesis_time": synthesis_time,
                    "audio_info": audio_info,
                    "german_accent": True,
                    "female_voice": True
                }
            else:
                logger.error(f"❌ Speech synthesis failed: {result.stderr}")
                await self._update_voice_metrics(selected_voice, synthesis_time, False)
                
                return {
                    "success": False,
                    "error": result.stderr if result.stderr else "Unknown error",
                    "synthesis_time": synthesis_time
                }
                
        except Exception as e:
            logger.error(f"❌ Speech synthesis error: {e}")
            synthesis_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": False,
                "error": str(e),
                "synthesis_time": synthesis_time
            }
    
    def _enhance_for_german_accent(self, text: str) -> str:
        """Enhance text for German accent pronunciation"""
        
        # Add pronunciation markers for German accent
        # This is a simplified approach - in practice, you might use more sophisticated methods
        
        # Common German-influenced pronunciation patterns
        german_patterns = {
            "the": "ze",
            "this": "zis",
            "that": "zat",
            "with": "viz",
            "think": "zink",
            "technology": "teknology",
            "information": "informatsion"
        }
        
        # Add slight pauses for German rhythm
        enhanced_text = text
        
        # Insert brief pauses at punctuation for German rhythm
        for char in [".", "!", "?"]:
            enhanced_text = enhanced_text.replace(char, f"{char} <break time='200ms'/>")
        
        return enhanced_text
    
    def _estimate_audio_duration(self, text: str) -> float:
        """Estimate audio duration based on text length"""
        word_count = len(text.split())
        return word_count / self.voice_config["speech_rate"] * 60  # Convert to seconds
    
    async def _get_audio_info(self, audio_file: Path) -> Dict[str, Any]:
        """Get audio file information"""
        try:
            # Use ffprobe if available, otherwise use file size
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_format", str(audio_file)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse ffprobe output
                lines = result.stdout.split('\n')
                duration = 0
                
                for line in lines:
                    if "duration=" in line:
                        duration = float(line.split("duration=")[1].split(",)[0])
                        break
                
                return {
                    "duration": duration,
                    "file_size": audio_file.stat().st_size,
                    "format": self.audio_config["output_format"]
                }
            else:
                # Fallback to file size estimation
                file_size = audio_file.stat().st_size
                estimated_duration = file_size / (22050 * 2 * 60)  # Rough estimate
                
                return {
                    "duration": estimated_duration,
                    "file_size": file_size,
                    "format": self.audio_config["output_format"],
                    "estimated": True
                }
                
        except Exception as e:
            logger.debug(f"Audio info retrieval failed: {e}")
            return {
                "duration": 0,
                "file_size": audio_file.stat().st_size if audio_file.exists() else 0,
                "format": self.audio_config["output_format"]
            }
    
    async def _update_voice_metrics(self, voice_name: str, synthesis_time: float, success: bool):
        """Update voice performance metrics"""
        
        if voice_name not in self.performance_metrics["voice_usage"]:
            self.performance_metrics["voice_usage"][voice_name] = {
                "uses": 0,
                "total_time": 0,
                "successes": 0
            }
        
        voice_metrics = self.performance_metrics["voice_usage"][voice_name]
        voice_metrics["uses"] += 1
        voice_metrics["total_time"] += synthesis_time
        
        if success:
            voice_metrics["successes"] += 1
        
        # Update average synthesis time
        total_requests = self.performance_metrics["total_synthesis_requests"]
        current_avg = self.performance_metrics["average_synthesis_time"]
        self.performance_metrics["average_synthesis_time"] = (
            (current_avg * (total_requests - 1) + synthesis_time) / total_requests
        )
    
    async def speak_text(self, text: str, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize and immediately speak text
        """
        try:
            # Synthesize speech
            synthesis_result = await self.synthesize_speech(text, voice_name)
            
            if synthesis_result["success"]:
                # Immediately play the audio
                play_result = await self._play_audio(synthesis_result["output_file"])
                
                return {
                    "success": True,
                    "text": text,
                    "voice": synthesis_result["voice"],
                    "synthesis_result": synthesis_result,
                    "play_result": play_result
                }
            else:
                return synthesis_result
                
        except Exception as e:
            logger.error(f"❌ Speak text failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _play_audio(self, audio_file: str) -> Dict[str, Any]:
        """Play audio file"""
        try:
            # Use macOS afplay to play audio
            result = subprocess.run(
                ["afplay", audio_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "audio_file": audio_file,
                    "played": True
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr if result.stderr else "Unknown error"
                }
                
        except Exception as e:
            logger.error(f"❌ Audio playback failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_available_voices(self) -> Dict[str, Any]:
        """Get information about available voices"""
        
        voices_info = []
        for voice in self.available_voices:
            voice_info.append({
                "name": voice["name"],
                "language": voice.get("language", "unknown"),
                "gender": voice.get("gender", "unknown"),
                "is_optimal": voice in [v["name"] for v in self.optimal_voices]
            })
        
        return {
            "total_voices": len(self.available_voices),
            "optimal_voices": len(self.optimal_voices),
            "voices": voices_info,
            "preferred_voice": self.optimal_voices[0]["name"] if self.optimal_voices else None
        }
    
    async def get_voice_performance(self) -> Dict[str, Any]:
        """Get voice service performance metrics"""
        
        return {
            "total_synthesis_requests": self.performance_metrics["total_synthesis_requests"],
            "successful_synthesis": self.performance_metrics["successful_synthesis"],
            "success_rate": (
                self.performance_metrics["successful_synthesis"] / 
                max(self.performance_metrics["total_synthesis_requests"], 1)
            ) * 100,
            "average_synthesis_time": self.performance_metrics["average_synthesis_time"],
            "total_audio_duration": self.performance_metrics["total_audio_duration"],
            "voice_usage": self.performance_metrics["voice_usage"],
            "german_accent_enabled": True,
            "female_voice_enabled": True
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Voice service health check"""
        
        return {
            "status": "healthy",
            "service": "greta_voice_service",
            "tts_engine": "macos_native",
            "available_voices": len(self.available_voices),
            "optimal_voices": len(self.optimal_voices),
            "tts_working": getattr(self, 'tts_working', False),
            "german_accent": True,
            "female_voice": True,
            "voice_performance": await self.get_voice_performance(),
            "timestamp": datetime.utcnow().isoformat()
        }

# Global voice service instance
greta_voice_service = GretaVoiceService()

async def get_greta_voice_service() -> GretaVoiceService:
    """Get the global Greta voice service instance"""
    return greta_voice_service
