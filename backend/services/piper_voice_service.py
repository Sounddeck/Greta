"""
GRETA PAI Piper TTS Voice Service
Open-source voice synthesis for GRETA PAI with German accent support
"""

import subprocess
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class PiperVoiceService:
    """
    Piper TTS Voice Service for GRETA PAI
    High-quality open-source voice synthesis with German accent support
    """

    def __init__(self):
        self.models_dir = Path("/Users/macone/Desktop/Greta/models/piper")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.voice_models = {
            "de_DE-karl-medium": {
                "name": "Karl (Male)",
                "language": "German",
                "accent": "Standard German",
                "quality": "High"
            },
            "de_DE-ramona-medium": {
                "name": "Ramona (Female)",
                "language": "German",
                "accent": "Standard German",
                "quality": "High"
            },
            "en_US-lessac-medium": {
                "name": "Sarah (Female)",
                "language": "English",
                "accent": "American",
                "quality": "High"
            }
        }

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_generations": 0,
            "average_generation_time": 0,
            "last_generation_time": 0
        }

    async def generate_audio(self, text: str, voice_model: str = "de_DE-karl-medium",
                           speed: float = 1.0) -> bytes:
        """
        Generate audio from text using Piper TTS

        Args:
            text: Text to synthesize
            voice_model: Voice model to use
            speed: Speech speed (0.5-2.0)

        Returns:
            WAV audio data as bytes
        """
        start_time = time.time()

        try:
            self.metrics["total_requests"] += 1

            if voice_model not in self.voice_models:
                raise ValueError("Unknown voice model")

            # Find model and config files
            model_file = self._find_model_file(voice_model)
            config_file = model_file.with_suffix('.onxx.conf')

            if not model_file.exists():
                await self._download_voice_model(voice_model)

            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                output_file = temp_file.name

            try:
                # Run Piper TTS in thread pool to not block async
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._run_piper,
                    text, str(model_file), str(config_file), output_file, speed
                )

                # Read the generated audio
                with open(output_file, 'rb') as f:
                    audio_data = f.read()

                generation_time = time.time() - start_time
                self.metrics["last_generation_time"] = generation_time
                self.metrics["successful_generations"] += 1

                # Update average generation time
                total_time = self.metrics["average_generation_time"] * (self.metrics["successful_generations"] - 1)
                self.metrics["average_generation_time"] = (total_time + generation_time) / self.metrics["successful_generations"]

                logger.info(f"✅ Generated {len(audio_data)} bytes of audio in {generation_time:.2f}s")
                return audio_data

            finally:
                # Cleanup temp file
                try:
                    os.unlink(output_file)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"❌ Voice synthesis failed: {e}")
            raise

    def _run_piper(self, text: str, model_path: str, config_path: str, output_file: str, speed: float):
        """Run Piper TTS subprocess (synchronous)"""
        cmd = [
            "piper",
            "--model", model_path,
            "--config", config_path,
            "--output_file", output_file,
            "--length_scale", str(1.0/speed)  # Inverse relationship
        ]

        # Pipe text input
        process = subprocess.run(
            cmd,
            input=text,
            text=True,
            capture_output=True,
            check=True,
            timeout=30  # 30 second timeout
        )

        if process.returncode != 0:
            raise Exception(f"Piper TTS failed: {process.stderr}")

    def _find_model_file(self, voice_model: str) -> Path:
        """Find the model file path"""
        model_file = self.models_dir / f"{voice_model}.onnx"

        # Check for alternative extensions
        if not model_file.exists():
            alt_file = self.models_dir / f"{voice_model}.onnx"
            if alt_file.exists():
                return alt_file

        return model_file

    async def _download_voice_model(self, voice_model: str):
        """Download voice model if not available"""
        import requests

        # HuggingFace Piper voices URL pattern
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
        model_parts = voice_model.split("-")

        if len(model_parts) >= 2:
            lang_region = model_parts[0]
            voice_name = model_parts[1]
            quality = model_parts[2] if len(model_parts) > 2 else "medium"

            # Download model file
            model_url = f"{base_url}/{lang_region}/{voice_name}/{quality}/{voice_model}.onnx"
            config_url = f"{base_url}/{lang_region}/{voice_name}/{quality}/{voice_model}.onnx.conf"

            self.models_dir.mkdir(parents=True, exist_ok=True)

            model_file = self.models_dir / f"{voice_model}.onnx"
            config_file = self.models_dir / f"{voice_model}.onnx.conf"

            logger.info(f"📥 Downloading voice model: {voice_model}")

            # Download files
            for url, file_path in [(model_url, model_file), (config_url, config_file)]:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            logger.info(f"✅ Downloaded voice model: {voice_model}")

    def list_available_models(self) -> List[str]:
        """List all available voice model names"""
        return list(self.voice_models.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific voice model"""
        return self.voice_models.get(model_name)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get voice service performance statistics"""
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": self.metrics["successful_generations"] / max(self.metrics["total_requests"], 1),
            "average_generation_time": self.metrics["average_generation_time"],
            "last_generation_time": self.metrics["last_generation_time"]
        }

    def preload_models(self, models: List[str]):
        """Preload voice models into memory"""
        # Note: Piper doesn't support model preloading in the same way
        # but we can validate they exist
        for model in models:
            if not self._find_model_file(model).exists():
                asyncio.create_task(self._download_voice_model(model))

    async def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("🔄 Piper voice service shut down")

# Queue management for background voice processing
class VoiceQueue:
    """Background voice processing queue"""

    def __init__(self, voice_service: PiperVoiceService):
        self.voice_service = voice_service
        self.queue = asyncio.Queue()
        self.processing_task = None
        self.is_running = False

    async def start(self):
        """Start the background processing task"""
        if not self.is_running:
            self.is_running = True
            self.processing_task = asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stop the background processing"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, request: Dict[str, Any], callback=None):
        """Add voice request to queue"""
        await self.queue.put({
            "request": request,
            "callback": callback,
            "timestamp": time.time()
        })

    async def _process_queue(self):
        """Background queue processing"""
        while self.is_running:
            try:
                # Wait for next request with timeout
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                request = item["request"]
                callback = item["callback"]

                try:
                    # Process the voice request
                    audio_data = await self.voice_service.generate_audio(
                        request.get("text", ""),
                        request.get("voice_model", "de_DE-karl-medium"),
                        request.get("speed", 1.0)
                    )

                    # Call callback with result
                    if callback:
                        callback({
                            "success": True,
                            "audio_data": audio_data,
                            "request": request
                        })

                except Exception as e:
                    logger.error(f"Voice queue processing failed: {e}")

                    # Call callback with error
                    if callback:
                        callback({
                            "success": False,
                            "error": str(e),
                            "request": request
                        })

                self.queue.task_done()

            except asyncio.TimeoutError:
                continue  # No items in queue, continue waiting
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

# Global voice service instance
voice_service = PiperVoiceService()
voice_queue = VoiceQueue(voice_service)
