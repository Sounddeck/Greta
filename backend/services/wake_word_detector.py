"""
Wake Word Detection Service for Greta
Uses openWakeWord for accurate, low-latency wake word detection

Supports wake words:
- "Greta" (custom trained model)
- "Hey Jarvis" (pre-trained, temporary until Greta model is trained)
- Other pre-trained models: "alexa", "hey mycroft"

Features:
- No API keys required (fully offline)
- Built-in VAD (Voice Activity Detection)
- Optional Speex noise suppression
- Custom model training support

Author: GRETA PAI System
Date: October 20, 2025
Updated: October 20, 2025 - Migrated from Porcupine to openWakeWord
"""

from openwakeword.model import Model
import openwakeword
import pyaudio
import struct
import numpy as np
from typing import Callable, Optional, List, Dict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Wake word detector for "Greta" voice assistant

    Uses openWakeWord for high-accuracy wake word detection.
    Supports custom wake words and adjustable sensitivity.

    Features:
    - No API keys required (fully offline)
    - Built-in VAD for false-positive reduction
    - Optional Speex noise suppression
    - Custom model training support
    """

    # openWakeWord audio parameters
    SAMPLE_RATE = 16000  # 16kHz
    CHUNK_SIZE = 1280    # 80ms frames (1280 samples at 16kHz)

    def __init__(
        self,
        model_paths: List[str] = None,
        inference_framework: str = "onnx",
        enable_vad: bool = True,
        vad_threshold: float = 0.5,
        enable_noise_suppression: bool = False,
        detection_threshold: float = 0.5
    ):
        """
        Initialize wake word detector

        Args:
            model_paths: List of paths to .tflite or .onnx model files
                        If None, uses pre-trained "hey jarvis" model
            inference_framework: "onnx" or "tflite" (default: "onnx")
            enable_vad: Enable Voice Activity Detection (default: True)
            vad_threshold: VAD threshold 0.0-1.0 (default: 0.5)
            enable_noise_suppression: Enable Speex noise suppression (default: False)
            detection_threshold: Wake word detection threshold 0.0-1.0 (default: 0.5)
        """
        self.model_paths = model_paths
        self.inference_framework = inference_framework
        self.enable_vad = enable_vad
        self.vad_threshold = vad_threshold
        self.enable_noise_suppression = enable_noise_suppression
        self.detection_threshold = detection_threshold

        self.model = None
        self.audio_stream = None
        self.is_listening = False
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Statistics
        self.detections_count = {}  # Per-model detection counts
        self.false_positives = 0
        self.start_time = None

        logger.info(f"🎤 WakeWordDetector initialized")
        logger.info(f"   Framework: {inference_framework}")
        logger.info(f"   VAD enabled: {enable_vad} (threshold: {vad_threshold})")
        logger.info(f"   Noise suppression: {enable_noise_suppression}")
        logger.info(f"   Detection threshold: {detection_threshold}")
    
    def initialize(self):
        """Initialize openWakeWord model"""
        try:
            # Download pre-trained models if needed
            if not self.model_paths:
                logger.info("📥 Downloading pre-trained models...")
                openwakeword.utils.download_models()

                # Use "hey jarvis" as default (temporary until Greta model is trained)
                self.model_paths = ["hey_jarvis_v0.1"]
                logger.info("   Using pre-trained 'hey jarvis' model (temporary)")
                logger.info("   Train custom 'Greta' model for production use")

            # Initialize model
            self.model = Model(
                wakeword_models=self.model_paths,
                inference_framework=self.inference_framework,
                vad_threshold=self.vad_threshold if self.enable_vad else 0,
                enable_speex_noise_suppression=self.enable_noise_suppression
            )

            # Initialize detection counts for each model
            for model_name in self.model.models.keys():
                self.detections_count[model_name] = 0

            logger.info(f"✅ openWakeWord initialized successfully")
            logger.info(f"   Models loaded: {list(self.model.models.keys())}")
            logger.info(f"   Sample rate: {self.SAMPLE_RATE} Hz")
            logger.info(f"   Frame size: {self.CHUNK_SIZE} samples (80ms)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize openWakeWord: {e}")
            logger.error(f"   Check that openwakeword is installed: pip install openwakeword")
            raise
    
    def start_audio_stream(self):
        """Start audio input stream from microphone"""
        try:
            pa = pyaudio.PyAudio()

            # Find default input device
            default_device = pa.get_default_input_device_info()
            logger.info(f"🎤 Using microphone: {default_device['name']}")

            self.audio_stream = pa.open(
                rate=self.SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
                input_device_index=None  # Use default microphone
            )

            logger.info("✅ Audio stream started")

        except Exception as e:
            logger.error(f"❌ Failed to start audio stream: {e}")
            logger.error(f"   Check microphone permissions in System Preferences")
            raise
    
    async def start_listening(self, on_wake_word: Callable[[str], None]):
        """
        Start listening for wake words

        Args:
            on_wake_word: Async callback function called when wake word is detected
                         Receives the detected keyword as argument

        Example:
            async def on_wake_word_detected(keyword: str):
                print(f"Wake word detected: {keyword}")
                await voice_service.speak("Yes? How can I help you?")

            await detector.start_listening(on_wake_word_detected)
        """
        if self.is_listening:
            logger.warning("⚠️ Already listening for wake words")
            return

        try:
            # Initialize if not already done
            if not self.model:
                self.initialize()

            # Start audio stream
            if not self.audio_stream:
                self.start_audio_stream()

            self.is_listening = True
            self.start_time = asyncio.get_event_loop().time()

            logger.info(f"🎤 Wake word detection started")
            logger.info(f"   Listening for: {list(self.model.models.keys())}")
            logger.info(f"   Detection threshold: {self.detection_threshold}")

            # Run detection loop in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._detection_loop,
                on_wake_word
            )

        except Exception as e:
            logger.error(f"❌ Wake word detection error: {e}")
            self.is_listening = False
            raise
    
    def _detection_loop(self, on_wake_word: Callable[[str], None]):
        """
        Main detection loop (runs in thread pool)

        Continuously reads audio frames and processes them for wake word detection.
        Calls the callback function when a wake word is detected.

        Args:
            on_wake_word: Callback function for wake word detection
        """
        try:
            while self.is_listening:
                # Read audio frame from microphone
                audio_data = self.audio_stream.read(
                    self.CHUNK_SIZE,
                    exception_on_overflow=False
                )

                # Convert bytes to numpy array (int16)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Get predictions from openWakeWord
                predictions = self.model.predict(audio_array)

                # Check each model for wake word detection
                for model_name, score in predictions.items():
                    if score >= self.detection_threshold:
                        # Wake word detected!
                        self.detections_count[model_name] = self.detections_count.get(model_name, 0) + 1
                        total_detections = sum(self.detections_count.values())

                        logger.info(f"✅ Wake word detected: '{model_name}' (score: {score:.3f}, #{total_detections})")

                        # Call callback (run in event loop)
                        asyncio.run(on_wake_word(model_name))

                        # Reset model to avoid multiple detections
                        self.model.reset()
                        break  # Only trigger once per frame

        except Exception as e:
            logger.error(f"❌ Detection loop error: {e}")
            self.is_listening = False
    
    def stop_listening(self):
        """Stop listening for wake words and clean up resources"""
        logger.info("🛑 Stopping wake word detection...")
        self.is_listening = False

        # Close audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            logger.info("   Audio stream closed")

        # Reset model (no need to delete, just reset state)
        if self.model:
            self.model.reset()
            logger.info("   Model reset")

        # Calculate uptime
        if self.start_time:
            uptime = asyncio.get_event_loop().time() - self.start_time
            logger.info(f"   Uptime: {uptime:.1f} seconds")

        total_detections = sum(self.detections_count.values())
        logger.info(f"✅ Wake word detection stopped (Total detections: {total_detections})")
        logger.info(f"   Per-model detections: {self.detections_count}")
    
    def get_statistics(self) -> dict:
        """
        Get detection statistics

        Returns:
            Dictionary with detection statistics including:
            - is_listening: Whether currently listening
            - models: List of loaded models
            - detection_threshold: Current detection threshold
            - total_detections: Total number of detections
            - per_model_detections: Detections per model
            - false_positives: Number of false positives (user-reported)
            - accuracy: Detection accuracy (0.0-1.0)
            - uptime_seconds: Time since detection started
        """
        total_detections = sum(self.detections_count.values())
        uptime = 0
        if self.start_time:
            uptime = asyncio.get_event_loop().time() - self.start_time

        return {
            "is_listening": self.is_listening,
            "models": list(self.model.models.keys()) if self.model else [],
            "detection_threshold": self.detection_threshold,
            "total_detections": total_detections,
            "per_model_detections": self.detections_count,
            "false_positives": self.false_positives,
            "accuracy": (total_detections - self.false_positives) / max(total_detections, 1),
            "uptime_seconds": uptime,
            "vad_enabled": self.enable_vad,
            "noise_suppression_enabled": self.enable_noise_suppression
        }
    
    def adjust_threshold(self, threshold: float):
        """
        Adjust detection threshold for all models

        Args:
            threshold: New detection threshold (0.0-1.0)
                      0.0 = very conservative (fewer false positives)
                      1.0 = very aggressive (more detections)

        Note: Takes effect immediately, no reinitialization needed
        """
        old_threshold = self.detection_threshold
        self.detection_threshold = max(0.0, min(1.0, threshold))

        logger.info(f"🔧 Adjusted detection threshold: {old_threshold:.2f} → {self.detection_threshold:.2f}")
        logger.info(f"   Lower threshold = more detections (may increase false positives)")
        logger.info(f"   Higher threshold = fewer detections (may miss some wake words)")

    def adjust_vad_threshold(self, vad_threshold: float):
        """
        Adjust VAD (Voice Activity Detection) threshold

        Args:
            vad_threshold: New VAD threshold (0.0-1.0)
                          Higher values = stricter voice detection

        Note: Requires reinitialization if currently listening
        """
        old_vad = self.vad_threshold
        self.vad_threshold = max(0.0, min(1.0, vad_threshold))

        logger.info(f"🔧 Adjusted VAD threshold: {old_vad:.2f} → {self.vad_threshold:.2f}")

        # Reinitialize if currently listening
        if self.is_listening:
            logger.info("   Reinitializing model with new VAD threshold...")
            was_listening = True
            self.stop_listening()
            self.initialize()
        else:
            logger.info("   Will take effect on next start_listening() call")
    
    def report_false_positive(self):
        """Report a false positive detection (for accuracy tracking)"""
        self.false_positives += 1
        logger.info(f"⚠️ False positive reported (Total: {self.false_positives})")
        logger.info(f"   Consider increasing detection threshold to reduce false positives")


# Global instance
_wake_word_detector: Optional[WakeWordDetector] = None


def get_wake_word_detector(
    model_paths: List[str] = None,
    enable_vad: bool = True,
    enable_noise_suppression: bool = False
) -> WakeWordDetector:
    """
    Get or create wake word detector instance (singleton pattern)

    Args:
        model_paths: List of paths to wake word models (optional)
                    If None, uses pre-trained "hey jarvis" model
        enable_vad: Enable Voice Activity Detection (default: True)
        enable_noise_suppression: Enable Speex noise suppression (default: False)

    Returns:
        WakeWordDetector instance

    Note:
        No API keys required! openWakeWord is fully offline.

        To use custom "Greta" model:
        1. Train model using Google Colab: https://github.com/dscripka/openWakeWord
        2. Download .tflite or .onnx file
        3. Pass path in model_paths parameter

        Example:
            detector = get_wake_word_detector(
                model_paths=["models/greta.tflite"],
                enable_vad=True
            )
    """
    global _wake_word_detector

    if _wake_word_detector is None:
        logger.info("🎤 Creating new WakeWordDetector instance")
        logger.info("   Using openWakeWord (no API keys required!)")

        _wake_word_detector = WakeWordDetector(
            model_paths=model_paths,
            enable_vad=enable_vad,
            enable_noise_suppression=enable_noise_suppression,
            detection_threshold=0.5
        )

    return _wake_word_detector

