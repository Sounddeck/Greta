"""
Image Generation Providers
Modular architecture for swapping between different image generation services
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class ImageGenerationProvider(ABC):
    """Base class for image generation providers"""
    
    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image from text prompt
        
        Returns:
            Dict with keys: image_data (base64), format, metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        pass


class StableDiffusionProvider(ImageGenerationProvider):
    """Stable Diffusion provider using diffusers library (local, free)"""
    
    def __init__(self):
        self.model_id = "runwayml/stable-diffusion-v1-5"  # Lighter than SDXL, faster on M2
        self.pipeline = None
        self.device = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of the model"""
        if self._initialized:
            return
            
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            
            # Detect device (MPS for M2 Mac, CUDA for NVIDIA, CPU fallback)
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("🎨 Using MPS (Metal) for Stable Diffusion on M2 Mac")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("🎨 Using CUDA for Stable Diffusion")
            else:
                self.device = "cpu"
                logger.warning("⚠️ Using CPU for Stable Diffusion (will be slow)")
            
            # Load pipeline with optimizations for M2 Mac
            logger.info(f"📥 Loading Stable Diffusion model: {self.model_id}")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                safety_checker=None,  # Disable for speed
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.device == "mps":
                # MPS-specific optimizations
                self.pipeline.enable_attention_slicing()
                logger.info("✅ Enabled attention slicing for lower memory usage")
            elif self.device == "cuda":
                # CUDA-specific optimizations
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            self._initialized = True
            logger.info("✅ Stable Diffusion initialized successfully")
            
        except ImportError as e:
            logger.error(f"❌ Missing dependencies for Stable Diffusion: {e}")
            logger.error("Install with: pip install diffusers transformers accelerate torch torchvision")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Stable Diffusion: {e}")
            raise
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Stable Diffusion"""
        try:
            # Initialize on first use
            if not self._initialized:
                self._initialize()
            
            logger.info(f"🎨 Generating image: '{prompt[:50]}...'")
            start_time = datetime.now()
            
            # Default negative prompt if not provided
            if negative_prompt is None:
                negative_prompt = "blurry, bad quality, distorted, ugly, low resolution"
            
            # Generate image
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )
            
            # Get the image
            image = result.images[0]
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Image generated in {generation_time:.2f}s")
            
            return {
                "image_data": image_base64,
                "format": "png",
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "model": self.model_id,
                    "device": self.device,
                    "generation_time_seconds": generation_time,
                    "provider": "stable_diffusion"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Stable Diffusion is available"""
        try:
            import torch
            import diffusers
            return True
        except ImportError:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": "Stable Diffusion",
            "provider": "stable_diffusion",
            "model": self.model_id,
            "cost": "$0/month (local)",
            "device": self.device if self._initialized else "not_initialized",
            "available": self.is_available(),
            "features": [
                "Text-to-image generation",
                "Negative prompts",
                "Customizable resolution",
                "Local execution (privacy)",
                "No API costs"
            ]
        }


class DalleProvider(ImageGenerationProvider):
    """DALL-E provider (OpenAI) - Stub for future implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using DALL-E 3"""
        # TODO: Implement when needed
        raise NotImplementedError("DALL-E provider not yet implemented")
    
    def is_available(self) -> bool:
        """Check if DALL-E is available"""
        return self.api_key is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": "DALL-E 3",
            "provider": "dalle",
            "model": "dall-e-3",
            "cost": "$0.04 per image (1024x1024)",
            "available": self.is_available(),
            "features": [
                "High-quality images",
                "1024x1024 resolution",
                "Fast generation",
                "Requires OpenAI API key"
            ]
        }


class MidjourneyProvider(ImageGenerationProvider):
    """Midjourney provider - Stub for future implementation"""
    
    def __init__(self):
        self.api_key = os.getenv("MIDJOURNEY_API_KEY")
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Midjourney"""
        # TODO: Implement when needed
        raise NotImplementedError("Midjourney provider not yet implemented")
    
    def is_available(self) -> bool:
        """Check if Midjourney is available"""
        return self.api_key is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": "Midjourney",
            "provider": "midjourney",
            "model": "midjourney-v6",
            "cost": "$10/month (basic plan)",
            "available": self.is_available(),
            "features": [
                "Artistic style",
                "High quality",
                "Multiple variations",
                "Requires Midjourney subscription"
            ]
        }


class ImageGenerationService:
    """Service that manages multiple image generation providers"""
    
    def __init__(self, default_provider: str = "stable_diffusion"):
        self.providers = {
            "stable_diffusion": StableDiffusionProvider(),
            "dalle": DalleProvider(),
            "midjourney": MidjourneyProvider()
        }
        self.default_provider = default_provider
        
        # Override from environment variable
        env_provider = os.getenv("IMAGE_GENERATION_PROVIDER", "").lower()
        if env_provider in self.providers:
            self.default_provider = env_provider
    
    async def generate_image(
        self,
        prompt: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using specified or default provider"""
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_instance = self.providers[provider_name]
        
        if not provider_instance.is_available():
            raise RuntimeError(f"Provider {provider_name} is not available")
        
        return await provider_instance.generate_image(prompt, **kwargs)
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all available providers"""
        return [
            provider.get_info()
            for provider in self.providers.values()
        ]
    
    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.providers[provider_name].get_info()
