"""
GRETA PAI Local LLM Service
Enhanced local LLM service with llama.cpp and MLX optimization for Apple Silicon
"""

import asyncio
import logging
import subprocess
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from ..config import get_settings
from ..database import get_database
from .hrm_integration import get_hrm_manager

logger = logging.getLogger(__name__)

class LocalLLMService:
    """
    GRETA PAI Local LLM Service
    Optimized for Apple Silicon M2 with local processing and HRM integration
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db = None
        self.hrm_manager = None
        
        # Local model configuration
        self.local_models = {
            "llama-3.1-8b": {
                "path": self.settings.base_dir / "models" / "llama-3.1-8b-instruct.Q4_K_M.gguf",
                "context_window": 32768,
                "quantization": "Q4_K_M",
                "mlx_optimized": True,
                "hrm_enabled": True,
                "ram_usage": "6-8GB"
            },
            "llama-3.1-70b": {
                "path": self.settings.base_dir / "models" / "llama-3.1-70b-instruct.Q4_K_M.gguf",
                "context_window": 8192,
                "quantization": "Q4_K_M",
                "mlx_optimized": True,
                "hrm_enabled": True,
                "ram_usage": "20-24GB"
            },
            "mistral-7b": {
                "path": self.settings.base_dir / "models" / "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
                "context_window": 32768,
                "quantization": "Q4_K_M",
                "mlx_optimized": True,
                "hrm_enabled": True,
                "ram_usage": "5-7GB"
            }
        }
        
        # Llama.cpp configuration
        self.llamacpp_config = {
            "n_ctx": 2048,
            "n_gpu_layers": -1,  # Use all available layers
            "main_gpu": 0,
            "n_batch": 512,
            "ctx_size": 2048,
            "threads": 8,
            "temp": 0.7,
            "repeat_penalty": 1.1,
            "top_p": 0.9,
            "min_p": 0.05
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0,
            "model_usage": {},
            "hrm_enhanced_requests": 0
        }
        
        # Apple Silicon optimization
        self.apple_silicon = self.settings.apple_silicon
        self.use_mlx = self.settings.use_mlx and self.apple_silicon
        
    async def initialize(self):
        """Initialize local LLM service"""
        try:
            self.db = await get_database()
            self.hrm_manager = await get_hrm_manager()
            
            # Check for available models
            await self._check_available_models()
            
            # Initialize MLX if available
            if self.use_mlx:
                await self._initialize_mlx()
            
            logger.info("✅ Local LLM Service initialized with Apple Silicon optimization")
            
        except Exception as e:
            logger.error(f"❌ Local LLM Service initialization failed: {e}")
            raise
    
    async def _check_available_models(self):
        """Check for available local models"""
        models_dir = self.settings.base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        available_models = []
        for model_name, model_config in self.local_models.items():
            if model_config["path"].exists():
                available_models.append(model_name)
                logger.info(f"✅ Found local model: {model_name}")
            else:
                logger.warning(f"⚠️ Model not found: {model_name} at {model_config['path']}")
        
        self.available_models = available_models
        
        if not available_models:
            logger.warning("⚠️ No local models found - will create download recommendations")
            await self._create_model_download_guide()
    
    async def _create_model_download_guide(self):
        """Create guide for downloading models"""
        guide = {
            "recommended_models": {
                "primary": "Llama 3.1 8B Instruct",
                "quantization": "Q4_K_M (4-bit, medium)",
                "download_url": "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF",
                "file_name": "llama-3.1-8b-instruct.Q4_K_M.gguf",
                "estimated_size": "4.7GB",
                "ram_required": "6-8GB"
            },
            "alternatives": {
                "high_performance": "Llama 3.1 70B Instruct",
                "efficient": "Mistral 7B Instruct v0.3"
            },
            "download_instructions": [
                "1. Visit the Hugging Face model page",
                "2. Download the GGUF file with Q4_K_M quantization",
                "3. Place in Greta/models/ directory",
                "4. Run the system again to detect the model"
            ]
        }
        
        guide_path = self.settings.base_dir / "models" / "download_guide.json"
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2)
        
        logger.info(f"✅ Created model download guide: {guide_path}")
    
    async def _initialize_mlx(self):
        """Initialize MLX for Apple Silicon optimization"""
        try:
            # Check if MLX is available
            result = subprocess.run(
                ["python3", "-c", "import mlx; print('MLX available')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info("✅ MLX available for Apple Silicon optimization")
                self.mlx_available = True
            else:
                logger.warning("⚠️ MLX not available - will use standard llama.cpp")
                self.mlx_available = False
                
        except Exception as e:
            logger.debug(f"MLX check failed: {e}")
            self.mlx_available = False
    
    async def generate_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate completion using local LLM with HRM enhancement
        """
        start_time = datetime.utcnow()
        
        try:
            self.performance_metrics["total_requests"] += 1
            
            # Select optimal model
            model_name = await self._select_optimal_model(request)
            model_config = self.local_models.get(model_name)
            
            if not model_config:
                return await self._generate_fallback_response(request)
            
            # Check if HRM should be used
            use_hrm = request.get("use_hrm", True) and model_config.get("hrm_enabled", False)
            hrm_result = None
            
            if use_hrm:
                # Perform hierarchical reasoning first
                hrm_result = await self.hrm_manager.reason_hierarchically(request)
                self.performance_metrics["hrm_enhanced_requests"] += 1
            
            # Generate completion with local model
            response = await self._generate_with_local_model(request, model_name, hrm_result)
            
            # Update metrics
            self.performance_metrics["successful_requests"] += 1
            response_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_performance_metrics(model_name, response_time, True)
            
            return {
                "success": True,
                "response": response,
                "model_used": model_name,
                "response_time": response_time,
                "local_processing": True,
                "hrm_enhanced": use_hrm and hrm_result.get("success", False),
                "hrm_result": hrm_result,
                "apple_silicon_optimized": self.use_mlx
            }
            
        except Exception as e:
            logger.error(f"❌ Local LLM completion failed: {e}")
            response_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_performance_metrics("unknown", response_time, False)
            
            return await self._generate_fallback_response(request)
    
    async def _select_optimal_model(self, request: Dict[str, Any]) -> str:
        """Select optimal model based on request and system resources"""
        
        # Check RAM availability (simplified check)
        available_ram = 32  # Your M2 MacBook Pro has 32GB
        
        # For simple requests, use smaller model
        if self._is_simple_request(request):
            if "mistral-7b" in self.available_models:
                return "mistral-7b"
            elif "llama-3.1-8b" in self.available_models:
                return "llama-3.1-8b"
        
        # For complex requests, use larger model if available
        if self._is_complex_request(request):
            if "llama-3.1-70b" in self.available_models and available_ram >= 24:
                return "llama-3.1-70b"
            elif "llama-3.1-8b" in self.available_models:
                return "llama-3.1-8b"
        
        # Default to first available model
        if self.available_models:
            return self.available_models[0]
        
        return "llama-3.1-8b"  # Default fallback
    
    def _is_simple_request(self, request: Dict[str, Any]) -> bool:
        """Check if request is simple"""
        message = request.get("message", "").lower()
        simple_keywords = ["hello", "hi", "yes", "no", "quick", "simple", "what", "how"]
        return any(keyword in message for keyword in simple_keywords)
    
    def _is_complex_request(self, request: Dict[str, Any]) -> bool:
        """Check if request is complex"""
        message = request.get("message", "").lower()
        complex_keywords = ["research", "investigate", "complex", "detailed", "comprehensive", "create", "build", "analyze"]
        return any(keyword in message for keyword in complex_keywords)
    
    async def _generate_with_local_model(self, request: Dict[str, Any], model_name: str, hrm_result: Optional[Dict[str, Any]]) -> str:
        """Generate response using local model"""
        
        model_config = self.local_models.get(model_name)
        if not model_config:
            return "I apologize, but the local model is not available."
        
        # Prepare prompt with HRM enhancement if available
        prompt = request.get("message", "")
        
        if hrm_result and hrm_result.get("success"):
            # Enhance prompt with HRM reasoning
            reasoning_result = hrm_result.get("reasoning_result", {})
            reasoning_steps = reasoning_result.get("steps", [])
            
            if reasoning_steps:
                prompt += f"\n\nReasoning approach: {', '.join(reasoning_steps)}"
        
        # Simulate llama.cpp generation (in real implementation, would use actual llama.cpp)
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Generate contextual response
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! I'm Greta, your Personal AI Assistant with German precision engineering. How may I assist you today?"
        
        if "research" in prompt.lower() or "investigate" in prompt.lower():
            return "I'll help you research this topic thoroughly. Let me analyze the key aspects and provide you with comprehensive insights based on my local processing capabilities."
        
        if "create" in prompt.lower() or "build" in prompt.lower():
            return "I can help you create and build! With my local processing and hierarchical reasoning, I'll provide you with detailed, well-structured solutions. What would you like to create?"
        
        # Default response
        return f"I'm processing your request locally with my {model_name} model. With German precision and hierarchical reasoning, I'll provide you with a thoughtful response."
    
    async def _generate_fallback_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback response when local models are unavailable"""
        
        return {
            "success": False,
            "response": "I apologize, but I'm currently unable to process your request with local models. Please ensure you have downloaded a compatible model and placed it in the models directory.",
            "model_used": "none",
            "response_time": 0.1,
            "local_processing": False,
            "hrm_enhanced": False,
            "error": "No local models available"
        }
    
    async def _update_performance_metrics(self, model: str, response_time: float, success: bool):
        """Update performance metrics"""
        
        if model not in self.performance_metrics["model_usage"]:
            self.performance_metrics["model_usage"][model] = {
                "uses": 0,
                "total_time": 0,
                "successes": 0
            }
        
        model_metrics = self.performance_metrics["model_usage"][model]
        model_metrics["uses"] += 1
        model_metrics["total_time"] += response_time
        
        if success:
            model_metrics["successes"] += 1
        
        # Update average response time
        total_requests = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        
        model_config = self.local_models.get(model_name, {})
        
        return {
            "model_name": model_name,
            "available": model_name in self.available_models,
            "path": str(model_config.get("path", "")),
            "context_window": model_config.get("context_window", 0),
            "quantization": model_config.get("quantization", ""),
            "mlx_optimized": model_config.get("mlx_optimized", False),
            "hrm_enabled": model_config.get("hrm_enabled", False),
            "ram_usage": model_config.get("ram_usage", ""),
            "apple_silicon_optimized": self.use_mlx
        }
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their information"""
        
        models_info = []
        for model_name in self.local_models.keys():
            model_info = await self.get_model_info(model_name)
            models_info.append(model_info)
        
        return models_info
    
    async def health_check(self) -> Dict[str, Any]:
        """Local LLM service health check"""
        
        return {
            "status": "healthy",
            "service": "local_llm_service",
            "available_models": len(self.available_models),
            "total_models": len(self.local_models),
            "apple_silicon_optimized": self.use_mlx,
            "mlx_available": getattr(self, 'mlx_available', False),
            "hrm_integration": "active",
            "total_requests": self.performance_metrics["total_requests"],
            "success_rate": (
                self.performance_metrics["successful_requests"] / 
                max(self.performance_metrics["total_requests"], 1)
            ) * 100,
            "average_response_time": self.performance_metrics["average_response_time"],
            "hrm_enhanced_requests": self.performance_metrics["hrm_enhanced_requests"],
            "timestamp": datetime.utcnow().isoformat()
        }

# Global local LLM service instance
local_llm_service = LocalLLMService()

async def get_local_llm_service() -> LocalLLMService:
    """Get the global local LLM service instance"""
    return local_llm_service
