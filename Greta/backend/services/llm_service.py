"""
LLM Service - llama.cpp Integration
Week 1 Implementation
"""

from typing import Optional, Dict, Any, AsyncIterator
from pathlib import Path
import asyncio
from llama_cpp import Llama
from loguru import logger
import os


class LLMService:
    """Service for managing llama.cpp LLM inference"""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize LLM Service
        
        Args:
            models_dir: Directory containing .gguf model files
        """
        self.models_dir = Path(models_dir or os.getenv('CPAS4_MODELS', Path.home() / '.cpas4' / 'models'))
        self.loaded_models: Dict[str, Llama] = {}
        self.current_model: Optional[str] = None
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LLM Service initialized with models directory: {self.models_dir}")
    
    async def load_model(
        self,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 35,
        verbose: bool = False
    ) -> bool:
        """
        Load a model into memory
        
        Args:
            model_name: Name of the model file (without .gguf extension)
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (Metal on macOS)
            verbose: Enable verbose logging
            
        Returns:
            True if model loaded successfully
        """
        try:
            model_path = self.models_dir / f"{model_name}.gguf"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading model: {model_name} from {model_path}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            llm = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,  # Use Metal GPU on macOS
                    verbose=verbose,
                    n_threads=8,  # Optimize for multi-core
                    use_mlock=True,  # Keep model in RAM
                )
            )
            
            self.loaded_models[model_name] = llm
            self.current_model = model_name
            
            logger.success(f"Model {model_name} loaded successfully")
            logger.info(f"Context window: {n_ctx}, GPU layers: {n_gpu_layers}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model unloaded successfully
        """
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                
                if self.current_model == model_name:
                    self.current_model = None
                
                logger.info(f"Model {model_name} unloaded")
                return True
            else:
                logger.warning(f"Model {model_name} not loaded")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[list] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text completion
        
        Args:
            prompt: Input prompt
            model_name: Model to use (uses current model if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Penalty for repeating tokens
            stop: List of stop sequences
            stream: Enable streaming (not implemented yet)
            
        Returns:
            Generated text
        """
        try:
            # Use specified model or current model
            model_name = model_name or self.current_model
            
            if not model_name or model_name not in self.loaded_models:
                raise ValueError(f"Model {model_name} not loaded")
            
            llm = self.loaded_models[model_name]
            
            logger.info(f"Generating with model {model_name}, max_tokens={max_tokens}, temp={temperature}")
            
            # Generate in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    echo=False
                )
            )
            
            generated_text = response['choices'][0]['text']
            
            logger.success(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming (for future implementation)
        
        Args:
            prompt: Input prompt
            model_name: Model to use
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        # TODO: Implement streaming in Week 2
        raise NotImplementedError("Streaming not yet implemented")
    
    def list_models(self) -> list[str]:
        """
        List available models in models directory
        
        Returns:
            List of model names (without .gguf extension)
        """
        try:
            models = [f.stem for f in self.models_dir.glob("*.gguf")]
            logger.info(f"Found {len(models)} models: {models}")
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def list_loaded_models(self) -> list[str]:
        """
        List currently loaded models
        
        Returns:
            List of loaded model names
        """
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dict or None
        """
        try:
            model_path = self.models_dir / f"{model_name}.gguf"
            
            if not model_path.exists():
                return None
            
            file_size = model_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            is_loaded = model_name in self.loaded_models
            
            return {
                "name": model_name,
                "path": str(model_path),
                "size_bytes": file_size,
                "size_mb": round(file_size_mb, 2),
                "is_loaded": is_loaded,
                "is_current": model_name == self.current_model
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None


# Global instance
llm_service = LLMService()
