"""
GRETA Local LLM Service with llama.cpp Integration
Provides local AI processing with continuous learning capabilities
"""

import asyncio
import logging
import json
import os
import threading
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path
import hashlib

# llama.cpp Python bindings
try:
    from llama_cpp import Llama, LlamaGrammar
    from llama_cpp.llama_tokenizer import LlamaHFTokenizer
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logging.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    LLAMA_CPP_AVAILABLE = False

# Additional dependencies for model management
try:
    import requests
    import huggingface_hub
    DOWNLOAD_AVAILABLE = True
except ImportError:
    DOWNLOAD_AVAILABLE = False

logger = logging.getLogger(__name__)

class LocalLLMConfig:
    """Configuration for local LLM service"""
    
    def __init__(self):
        self.models_dir = Path("models/local_llm")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Default model configurations with hierarchical reasoning
        self.default_models = {
            "llama3-8b": {
                "filename": "llama-3-8b-instruct.Q4_K_M.gguf",
                "url": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
                "size": "4.8GB",
                "context_length": 8192,
                "description": "Llama 3 8B Instruct - Advanced reasoning with HRM integration",
                "reasoning_capable": True,
                "training_data_compatible": True
            },
            "llama3-70b": {
                "filename": "llama-3-70b-instruct.Q4_K_M.gguf",
                "url": "https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
                "size": "40GB",
                "context_length": 8192,
                "description": "Llama 3 70B Instruct - Maximum reasoning capability",
                "reasoning_capable": True,
                "training_data_compatible": True
            },
            "phi3-mini": {
                "filename": "phi-3-mini-4k-instruct-q4.gguf", 
                "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
                "size": "2.4GB",
                "context_length": 4096,
                "description": "Phi-3 Mini - Fast and efficient for most tasks"
            },
            "codellama-7b": {
                "filename": "codellama-7b-instruct.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf",
                "size": "4.1GB",
                "context_length": 4096,
                "description": "Code Llama 7B - Specialized for code generation and analysis",
                "reasoning_capable": True,
                "coding_specialized": True
            },
            "deepseek-coder-6.7b": {
                "filename": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
                "size": "4.0GB",
                "context_length": 16384,
                "description": "DeepSeek Coder 6.7B - Advanced coding with long context",
                "reasoning_capable": True,
                "coding_specialized": True
            }
        }
        
        # Runtime configuration
        self.gpu_layers = self._detect_gpu_layers()

        # Hierarchical Reasoning Model (HRM) configuration
        self.hrm_config = {
            "reasoning_levels": ["surface", "intermediate", "deep", "meta"],
            "reasoning_strategies": ["analytical", "creative", "logical", "intuitive"],
            "context_integration": True,
            "learning_enabled": True,
            "user_data_training": True
        }

        # Learning and adaptation
        self.learning_data_dir = Path("learning_data")
        self.learning_data_dir.mkdir(exist_ok=True)

        self.user_interactions = []
        self.model_performance_metrics = {
            "accuracy": 0.0,
            "response_quality": 0.0,
            "user_satisfaction": 0.0,
            "learning_rate": 0.1
        }
        self.threads = os.cpu_count() or 4
        self.context_length = 4096
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 512
        
    def _detect_gpu_layers(self) -> int:
        """Detect optimal GPU layers based on system"""
        try:
            # Try to detect CUDA/Metal availability
            import torch
            if torch.cuda.is_available():
                return 35  # Use GPU acceleration
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 35  # Apple Silicon Metal
            else:
                return 0   # CPU only
        except ImportError:
            return 0

class LocalLLMService:
    """Local LLM service with llama.cpp integration"""
    
    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig()
        self.models: Dict[str, Llama] = {}
        self.active_model = None
        self.model_lock = threading.Lock()
        self.user_interactions = []
        self.learning_data = []
        
    async def initialize(self, model_name: str = "phi3-mini") -> bool:
        """Initialize the local LLM service"""
        try:
            if not LLAMA_CPP_AVAILABLE:
                logger.error("llama-cpp-python not available. Cannot initialize local LLM.")
                return False
            
            # Download model if needed
            model_path = await self._ensure_model_available(model_name)
            if not model_path:
                return False
            
            # Load model
            success = await self._load_model(model_name, model_path)
            if success:
                logger.info(f"✅ Local LLM initialized with model: {model_name}")
                return True
            else:
                logger.error(f"❌ Failed to load model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Local LLM initialization failed: {e}")
            return False
    
    async def _ensure_model_available(self, model_name: str) -> Optional[Path]:
        """Ensure model is downloaded and available"""
        if model_name not in self.config.default_models:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        model_info = self.config.default_models[model_name]
        model_path = self.config.models_dir / model_info["filename"]
        
        if model_path.exists():
            logger.info(f"Model {model_name} already available at {model_path}")
            return model_path
        
        if not DOWNLOAD_AVAILABLE:
            logger.error("Model download dependencies not available")
            return None
        
        # Download model
        logger.info(f"Downloading model {model_name} ({model_info['size']})...")
        try:
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                                logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info(f"✅ Model {model_name} downloaded successfully")
            return model_path
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            if model_path.exists():
                model_path.unlink()  # Clean up partial download
            return None
    
    async def _load_model(self, model_name: str, model_path: Path) -> bool:
        """Load model into memory"""
        try:
            with self.model_lock:
                logger.info(f"Loading model {model_name} from {model_path}")
                
                # Initialize Llama model
                model = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=self.config.gpu_layers,
                    n_threads=self.config.threads,
                    n_ctx=self.config.context_length,
                    verbose=False
                )
                
                self.models[model_name] = model
                self.active_model = model_name
                
                logger.info(f"✅ Model {model_name} loaded successfully")
                return True
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    async def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using local LLM"""
        if not self.active_model or self.active_model not in self.models:
            return {
                'success': False,
                'error': 'No active model available',
                'text': ''
            }
        
        try:
            model = self.models[self.active_model]
            
            # Merge generation parameters
            gen_params = {
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                'temperature': kwargs.get('temperature', self.config.temperature),
                'top_p': kwargs.get('top_p', self.config.top_p),
                'stop': kwargs.get('stop', []),
                'echo': False
            }
            
            # Generate text
            start_time = datetime.now()
            
            with self.model_lock:
                response = model(prompt, **gen_params)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Extract generated text
            generated_text = response['choices'][0]['text'] if response['choices'] else ''
            
            # Store interaction for learning
            await self._store_interaction(prompt, generated_text, generation_time)
            
            return {
                'success': True,
                'text': generated_text,
                'model': self.active_model,
                'generation_time': generation_time,
                'tokens_generated': len(generated_text.split()),
                'prompt_tokens': len(prompt.split())
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming response"""
        if not self.active_model or self.active_model not in self.models:
            yield json.dumps({'error': 'No active model available'})
            return
        
        try:
            model = self.models[self.active_model]
            
            gen_params = {
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                'temperature': kwargs.get('temperature', self.config.temperature),
                'top_p': kwargs.get('top_p', self.config.top_p),
                'stop': kwargs.get('stop', []),
                'stream': True
            }
            
            full_response = ""
            
            with self.model_lock:
                for output in model(prompt, **gen_params):
                    if 'choices' in output and output['choices']:
                        token = output['choices'][0].get('text', '')
                        if token:
                            full_response += token
                            yield json.dumps({
                                'token': token,
                                'partial_text': full_response
                            })
            
            # Store complete interaction
            await self._store_interaction(prompt, full_response, 0.0)
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield json.dumps({'error': str(e)})
    
    async def _store_interaction(self, prompt: str, response: str, generation_time: float):
        """Store user interaction for continuous learning"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response,
            'model': self.active_model,
            'generation_time': generation_time,
            'prompt_hash': hashlib.md5(prompt.encode()).hexdigest()
        }
        
        self.user_interactions.append(interaction)
        
        # Keep only recent interactions (last 1000)
        if len(self.user_interactions) > 1000:
            self.user_interactions = self.user_interactions[-1000:]
        
        # Analyze for learning opportunities
        await self._analyze_for_learning(interaction)
    
    async def _analyze_for_learning(self, interaction: Dict[str, Any]):
        """Analyze interaction for learning opportunities"""
        # Simple learning heuristics
        prompt = interaction['prompt'].lower()
        
        # Identify patterns for improvement
        learning_signals = []
        
        if 'error' in interaction['response'].lower():
            learning_signals.append('error_response')
        
        if interaction['generation_time'] > 5.0:
            learning_signals.append('slow_generation')
        
        if len(interaction['response']) < 10:
            learning_signals.append('short_response')
        
        if learning_signals:
            self.learning_data.append({
                'interaction_id': interaction['prompt_hash'],
                'signals': learning_signals,
                'timestamp': interaction['timestamp']
            })
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and statistics"""
        return {
            'active_model': self.active_model,
            'available_models': list(self.models.keys()),
            'total_interactions': len(self.user_interactions),
            'learning_data_points': len(self.learning_data),
            'gpu_layers': self.config.gpu_layers,
            'context_length': self.config.context_length,
            'models_directory': str(self.config.models_dir)
        }
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        if model_name in self.models:
            self.active_model = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
        else:
            # Try to load the model
            success = await self.initialize(model_name)
            return success

# Global service instance
local_llm_service = LocalLLMService()

# Async initialization function
async def initialize_local_llm(model_name: str = "phi3-mini") -> bool:
    """Initialize the global local LLM service"""
    return await local_llm_service.initialize(model_name)
