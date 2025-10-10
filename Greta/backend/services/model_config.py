"""
Model Configuration System
Week 2 - Component 3: Model Configuration Management

Manages model-specific configurations with YAML storage and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from loguru import logger


class ModelConfig(BaseModel):
    """Model configuration schema"""
    name: str
    context_window: int = 4096
    n_gpu_layers: int = 35
    n_threads: int = 8
    use_mlock: bool = True
    use_mmap: bool = True
    
    # Generation defaults
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2000
    
    # Stop sequences
    stop_sequences: list[str] = Field(default_factory=list)
    
    # System prompt
    system_prompt: Optional[str] = None
    
    # Chat template
    chat_template: Optional[str] = None
    
    # Performance settings
    batch_size: int = 512
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    
    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Top-p must be between 0.0 and 1.0')
        return v


class ModelConfigManager:
    """
    Model Configuration Manager
    
    Manages model-specific configurations with YAML storage.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or os.getenv('CPAS4_DATA', Path.home() / '.cpas4' / 'configs'))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.configs: Dict[str, ModelConfig] = {}
        
        # Default config
        self.default_config = ModelConfig(name="default")
        
        logger.info(f"Model Config Manager initialized")
        logger.info(f"Config directory: {self.config_dir}")
    
    async def initialize(self):
        """Load all configurations"""
        await self._load_all_configs()
        logger.info(f"Loaded {len(self.configs)} model configurations")
    
    async def _load_all_configs(self):
        """Load all YAML configs from directory"""
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
                    config = ModelConfig(**data)
                    self.configs[config.name] = config
                    logger.debug(f"Loaded config: {config.name}")
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
    
    async def get_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a model (returns default if not found)"""
        if model_name in self.configs:
            return self.configs[model_name]
        
        logger.debug(f"No config found for {model_name}, using default")
        return self.default_config
    
    async def save_config(self, config: ModelConfig) -> bool:
        """Save model configuration"""
        try:
            # Add to cache
            self.configs[config.name] = config
            
            # Save to YAML
            config_file = self.config_dir / f"{config.name}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
            
            logger.success(f"Saved config: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config {config.name}: {e}")
            return False
    
    async def delete_config(self, model_name: str) -> bool:
        """Delete model configuration"""
        try:
            # Remove from cache
            if model_name in self.configs:
                del self.configs[model_name]
            
            # Delete file
            config_file = self.config_dir / f"{model_name}.yaml"
            if config_file.exists():
                config_file.unlink()
            
            logger.success(f"Deleted config: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete config {model_name}: {e}")
            return False
    
    async def create_default_configs(self):
        """Create default configurations for common models"""
        defaults = {
            "llama-2-7b": ModelConfig(
                name="llama-2-7b",
                context_window=4096,
                n_gpu_layers=35,
                temperature=0.7,
                system_prompt="You are a helpful AI assistant.",
                stop_sequences=["</s>", "[INST]", "[/INST]"]
            ),
            "mistral-7b": ModelConfig(
                name="mistral-7b",
                context_window=8192,
                n_gpu_layers=35,
                temperature=0.7,
                system_prompt="You are Mistral, a helpful AI assistant.",
                stop_sequences=["</s>"]
            ),
            "phi-2": ModelConfig(
                name="phi-2",
                context_window=2048,
                n_gpu_layers=32,
                temperature=0.8,
                system_prompt="You are Phi, a helpful AI assistant.",
                stop_sequences=["<|endoftext|>"]
            )
        }
        
        for name, config in defaults.items():
            await self.save_config(config)
        
        logger.info(f"Created {len(defaults)} default configurations")


# Global instance
model_config_manager = ModelConfigManager()
