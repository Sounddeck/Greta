"""
GRETA PAI Configuration Settings
Core configuration management for the Personal AI Infrastructure
"""

import os
from typing import Optional, List
from pathlib import Path

class Settings:
    """
    GRETA PAI Settings - Configuration management with PAI philosophy
    The whole system is smarter than individual components
    """
    
    def __init__(self):
        # Core System Settings
        self.app_name = "GRETA PAI"
        self.version = "2.2.0"
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Database Configuration
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.database_name = os.getenv("DATABASE_NAME", "greta_pai")
        
        # AI/ML Configuration
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.llamacpp_model_path = os.getenv("LLAMACPP_MODEL_PATH", "./models/llama-2-7b-chat.gguf")
        self.use_mlx = os.getenv("USE_MLX", "true").lower() == "true"
        self.apple_silicon = os.getenv("APPLE_SILICON", "true").lower() == "true"
        
        # Voice Configuration
        self.voice_language = os.getenv("VOICE_LANGUAGE", "en")
        self.voice_accent = os.getenv("VOICE_ACCENT", "german")
        self.tts_engine = os.getenv("TTS_ENGINE", "pyttsx3")
        
        # Learning Configuration
        self.auto_fine_tune_threshold = int(os.getenv("AUTO_FINE_TUNE_THRESHOLD", "100"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        self.memory_consolidation_interval = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "3600"))
        
        # System Configuration
        self.max_agents = int(os.getenv("MAX_AGENTS", "10"))
        self.max_memory_usage = int(os.getenv("MAX_MEMORY_USAGE", "80"))
        self.enable_performance_monitoring = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
        
        # Security Configuration
        self.jwt_secret = os.getenv("JWT_SECRET", "greta-enhanced-jwt-secret-2024")
        self.encryption_key = os.getenv("ENCRYPTION_KEY")
        self.api_key = os.getenv("API_KEY")
        
        # Web Server Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.reload = os.getenv("RELOAD", "true").lower() == "true"
        
        # PAI Configuration
        self.pai_dir = os.getenv("PAI_DIR")
        self.digital_assistant_name = os.getenv("DA", "Greta")
        self.da_color = os.getenv("DA_COLOR", "blue")
        
        # Experimental Features
        self.enable_experimental_features = os.getenv("ENABLE_EXPERIMENTAL_FEATURES", "true").lower() == "true"
        self.quantum_reasoning_enabled = os.getenv("QUANTUM_REASONING_ENABLED", "true").lower() == "true"
        self.emotional_intelligence_enabled = os.getenv("EMOTIONAL_INTELLIGENCE_ENABLED", "true").lower() == "true"
        self.nlp_communication_enabled = os.getenv("NLP_COMMUNICATION_ENABLED", "true").lower() == "true"
        self.enable_ai_training = os.getenv("ENABLE_AI_TRAINING", "true").lower() == "true"
        self.personalized_learning = os.getenv("PERSONALIZED_LEARNING", "true").lower() == "true"
        
        # Path Configuration
        self.pythonpath = os.getenv("PYTHONPATH")
        self.base_dir = Path(__file__).parent.parent
        
        # Set dynamic paths
        if not self.pai_dir:
            self.pai_dir = str(self.base_dir / "PAI_DIRECTORY")
        
        if not self.pythonpath:
            self.pythonpath = str(self.base_dir)
        
        # Apple Silicon detection
        import platform
        if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
            self.apple_silicon = True
        
        # Model path resolution
        if not os.path.exists(self.llamacpp_model_path):
            self.llamacpp_model_path = str(self.base_dir / "models" / "llama-3-8b-instruct.gguf")
        
        # CORS Configuration
        cors_origins_env = os.getenv("CORS_ORIGINS", "[\"*\"]")
        try:
            import ast
            self.cors_origins = ast.literal_eval(cors_origins_env)
        except:
            self.cors_origins = ["*"]

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def reload_settings() -> Settings:
    """Reload settings from environment"""
    global settings
    settings = Settings()
    return settings
