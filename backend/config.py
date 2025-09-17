"""Enhanced GRETA Configuration"""
from pydantic_settings import BaseSettings
from typing import Optional
class Settings(BaseSettings):
    # Database
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "greta_pai"
    
    # Security  
    jwt_secret: str = "greta-enhanced-jwt-secret-2024"
    encryption_key: Optional[str] = None
    
    # AI Services
    ollama_url: str = "http://localhost:11434"
    llamacpp_model_path: str = "./models/llama-2-7b-chat.gguf"
    
    # Voice
    voice_language: str = "en"
    voice_accent: str = "german"
    tts_engine: str = "pyttsx3"
    
    # Learning
    auto_fine_tune_threshold: int = 100
    learning_rate: float = 0.001
    memory_consolidation_interval: int = 3600
    
    # System
    debug: bool = True
    log_level: str = "INFO"
    max_agents: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False
