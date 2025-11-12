"""
GRETA PAI Configuration Settings
Core configuration management for the Personal AI Infrastructure

This module now supports secure secrets management via:
- HashiCorp Vault (recommended for production)
- AWS Secrets Manager (for AWS deployments)
- Environment variables (fallback for development)
"""

import os
import asyncio
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the parent directory (project root)
try:
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

# Import secrets manager
try:
    from services.secrets_manager import get_secrets_manager
    SECRETS_MANAGER_AVAILABLE = True
except ImportError:
    SECRETS_MANAGER_AVAILABLE = False
    print("Warning: Secrets manager not available, using environment variables only")

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

        # Structured Logging Configuration (Phase 3)
        self.log_format = os.getenv("LOG_FORMAT", "json")  # json or text
        self.log_file_path = os.getenv("LOG_FILE_PATH", "logs/cpas5.log")
        self.log_rotation = os.getenv("LOG_ROTATION", "1 day")
        self.log_retention = os.getenv("LOG_RETENTION", "30 days")
        self.log_compression = os.getenv("LOG_COMPRESSION", "zip")
        self.log_enable_console = os.getenv("LOG_ENABLE_CONSOLE", "true").lower() == "true"
        self.log_enable_file = os.getenv("LOG_ENABLE_FILE", "true").lower() == "true"
        self.slow_request_threshold_ms = float(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "1000"))
        
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

        # Wake Word Detection (openWakeWord - No API keys required!)
        # Migrated from Porcupine to openWakeWord (Oct 20, 2025)
        self.wake_word_enabled = os.getenv("WAKE_WORD_ENABLED", "true").lower() == "true"
        self.wake_word_model_paths = os.getenv("WAKE_WORD_MODEL_PATHS", "hey_jarvis_v0.1").split(",")
        self.wake_word_detection_threshold = float(os.getenv("WAKE_WORD_DETECTION_THRESHOLD", "0.5"))
        self.wake_word_enable_vad = os.getenv("WAKE_WORD_ENABLE_VAD", "true").lower() == "true"
        self.wake_word_vad_threshold = float(os.getenv("WAKE_WORD_VAD_THRESHOLD", "0.5"))
        self.wake_word_enable_noise_suppression = os.getenv("WAKE_WORD_ENABLE_NOISE_SUPPRESSION", "false").lower() == "true"
        
        # Learning Configuration
        self.auto_fine_tune_threshold = int(os.getenv("AUTO_FINE_TUNE_THRESHOLD", "100"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        self.memory_consolidation_interval = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "3600"))
        
        # System Configuration
        self.max_agents = int(os.getenv("MAX_AGENTS", "10"))
        self.max_memory_usage = int(os.getenv("MAX_MEMORY_USAGE", "80"))
        self.enable_performance_monitoring = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
        
        # Security Configuration
        # CRITICAL: Secrets should be loaded from secrets manager in production
        # For development, environment variables are acceptable

        # Check if we should use secrets manager
        self.use_secrets_manager = os.getenv("SECRETS_BACKEND", "env") != "env"

        if self.use_secrets_manager and SECRETS_MANAGER_AVAILABLE:
            # Secrets will be loaded asynchronously via load_secrets_async()
            # For now, set placeholders
            self.jwt_secret = None
            self.jwt_refresh_secret = None
            self.encryption_key = None
            self.api_key = None
            print("ℹ️  Secrets will be loaded from secrets manager")
        else:
            # Load from environment variables (development mode)
            self.jwt_secret = os.getenv("JWT_SECRET")
            if not self.jwt_secret:
                raise ValueError(
                    "JWT_SECRET environment variable is required! "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )

            self.jwt_refresh_secret = os.getenv("JWT_REFRESH_SECRET")
            if not self.jwt_refresh_secret:
                raise ValueError(
                    "JWT_REFRESH_SECRET environment variable is required! "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )

            self.encryption_key = os.getenv("ENCRYPTION_KEY")
            self.api_key = os.getenv("API_KEY")

        # Security Features
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.enable_https_redirect = os.getenv("ENABLE_HTTPS_REDIRECT", "false").lower() == "true"
        self.enable_input_validation = os.getenv("ENABLE_INPUT_VALIDATION", "true").lower() == "true"
        self.enable_security_headers = os.getenv("ENABLE_SECURITY_HEADERS", "true").lower() == "true"

        # Password Policy
        self.password_min_length = int(os.getenv("PASSWORD_MIN_LENGTH", "12"))
        self.password_require_uppercase = os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true"
        self.password_require_lowercase = os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true"
        self.password_require_numbers = os.getenv("PASSWORD_REQUIRE_NUMBERS", "true").lower() == "true"
        self.password_require_special = os.getenv("PASSWORD_REQUIRE_SPECIAL", "true").lower() == "true"
        self.password_check_common = os.getenv("PASSWORD_CHECK_COMMON", "true").lower() == "true"

        # Session Management
        self.session_timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
        self.refresh_token_expiry_days = int(os.getenv("REFRESH_TOKEN_EXPIRY_DAYS", "7"))
        self.access_token_expiry_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRY_MINUTES", "15"))
        self.access_token_expire_minutes = self.access_token_expiry_minutes  # Alias for compatibility

        # JWT Configuration
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")

        # CORS Configuration
        cors_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
        if cors_origins_str:
            self.cors_allowed_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        else:
            # Default for development
            self.cors_allowed_origins = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:5174",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:5174",
            ]

        # IP Filtering
        ip_whitelist_str = os.getenv("IP_WHITELIST", "")
        self.ip_whitelist = [ip.strip() for ip in ip_whitelist_str.split(",") if ip.strip()]

        ip_blacklist_str = os.getenv("IP_BLACKLIST", "")
        self.ip_blacklist = [ip.strip() for ip in ip_blacklist_str.split(",") if ip.strip()]

        # DDoS Protection
        self.max_connections_per_ip = int(os.getenv("MAX_CONNECTIONS_PER_IP", "100"))
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

        # API Keys for External Services
        self.apollo_io_api_key = os.getenv("APOLLO_IO_API_KEY")
        self.clearbit_api_key = os.getenv("CLEARBIT_API_KEY")
        self.hunter_io_api_key = os.getenv("HUNTER_IO_API_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")

        # Google Workspace OAuth
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.google_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/oauth/callback/google")
        self.google_mock_mode = os.getenv("GOOGLE_MOCK_MODE", "true").lower() == "true"
        
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
        
        # CORS Configuration - Use cors_allowed_origins for consistency
        # Environment variable can override with comma-separated list
        cors_origins_env = os.getenv("CORS_ORIGINS", "")
        if cors_origins_env:
            try:
                import ast
                self.cors_origins = ast.literal_eval(cors_origins_env)
            except:
                # If not valid Python literal, treat as comma-separated string
                self.cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
        else:
            # Use the same origins as cors_allowed_origins for consistency
            self.cors_origins = self.cors_allowed_origins

        # Master Agent Orchestration Feature Flags (Phase 2.1)
        # Allow toggling orchestration frameworks on/off for safe rollback
        self.enable_autogen_orchestration = os.getenv("ENABLE_AUTOGEN", "true").lower() == "true"
        self.enable_crewai_orchestration = os.getenv("ENABLE_CREWAI", "true").lower() == "true"
        self.enable_smolagents_orchestration = os.getenv("ENABLE_SMOLAGENTS", "true").lower() == "true"
        self.enable_langgraph_orchestration = os.getenv("ENABLE_LANGGRAPH", "true").lower() == "true"

    async def load_secrets_async(self):
        """
        Load secrets from secrets manager asynchronously
        This should be called during application startup
        """
        if not self.use_secrets_manager or not SECRETS_MANAGER_AVAILABLE:
            return

        try:
            secrets_manager = get_secrets_manager()

            # Load JWT secrets
            self.jwt_secret = await secrets_manager.get_secret("cpas5/jwt_secret")
            if not self.jwt_secret:
                raise ValueError("JWT secret not found in secrets manager")

            self.jwt_refresh_secret = await secrets_manager.get_secret("cpas5/jwt_refresh_secret")
            if not self.jwt_refresh_secret:
                raise ValueError("JWT refresh secret not found in secrets manager")

            # Load encryption key
            self.encryption_key = await secrets_manager.get_secret("cpas5/encryption_key")

            # Load API key
            self.api_key = await secrets_manager.get_secret("cpas5/api_key")

            print("✅ Secrets loaded from secrets manager")

        except Exception as e:
            print(f"❌ Failed to load secrets from secrets manager: {e}")
            raise

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

async def init_settings_async():
    """
    Initialize settings asynchronously
    This should be called during application startup to load secrets
    """
    await settings.load_secrets_async()
    return settings
