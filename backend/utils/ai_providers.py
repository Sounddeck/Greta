"""
GRETA PAI - Multi-Vendor AI Architecture
Core PAI Feature: Intelligent AI provider routing
Supports Claude, GPT, Gemini, Ollama with automatic fallbacks
"""
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import asyncio
import json
import logging
from datetime import datetime
import os
import time

# AI Provider Imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from utils.error_handling import GretaException, handle_errors, metrics_collector
from utils.performance import performance_monitor

logger = logging.getLogger(__name__)


class AIProviderError(GretaException):
    """AI provider-related errors"""


class AIProviderBase(ABC):
    """Base class for AI providers"""

    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.connected = False
        self.last_used = None
        self.usage_count = 0
        self.error_count = 0
        self.total_tokens = 0

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider connection"""
        pass

    @abstractmethod
    async def run_pattern(self, system_prompt: str, user_prompt: str, pattern_name: str) -> str:
        """Run a PAI pattern with this provider"""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available"""
        pass

    @abstractmethod
    async def get_token_count(self, text: str) -> int:
        """Estimate token count for text"""
        pass

    def update_stats(self, tokens_used: int, success: bool):
        """Update provider statistics"""
        self.usage_count += 1
        self.total_tokens += tokens_used
        self.last_used = datetime.utcnow()

        if not success:
            self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            'name': self.name,
            'connected': self.connected,
            'usage_count': self.usage_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.usage_count),
            'total_tokens': self.total_tokens,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


class ClaudeProvider(AIProviderBase):
    """Anthropic Claude provider"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("claude", api_key)
        self.client = None
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    async def initialize(self) -> bool:
        """Initialize Claude client"""
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic library not available")
            return False

        api_key = self.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("No Claude API key provided")
            return False

        try:
            self.client = anthropic.Anthropic(api_key=api_key)

            # Test connection with minimal request
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )
            )

            self.connected = True
            logger.info("✅ Claude provider initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
            return False

    async def is_available(self) -> bool:
        """Check if Claude is available"""
        return self.connected and self.client is not None

    async def run_pattern(self, system_prompt: str, user_prompt: str, pattern_name: str) -> str:
        """Run PAI pattern with Claude"""
        if not self.is_available():
            raise AIProviderError("Claude provider not available")

        try:
            # Estimate input tokens
            input_tokens = await self.get_token_count(system_prompt + user_prompt)

            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-sonnet-20240229",  # PAI's preferred Claude model
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
            )

            execution_time = time.time() - start_time
            output_tokens = await self.get_token_count(response.content[0].text)

            # Update statistics
            self.update_stats(input_tokens + output_tokens, True)

            logger.info(f"🎯 Claude executed pattern '{pattern_name}' in {execution_time:.2f}s")
            return response.content[0].text

        except Exception as e:
            self.update_stats(0, False)
            raise AIProviderError(f"Claude execution failed: {str(e)}")

    async def get_token_count(self, text: str) -> int:
        """Estimate tokens for Claude"""
        # Rough estimation: ~4 chars per token
        return len(text) // 4


class OpenAIProvider(AIProviderBase):
    """OpenAI GPT provider"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("gpt", api_key)
        self.client = None
        if OPENAI_AVAILABLE and api_key:
            self.client = openai.OpenAI(api_key=api_key)

    async def initialize(self) -> bool:
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available")
            return False

        api_key = self.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("No OpenAI API key provided")
            return False

        try:
            self.client = openai.OpenAI(api_key=api_key)

            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10
                )
            )

            self.connected = True
            logger.info("✅ GPT provider initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False

    async def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self.connected and self.client is not None

    async def run_pattern(self, system_prompt: str, user_prompt: str, pattern_name: str) -> str:
        """Run PAI pattern with OpenAI"""
        if not self.is_available():
            raise AIProviderError("OpenAI provider not available")

        try:
            input_tokens = await self.get_token_count(system_prompt + user_prompt)

            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",  # PAI's preferred GPT model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4000
                )
            )

            execution_time = time.time() - start_time
            output_tokens = await self.get_token_count(response.choices[0].message.content)

            self.update_stats(input_tokens + output_tokens, True)

            logger.info(f"🧠 GPT executed pattern '{pattern_name}' in {execution_time:.2f}s")
            return response.choices[0].message.content

        except Exception as e:
            self.update_stats(0, False)
            raise AIProviderError(f"OpenAI execution failed: {str(e)}")

    async def get_token_count(self, text: str) -> int:
        """Estimate tokens for GPT (more accurate than Claude)"""
        # Rough GPT token estimation
        return len(text) // 4


class GeminiProvider(AIProviderBase):
    """Google Gemini provider"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("gemini", api_key)

    async def initialize(self) -> bool:
        """Initialize Gemini client"""
        if not GOOGLE_AVAILABLE:
            logger.warning("Google Generative AI library not available")
            return False

        api_key = self.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("No Google API key provided")
            return False

        try:
            genai.configure(api_key=api_key)

            # Test connection
            model = genai.GenerativeModel('gemini-pro')
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content("Hi", generation_config={"max_output_tokens": 10})
            )

            self.connected = True
            logger.info("✅ Gemini provider initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False

    async def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self.connected

    async def run_pattern(self, system_prompt: str, user_prompt: str, pattern_name: str) -> str:
        """Run PAI pattern with Gemini"""
        if not self.is_available():
            raise AIProviderError("Gemini provider not available")

        try:
            # Combine system and user prompts for Gemini
            combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            input_tokens = await self.get_token_count(combined_prompt)
            start_time = time.time()

            model = genai.GenerativeModel('gemini-1.5-flash')  # PAI's preferred Gemini model
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    combined_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 4000
                    }
                )
            )

            execution_time = time.time() - start_time
            output_tokens = await self.get_token_count(response.text)

            self.update_stats(input_tokens + output_tokens, True)

            logger.info(f"🌟 Gemini executed pattern '{pattern_name}' in {execution_time:.2f}s")
            return response.text

        except Exception as e:
            self.update_stats(0, False)
            raise AIProviderError(f"Gemini execution failed: {str(e)}")

    async def get_token_count(self, text: str) -> int:
        """Estimate tokens for Gemini"""
        return len(text) // 4


class OllamaProvider(AIProviderBase):
    """Local Ollama provider for privacy-first operations"""

    def __init__(self, model_name: str = "llama2"):
        super().__init__("ollama")
        self.model_name = model_name
        self.base_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')

    async def initialize(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m['name'] for m in models]

                    if self.model_name in model_names or any(self.model_name in name for name in model_names):
                        self.connected = True
                        logger.info(f"✅ Ollama provider initialized with model '{self.model_name}'")
                        return True
                    else:
                        logger.warning(f"Model '{self.model_name}' not available in Ollama")
                else:
                    logger.warning("Ollama service not reachable")

        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {e}")

        return False

    async def is_available(self) -> bool:
        """Check if Ollama is available"""
        return self.connected

    async def run_pattern(self, system_prompt: str, user_prompt: str, pattern_name: str) -> str:
        """Run PAI pattern with local Ollama"""
        if not self.is_available():
            raise AIProviderError("Ollama provider not available")

        try:
            import httpx

            # Combine prompts for Ollama (no separate system prompt support)
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            input_tokens = await self.get_token_count(full_prompt)
            start_time = time.time()

            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                result = response.json()

            execution_time = time.time() - start_time
            output_tokens = await self.get_token_count(result.get('response', ''))

            self.update_stats(input_tokens + output_tokens, True)

            logger.info(f"🏠 Ollama executed pattern '{pattern_name}' in {execution_time:.2f}s")
            return result.get('response', '')

        except Exception as e:
            self.update_stats(0, False)
            raise AIProviderError(f"Ollama execution failed: {str(e)}")

    async def get_token_count(self, text: str) -> int:
        """Estimate tokens for Ollama"""
        return len(text) // 4


class AIOrchestrator:
    """
    PAI's Intelligent AI Provider Orchestrator
    Routes requests to the best AI provider based on task characteristics
    """

    def __init__(self):
        self.providers: Dict[str, AIProviderBase] = {}
        self.routing_rules = {
            # Model preference routing (from PAI patterns)
            'claude': 'claude',
            'gpt': 'gpt',
            'gemini': 'gemini',
            'ollama': 'ollama',

            # Task-based intelligent routing
            'creative_writing': 'claude',      # Best for creative tasks
            'analytical': 'gpt',              # Best for analysis
            'coding': 'claude',              # Best for programming
            'research': 'claude',            # Best for complex research
            'business': 'gpt',               # Best for business logic
            'personal': 'ollama',            # Privacy-first for personal
            'fast': 'ollama',               # Local model for speed
            'complex': 'claude',            # Complex reasoning
        }

        self.fallback_order = ['ollama', 'claude', 'gpt', 'gemini']

        logger.info("🎭 AI Orchestrator initialized")

    async def initialize_providers(self) -> bool:
        """Initialize all available AI providers"""
        success_count = 0

        # Initialize Claude
        if ANTHROPIC_AVAILABLE:
            claude_provider = ClaudeProvider()
            if await claude_provider.initialize():
                self.providers['claude'] = claude_provider
                success_count += 1

        # Initialize OpenAI
        if OPENAI_AVAILABLE:
            gpt_provider = OpenAIProvider()
            if await gpt_provider.initialize():
                self.providers['gpt'] = gpt_provider
                success_count += 1

        # Initialize Gemini
        if GOOGLE_AVAILABLE:
            gemini_provider = GeminiProvider()
            if await gemini_provider.initialize():
                self.providers['gemini'] = gemini_provider
                success_count += 1

        # Initialize Ollama (local)
        ollama_provider = OllamaProvider()
        if await ollama_provider.initialize():
            self.providers['ollama'] = ollama_provider
            success_count += 1

        logger.info(f"📊 Initialized {success_count}/{4} AI providers")
        return success_count > 0

    def get_provider(self, preference: Optional[str] = None) -> AIProviderBase:
        """
        Get the best available provider for a task

        Args:
            preference: Preferred provider ('claude', 'gpt', 'gemini', 'ollama')
                       or intelligent routing keyword

        Returns:
            Best available provider for the request
        """
        # Direct model preference
        if preference in self.providers and self.providers[preference].is_available():
            return self.providers[preference]

        # Intelligent routing based on task type
        if preference in self.routing_rules:
            routed_provider = self.routing_rules[preference]
            if routed_provider in self.providers and self.providers[routed_provider].is_available():
                return self.providers[routed_provider]

        # Fallback to available providers in priority order
        for provider_name in self.fallback_order:
            if provider_name in self.providers and self.providers[provider_name].is_available():
                logger.info(f"🔄 Falling back to {provider_name} provider")
                return self.providers[provider_name]

        raise AIProviderError("No AI providers available")

    async def execute_with_routing(self, system_prompt: str, user_prompt: str,
                                 pattern_name: str, provider_preference: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a pattern with intelligent provider routing and error recovery

        Returns:
            Dict with result, provider used, and execution metadata
        """
        # Try preferred provider first
        try:
            provider = self.get_provider(provider_preference)
            result = await provider.run_pattern(system_prompt, user_prompt, pattern_name)

            return {
                'result': result,
                'provider': provider.name,
                'success': True,
                'fallback_used': False
            }

        except Exception as e:
            logger.warning(f"Primary provider failed for {pattern_name}: {e}")

            # Try fallback providers
            for fallback_name in self.fallback_order:
                if fallback_name != provider_preference and fallback_name in self.providers:
                    try:
                        fallback_provider = self.providers[fallback_name]
                        result = await fallback_provider.run_pattern(system_prompt, user_prompt, pattern_name)

                        logger.info(f"✅ Fallback to {fallback_name} succeeded for {pattern_name}")
                        return {
                            'result': result,
                            'provider': fallback_provider.name,
                            'success': True,
                            'fallback_used': True,
                            'original_error': str(e)
                        }

                    except Exception as fallback_error:
                        logger.warning(f"Fallback to {fallback_name} also failed: {fallback_error}")
                        continue

            # All providers failed
            raise AIProviderError(f"All AI providers failed for pattern '{pattern_name}'. Last error: {str(e)}")

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all providers"""
        stats = {}

        for name, provider in self.providers.items():
            stats[name] = provider.get_stats()

        stats['routing_rules'] = self.routing_rules.copy()
        stats['fallback_order'] = self.fallback_order.copy()

        return stats


# Global PAI AI Orchestrator instance
ai_orchestrator = AIOrchestrator()


class MCPBridge:
    """
    PAI's MCP (Multi-Client Protocol) Bridge
    Integrates external services like browser automation, payments, etc.
    """

    def __init__(self):
        self.services = {
            'web_browser': None,  # Playwright/Selenium
            'financial': None,    # Stripe/Square
            'communication': None,  # Twilio/SendGrid
            'analytics': None     # Google Analytics
        }

    async def initialize_services(self):
        """Initialize available MCP services"""
        # This would initialize external integrations
        # For now, just log availability
        logger.info("🔌 MCP Bridge initialized - ready for external integrations")

    async def execute_service(self, service_name: str, action: str, params: Dict) -> Any:
        """Execute action on MCP service"""
        # Placeholder for real MCP implementations
        logger.info(f"🔌 MCP: {service_name}.{action} with params {params}")
        return {"status": "not_implemented_yet", "service": service_name, "action": action}


# Global MCP Bridge
mcp_bridge = MCPBridge()


class AIFactory:
    """Factory for creating AI provider instances"""

    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> AIProviderBase:
        """Create AI provider instance"""
        if provider_type.lower() == 'claude':
            return ClaudeProvider(**kwargs)
        elif provider_type.lower() == 'gpt':
            return OpenAIProvider(**kwargs)
        elif provider_type.lower() == 'gemini':
            return GeminiProvider(**kwargs)
        elif provider_type.lower() == 'ollama':
            return OllamaProvider(**kwargs)
        else:
            raise AIProviderError(f"Unknown provider type: {provider_type}")


__all__ = [
    'AIProviderBase',
    'ClaudeProvider',
    'OpenAIProvider',
    'GeminiProvider',
    'OllamaProvider',
    'AIOrchestrator',
    'MCPBridge',
    'AIFactory',
    'ai_orchestrator',
    'mcp_bridge'
]
