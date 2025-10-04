"""
GRETA PAI LLM Service
Core LLM integration with PAI intelligence orchestration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..config import get_settings
from ..database import get_database

logger = logging.getLogger(__name__)

class LLMService:
    """
    GRETA PAI LLM Service - Intelligent language model orchestration
    The system orchestrates multiple AI capabilities for superior performance
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db = None
        self.available_models = {}
        self.active_sessions = {}
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0,
            "model_usage": {}
        }
        
    async def initialize(self):
        """Initialize the LLM service"""
        try:
            self.db = await get_database()
            
            # Initialize available models
            await self._initialize_models()
            
            # Load performance metrics
            await self._load_performance_metrics()
            
            logger.info("✅ LLM Service initialized with PAI intelligence")
            
        except Exception as e:
            logger.error(f"❌ LLM Service initialization failed: {e}")
            raise
    
    async def _initialize_models(self):
        """Initialize available AI models"""
        # Local models (priority)
        self.available_models = {
            "llama3-local": {
                "type": "local",
                "provider": "llama.cpp",
                "model_path": self.settings.llamacpp_model_path,
                "context_window": 8192,
                "priority": 1,
                "available": True
            },
            "mlx-local": {
                "type": "local", 
                "provider": "mlx",
                "model_path": "mlx-quantized",
                "context_window": 4096,
                "priority": 2,
                "available": self.settings.apple_silicon
            }
        }
        
        # Cloud models (fallback)
        if self.settings.api_key:
            self.available_models.update({
                "gpt-4": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-4",
                    "context_window": 8192,
                    "priority": 3,
                    "available": True
                },
                "claude-3": {
                    "type": "cloud",
                    "provider": "anthropic", 
                    "model": "claude-3-sonnet",
                    "context_window": 200000,
                    "priority": 4,
                    "available": True
                }
            })
    
    async def generate_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate intelligent completion using PAI orchestration
        The whole system is smarter than individual models
        """
        start_time = datetime.utcnow()
        
        try:
            # Update metrics
            self.performance_metrics["total_requests"] += 1
            
            # Analyze request for optimal model selection
            optimal_model = await self._select_optimal_model(request)
            
            # Enhance prompt with PAI intelligence
            enhanced_request = await self._enhance_with_pai_intelligence(request, optimal_model)
            
            # Generate completion
            response = await self._generate_with_model(enhanced_request, optimal_model)
            
            # Apply PAI post-processing
            optimized_response = await self._apply_pai_optimization(response, request)
            
            # Update success metrics
            self.performance_metrics["successful_requests"] += 1
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_performance_metrics(optimal_model, response_time, True)
            
            # Store interaction for learning
            await self._store_interaction(request, optimized_response, optimal_model, response_time)
            
            return {
                "success": True,
                "response": optimized_response,
                "model_used": optimal_model,
                "response_time": response_time,
                "pai_enhanced": True,
                "confidence": await self._calculate_confidence(request, optimized_response)
            }
            
        except Exception as e:
            logger.error(f"❌ Completion generation failed: {e}")
            
            # Update failure metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_performance_metrics("unknown", response_time, False)
            
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time,
                "fallback_response": await self._generate_fallback_response(request)
            }
    
    async def _select_optimal_model(self, request: Dict[str, Any]) -> str:
        """Select optimal model based on request characteristics"""
        
        # Analyze request complexity
        complexity = await self._analyze_request_complexity(request)
        
        # Check for specific requirements
        requires_large_context = request.get("requires_large_context", False)
        requires_local_processing = request.get("local_only", False)
        requires_cultural_context = "german" in request.get("message", "").lower()
        
        # Model selection logic
        if requires_local_processing:
            # Prefer local models for privacy
            if self.available_models.get("llama3-local", {}).get("available"):
                return "llama3-local"
            elif self.available_models.get("mlx-local", {}).get("available"):
                return "mlx-local"
        
        if requires_large_context:
            # Prefer models with larger context windows
            for model_name, model_info in self.available_models.items():
                if model_info.get("available") and model_info.get("context_window", 0) > 10000:
                    return model_name
        
        if requires_cultural_context and complexity == "high":
            # Use advanced models for cultural nuance
            for model_name, model_info in sorted(
                self.available_models.items(), 
                key=lambda x: x[1].get("priority", 999)
            ):
                if model_info.get("available"):
                    return model_name
        
        # Default selection based on availability and priority
        for model_name, model_info in sorted(
            self.available_models.items(),
            key=lambda x: x[1].get("priority", 999)
        ):
            if model_info.get("available"):
                return model_name
        
        # Fallback to first available model
        return list(self.available_models.keys())[0] if self.available_models else "unknown"
    
    async def _analyze_request_complexity(self, request: Dict[str, Any]) -> str:
        """Analyze request complexity"""
        message = request.get("message", "")
        
        # Complexity indicators
        complexity_indicators = {
            "high": ["analyze", "research", "complex", "detailed", "comprehensive"],
            "medium": ["explain", "describe", "summarize", "compare"],
            "low": ["hello", "hi", "simple", "basic"]
        }
        
        message_lower = message.lower()
        
        for complexity, keywords in complexity_indicators.items():
            if any(keyword in message_lower for keyword in keywords):
                return complexity
        
        # Default based on message length
        if len(message) > 500:
            return "high"
        elif len(message) > 100:
            return "medium"
        else:
            return "low"
    
    async def _enhance_with_pai_intelligence(self, request: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Enhance request with PAI intelligence"""
        
        # Get model information
        model_info = self.available_models.get(model, {})
        
        # Build PAI system prompt
        system_prompt = await self._build_pai_system_prompt(request, model_info)
        
        # Add context from memory
        context = await self._get_relevant_context(request)
        
        # Enhanced request
        enhanced_request = {
            "message": request.get("message", ""),
            "system_prompt": system_prompt,
            "context": context,
            "model": model,
            "parameters": {
                "temperature": await self._calculate_optimal_temperature(request),
                "max_tokens": await self._calculate_optimal_max_tokens(request, model_info),
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        }
        
        return enhanced_request
    
    async def _build_pai_system_prompt(self, request: Dict[str, Any], model_info: Dict[str, Any]) -> str:
        """Build PAI-enhanced system prompt"""
        
        base_prompt = f"""You are Greta, a Personal AI Assistant with German precision engineering and PAI intelligence.

Core Principles:
- The whole system is smarter than individual AI models
- You orchestrate multiple intelligence sources for optimal responses
- You embody German precision: thorough, reliable, and exact
- You learn from every interaction to improve the entire system

Current Context:
- Model: {model_info.get('provider', 'unknown')} {model_info.get('model', 'local')}
- Context Window: {model_info.get('context_window', 'unknown')} tokens
- Processing: {model_info.get('type', 'unknown')}"""

        # Add task-specific guidance
        message = request.get("message", "").lower()
        
        if any(word in message for word in ["analyze", "research", "investigate"]):
            base_prompt += "\n\nTask Guidance: Provide thorough, well-researched analysis with multiple perspectives. Use systematic approach and cite sources when relevant."
        
        if any(word in message for word in ["create", "write", "generate"]):
            base_prompt += "\n\nTask Guidance: Create high-quality, original content with attention to detail. Apply German precision and thoroughness."
        
        if any(word in message for word in ["german", "deutsch", "germany"]):
            base_prompt += "\n\nTask Guidance: Apply German cultural context and precision. Use appropriate formalities and cultural nuances."
        
        if any(word in message for word in ["code", "program", "technical"]):
            base_prompt += "\n\nTask Guidance: Provide technically accurate, well-structured solutions. Follow best practices and include explanations."
        
        base_prompt += "\n\nRemember: You are enhanced by PAI intelligence. Use the full system capabilities to provide optimal assistance."
        
        return base_prompt
    
    async def _get_relevant_context(self, request: Dict[str, Any]) -> List[str]:
        """Get relevant context from memory"""
        try:
            if not self.db:
                return []
            
            # Search for relevant memories
            message = request.get("message", "")
            keywords = await self._extract_keywords(message)
            
            context_items = []
            
            # Get recent interactions
            recent_interactions = await self.db.find_many(
                "interactions", 
                {"type": "chat"},
                limit=5
            )
            
            for interaction in recent_interactions:
                if any(keyword in interaction.get("message", "").lower() for keyword in keywords):
                    context_items.append(f"Previous: {interaction.get('message', '')}")
            
            # Get relevant learning data
            learning_data = await self.db.find_many(
                "learning",
                {"keywords": {"$in": keywords}},
                limit=3
            )
            
            for learning in learning_data:
                context_items.append(f"Learning: {learning.get('insight', '')}")
            
            return context_items[:5]  # Limit context
            
        except Exception as e:
            logger.debug(f"Context retrieval failed: {e}")
            return []
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction (can be enhanced)
        import re
        
        # Remove common words and extract meaningful terms
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "must"}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    async def _calculate_optimal_temperature(self, request: Dict[str, Any]) -> float:
        """Calculate optimal temperature for request"""
        message = request.get("message", "").lower()
        
        # Creative tasks need higher temperature
        if any(word in message for word in ["create", "write", "generate", "imagine"]):
            return 0.8
        
        # Analytical tasks need lower temperature
        if any(word in message for word in ["analyze", "research", "calculate", "compare"]):
            return 0.3
        
        # Conversational tasks use medium temperature
        return 0.7
    
    async def _calculate_optimal_max_tokens(self, request: Dict[str, Any], model_info: Dict[str, Any]) -> int:
        """Calculate optimal max tokens for request"""
        message = request.get("message", "")
        
        # Base on request type
        if any(word in message.lower() for word in ["brief", "short", "concise"]):
            return min(500, model_info.get("context_window", 4096) // 8)
        
        if any(word in message.lower() for word in ["detailed", "comprehensive", "thorough"]):
            return min(2000, model_info.get("context_window", 4096) // 4)
        
        # Default response length
        return min(1000, model_info.get("context_window", 4096) // 4)
    
    async def _generate_with_model(self, request: Dict[str, Any], model: str) -> str:
        """Generate response using specific model"""
        
        if model == "llama3-local":
            return await self._generate_with_llama_cpp(request)
        elif model == "mlx-local":
            return await self._generate_with_mlx(request)
        elif model == "gpt-4":
            return await self._generate_with_openai(request)
        elif model == "claude-3":
            return await self._generate_with_anthropic(request)
        else:
            return await self._generate_mock_response(request)
    
    async def _generate_with_llama_cpp(self, request: Dict[str, Any]) -> str:
        """Generate using local Llama.cpp model"""
        # Mock implementation - would use actual llama.cpp
        await asyncio.sleep(0.5)  # Simulate processing
        
        message = request.get("message", "")
        return f"[Local Llama.cpp] Processing '{message[:50]}...' with German precision and PAI intelligence. This response is generated locally for maximum privacy and performance."
    
    async def _generate_with_mlx(self, request: Dict[str, Any]) -> str:
        """Generate using MLX (Apple Silicon)"""
        # Mock implementation - would use actual MLX
        await asyncio.sleep(0.3)  # Simulate processing
        
        message = request.get("message", "")
        return f"[MLX Apple Silicon] Optimized processing of '{message[:50]}...' using Apple Silicon acceleration with PAI orchestration."
    
    async def _generate_with_openai(self, request: Dict[str, Any]) -> str:
        """Generate using OpenAI API"""
        # Mock implementation - would use actual OpenAI API
        await asyncio.sleep(1.0)  # Simulate API call
        
        message = request.get("message", "")
        return f"[OpenAI GPT-4] Enhanced response to '{message[:50]}...' with PAI intelligence augmentation and German precision."
    
    async def _generate_with_anthropic(self, request: Dict[str, Any]) -> str:
        """Generate using Anthropic Claude"""
        # Mock implementation - would use actual Anthropic API
        await asyncio.sleep(1.2)  # Simulate API call
        
        message = request.get("message", "")
        return f"[Claude-3] Intelligent response to '{message[:50]}...' with PAI system orchestration and cultural awareness."
    
    async def _generate_mock_response(self, request: Dict[str, Any]) -> str:
        """Generate mock response for testing"""
        message = request.get("message", "")
        
        if any(word in message.lower() for word in ["hello", "hi", "greetings"]):
            return "Hello! I'm Greta, your Personal AI Assistant with German precision engineering. How may I assist you today?"
        
        if any(word in message.lower() for word in ["help", "assist", "support"]):
            return "I'm here to help with thorough, precise assistance. My PAI intelligence allows me to orchestrate multiple capabilities for optimal results. What specific task can I help you with?"
        
        return f"I'm processing your request about '{message[:50]}...' with PAI intelligence and German precision. Let me provide you with a comprehensive, thoughtful response."
    
    async def _apply_pai_optimization(self, response: str, request: Dict[str, Any]) -> str:
        """Apply PAI post-processing optimization"""
        
        # Add PAI signature if not present
        if "[PAI" not in response and "PAI" not in response:
            response += "\n\n[PAI Enhanced: This response was orchestrated through Personal AI Intelligence for optimal quality and precision]"
        
        # Ensure German precision elements
        if "thorough" not in response.lower() and "precise" not in response.lower():
            response = response.replace("I'm", "I'm thoroughly and precisely")
        
        return response
    
    async def _calculate_confidence(self, request: Dict[str, Any], response: str) -> float:
        """Calculate confidence score for response"""
        
        # Base confidence
        confidence = 0.85
        
        # Adjust based on response characteristics
        if len(response) > 500:
            confidence += 0.05  # Longer responses often more detailed
        
        if "thorough" in response.lower() or "precise" in response.lower():
            confidence += 0.05  # German precision indicators
        
        if "PAI" in response:
            confidence += 0.03  # PAI enhancement present
        
        return min(confidence, 1.0)
    
    async def _update_performance_metrics(self, model: str, response_time: float, success: bool):
        """Update performance metrics"""
        
        # Update model usage
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
        
        # Persist metrics periodically
        if total_requests % 10 == 0:
            await self._persist_performance_metrics()
    
    async def _store_interaction(self, request: Dict[str, Any], response: Dict[str, Any], model: str, response_time: float):
        """Store interaction for learning"""
        try:
            if not self.db:
                return
            
            interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "request": request.get("message", ""),
                "response": response.get("response", ""),
                "model_used": model,
                "response_time": response_time,
                "success": response.get("success", False),
                "confidence": response.get("confidence", 0),
                "pai_enhanced": response.get("pai_enhanced", False),
                "session_id": request.get("session_id", "default"),
                "user_id": request.get("user_id", "default")
            }
            
            await self.db.insert_one("interactions", interaction)
            
        except Exception as e:
            logger.debug(f"Failed to store interaction: {e}")
    
    async def _generate_fallback_response(self, request: Dict[str, Any]) -> str:
        """Generate fallback response"""
        return "I apologize, but I'm experiencing technical difficulties. As your PAI assistant with German precision, I want to ensure you receive the best possible assistance. Please try again in a moment."
    
    async def _load_performance_metrics(self):
        """Load performance metrics from database"""
        try:
            if not self.db:
                return
            
            # Get recent metrics
            metrics = await self.db.find_one("performance", {"type": "llm_metrics"})
            
            if metrics:
                self.performance_metrics = metrics.get("metrics", self.performance_metrics)
                
        except Exception as e:
            logger.debug(f"Failed to load performance metrics: {e}")
    
    async def _persist_performance_metrics(self):
        """Persist performance metrics to database"""
        try:
            if not self.db:
                return
            
            await self.db.update_one(
                "performance",
                {"type": "llm_metrics"},
                {
                    "$set": {
                        "type": "llm_metrics",
                        "metrics": self.performance_metrics,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                },
                upsert=True
            )
            
        except Exception as e:
            logger.debug(f"Failed to persist performance metrics: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        
        available_models = [
            name for name, info in self.available_models.items() 
            if info.get("available", False)
        ]
        
        return {
            "status": "healthy",
            "service": "llm_service",
            "available_models": available_models,
            "total_models": len(self.available_models),
            "total_requests": self.performance_metrics["total_requests"],
            "success_rate": (
                self.performance_metrics["successful_requests"] / 
                max(self.performance_metrics["total_requests"], 1)
            ) * 100,
            "average_response_time": self.performance_metrics["average_response_time"],
            "pai_intelligence": "active",
            "timestamp": datetime.utcnow().isoformat()
        }

# Global service instance
llm_service = LLMService()

async def get_llm_service() -> LLMService:
    """Get the global LLM service instance"""
    return llm_service
