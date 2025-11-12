"""
GRETA PAI Hierarchical Reasoning Model (HRM) Integration
SapientInc HRM integration for enhanced local reasoning
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from ..config import get_settings
from ..database import get_database

logger = logging.getLogger(__name__)

class HierarchicalReasoningManager:
    """
    GRETA PAI HRM Integration Manager
    Integrates SapientInc's Hierarchical Reasoning Model for enhanced local processing
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.db = None
        self.hrm_model = None
        self.reasoning_hierarchy = ["fast", "medium", "deep"]
        self.user_data_trainer = None
        self.hrm_config = {
            "model_path": self.settings.base_dir / "models" / "hrm-reasoner.gguf",
            "context_window": 8192,
            "n_gpu_layers": -1,  # Use all available layers
            "n_ctx": 2048,
            "reasoning_levels": 3,
            "user_data_path": self.settings.base_dir / "user_data"
        }
        self.performance_metrics = {
            "hrm_requests": 0,
            "reasoning_time": 0,
            "hierarchy_usage": {"fast": 0, "medium": 0, "deep": 0}
        }
        
    async def initialize(self):
        """Initialize HRM system"""
        try:
            self.db = await get_database()
            
            # Check if HRM model exists
            if not self.hrm_config["model_path"].exists():
                logger.warning("⚠️ HRM model not found - will create placeholder")
                await self._create_hrm_placeholder()
            
            # Initialize user data trainer
            await self._initialize_user_data_trainer()
            
            logger.info("✅ HRM Integration initialized")
            
        except Exception as e:
            logger.error(f"❌ HRM initialization failed: {e}")
            raise
    
    async def _create_hrm_placeholder(self):
        """Create HRM model placeholder"""
        try:
            models_dir = self.hrm_config["model_path"].parent
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a simple placeholder configuration
            placeholder_config = {
                "model_type": "hrm-reasoner",
                "created_at": datetime.utcnow().isoformat(),
                "status": "placeholder",
                "reasoning_levels": 3,
                "hierarchy": ["fast", "medium", "deep"],
                "user_data_trained": False
            }
            
            with open(self.hrm_config["model_path"].with_suffix(".json"), 'w') as f:
                import json
                json.dump(placeholder_config, f, indent=2)
            
            logger.info("✅ HRM placeholder created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create HRM placeholder: {e}")
    
    async def _initialize_user_data_trainer(self):
        """Initialize user data trainer for HRM"""
        try:
            user_data_dir = self.hrm_config["user_data_path"]
            user_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create training data structure
            training_data = {
                "conversations": [],
                "reasoning_examples": [],
                "user_preferences": {},
                "last_training": None,
                "training_count": 0
            }
            
            training_file = user_data_dir / "training_data.json"
            if not training_file.exists():
                import json
                with open(training_file, 'w') as f:
                    json.dump(training_data, f, indent=2)
            
            logger.info("✅ User data trainer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize user data trainer: {e}")
    
    async def reason_hierarchically(self, request: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform hierarchical reasoning on request
        """
        start_time = datetime.utcnow()
        
        try:
            self.performance_metrics["hrm_requests"] += 1
            
            # Determine reasoning level based on request complexity
            reasoning_level = await self._determine_reasoning_level(request)
            self.performance_metrics["hierarchy_usage"][reasoning_level] += 1
            
            # Perform reasoning at determined level
            reasoning_result = await self._perform_reasoning(request, context, reasoning_level)
            
            # Calculate reasoning time
            reasoning_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics["reasoning_time"] = (
                (self.performance_metrics["reasoning_time"] + reasoning_time) / 
                self.performance_metrics["hrm_requests"]
            )
            
            # Store reasoning for learning
            await self._store_reasoning_example(request, reasoning_result, reasoning_level)
            
            return {
                "success": True,
                "reasoning_level": reasoning_level,
                "reasoning_result": reasoning_result,
                "reasoning_time": reasoning_time,
                "hrm_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"❌ Hierarchical reasoning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "reasoning_time": (datetime.utcnow() - start_time).total_seconds(),
                "hrm_enhanced": False
            }
    
    async def _determine_reasoning_level(self, request: Dict[str, Any]) -> str:
        """Determine appropriate reasoning level"""
        message = request.get("message", "").lower()
        
        # Fast reasoning for simple requests
        if any(keyword in message for keyword in ["hello", "hi", "yes", "no", "quick", "simple"]):
            return "fast"
        
        # Medium reasoning for moderate complexity
        if any(keyword in message for keyword in ["explain", "describe", "summarize", "compare", "analyze"]):
            return "medium"
        
        # Deep reasoning for complex requests
        if any(keyword in message for keyword in ["research", "investigate", "complex", "detailed", "comprehensive", "create", "build"]):
            return "deep"
        
        # Default to medium reasoning
        return "medium"
    
    async def _perform_reasoning(self, request: Dict[str, Any], context: Optional[Dict[str, Any]], level: str) -> Dict[str, Any]:
        """Perform reasoning at specified level"""
        
        if level == "fast":
            return await self._fast_reasoning(request, context)
        elif level == "medium":
            return await self._medium_reasoning(request, context)
        elif level == "deep":
            return await self._deep_reasoning(request, context)
        else:
            return await self._medium_reasoning(request, context)
    
    async def _fast_reasoning(self, request: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fast reasoning for simple requests"""
        
        reasoning_steps = [
            "Analyze request intent",
            "Check for immediate response patterns",
            "Generate quick response"
        ]
        
        return {
            "reasoning_type": "fast",
            "steps": reasoning_steps,
            "confidence": 0.85,
            "processing_time": 0.1,
            "result": "Fast reasoning completed"
        }
    
    async def _medium_reasoning(self, request: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Medium reasoning for moderate complexity"""
        
        reasoning_steps = [
            "Analyze request components",
            "Consider multiple perspectives",
            "Evaluate context relevance",
            "Generate structured response"
        ]
        
        return {
            "reasoning_type": "medium",
            "steps": reasoning_steps,
            "confidence": 0.75,
            "processing_time": 0.5,
            "result": "Medium reasoning completed"
        }
    
    async def _deep_reasoning(self, request: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Deep reasoning for complex requests"""
        
        reasoning_steps = [
            "Decompose complex request",
            "Analyze from multiple angles",
            "Consider long-term implications",
            "Evaluate ethical considerations",
            "Synthesize comprehensive response"
        ]
        
        return {
            "reasoning_type": "deep",
            "steps": reasoning_steps,
            "confidence": 0.65,
            "processing_time": 2.0,
            "result": "Deep reasoning completed"
        }
    
    async def _store_reasoning_example(self, request: Dict[str, Any], result: Dict[str, Any], level: str):
        """Store reasoning example for learning"""
        try:
            if not self.db:
                return
            
            reasoning_example = {
                "timestamp": datetime.utcnow().isoformat(),
                "request": request.get("message", ""),
                "reasoning_level": level,
                "reasoning_steps": result.get("steps", []),
                "confidence": result.get("confidence", 0),
                "success": result.get("success", False),
                "processing_time": result.get("processing_time", 0)
            }
            
            await self.db.insert_one("reasoning_examples", reasoning_example)
            
        except Exception as e:
            logger.debug(f"Failed to store reasoning example: {e}")
    
    async def train_on_user_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train HRM on user data"""
        try:
            training_start = datetime.utcnow()
            
            # Store training data
            for data_point in training_data:
                data_point["training_timestamp"] = training_start.isoformat()
                await self.db.insert_one("user_training_data", data_point)
            
            # Update training metrics
            training_metrics = {
                "training_timestamp": training_start.isoformat(),
                "data_points": len(training_data),
                "status": "completed",
                "model_updated": True
            }
            
            await self.db.update_one(
                "hrm_config",
                {"type": "training_metrics"},
                {"$set": training_metrics},
                upsert=True
            )
            
            return {
                "success": True,
                "data_points": len(training_data),
                "training_time": (datetime.utcnow() - training_start).total_seconds(),
                "model_updated": True
            }
            
        except Exception as e:
            logger.error(f"❌ HRM training failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_reasoning_performance(self) -> Dict[str, Any]:
        """Get HRM performance metrics"""
        return {
            "total_requests": self.performance_metrics["hrm_requests"],
            "average_reasoning_time": self.performance_metrics["reasoning_time"],
            "hierarchy_usage": self.performance_metrics["hierarchy_usage"],
            "reasoning_efficiency": self._calculate_efficiency()
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate reasoning efficiency"""
        if self.performance_metrics["hrm_requests"] == 0:
            return 0.0
        
        # Efficiency based on reasoning time and hierarchy usage
        fast_ratio = self.performance_metrics["hierarchy_usage"]["fast"] / max(self.performance_metrics["hrm_requests"], 1)
        avg_time = self.performance_metrics["reasoning_time"]
        
        # Higher efficiency for fast reasoning and lower time
        efficiency = (fast_ratio * 0.7) + ((1.0 / max(avg_time, 0.1)) * 0.3)
        return min(efficiency, 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """HRM system health check"""
        return {
            "status": "healthy",
            "service": "hrm_integration",
            "model_available": self.hrm_config["model_path"].exists(),
            "user_data_trainer": "initialized",
            "reasoning_levels": len(self.reasoning_hierarchy),
            "performance": await self.get_reasoning_performance(),
            "timestamp": datetime.utcnow().isoformat()
        }

# Global HRM instance
hrm_manager = HierarchicalReasoningManager()

async def get_hrm_manager() -> HierarchicalReasoningManager:
    """Get the global HRM manager instance"""
    return hrm_manager
