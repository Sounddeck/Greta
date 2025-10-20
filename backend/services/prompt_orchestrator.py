"""
PAI Dynamic Prompt Orchestrator
Intelligent prompt management system that evolves Greta's personality, context, and strategy over time based on learning and feedback
"""

import asyncio
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re
import os

# Import CoT Prompt Enhancer
from services.cot_prompt_enhancer import CoTPromptEnhancer, TaskType

logger = logging.getLogger(__name__)


class PromptOrchestrator:
    """
    PAI Dynamic Prompt System - Makes prompt management intelligent and adaptable
    Enables Greta's personality and context to evolve based on learning and user feedback
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.master_prompts = {}        # Multiple personality modes/versions
        self.context_prompts = {}       # Dynamic context additions
        self.prompt_performance = {}    # Track which prompts work best
        self.prompt_evolution = deque(maxlen=50)  # Evolution tracking
        self.user_feedback_loop = {}    # Learn from user responses

        # Current active prompts
        self.active_master_prompt = None
        self.active_context_prompts = []

        # Evolution parameters
        self.feedback_threshold = 0.7   # Minimum effectiveness for prompt adaptation
        self.evolution_interval = 7     # Days between automatic evolutions

        # Initialize CoT Enhancer
        self.cot_enhancer = CoTPromptEnhancer()
        self.cot_enabled = True  # Enable CoT by default

        # Ensure prompts directory exists
        os.makedirs(prompts_dir, exist_ok=True)

        logger.info("🎯 Prompt Orchestrator initialized with CoT enhancement")

    async def generate_pai_prompt(
        self, 
        user_context: Dict[str, Any], 
        task_type: str,
        enable_cot: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete PAI prompt by intelligently combining master prompt, context, and learned adaptations
        Now includes Chain-of-Thought enhancement for improved reasoning
        
        Args:
            user_context: Context about the user and current interaction
            task_type: Type of task being performed
            enable_cot: Whether to apply CoT enhancement (default: True)
        
        Returns:
            Dictionary with complete prompt and metadata
        """

        # Select optimal master prompt for this context
        master_prompt = await self.select_master_prompt(user_context, task_type)

        # Select relevant context prompts
        context_additions = await self.select_context_prompts(user_context, task_type)

        # Generate dynamic adaptations based on learning
        adaptations = await self.generate_adaptations(user_context, task_type)

        # Combine into comprehensive PAI prompt
        complete_prompt = await self.combine_prompts(master_prompt, context_additions, adaptations)

        # Apply Chain-of-Thought enhancement if enabled
        cot_metadata = {}
        if enable_cot and self.cot_enabled:
            cot_result = self.cot_enhancer.enhance_with_cot(
                prompt=complete_prompt,
                task_type=self._map_task_type_to_cot(task_type)
            )
            
            if cot_result.get("success"):
                complete_prompt = cot_result["enhanced_prompt"]
                cot_metadata = {
                    "cot_applied": True,
                    "cot_task_type": cot_result["task_type"],
                    "cot_template": cot_result["cot_template_used"],
                    "enhancement_ratio": cot_result["metadata"]["enhancement_ratio"]
                }
                logger.info(f"✅ Applied CoT enhancement ({cot_result['task_type']})")
            else:
                cot_metadata = {"cot_applied": False, "cot_error": cot_result.get("error")}
                logger.warning(f"⚠️ CoT enhancement failed: {cot_result.get('error')}")
        else:
            cot_metadata = {"cot_applied": False, "reason": "disabled"}

        return {
            "prompt": complete_prompt,
            "metadata": {
                "master_version": master_prompt.get("name", "default"),
                "context_count": len(context_additions),
                "adaptations_count": len(adaptations),
                "task_type": task_type,
                **cot_metadata
            },
            "components": {
                "master": master_prompt,
                "contexts": context_additions,
                "adaptations": adaptations
            }
        }

    def _map_task_type_to_cot(self, task_type: str) -> Optional[TaskType]:
        """Map general task type to CoT TaskType enum"""
        task_type_mapping = {
            "reasoning": TaskType.REASONING,
            "analysis": TaskType.ANALYSIS,
            "problem_solving": TaskType.PROBLEM_SOLVING,
            "creative": TaskType.CREATIVE,
            "research": TaskType.RESEARCH,
            "coding": TaskType.CODING,
            "code": TaskType.CODING,
            "planning": TaskType.PLANNING,
            "decision": TaskType.DECISION_MAKING,
            "decision_making": TaskType.DECISION_MAKING,
            "general": TaskType.GENERAL
        }
        
        return task_type_mapping.get(task_type.lower(), None)

    async def select_master_prompt(self, user_context: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Select the optimal master prompt based on context and task type
        """
        # If we have master prompts loaded, select the best one
        if self.master_prompts:
            # Simple selection logic - can be enhanced with ML
            for name, prompt in self.master_prompts.items():
                if task_type in prompt.get("adapted_for", []):
                    return prompt
            
            # Return first available if no match
            return list(self.master_prompts.values())[0]
        
        # Default master prompt
        return {
            "name": "default_master",
            "content": "You are Greta, a highly capable Personal AI Assistant with German precision and intelligence.",
            "version": "1.0",
            "adapted_for": ["general"]
        }

    async def select_context_prompts(self, user_context: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """
        Select relevant context prompts to enhance the interaction
        """
        selected_contexts = []

        for context_name, context_data in self.context_prompts.items():
            if await self.is_context_relevant(context_data, user_context, task_type):
                selected_contexts.append(context_data)

                # Track usage for learning
                if context_name not in self.prompt_performance:
                    self.prompt_performance[context_name] = {"uses": 0, "effectiveness": []}
                self.prompt_performance[context_name]["uses"] += 1

        # Limit context additions to prevent prompt bloat
        selected_contexts = sorted(selected_contexts, key=lambda x: x.get("priority", 1), reverse=True)[:3]

        return selected_contexts

    async def is_context_relevant(self, context_data: Dict, user_context: Dict, task_type: str) -> bool:
        """Check if a context prompt is relevant to current situation"""
        # Simple relevance check - can be enhanced
        context_tags = set(context_data.get("tags", []))
        user_tags = set(user_context.get("tags", []))
        
        # Check for tag overlap
        if context_tags & user_tags:
            return True
        
        # Check if task type matches
        if task_type in context_data.get("applicable_tasks", []):
            return True
        
        return False

    async def generate_adaptations(self, user_context: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """
        Generate dynamic adaptations based on learned user preferences and system evolution
        """
        adaptations = []

        # User preference adaptations
        user_preferences = await self.generate_user_adaptations(user_context)
        if user_preferences:
            adaptations.extend(user_preferences)

        # Task-specific adaptations
        task_adaptations = await self.generate_task_adaptations(task_type, user_context)
        if task_adaptations:
            adaptations.extend(task_adaptations)

        # Time/contextual adaptations
        temporal_adaptations = await self.generate_temporal_adaptations(user_context)
        if temporal_adaptations:
            adaptations.extend(temporal_adaptations)

        return adaptations

    async def generate_user_adaptations(self, user_context: Dict) -> List[Dict]:
        """Generate adaptations based on user preferences"""
        # Placeholder - would analyze user history
        return []

    async def generate_task_adaptations(self, task_type: str, user_context: Dict) -> List[Dict]:
        """Generate task-specific adaptations"""
        # Placeholder - would provide task-specific guidance
        return []

    async def generate_temporal_adaptations(self, user_context: Dict) -> List[Dict]:
        """Generate time-based adaptations"""
        # Placeholder - would consider time of day, urgency, etc.
        return []

    async def combine_prompts(self, master: Dict[str, Any], contexts: List[Dict], adaptations: List[Dict]) -> str:
        """
        Intelligently combine all prompt components into a coherent, effective prompt
        """
        # Start with master prompt
        combined = [master["content"]]

        # Add context prompts in priority order
        for context in contexts:
            if context.get("position") == "before_master":
                combined.insert(0, context["content"])
            else:
                combined.append(context["content"])

        # Add adaptations
        for adaptation in adaptations:
            # Insert adaptations strategically
            if adaptation.get("type") == "instruction_enhancement":
                combined.append(adaptation["content"])
            elif adaptation.get("type") == "behavior_modification":
                combined.insert(0, adaptation["content"])  # Early in prompt

        # Add PAI system signature
        combined.append("\n--- PAI SYSTEM CONTEXT ---\nYou are operating within a Personal AI (PAI) system that is smarter than individual LLM capabilities. Leverage hierarchical reasoning, context awareness, and proactive intelligence to provide superior assistance.")

        return "\n\n".join(combined)

    async def process_user_feedback(self, prompt_used: Dict, user_response: Dict, interaction_quality: float):
        """
        Process user feedback to improve future prompt selection and adaptation
        """
        prompt_components = prompt_used.get("components", {})
        master_name = prompt_used.get("metadata", {}).get("master_version")

        if master_name and master_name in self.prompt_performance:
            self.prompt_performance[master_name]["effectiveness"].append(interaction_quality)

        # Track evolution
        self.prompt_evolution.append({
            "timestamp": datetime.now().isoformat(),
            "quality": interaction_quality,
            "prompt_version": master_name
        })

    def enable_cot_enhancement(self, enabled: bool = True):
        """Enable or disable CoT enhancement"""
        self.cot_enabled = enabled
        logger.info(f"🧠 CoT enhancement {'enabled' if enabled else 'disabled'}")

    def get_cot_stats(self) -> Dict[str, Any]:
        """Get statistics about CoT usage"""
        return {
            "cot_enabled": self.cot_enabled,
            "available_task_types": self.cot_enhancer.get_available_task_types()
        }

