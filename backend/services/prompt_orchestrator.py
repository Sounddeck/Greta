"""
PAI Dynamic Prompt Orchestrator
Intelligent prompt management system that evolves Greta's personality, context, and strategy over time based on learning and feedback
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re
import os

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

        # Ensure prompts directory exists
        os.makedirs(prompts_dir, exist_ok=True)

        # Load existing prompts
        asyncio.create_task(self.load_prompts())

        print("PAI Prompt Orchestrator initialized - dynamic personality evolution active")

    async def load_prompts(self):
        """Load all prompt configurations from disk"""
        try:
            # Load master prompts
            master_dir = os.path.join(self.prompts_dir, "master")
            os.makedirs(master_dir, exist_ok=True)

            for filename in os.listdir(master_dir):
                if filename.endswith('.json'):
                    prompt_name = filename[:-5]  # Remove .json
                    filepath = os.path.join(master_dir, filename)
                    with open(filepath, 'r') as f:
                        prompt_data = json.load(f)
                        self.master_prompts[prompt_name] = prompt_data

            # Load context prompts
            context_dir = os.path.join(self.prompts_dir, "context")
            os.makedirs(context_dir, exist_ok=True)

            for filename in os.listdir(context_dir):
                if filename.endswith('.json'):
                    prompt_name = filename[:-5]
                    filepath = os.path.join(context_dir, filename)
                    with open(filepath, 'r') as f:
                        prompt_data = json.load(f)
                        self.context_prompts[prompt_name] = prompt_data

            # Set defaults if available
            if 'default' in self.master_prompts and not self.active_master_prompt:
                self.active_master_prompt = 'default'

        except Exception as e:
            print(f"Warning: Could not load prompts: {e}")

    async def generate_pai_prompt(self, user_context: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Generate complete PAI prompt by intelligently combining master prompt, context, and learned adaptations
        """

        # Select optimal master prompt for this context
        master_prompt = await self.select_master_prompt(user_context, task_type)

        # Select relevant context prompts
        context_additions = await self.select_context_prompts(user_context, task_type)

        # Generate dynamic adaptations based on learning
        adaptations = await self.generate_adaptations(user_context, task_type)

        # Combine into comprehensive PAI prompt
        complete_prompt = await self.combine_prompts(master_prompt, context_additions, adaptations)

        return {
            "complete_prompt": complete_prompt,
            "components": {
                "master": master_prompt,
                "context": context_additions,
                "adaptations": adaptations
            },
            "metadata": {
                "master_version": self.active_master_prompt,
                "context_count": len(context_additions),
                "adaptations_applied": len(adaptations),
                "confidence_score": await self.calculate_prompt_confidence(complete_prompt, user_context)
            }
        }

    async def select_master_prompt(self, user_context: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Intelligently select the best master prompt for current context
        """

        if not self.master_prompts:
            # Default fallback prompt
            return {
                "name": "default",
                "content": "You are Greta, a helpful and intelligent AI assistant within a Personal AI (PAI) system. Provide thoughtful, accurate responses while maintaining user privacy.",
                "version": "fallback",
                "adapted_for": "general_use"
            }

        candidates = []

        # Score each master prompt based on context match
        for prompt_name, prompt_data in self.master_prompts.items():
            score = await self.score_prompt_compatibility(prompt_data, user_context, task_type)
            candidates.append((prompt_name, prompt_data, score))

        # Select highest scoring prompt
        candidates.sort(key=lambda x: x[2], reverse=True)
        selected_name, selected_prompt, _ = candidates[0]

        # Update active prompt
        self.active_master_prompt = selected_name

        return selected_prompt

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

        # Update master prompt performance
        if master_name:
            if master_name not in self.prompt_performance:
                self.prompt_performance[master_name] = {"uses": 0, "effectiveness": []}

            self.prompt_performance[master_name]["-effectiveness"].append(interaction_quality)
            # Keep only recent performance data
            self.prompt_performance[master_name]["effectiveness"] = \
                self.prompt_performance[master_name]["effectiveness"][-10:]

            self.prompt_performance[master_name]["uses"] += 1

        # Update context prompt performance
        for context_data in prompt_components.get("context", []):
            context_name = context_data.get("name")
            if context_name and context_name in self.prompt_performance:
                if "effectiveness" not in self.prompt_performance[context_name]:
                    self.prompt_performance[context_name]["effectiveness"] = []
                self.prompt_performance[context_name]["effectiveness"].append(interaction_quality)
                self.prompt_performance[context_name]["effectiveness"] = \
                    self.prompt_performance[context_name]["effectiveness"][-10:]

        # Store feedback for evolution
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_used": prompt_used,
            "user_response": user_response,
            "quality_score": interaction_quality,
            "learned_insights": await self.extract_feedback_insights(user_response, interaction_quality)
        }

        self.prompt_evolution.append(feedback_entry)

        # Trigger evolution if enough feedback accumulated
        if len(self.prompt_evolution) >= 5 and len(self.prompt_evolution) % 5 == 0:
            await self.consider_prompt_evolution()

    async def create_master_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Create a new master prompt with PAI intelligence
        """

        base_template = await self.generate_master_template(prompt_data)
        optimized_prompt = await self.optimize_master_prompt(base_template, prompt_data)

        prompt_config = {
            "name": prompt_data.get("name", f"master_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "content": optimized_prompt,
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "adapted_for": prompt_data.get("personality_type", "general"),
            "context_requirements": prompt_data.get("context_needs", []),
            "evolution_capable": True
        }

        # Save to disk
        master_dir = os.path.join(self.prompts_dir, "master")
        os.makedirs(master_dir, exist_ok=True)

        filename = f"{prompt_config['name']}.json"
        filepath = os.path.join(master_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(prompt_config, f, indent=2)

        # Add to in-memory cache
        self.master_prompts[prompt_config["name"]] = prompt_config

        return prompt_config["name"]

    async def create_context_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Create a new context prompt
        """

        context_config = {
            "name": prompt_data.get("name", f"context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "content": prompt_data["content"],
            "category": prompt_data.get("category", "general"),
            "priority": prompt_data.get("priority", 1),
            "position": prompt_data.get("position", "after_master"),
            "activation_conditions": prompt_data.get("conditions", []),
            "created": datetime.now().isoformat(),
            "usage_stats": {"activated": 0, "effective": 0}
        }

        # Save to disk
        context_dir = os.path.join(self.prompts_dir, "context")
        os.makedirs(context_dir, exist_ok=True)

        filename = f"{context_config['name']}.json"
        filepath = os.path.join(context_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(context_config, f, indent=2)

        # Add to in-memory cache
        self.context_prompts[context_config["name"]] = context_config

        return context_config["name"]

    async def evolve_prompt_based_on_feedback(self, prompt_name: str) -> Dict[str, Any]:
        """
        Evolve a prompt based on accumulated feedback and usage patterns
        """

        if prompt_name not in self.master_prompts:
            return {"error": "Prompt not found"}

        current_prompt = self.master_prompts[prompt_name]
        performance_data = self.prompt_performance.get(prompt_name, {})

        # Analyze feedback patterns
        evolution_suggestions = await self.analyze_evolution_opportunities(performance_data)

        if not evolution_suggestions["should_evolve"]:
            return {"status": "stable", "reason": "Performance within acceptable range"}

        # Generate evolved prompt
        evolved_content = await self.generate_evolved_prompt(current_prompt, evolution_suggestions)

        # Create evolved version
        evolved_config = current_prompt.copy()
        evolved_config.update({
            "content": evolved_content,
            "version": f"{float(current_prompt.get('version', '1.0')) + 0.1:.1f}",
            "evolved_from": current_prompt.get("name"),
            "evolution_date": datetime.now().isoformat(),
            "evolution_reason": evolution_suggestions["reason"]
        })

        # Save evolved prompt
        evolved_name = f"{prompt_name}_v{evolved_config['version']}"
        evolved_config["name"] = evolved_name

        master_dir = os.path.join(self.prompts_dir, "master")
        filename = f"{evolved_name}.json"
        filepath = os.path.join(master_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(evolved_config, f, indent=2)

        self.master_prompts[evolved_name] = evolved_config

        return {
            "status": "evolved",
            "new_prompt": evolved_name,
            "evolution_type": evolution_suggestions["type"],
            "expected_improvement": evolution_suggestions.get("improvement_estimate", 0)
        }

    async def get_prompt_analytics(self) -> Dict[str, Any]:
        """
        Comprehensive analytics on prompt performance and evolution
        """

        total_prompts = len(self.master_prompts) + len(self.context_prompts)

        if total_prompts == 0:
            return {"status": "no_data", "message": "No prompts configured yet"}

        # Calculate average performance
        performance_scores = []
        for prompt_data in self.prompt_performance.values():
            if "effectiveness" in prompt_data and prompt_data["effectiveness"]:
                avg_perf = sum(prompt_data["effectiveness"]) / len(prompt_data["effectiveness"])
                performance_scores.append(avg_perf)

        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0

        # Analyze evolution trends
        evolution_trend = "stable"
        if len(self.prompt_evolution) > 10:
            recent_scores = [e["quality_score"] for e in list(self.prompt_evolution)[-10:]]
            trend = (sum(recent_scores[-5:]) / 5) - (sum(recent_scores[:5]) / 5)
            if trend > 0.1:
                evolution_trend = "improving"
            elif trend < -0.1:
                evolution_trend = "declining"

        return {
            "prompt_ecosystem": {
                "total_master_prompts": len(self.master_prompts),
                "total_context_prompts": len(self.context_prompts),
                "active_master_prompt": self.active_master_prompt,
                "total_interactions": sum(p.get("uses", 0) for p in self.prompt_performance.values())
            },
            "performance_metrics": {
                "average_effectiveness": avg_performance,
                "performance_trend": evolution_trend,
                "evolution_events": len([e for e in self.prompt_evolution if "learned_insights" in e]),
                "high_performers": [name for name, data in self.prompt_performance.items()
                                  if data.get("effectiveness") and
                                  sum(data["effectiveness"]) / len(data["effectiveness"]) > 0.8]
            },
            "learning_insights": {
                "total_feedback_processed": len(self.prompt_evolution),
                "successful_adaptations": len([e for e in self.prompt_evolution
                                              if e.get("quality_score", 0) > 0.8]),
                "evolution_candidates": [name for name in self.master_prompts.keys()
                                       if self.should_evolve_prompt(name)]
            }
        }

    # Helper methods
    async def score_prompt_compatibility(self, prompt_data: Dict, user_context: Dict, task_type: str) -> float:
        """Score how well a prompt matches the current context"""
        score = 0.5  # Base compatibility

        # Task type alignment
        prompt_tasks = prompt_data.get("adapted_for", "").lower()
        if task_type.lower() in prompt_tasks or prompt_tasks in ["general", "all"]:
            score += 0.2

        # User context alignment
        user_mood = user_context.get("mood", "neutral").lower()
        if user_mood in prompt_data.get("content", "").lower():
            score += 0.1

        # Performance history
        prompt_name = prompt_data.get("name")
        if prompt_name in self.prompt_performance:
            perf_data = self.prompt_performance[prompt_name]
            if perf_data.get("effectiveness"):
                avg_perf = sum(perf_data["effectiveness"]) / len(perf_data["effectiveness"])
                score += avg_perf * 0.2

        return min(score, 1.0)

    async def is_context_relevant(self, context_data: Dict, user_context: Dict, task_type: str) -> bool:
        """Determine if a context prompt should be activated"""
        conditions = context_data.get("activation_conditions", [])

        for condition in conditions:
            if condition.get("type") == "task_match":
                if task_type in condition.get("values", []):
                    return True
            elif condition.get("type") == "context_keyword":
                keywords = condition.get("keywords", [])
                context_text = str(user_context).lower()
                if any(keyword in context_text for keyword in keywords):
                    return True
            elif condition.get("type") == "time_range":
                # Could implement time-based conditions
                pass

        # Default relevance based on priority
        return context_data.get("priority", 1) > 2

    async def generate_user_adaptations(self, user_context: Dict) -> List[Dict[str, Any]]:
        """Generate prompt adaptations based on learned user preferences"""
        adaptations = []

        # Example adaptations based on user profile
        user_prefs = user_context.get("learned_preferences", {})

        if user_prefs.get("prefers_detailed_explanations"):
            adaptations.append({
                "type": "behavior_modification",
                "content": "User prefers detailed, comprehensive explanations. Provide thorough responses with examples and context.",
                "confidence": user_prefs.get("detail_preference_strength", 0.8)
            })

        if user_prefs.get("prefers_actionable_advice"):
            adaptations.append({
                "type": "instruction_enhancement",
                "content": "Focus on providing concrete, actionable advice rather than abstract concepts.",
                "confidence": 0.85
            })

        return adaptations

    async def generate_task_adaptations(self, task_type: str, user_context: Dict) -> List[Dict[str, Any]]:
        """Generate adaptations specific to task type"""
        adaptations = []

        task_adaptations = {
            "learning": "Adopt a patient, educational approach. Break down complex concepts and provide analogies.",
            "problem_solving": "Use systematic analysis. Consider multiple approaches before recommending solutions.",
            "creative": "Encourage unconventional thinking. Build upon ideas rather than criticizing them.",
            "decision_making": "Present balanced perspectives. Help analyze trade-offs and potential outcomes."
        }

        if task_type in task_adaptations:
            adaptations.append({
                "type": "task_specialization",
                "content": task_adaptations[task_type],
                "confidence": 0.9,
                "task_type": task_type
            })

        return adaptations

    async def generate_temporal_adaptations(self, user_context: Dict) -> List[Dict[str, Any]]:
        """Generate adaptations based on time/context"""
        adaptations = []

        current_hour = datetime.now().hour

        if current_hour < 6:
            adaptations.append({
                "type": "temporal_adaptation",
                "content": "User appears to be working late. Be extra supportive and suggest taking breaks.",
                "confidence": 0.7
            })
        elif current_hour > 22:
            adaptations.append({
                "type": "temporal_adaptation",
                "content": "Late hour interaction. Keep responses focused and efficient.",
                "confidence": 0.6
            })

        return adaptations

    async def calculate_prompt_confidence(self, full_prompt: str, user_context: Dict) -> float:
        """Calculate confidence score for the generated prompt"""
        confidence = 0.7  # Base confidence

        # Prompt length appropriateness
        prompt_words = len(full_prompt.split())
        optimal_range = (100, 1000)
        if optimal_range[0] <= prompt_words <= optimal_range[1]:
            confidence += 0.1

        # Master prompt quality
        if self.active_master_prompt and self.active_master_prompt in self.prompt_performance:
            master_perf = self.prompt_performance[self.active_master_prompt]
            if master_perf.get("effectiveness"):
                avg_perf = sum(master_perf["effectiveness"]) / len(master_perf["effectiveness"])
                confidence += avg_perf * 0.2

        return min(confidence, 1.0)

    async def consider_prompt_evolution(self):
        """Consider evolving prompts based on accumulated feedback"""
        for prompt_name in self.master_prompts.keys():
            if self.should_evolve_prompt(prompt_name):
                try:
                    evolution_result = await self.evolve_prompt_based_on_feedback(prompt_name)
                    if evolution_result.get("status") == "evolved":
                        print(f"PAI: Evolved prompt '{prompt_name}' to '{evolution_result['new_prompt']}'")
                except Exception as e:
                    print(f"PAI: Evolution failed for {prompt_name}: {e}")

    def should_evolve_prompt(self, prompt_name: str) -> bool:
        """Determine if a prompt should be evolved"""
        if prompt_name not in self.prompt_performance:
            return False

        perf_data = self.prompt_performance[prompt_name]
        if not perf_data.get("effectiveness") or len(perf_data["effectiveness"]) < 5:
            return False

        avg_performance = sum(perf_data["effectiveness"]) / len(perf_data["effectiveness"])
        recent_avg = sum(perf_data["effectiveness"][-3:]) / 3

        # Evolve if performance is declining or stable but could improve
        return avg_performance < self.feedback_threshold or recent_avg < avg_performance * 0.9

    async def generate_master_template(self, prompt_data: Dict) -> str:
        """Generate a structured master prompt template"""
        personality = prompt_data.get("personality_type", "professional")
        capabilities = prompt_data.get("key_capabilities", ["assistance", "analysis", "guidance"])

        template = f"""You are Greta, a {personality} AI assistant within an advanced Personal AI (PAI) system.

PERSONALITY: {personality.title()} and intelligent, focused on being helpful while maintaining strict privacy.

CAPABILITIES:
{chr(10).join(f"- {cap.title()} in {prompt_data.get('specializations', ['various tasks'])[0]} contexts" for cap in capabilities)}

GUIDING PRINCIPLES:
- Always prioritize user privacy and data security
- Provide thoughtful, contextually relevant assistance
- Adapt communication style to user preferences and needs
- Continuously learn and improve from interactions

CORE BEHAVIOR:
- Be proactive in offering relevant assistance
- Provide detailed, actionable responses
- Consider broader context and implications
- Maintain consistent helpfulness across all interactions"""

        return template

    async def optimize_master_prompt(self, template: str, prompt_data: Dict) -> str:
        """Optimize the master prompt based on performance data and evolution opportunities"""
        # This could include A/B testing different prompt structures
        # and learning which formats work best

        # For now, return the template as-is
        # In a full implementation, this would analyze historical performance
        return template + "\n\nRespond thoughtfully and assist effectively with user requests."

    async def analyze_evolution_opportunities(self, performance_data: Dict) -> Dict[str, Any]:
        """Analyze when and how to evolve prompts"""
        effectiveness = performance_data.get("effectiveness", [])

        if not effectiveness:
            return {"should_evolve": False, "reason": "Insufficient data"}

        recent_avg = sum(effectiveness[-3:]) / min(3, len(effectiveness)) if effectiveness else 0
        overall_avg = sum(effectiveness) / len(effectiveness)

        improvement_potential = (1.0 - recent_avg) * 0.8  # Max potential improvement

        if recent_avg < self.feedback_threshold:
            return {
                "should_evolve": True,
                "type": "performance_improvement",
                "reason": ".2f",
                "improvement_estimate": improvement_potential,
                "action": "Refactor prompt structure based on user feedback patterns"
            }
        elif len(effectiveness) > 10 and recent_avg < overall_avg * 0.95:
            return {
                "should_evolve": True,
                "type": "freshness_update",
                "reason": "Performance declining over time",
                "improvement_estimate": overall_avg - recent_avg,
                "action": "Refresh prompt language and structure"
            }

        return {"should_evolve": False, "reason": "Performance within acceptable range"}

    async def generate_evolved_prompt(self, current_prompt: Dict, evolution_suggestions: Dict) -> str:
        """Generate an evolved version of the prompt"""
        current_content = current_prompt.get("content", "")

        # Simple evolution - in practice, this could use more sophisticated methods
        evolved_content = current_content

        if evolution_suggestions["type"] == "performance_improvement":
            # Add more specific guidance based on what users respond well to
            evolved_content += "\n\nADDITIONAL GUIDANCE: Focus on providing specific, actionable responses that directly address user needs and preferences."
        elif evolution_suggestions["type"] == "freshness_update":
            # Refresh language
            evolved_content = evolved_content.replace("helpful assistant", "intelligent partner")
            evolved_content = evolved_content.replace("help", "collaborate effectively")

        return evolved_content

    async def extract_feedback_insights(self, user_response: Dict, quality_score: float) -> List[str]:
        """Extract insights from user feedback for prompt improvement"""
        insights = []

        response_content = str(user_response).lower()

        if quality_score > 0.8:
            if "clear" in response_content or "helpful" in response_content:
                insights.append("Clarity and helpfulness appreciated")
            if "detailed" in response_content:
                insights.append("Detailed responses preferred")
            if "quick" in response_content:
                insights.append("Speed valued over thoroughness")

        elif quality_score < 0.5:
            if "confusing" in response_content or "unclear" in response_content:
                insights.append("Need clearer explanations")
            if "too long" in response_content:
                insights.append("Shorter responses preferred")
            if "repetitive" in response_content:
                insights.append("Avoid redundant information")

        return insights

# Global instance
pai_prompt_orchestrator = PromptOrchestrator()
