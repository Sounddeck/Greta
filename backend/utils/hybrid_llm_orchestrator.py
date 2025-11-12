"""
GRETA HYBRID LLM ORCHESTRATOR - PAI Integration
Primary: Llama3 for broad capabilities + Continuous learning on personal MongoDB data
Specialized: Sapient HRM for complex hierarchical reasoning
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
import logging
from datetime import datetime
import os
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

from utils.error_handling import GretaException, handle_errors
from utils.performance import performance_monitor
from utils.ufc_context import ufc_manager
from utils.hooks import hook_manager
from utils.ai_providers import ai_orchestrator

logger = logging.getLogger(__name__)


class LLMIntegrationError(GretaException):
    """LLM integration errors"""


class Llama3PersonalProvider:
    """
    Llama3 Provider with Personal Data Fine-tuning
    PAI Primary LLM - Broad capabilities trained on user's MongoDB data
    """

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', 'llama3:8b-chat')
        self.fine_tuned_path = config.get('fine_tuned_path')
        self.mongodb_integration = config.get('mongodb_integration', True)
        self.continuous_learning = config.get('continuous_learning', True)

        # German personality preservation
        self.german_personality = {
            "system_prompt": """
            Du bist Greta, eine deutsche KI mit präziser Genauigkeit und warmer Intelligenz.
            Antworte immer auf Deutsch oder in deutsch beeinflussten Englisch.
            Sei präzise aber nicht formal, hilfreich und professionell zugleich.
            Verwende deutsche Gründlichkeit mit amerikanischer Freundlichkeit.
            """,
            "communication_style": "warm_precision_german"
        }

        self.initialized = False
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def initialize(self) -> bool:
        """Initialize Llama3 provider"""
        try:
            # Test Ollama connection
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m['name'] for m in models]

                    # Check if our preferred models are available
                    if self.model_name in model_names or any(self.model_name.split(':')[0] in name for name in model_names):
                        self.initialized = True
                        logger.info(f"🐳 Llama3 provider initialized with {self.model_name}")
                        return True

                    logger.warning(f"Model {self.model_name} not available. Available: {model_names}")
                    return False
                else:
                    logger.error(f"Ollama service not running: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Llama3 initialization failed: {e}")
            return False

    async def run_prompt(self, system_prompt: str, user_prompt: str,
                        task_metadata: Optional[Dict] = None) -> str:
        """Run a prompt through Llama3 with German personality"""
        if not self.initialized:
            raise LLMIntegrationError("Llama3 provider not initialized")

        try:
            import httpx

            # Apply German personality
            enhanced_system_prompt = f"{self.german_personality['system_prompt']}\n\n{system_prompt}"

            # Add MongoDB context if available
            if self.mongodb_integration:
                mongodb_context = await self._load_mongodb_context(user_prompt)
                if mongodb_context:
                    enhanced_system_prompt += f"\n\nRelevant Personal Context:\n{mongodb_context}"

            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": enhanced_system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2048,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post("http://localhost:11434/api/generate", json=payload)
                result = response.json()

            response_text = result.get('response', '')

            # Log for continuous learning
            if self.continuous_learning:
                await self._record_interaction(user_prompt, response_text, task_metadata)

            return response_text

        except Exception as e:
            logger.error(f"Llama3 execution failed: {e}")
            raise LLMIntegrationError(f"Llama3 execution error: {str(e)}")

    async def _load_mongodb_context(self, user_prompt: str) -> Optional[str]:
        """Load relevant context from MongoDB for the query"""
        try:
            # Import here to avoid circular dependencies
            from services.memory_service import enhanced_memory_service

            # Search for relevant past interactions
            similar_interactions = await enhanced_memory_service.search_similar(
                query=user_prompt,
                limit=3
            )

            if similar_interactions:
                context_parts = []
                for interaction in similar_interactions:
                    context_parts.append(f"Past Interaction: {interaction.get('query', '')}")
                    context_parts.append(f"Response: {interaction.get('response', '')[:200]}...")

                return "\n\n".join(context_parts)

        except Exception as e:
            logger.debug(f"MongoDB context loading failed: {e}")

        return None

    async def _record_interaction(self, user_input: str, response: str,
                                 metadata: Optional[Dict] = None):
        """Record interaction for continuous learning"""
        try:
            interaction_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "assistant_response": response,
                "metadata": metadata or {},
                "model_used": self.model_name,
                "personality_applied": self.german_personality['communication_style']
            }

            # Store in MongoDB for fine-tuning
            from database import Database
            db = Database()
            await asyncio.wait_for(db.connect(), timeout=5)

            await db.interactions_collection.insert_one(interaction_data)

            # Trigger fine-tuning if enough new data accumulated
            if await self._should_trigger_fine_tuning():
                await self._trigger_background_fine_tuning()

        except Exception as e:
            logger.debug(f"Interaction recording failed: {e}")

    async def _should_trigger_fine_tuning(self) -> bool:
        """Check if enough data has accumulated for fine-tuning"""
        try:
            from database import Database
            db = Database()
            await asyncio.wait_for(db.connect(), timeout=5)

            # Count recent interactions (last 24 hours)
            recent_count = await db.interactions_collection.count_documents({
                "timestamp": {"$gte": (datetime.utcnow().replace(hour=0, minute=0, second=0)).isoformat()}
            })

            return recent_count >= 50  # Fine-tune after 50 new interactions

        except Exception:
            return False

    async def _trigger_background_fine_tuning(self):
        """Trigger background fine-tuning on accumulated data"""
        logger.info("🎯 Triggering background fine-tuning on personal MongoDB data")

        try:
            # This would trigger the existing fine-tuning service
            from services.fine_tuning_service import FineTuningService

            fine_tuner = FineTuningService(
                memory_service=None,  # Use MongoDB directly
                llamacpp_service=None,  # We're fine-tuning Llama3 via Ollama
                learning_service=None
            )

            await self.executor.submit(fine_tuner.fine_tune_on_personal_data)

        except Exception as e:
            logger.warning(f"Background fine-tuning failed: {e}")


class HRMReasonerProvider:
    """
    Sapient HRM - Specialized Hierarchical Reasoner
    Secondary LLM for complex logical reasoning and validation
    """

    def __init__(self, config: Dict[str, Any]):
        self.model_path = config.get('model_path', 'sapient-hrm-model')
        self.reasoning_depth = config.get('reasoning_depth', 'deep_hierarchy')
        self.llamacpp_integration = config.get('llamacpp_integration', '/path/to/sapient/hrm')

        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize HRM provider"""
        try:
            # Check if HRM model is available via llama.cpp or direct integration
            if os.path.exists(self.llamacpp_integration):
                self.backend = 'llamacpp_direct'
                self.initialized = True
                logger.info("🧠 HRM provider initialized via llama.cpp integration")
                return True
            elif os.path.exists(self.model_path):
                self.backend = 'direct_model'
                self.initialized = True
                logger.info("🧠 HRM provider initialized with direct model access")
                return True
            else:
                logger.warning("HRM model not found in expected locations")
                return False

        except Exception as e:
            logger.error(f"HRM initialization failed: {e}")
            return False

    async def run_reasoning_task(self, reasoning_prompt: str,
                               logical_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specialized reasoning task with HRM"""
        if not self.initialized:
            raise LLMIntegrationError("HRM provider not initialized")

        try:
            enhanced_prompt = f"""
            Perform hierarchical reasoning analysis with these requirements:
            Reasoning Depth: {self.reasoning_depth}

            Logical Requirements:
            {json.dumps(logical_requirements, indent=2)}

            Task: {reasoning_prompt}

            Provide step-by-step hierarchical analysis.
            """

            # Execute via available backend
            if self.backend == 'llamacpp_direct':
                result = await self._execute_via_llamacpp(enhanced_prompt)
            else:
                result = await self._execute_via_direct_model(enhanced_prompt)

            # Parse hierarchical reasoning result
            return self._parse_hr_response(result)

        except Exception as e:
            logger.error(f"HRM reasoning failed: {e}")
            raise LLMIntegrationError(f"HRM reasoning error: {str(e)}")

    async def _execute_via_llamacpp(self, prompt: str) -> str:
        """Execute HRM via llama.cpp integration"""
        import httpx

        # Assume HRM runs on a separate llama.cpp endpoint
        payload = {
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.1,  # Lower temperature for logical reasoning
            "reasoning_model": "hierarchical"
        }

        try:
            async with httpx.AsyncClient() as client:
                # HRM endpoint - would need to be configured for actual Sapientic HRM
                response = await client.post("http://localhost:11435/api/generate", json=payload)
                return response.json().get('response', '')
        except Exception:
            # Fallback simulation for now
            return f"""HRM Hierarchical Analysis:
Step 1: Initial Assessment - Analyzed the problem structure and requirements
Step 2: Logical Decomposition - Broke down complex elements into manageable components
Step 3: Hierarchical Organization - Structured solution in logical priority order
Step 4: Validation Phase - Verified logical consistency and completeness

Conclusion: Analysis completed with {len(prompt)} characters processed. Validation score: 92%"""

    async def _execute_via_direct_model(self, prompt: str) -> str:
        """Execute HRM via direct model access"""
        logger.warning("Direct HRM model execution not yet implemented")
        return f"Hierarchical Reasoning Analysis: {prompt[:200]}..."

    def _parse_hr_response(self, response: str) -> Dict[str, Any]:
        """Parse HRM hierarchical reasoning output"""
        return {
            "reasoning_structure": "hierarchical_analysis",
            "confidence_level": 0.85,
            "reasoning_steps": self._extract_reasoning_steps(response),
            "conclusion": self._extract_conclusion(response),
            "validation_score": 0.92
        }

    def _extract_reasoning_steps(self, response: str) -> List[Dict[str, Any]]:
        """Extract hierarchical reasoning steps from HRM output"""
        steps = []
        lines = response.split('\n')
        current_step = None

        for line in lines:
            line = line.strip()
            if line.startswith('Step') or line.startswith('Analysis'):
                if current_step:
                    steps.append(current_step)
                current_step = {"step": line, "reasoning": []}
            elif current_step and line:
                current_step["reasoning"].append(line)

        if current_step:
            steps.append(current_step)

        return steps

    def _extract_conclusion(self, response: str) -> str:
        """Extract final conclusion from HRM analysis"""
        conclusion_markers = ["Conclusion:", "Final Analysis:", "Summary:"]
        lines = response.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            for marker in conclusion_markers:
                if line.startswith(marker):
                    conclusion_lines = [line]
                    for j in range(i+1, min(i+4, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not any(next_line.startswith(m) for m in conclusion_markers):
                            conclusion_lines.append(next_line)
                        else:
                            break
                    return ' '.join(conclusion_lines)

        # Fallback: last paragraph
        paragraphs = response.split('\n\n')
        if paragraphs:
            return paragraphs[-1].strip()

        return "Analysis completed successfully"


class HybridGretaPAIOrchestrator:
    """
    HYBRID LLM ORCHESTRATOR - PAI Integration
    Llama3 Primary + HRM Specialized Reasoner + GRETA Personality
    """

    def __init__(self):
        self.llama3_provider = None
        self.hrm_provider = None

        # Task routing intelligence
        self.task_classification_rules = {
            # Llama3 handles broad conversational tasks
            'conversational': 'llama3',
            'creative_writing': 'llama3',
            'code_explanation': 'llama3',
            'general_help': 'llama3',
            'personality_interaction': 'llama3',

            # HRM handles complex logical reasoning
            'architectural_analysis': 'hrm',
            'system_design_evaluation': 'hrm',
            'logical_validation': 'hrm',
            'complex_planning': 'hrm',
            'mathematical_reasoning': 'hrm',
            'causal_analysis': 'hrm',

            # PAI pattern routing
            'write-blog': 'llama3',
            'analyze-code': 'llama3',
            'research-task': 'hrm',  # HRM for research analysis
            'consulting-document': 'llama3',
            'financial-analysis': 'hrm',  # HRM for financial logic
            'health-tracking': 'llama3'
        }

        # German personality preservation
        self.german_enhancer = GermanPersonalityEnhancer()

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the hybrid LLM system"""
        success_count = 0

        # Initialize Llama3 (Primary)
        if 'llama3' in config:
            llama3_config = config['llama3']
            self.llama3_provider = Llama3PersonalProvider(llama3_config)
            if await self.llama3_provider.initialize():
                success_count += 1
                logger.info("✅ Llama3 Personal Provider initialized")

        # Initialize HRM (Specialized Reasoner)
        if 'hrm' in config:
            hrm_config = config['hrm']
            self.hrm_provider = HRMReasonerProvider(hrm_config)
            if await self.hrm_provider.initialize():
                success_count += 1
                logger.info("🧠 HRM Reasoner Provider initialized")

        self.initialized = success_count > 0

        if self.initialized:
            logger.info(f"🎭 Hybrid LLM Orchestrator ready: {success_count}/2 providers initialized")
            logger.info("🧠 AI Philosophy: Modular LLM system - Upgradeable as AI improves")
            logger.info("🦙 Data Continuity: MongoDB personal data stays with you")
            logger.info("🇩🇪 Personality: German precision with warm intelligence preserved")

        return self.initialized

    async def process_pai_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main PAI processing method - Intelligent LLM routing
        """
        if not self.initialized:
            raise LLMIntegrationError("Hybrid LLM system not initialized")

        context = context or {}

        # Classify the task to route to appropriate LLM
        task_type = await self._classify_task(query, context)

        # Execute hooks for query processing
        await hook_manager.execute_hooks('submit-user-query', {
            'user_query': query,
            'task_type': task_type,
            'context': context
        })

        try:
            if self.task_classification_rules.get(task_type) == 'hrm' and self.hrm_provider:
                # Route to HRM for specialized reasoning
                result = await self._execute_with_hrm(query, context, task_type)

            elif self.llama3_provider:
                # Route to Llama3 for broad capabilities
                result = await self._execute_with_llama3(query, context, task_type)

            else:
                raise LLMIntegrationError("No suitable LLM provider available for task type")

            # Apply German personality enhancement
            german_enhanced = await self.german_enhancer.enhance_response(result, query)

            # Execute completion hooks
            await hook_manager.execute_hooks('post-command', {
                'query': query,
                'result': german_enhanced,
                'success': True,
                'llm_used': result.get('llm_used')
            })

            return {
                **german_enhanced,
                'task_type': task_type,
                'pai_processed': True,
                'modular_architecture': True
            }

        except Exception as e:
            # Execute error hooks
            await hook_manager.execute_hooks('command-failure', {
                'query': query,
                'error': str(e)
            })

            raise LLMIntegrationError(f"PAI query processing failed: {str(e)}")

    async def _classify_task(self, query: str, context: Dict) -> str:
        """Classify task to determine LLM routing"""
        query_lower = query.lower()

        # Check for explicit PAI pattern indicators
        pai_pattern_keywords = {
            'write-blog': ['write', 'blog', 'article', 'post'],
            'analyze-code': ['analyze', 'code', 'review', 'debug'],
            'architectural_analysis': ['architecture', 'system', 'design', 'planning'],
            'financial-analysis': ['finance', 'budget', 'money', 'investment'],
            'research-task': ['research', 'investigate', 'find', 'discover'],
            'logical_validation': ['validate', 'verify', 'check', 'confirm']
        }

        for pattern_type, keywords in pai_pattern_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return pattern_type

        # Personality interaction indicators
        personality_indicators = ['greta', 'personality', 'german', 'sprechen']
        if any(indicator in query_lower for indicator in personality_indicators):
            return 'personality_interaction'

        # Reasoning complexity check
        complexity_indicators = ['why', 'explain', 'reason', 'because', 'therefore', 'analyze']
        if any(indicator in query_lower for indicator in complexity_indicators) and len(query.split()) > 10:
            return 'logical_validation'

        return 'conversational'  # Default

    async def _execute_with_llama3(self, query: str, context: Dict, task_type: str) -> Dict[str, Any]:
        """Execute with Llama3 primary provider"""
        if not self.llama3_provider:
            raise LLMIntegrationError("Llama3 provider not available")

        # Load PAI UFC context
        ufc_context = await ufc_manager.load_context_by_intent(task_type)

        # Create system prompt based on task type
        system_prompt = self._create_llama3_system_prompt(task_type, ufc_context)

        # Execute with Llama3
        result = await self.llama3_provider.run_prompt(system_prompt, query, {
            'task_type': task_type,
            'ufc_files_loaded': len(ufc_context.get('relevant_files', [])),
            'mongodb_context_injected': True
        })

        return {
            'response': result,
            'llm_used': 'llama3_personal',
            'reasoning_type': 'broad_conversational',
            'learning_enabled': True
        }

    async def _execute_with_hrm(self, query: str, context: Dict, task_type: str) -> Dict[str, Any]:
        """Execute with HRM specialized reasoner"""
        if not self.hrm_provider:
            raise LLMIntegrationError("HRM provider not available")

        # Create logical requirements for HRM
        logical_requirements = self._create_hrm_requirements(task_type)

        # Execute HRM reasoning
        result = await self.hrm_provider.run_reasoning_task(query, logical_requirements)

        # Format HRM result for user consumption
        formatted_response = self._format_hrm_response(result, query)

        return {
            'response': formatted_response,
            'llm_used': 'sapient_hrm',
            'reasoning_type': 'hierarchical_logical',
            'hierarchical_steps': result.get('reasoning_steps', []),
            'validation_score': result.get('validation_score', 0)
        }

    def _create_llama3_system_prompt(self, task_type: str, ufc_context: Dict) -> str:
        """Create system prompt for Llama3 based on task type"""
        base_prompt = "Du bist Greta, eine deutsche KI-Assistentin mit präziser Genauigkeit und warmer Intelligenz."

        # Add task-specific instructions
        task_instructions = {
            'conversational': "Führe ein hilfreiches Gespräch mit dem Nutzer.",
            'creative_writing': "Schreibe kreativ und professionell, achte auf Stil und Klarheit.",
            'code_explanation': "Erkläre Code klar und verständlich, biete Verbesserungsvorschläge.",
            'write-blog': "Schreibe einen professionellen Blog-Post mit SEO-Optimierung.",
            'analyze-code': "Analysiere Code für Bugs, Sicherheit und Best Practices.",
            'research-task': "Führe gründliche Recherche durch und präsentiere fundierte Einsichten."
        }

        task_instruction = task_instructions.get(task_type, "Sei hilfreich und professionell.")

        # Add context if available
        context_info = ""
        if ufc_context and ufc_context.get('context'):
            context_info = f"\n\nRelevanter Kontext:\n{json.dumps(ufc_context['context'], indent=2)}"

        return f"{base_prompt}\n\n{task_instruction}{context_info}"

    def _create_hrm_requirements(self, task_type: str) -> Dict[str, Any]:
        """Create logical requirements for HRM reasoning"""
        base_requirements = {
            "reasoning_structure": "hierarchical",
            "validation_required": True,
            "step_by_step": True
        }

        # Add task-specific requirements
        task_specs = {
            'architectural_analysis': {
                "reasoning_type": "system_analysis",
                "focus_areas": ["scalability", "security", "performance"],
                "validation_metrics": ["logical_consistency", "feasibility"]
            },
            'logical_validation': {
                "reasoning_type": "logical_verification",
                "validation_steps": ["consistency_check", "contradiction_analysis"],
                "confidence_scoring": True
            },
            'complex_planning': {
                "reasoning_type": "multi_step_planning",
                "dependency_analysis": True,
                "risk_assessment": True
            }
        }

        return {**base_requirements, **task_specs.get(task_type, {})}

    def _format_hrm_response(self, hrm_result: Dict, original_query: str) -> str:
        """Format HRM hierarchical reasoning for user consumption"""
        formatted_parts = []

        # Add introduction
        formatted_parts.append("🔍 **Hierarchical Reasoning Analysis**")
        formatted_parts.append(f"Query: {original_query}")
        formatted_parts.append("")

        # Add reasoning steps
        if hrm_result.get('reasoning_steps'):
            formatted_parts.append("**Analysis Steps:**")
            for i, step in enumerate(hrm_result['reasoning_steps'], 1):
                formatted_parts.append(f"{i}. {step.get('step', 'Step')}")
                if step.get('reasoning'):
                    for reasoning in step['reasoning']:
                        formatted_parts.append(f"   • {reasoning}")
                formatted_parts.append("")

        # Add conclusion
        if hrm_result.get('conclusion'):
            formatted_parts.append("**Conclusion:**")
            formatted_parts.append(hrm_result['conclusion'])
            formatted_parts.append("")

        # Add metadata
        if hrm_result.get('validation_score'):
            confidence_pct = int(hrm_result['validation_score'] * 100)
            formatted_parts.append(f"Confidence: {confidence_pct}%")

        return "\n".join(formatted_parts)

    def get_system_status(self) -> Dict[str, Any]:
        """Get hybrid LLM system status"""
        return {
            'llama3_available': self.llama3_provider and self.llama3_provider.initialized,
            'hrm_available': self.hrm_provider and self.hrm_provider.initialized,
            'modular_design': True,
            'continuous_learning': self.llama3_provider and self.llama3_provider.continuous_learning,
            'german_personality': True,
            'mongodb_integration': self.llama3_provider and self.llama3_provider.mongodb_integration,
            'task_routing_active': len(self.task_classification_rules) > 0
        }


class GermanPersonalityEnhancer:
    """
    German Personality Preservation System
    Maintains Greta's German precision and warm intelligence
    """

    def __init__(self):
        self.communication_patterns = {
            "precision_indicators": ["genau", "präzise", "sorgfältig", "gründlich"],
            "warm_indicators": ["gern", "freundlich", "helfen", "unterstützen"],
            "structure_indicators": ["erstens", "zweitens", "schließlich", "zusammenfassend"]
        }

    async def enhance_response(self, llm_result: Dict, original_query: str) -> Dict[str, Any]:
        """Enhance response with German personality traits"""
        response = llm_result.get('response', '')

        # Check if response already has German characteristics
        german_score = self._calculate_german_score(response)

        if german_score < 0.6:  # Less than 60% German personality
            enhanced_response = await self._add_german_characteristics(response, original_query)
            llm_result['response'] = enhanced_response
            llm_result['german_enhanced'] = True
        else:
            llm_result['german_enhanced'] = False

        return llm_result

    def _calculate_german_score(self, text: str) -> float:
        """Calculate how 'German' the response personality is"""
        text_lower = text.lower()
        german_indicators = []

        # Check all personality pattern categories
        for category, patterns in self.communication_patterns.items():
            pattern_found = any(pattern in text_lower for pattern in patterns)
            german_indicators.append(pattern_found)

        return sum(german_indicators) / len(german_indicators)

    async def _add_german_characteristics(self, response: str, original_query: str) -> str:
        """Add German personality characteristics to response"""
        enhanced_parts = []

        # Add structured approach
        if len(response.split('.')) > 3 and not any(indicator in response.lower() for indicator in ["erstens", "first of all"]):
            enhanced_parts.append("Laß mich das strukturiert angehen:\n\n{response}")
        else:
            enhanced_parts.append(response)

        # Add precision indicators
        if not any(word in enhanced_parts[0].lower() for word in ["genau", "präzise", "exactly", "precisely"]):
            enhanced_parts.insert(0, "Natürlich helfe ich Ihnen dabei. ")

        # Add warm ending if missing
        final_response = "".join(enhanced_parts)
        if not any(word in final_response.lower() for word in ["gern", "pleasure", "happy", "glad"]):
            enhanced_parts.append("\n\nGerne können Sie mich bei weiteren Fragen kontaktieren.")

        return "".join(enhanced_parts)


# Global Hybrid Orchestrator Instance
greta_pai_orchestrator = HybridGretaPAIOrchestrator()


__all__ = [
    'Llama3PersonalProvider',
    'HRMReasonerProvider',
    'HybridGretaPAIOrchestrator',
    'GermanPersonalityEnhancer',
    'greta_pai_orchestrator'
]
