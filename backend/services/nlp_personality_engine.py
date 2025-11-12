"""
PAI Neuro-Linguistic Programming (NLP) Personality Engine
Integrates NLP techniques to create deeply personalized communication and learning patterns.
Greta learns and adapts communication styles based on NLP principles for optimal personal interaction.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re
import statistics

class NLP_PersonalityEngine:
    """
    PAI NLP Personality Engine: Integrates Neuro-Linguistic Programming techniques
    to create deeply personalized communication patterns that adapt to user preferences.

    Core NLP Components:
    - Sensory Preferences (Visual/Auditory/Kinesthetic)
    - Language Patterns & Pace
    - Belief Structures & Framing
    - Anchoring & Emotional Triggers
    - Meta-Programs (Decision Making, Motivation)
    """

    def __init__(self):
        # NLP Sensory Modalities
        self.modality_detection = self._initialize_modality_system()
        self.sensory_patterns = defaultdict(list)

        # Language Pattern Analysis
        self.language_patterns = defaultdict(list)
        self.meta_programs = defaultdict(list)
        self.belief_structures = defaultdict(dict)

        # Communication Rhythm & Pace
        self.communication_rhythms = {}
        self.pacing_patterns = {}

        # Emotional Anchoring & Triggers
        self.emotional_responses = defaultdict(list)
        self.anchor_activations = defaultdict(list)

        # Personal Communication Preferences
        self.preferred_metaphors = []
        self.communication_style = {}

        # Learning Evolution
        self.nlp_learning_history = deque(maxlen=200)
        self.anchor_effectiveness = {}

        # NLP Constants
        self.sensory_keywords = {
            'visual': ['see', 'look', 'view', 'appear', 'image', 'picture', 'show', 'display', 'watch', 'observe', 'perspective', 'vision', 'bright', 'dim', 'clear', 'fuzzy', 'focus'],
            'auditory': ['hear', 'listen', 'sound', 'music', 'tell', 'talk', 'speak', 'say', 'ask', 'discuss', 'volume', 'loud', 'quiet', 'noise', 'resonate'],
            'kinesthetic': ['feel', 'touch', 'sense', 'emotion', 'feelings', 'gut', 'warm', 'cold', 'pressure', 'texture', 'get', 'comfortable', 'uneasy'],
            'auditory_digital': ['think', 'understand', 'know', 'process', 'analyze', 'consider', 'evaluate', 'judge', 'reason', 'logic', 'sense', 'make_sense']
        }

        print("PAI NLP Personality Engine initialized - learning your communication preferences")

    def _initialize_modality_system(self) -> Dict[str, Any]:
        """Initialize NLP sensory modality detection system"""
        return {
            'visual_words': [],
            'auditory_words': [],
            'kinesthetic_words': [],
            'predicates': defaultdict(int),
            'modalities_detected': {},
            'dominant_modality': None,
            'secondary_modality': None
        }

    async def analyze_user_nlp_patterns(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze user message through NLP lens to understand communication preferences
        """

        analysis_start = asyncio.get_event_loop().time()

        # Multi-dimensional NLP analysis
        sensory_patterns = await self._analyze_sensory_modalities(user_message)
        language_patterns = await self._analyze_language_patterns(user_message)
        meta_programs = await self._identify_meta_programs(user_message, context or {})
        communication_preferences = await self._assess_communication_style(user_message)

        # Learning adaptation
        self._update_nlp_learning(sensory_patterns, language_patterns, meta_programs)

        # Generate NLP-adapted response guidance
        adaptation_recommendations = await self._generate_nlp_adaptations(
            sensory_patterns, language_patterns, meta_programs, communication_preferences
        )

        processing_time = asyncio.get_event_loop().time() - analysis_start

        analysis_result = {
            "nlp_patterns": {
                "sensory_modalities": sensory_patterns,
                "language_patterns": language_patterns,
                "meta_programs": meta_programs,
                "communication_preferences": communication_preferences
            },
            "adaptation_recommendations": adaptation_recommendations,
            "nlp_confidence": self._calculate_nlp_confidence(sensory_patterns),
            "modality_match_score": sensory_patterns.get('dominance_confidence', 0),
            "processing_time": round(processing_time, 3),
            "nlp_learning_integrated": True
        }

        # Record learning instance
        self.nlp_learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis_result,
            "user_message": user_message,
            "context": context or {}
        })

        return analysis_result

    async def generate_nlp_guided_response(self, user_analysis: Dict[str, Any], base_response: str, task_context: str) -> Dict[str, Any]:
        """
        Generate NLP-enhanced response tailored to user's communication style
        """

        modalities = user_analysis.get("nlp_patterns", {}).get("sensory_modalities", {})

        # Apply NLP adaptations
        adapted_response = await self._adapt_response_with_nlp(base_response, modalities, task_context)

        # Enhance with meta-program alignment
        meta_enhanced = await self._enhance_with_meta_programs(adapted_response, user_analysis)

        # Apply pacing and rhythm adjustments
        rhythm_optimized = await self._optimize_communication_rhythm(meta_enhanced, user_analysis)

        return {
            "original_response": base_response,
            "nlp_enhanced_response": rhythm_optimized,
            "nlp_adaptations_applied": [
                "sensory_modality_alignment",
                "language_pattern_matching",
                "meta_program_resonance",
                "communication_rhythm_optimization"
            ],
            "personalization_score": await self._calculate_personalization_score(user_analysis),
            "nlp_effectiveness_projection": await self._project_nlp_effectiveness(user_analysis, task_context)
        }

    async def _analyze_sensory_modalities(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to determine user's dominant sensory modality preferences
        """

        text_lower = text.lower()
        modality_counts = {}

        # Count sensory predicate usage
        for modality, keywords in self.sensory_keywords.items():
            count = 0
            for keyword in keywords:
                # Use word boundaries to match whole words
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                count += matches

            modality_counts[modality] = count
            self.modality_detection['predicates'][modality] += count

        # Determine dominant modality
        if modality_counts:
            sorted_modalities = sorted(modality_counts.items(), key=lambda x: x[1], reverse=True)
            dominant = sorted_modalities[0][0]
            secondary = sorted_modalities[1][0] if len(sorted_modalities) > 1 else None

            total_words = sum(modality_counts.values())
            confidence = modality_counts[dominant] / total_words if total_words > 0 else 0

            self.modality_detection['dominant_modality'] = dominant
            self.modality_detection['secondary_modality'] = secondary

        return {
            "modality_counts": modality_counts,
            "dominant_modality": self.modality_detection.get('dominant_modality'),
            "secondary_modality": self.modality_detection.get('secondary_modality'),
            "dominance_confidence": confidence if 'confidence' in locals() else 0,
            "nlp_patterns_detected": len([c for c in modality_counts.values() if c > 0])
        }

    async def _analyze_language_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze language patterns, structure, and complexity preferences
        """

        patterns = {}

        # Sentence structure analysis
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.strip().split()) for s in sentences if s.strip()]

        if sentence_lengths:
            patterns.update({
                "average_sentence_length": statistics.mean(sentence_lengths),
                "sentence_complexity": max(sentence_lengths) - min(sentence_lengths),
                "sentence_variability": statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
            })

        # Word choice analysis
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        unique_words = set(words)

        # Complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        vocab_richness = len(unique_words) / len(words) if words else 0

        # Question patterns
        questions = len(re.findall(r'\?', text))
        imperatives = len(re.findall(r'\b(please|do|make|create|show|tell)\b', text.lower()))

        patterns.update({
            "vocabulary_richness": vocab_richness,
            "average_word_length": avg_word_length,
            "question_frequency": questions / len(sentences) if sentences else 0,
            "imperative_frequency": imperatives / len(sentences) if sentences else 0,
            "directness_preference": imperatives > questions  # More commands vs questions
        })

        # Learning style indicators
        patterns["learning_modality"] = "visual" if any(word in text.lower() for word in ["see", "show", "display"]) else \
                                      "auditory" if any(word in text.lower() for word in ["hear", "tell", "explain"]) else \
                                      "kinesthetic" if any(word in text.lower() for word in ["feel", "touch", "experience"]) else "general"

        return patterns

    async def _identify_meta_programs(self, text: str, context: Dict) -> Dict[str, Any]:
        """
        Identify NLP meta-programs (deep unconscious patterns)
        """

        meta_programs = {}

        # Toward vs Away From motivation
        toward_indicators = ["want", "achieve", "reach", "get", "obtain", "accomplish"]
        away_indicators = ["avoid", "prevent", "stop", "eliminate", "reduce", "escape"]

        toward_count = sum(1 for word in toward_indicators if word in text.lower())
        away_count = sum(1 for word in away_indicators if word in text.lower())

        meta_programs["motivation_pattern"] = "toward" if toward_count > away_count else "away_from"

        # Big Chunk vs Small Chunk processing
        big_chunk_indicators = ["overview", "big picture", "broad", "general", "comprehensive", "holistic"]
        small_chunk_indicators = ["detail", "specific", "step", "exact", "precise", "particular"]

        big_count = sum(1 for word in big_chunk_indicators if word in text.lower())
        small_count = sum(1 for word in small_chunk_indicators if word in text.lower())

        meta_programs["chunk_size_preference"] = "big_chunk" if big_count > small_count else "small_chunk"

        # Options vs Procedures
        options_indicators = ["choice", "option", "alternative", "flexible", "variety"]
        procedures_indicators = ["steps", "process", "method", "standard", "protocol"]

        options_count = sum(1 for word in options_indicators if word in text.lower())
        procedures_count = sum(1 for word in procedures_indicators if word in text.lower())

        meta_programs["decision_style"] = "options" if options_count > procedures_count else "procedures"

        return meta_programs

    async def _assess_communication_style(self, text: str) -> Dict[str, Any]:
        """
        Assess overall communication style preferences
        """

        style = {}

        # Formality level
        formal_indicators = ["therefore", "consequently", "accordingly", "furthermore", "moreover"]
        informal_indicators = ["like", "yeah", "kinda", "sorta", "actually"]

        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())

        style["formality_preference"] = "formal" if formal_count > informal_count * 2 else \
                                      "casual" if informal_count > formal_count else "mixed"

        # Pacing preference (inferred from sentence structure)
        sentences = re.split(r'[.!?]+', text)
        total_sentences = len([s for s in sentences if s.strip()])

        words_per_sentence = len(re.findall(r'\w+', text)) / max(total_sentences, 1)

        style["pace_preference"] = "detailed" if words_per_sentence > 15 else \
                                 "concise" if words_per_sentence < 8 else "balanced"

        # Metaphor usage
        metaphors = re.findall(r'(?:like|as|similar to|compared to|metaphor|analogy)', text.lower())
        style["metaphor_preference"] = "high" if len(metaphors) > 2 else "low"

        return style

    async def _generate_nlp_adaptations(self, sensory: Dict, language: Dict, meta: Dict, communication: Dict) -> List[Dict[str, Any]]:
        """
        Generate specific NLP-based adaptation recommendations
        """

        adaptations = []

        # Sensory modality alignment
        if sensory.get('dominant_modality'):
            adaptations.append({
                "type": "sensory_alignment",
                "strategy": f"use_{sensory['dominant_modality']}_language",
                "description": f"Match user's preferred {sensory['dominant_modality']} sensory system",
                "priority": "high"
            })

        # Language pattern matching
        if language.get('pace_preference') == 'detailed':
            adaptations.append({
                "type": "content_density",
                "strategy": "provide_comprehensive_information",
                "description": "User prefers detailed explanations with comprehensive information",
                "priority": "medium"
            })
        elif language.get('pace_preference') == 'concise':
            adaptations.append({
                "type": "content_density",
                "strategy": "be_direct_efficient",
                "description": "Keep responses focused and to-the-point",
                "priority": "medium"
            })

        # Meta-program resonance
        if meta.get('motivation_pattern') == 'toward':
            adaptations.append({
                "type": "motivation_alignment",
                "strategy": "emphasize_benefits_achievements",
                "description": "Focus on goals, achievements, and positive outcomes",
                "priority": "high"
            })
        elif meta.get('motivation_pattern') == 'away_from':
            adaptations.append({
                "type": "motivation_alignment",
                "strategy": "emphasize_problem_solutions",
                "description": "Focus on solving problems and avoiding negative outcomes",
                "priority": "high"
            })

        return adaptations

    async def _adapt_response_with_nlp(self, response: str, sensory_modalities: Dict, task_context: str) -> str:
        """
        Adapt response to match user's sensory modality preferences
        """

        adapted_response = response
        dominant_modality = sensory_modalities.get('dominant_modality')

        if dominant_modality == 'visual':
            # Add visual language elements
            visual_enhancements = [
                "imagine", "picture this", "visualize", "see how", "appears like",
                "looks as though", "clearly visible", "perspective shows"
            ]
            adapted_response = await self._infuse_nlp_language(adapted_response, visual_enhancements)

        elif dominant_modality == 'auditory':
            # Add auditory language elements
            auditory_enhancements = [
                "hear this", "sounds like", "listen to", "resonates as", "echoes",
                "harmonizes with", "tune into", "rhythm of"
            ]
            adapted_response = await self._infuse_nlp_language(adapted_response, auditory_enhancements)

        elif dominant_modality == 'kinesthetic':
            # Add kinesthetic language elements
            kinesthetic_enhancements = [
                "feel this", "get a sense of", "grasp the", "touch on", "solid feel",
                "smooth as", "rough around", "warm to", "cold reality"
            ]
            adapted_response = await self._infuse_nlp_language(adapted_response, kinesthetic_enhancements)

        return adapted_response

    async def _infuse_nlp_language(self, response: str, nlp_phrases: List[str]) -> str:
        """
        Strategically infuse NLP language patterns into response
        """
        # Simple implementation - in production would use more sophisticated NLP
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())

        if len(sentences) > 2:
            # Infuse at strategic points without overdoing it
            inflection_points = [1, max(2, len(sentences) // 2)]

            for i, phrase in zip(inflection_points, nlp_phrases[:len(inflection_points)]):
                if i < len(sentences):
                    sentences[i] = f"{phrase}, {sentences[i].lower()}"

        return ' '.join(sentences)

    async def _enhance_with_meta_programs(self, response: str, user_analysis: Dict) -> str:
        """
        Enhance response based on user's meta-program preferences
        """

        meta_programs = user_analysis.get("nlp_patterns", {}).get("meta_programs", {})

        # Apply meta-program specific enhancements
        if meta_programs.get('chunk_size_preference') == 'big_chunk':
            # Add overview elements
            response = f"For the big picture: {response}"
        elif meta_programs.get('chunk_size_preference') == 'small_chunk':
            # Add detailed structure
            response = f"Let me break this down step by step: {response}"

        return response

    async def _optimize_communication_rhythm(self, response: str, user_analysis: Dict) -> str:
        """
        Optimize response pacing and rhythm based on user communication style
        """

        communication_style = user_analysis.get("nlp_patterns", {}).get("communication_preferences", {})

        if communication_style.get('pace_preference') == 'detailed':
            # Break into smaller units with more structure
            sentences = re.split(r'(?<=[.!?])\s+', response)
            if len(sentences) > 3:
                # Add transition phrases
                transitions = ["Moving forward,", "Additionally,", "Furthermore,", "Importantly,"]
                for i in range(1, min(len(sentences), len(transitions) + 1)):
                    sentences[i] = f"{transitions[i-1]} {sentences[i].lower()}"

                response = ' '.join(sentences)

        elif communication_style.get('pace_preference') == 'concise':
            # Make more direct and to the point
            response = response.replace("In addition", "Plus").replace("Furthermore", "Also")

        return response

    async def _update_nlp_learning(self, sensory: Dict, language: Dict, meta: Dict):
        """
        Update NLP learning models with new interaction data
        """

        # Update sensory modality learning
        dominant = sensory.get('dominant_modality')
        if dominant:
            self.sensory_patterns[dominant].append({
                "timestamp": datetime.now().isoformat(),
                "confidence": sensory.get('dominance_confidence', 0)
            })

        # Limit historical data
        for modality in self.sensory_patterns:
            if len(self.sensory_patterns[modality]) > 50:
                self.sensory_patterns[modality] = self.sensory_patterns[modality][-50:]

    def _calculate_nlp_confidence(self, sensory_modalities: Dict) -> float:
        """Calculate confidence in NLP analysis"""
        confidence_factors = []

        if sensory_modalities.get('dominance_confidence'):
            confidence_factors.append(sensory_modalities['dominance_confidence'])

        if sensory_modalities.get('nlp_patterns_detected', 0) > 0:
            confidence_factors.append(min(sensory_modalities['nlp_patterns_detected'] / 10, 1.0))

        return statistics.mean(confidence_factors) if confidence_factors else 0.0

    async def _calculate_personalization_score(self, user_analysis: Dict) -> float:
        """Calculate how well the response is personalized"""
        score = 0.5  # Base score

        # Sensory alignment
        if user_analysis.get("nlp_patterns", {}).get("sensory_modalities", {}).get("dominant_modality"):
            score += 0.2

        # Language pattern matching
        language_patterns = user_analysis.get("nlp_patterns", {}).get("language_patterns", {})
        if language_patterns and language_patterns.get("vocabulary_richness"):
            score += 0.15

        # Meta-program resonance
        if user_analysis.get("nlp_patterns", {}).get("meta_programs"):
            score += 0.15

        return min(score, 1.0)

    async def _project_nlp_effectiveness(self, user_analysis: Dict, task_context: str) -> float:
        """Project how effective NLP adaptations will be"""
        base_effectiveness = 0.7

        # Task alignment bonus
        task_keywords = task_context.lower().split()
        sensory_keywords = [word for keywords in self.sensory_keywords.values() for word in keywords]

        sensory_matches = len(set(task_keywords) & set(sensory_keywords))
        if sensory_matches > 0:
            base_effectiveness += 0.1

        # Learning history bonus
        if len(self.nlp_learning_history) > 10:
            base_effectiveness += 0.1

        return min(base_effectiveness, 0.95)

    async def get_nlp_personality_insights(self) -> Dict[str, Any]:
        """
        Comprehensive insights into NLP-based personality profile development
        """

        current_modality = self.modality_detection.get('dominant_modality', 'unknown')
        learning_depth = len(self.nlp_learning_history)

        # Calculate modality stability
        modality_stability = {}
        for modality, patterns in self.sensory_patterns.items():
            if patterns:
                confidences = [p.get('confidence', 0) for p in patterns[-10:]]  # Recent patterns
                modality_stability[modality] = statistics.mean(confidences) if confidences else 0

        # Determine learning maturity
        maturity_indicators = {
            "interactions_analyzed": learning_depth,
            "modalities_learned": len([k for k in self.sensory_patterns.keys() if self.sensory_patterns[k]]),
            "stability_achieved": learning_depth > 20 and max(modality_stability.values()) if modality_stability else 0 > 0.6,
            "adaptation_readiness": learning_depth > 10
        }

        return {
            "personality_profile": {
                "dominant_sensory_modality": current_modality,
                "secondary_modality": self.modality_detection.get('secondary_modality'),
                "nlp_learning_maturity": "mature" if maturity_indicators["stability_achieved"] else "developing",
                "communication_style": self.communication_style
            },
            "learning_progress": {
                "total_interactions_processed": learning_depth,
                "modality_confidences": modality_stability,
                "learning_velocity": maturity_indicators["interactions_analyzed"] / max(maturity_indicators["modalities_learned"], 1)
            },
            "adaptation_capabilities": {
                "response_personalization_available": maturity_indicators["adaptation_readiness"],
                "modalities_mastered": list(modality_stability.keys()),
                "best_adaptation_performance": max(modality_stability.values()) if modality_stability else 0
            },
            "nlp_insights": {
                "maturity_level": "expert" if maturity_indicators["stability_achieved"] else "intermediate",
                "personality_understanding_depth": len(self.communication_style),
                "adaptation_readiness_score": min(learning_depth / 50, 1.0)
            }
        }

# Global instance
pai_nlp_personality_engine = NLP_PersonalityEngine()
