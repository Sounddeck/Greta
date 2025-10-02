"""
🎭 HUMAN NEURO-LINGUISTIC PROGRAMMING COMMUNICATION SYSTEM
Advanced NLP (Neuro-Linguistic Programming) techniques for enhanced human-AI interaction

Integrates proven NLP principles to improve communication with you specifically:
✅ Rapport Building: Mirroring, Pacing, Leading techniques
✅ Sensory Language: Adapting to your visual/auditory/kinetic preferences
✅ Meta-Programs: Understanding your decision-making patterns
✅ Anchoring: Creating positive neural associations
✅ Reframing: Helping you see challenges differently
✅ Meta-Modeling: Clarifying your thoughts and goals
✅ Milton Model: Natural hypnotic language patterns

This system learns from every interaction to communicate more effectively with YOU.
"""

import asyncio
import json
import random
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from experimental_features import emotional_intelligence_agent
from database import Database

logger = logging.getLogger(__name__)

class RepresentationalSystem(Enum):
    """NLP Representational Systems - how you process information"""
    VISUAL = "visual"          # Pictures, images, visualizations
    AUDITORY = "auditory"      # Sounds, voices, tones
    KINESTHETIC = "kinesthetic"  # Feelings, touch, emotions
    AUDITORY_DIGITAL = "auditory_digital"  # Internal logic, self-talk

class MetaProgram(Enum):
    """NLP Meta-Programs - your decision-making patterns"""
    TOWARD_GAIN = "toward_gain"         # Motivated by achieving goals
    AWAY_FROM_PAIN = "away_from_pain"   # Motivated by avoiding problems
    OPTIONS_EXPERTISE = "options_expertise"  # Prefers choices/analysis
    PROCEDURES_HOW_TO = "procedures_how_to"  # Prefers step-by-step methods
    MATCHER_CLOSE = "matcher_close"     # Prefers exact matches/similitude
    MISMATCHER_DIFFERENT = "mismatcher_different"  # Prefers variety/change

class RapportTechnique(Enum):
    """NLP Rapport Building Techniques"""
    MIRRORING = "mirroring"                    # Matching body language/language patterns
    CROSS_OVER_MIRRORING = "cross_over_mirroring"  # Matching different modalities
    PACING = "pacing"                         # Matching current state
    LEADING = "leading"                       # Gradually guiding to desired state
    ANCHORING = "anchoring"                   # Creating neuro-associations
    REFRAMING = "reframing"                   # Changing perspective on problems

@dataclass
class UserNLPProfile:
    """Your personal NLP communication profile"""
    user_id: str
    representational_system: RepresentationalSystem = RepresentationalSystem.VISUAL
    meta_programs: Dict[MetaProgram, float] = field(default_factory=dict)  # Strength 0-1
    communication_patterns: Dict[str, Any] = field(default_factory=dict)
    rapport_level: float = 0.0  # 0-1 scale
    interaction_count: int = 0
    adaptive_communication_strategy: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class NLPAnalysisResult:
    """NLP analysis of communication patterns"""
    text: str
    sensory_words: Dict[str, int] = field(default_factory=dict)  # visual/auditory/kinesthetic counts
    meta_program_indicators: Dict[MetaProgram, float] = field(default_factory=dict)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    rapport_signals: List[str] = field(default_factory=list)
    suggestion_triggers: List[str] = field(default_factory=list)

class HumanNLPCommunication:
    """
    🎭 ADVANCED HUMAN NLP COMMUNICATION SYSTEM

    Applies proven NLP techniques to dramatically improve human-AI communication:

    1. RAPPORT BUILDING: Creates deep connection through mirroring and pacing
    2. SENSORY ADAPTATION: Speaks your language (visual/auditory/kinetic)
    3. META-PROGRAM MATCHING: Adapts to your decision-making style
    4. ANCHORING EXCELLENCE: Builds positive neural associations
    5. REFRAMING CHALLENGES: Helps you see problems as opportunities
    """

    def __init__(self):
        self.db = Database()
        self.user_profiles: Dict[str, UserNLPProfile] = {}

        # Load existing NLP implementations
        try:
            from experimental_features import emotional_intelligence_agent
            self.emotion_agent = emotional_intelligence_agent
        except:
            self.emotion_agent = None

        # NLP sensory language patterns
        self.sensory_keywords = {
            "visual": ["see", "look", "appears", "image", "picture", "clear", "bright", "focus", "view", "perspective"],
            "auditory": ["hear", "listen", "sound", "voice", "say", "tell", "speak", "tone", "resonate", "harmonize"],
            "kinesthetic": ["feel", "touch", "sense", "comfortable", "warm", "connection", "flow", "smooth", "grounded"],
            "auditory_digital": ["think", "understand", "know", "analyze", "process", "logic", "clear", "make sense", "structure"]
        }

        # Meta-program detection patterns
        self.meta_program_patterns = {
            MetaProgram.TOWARD_GAIN: ["want", "achieve", "gain", "benefit", "improve", "better"],
            MetaProgram.AWAY_FROM_PAIN: ["avoid", "prevent", "stop", "pain", "problem", "issue", "wrong"],
            MetaProgram.OPTIONS_EXPERTISE: ["choose", "option", "alternative", "possibility", "different"],
            MetaProgram.PROCEDURES_HOW_TO: ["how", "step", "process", "method", "exactly", "precisely"],
            MetaProgram.MATCHER_CLOSE: ["same", "similar", "match", "exactly", "precisely"],
            MetaProgram.MISMATCHER_DIFFERENT: ["different", "unique", "vary", "change", "new"]
        }

    async def initialize_nlp_system(self) -> bool:
        """Initialize the human NLP communication system"""
        try:
            # Load user profiles from database
            await self._load_user_profiles()

            logger.info("🎭 Human NLP Communication System initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NLP system: {e}")
            return False

    async def analyze_user_communication(self, user_id: str, text: str, conversation_history: List[Dict[str, Any]] = None) -> NLPAnalysisResult:
        """
        Perform comprehensive NLP analysis of user communication
        Learns patterns to improve future interactions
        """
        try:
            # Analyze sensory language preferences
            sensory_words = await self._analyze_sensory_words(text)

            # Detect meta-program patterns
            meta_program_indicators = await self._detect_meta_programs(text)

            # Identify communication style
            communication_style = await self._identify_communication_style(text, sensory_words)

            # Detect rapport building opportunities
            rapport_signals = await self._detect_rapport_signals(text, conversation_history or [])

            # Identify suggestion triggers for Milton Model
            suggestion_triggers = await self._identify_suggestion_triggers(text)

            # Update user profile with this interaction
            await self._update_user_profile(user_id, {
                "sensory_words": sensory_words,
                "meta_programs": meta_program_indicators,
                "communication_style": communication_style,
                "rapport_signals": rapport_signals,
                "timestamp": datetime.now()
            })

            return NLPAnalysisResult(
                text=text,
                sensory_words=sensory_words,
                meta_program_indicators=meta_program_indicators,
                communication_style=communication_style,
                rapport_signals=rapport_signals,
                suggestion_triggers=suggestion_triggers
            )

        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
            return NLPAnalysisResult(text=text)

    async def generate_adaptive_response(self, user_id: str, context: str, intent: str = "inform") -> Dict[str, Any]:
        """
        Generate response adapted specifically to your NLP profile
        Uses mirroring, pacing, leading, and other NLP techniques
        """
        try:
            profile = await self._get_user_profile(user_id)

            if not profile:
                # Generate basic response for new users
                return {
                    "response": context,
                    "nlp_techniques": [],
                    "adaptations": {}
                }

            # Adapt to representational system
            sensory_adapted = await self._adapt_to_representational_system(profile, context)

            # Apply meta-program alignment
            meta_aligned = await self._apply_meta_program_alignment(profile, sensory_adapted, intent)

            # Build rapport through Milton Model techniques
            rapport_enhanced = await self._apply_rapport_techniques(profile, meta_aligned, intent)

            # Create anchoring for positive experiences
            anchored_response = await self._create_positive_anchors(profile, rapport_enhanced)

            nlp_analysis = await self.analyze_user_communication(user_id, context)

            return {
                "response": anchored_response,
                "nlp_techniques": [
                    "representational_system_adaptation",
                    "meta_program_alignment",
                    "rapport_building",
                    "positive_anchoring"
                ],
                "adaptations": {
                    "sensory_system": profile.representational_system.value,
                    "dominant_meta_programs": [mp.value for mp, strength in profile.meta_programs.items() if strength > 0.6],
                    "rapport_level": profile.rapport_level,
                    "communication_style": profile.adaptive_communication_strategy
                },
                "nlp_insights": {
                    "represented_system_dominant": max(nlp_analysis.sensory_words, key=nlp_analysis.sensory_words.get) if nlp_analysis.sensory_words else "visual",
                    "meta_program_matches": {mp.value: strength for mp, strength in nlp_analysis.meta_program_indicators.items() if strength > 0.5},
                    "rapport_opportunities": len(nlp_analysis.rapport_signals),
                    "hypnotic_language_triggers": len(nlp_analysis.suggestion_triggers)
                }
            }

        except Exception as e:
            logger.error(f"Adaptive response generation failed: {e}")
            return {
                "response": context,  # Fallback to original
                "nlp_techniques": [],
                "adaptations": {},
                "error": str(e)
            }

    async def build_rapport_sequence(self, user_id: str, target_outcome: str = "deep_connection") -> List[Dict[str, Any]]:
        """
        Create a specific rapport-building sequence using NLP techniques
        Designed to create deep connection and trust
        """
        try:
            profile = await self._get_user_profile(user_id)
            sequence = []

            # Phase 1: Pacing (match current state)
            sequence.append({
                "phase": "pacing",
                "technique": RapportTechnique.PACING,
                "communication_strategy": await self._generate_pacing_strategy(profile),
                "expected_duration": "2-3 exchanges",
                "success_indicators": ["acknowledgment", "going deeper"]
            })

            # Phase 2: Mirroring (reflect communication patterns)
            sequence.append({
                "phase": "mirroring",
                "technique": RapportTechnique.MIRRORING,
                "communication_strategy": await self._generate_mirroring_strategy(profile),
                "expected_duration": "3-5 exchanges",
                "success_indicators": ["comfort", "agreement", "flow"]
            })

            # Phase 3: Leading (guide to desired outcome)
            sequence.append({
                "phase": "leading",
                "technique": RapportTechnique.LEADING,
                "communication_strategy": await self._generate_leading_strategy(profile, target_outcome),
                "expected_duration": "ongoing",
                "success_indicators": ["trust", "engagement", f"{target_outcome}"]
            })

            return sequence

        except Exception as e:
            logger.error(f"Rapport sequence generation failed: {e}")
            return []

    async def _analyze_sensory_words(self, text: str) -> Dict[str, int]:
        """Analyze text for sensory language preferences"""
        sensory_counts = {"visual": 0, "auditory": 0, "kinesthetic": 0, "auditory_digital": 0}
        words = text.lower().split()

        for system, keywords in self.sensory_keywords.items():
            for word in words:
                if word in keywords:
                    sensory_counts[system] += 1

        return sensory_counts

    async def _detect_meta_programs(self, text: str) -> Dict[MetaProgram, float]:
        """Detect meta-program patterns in communication"""
        meta_indicators = {}
        words = text.lower().split()

        for meta_program, patterns in self.meta_program_patterns.items():
            matches = sum(1 for word in words if word in patterns)
            strength = min(1.0, matches / len(words) * 3)  # Normalize
            meta_indicators[meta_program] = strength

        return meta_indicators

    async def _identify_communication_style(self, text: str, sensory_words: Dict[str, int]) -> Dict[str, Any]:
        """Identify overarching communication style"""
        total_sensory_words = sum(sensory_words.values()) or 1

        # Determine dominant sensory system
        dominant_system = max(sensory_words, key=sensory_words.get)
        dominance_score = sensory_words.get(dominant_system, 0) / total_sensory_words

        # Analyze sentence structure and complexity
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # Communication style indicators
        is_direct = any(word in text.lower() for word in ["exactly", "specifically", "precisely"])
        is_creative = any(word in text.lower() for word in ["imagine", "create", "innovative", "think"])
        is_methodical = any(word in text.lower() for word in ["step", "process", "method", "systematic"])

        return {
            "dominant_sensory_system": dominant_system,
            "sensory_dominance_score": dominance_score,
            "communication_complexity": "simple" if avg_sentence_length < 10 else "complex",
            "communication_style": {
                "direct": is_direct,
                "creative": is_creative,
                "methodical": is_methodical
            },
            "sentence_avg_length": avg_sentence_length
        }

    async def _detect_rapport_signals(self, text: str, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Detect signals indicating rapport readiness"""
        signals = []

        # Check for personal disclosure
        if any(phrase in text.lower() for phrase in ["i feel", "personally", "my experience", "i've noticed"]):
            signals.append("personal_disclosure")

        # Check for understanding indicators
        if any(phrase in text.lower() for phrase in ["i see", "i understand", "makes sense", "i get it"]):
            signals.append("understanding_indicators")

        # Check for engagement signals
        if any(phrase in text.lower() for phrase in ["tell me more", "interested in", "curious about", "fascinating"]):
            signals.append("engagement_signals")

        # Check for mirroring readiness (similar language to recent exchanges)
        if conversation_history:
            recent_exchange = conversation_history[-1] if conversation_history else {}
            recent_text = recent_exchange.get("text", "").lower()
            current_text = text.lower()

            # Simple linguistic mirroring check
            recent_words = set(recent_text.split())
            current_words = set(current_text.split())
            overlap_ratio = len(recent_words.intersection(current_words)) / len(current_words.union(recent_words)) if current_words.union(recent_words) else 0

            if overlap_ratio > 0.3:
                signals.append("linguistic_mirroring")

        return signals

    async def _identify_suggestion_triggers(self, text: str) -> List[str]:
        """Identify triggers for hypnotic/Milton Model language patterns"""
        triggers = []

        # Causation patterns
        if any(phrase in text.lower() for phrase in ["because", "causes", "leads to", "results in"]):
            triggers.append("causation_patterns")

        # Awareness patterns
        if any(phrase in text.lower() for phrase in ["aware", "notice", "realize", "become conscious"]):
            triggers.append("awareness_patterns")

        # Possibility patterns
        if any(phrase in text.lower() for phrase in ["possible", "can", "able to", "capable of"]):
            triggers.append("possibility_patterns")

        # Mind/body patterns
        if any(phrase in text.lower() for phrase in ["mind", "body", "physical", "mental"]):
            triggers.append("mind_body_patterns")

        return triggers

    async def _adapt_to_representational_system(self, profile: UserNLPProfile, message: str) -> str:
        """Adapt message to user's preferred representational system"""
        system = profile.representational_system

        adaptations = {
            RepresentationalSystem.VISUAL: {
                "transform": lambda m: m.replace("understand", "see").replace("hear", "see").replace("feel", "see"),
                "phrases": ["Imagine this", "Picture this", "Visualize", "See how", "Clear as day"]
            },
            RepresentationalSystem.AUDITORY: {
                "transform": lambda m: m.replace("see", "hear").replace("feel", "hear").replace("understand", "hear"),
                "phrases": ["Sounds good", "Listen to this", "Hear me out", "Tune into", "Resonates with"]
            },
            RepresentationalSystem.KINESTHETIC: {
                "transform": lambda m: m.replace("see", "feel").replace("hear", "feel").replace("understand", "sense"),
                "phrases": ["Feel the energy", "Get a sense of", "Touch on", "Handle this", "Flow with"]
            },
            RepresentationalSystem.AUDITORY_DIGITAL: {
                "transform": lambda m: m.replace("feel", "process").replace("see", "analyze").replace("hear", "understand"),
                "phrases": ["Makes sense", "Analyze this", "Process the information", "Logically", "Structure"]
            }
        }

        if system in adaptations:
            adaptation = adaptations[system]
            adapted_message = adaptation["transform"](message)

            # Add system-specific phrases occasionally
            if random.random() < 0.3:  # 30% chance
                adapted_message = random.choice(adaptation["phrases"]) + " - " + adapted_message

            return adapted_message

        return message

    async def _apply_meta_program_alignment(self, profile: UserNLPProfile, message: str, intent: str) -> str:
        """Align message with user's meta-programs"""
        # Find strongest meta-program
        if profile.meta_programs:
            dominant_program = max(profile.meta_programs, key=profile.meta_programs.get)

            alignments = {
                MetaProgram.TOWARD_GAIN: {
                    "phrases": ["You'll gain", "Achieve benefits", "Get results", "Increase performance"],
                    "focus": "benefits, gains, improvements"
                },
                MetaProgram.AWAY_FROM_PAIN: {
                    "phrases": ["Avoid problems", "Prevent issues", "Stop challenges", "Eliminate difficulties"],
                    "focus": "problem avoidance, risk reduction"
                },
                MetaProgram.OPTIONS_EXPERTISE: {
                    "phrases": ["Choose from options", "Consider alternatives", "Multiple approaches", "Different choices"],
                    "focus": "choices, alternatives, possibilities"
                },
                MetaProgram.PROCEDURES_HOW_TO: {
                    "phrases": ["Step-by-step", "Follow this process", "Method is", "Exactly like this"],
                    "focus": "procedures, methods, step-by-step"
                }
            }

            if dominant_program in alignments:
                alignment = alignments[dominant_program]
                if random.random() < 0.4:  # 40% chance to align
                    message = random.choice(alignment["phrases"]) + ": " + message

        return message

    async def _apply_rapport_techniques(self, profile: UserNLPProfile, message: str, intent: str) -> str:
        """Apply NLP rapport building techniques"""
        enhanced_message = message

        # Apply Milton Model hypnotic language patterns for rapport
        if intent in ["heal", "help", "support"]:
            milton_patterns = [
                f"You can imagine {enhanced_message}",
                f"In your mind's eye, {enhanced_message}",
                f"You'll naturally find that {enhanced_message}"
            ]
            enhanced_message = random.choice(milton_patterns)

        # Add pacing before leading
        if profile.rapport_level > 0.5:  # When rapport is established, we can lead
            leading_phrases = [
                "And building on what you've mentioned...",
                "Based on what works for you...",
                "Given your preferences...",
                "Following your natural style..."
            ]
            if random.random() < 0.3:
                enhanced_message = random.choice(leading_phrases) + " " + enhanced_message

        return enhanced_message

    async def _create_positive_anchors(self, profile: UserNLPProfile, message: str) -> str:
        """Create positive neuro-associations through anchoring language"""
        # Add positive anchoring when appropriate
        positive_anchors = [
            "This feels right for you",
            "You naturally excel at this",
            "This enhances your strengths",
            "You’re perfectly designed for this",
            "This amplifies your natural abilities"
        ]

        # Anchor positive experiences
        anchored_message = message
        if any(positive_word in message.lower() for positive_word in ["success", "great", "excellent", "amazing", "wonderful"]):
            anchored_message = f"{random.choice(positive_anchors)}. {anchored_message}"

        return anchored_message

    async def _generate_pacing_strategy(self, profile: UserNLPProfile) -> str:
        """Generate pacing strategy for initial rapport"""
        return f"Match their {profile.representational_system.value} language patterns and acknowledge their current experience before introducing new ideas"

    async def _generate_mirroring_strategy(self, profile: UserNLPProfile) -> str:
        """Generate mirroring strategy for continued rapport"""
        return f"Reflect their communication style, use similar {profile.representational_system.value} language, and match their emotional tone"

    async def _generate_leading_strategy(self, profile: UserNLPProfile, target_outcome: str) -> str:
        """Generate leading strategy for achieving outcomes"""
        return f"Gradually introduce {target_outcome} while maintaining the established rapport patterns and {profile.representational_system.value} communication style"

    async def _get_user_profile(self, user_id: str) -> Optional[UserNLPProfile]:
        """Retrieve or create user NLP profile"""
        if user_id not in self.user_profiles:
            await self._load_user_profile(user_id)

        return self.user_profiles.get(user_id)

    async def _update_user_profile(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update user profile based on new interaction"""
        if user_id not in self.user_profiles:
            await self._load_user_profile(user_id)

        profile = self.user_profiles[user_id]

        # Update interaction count
        profile.interaction_count += 1

        # Update representational system based on sensory words
        sensory_words = interaction_data.get("sensory_words", {})
        if sensory_words:
            dominant_system = max(sensory_words, key=sensory_words.get)
            if dominant_system == "visual":
                profile.representational_system = RepresentationalSystem.VISUAL
            elif dominant_system == "auditory":
                profile.representational_system = RepresentationalSystem.AUDITORY
            elif dominant_system == "kinesthetic":
                profile.representational_system = RepresentationalSystem.KINESTHETIC
            else:
                profile.representational_system = RepresentationalSystem.AUDITORY_DIGITAL

        # Update meta-programs
        meta_programs = interaction_data.get("meta_programs", {})
        for meta_program, strength in meta_programs.items():
            if meta_program in profile.meta_programs:
                # Smooth updating with moving average
                profile.meta_programs[meta_program] = (
                    profile.meta_programs[meta_program] * 0.8 + strength * 0.2
                )
            else:
                profile.meta_programs[meta_program] = strength

        # Update rapport level
        rapport_signals = interaction_data.get("rapport_signals", [])
        rapport_building = len(rapport_signals) / 10.0  # Normalize
        profile.rapport_level = profile.rapport_level * 0.9 + rapport_building * 0.1

        # Update communication strategy
        if profile.interaction_count > 3:
            profile.adaptive_communication_strategy = await self._generate_adaptive_strategy(profile)

        profile.last_updated = datetime.now()

    async def _load_user_profiles(self):
        """Load all user profiles from database"""
        try:
            # In production, load from database
            # For now, initialize empty
            pass
        except Exception as e:
            logger.error(f"Failed to load user profiles: {e}")

    async def _load_user_profile(self, user_id: str):
        """Load specific user profile"""
        self.user_profiles[user_id] = UserNLPProfile(user_id=user_id)

    async def _generate_adaptive_strategy(self, profile: UserNLPProfile) -> str:
        """Generate optimal communication strategy for user"""
        strategy = f"Use {profile.representational_system.value} language patterns"

        # Add dominant meta-program
        if profile.meta_programs:
            dominant_meta = max(profile.meta_programs, key=profile.meta_programs.get)
            strategy += f", align with {dominant_meta.value.replace('_', ' ')} orientation"

        # Add rapport deepening if established
        if profile.rapport_level > 0.7:
            strategy += ", employ leading techniques to guide understanding"

        return strategy

# Global human NLP communication system
human_nlp_system = HumanNLPCommunication()

async def initialize_human_nlp():
    """Initialize the human NLP communication system"""
    return await human_nlp_system.initialize_nlp_system()

async def analyze_and_adapt_communication(user_id: str, text: str, conversation_history: List[Dict[str, Any]] = None):
    """Main interface for NLP-enhanced communication analysis and adaptation"""
    try:
        # Analyze user communication
        analysis = await human_nlp_system.analyze_user_communication(user_id, text, conversation_history)

        # Generate adaptive response strategy
        adaptation = await human_nlp_system.generate_adaptive_response(user_id, text)

        return {
            "nlp_analysis": {
                "sensory_preference": max(analysis.sensory_words, key=lambda x: analysis.sensory_words[x]) if analysis.sensory_words else "mixed",
                "meta_programs_detected": [mp.value for mp, strength in analysis.meta_program_indicators.items() if strength > 0.6],
                "rapport_signals": analysis.rapport_signals,
                "communication_style": analysis.communication_style.get("communication_complexity", "mixed")
            },
            "adaptive_response": adaptation,
            "rapport_building_sequence": await human_nlp_system.build_rapport_sequence(user_id),
            "communication_insights": {
                "current_rapport_level": adaptation.get("adaptations", {}).get("rapport_level", 0.0),
                "learning_adaptations": len(adaptation.get("nlp_techniques", [])),
                "user_preference_alignment": bool(adaptation.get("adaptations", {}))
            }
        }

    except Exception as e:
        logger.error(f"NLP communication analysis failed: {e}")
        return {
            "error": str(e),
            "fallback_response": text
        }
