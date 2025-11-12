"""
PAI Proactive Intelligence Pipeline
Transcends reactive assistance to provide anticipatory, context-aware intelligence that enhances human capability
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re
import math
from backend.services.memory_orchestrator import pai_memory_orchestrator

class ProactiveIntelligencePipeline:
    """
    PAI Proactive Intelligence: Transforms Greta from reactive assistant to intelligence partner.
    Anticipates needs, provides context-aware assistance, and learns from interaction patterns.
    """

    def __init__(self):
        # Proactive intelligence state
        self.anticipatory_memory = {}      # Learned anticipatory patterns
        self.workflow_predictor = {}      # Predicts next steps in user workflows
        self.opportunity_detector = {}     # Identifies helping moments
        self.proactive_history = deque(maxlen=100)  # Track proactive success

        # Learning and adaptation
        self.user_workflow_patterns = defaultdict(list)
        self.successful_interventions = []
        self.missed_opportunities = []

        # Confidence thresholds
        self.min_suggestion_confidence = 0.75
        self.max_false_positive_rate = 0.1

        print("PAI Proactive Intelligence Pipeline initialized - anticipatory assistance active")

    async def analyze_current_context(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proactively analyze current context to identify assistance opportunities
        """

        analysis_start = asyncio.get_event_loop().time()

        # Multi-dimensional context assessment
        temporal_signals = await self._analyze_temporal_signals(user_context)
        behavior_patterns = await self._analyze_behavior_patterns(user_context)
        workflow_projection = await self._project_workflow_needs(user_context)
        resource_optimization = await self._optimizing_resource_opportunities(user_context)

        # Synthesize proactive opportunities
        proactive_opportunities = await self._synthesize_opportunities(
            temporal_signals, behavior_patterns, workflow_projection, resource_optimization
        )

        # Ensure quality and relevance
        quality_filtered = await self._quality_filter_opportunities(proactive_opportunities)

        processing_time = asyncio.get_event_loop().time() - analysis_start

        return {
            "context_analysis": {
                "temporal_signals": temporal_signals,
                "behavior_patterns": behavior_patterns,
                "workflow_projection": workflow_projection,
                "resource_optimization": resource_optimization
            },
            "proactive_opportunities": quality_filtered,
            "analysis_confidence": self._calculate_analysis_confidence(quality_filtered),
            "processing_time": round(processing_time, 2),
            "pai_proactive_stage": "analysis_complete"
        }

    async def generate_proactive_assistance(self, context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable proactive assistance based on context analysis
        """

        opportunities = context_analysis["proactive_opportunities"]
        recommendations = []

        for opportunity in opportunities:
            if opportunity["confidence"] >= self.min_suggestion_confidence:

                # Generate contextually relevant assistance
                assistance = await self._create_targeted_assistance(opportunity)

                if assistance:
                    recommendations.append({
                        "opportunity": opportunity,
                        "assistance": assistance,
                        "timeliness_score": await self._calculate_timeliness_score(opportunity),
                        "expected_value": opportunity.get("expected_impact", "medium")
                    })

        # Rank and filter based on predicted effectiveness
        ranked_recommendations = await self._rank_recommendations(recommendations)

        return {
            "proactive_recommendations": ranked_recommendations,
            "total_opportunities": len(opportunities),
            "high_confidence_suggestions": len([r for r in ranked_recommendations if r.get("timeliness_score", 0) > 0.8]),
            "anticipatory_intelligence_active": True
        }

    async def learn_from_proactive_feedback(self, assistance_delivered: Dict[str, Any], user_response: Dict[str, Any]):
        """
        Learn from proactive assistance effectiveness to improve future suggestions
        """

        # Analyze effectiveness
        effectiveness = await self._evaluate_assistance_effectiveness(assistance_delivered, user_response)

        # Update learning models
        await self._update_user_workflow_models(assistance_delivered, effectiveness)
        await self._update_anticipatory_memory(assistance_delivered, effectiveness)
        await self._update_opportunity_detection_models(assistance_delivered, effectiveness)

        # Store learning instance
        learning_instance = {
            "assistance": assistance_delivered,
            "user_response": user_response,
            "effectiveness_score": effectiveness,
            "timestamp": datetime.now().isoformat(),
            "learning_processed": True
        }

        self.proactive_history.append(learning_instance)

    async def get_proactive_statistics(self) -> Dict[str, Any]:
        """
        Comprehensive statistics on proactive intelligence performance
        """

        total_interventions = len(self.proactive_history)
        if total_interventions == 0:
            return {"status": "no_data", "message": "Proactive intelligence learning in progress"}

        # Calculate effectiveness metrics
        effectiveness_scores = [h["effectiveness_score"] for h in self.proactive_history]
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)

        # Calculate hit rates
        successful_interventions = len([s for s in effectiveness_scores if s >= 0.7])
        hit_rate = successful_interventions / total_interventions if total_interventions > 0 else 0

        # Calculate learning growth over time
        recent_performance = effectiveness_scores[-20:] if len(effectiveness_scores) > 20 else effectiveness_scores
        recent_avg = sum(recent_performance) / len(recent_performance)

        return {
            "overall_performance": {
                "total_interventions": total_interventions,
                "average_effectiveness": round(avg_effectiveness, 2),
                "hit_rate": round(hit_rate, 2),
                "learning_improvement": round(recent_avg - effectiveness_scores[0], 2) if len(effectiveness_scores) > 1 else 0
            },
            "recent_performance": {
                "last_20_interventions": recent_avg,
                "trend": "improving" if recent_avg > avg_effectiveness + 0.05 else "stable",
                "confidence_trend": await self._analyze_confidence_trend()
            },
            "proactive_capabilities": {
                "active_opportunity_detectors": len(self.opportunity_detector),
                "learned_workflow_patterns": len(self.user_workflow_patterns),
                "anticipatory_memory_items": len(self.anticipatory_memory)
            },
            "system_health": {
                "learning_active": len(self.proactive_history) >= 10,
                "pattern_recognition_mature": len(self.user_workflow_patterns) >= 5,
                "false_positive_rate": await self._calculate_false_positive_rate()
            }
        }

    # Core proactive analysis methods
    async def _analyze_temporal_signals(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal patterns that suggest proactive assistance opportunities
        """

        current_time = datetime.now()

        # Time-based signals
        signals = {
            "end_of_workday_warning": False,
            "start_of_competitive_period": False,
            "deadline_approaching": False,
            "regular_pattern_interrupt": False,
            "time_pressure_indicators": []
        }

        # Analyze recent interactions for temporal patterns
        recent_memory = await pai_memory_orchestrator.contextual_retrieval({
            "time_range": "24h",
            "query_type": "temporal_patterns"
        })

        # Detect common proactive moments
        if await self._is_end_of_focused_work_block(user_context):
            signals["end_of_workday_warning"] = True

        if await self._detect_periodic_activity_drop(user_context):
            signals["regular_pattern_interrupt"] = True

        approaching_deadlines = await self._analyze_approaching_deadlines(user_context)
        if approaching_deadlines:
            signals["deadline_approaching"] = True
            signals["time_pressure_indicators"] = approaching_deadlines

        return signals

    async def _analyze_behavior_patterns(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user behavior patterns to predict proactive needs
        """

        patterns = {
            "problem_solving_phase": False,
            "information_gathering_mode": False,
            "decision_making_context": False,
            "creative_block_indicators": False,
            "learning_opportunity": False
        }

        # Analyze current and recent interactions
        recent_interactions = await pai_memory_orchestrator.contextual_retrieval({
            "time_range": "1h",
            "query_type": "behavior_analysis"
        })

        # Detect behavioral indicators
        if await self._detect_repeated_questions(user_context):
            patterns["information_gathering_mode"] = True

        if await self._detect_frustration_indicators(user_context):
            patterns["problem_solving_phase"] = True

        if await self._detect_decision_indicators(user_context):
            patterns["decision_making_context"] = True

        if await self._detect_learning_opportunities(user_context):
            patterns["learning_opportunity"] = True

        return patterns

    async def _project_workflow_needs(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project what user might need next based on current workflow
        """

        current_activity = user_context.get("current_activity", "")
        recent_activities = await self._get_recent_activity_sequence(user_context)

        # Project next logical steps
        projections = await self._calculate_workflow_probabilities(recent_activities, current_activity)

        return {
            "current_context": current_activity,
            "likely_next_steps": projections.get("top_predictions", []),
            "workflow_confidence": projections.get("confidence", 0),
            "alternative_paths": projections.get("alternative_paths", [])
        }

    async def _synthesizing_opportunities(self, temporal: Dict, behavioral: Dict, workflow: Dict, resource: Dict, user_context: Dict) -> List[Dict[str, Any]]:
        """
        Synthesize all signals into proactive assistance opportunities
        """

        opportunities = []

        # Create opportunities based on signal combinations

        # Time + Workflow opportunities
        if temporal.get("deadline_approaching") and workflow.get("workflow_confidence", 0) > 0.6:
            opportunities.append({
                "type": "deadline_workflow_assistance",
                "description": "Help optimize workflow completion before deadline",
                "confidence": min(temporal.get("deadline_importance", 0) * workflow.get("workflow_confidence", 0), 0.95),
                "priority": "high",
                "suggestion_trigger": "deadline_detected",
                "expected_impact": "high"
            })

        # Behavior pattern opportunities
        if behavioral.get("problem_solving_phase"):
            opportunities.append({
                "type": "problem_solving_support",
                "description": "Provide structured problem-solving guidance",
                "confidence": 0.85,
                "priority": "medium",
                "suggestion_trigger": "problem_indicators_detected",
                "expected_impact": "high"
            })

        # Learning opportunities
        if behavioral.get("learning_opportunity") and user_context.get("skill_level") != "expert":
            opportunities.append({
                "type": "learning_enhancement",
                "description": "Offer targeted learning resources or examples",
                "confidence": 0.75,
                "priority": "low",
                "suggestion_trigger": "learning_moment_detected",
                "expected_impact": "medium"
            })

        # Resource optimization opportunities
        if resource.get("resource_constraint"):
            opportunities.append({
                "type": "resource_optimization",
                "description": "Suggest alternate approaches to resource constraints",
                "confidence": resource.get("constraint_confidence", 0.6),
                "priority": "medium",
                "suggestion_trigger": "resource_limit_detected",
                "expected_impact": "high"
            })

        return opportunities

    async def _create_targeted_assistance(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create specific, actionable assistance for the identified opportunity
        """

        opportunity_type = opportunity["type"]

        if opportunity_type == "deadline_workflow_assistance":
            return {
                "title": "Deadline Workflow Optimization",
                "message": "I notice you might be working towards a deadline. Would you like me to help organize your remaining tasks and optimize your workflow?",
                "suggested_actions": [
                    "Prioritize remaining tasks",
                    "Delegate lower-priority items",
                    "Set up focused work sessions"
                ],
                "value_proposition": "Complete more efficiently while reducing stress",
                "response_type": "workflow_optimization"
            }

        elif opportunity_type == "problem_solving_support":
            return {
                "title": "Problem Solving Framework",
                "message": "I can see you're tackling a complex problem. This structured approach might help:",
                "suggested_actions": [
                    "Break down the problem into smaller components",
                    "Gather all relevant information first",
                    "Consider multiple solution approaches",
                    "Test assumptions systematically"
                ],
                "value_proposition": "More systematic and thorough problem resolution",
                "response_type": "structured_guidance"
            }

        elif opportunity_type == "resource_optimization":
            return {
                "title": "Resource Optimization Suggestions",
                "message": "Working with limited resources? Here are some proven optimization strategies:",
                "suggested_actions": [
                    "Identify highest-impact activities",
                    "Look for process efficiencies",
                    "Consider alternate approaches",
                    "Prioritize based on expected outcomes"
                ],
                "value_proposition": "Achieve goals with available resources",
                "response_type": "resource_strategy"
            }

        elif opportunity_type == "learning_enhancement":
            return {
                "title": "Learning Enhancement",
                "message": "This seems like a great learning opportunity. Here are ways to make it more effective:",
                "suggested_actions": [
                    "Focus on understanding principles, not just procedures",
                    "Try a practical example",
                    "Ask specific questions about confusing parts",
                    "Connect to prior knowledge"
                ],
                "value_proposition": "Deeper, more lasting learning",
                "response_type": "educational_support"
            }

        return None

    async def _calculate_timeliness_score(self, opportunity: Dict[str, Any]) -> float:
        """
        Calculate how timely the assistance would be (higher = more urgent/valuable)
        """

        base_timeliness = opportunity.get("confidence", 0.5)

        # Adjust based on opportunity type urgency
        urgency_multipliers = {
            "deadline_workflow_assistance": 1.3,
            "problem_solving_support": 1.1,
            "resource_optimization": 1.2,
            "learning_enhancement": 0.9
        }

        urgency_multiplier = urgency_multipliers.get(opportunity.get("type"), 1.0)

        # Cap at reasonable maximum
        return min(base_timeliness * urgency_multiplier, 0.95)

    async def _rank_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank recommendations by effectiveness and timeliness
        """

        for rec in recommendations:
            # Combined score of confidence and timeliness
            effectiveness_score = (
                rec["opportunity"]["confidence"] * 0.6 +
                rec["timeliness_score"] * 0.4
            )

            # Bonus for high expected impact
            if rec.get("expected_value") == "high":
                effectiveness_score *= 1.1

            rec["overall_effectiveness"] = min(effectiveness_score, 1.0)

        # Sort by effectiveness (highest first)
        recommendations.sort(key=lambda x: x["overall_effectiveness"], reverse=True)

        return recommendations[:5]  # Return top 5 most effective

    # Helper and utility methods
    async def _is_end_of_focused_work_block(self, user_context: Dict) -> bool:
        # Placeholder - would analyze user activity patterns
        return False

    async def _detect_periodic_activity_drop(self, user_context: Dict) -> bool:
        # Placeholder - would analyze activity levels
        return False

    async def _analyze_approaching_deadlines(self, user_context: Dict) -> List[str]:
        # Placeholder - would analyze user calendar/tasks
        return []

    async def _optimizing_resource_opportunities(self, user_context: Dict) -> Dict[str, Any]:
        # Placeholder - would analyze resource availability
        return {"resource_constraint": False, "constraint_confidence": 0}

    async def _quality_filter_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter opportunities based on quality thresholds"""
        return [opp for opp in opportunities if opp["confidence"] >= self.min_suggestion_confidence]

    async def _calculate_analysis_confidence(self, opportunities: List[Dict]) -> float:
        """Calculate overall confidence in the proactive analysis"""
        if not opportunities:
            return 0.0

        confidence_avg = sum(opp["confidence"] for opp in opportunities) / len(opportunities)
        confidence_std = 0.8  # Simplified - real implementation would calculate standard deviation

        # Higher confidence when we have more opportunities with consistent confidence
        return min(confidence_avg * (1 + len(opportunities) * 0.1), 0.95)

    async def _evaluate_assistance_effectiveness(self, assistance: Dict, user_response: Dict) -> float:
        """Evaluate how effective the proactive assistance was"""

        # Analyze user response to gauge effectiveness
        response_sentiment = await self._analyze_response_sentiment(user_response)
        engagement_level = self._measure_user_engagement(user_response)
        follow_through_indicators = await self._check_follow_through(assistance, user_response)

        # Combined effectiveness score
        effectiveness = (
            response_sentiment * 0.3 +
            engagement_level * 0.4 +
            follow_through_indicators * 0.3
        )

        return min(max(effectiveness, 0.0), 1.0)

    async def _update_user_workflow_models(self, assistance: Dict, effectiveness: float):
        """Update learned user workflow patterns"""
        pass  # Implementation would update user workflow models

    async def _update_anticipatory_memory(self, assistance: Dict, effectiveness: float):
        """Update proactive memory with new learning"""
        pass  # Implementation would update anticipatory patterns

    async def _update_opportunity_detection_models(self, assistance: Dict, effectiveness: float):
        """Update opportunity detection based on success/failure"""
        pass  # Implementation would refine detection algorithms

    async def _analyze_response_sentiment(self, user_response: Dict) -> float:
        # Simplified sentiment analysis
        content = str(user_response).lower()
        positive_indicators = ["thanks", "helpful", "great", "useful", "yes", "good"]
        negative_indicators = ["no", "don't", "stop", "not helpful", "ignore"]

        positive_count = sum(1 for word in positive_indicators if word in content)
        negative_count = sum(1 for word in negative_indicators if word in content)

        if positive_count > negative_count:
            return 0.8
        elif negative_count > positive_count:
            return 0.2
        else:
            return 0.5

    def _measure_user_engagement(self, user_response: Dict) -> float:
        # Simplified engagement measurement
        response_length = len(str(user_response))
        interaction_complexity = 0.5  # Placeholder

        return min((response_length / 1000) + interaction_complexity, 1.0)

    async def _check_follow_through(self, assistance: Dict, user_response: Dict) -> float:
        # Check if user acted on suggestions
        return 0.6  # Placeholder

    async def _detect_repeated_questions(self, user_context: Dict) -> bool:
        # Has user asked multiple questions recently?
        return False  # Placeholder

    async def _detect_frustration_indicators(self, user_context: Dict) -> bool:
        # Signs of frustration or struggle?
        return False  # Placeholder

    async def _detect_decision_indicators(self, user_context: Dict) -> bool:
        # Is user making decisions?
        return False  # Placeholder

    async def _detect_learning_opportunities(self, user_context: Dict) -> bool:
        # Good moment for learning support?
        return user_context.get("knowledge_gap_indicators", 0) > 0.6

    async def _get_recent_activity_sequence(self, user_context: Dict) -> List[str]:
        # Get recent user activity sequence
        return ["task_start", "research", "analysis"]  # Placeholder

    async def _calculate_workflow_probabilities(self, activities: List[str], current: str) -> Dict:
        # Calculate next likely activities
        return {
            "top_predictions": ["documentation", "testing", "deployment"],
            "confidence": 0.7,
            "alternative_paths": ["review", "refinement"]
        }

    async def _analyze_confidence_trend(self) -> str:
        # Analyze if proactive confidence is improving
        return "stable"  # Placeholder

    async def _calculate_false_positive_rate(self) -> float:
        # Calculate rate of unhelpful suggestions
        return 0.05  # Placeholder

# Global instance
pai_proactive_intelligence = ProactiveIntelligencePipeline()
