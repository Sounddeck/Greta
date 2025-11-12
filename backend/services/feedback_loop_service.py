"""
Continuous Learning Feedback Loop Service

Implements continuous learning through:
- Success/failure tracking
- Automatic pattern refinement
- Skill confidence adjustment
- User feedback integration
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import uuid4
from pydantic import BaseModel, Field

from services.reasoning_bank_service import ReasoningBankService
from services.skills_service import skills_service
from services.tool_recommendation_service import ToolRecommendationService
from database import Database

logger = logging.getLogger(__name__)


# ==================== MODELS ====================

class FeedbackType(str):
    """Types of feedback"""
    SUCCESS = "success"
    FAILURE = "failure"
    USER_RATING = "user_rating"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"


class FeedbackEntry(BaseModel):
    """A single feedback entry"""
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    feedback_type: str  # FeedbackType
    entity_type: str  # "workflow", "tool", "skill", "pattern"
    entity_id: str
    success: Optional[bool] = None
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    user_comment: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"


class LearningMetrics(BaseModel):
    """Metrics for learning progress"""
    total_feedback_entries: int = 0
    success_rate: float = 0.0
    average_rating: float = 0.0
    patterns_refined: int = 0
    skills_adjusted: int = 0
    tools_improved: int = 0
    last_learning_cycle: Optional[datetime] = None


class SkillConfidenceAdjustment(BaseModel):
    """Adjustment to skill confidence"""
    skill_id: str
    old_confidence: float
    new_confidence: float
    reason: str
    adjusted_at: datetime = Field(default_factory=datetime.utcnow)


# ==================== SERVICE ====================

class FeedbackLoopService:
    """
    Continuous learning feedback loop service
    
    Tracks success/failure, refines patterns, adjusts skill confidence,
    and integrates user feedback to improve system performance over time.
    """
    
    def __init__(
        self,
        db: Database,
        reasoning_bank: Optional[ReasoningBankService] = None,
        tool_recommendation: Optional[ToolRecommendationService] = None
    ):
        self.db = db
        self.reasoning_bank = reasoning_bank
        self.tool_recommendation = tool_recommendation
        
        # Collections
        self.feedback_collection = db.get_collection("feedback_entries")
        self.metrics_collection = db.get_collection("learning_metrics")
        self.adjustments_collection = db.get_collection("skill_adjustments")
        
        # In-memory cache
        self.recent_feedback: List[FeedbackEntry] = []
        self.learning_metrics = LearningMetrics()
        
        # Configuration
        self.learning_cycle_interval = timedelta(hours=1)  # Run learning cycle every hour
        self.min_feedback_for_learning = 5  # Minimum feedback entries before learning
        self.confidence_adjustment_rate = 0.1  # How much to adjust confidence per feedback
        
        logger.info("✅ Feedback Loop Service initialized")
    
    
    async def initialize(self):
        """Initialize service and load cached data"""
        try:
            # Load recent feedback
            feedback_docs = await self.feedback_collection.find({}).sort("created_at", -1).limit(100).to_list(length=100)
            self.recent_feedback = [FeedbackEntry(**doc) for doc in feedback_docs]
            
            # Load learning metrics
            metrics_doc = await self.metrics_collection.find_one({"_id": "global"})
            if metrics_doc:
                self.learning_metrics = LearningMetrics(**metrics_doc)
            
            logger.info(f"✅ Loaded {len(self.recent_feedback)} recent feedback entries")
            
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
    
    
    async def record_feedback(
        self,
        entity_type: str,
        entity_id: str,
        feedback_type: str,
        success: Optional[bool] = None,
        rating: Optional[float] = None,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None,
        user_comment: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: str = "system"
    ) -> FeedbackEntry:
        """
        Record feedback for learning
        
        Args:
            entity_type: Type of entity (workflow, tool, skill, pattern)
            entity_id: ID of the entity
            feedback_type: Type of feedback (success, failure, user_rating, etc.)
            success: Whether the execution was successful
            rating: User rating (0-5)
            execution_time: Execution time in seconds
            error_message: Error message if failed
            user_comment: User's comment
            context: Additional context
            user_id: User who provided feedback
        
        Returns:
            FeedbackEntry object
        """
        try:
            feedback = FeedbackEntry(
                feedback_type=feedback_type,
                entity_type=entity_type,
                entity_id=entity_id,
                success=success,
                rating=rating,
                execution_time=execution_time,
                error_message=error_message,
                user_comment=user_comment,
                context=context or {},
                created_by=user_id
            )
            
            # Save to database
            await self.feedback_collection.insert_one(feedback.dict())
            
            # Add to recent feedback
            self.recent_feedback.insert(0, feedback)
            if len(self.recent_feedback) > 100:
                self.recent_feedback.pop()
            
            # Update metrics
            self.learning_metrics.total_feedback_entries += 1
            
            # Trigger learning if enough feedback accumulated
            if len(self.recent_feedback) >= self.min_feedback_for_learning:
                await self._trigger_learning_cycle()
            
            logger.info(f"✅ Recorded {feedback_type} feedback for {entity_type} {entity_id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            raise
    
    
    async def _trigger_learning_cycle(self):
        """Trigger a learning cycle to refine patterns and adjust confidence"""
        try:
            # Check if enough time has passed since last cycle
            if self.learning_metrics.last_learning_cycle:
                time_since_last = datetime.utcnow() - self.learning_metrics.last_learning_cycle
                if time_since_last < self.learning_cycle_interval:
                    return  # Too soon
            
            logger.info("🧠 Starting learning cycle...")
            
            # Step 1: Refine patterns based on feedback
            patterns_refined = await self._refine_patterns()
            
            # Step 2: Adjust skill confidence
            skills_adjusted = await self._adjust_skill_confidence()
            
            # Step 3: Update tool performance
            tools_improved = await self._update_tool_performance()
            
            # Update metrics
            self.learning_metrics.patterns_refined += patterns_refined
            self.learning_metrics.skills_adjusted += skills_adjusted
            self.learning_metrics.tools_improved += tools_improved
            self.learning_metrics.last_learning_cycle = datetime.utcnow()
            
            # Calculate success rate
            recent_success_feedback = [
                f for f in self.recent_feedback
                if f.success is not None
            ]
            if recent_success_feedback:
                success_count = sum(1 for f in recent_success_feedback if f.success)
                self.learning_metrics.success_rate = success_count / len(recent_success_feedback)
            
            # Calculate average rating
            recent_ratings = [f.rating for f in self.recent_feedback if f.rating is not None]
            if recent_ratings:
                self.learning_metrics.average_rating = sum(recent_ratings) / len(recent_ratings)
            
            # Save metrics
            await self.metrics_collection.update_one(
                {"_id": "global"},
                {"$set": self.learning_metrics.dict()},
                upsert=True
            )
            
            logger.info(f"✅ Learning cycle complete: {patterns_refined} patterns refined, {skills_adjusted} skills adjusted, {tools_improved} tools improved")
            
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}")
    
    
    async def _refine_patterns(self) -> int:
        """Refine patterns based on feedback"""
        if not self.reasoning_bank:
            return 0
        
        try:
            refined_count = 0
            
            # Get feedback for patterns
            pattern_feedback = [
                f for f in self.recent_feedback
                if f.entity_type == "pattern"
            ]
            
            # Group by pattern ID
            pattern_groups: Dict[str, List[FeedbackEntry]] = {}
            for feedback in pattern_feedback:
                if feedback.entity_id not in pattern_groups:
                    pattern_groups[feedback.entity_id] = []
                pattern_groups[feedback.entity_id].append(feedback)
            
            # Refine each pattern
            for pattern_id, feedbacks in pattern_groups.items():
                # Calculate success rate for this pattern
                success_feedbacks = [f for f in feedbacks if f.success is not None]
                if not success_feedbacks:
                    continue
                
                success_rate = sum(1 for f in success_feedbacks if f.success) / len(success_feedbacks)
                
                # If success rate is low, mark pattern for refinement
                if success_rate < 0.5:
                    logger.info(f"Pattern {pattern_id} has low success rate ({success_rate:.1%}), marking for refinement")
                    # TODO: Implement pattern refinement logic
                    refined_count += 1
            
            return refined_count
            
        except Exception as e:
            logger.error(f"Pattern refinement failed: {e}")
            return 0
    
    
    async def _adjust_skill_confidence(self) -> int:
        """Adjust skill confidence based on feedback"""
        try:
            adjusted_count = 0
            
            # Get feedback for skills
            skill_feedback = [
                f for f in self.recent_feedback
                if f.entity_type == "skill"
            ]
            
            # Group by skill ID
            skill_groups: Dict[str, List[FeedbackEntry]] = {}
            for feedback in skill_feedback:
                if feedback.entity_id not in skill_groups:
                    skill_groups[feedback.entity_id] = []
                skill_groups[feedback.entity_id].append(feedback)
            
            # Adjust each skill
            for skill_id, feedbacks in skill_groups.items():
                # Calculate success rate
                success_feedbacks = [f for f in feedbacks if f.success is not None]
                if not success_feedbacks:
                    continue
                
                success_rate = sum(1 for f in success_feedbacks if f.success) / len(success_feedbacks)
                
                # Get current skill
                skill = skills_service.get_skill(skill_id)
                if not skill:
                    continue
                
                # Calculate new confidence
                old_confidence = skill.confidence
                
                # Adjust based on success rate
                if success_rate > 0.8:
                    # High success - increase confidence
                    new_confidence = min(1.0, old_confidence + self.confidence_adjustment_rate)
                elif success_rate < 0.5:
                    # Low success - decrease confidence
                    new_confidence = max(0.0, old_confidence - self.confidence_adjustment_rate)
                else:
                    # Moderate success - small adjustment
                    new_confidence = old_confidence + (success_rate - 0.65) * self.confidence_adjustment_rate
                
                # Update skill confidence
                if abs(new_confidence - old_confidence) > 0.01:
                    skill.confidence = new_confidence
                    
                    # Record adjustment
                    adjustment = SkillConfidenceAdjustment(
                        skill_id=skill_id,
                        old_confidence=old_confidence,
                        new_confidence=new_confidence,
                        reason=f"Success rate: {success_rate:.1%} over {len(success_feedbacks)} executions"
                    )
                    await self.adjustments_collection.insert_one(adjustment.dict())
                    
                    adjusted_count += 1
                    logger.info(f"Adjusted skill {skill_id} confidence: {old_confidence:.2f} → {new_confidence:.2f}")
            
            return adjusted_count
            
        except Exception as e:
            logger.error(f"Skill confidence adjustment failed: {e}")
            return 0
    
    
    async def _update_tool_performance(self) -> int:
        """Update tool performance based on feedback"""
        if not self.tool_recommendation:
            return 0
        
        try:
            updated_count = 0
            
            # Get feedback for tools
            tool_feedback = [
                f for f in self.recent_feedback
                if f.entity_type == "tool"
            ]
            
            # Update tool performance for each feedback
            for feedback in tool_feedback:
                if feedback.success is not None and feedback.execution_time is not None:
                    await self.tool_recommendation.record_tool_execution(
                        tool_id=feedback.entity_id,
                        success=feedback.success,
                        execution_time=feedback.execution_time,
                        user_rating=feedback.rating
                    )
                    updated_count += 1
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Tool performance update failed: {e}")
            return 0
    
    
    async def get_learning_metrics(self) -> LearningMetrics:
        """Get current learning metrics"""
        return self.learning_metrics
    
    
    async def get_recent_feedback(self, limit: int = 20) -> List[FeedbackEntry]:
        """Get recent feedback entries"""
        return self.recent_feedback[:limit]
    
    
    async def get_skill_adjustments(self, skill_id: Optional[str] = None, limit: int = 20) -> List[SkillConfidenceAdjustment]:
        """Get skill confidence adjustments"""
        try:
            query = {"skill_id": skill_id} if skill_id else {}
            docs = await self.adjustments_collection.find(query).sort("adjusted_at", -1).limit(limit).to_list(length=limit)
            return [SkillConfidenceAdjustment(**doc) for doc in docs]
        except Exception as e:
            logger.error(f"Failed to get skill adjustments: {e}")
            return []

