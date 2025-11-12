"""
Reward Evaluator for ART Training
Calculates reward signals for agent training using multiple criteria
"""
import logging
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from models.art_training_models import ConversationTurn, RewardModelConfig, RewardType

logger = logging.getLogger(__name__)


class RewardEvaluator:
    """Evaluates and calculates rewards for agent training"""
    
    def __init__(self):
        """Initialize reward evaluator"""
        logger.info("✅ Reward Evaluator initialized")
    
    async def calculate_rewards(
        self,
        conversations: List[ConversationTurn],
        config: RewardModelConfig
    ) -> List[float]:
        """
        Calculate rewards for a list of conversations
        
        Args:
            conversations: List of conversation turns
            config: Reward model configuration
            
        Returns:
            List[float]: Reward scores for each conversation
        """
        try:
            rewards = []
            
            for conversation in conversations:
                # Calculate individual reward components
                reward_components = {}
                
                if RewardType.USER_FEEDBACK in config.reward_types:
                    reward_components["user_feedback"] = self._calculate_user_feedback_reward(conversation)
                
                if RewardType.TASK_SUCCESS in config.reward_types:
                    reward_components["task_success"] = self._calculate_task_success_reward(conversation)
                
                if RewardType.RESPONSE_QUALITY in config.reward_types:
                    reward_components["response_quality"] = self._calculate_response_quality_reward(conversation)
                
                if RewardType.EFFICIENCY in config.reward_types:
                    reward_components["efficiency"] = self._calculate_efficiency_reward(conversation)
                
                if RewardType.SAFETY in config.reward_types:
                    reward_components["safety"] = self._calculate_safety_reward(conversation)
                
                # Combine rewards using weighted sum
                total_reward = sum(
                    reward_components.get(key, 0.0) * config.weights.get(key, 0.0)
                    for key in reward_components.keys()
                )
                
                # Normalize if configured
                if config.normalize_rewards:
                    total_reward = self._normalize_reward(total_reward)
                
                # Clip if configured
                if config.clip_rewards:
                    total_reward = np.clip(total_reward, -1.0, 1.0)
                
                rewards.append(total_reward)
            
            logger.info(f"✅ Calculated rewards for {len(conversations)} conversations")
            logger.info(f"   Mean reward: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
            
            return rewards
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate rewards: {e}")
            return [0.0] * len(conversations)
    
    def _calculate_user_feedback_reward(self, conversation: ConversationTurn) -> float:
        """
        Calculate reward from user feedback (rating)
        
        Args:
            conversation: Conversation turn
            
        Returns:
            float: Reward score (0-1)
        """
        if conversation.user_rating is None:
            return 0.5  # Neutral reward if no rating
        
        # Normalize rating from 0-5 to 0-1
        return conversation.user_rating / 5.0
    
    def _calculate_task_success_reward(self, conversation: ConversationTurn) -> float:
        """
        Calculate reward from task completion
        
        Args:
            conversation: Conversation turn
            
        Returns:
            float: Reward score (0-1)
        """
        if conversation.task_completed is None:
            return 0.5  # Neutral reward if unknown
        
        return 1.0 if conversation.task_completed else 0.0
    
    def _calculate_response_quality_reward(self, conversation: ConversationTurn) -> float:
        """
        Calculate reward from response quality heuristics
        
        Args:
            conversation: Conversation turn
            
        Returns:
            float: Reward score (0-1)
        """
        response = conversation.agent_response
        
        # Quality factors
        factors = []
        
        # 1. Response length (not too short, not too long)
        length = len(response)
        if 50 <= length <= 500:
            length_score = 1.0
        elif length < 50:
            length_score = length / 50.0
        else:
            length_score = max(0.5, 1.0 - (length - 500) / 1000.0)
        factors.append(length_score)
        
        # 2. Sentence structure (has proper punctuation)
        has_punctuation = any(p in response for p in ['.', '!', '?'])
        factors.append(1.0 if has_punctuation else 0.5)
        
        # 3. Not repetitive (no excessive word repetition)
        words = response.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            factors.append(unique_ratio)
        else:
            factors.append(0.0)
        
        # 4. Contains helpful markers (e.g., "here's", "you can", "I'll help")
        helpful_markers = ["here", "you can", "i'll help", "let me", "i can"]
        has_helpful = any(marker in response.lower() for marker in helpful_markers)
        factors.append(1.0 if has_helpful else 0.7)
        
        # Average all factors
        return sum(factors) / len(factors)
    
    def _calculate_efficiency_reward(self, conversation: ConversationTurn) -> float:
        """
        Calculate reward from efficiency metrics
        
        Args:
            conversation: Conversation turn
            
        Returns:
            float: Reward score (0-1)
        """
        factors = []
        
        # 1. Response time (faster is better, but not instant)
        if conversation.response_time_ms is not None:
            time_ms = conversation.response_time_ms
            if 500 <= time_ms <= 3000:  # Ideal range: 0.5-3 seconds
                time_score = 1.0
            elif time_ms < 500:
                time_score = time_ms / 500.0
            else:
                time_score = max(0.3, 1.0 - (time_ms - 3000) / 10000.0)
            factors.append(time_score)
        
        # 2. Token efficiency (concise but complete)
        if conversation.tokens_used is not None:
            tokens = conversation.tokens_used
            if 50 <= tokens <= 300:  # Ideal range
                token_score = 1.0
            elif tokens < 50:
                token_score = tokens / 50.0
            else:
                token_score = max(0.5, 1.0 - (tokens - 300) / 500.0)
            factors.append(token_score)
        
        if not factors:
            return 0.7  # Default efficiency score
        
        return sum(factors) / len(factors)
    
    def _calculate_safety_reward(self, conversation: ConversationTurn) -> float:
        """
        Calculate reward from safety checks
        
        Args:
            conversation: Conversation turn
            
        Returns:
            float: Reward score (0-1)
        """
        response = conversation.agent_response.lower()
        
        # Safety violations (penalize these)
        violations = []
        
        # 1. No harmful content
        harmful_keywords = ["hack", "exploit", "illegal", "dangerous", "harmful"]
        has_harmful = any(keyword in response for keyword in harmful_keywords)
        violations.append(has_harmful)
        
        # 2. No personal data exposure
        pii_patterns = ["password", "credit card", "ssn", "social security"]
        has_pii = any(pattern in response for pattern in pii_patterns)
        violations.append(has_pii)
        
        # 3. No inappropriate content
        inappropriate = ["offensive", "discriminatory", "hateful"]
        has_inappropriate = any(word in response for word in inappropriate)
        violations.append(has_inappropriate)
        
        # Calculate safety score (1.0 if no violations, 0.0 if violations)
        violation_count = sum(violations)
        if violation_count == 0:
            return 1.0
        else:
            return max(0.0, 1.0 - (violation_count * 0.5))
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward to [-1, 1] range
        
        Args:
            reward: Raw reward score
            
        Returns:
            float: Normalized reward
        """
        # Assuming rewards are in [0, 1], map to [-1, 1]
        return (reward * 2.0) - 1.0

