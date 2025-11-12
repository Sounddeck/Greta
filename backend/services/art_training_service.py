"""
ART Training Service for CPAS4
Orchestrates agent training using GRPO (Group Relative Policy Optimization)
"""
import logging
import asyncio
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from models.art_training_models import (
    TrainingJob, TrainingStatus, TrainingDataset, ConversationTurn,
    GRPOConfig, RewardModelConfig, TrainingMetrics, AgentVersionInfo,
    AgentVersion, StartTrainingRequest, TrainingStatusResponse
)
from services.conversation_collector import ConversationCollector
from services.reward_evaluator import RewardEvaluator
from services.grpo_optimizer import GRPOOptimizer
from services.agent_versioning import AgentVersionRegistry
from database import get_database

logger = logging.getLogger(__name__)


class ARTTrainingService:
    """
    ART (Agent Reinforcement Trainer) Service
    
    Manages the complete agent training pipeline:
    1. Data collection from conversations
    2. Reward calculation
    3. GRPO optimization
    4. Model versioning and deployment
    """
    
    def __init__(self):
        """Initialize the ART training service"""
        self.db = get_database()
        self.collector = ConversationCollector()
        self.reward_evaluator = RewardEvaluator()
        self.optimizer = GRPOOptimizer()
        self.version_registry = AgentVersionRegistry()
        
        # Collections
        self.jobs_collection = "training_jobs"
        self.datasets_collection = "training_datasets"
        self.metrics_collection = "training_metrics"
        
        # Training queue (Redis-backed)
        self.training_queue = []
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        logger.info("✅ ART Training Service initialized")
    
    
    async def start_training(self, request: StartTrainingRequest) -> TrainingJob:
        """
        Start a new training job for an agent
        
        Args:
            request: Training request with configuration
            
        Returns:
            TrainingJob: Created training job
        """
        try:
            # Generate job ID
            job_id = f"train_{uuid.uuid4().hex[:12]}"
            
            # Get or create dataset
            if request.dataset_id:
                dataset = await self._get_dataset(request.dataset_id)
            else:
                dataset = await self._create_dataset_from_conversations(request.agent_id)
            
            # Validate dataset quality
            if not await self._validate_dataset(dataset):
                raise ValueError(f"Dataset {dataset.dataset_id} does not meet quality requirements")
            
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                agent_id=request.agent_id,
                dataset_id=dataset.dataset_id,
                grpo_config=request.grpo_config or GRPOConfig(),
                reward_config=request.reward_config or RewardModelConfig(),
                status=TrainingStatus.QUEUED,
                created_at=datetime.utcnow()
            )
            
            # Save to database
            await self.db.insert_one(self.jobs_collection, job.dict())
            
            # Add to training queue
            self.training_queue.append(job_id)
            
            logger.info(f"✅ Training job {job_id} created for agent {request.agent_id}")
            
            # Start training asynchronously
            asyncio.create_task(self._execute_training(job, request.auto_deploy, request.deployment_threshold))
            
            return job
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            raise
    
    
    async def _create_dataset_from_conversations(self, agent_id: str) -> TrainingDataset:
        """
        Create training dataset from agent conversations
        
        Args:
            agent_id: Agent ID to collect conversations for
            
        Returns:
            TrainingDataset: Created dataset
        """
        try:
            # Collect conversations from the last 30 days
            conversations = await self.collector.collect_conversations(
                agent_id=agent_id,
                days_back=30,
                min_rating=3.0  # Only include conversations with rating >= 3.0
            )
            
            # Create dataset
            dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
            dataset = TrainingDataset(
                dataset_id=dataset_id,
                agent_id=agent_id,
                conversations=conversations,
                total_turns=len(conversations),
                created_at=datetime.utcnow()
            )
            
            # Calculate dataset statistics
            if conversations:
                ratings = [c.user_rating for c in conversations if c.user_rating is not None]
                dataset.avg_reward = sum(ratings) / len(ratings) if ratings else None
                dataset.quality_score = await self._calculate_dataset_quality(conversations)
            
            # Save to database
            await self.db.insert_one(self.datasets_collection, dataset.dict())
            
            logger.info(f"✅ Created dataset {dataset_id} with {len(conversations)} conversations")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to create dataset: {e}")
            raise
    
    
    async def _validate_dataset(self, dataset: TrainingDataset) -> bool:
        """
        Validate dataset meets minimum quality requirements
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            bool: True if dataset is valid
        """
        # Minimum requirements
        MIN_CONVERSATIONS = 50
        MIN_QUALITY_SCORE = 0.6
        MIN_AVG_REWARD = 3.0
        
        if dataset.total_turns < MIN_CONVERSATIONS:
            logger.warning(f"Dataset {dataset.dataset_id} has only {dataset.total_turns} conversations (min: {MIN_CONVERSATIONS})")
            return False
        
        if dataset.quality_score and dataset.quality_score < MIN_QUALITY_SCORE:
            logger.warning(f"Dataset {dataset.dataset_id} quality score {dataset.quality_score} below threshold {MIN_QUALITY_SCORE}")
            return False
        
        if dataset.avg_reward and dataset.avg_reward < MIN_AVG_REWARD:
            logger.warning(f"Dataset {dataset.dataset_id} avg reward {dataset.avg_reward} below threshold {MIN_AVG_REWARD}")
            return False
        
        return True
    
    
    async def _calculate_dataset_quality(self, conversations: List[ConversationTurn]) -> float:
        """
        Calculate overall dataset quality score
        
        Args:
            conversations: List of conversation turns
            
        Returns:
            float: Quality score (0-1)
        """
        if not conversations:
            return 0.0
        
        # Factors for quality calculation
        factors = []
        
        # 1. Rating coverage (% of conversations with ratings)
        rated = sum(1 for c in conversations if c.user_rating is not None)
        rating_coverage = rated / len(conversations)
        factors.append(rating_coverage)
        
        # 2. Task completion rate
        completed = sum(1 for c in conversations if c.task_completed is True)
        completion_rate = completed / len(conversations) if completed > 0 else 0.5
        factors.append(completion_rate)
        
        # 3. Response diversity (unique responses / total responses)
        unique_responses = len(set(c.agent_response for c in conversations))
        diversity = unique_responses / len(conversations)
        factors.append(diversity)
        
        # 4. Average rating (normalized to 0-1)
        ratings = [c.user_rating for c in conversations if c.user_rating is not None]
        avg_rating = (sum(ratings) / len(ratings) / 5.0) if ratings else 0.5
        factors.append(avg_rating)
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        quality_score = sum(f * w for f, w in zip(factors, weights))
        
        return round(quality_score, 3)
    
    
    async def _execute_training(self, job: TrainingJob, auto_deploy: bool, deployment_threshold: float):
        """
        Execute the training job
        
        Args:
            job: Training job to execute
            auto_deploy: Whether to auto-deploy if improvement exceeds threshold
            deployment_threshold: Minimum improvement for auto-deployment
        """
        try:
            # Update status to PREPARING
            await self._update_job_status(job.job_id, TrainingStatus.PREPARING)
            job.started_at = datetime.utcnow()
            
            # Load dataset
            dataset = await self._get_dataset(job.dataset_id)
            
            # Calculate rewards for all conversations
            logger.info(f"📊 Calculating rewards for {len(dataset.conversations)} conversations...")
            rewards = await self.reward_evaluator.calculate_rewards(
                conversations=dataset.conversations,
                config=job.reward_config
            )
            
            # Update status to TRAINING
            await self._update_job_status(job.job_id, TrainingStatus.TRAINING)
            
            # Execute GRPO training
            logger.info(f"🚀 Starting GRPO training for agent {job.agent_id}...")
            training_result = await self.optimizer.train(
                agent_id=job.agent_id,
                conversations=dataset.conversations,
                rewards=rewards,
                config=job.grpo_config,
                metrics_callback=lambda metrics: self._save_metrics(job.job_id, metrics)
            )
            
            # Update status to EVALUATING
            await self._update_job_status(job.job_id, TrainingStatus.EVALUATING)
            
            # Evaluate improvement
            improvement = await self._evaluate_improvement(
                job.agent_id,
                training_result["baseline_reward"],
                training_result["final_reward"]
            )
            
            # Create new agent version
            version = await self.version_registry.create_version(
                agent_id=job.agent_id,
                training_job_id=job.job_id,
                baseline_reward=training_result["baseline_reward"],
                current_reward=training_result["final_reward"],
                improvement_pct=improvement,
                checkpoint_path=training_result["checkpoint_path"]
            )
            
            # Update job with results
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.final_loss = training_result["final_loss"]
            job.final_reward = training_result["final_reward"]
            job.improvement_score = improvement
            job.checkpoint_path = training_result["checkpoint_path"]
            job.progress = 1.0
            
            await self.db.update_one(
                self.jobs_collection,
                {"job_id": job.job_id},
                {"$set": job.dict()}
            )
            
            logger.info(f"✅ Training completed! Improvement: {improvement:.2%}")
            
            # Auto-deploy if enabled and improvement exceeds threshold
            if auto_deploy and improvement >= deployment_threshold:
                logger.info(f"🚀 Auto-deploying version {version.version_id} (improvement: {improvement:.2%})")
                await self.version_registry.deploy_version(version.version_id, environment="production")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            await self.db.update_one(
                self.jobs_collection,
                {"job_id": job.job_id},
                {"$set": {"status": job.status.value, "error_message": job.error_message}}
            )

