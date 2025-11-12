"""
ART (Agent Reinforcement Trainer) Models for CPAS4
Pydantic models for agent training, GRPO optimization, and versioning
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# Enums
class TrainingStatus(str, Enum):
    """Training job status"""
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RewardType(str, Enum):
    """Types of reward signals"""
    USER_FEEDBACK = "user_feedback"
    TASK_SUCCESS = "task_success"
    RESPONSE_QUALITY = "response_quality"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    COMPOSITE = "composite"


class AgentVersion(str, Enum):
    """Agent version types"""
    BASELINE = "baseline"
    EXPERIMENTAL = "experimental"
    PRODUCTION = "production"
    ARCHIVED = "archived"


# Training Configuration
class GRPOConfig(BaseModel):
    """GRPO (Group Relative Policy Optimization) configuration"""
    learning_rate: float = Field(1e-5, ge=1e-7, le=1e-3, description="Learning rate")
    batch_size: int = Field(8, ge=1, le=128, description="Training batch size")
    num_epochs: int = Field(3, ge=1, le=100, description="Number of training epochs")
    max_steps: Optional[int] = Field(None, description="Maximum training steps")
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    gradient_accumulation_steps: int = Field(4, ge=1, description="Gradient accumulation")
    max_grad_norm: float = Field(1.0, ge=0.0, description="Gradient clipping")
    
    # GRPO-specific parameters
    group_size: int = Field(4, ge=2, le=16, description="Group size for relative comparison")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    kl_penalty: float = Field(0.1, ge=0.0, le=1.0, description="KL divergence penalty")
    value_loss_coef: float = Field(0.5, ge=0.0, le=1.0, description="Value loss coefficient")
    
    # Hardware configuration
    use_gpu: bool = Field(True, description="Use GPU if available")
    mixed_precision: bool = Field(True, description="Use mixed precision training")
    num_workers: int = Field(4, ge=1, le=16, description="Data loader workers")


class RewardModelConfig(BaseModel):
    """Reward model configuration"""
    reward_types: List[RewardType] = Field(
        default=[RewardType.USER_FEEDBACK, RewardType.TASK_SUCCESS],
        description="Types of rewards to use"
    )
    weights: Dict[str, float] = Field(
        default={
            "user_feedback": 0.4,
            "task_success": 0.3,
            "response_quality": 0.2,
            "efficiency": 0.1
        },
        description="Reward type weights (must sum to 1.0)"
    )
    normalize_rewards: bool = Field(True, description="Normalize rewards to [-1, 1]")
    clip_rewards: bool = Field(True, description="Clip extreme reward values")
    reward_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum reward threshold")


# Training Data
class ConversationTurn(BaseModel):
    """Single conversation turn for training"""
    turn_id: str = Field(..., description="Unique turn ID")
    agent_id: str = Field(..., description="Agent that generated this turn")
    user_message: str = Field(..., description="User input")
    agent_response: str = Field(..., description="Agent response")
    context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Feedback and rewards
    user_rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="User rating (0-5)")
    task_completed: Optional[bool] = Field(None, description="Whether task was completed")
    response_time_ms: Optional[float] = Field(None, ge=0.0, description="Response time")
    tokens_used: Optional[int] = Field(None, ge=0, description="Tokens consumed")


class TrainingDataset(BaseModel):
    """Training dataset for an agent"""
    dataset_id: str = Field(..., description="Unique dataset ID")
    agent_id: str = Field(..., description="Agent this dataset is for")
    conversations: List[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Dataset statistics
    total_turns: int = Field(0, ge=0, description="Total conversation turns")
    avg_reward: Optional[float] = Field(None, description="Average reward score")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dataset quality")


# Training Job
class TrainingJob(BaseModel):
    """Agent training job"""
    job_id: str = Field(..., description="Unique job ID")
    agent_id: str = Field(..., description="Agent being trained")
    dataset_id: str = Field(..., description="Training dataset ID")
    
    # Configuration
    grpo_config: GRPOConfig = Field(default_factory=GRPOConfig)
    reward_config: RewardModelConfig = Field(default_factory=RewardModelConfig)
    
    # Status
    status: TrainingStatus = Field(TrainingStatus.QUEUED)
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Training progress (0-1)")
    current_epoch: int = Field(0, ge=0)
    current_step: int = Field(0, ge=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Results
    final_loss: Optional[float] = Field(None, description="Final training loss")
    final_reward: Optional[float] = Field(None, description="Final average reward")
    improvement_score: Optional[float] = Field(None, description="Improvement over baseline")
    
    # Metadata
    error_message: Optional[str] = Field(None, description="Error message if failed")
    logs_path: Optional[str] = Field(None, description="Path to training logs")
    checkpoint_path: Optional[str] = Field(None, description="Path to model checkpoint")


# Agent Versioning
class AgentVersionInfo(BaseModel):
    """Agent version information"""
    version_id: str = Field(..., description="Unique version ID")
    agent_id: str = Field(..., description="Base agent ID")
    version_number: str = Field(..., description="Semantic version (e.g., 1.2.3)")
    version_type: AgentVersion = Field(..., description="Version type")
    
    # Training info
    trained_from_job: Optional[str] = Field(None, description="Training job ID")
    parent_version: Optional[str] = Field(None, description="Parent version ID")
    
    # Performance metrics
    baseline_reward: float = Field(..., description="Baseline performance")
    current_reward: float = Field(..., description="Current performance")
    improvement_pct: float = Field(..., description="Improvement percentage")
    
    # Deployment info
    is_deployed: bool = Field(False, description="Whether deployed to production")
    deployed_at: Optional[datetime] = Field(None)
    deployment_count: int = Field(0, ge=0, description="Number of times deployed")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="User who created this version")
    notes: Optional[str] = Field(None, description="Version notes")
    tags: List[str] = Field(default_factory=list, description="Version tags")


# Training Metrics
class TrainingMetrics(BaseModel):
    """Training metrics for monitoring"""
    job_id: str = Field(..., description="Training job ID")
    epoch: int = Field(..., ge=0)
    step: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Loss metrics
    total_loss: float = Field(..., description="Total loss")
    policy_loss: float = Field(..., description="Policy loss")
    value_loss: float = Field(..., description="Value loss")
    kl_divergence: float = Field(..., description="KL divergence from baseline")
    
    # Reward metrics
    avg_reward: float = Field(..., description="Average reward")
    min_reward: float = Field(..., description="Minimum reward")
    max_reward: float = Field(..., description="Maximum reward")
    reward_std: float = Field(..., description="Reward standard deviation")
    
    # Performance metrics
    learning_rate: float = Field(..., description="Current learning rate")
    gradient_norm: float = Field(..., description="Gradient norm")
    tokens_per_second: Optional[float] = Field(None, description="Training throughput")
    gpu_memory_mb: Optional[float] = Field(None, description="GPU memory usage")


# API Request/Response Models
class StartTrainingRequest(BaseModel):
    """Request to start agent training"""
    agent_id: str = Field(..., description="Agent to train")
    dataset_id: Optional[str] = Field(None, description="Dataset ID (auto-create if None)")
    grpo_config: Optional[GRPOConfig] = Field(None, description="GRPO configuration")
    reward_config: Optional[RewardModelConfig] = Field(None, description="Reward configuration")
    auto_deploy: bool = Field(False, description="Auto-deploy if improvement > threshold")
    deployment_threshold: float = Field(0.1, ge=0.0, description="Min improvement for auto-deploy")


class TrainingStatusResponse(BaseModel):
    """Training status response"""
    job: TrainingJob
    latest_metrics: Optional[TrainingMetrics] = Field(None)
    estimated_time_remaining: Optional[int] = Field(None, description="Seconds remaining")


class DeployAgentVersionRequest(BaseModel):
    """Request to deploy an agent version"""
    version_id: str = Field(..., description="Version to deploy")
    environment: Literal["production", "staging", "development"] = Field("production")
    rollback_on_error: bool = Field(True, description="Auto-rollback on errors")
    canary_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Canary deployment %")


class AgentPerformanceReport(BaseModel):
    """Agent performance report"""
    agent_id: str
    version_id: str
    time_period: str = Field(..., description="Report time period (e.g., '7d', '30d')")
    
    # Performance metrics
    total_conversations: int = Field(0, ge=0)
    avg_user_rating: float = Field(0.0, ge=0.0, le=5.0)
    task_success_rate: float = Field(0.0, ge=0.0, le=1.0)
    avg_response_time_ms: float = Field(0.0, ge=0.0)
    
    # Comparison to baseline
    baseline_version: Optional[str] = Field(None)
    improvement_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Recommendations
    needs_retraining: bool = Field(False)
    recommended_actions: List[str] = Field(default_factory=list)

