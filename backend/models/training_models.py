"""
Training Models for Local LLM Training Automation
Pydantic models for training datasets, jobs, and model versions
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
from enum import Enum


class DatasetStatus(str, Enum):
    """Dataset status"""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class TrainingJobStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuantizationType(str, Enum):
    """Model quantization types"""
    Q4_K_M = "Q4_K_M"  # 4-bit, medium quality (recommended)
    Q5_K_M = "Q5_K_M"  # 5-bit, higher quality
    Q8_0 = "Q8_0"      # 8-bit, highest quality
    NONE = "none"      # No quantization (full precision)


class DeploymentStatus(str, Enum):
    """Model deployment status"""
    TESTING = "testing"      # Model is being tested
    DEPLOYED = "deployed"    # Model is deployed and active
    ARCHIVED = "archived"    # Model is archived (not in use)


# Dataset Models

class DatasetFilters(BaseModel):
    """Filters for dataset creation"""
    min_rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Minimum user rating")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence score")
    success_only: bool = Field(True, description="Include only successful interactions")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")
    agent_ids: Optional[List[str]] = Field(None, description="Filter by specific agent IDs")
    exclude_duplicates: bool = Field(True, description="Remove duplicate conversations")


class TrainingDatasetCreate(BaseModel):
    """Create training dataset request"""
    name: str = Field(..., min_length=1, max_length=200, description="Dataset name")
    description: Optional[str] = Field(None, max_length=1000, description="Dataset description")
    source_collections: List[str] = Field(
        default=["agent_conversations", "interactions"],
        description="MongoDB collections to collect from"
    )
    filters: DatasetFilters = Field(default_factory=DatasetFilters, description="Data filters")
    max_samples: Optional[int] = Field(None, gt=0, description="Maximum number of samples")


class TrainingDataset(BaseModel):
    """Training dataset model"""
    dataset_id: str = Field(..., description="Unique dataset ID")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    source_collections: List[str] = Field(..., description="Source collections")
    filters: DatasetFilters = Field(..., description="Applied filters")
    total_samples: int = Field(0, description="Total number of samples")
    file_path: str = Field(..., description="Path to JSONL file")
    file_size_mb: float = Field(0.0, description="File size in MB")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Creator (null for single-user)")
    status: DatasetStatus = Field(DatasetStatus.PENDING, description="Dataset status")
    error: Optional[str] = Field(None, description="Error message if failed")


# Training Job Models

class TrainingConfig(BaseModel):
    """Training configuration"""
    lora_rank: int = Field(8, ge=4, le=64, description="LoRA rank")
    lora_alpha: int = Field(16, ge=8, le=128, description="LoRA alpha")
    learning_rate: float = Field(1e-4, gt=0, description="Learning rate")
    batch_size: int = Field(4, ge=1, le=32, description="Batch size")
    epochs: int = Field(3, ge=1, le=20, description="Number of epochs")
    ctx_size: int = Field(2048, ge=512, le=8192, description="Context size")
    threads: int = Field(8, ge=1, le=32, description="Number of threads")
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    save_every: int = Field(100, ge=10, description="Save checkpoint every N steps")


class TrainingProgress(BaseModel):
    """Training progress"""
    current_epoch: int = Field(0, description="Current epoch")
    total_epochs: int = Field(0, description="Total epochs")
    current_step: int = Field(0, description="Current step")
    total_steps: int = Field(0, description="Total steps")
    loss: float = Field(0.0, description="Current loss")
    learning_rate: float = Field(0.0, description="Current learning rate")
    eta_seconds: int = Field(0, description="Estimated time remaining (seconds)")
    samples_per_second: float = Field(0.0, description="Training speed")


class TrainingJobCreate(BaseModel):
    """Create training job request"""
    dataset_id: str = Field(..., description="Dataset ID to train on")
    base_model: str = Field("llama-3-8b-instruct", description="Base model name")
    base_model_path: str = Field(..., description="Path to base model file")
    training_config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training config")
    output_name: Optional[str] = Field(None, description="Output model name")
    quantize: bool = Field(True, description="Quantize model after training")
    quantization_type: QuantizationType = Field(QuantizationType.Q4_K_M, description="Quantization type")


class TrainingJob(BaseModel):
    """Training job model"""
    job_id: str = Field(..., description="Unique job ID")
    dataset_id: str = Field(..., description="Dataset ID")
    base_model: str = Field(..., description="Base model name")
    base_model_path: str = Field(..., description="Base model path")
    training_config: TrainingConfig = Field(..., description="Training configuration")
    status: TrainingJobStatus = Field(TrainingJobStatus.PENDING, description="Job status")
    progress: TrainingProgress = Field(default_factory=TrainingProgress, description="Training progress")
    output_paths: Dict[str, str] = Field(default_factory=dict, description="Output file paths")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Creator (null for single-user)")
    logs: List[str] = Field(default_factory=list, description="Training logs")


# Model Version Models

class PerformanceMetrics(BaseModel):
    """Model performance metrics"""
    perplexity: Optional[float] = Field(None, description="Model perplexity")
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time")
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=5.0, description="User satisfaction score")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Success rate")
    total_inferences: int = Field(0, description="Total number of inferences")


class ModelConversionRequest(BaseModel):
    """Request to convert a training job to a model"""
    job_id: str = Field(..., description="Training job ID")
    output_name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    quantization_types: List[QuantizationType] = Field(
        default_factory=lambda: [QuantizationType.Q4_K_M],
        description="Quantization types to create"
    )


class ModelDeployRequest(BaseModel):
    """Request to deploy a model version"""
    version_id: str = Field(..., description="Model version ID to deploy")


class ModelVersion(BaseModel):
    """Model version model"""
    version_id: str = Field(..., description="Version ID")
    job_id: str = Field(..., description="Training job ID")
    dataset_id: str = Field(..., description="Training dataset ID")
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    base_model: str = Field(..., description="Base model name")
    model_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to model files")
    model_sizes: Dict[str, float] = Field(default_factory=dict, description="Model sizes in MB")
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    deployment_status: DeploymentStatus = Field(DeploymentStatus.TESTING, description="Deployment status")
    is_active: bool = Field(False, description="Currently active for inference")
    deployed_at: Optional[datetime] = Field(None, description="Deployment time")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Creator (null for single-user)")
    tags: List[str] = Field(default_factory=list, description="Model tags")


# Response Models

class DatasetListResponse(BaseModel):
    """Dataset list response"""
    success: bool = True
    datasets: List[TrainingDataset]
    total: int


class TrainingJobListResponse(BaseModel):
    """Training job list response"""
    success: bool = True
    jobs: List[TrainingJob]
    total: int


class ModelVersionListResponse(BaseModel):
    """Model version list response"""
    success: bool = True
    versions: List[ModelVersion]
    total: int


class TrainingStatsResponse(BaseModel):
    """Training statistics response"""
    success: bool = True
    total_datasets: int
    total_training_samples: int
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_models: int
    active_model: Optional[ModelVersion]
    disk_usage_mb: float


# Export Format Models

class ConversationSample(BaseModel):
    """Single conversation sample for export"""
    text: str = Field(..., description="Formatted conversation text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Sample metadata")


class InstructionSample(BaseModel):
    """Instruction-based sample for export"""
    instruction: str = Field(..., description="User instruction")
    input: str = Field("", description="Additional input context")
    output: str = Field(..., description="Expected output")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Sample metadata")

