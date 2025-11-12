"""
Training Router
API endpoints for LLM training automation
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from models.training_models import (
    TrainingDataset,
    TrainingDatasetCreate,
    DatasetListResponse,
    DatasetStatus,
    TrainingStatsResponse,
    TrainingJob,
    TrainingJobCreate,
    TrainingJobListResponse,
    TrainingJobStatus,
    ModelVersion,
    ModelConversionRequest,
    ModelDeployRequest,
    ModelVersionListResponse,
    DeploymentStatus
)
from services.training_data_service import TrainingDataService
from services.unsloth_training_service import UnslothTrainingService
from services.model_conversion_service import ModelConversionService
from services.model_version_service import ModelVersionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm-training", tags=["LLM Training"])

# Initialize services
training_data_service = TrainingDataService()
training_service = UnslothTrainingService()
conversion_service = ModelConversionService()
version_service = ModelVersionService()


# Dataset Endpoints

@router.post("/datasets", response_model=TrainingDataset)
async def create_dataset(dataset_create: TrainingDatasetCreate) -> TrainingDataset:
    """
    Create a new training dataset
    
    Collects conversation data from MongoDB and exports to JSONL format.
    """
    try:
        dataset = await training_data_service.create_dataset(dataset_create)
        return dataset
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    status: Optional[DatasetStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of datasets")
) -> DatasetListResponse:
    """
    List training datasets
    
    Returns all datasets with optional status filter.
    """
    try:
        datasets = await training_data_service.list_datasets(status=status, limit=limit)
        return DatasetListResponse(
            success=True,
            datasets=datasets,
            total=len(datasets)
        )
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}", response_model=TrainingDataset)
async def get_dataset(dataset_id: str) -> TrainingDataset:
    """
    Get dataset by ID
    
    Returns detailed information about a specific dataset.
    """
    try:
        dataset = await training_data_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete dataset
    
    Deletes the dataset and its associated JSONL file.
    """
    try:
        success = await training_data_service.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"success": True, "message": "Dataset deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=TrainingStatsResponse)
async def get_training_stats() -> TrainingStatsResponse:
    """
    Get training statistics

    Returns overview of datasets, jobs, and models.
    """
    try:
        # Get datasets
        datasets = await training_data_service.list_datasets()
        total_samples = sum(d.total_samples for d in datasets)

        # Get jobs
        all_jobs = await training_service.list_jobs()
        active_jobs = len([j for j in all_jobs if j.status == TrainingJobStatus.RUNNING])
        completed_jobs = len([j for j in all_jobs if j.status == TrainingJobStatus.COMPLETED])
        failed_jobs = len([j for j in all_jobs if j.status == TrainingJobStatus.FAILED])

        # Get models
        all_models = await version_service.list_versions()
        active_model = await version_service.get_active_version()

        # Calculate disk usage
        disk_usage_mb = sum(d.file_size_mb for d in datasets)

        return TrainingStatsResponse(
            success=True,
            total_datasets=len(datasets),
            total_training_samples=total_samples,
            total_jobs=len(all_jobs),
            active_jobs=active_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            total_models=len(all_models),
            active_model=active_model.name if active_model else None,
            disk_usage_mb=disk_usage_mb
        )
    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training Job Endpoints (Phase 2)

@router.post("/jobs", response_model=TrainingJob)
async def create_training_job(job_create: TrainingJobCreate) -> TrainingJob:
    """
    Create a new training job

    Starts a new LLM training job using Unsloth for LoRA fine-tuning.
    """
    try:
        job = await training_service.create_training_job(job_create)
        return job
    except Exception as e:
        logger.error(f"Failed to create training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    status: Optional[TrainingJobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of jobs")
) -> TrainingJobListResponse:
    """
    List training jobs

    Returns all training jobs with optional status filter.
    """
    try:
        jobs = await training_service.list_jobs(status=status, limit=limit)
        return TrainingJobListResponse(
            success=True,
            jobs=jobs,
            total=len(jobs)
        )
    except Exception as e:
        logger.error(f"Failed to list training jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str) -> TrainingJob:
    """
    Get training job by ID

    Returns detailed information about a training job including progress.
    """
    try:
        job = await training_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    """
    Cancel training job

    Cancels a running training job.
    """
    try:
        success = await training_service.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Training job not found or not running")
        return {"success": True, "message": "Training job cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Version Endpoints (Phase 3)

@router.post("/models/convert", response_model=ModelVersion)
async def convert_job_to_model(request: ModelConversionRequest) -> ModelVersion:
    """
    Convert a training job to a deployable model

    Merges LoRA adapter, converts to GGUF, and quantizes the model.
    """
    try:
        version = await conversion_service.convert_job_to_model(
            job_id=request.job_id,
            output_name=request.output_name,
            description=request.description or "",
            quantization_types=request.quantization_types
        )
        return version
    except Exception as e:
        logger.error(f"Failed to convert job to model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelVersionListResponse)
async def list_model_versions(
    status: Optional[DeploymentStatus] = Query(None, description="Filter by deployment status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of versions")
) -> ModelVersionListResponse:
    """
    List model versions

    Returns all model versions with optional deployment status filter.
    """
    try:
        versions = await version_service.list_versions(status=status, limit=limit)
        return ModelVersionListResponse(
            success=True,
            versions=versions,
            total=len(versions)
        )
    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/active", response_model=ModelVersion)
async def get_active_model() -> ModelVersion:
    """
    Get currently active model version

    Returns the model version currently deployed for inference.
    """
    try:
        version = await version_service.get_active_version()
        if not version:
            raise HTTPException(status_code=404, detail="No active model version found")
        return version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get active model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{version_id}", response_model=ModelVersion)
async def get_model_version(version_id: str) -> ModelVersion:
    """
    Get model version by ID

    Returns detailed information about a model version.
    """
    try:
        version = await version_service.get_version(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Model version not found")
        return version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{version_id}/deploy")
async def deploy_model_version(version_id: str):
    """
    Deploy model version

    Deploys a model version for inference.
    """
    try:
        success = await version_service.deploy_version(version_id)
        return {
            "success": success,
            "message": f"Model version {version_id} deployed successfully",
            "note": "LLM service restart may be required"
        }
    except Exception as e:
        logger.error(f"Failed to deploy model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{version_id}/rollback")
async def rollback_model_version(version_id: str):
    """
    Rollback to previous model version

    Rolls back to a previous model version.
    """
    try:
        success = await version_service.rollback_to_version(version_id)
        return {
            "success": success,
            "message": f"Rolled back to model version {version_id}",
            "note": "LLM service restart may be required"
        }
    except Exception as e:
        logger.error(f"Failed to rollback model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{version_id}")
async def delete_model_version(
    version_id: str,
    delete_files: bool = Query(False, description="Delete model files from disk")
):
    """
    Delete model version

    Deletes a model version and optionally its files.
    """
    try:
        success = await version_service.delete_version(version_id, delete_files=delete_files)
        return {
            "success": success,
            "message": f"Model version {version_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to delete model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check

@router.get("/health")
async def training_health_check():
    """
    Training system health check

    Returns status of training components.
    """
    # Check if Unsloth is available
    try:
        import unsloth
        training_available = True
    except ImportError:
        training_available = False

    # Check if llama.cpp tools are available
    llamacpp_available = conversion_service.convert_script.exists() and conversion_service.quantize_bin.exists()

    return {
        "status": "healthy",
        "components": {
            "data_collection": "active",
            "training_pipeline": "active" if training_available else "unavailable",
            "model_management": "active" if llamacpp_available else "partial"
        },
        "dependencies": {
            "unsloth": training_available,
            "torch": True,  # Assume torch is available
            "llamacpp_convert": conversion_service.convert_script.exists(),
            "llamacpp_quantize": conversion_service.quantize_bin.exists()
        },
        "version": "3.0.0"
    }

