"""
GRETA Multi-Modal Router
Complete API endpoints for multi-modal processing
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import database
from services.llm_service import get_llm_service
from services.learning_service import LearningService
from services.multimodal_service import MultiModalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/multimodal", tags=["Multi-Modal"])

# Global instance
multimodal_instance: Optional[MultiModalService] = None

# Pydantic models
class ProcessMediaRequest(BaseModel):
    """Media processing request"""
    media_type: str = Field(..., description="Media type (audio, video, image, document)")
    processor: str = Field(..., description="Processor type")
    input: Dict[str, Any] = Field(..., description="Input data (file_path, url, or base64)")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")

async def get_multimodal() -> MultiModalService:
    """Get or initialize Multi-Modal instance"""
    global multimodal_instance
    
    if multimodal_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()
        
        multimodal_instance = MultiModalService(db, llm_service, learning_service)
        await multimodal_instance.initialize()
    
    return multimodal_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "multimodal",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(service: MultiModalService = Depends(get_multimodal)):
    """Get service metrics"""
    return await service.get_metrics()

# ==================== MEDIA PROCESSING ====================

@router.post("/process")
async def process_media(
    request: ProcessMediaRequest,
    service: MultiModalService = Depends(get_multimodal)
):
    """Process media (audio, video, image, document)"""
    try:
        result = await service.process_media({
            "media_type": request.media_type,
            "processor": request.processor,
            "input": request.input,
            "options": request.options
        })

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing media: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== JOB MANAGEMENT ====================

@router.get("/jobs")
async def get_all_jobs(
    media_type: Optional[str] = None,
    status: Optional[str] = None,
    service: MultiModalService = Depends(get_multimodal)
):
    """Get all processing jobs"""
    try:
        result = await service.get_all_jobs(media_type, status)
        return result
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    service: MultiModalService = Depends(get_multimodal)
):
    """Get a specific job"""
    try:
        result = await service.get_job(job_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    service: MultiModalService = Depends(get_multimodal)
):
    """Cancel a processing job"""
    try:
        result = await service.cancel_job(job_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    service: MultiModalService = Depends(get_multimodal)
):
    """Delete a job"""
    try:
        result = await service.delete_job(job_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PROCESSORS INFO ====================

@router.get("/processors")
async def get_processors(service: MultiModalService = Depends(get_multimodal)):
    """Get available processors"""
    try:
        result = await service.get_processors()
        return result
    except Exception as e:
        logger.error(f"Error getting processors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    service: MultiModalService = Depends(get_multimodal)
):
    """Get activity log"""
    try:
        result = await service.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

