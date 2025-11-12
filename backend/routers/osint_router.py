"""
GRETA OSINT Router
Complete API endpoints for OSINT management
"""

from fastapi import APIRouter, HTTPException, Depends
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
from services.osint_service import OSINTService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/osint", tags=["OSINT"])

# Global instance
osint_instance: Optional[OSINTService] = None

# Pydantic models
class InvestigationCreateRequest(BaseModel):
    """Investigation creation request"""
    name: str = Field(..., description="Investigation name")
    target: str = Field(..., description="Investigation target")
    description: Optional[str] = Field("", description="Investigation description")
    sources: Optional[List[str]] = Field(None, description="Data sources to use")
    priority: Optional[str] = Field("medium", description="Priority (low, medium, high)")
    tags: Optional[List[str]] = Field(None, description="Tags")
    created_by: Optional[str] = Field("system", description="Creator")

class InvestigationUpdateRequest(BaseModel):
    """Investigation update request"""
    updates: Dict[str, Any] = Field(..., description="Fields to update")

class DataGatherRequest(BaseModel):
    """Data gathering request"""
    source: str = Field(..., description="Source to gather from")
    tool: Optional[str] = Field(None, description="Specific tool to use")
    params: Optional[Dict[str, Any]] = Field(None, description="Tool parameters")

class AnalyzeRequest(BaseModel):
    """Analysis request"""
    pipeline_id: Optional[str] = Field(None, description="Analysis pipeline to use")

class CompleteRequest(BaseModel):
    """Investigation completion request"""
    report: Optional[Dict[str, Any]] = Field(None, description="Final report")

async def get_osint() -> OSINTService:
    """Get or initialize OSINT instance"""
    global osint_instance
    
    if osint_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()
        
        osint_instance = OSINTService(db, llm_service, learning_service)
        await osint_instance.initialize()
    
    return osint_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "osint",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(service: OSINTService = Depends(get_osint)):
    """Get service metrics"""
    return await service.get_metrics()

# ==================== INVESTIGATIONS ====================

@router.post("/investigations")
async def create_investigation(
    request: InvestigationCreateRequest,
    service: OSINTService = Depends(get_osint)
):
    """Create a new investigation"""
    try:
        result = await service.create_investigation(request.dict())

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating investigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/investigations")
async def list_investigations(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    service: OSINTService = Depends(get_osint)
):
    """List all investigations"""
    try:
        filters = {}
        if status:
            filters["status"] = status
        if priority:
            filters["priority"] = priority

        result = await service.list_investigations(filters)
        return result
    except Exception as e:
        logger.error(f"Error listing investigations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/investigations/{investigation_id}")
async def get_investigation(
    investigation_id: str,
    service: OSINTService = Depends(get_osint)
):
    """Get investigation details"""
    try:
        result = await service.get_investigation(investigation_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting investigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/investigations/{investigation_id}")
async def update_investigation(
    investigation_id: str,
    request: InvestigationUpdateRequest,
    service: OSINTService = Depends(get_osint)
):
    """Update an investigation"""
    try:
        result = await service.update_investigation(investigation_id, request.updates)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating investigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/investigations/{investigation_id}")
async def delete_investigation(
    investigation_id: str,
    service: OSINTService = Depends(get_osint)
):
    """Delete an investigation"""
    try:
        result = await service.delete_investigation(investigation_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting investigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/investigations/{investigation_id}/complete")
async def complete_investigation(
    investigation_id: str,
    request: CompleteRequest,
    service: OSINTService = Depends(get_osint)
):
    """Mark investigation as complete"""
    try:
        result = await service.complete_investigation(investigation_id, request.report)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing investigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DATA GATHERING ====================

@router.post("/investigations/{investigation_id}/gather")
async def gather_data(
    investigation_id: str,
    request: DataGatherRequest,
    service: OSINTService = Depends(get_osint)
):
    """Gather data for an investigation"""
    try:
        result = await service.gather_data(
            investigation_id,
            request.source,
            request.tool,
            request.params
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error gathering data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ANALYSIS ====================

@router.post("/investigations/{investigation_id}/analyze")
async def analyze_data(
    investigation_id: str,
    request: AnalyzeRequest,
    service: OSINTService = Depends(get_osint)
):
    """Analyze collected data"""
    try:
        result = await service.analyze_data(investigation_id, request.pipeline_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SOURCES ====================

@router.get("/sources")
async def get_sources(service: OSINTService = Depends(get_osint)):
    """Get available OSINT sources"""
    try:
        result = await service.get_sources()
        return result
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    service: OSINTService = Depends(get_osint)
):
    """Get activity log"""
    try:
        result = await service.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

