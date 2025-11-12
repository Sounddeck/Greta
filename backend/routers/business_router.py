"""
Business Router for Multi-Business Management

Provides API endpoints for CRUD operations on businesses.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from database import Database
from services.business_service_new import BusinessService
from services.executive_team_service import ExecutiveTeamService
from models.business_models import (
    Business,
    BusinessCreate,
    BusinessUpdate,
    BusinessStatus,
    BusinessStats,
    ExecutiveDecision
)

router = APIRouter(prefix="/api/v1/businesses", tags=["Businesses"])


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_business_service() -> BusinessService:
    """Get BusinessService instance"""
    db = Database()
    await db.connect()
    return BusinessService(db)


async def get_executive_service() -> ExecutiveTeamService:
    """Get ExecutiveTeamService instance"""
    db = Database()
    await db.connect()
    return ExecutiveTeamService(db)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class BusinessCreateRequest(BaseModel):
    """Request to create a business"""
    name: str
    description: str
    industry: str
    tags: List[str] = []
    workspace_path: Optional[str] = None
    isolated_data: bool = True
    enable_executive_team: bool = True
    autonomous_mode: bool = False


class BusinessListResponse(BaseModel):
    """Response for business list"""
    businesses: List[Business]
    total: int


# ============================================================================
# BUSINESS CRUD ENDPOINTS
# ============================================================================

@router.post("", response_model=Business)
async def create_business(
    request: BusinessCreateRequest,
    service: BusinessService = Depends(get_business_service),
    exec_service: ExecutiveTeamService = Depends(get_executive_service)
):
    """
    Create a new business with optional executive team
    
    - Creates business entity
    - Optionally creates default executive team (8 roles)
    - Sets up workspace directory
    """
    try:
        # Create business data object
        business_data = BusinessCreate(
            name=request.name,
            description=request.description,
            industry=request.industry,
            tags=request.tags,
            workspace_path=request.workspace_path,
            isolated_data=request.isolated_data,
            enable_executive_team=request.enable_executive_team,
            autonomous_mode=request.autonomous_mode
        )
        
        # Create business (owner_id should come from auth, using "system" for now)
        business = await service.create_business(business_data, owner_id="system")

        # Note: Executive team creation is handled separately via the executive team service
        # The frontend should call POST /api/v1/executive-teams/businesses/{business_id}/create-team
        # after the business is created if enable_executive_team is true

        return business
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=BusinessListResponse)
async def list_businesses(
    status: Optional[BusinessStatus] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of records to return"),
    service: BusinessService = Depends(get_business_service)
):
    """
    List all businesses with optional filtering
    """
    try:
        businesses, total = await service.list_businesses(
            status=status,
            skip=skip,
            limit=limit
        )

        return BusinessListResponse(
            businesses=businesses,
            total=total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{business_id}", response_model=Business)
async def get_business(
    business_id: str,
    service: BusinessService = Depends(get_business_service)
):
    """
    Get a business by ID
    """
    try:
        business = await service.get_business(business_id)
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        return business
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{business_id}", response_model=Business)
async def update_business(
    business_id: str,
    update_data: BusinessUpdate,
    service: BusinessService = Depends(get_business_service)
):
    """
    Update a business
    """
    try:
        business = await service.update_business(business_id, update_data)
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        return business
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{business_id}")
async def delete_business(
    business_id: str,
    service: BusinessService = Depends(get_business_service)
):
    """
    Delete (archive) a business
    """
    try:
        success = await service.delete_business(business_id)
        if not success:
            raise HTTPException(status_code=404, detail="Business not found")
        return {"message": "Business archived successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{business_id}/stats", response_model=BusinessStats)
async def get_business_stats(
    business_id: str,
    service: BusinessService = Depends(get_business_service)
):
    """
    Get business statistics
    """
    try:
        stats = await service.get_business_stats(business_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{business_id}/decisions/pending", response_model=List[ExecutiveDecision])
async def get_pending_decisions(
    business_id: str,
    service: BusinessService = Depends(get_business_service)
):
    """
    Get pending decisions for a business
    """
    try:
        decisions = await service.get_pending_decisions(business_id)
        return decisions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{business_id}/decisions/{decision_id}/approve", response_model=ExecutiveDecision)
async def approve_decision(
    business_id: str,
    decision_id: str,
    approved: bool,
    service: BusinessService = Depends(get_business_service)
):
    """
    Approve or reject a decision
    """
    try:
        decision = await service.approve_decision(
            decision_id=decision_id,
            approved_by="system",  # Should come from auth
            approved=approved
        )
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")
        return decision
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

