"""
Executive Team Router for Multi-Business Management

Provides API endpoints for managing AI executive teams and their decisions.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from database import Database
from services.executive_team_service import ExecutiveTeamService
from models.business_models import (
    ExecutiveTeam,
    ExecutiveDecision,
    AgentConfig,
    DecisionAuthority
)

router = APIRouter(prefix="/api/v1/executive-teams", tags=["Executive Teams"])


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_service() -> ExecutiveTeamService:
    """Get ExecutiveTeamService instance"""
    db = Database()
    await db.connect()
    return ExecutiveTeamService(db)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ConsultationRequest(BaseModel):
    """Request for advisor consultation"""
    query: str
    context: Optional[Dict[str, Any]] = None


class ConsultationResponse(BaseModel):
    """Response from advisor consultation"""
    execution_id: str
    status: str
    recommendations: Dict[str, Any]
    requires_approval: bool
    timestamp: str


class TeamDecisionRequest(BaseModel):
    """Request for team decision"""
    decision_topic: str
    context: Dict[str, Any]


class TeamDecisionResponse(BaseModel):
    """Response from team decision"""
    decision_topic: str
    executive_inputs: List[Dict[str, Any]]
    synthesis: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: str


class DecisionApprovalRequest(BaseModel):
    """Request to approve/reject a decision"""
    approved: bool
    approved_by: str


# ============================================================================
# EXECUTIVE TEAM ENDPOINTS
# ============================================================================

@router.post("/businesses/{business_id}/create-team", response_model=ExecutiveTeam)
async def create_executive_team(
    business_id: str,
    business_name: str = Query(..., description="Name of the business"),
    team_type: str = Query("elite_advisor", description="Type of team: elite_advisor or full_team"),
    service: ExecutiveTeamService = Depends(get_service)
):
    """
    Create an executive team for a business
    
    - **elite_advisor**: Single elite business advisor (recommended)
    - **full_team**: Full C-suite team (CEO, CFO, COO, CTO)
    """
    try:
        team = await service.create_executive_team(
            business_id=business_id,
            business_name=business_name,
            team_type=team_type
        )
        return team
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create executive team: {str(e)}")


@router.post("/businesses/{business_id}/consult", response_model=ConsultationResponse)
async def consult_advisor(
    business_id: str,
    agent_id: str = Query(..., description="ID of the advisor agent"),
    request: ConsultationRequest = ...,
    service: ExecutiveTeamService = Depends(get_service)
):
    """
    Consult with the elite business advisor
    
    Provide a query and optional context to get strategic recommendations.
    """
    try:
        result = await service.execute_advisor_consultation(
            business_id=business_id,
            agent_id=agent_id,
            query=request.query,
            context=request.context
        )
        return ConsultationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consultation failed: {str(e)}")


@router.post("/businesses/{business_id}/team-decision", response_model=TeamDecisionResponse)
async def coordinate_team_decision(
    business_id: str,
    request: TeamDecisionRequest,
    service: ExecutiveTeamService = Depends(get_service)
):
    """
    Coordinate a decision across the executive team
    
    Involves relevant executives based on the decision topic.
    """
    try:
        # Get business to retrieve executive team
        from services.business_service import BusinessService
        business_service = BusinessService(service.db)
        business = await business_service.get_business(business_id)
        
        if not business:
            raise HTTPException(status_code=404, detail="Business not found")
        
        result = await service.coordinate_team_decision(
            business_id=business_id,
            executive_team=business.executive_team,
            decision_topic=request.decision_topic,
            context=request.context
        )
        return TeamDecisionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Team decision failed: {str(e)}")


# ============================================================================
# DECISION MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/businesses/{business_id}/decisions", response_model=ExecutiveDecision)
async def create_decision(
    business_id: str,
    agent_role: str = Query(..., description="Role of the agent making the decision"),
    decision_type: str = Query(..., description="Type of decision"),
    description: str = Query(..., description="Decision description"),
    amount: Optional[float] = Query(None, description="Financial amount if applicable"),
    requires_approval: bool = Query(True, description="Whether approval is required"),
    service: ExecutiveTeamService = Depends(get_service)
):
    """
    Create a decision record for tracking and approval
    """
    try:
        decision = await service.create_decision_record(
            business_id=business_id,
            agent_role=agent_role,
            decision_type=decision_type,
            description=description,
            amount=amount,
            requires_approval=requires_approval
        )
        return decision
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create decision: {str(e)}")


@router.post("/decisions/{decision_id}/approve", response_model=ExecutiveDecision)
async def approve_decision(
    decision_id: str,
    request: DecisionApprovalRequest,
    service: ExecutiveTeamService = Depends(get_service)
):
    """
    Approve or reject a decision
    """
    try:
        decision = await service.approve_decision(
            decision_id=decision_id,
            approved_by=request.approved_by,
            approved=request.approved
        )
        return decision
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve decision: {str(e)}")


@router.get("/businesses/{business_id}/decisions/pending", response_model=List[ExecutiveDecision])
async def get_pending_decisions(
    business_id: str,
    service: ExecutiveTeamService = Depends(get_service)
):
    """
    Get all pending decisions for a business
    """
    try:
        decisions = await service.get_pending_decisions(business_id)
        return decisions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pending decisions: {str(e)}")


# ============================================================================
# AGENT TEMPLATES ENDPOINTS
# ============================================================================

@router.get("/templates/elite-advisor")
async def get_elite_advisor_template(
    service: ExecutiveTeamService = Depends(get_service)
):
    """Get the elite business advisor template"""
    return service.get_elite_business_advisor_template()


@router.get("/templates/ceo")
async def get_ceo_template(
    service: ExecutiveTeamService = Depends(get_service)
):
    """Get the CEO agent template"""
    return service.get_ceo_template()


@router.get("/templates/cfo")
async def get_cfo_template(
    service: ExecutiveTeamService = Depends(get_service)
):
    """Get the CFO agent template"""
    return service.get_cfo_template()


@router.get("/templates/coo")
async def get_coo_template(
    service: ExecutiveTeamService = Depends(get_service)
):
    """Get the COO agent template"""
    return service.get_coo_template()


@router.get("/templates/cto")
async def get_cto_template(
    service: ExecutiveTeamService = Depends(get_service)
):
    """Get the CTO agent template"""
    return service.get_cto_template()


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "module": "Executive Teams",
        "features": [
            "Elite Business Advisor Creation",
            "Full Executive Team Creation",
            "Advisor Consultation",
            "Team Decision Coordination",
            "Decision Tracking & Approval",
            "Executive Templates",
        ],
        "endpoints": 11,
        "team_types": ["elite_advisor", "full_team"],
    }

