"""
CRM Router for CPAS4
REST API endpoints for Customer Relationship Management
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import logging

from database import Database
from services.crm_service_new import CRMService
from models.crm_models import (
    Contact, ContactCreate, ContactUpdate,
    Company, CompanyCreate, CompanyUpdate,
    Deal, DealCreate, DealUpdate,
    Activity, ActivityCreate, ActivityUpdate,
    LeadScore, NextAction, CRMStats,
    ContactStatus, DealStage, ActivityType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/crm", tags=["CRM"])


# Dependency to get CRM service
async def get_crm_service() -> CRMService:
    """Get CRM service instance"""
    db = Database()
    await db.connect()
    return CRMService(db)


# ============================================================================
# CONTACT ENDPOINTS
# ============================================================================

@router.post("/contacts", response_model=Contact)
async def create_contact(
    contact_create: ContactCreate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Create a new contact"""
    try:
        contact = await crm_service.create_contact(contact_create)
        return contact
    except Exception as e:
        logger.error(f"Error creating contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contacts/{contact_id}", response_model=Contact)
async def get_contact(
    contact_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Get contact by ID"""
    contact = await crm_service.get_contact(contact_id)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


@router.patch("/contacts/{contact_id}", response_model=Contact)
async def update_contact(
    contact_id: str,
    contact_update: ContactUpdate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Update contact"""
    contact = await crm_service.update_contact(contact_id, contact_update)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


@router.delete("/contacts/{contact_id}")
async def delete_contact(
    contact_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Delete contact"""
    success = await crm_service.delete_contact(contact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"message": "Contact deleted successfully"}


@router.get("/contacts", response_model=List[Contact])
async def list_contacts(
    status: Optional[ContactStatus] = None,
    company_id: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 100,
    crm_service: CRMService = Depends(get_crm_service)
):
    """List contacts with filters"""
    tag_list = tags.split(",") if tags else None
    contacts = await crm_service.list_contacts(
        status=status,
        company_id=company_id,
        tags=tag_list,
        limit=limit
    )
    return contacts


# ============================================================================
# COMPANY ENDPOINTS
# ============================================================================

@router.post("/companies", response_model=Company)
async def create_company(
    company_create: CompanyCreate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Create a new company"""
    try:
        company = await crm_service.create_company(company_create)
        return company
    except Exception as e:
        logger.error(f"Error creating company: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/companies/{company_id}", response_model=Company)
async def get_company(
    company_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Get company by ID"""
    company = await crm_service.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


@router.patch("/companies/{company_id}", response_model=Company)
async def update_company(
    company_id: str,
    company_update: CompanyUpdate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Update company"""
    company = await crm_service.update_company(company_id, company_update)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


@router.delete("/companies/{company_id}")
async def delete_company(
    company_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Delete company"""
    success = await crm_service.delete_company(company_id)
    if not success:
        raise HTTPException(status_code=404, detail="Company not found")
    return {"message": "Company deleted successfully"}


@router.get("/companies", response_model=List[Company])
async def list_companies(
    industry: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 100,
    crm_service: CRMService = Depends(get_crm_service)
):
    """List companies with filters"""
    tag_list = tags.split(",") if tags else None
    companies = await crm_service.list_companies(
        industry=industry,
        tags=tag_list,
        limit=limit
    )
    return companies


# ============================================================================
# DEAL ENDPOINTS
# ============================================================================

@router.post("/deals", response_model=Deal)
async def create_deal(
    deal_create: DealCreate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Create a new deal"""
    try:
        deal = await crm_service.create_deal(deal_create)
        return deal
    except Exception as e:
        logger.error(f"Error creating deal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deals/{deal_id}", response_model=Deal)
async def get_deal(
    deal_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Get deal by ID"""
    deal = await crm_service.get_deal(deal_id)
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    return deal


@router.patch("/deals/{deal_id}", response_model=Deal)
async def update_deal(
    deal_id: str,
    deal_update: DealUpdate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Update deal"""
    deal = await crm_service.update_deal(deal_id, deal_update)
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    return deal


@router.delete("/deals/{deal_id}")
async def delete_deal(
    deal_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Delete deal"""
    success = await crm_service.delete_deal(deal_id)
    if not success:
        raise HTTPException(status_code=404, detail="Deal not found")
    return {"message": "Deal deleted successfully"}


@router.get("/deals", response_model=List[Deal])
async def list_deals(
    stage: Optional[DealStage] = None,
    company_id: Optional[str] = None,
    contact_id: Optional[str] = None,
    limit: int = 100,
    crm_service: CRMService = Depends(get_crm_service)
):
    """List deals with filters"""
    deals = await crm_service.list_deals(
        stage=stage,
        company_id=company_id,
        contact_id=contact_id,
        limit=limit
    )
    return deals


# ============================================================================
# ACTIVITY ENDPOINTS
# ============================================================================

@router.post("/activities", response_model=Activity)
async def create_activity(
    activity_create: ActivityCreate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Create a new activity"""
    try:
        activity = await crm_service.create_activity(activity_create)
        return activity
    except Exception as e:
        logger.error(f"Error creating activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activities/{activity_id}", response_model=Activity)
async def get_activity(
    activity_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Get activity by ID"""
    activity = await crm_service.get_activity(activity_id)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    return activity


@router.patch("/activities/{activity_id}", response_model=Activity)
async def update_activity(
    activity_id: str,
    activity_update: ActivityUpdate,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Update activity"""
    activity = await crm_service.update_activity(activity_id, activity_update)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    return activity


@router.delete("/activities/{activity_id}")
async def delete_activity(
    activity_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Delete activity"""
    success = await crm_service.delete_activity(activity_id)
    if not success:
        raise HTTPException(status_code=404, detail="Activity not found")
    return {"message": "Activity deleted successfully"}


@router.get("/activities", response_model=List[Activity])
async def list_activities(
    activity_type: Optional[ActivityType] = None,
    contact_id: Optional[str] = None,
    company_id: Optional[str] = None,
    deal_id: Optional[str] = None,
    completed: Optional[bool] = None,
    limit: int = 100,
    crm_service: CRMService = Depends(get_crm_service)
):
    """List activities with filters"""
    activities = await crm_service.list_activities(
        activity_type=activity_type,
        contact_id=contact_id,
        company_id=company_id,
        deal_id=deal_id,
        completed=completed,
        limit=limit
    )
    return activities


# ============================================================================
# AI-POWERED ENDPOINTS
# ============================================================================

@router.post("/contacts/{contact_id}/lead-score", response_model=LeadScore)
async def calculate_lead_score(
    contact_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Calculate AI-powered lead score for a contact"""
    try:
        lead_score = await crm_service.calculate_lead_score(contact_id)
        return lead_score
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating lead score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/next-action", response_model=NextAction)
async def suggest_next_action(
    entity_type: str,
    entity_id: str,
    crm_service: CRMService = Depends(get_crm_service)
):
    """Suggest AI-powered next action for an entity"""
    try:
        next_action = await crm_service.suggest_next_action(entity_type, entity_id)
        return next_action
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error suggesting next action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STATISTICS ENDPOINT
# ============================================================================

@router.get("/stats", response_model=CRMStats)
async def get_crm_stats(
    crm_service: CRMService = Depends(get_crm_service)
):
    """Get comprehensive CRM statistics"""
    try:
        stats = await crm_service.get_crm_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting CRM stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """CRM module health check"""
    return {
        "status": "healthy",
        "module": "CRM",
        "features": [
            "Contact Management",
            "Company Management",
            "Deal Pipeline",
            "Activity Tracking",
            "AI Lead Scoring",
            "AI Next Action Suggestions",
            "Statistics & Analytics"
        ],
        "endpoints": 27,
        "database": "MongoDB/File Storage"
    }

