"""
GRETA Business Tools Router
CRM and ATS endpoints for business management
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

from services.crm_service import crm_service
from services.ats_service import ats_service

logger = logging.getLogger(__name__)

router = APIRouter()

# CRM Models
class ContactCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = ""
    company: Optional[str] = ""
    position: Optional[str] = ""
    status: Optional[str] = "lead"
    source: Optional[str] = "manual"
    tags: Optional[List[str]] = []
    value: Optional[float] = 0.0
    probability: Optional[float] = 0.0

class DealCreate(BaseModel):
    title: str
    contact_id: str
    value: float
    stage: Optional[str] = "qualification"
    probability: Optional[float] = 0.0
    expected_close: Optional[str] = None

# ATS Models
class JobCreate(BaseModel):
    title: str
    department: str
    location: str
    type: Optional[str] = "full-time"
    level: Optional[str] = "mid"
    description: str
    requirements: Optional[List[str]] = []
    skills: Optional[List[str]] = []
    salary_min: Optional[float] = 0
    salary_max: Optional[float] = 0
    hiring_manager: Optional[str] = ""

class CandidateCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = ""
    location: Optional[str] = ""
    resume_text: Optional[str] = ""
    skills: Optional[List[str]] = []
    experience_years: Optional[int] = 0
    education: Optional[List[str]] = []
    previous_roles: Optional[List[str]] = []
    source: Optional[str] = "direct"

class ApplicationCreate(BaseModel):
    job_id: str
    candidate_id: str
    salary_expectation: Optional[float] = 0

# CRM Endpoints
@router.get("/crm/contacts")
async def get_contacts(
    status: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    """Get all contacts with optional filters"""
    try:
        filters = {}
        if status:
            filters['status'] = status
        if company:
            filters['company'] = company
        if search:
            filters['search'] = search
        
        result = await crm_service.get_contacts(filters)
        return result
    except Exception as e:
        logger.error(f"Error getting contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crm/contacts")
async def create_contact(contact: ContactCreate):
    """Create a new contact"""
    try:
        result = await crm_service.create_contact(contact.dict())
        return result
    except Exception as e:
        logger.error(f"Error creating contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crm/deals")
async def get_deals(
    stage: Optional[str] = Query(None),
    contact_id: Optional[str] = Query(None)
):
    """Get all deals with optional filters"""
    try:
        filters = {}
        if stage:
            filters['stage'] = stage
        if contact_id:
            filters['contact_id'] = contact_id
        
        result = await crm_service.get_deals(filters)
        return result
    except Exception as e:
        logger.error(f"Error getting deals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crm/deals")
async def create_deal(deal: DealCreate):
    """Create a new deal"""
    try:
        result = await crm_service.create_deal(deal.dict())
        return result
    except Exception as e:
        logger.error(f"Error creating deal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crm/analytics")
async def get_crm_analytics():
    """Get CRM analytics and insights"""
    try:
        result = await crm_service.get_analytics()
        return result
    except Exception as e:
        logger.error(f"Error getting CRM analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crm/insights")
async def get_crm_insights(contact_id: Optional[str] = Query(None)):
    """Get AI-powered CRM insights"""
    try:
        result = await crm_service.ai_insights(contact_id)
        return result
    except Exception as e:
        logger.error(f"Error getting CRM insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ATS Endpoints
@router.get("/ats/jobs")
async def get_jobs(
    status: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    type: Optional[str] = Query(None)
):
    """Get all job postings with optional filters"""
    try:
        filters = {}
        if status:
            filters['status'] = status
        if department:
            filters['department'] = department
        if type:
            filters['type'] = type
        
        result = await ats_service.get_jobs(filters)
        return result
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ats/jobs")
async def create_job(job: JobCreate):
    """Create a new job posting"""
    try:
        result = await ats_service.create_job(job.dict())
        return result
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ats/candidates")
async def get_candidates():
    """Get all candidates"""
    try:
        result = await ats_service.candidates
        candidates_list = [candidate.__dict__ for candidate in result.values()]
        return {
            "success": True,
            "candidates": candidates_list,
            "total": len(candidates_list)
        }
    except Exception as e:
        logger.error(f"Error getting candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ats/candidates")
async def create_candidate(candidate: CandidateCreate):
    """Create a new candidate"""
    try:
        result = await ats_service.create_candidate(candidate.dict())
        return result
    except Exception as e:
        logger.error(f"Error creating candidate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ats/applications")
async def get_applications(
    job_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None)
):
    """Get all applications with optional filters"""
    try:
        filters = {}
        if job_id:
            filters['job_id'] = job_id
        if status:
            filters['status'] = status
        if min_score:
            filters['min_score'] = min_score
        
        result = await ats_service.get_applications(filters)
        return result
    except Exception as e:
        logger.error(f"Error getting applications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ats/applications")
async def create_application(application: ApplicationCreate):
    """Create a new job application"""
    try:
        result = await ats_service.create_application(application.dict())
        return result
    except Exception as e:
        logger.error(f"Error creating application: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ats/analytics")
async def get_ats_analytics():
    """Get ATS analytics and insights"""
    try:
        result = await ats_service.get_analytics()
        return result
    except Exception as e:
        logger.error(f"Error getting ATS analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Combined Business Dashboard
@router.get("/dashboard")
async def get_business_dashboard():
    """Get combined business dashboard data"""
    try:
        crm_analytics = await crm_service.get_analytics()
        ats_analytics = await ats_service.get_analytics()
        crm_insights = await crm_service.ai_insights()
        
        return {
            "success": True,
            "dashboard": {
                "crm": crm_analytics.get("analytics", {}),
                "ats": ats_analytics.get("analytics", {}),
                "insights": crm_insights.get("insights", []),
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    except Exception as e:
        logger.error(f"Error getting business dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@router.get("/health")
async def business_tools_health():
    """Health check for business tools"""
    return {
        "status": "healthy",
        "services": {
            "crm": "active",
            "ats": "active"
        },
        "version": "1.0.0"
    }
