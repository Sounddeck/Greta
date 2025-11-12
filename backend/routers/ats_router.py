"""
ATS Router for CPAS4
REST API endpoints for Applicant Tracking System
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime
import logging

from services.ats_service_real import ATSService
from models.ats_models import (
    Job, JobCreate, JobUpdate,
    Candidate, CandidateCreate, CandidateUpdate,
    Application, ApplicationCreate, ApplicationUpdate,
    Interview, InterviewCreate, InterviewUpdate,
    CandidateMatch, ResumeParseResult, ATSStats,
    JobStatus, CandidateStatus, ApplicationStatus, InterviewStatus,
    JobType, InterviewType
)
from database import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ats", tags=["ATS"])


# Dependency to get ATS service
async def get_ats_service() -> ATSService:
    """Get ATS service instance"""
    db = Database()
    await db.connect()
    return ATSService(db)


# ============================================================================
# JOB ENDPOINTS
# ============================================================================

@router.post("/jobs", response_model=Job, status_code=201)
async def create_job(
    job_create: JobCreate,
    service: ATSService = Depends(get_ats_service)
):
    """Create a new job posting"""
    try:
        job = await service.create_job(job_create)
        return job
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=Job)
async def get_job(
    job_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Get job by ID"""
    job = await service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.patch("/jobs/{job_id}", response_model=Job)
async def update_job(
    job_id: str,
    job_update: JobUpdate,
    service: ATSService = Depends(get_ats_service)
):
    """Update job"""
    job = await service.update_job(job_id, job_update)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(
    job_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Delete job"""
    success = await service.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/jobs", response_model=List[Job])
async def list_jobs(
    status: Optional[JobStatus] = None,
    job_type: Optional[JobType] = None,
    department: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    limit: int = Query(100, le=1000),
    service: ATSService = Depends(get_ats_service)
):
    """List jobs with filters"""
    jobs = await service.list_jobs(
        status=status,
        job_type=job_type,
        department=department,
        tags=tags,
        limit=limit
    )
    return jobs


# ============================================================================
# CANDIDATE ENDPOINTS
# ============================================================================

@router.post("/candidates", response_model=Candidate, status_code=201)
async def create_candidate(
    candidate_create: CandidateCreate,
    service: ATSService = Depends(get_ats_service)
):
    """Create a new candidate"""
    try:
        candidate = await service.create_candidate(candidate_create)
        return candidate
    except Exception as e:
        logger.error(f"Error creating candidate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candidates/{candidate_id}", response_model=Candidate)
async def get_candidate(
    candidate_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Get candidate by ID"""
    candidate = await service.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@router.patch("/candidates/{candidate_id}", response_model=Candidate)
async def update_candidate(
    candidate_id: str,
    candidate_update: CandidateUpdate,
    service: ATSService = Depends(get_ats_service)
):
    """Update candidate"""
    candidate = await service.update_candidate(candidate_id, candidate_update)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@router.delete("/candidates/{candidate_id}", status_code=204)
async def delete_candidate(
    candidate_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Delete candidate"""
    success = await service.delete_candidate(candidate_id)
    if not success:
        raise HTTPException(status_code=404, detail="Candidate not found")


@router.get("/candidates", response_model=List[Candidate])
async def list_candidates(
    status: Optional[CandidateStatus] = None,
    skills: Optional[List[str]] = Query(None),
    tags: Optional[List[str]] = Query(None),
    limit: int = Query(100, le=1000),
    service: ATSService = Depends(get_ats_service)
):
    """List candidates with filters"""
    candidates = await service.list_candidates(
        status=status,
        skills=skills,
        tags=tags,
        limit=limit
    )
    return candidates


# ============================================================================
# APPLICATION ENDPOINTS
# ============================================================================

@router.post("/applications", response_model=Application, status_code=201)
async def create_application(
    application_create: ApplicationCreate,
    service: ATSService = Depends(get_ats_service)
):
    """Create a new application"""
    try:
        application = await service.create_application(application_create)
        return application
    except Exception as e:
        logger.error(f"Error creating application: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/applications/{application_id}", response_model=Application)
async def get_application(
    application_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Get application by ID"""
    application = await service.get_application(application_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return application


@router.patch("/applications/{application_id}", response_model=Application)
async def update_application(
    application_id: str,
    application_update: ApplicationUpdate,
    service: ATSService = Depends(get_ats_service)
):
    """Update application"""
    application = await service.update_application(application_id, application_update)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return application


@router.delete("/applications/{application_id}", status_code=204)
async def delete_application(
    application_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Delete application"""
    success = await service.delete_application(application_id)
    if not success:
        raise HTTPException(status_code=404, detail="Application not found")


@router.get("/applications", response_model=List[Application])
async def list_applications(
    job_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    status: Optional[ApplicationStatus] = None,
    limit: int = Query(100, le=1000),
    service: ATSService = Depends(get_ats_service)
):
    """List applications with filters"""
    applications = await service.list_applications(
        job_id=job_id,
        candidate_id=candidate_id,
        status=status,
        limit=limit
    )
    return applications


# ============================================================================
# INTERVIEW ENDPOINTS
# ============================================================================

@router.post("/interviews", response_model=Interview, status_code=201)
async def create_interview(
    interview_create: InterviewCreate,
    service: ATSService = Depends(get_ats_service)
):
    """Create a new interview"""
    try:
        interview = await service.create_interview(interview_create)
        return interview
    except Exception as e:
        logger.error(f"Error creating interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interviews/{interview_id}", response_model=Interview)
async def get_interview(
    interview_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Get interview by ID"""
    interview = await service.get_interview(interview_id)
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    return interview


@router.patch("/interviews/{interview_id}", response_model=Interview)
async def update_interview(
    interview_id: str,
    interview_update: InterviewUpdate,
    service: ATSService = Depends(get_ats_service)
):
    """Update interview"""
    interview = await service.update_interview(interview_id, interview_update)
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    return interview


@router.delete("/interviews/{interview_id}", status_code=204)
async def delete_interview(
    interview_id: str,
    service: ATSService = Depends(get_ats_service)
):
    """Delete interview"""
    success = await service.delete_interview(interview_id)
    if not success:
        raise HTTPException(status_code=404, detail="Interview not found")


@router.get("/interviews", response_model=List[Interview])
async def list_interviews(
    application_id: Optional[str] = None,
    status: Optional[InterviewStatus] = None,
    interviewer: Optional[str] = None,
    limit: int = Query(100, le=1000),
    service: ATSService = Depends(get_ats_service)
):
    """List interviews with filters"""
    interviews = await service.list_interviews(
        application_id=application_id,
        status=status,
        interviewer=interviewer,
        limit=limit
    )
    return interviews


# ============================================================================
# AI-POWERED ENDPOINTS
# ============================================================================

@router.post("/candidates/{candidate_id}/parse-resume", response_model=ResumeParseResult)
async def parse_resume(
    candidate_id: str,
    resume_text: str,
    service: ATSService = Depends(get_ats_service)
):
    """Parse resume text and extract structured data"""
    try:
        result = await service.parse_resume(candidate_id, resume_text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match-candidate", response_model=CandidateMatch)
async def match_candidate_to_job(
    candidate_id: str = Query(..., description="Candidate ID"),
    job_id: str = Query(..., description="Job ID"),
    service: ATSService = Depends(get_ats_service)
):
    """Calculate match score between candidate and job"""
    try:
        match = await service.match_candidate_to_job(candidate_id, job_id)
        return match
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error matching candidate to job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STATISTICS ENDPOINT
# ============================================================================

@router.get("/stats", response_model=ATSStats)
async def get_ats_stats(
    service: ATSService = Depends(get_ats_service)
):
    """Get comprehensive ATS statistics"""
    try:
        stats = await service.get_ats_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting ATS stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """ATS module health check"""
    return {
        "status": "healthy",
        "module": "ATS",
        "features": [
            "Job Management",
            "Candidate Tracking",
            "Application Pipeline",
            "Interview Scheduling",
            "AI Resume Parsing",
            "AI Candidate Matching",
            "Statistics & Analytics"
        ],
        "endpoints": 27,
        "database": "MongoDB/File Storage"
    }

