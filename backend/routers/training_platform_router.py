"""
GRETA Training Platform Router
Complete API endpoints for training management
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
from services.training_platform_service import TrainingPlatformService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["Training Platform"])

# Global instance
training_instance: Optional[TrainingPlatformService] = None

# Pydantic models
class CourseCreateRequest(BaseModel):
    """Course creation request"""
    title: str = Field(..., description="Course title")
    description: str = Field(..., description="Course description")
    category: Optional[str] = Field("AI/ML", description="Course category")
    modules: Optional[List[Dict[str, Any]]] = Field([], description="Course modules")
    duration_hours: Optional[int] = Field(10, description="Duration in hours")
    difficulty: Optional[str] = Field("intermediate", description="Course difficulty")
    instructor: Optional[str] = Field("System", description="Instructor name")

class EnrollmentRequest(BaseModel):
    """Enrollment request"""
    course_id: str = Field(..., description="Course ID")
    user_id: str = Field(..., description="User ID")

class ProgressUpdateRequest(BaseModel):
    """Progress update request"""
    module_id: str = Field(..., description="Module ID")

class AssessmentCreateRequest(BaseModel):
    """Assessment creation request"""
    title: str = Field(..., description="Assessment title")
    questions: List[Dict[str, Any]] = Field(..., description="Questions")
    passing_score: Optional[int] = Field(70, description="Passing score percentage")
    duration_minutes: Optional[int] = Field(60, description="Duration in minutes")

class AssessmentSubmitRequest(BaseModel):
    """Assessment submission request"""
    assessment_id: str = Field(..., description="Assessment ID")
    answers: List[Any] = Field(..., description="Answers")

async def get_training() -> TrainingPlatformService:
    """Get or initialize Training Platform instance"""
    global training_instance
    
    if training_instance is None:
        db = database.database
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client if hasattr(db, 'client') else None)
        await learning_service.initialize()
        
        training_instance = TrainingPlatformService(db, llm_service, learning_service)
        await training_instance.initialize()
    
    return training_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "training_platform",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(service: TrainingPlatformService = Depends(get_training)):
    """Get service metrics"""
    return await service.get_metrics()

# ==================== COURSES ====================

@router.post("/courses")
async def create_course(
    request: CourseCreateRequest,
    service: TrainingPlatformService = Depends(get_training)
):
    """Create a new course"""
    try:
        result = await service.create_course(request.dict())

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating course: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/courses")
async def list_courses(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    service: TrainingPlatformService = Depends(get_training)
):
    """List all courses"""
    try:
        filters = {}
        if category:
            filters["category"] = category
        if difficulty:
            filters["difficulty"] = difficulty

        result = await service.list_courses(filters)
        return result
    except Exception as e:
        logger.error(f"Error listing courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/courses/{course_id}")
async def get_course(
    course_id: str,
    service: TrainingPlatformService = Depends(get_training)
):
    """Get course details"""
    try:
        result = await service.get_course(course_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting course: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENROLLMENTS ====================

@router.post("/enrollments")
async def enroll_user(
    request: EnrollmentRequest,
    service: TrainingPlatformService = Depends(get_training)
):
    """Enroll user in a course"""
    try:
        result = await service.enroll_user(request.course_id, request.user_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enrolling user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/enrollments/{enrollment_id}")
async def get_enrollment(
    enrollment_id: str,
    service: TrainingPlatformService = Depends(get_training)
):
    """Get enrollment details"""
    try:
        result = await service.get_enrollment(enrollment_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enrollment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/enrollments")
async def get_user_enrollments(
    user_id: str,
    service: TrainingPlatformService = Depends(get_training)
):
    """Get all enrollments for a user"""
    try:
        result = await service.get_user_enrollments(user_id)
        return result
    except Exception as e:
        logger.error(f"Error getting user enrollments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enrollments/{enrollment_id}/progress")
async def update_progress(
    enrollment_id: str,
    request: ProgressUpdateRequest,
    service: TrainingPlatformService = Depends(get_training)
):
    """Update course progress"""
    try:
        result = await service.update_progress(enrollment_id, request.module_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ASSESSMENTS ====================

@router.post("/courses/{course_id}/assessments")
async def create_assessment(
    course_id: str,
    request: AssessmentCreateRequest,
    service: TrainingPlatformService = Depends(get_training)
):
    """Create an assessment for a course"""
    try:
        result = await service.create_assessment(course_id, request.dict())

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enrollments/{enrollment_id}/submit-assessment")
async def submit_assessment(
    enrollment_id: str,
    request: AssessmentSubmitRequest,
    service: TrainingPlatformService = Depends(get_training)
):
    """Submit assessment answers"""
    try:
        result = await service.submit_assessment(enrollment_id, request.assessment_id, request.answers)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CERTIFICATES ====================

@router.post("/enrollments/{enrollment_id}/certificate")
async def issue_certificate(
    enrollment_id: str,
    service: TrainingPlatformService = Depends(get_training)
):
    """Issue a certificate for completed course"""
    try:
        result = await service.issue_certificate(enrollment_id)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error issuing certificate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/certificates/{certificate_id}")
async def get_certificate(
    certificate_id: str,
    service: TrainingPlatformService = Depends(get_training)
):
    """Get certificate details"""
    try:
        result = await service.get_certificate(certificate_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting certificate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CATEGORIES ====================

@router.get("/categories")
async def get_categories(service: TrainingPlatformService = Depends(get_training)):
    """Get available course categories"""
    try:
        result = await service.get_categories()
        return result
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    service: TrainingPlatformService = Depends(get_training)
):
    """Get activity log"""
    try:
        result = await service.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

