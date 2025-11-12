"""
ATS Models for CPAS4
Comprehensive data models for Applicant Tracking System
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class JobStatus(str, Enum):
    """Job posting status"""
    DRAFT = "draft"
    OPEN = "open"
    CLOSED = "closed"
    ON_HOLD = "on_hold"
    FILLED = "filled"


class JobType(str, Enum):
    """Job type"""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"


class ExperienceLevel(str, Enum):
    """Experience level"""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


class CandidateStatus(str, Enum):
    """Candidate status"""
    NEW = "new"
    ACTIVE = "active"
    HIRED = "hired"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    BLACKLISTED = "blacklisted"


class ApplicationStatus(str, Enum):
    """Application status in pipeline"""
    APPLIED = "applied"
    SCREENING = "screening"
    PHONE_SCREEN = "phone_screen"
    INTERVIEW = "interview"
    TECHNICAL_TEST = "technical_test"
    FINAL_INTERVIEW = "final_interview"
    OFFER = "offer"
    HIRED = "hired"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class InterviewType(str, Enum):
    """Interview type"""
    PHONE = "phone"
    VIDEO = "video"
    IN_PERSON = "in_person"
    TECHNICAL = "technical"
    PANEL = "panel"
    BEHAVIORAL = "behavioral"


class InterviewStatus(str, Enum):
    """Interview status"""
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"


# ============================================================================
# JOB MODELS
# ============================================================================

class JobBase(BaseModel):
    """Base job posting information"""
    title: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    skills_required: List[str] = Field(default_factory=list)
    skills_preferred: List[str] = Field(default_factory=list)
    job_type: JobType = JobType.FULL_TIME
    experience_level: ExperienceLevel = ExperienceLevel.MID
    location: str
    remote_allowed: bool = False
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: str = "USD"
    department: Optional[str] = None
    hiring_manager: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class JobCreate(JobBase):
    """Job creation request"""
    status: JobStatus = JobStatus.DRAFT


class JobUpdate(BaseModel):
    """Job update request"""
    title: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[List[str]] = None
    responsibilities: Optional[List[str]] = None
    skills_required: Optional[List[str]] = None
    skills_preferred: Optional[List[str]] = None
    job_type: Optional[JobType] = None
    experience_level: Optional[ExperienceLevel] = None
    location: Optional[str] = None
    remote_allowed: Optional[bool] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    status: Optional[JobStatus] = None
    department: Optional[str] = None
    hiring_manager: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class Job(JobBase):
    """Complete job model"""
    job_id: str
    status: JobStatus = JobStatus.DRAFT
    application_count: int = 0
    view_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    posted_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    created_by: str = "system"


# ============================================================================
# CANDIDATE MODELS
# ============================================================================

class CandidateBase(BaseModel):
    """Base candidate information"""
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    location: Optional[str] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    linkedin_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[int] = None
    education: List[Dict[str, Any]] = Field(default_factory=list)
    work_history: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class CandidateCreate(CandidateBase):
    """Candidate creation request"""
    pass


class CandidateUpdate(BaseModel):
    """Candidate update request"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    linkedin_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    skills: Optional[List[str]] = None
    experience_years: Optional[int] = None
    education: Optional[List[Dict[str, Any]]] = None
    work_history: Optional[List[Dict[str, Any]]] = None
    status: Optional[CandidateStatus] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    resume_text: Optional[str] = None


class Candidate(CandidateBase):
    """Complete candidate model"""
    candidate_id: str
    status: CandidateStatus = CandidateStatus.NEW
    resume_text: Optional[str] = None
    resume_url: Optional[str] = None
    application_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None
    source: str = "direct"


# ============================================================================
# APPLICATION MODELS
# ============================================================================

class ApplicationBase(BaseModel):
    """Base application information"""
    job_id: str
    candidate_id: str
    cover_letter: Optional[str] = None
    answers: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class ApplicationCreate(ApplicationBase):
    """Application creation request"""
    pass


class ApplicationUpdate(BaseModel):
    """Application update request"""
    status: Optional[ApplicationStatus] = None
    cover_letter: Optional[str] = None
    answers: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    rating: Optional[float] = None
    match_score: Optional[float] = None


class Application(ApplicationBase):
    """Complete application model"""
    application_id: str
    status: ApplicationStatus = ApplicationStatus.APPLIED
    notes: Optional[str] = None
    rating: Optional[float] = None
    match_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None


# ============================================================================
# INTERVIEW MODELS
# ============================================================================

class InterviewBase(BaseModel):
    """Base interview information"""
    application_id: str
    interview_type: InterviewType
    scheduled_at: datetime
    duration_minutes: int = 60
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    interviewer: str
    notes: Optional[str] = None


class InterviewCreate(InterviewBase):
    """Interview creation request"""
    pass


class InterviewUpdate(BaseModel):
    """Interview update request"""
    scheduled_at: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    location: Optional[str] = None
    meeting_link: Optional[str] = None
    interviewer: Optional[str] = None
    status: Optional[InterviewStatus] = None
    notes: Optional[str] = None
    feedback: Optional[str] = None
    rating: Optional[float] = None
    recommendation: Optional[str] = None


class Interview(InterviewBase):
    """Complete interview model"""
    interview_id: str
    status: InterviewStatus = InterviewStatus.SCHEDULED
    feedback: Optional[str] = None
    rating: Optional[float] = None
    recommendation: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# ============================================================================
# AI-POWERED FEATURES
# ============================================================================

class CandidateMatch(BaseModel):
    """AI-generated candidate-job match"""
    candidate_id: str
    job_id: str
    match_score: float  # 0-100
    matching_skills: List[str]
    missing_skills: List[str]
    experience_match: float  # 0-1
    location_match: bool
    salary_match: bool
    recommendations: List[str]
    confidence: float  # 0-1


class ResumeParseResult(BaseModel):
    """Resume parsing result"""
    candidate_id: str
    extracted_text: str
    parsed_data: Dict[str, Any]
    skills_found: List[str]
    experience_years: Optional[int] = None
    education: List[Dict[str, Any]] = Field(default_factory=list)
    work_history: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float  # 0-1


class ResumeAnalysis(BaseModel):
    """
    AI-powered resume analysis result.

    Phase 2, Task 2.3: Complete ATS AI Features
    Provides comprehensive resume quality assessment and improvement suggestions.
    """
    candidate_id: str
    quality_score: float  # 0-100
    completeness_score: float  # 0-100
    strengths: List[str]
    weaknesses: List[str]
    missing_sections: List[str]
    improvement_suggestions: List[str]
    keyword_density: Dict[str, int]
    readability_score: float  # 0-100
    formatting_issues: List[str]
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class InterviewScheduleSuggestion(BaseModel):
    """
    AI-powered interview scheduling suggestion.

    Phase 2, Task 2.3: Complete ATS AI Features
    Provides optimal interview scheduling recommendations.
    """
    application_id: str
    suggested_slots: List[Dict[str, Any]]  # List of {start_time, end_time, score}
    optimal_slot: Dict[str, Any]  # Best time slot
    interviewer_suggestions: List[str]
    urgency_score: float  # 0-1
    conflict_warnings: List[str]
    alternative_options: List[Dict[str, Any]]
    reasoning: str
    confidence: float  # 0-1


class CandidateRanking(BaseModel):
    """
    AI-powered candidate ranking for a job.

    Phase 2, Task 2.3: Complete ATS AI Features
    Ranks candidates based on multiple weighted factors.
    """
    job_id: str
    ranked_candidates: List[Dict[str, Any]]  # List of {candidate_id, rank, score, factors}
    ranking_criteria: Dict[str, float]  # Criteria weights
    total_candidates: int
    ranking_method: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CommunicationTemplate(BaseModel):
    """
    AI-generated communication template.

    Phase 2, Task 2.3: Complete ATS AI Features
    Provides personalized candidate communication templates.
    """
    candidate_id: str
    application_id: str
    template_type: str  # e.g., "interview_invitation", "rejection", "offer"
    subject: str
    body: str
    suggested_send_time: datetime
    personalization_tokens: Dict[str, str]
    tone: str  # e.g., "professional", "friendly", "formal"
    next_action: Optional[str] = None
    follow_up_days: Optional[int] = None


# ============================================================================
# STATISTICS MODELS
# ============================================================================

class ATSStats(BaseModel):
    """ATS statistics"""
    total_jobs: int
    total_candidates: int
    total_applications: int
    total_interviews: int
    jobs_by_status: Dict[str, int]
    candidates_by_status: Dict[str, int]
    applications_by_status: Dict[str, int]
    interviews_by_status: Dict[str, int]
    average_time_to_hire: float  # days
    application_to_interview_rate: float  # percentage
    interview_to_offer_rate: float  # percentage
    offer_acceptance_rate: float  # percentage


# ============================================================================
# CSV IMPORT MODELS
# ============================================================================

class CSVImportResult(BaseModel):
    """CSV import result"""
    success: bool
    total_rows: int
    imported: int
    failed: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    imported_ids: List[str] = Field(default_factory=list)
    message: str

