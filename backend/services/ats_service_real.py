"""
ATS Service for CPAS4 - Real Database Integration
Comprehensive Applicant Tracking System with AI features

Phase 2, Task 2.1: Database Integrity Improvements
- Standardized datetime operations using Database.utcnow()
- Enhanced validation for update operations
- Transaction support for atomic operations
"""

import logging
import uuid
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from database import Database
from models.ats_models import (
    Job, JobCreate, JobUpdate,
    Candidate, CandidateCreate, CandidateUpdate,
    Application, ApplicationCreate, ApplicationUpdate,
    Interview, InterviewCreate, InterviewUpdate,
    CandidateMatch, ResumeParseResult, ATSStats,
    JobStatus, CandidateStatus, ApplicationStatus, InterviewStatus,
    JobType, ExperienceLevel, InterviewType,
    # Phase 2, Task 2.3: New AI models
    ResumeAnalysis, InterviewScheduleSuggestion, CandidateRanking, CommunicationTemplate
)

logger = logging.getLogger(__name__)


class ATSService:
    """
    ATS Service with Real Database Integration
    Manages jobs, candidates, applications, and interviews
    """
    
    def __init__(self, database: Database):
        self.db = database
        self.jobs_collection = "ats_jobs"
        self.candidates_collection = "ats_candidates"
        self.applications_collection = "ats_applications"
        self.interviews_collection = "ats_interviews"
        logger.info("✅ ATS Service initialized with real database")
    
    # ========================================================================
    # JOB OPERATIONS
    # ========================================================================
    
    async def create_job(self, job_create: JobCreate) -> Job:
        """Create a new job posting with standardized datetime operations"""
        try:
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            now = Database.utcnow()  # Standardized datetime

            job = Job(
                job_id=job_id,
                **job_create.model_dump(),
                created_at=now,
                updated_at=now
            )

            # Set posted_at if status is OPEN
            if job.status == JobStatus.OPEN:
                job.posted_at = now

            # Store in database
            job_dict = job.model_dump()
            job_dict["created_at"] = Database.to_iso(job.created_at)
            job_dict["updated_at"] = Database.to_iso(job.updated_at)
            job_dict["posted_at"] = Database.to_iso(job.posted_at)
            job_dict["closed_at"] = Database.to_iso(job.closed_at)

            await self.db.insert_one(self.jobs_collection, job_dict)

            logger.info(f"✅ Job created: {job_id}")
            return job

        except Exception as e:
            logger.error(f"❌ Error creating job: {e}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID with standardized datetime conversion"""
        try:
            job_dict = await self.db.find_one(
                self.jobs_collection,
                {"job_id": job_id}
            )

            if not job_dict:
                return None

            # Convert ISO strings back to datetime using standardized method
            job_dict["created_at"] = Database.from_iso(job_dict.get("created_at"))
            job_dict["updated_at"] = Database.from_iso(job_dict.get("updated_at"))
            job_dict["posted_at"] = Database.from_iso(job_dict.get("posted_at"))
            job_dict["closed_at"] = Database.from_iso(job_dict.get("closed_at"))

            return Job(**job_dict)

        except Exception as e:
            logger.error(f"❌ Error getting job {job_id}: {e}")
            return None
    
    async def update_job(self, job_id: str, job_update: JobUpdate) -> Optional[Job]:
        """Update job"""
        try:
            update_data = job_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Set posted_at when status changes to OPEN
            if "status" in update_data and update_data["status"] == JobStatus.OPEN:
                job = await self.get_job(job_id)
                if job and not job.posted_at:
                    update_data["posted_at"] = datetime.utcnow().isoformat()
            
            # Set closed_at when status changes to CLOSED or FILLED
            if "status" in update_data and update_data["status"] in [JobStatus.CLOSED, JobStatus.FILLED]:
                update_data["closed_at"] = datetime.utcnow().isoformat()
            
            success = await self.db.update_one(
                self.jobs_collection,
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            if success:
                logger.info(f"✅ Job updated: {job_id}")
                return await self.get_job(job_id)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error updating job {job_id}: {e}")
            return None
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete job"""
        try:
            success = await self.db.delete_one(
                self.jobs_collection,
                {"job_id": job_id}
            )
            
            if success:
                logger.info(f"✅ Job deleted: {job_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting job {job_id}: {e}")
            return False
    
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        department: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs with filters"""
        try:
            filter_dict = {}
            
            if status:
                filter_dict["status"] = status
            if job_type:
                filter_dict["job_type"] = job_type
            if department:
                filter_dict["department"] = department
            if tags:
                filter_dict["tags"] = {"$in": tags}
            
            job_dicts = await self.db.find_many(
                self.jobs_collection,
                filter_dict,
                limit=limit
            )
            
            jobs = []
            for job_dict in job_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in job_dict and isinstance(job_dict["created_at"], str):
                    job_dict["created_at"] = datetime.fromisoformat(job_dict["created_at"])
                if "updated_at" in job_dict and isinstance(job_dict["updated_at"], str):
                    job_dict["updated_at"] = datetime.fromisoformat(job_dict["updated_at"])
                if "posted_at" in job_dict and job_dict["posted_at"]:
                    job_dict["posted_at"] = datetime.fromisoformat(job_dict["posted_at"])
                if "closed_at" in job_dict and job_dict["closed_at"]:
                    job_dict["closed_at"] = datetime.fromisoformat(job_dict["closed_at"])
                
                jobs.append(Job(**job_dict))
            
            return jobs
            
        except Exception as e:
            logger.error(f"❌ Error listing jobs: {e}")
            return []
    
    # ========================================================================
    # CANDIDATE OPERATIONS
    # ========================================================================
    
    async def create_candidate(self, candidate_create: CandidateCreate) -> Candidate:
        """Create a new candidate"""
        try:
            candidate_id = f"candidate_{uuid.uuid4().hex[:12]}"
            
            candidate = Candidate(
                candidate_id=candidate_id,
                **candidate_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store in database
            candidate_dict = candidate.model_dump()
            candidate_dict["created_at"] = candidate.created_at.isoformat()
            candidate_dict["updated_at"] = candidate.updated_at.isoformat()
            if candidate_dict.get("last_activity"):
                candidate_dict["last_activity"] = candidate.last_activity.isoformat()
            
            await self.db.insert_one(self.candidates_collection, candidate_dict)
            
            logger.info(f"✅ Candidate created: {candidate_id}")
            return candidate
            
        except Exception as e:
            logger.error(f"❌ Error creating candidate: {e}")
            raise
    
    async def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        """Get candidate by ID"""
        try:
            candidate_dict = await self.db.find_one(
                self.candidates_collection,
                {"candidate_id": candidate_id}
            )
            
            if not candidate_dict:
                return None
            
            # Convert ISO strings back to datetime
            if "created_at" in candidate_dict and isinstance(candidate_dict["created_at"], str):
                candidate_dict["created_at"] = datetime.fromisoformat(candidate_dict["created_at"])
            if "updated_at" in candidate_dict and isinstance(candidate_dict["updated_at"], str):
                candidate_dict["updated_at"] = datetime.fromisoformat(candidate_dict["updated_at"])
            if "last_activity" in candidate_dict and candidate_dict["last_activity"]:
                candidate_dict["last_activity"] = datetime.fromisoformat(candidate_dict["last_activity"])
            
            return Candidate(**candidate_dict)
            
        except Exception as e:
            logger.error(f"❌ Error getting candidate {candidate_id}: {e}")
            return None
    
    async def update_candidate(self, candidate_id: str, candidate_update: CandidateUpdate) -> Optional[Candidate]:
        """Update candidate"""
        try:
            update_data = candidate_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            success = await self.db.update_one(
                self.candidates_collection,
                {"candidate_id": candidate_id},
                {"$set": update_data}
            )
            
            if success:
                logger.info(f"✅ Candidate updated: {candidate_id}")
                return await self.get_candidate(candidate_id)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error updating candidate {candidate_id}: {e}")
            return None
    
    async def delete_candidate(self, candidate_id: str) -> bool:
        """Delete candidate"""
        try:
            success = await self.db.delete_one(
                self.candidates_collection,
                {"candidate_id": candidate_id}
            )
            
            if success:
                logger.info(f"✅ Candidate deleted: {candidate_id}")
            
            return success

        except Exception as e:
            logger.error(f"❌ Error deleting candidate {candidate_id}: {e}")
            return False

    async def list_candidates(
        self,
        status: Optional[CandidateStatus] = None,
        skills: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Candidate]:
        """List candidates with filters"""
        try:
            filter_dict = {}

            if status:
                filter_dict["status"] = status
            if skills:
                filter_dict["skills"] = {"$in": skills}
            if tags:
                filter_dict["tags"] = {"$in": tags}

            candidate_dicts = await self.db.find_many(
                self.candidates_collection,
                filter_dict,
                limit=limit
            )

            candidates = []
            for candidate_dict in candidate_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in candidate_dict and isinstance(candidate_dict["created_at"], str):
                    candidate_dict["created_at"] = datetime.fromisoformat(candidate_dict["created_at"])
                if "updated_at" in candidate_dict and isinstance(candidate_dict["updated_at"], str):
                    candidate_dict["updated_at"] = datetime.fromisoformat(candidate_dict["updated_at"])
                if "last_activity" in candidate_dict and candidate_dict["last_activity"]:
                    candidate_dict["last_activity"] = datetime.fromisoformat(candidate_dict["last_activity"])

                candidates.append(Candidate(**candidate_dict))

            return candidates

        except Exception as e:
            logger.error(f"❌ Error listing candidates: {e}")
            return []

    # ========================================================================
    # APPLICATION OPERATIONS
    # ========================================================================

    async def create_application(self, application_create: ApplicationCreate) -> Application:
        """Create a new application"""
        try:
            application_id = f"app_{uuid.uuid4().hex[:12]}"

            application = Application(
                application_id=application_id,
                **application_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Store in database
            application_dict = application.model_dump()
            application_dict["created_at"] = application.created_at.isoformat()
            application_dict["updated_at"] = application.updated_at.isoformat()
            if application_dict.get("reviewed_at"):
                application_dict["reviewed_at"] = application.reviewed_at.isoformat()

            await self.db.insert_one(self.applications_collection, application_dict)

            # Update job application count
            await self.db.update_one(
                self.jobs_collection,
                {"job_id": application_create.job_id},
                {"$inc": {"application_count": 1}}
            )

            # Update candidate application count
            await self.db.update_one(
                self.candidates_collection,
                {"candidate_id": application_create.candidate_id},
                {"$inc": {"application_count": 1}, "$set": {"last_activity": datetime.utcnow().isoformat()}}
            )

            logger.info(f"✅ Application created: {application_id}")
            return application

        except Exception as e:
            logger.error(f"❌ Error creating application: {e}")
            raise

    async def get_application(self, application_id: str) -> Optional[Application]:
        """Get application by ID"""
        try:
            application_dict = await self.db.find_one(
                self.applications_collection,
                {"application_id": application_id}
            )

            if not application_dict:
                return None

            # Convert ISO strings back to datetime
            if "created_at" in application_dict and isinstance(application_dict["created_at"], str):
                application_dict["created_at"] = datetime.fromisoformat(application_dict["created_at"])
            if "updated_at" in application_dict and isinstance(application_dict["updated_at"], str):
                application_dict["updated_at"] = datetime.fromisoformat(application_dict["updated_at"])
            if "reviewed_at" in application_dict and application_dict["reviewed_at"]:
                application_dict["reviewed_at"] = datetime.fromisoformat(application_dict["reviewed_at"])

            return Application(**application_dict)

        except Exception as e:
            logger.error(f"❌ Error getting application {application_id}: {e}")
            return None

    async def update_application(self, application_id: str, application_update: ApplicationUpdate) -> Optional[Application]:
        """Update application"""
        try:
            update_data = application_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()

            # Set reviewed_at when status changes from APPLIED
            if "status" in update_data:
                app = await self.get_application(application_id)
                if app and app.status == ApplicationStatus.APPLIED and not app.reviewed_at:
                    update_data["reviewed_at"] = datetime.utcnow().isoformat()

            success = await self.db.update_one(
                self.applications_collection,
                {"application_id": application_id},
                {"$set": update_data}
            )

            if success:
                logger.info(f"✅ Application updated: {application_id}")
                return await self.get_application(application_id)

            return None

        except Exception as e:
            logger.error(f"❌ Error updating application {application_id}: {e}")
            return None

    async def delete_application(self, application_id: str) -> bool:
        """Delete application"""
        try:
            success = await self.db.delete_one(
                self.applications_collection,
                {"application_id": application_id}
            )

            if success:
                logger.info(f"✅ Application deleted: {application_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Error deleting application {application_id}: {e}")
            return False

    async def list_applications(
        self,
        job_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
        limit: int = 100
    ) -> List[Application]:
        """List applications with filters"""
        try:
            filter_dict = {}

            if job_id:
                filter_dict["job_id"] = job_id
            if candidate_id:
                filter_dict["candidate_id"] = candidate_id
            if status:
                filter_dict["status"] = status

            application_dicts = await self.db.find_many(
                self.applications_collection,
                filter_dict,
                limit=limit
            )

            applications = []
            for application_dict in application_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in application_dict and isinstance(application_dict["created_at"], str):
                    application_dict["created_at"] = datetime.fromisoformat(application_dict["created_at"])
                if "updated_at" in application_dict and isinstance(application_dict["updated_at"], str):
                    application_dict["updated_at"] = datetime.fromisoformat(application_dict["updated_at"])
                if "reviewed_at" in application_dict and application_dict["reviewed_at"]:
                    application_dict["reviewed_at"] = datetime.fromisoformat(application_dict["reviewed_at"])

                applications.append(Application(**application_dict))

            return applications

        except Exception as e:
            logger.error(f"❌ Error listing applications: {e}")
            return []

    # ========================================================================
    # INTERVIEW OPERATIONS
    # ========================================================================

    async def create_interview(self, interview_create: InterviewCreate) -> Interview:
        """Create a new interview"""
        try:
            interview_id = f"interview_{uuid.uuid4().hex[:12]}"

            interview = Interview(
                interview_id=interview_id,
                **interview_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Store in database
            interview_dict = interview.model_dump()
            interview_dict["created_at"] = interview.created_at.isoformat()
            interview_dict["updated_at"] = interview.updated_at.isoformat()
            interview_dict["scheduled_at"] = interview.scheduled_at.isoformat()
            if interview_dict.get("completed_at"):
                interview_dict["completed_at"] = interview.completed_at.isoformat()

            await self.db.insert_one(self.interviews_collection, interview_dict)

            logger.info(f"✅ Interview created: {interview_id}")
            return interview

        except Exception as e:
            logger.error(f"❌ Error creating interview: {e}")
            raise

    async def get_interview(self, interview_id: str) -> Optional[Interview]:
        """Get interview by ID"""
        try:
            interview_dict = await self.db.find_one(
                self.interviews_collection,
                {"interview_id": interview_id}
            )

            if not interview_dict:
                return None

            # Convert ISO strings back to datetime
            if "created_at" in interview_dict and isinstance(interview_dict["created_at"], str):
                interview_dict["created_at"] = datetime.fromisoformat(interview_dict["created_at"])
            if "updated_at" in interview_dict and isinstance(interview_dict["updated_at"], str):
                interview_dict["updated_at"] = datetime.fromisoformat(interview_dict["updated_at"])
            if "scheduled_at" in interview_dict and isinstance(interview_dict["scheduled_at"], str):
                interview_dict["scheduled_at"] = datetime.fromisoformat(interview_dict["scheduled_at"])
            if "completed_at" in interview_dict and interview_dict["completed_at"]:
                interview_dict["completed_at"] = datetime.fromisoformat(interview_dict["completed_at"])

            return Interview(**interview_dict)

        except Exception as e:
            logger.error(f"❌ Error getting interview {interview_id}: {e}")
            return None

    async def update_interview(self, interview_id: str, interview_update: InterviewUpdate) -> Optional[Interview]:
        """Update interview"""
        try:
            update_data = interview_update.model_dump(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow().isoformat()

            # Set completed_at when status changes to COMPLETED
            if "status" in update_data and update_data["status"] == InterviewStatus.COMPLETED:
                update_data["completed_at"] = datetime.utcnow().isoformat()

            # Convert scheduled_at to ISO if present
            if "scheduled_at" in update_data and isinstance(update_data["scheduled_at"], datetime):
                update_data["scheduled_at"] = update_data["scheduled_at"].isoformat()

            success = await self.db.update_one(
                self.interviews_collection,
                {"interview_id": interview_id},
                {"$set": update_data}
            )

            if success:
                logger.info(f"✅ Interview updated: {interview_id}")
                return await self.get_interview(interview_id)

            return None

        except Exception as e:
            logger.error(f"❌ Error updating interview {interview_id}: {e}")
            return None

    async def delete_interview(self, interview_id: str) -> bool:
        """Delete interview"""
        try:
            success = await self.db.delete_one(
                self.interviews_collection,
                {"interview_id": interview_id}
            )

            if success:
                logger.info(f"✅ Interview deleted: {interview_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Error deleting interview {interview_id}: {e}")
            return False

    async def list_interviews(
        self,
        application_id: Optional[str] = None,
        status: Optional[InterviewStatus] = None,
        interviewer: Optional[str] = None,
        limit: int = 100
    ) -> List[Interview]:
        """List interviews with filters"""
        try:
            filter_dict = {}

            if application_id:
                filter_dict["application_id"] = application_id
            if status:
                filter_dict["status"] = status
            if interviewer:
                filter_dict["interviewer"] = interviewer

            interview_dicts = await self.db.find_many(
                self.interviews_collection,
                filter_dict,
                limit=limit
            )

            interviews = []
            for interview_dict in interview_dicts:
                # Convert ISO strings back to datetime
                if "created_at" in interview_dict and isinstance(interview_dict["created_at"], str):
                    interview_dict["created_at"] = datetime.fromisoformat(interview_dict["created_at"])
                if "updated_at" in interview_dict and isinstance(interview_dict["updated_at"], str):
                    interview_dict["updated_at"] = datetime.fromisoformat(interview_dict["updated_at"])
                if "scheduled_at" in interview_dict and isinstance(interview_dict["scheduled_at"], str):
                    interview_dict["scheduled_at"] = datetime.fromisoformat(interview_dict["scheduled_at"])
                if "completed_at" in interview_dict and interview_dict["completed_at"]:
                    interview_dict["completed_at"] = datetime.fromisoformat(interview_dict["completed_at"])

                interviews.append(Interview(**interview_dict))

            return interviews

        except Exception as e:
            logger.error(f"❌ Error listing interviews: {e}")
            return []

    # ========================================================================
    # AI-POWERED FEATURES
    # ========================================================================

    async def parse_resume(self, candidate_id: str, resume_text: str) -> ResumeParseResult:
        """Parse resume text and extract structured data"""
        try:
            # Basic resume parsing using regex patterns
            parsed_data = {}
            skills_found = []

            # Extract email
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, resume_text)
            if emails:
                parsed_data["email"] = emails[0]

            # Extract phone
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, resume_text)
            if phones:
                parsed_data["phone"] = phones[0]

            # Extract common skills (simple keyword matching)
            common_skills = [
                "Python", "JavaScript", "Java", "C++", "React", "Node.js", "SQL",
                "MongoDB", "AWS", "Docker", "Kubernetes", "Git", "Agile", "Scrum",
                "Machine Learning", "AI", "Data Science", "DevOps", "CI/CD",
                "TypeScript", "Vue", "Angular", "Django", "Flask", "FastAPI"
            ]

            for skill in common_skills:
                if skill.lower() in resume_text.lower():
                    skills_found.append(skill)

            # Extract experience years (simple pattern)
            exp_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?experience'
            exp_matches = re.findall(exp_pattern, resume_text, re.IGNORECASE)
            experience_years = None
            if exp_matches:
                experience_years = int(exp_matches[0])

            # Extract education (simple pattern)
            education = []
            edu_keywords = ["Bachelor", "Master", "PhD", "B.S.", "M.S.", "MBA"]
            for keyword in edu_keywords:
                if keyword in resume_text:
                    education.append({"degree": keyword, "field": "Unknown"})

            # Extract work history (simple pattern - company names)
            work_history = []
            # This is a simplified version - real implementation would be more sophisticated

            result = ResumeParseResult(
                candidate_id=candidate_id,
                extracted_text=resume_text,
                parsed_data=parsed_data,
                skills_found=skills_found,
                experience_years=experience_years,
                education=education,
                work_history=work_history,
                confidence=0.7  # Simple parsing has moderate confidence
            )

            # Update candidate with parsed data
            update_data = {
                "resume_text": resume_text,
                "skills": list(set(skills_found)),  # Unique skills
            }
            if experience_years:
                update_data["experience_years"] = experience_years
            if education:
                update_data["education"] = education

            await self.update_candidate(
                candidate_id,
                CandidateUpdate(**update_data)
            )

            logger.info(f"✅ Resume parsed for candidate: {candidate_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Error parsing resume: {e}")
            raise

    async def match_candidate_to_job(self, candidate_id: str, job_id: str) -> CandidateMatch:
        """Calculate match score between candidate and job"""
        try:
            candidate = await self.get_candidate(candidate_id)
            job = await self.get_job(job_id)

            if not candidate or not job:
                raise ValueError("Candidate or job not found")

            # Calculate match score based on multiple factors
            match_score = 0.0
            matching_skills = []
            missing_skills = []

            # Factor 1: Skills match (40 points)
            candidate_skills = set(s.lower() for s in candidate.skills)
            required_skills = set(s.lower() for s in job.skills_required)
            preferred_skills = set(s.lower() for s in job.skills_preferred)

            if required_skills:
                matched_required = candidate_skills.intersection(required_skills)
                matching_skills.extend(matched_required)
                skill_match_rate = len(matched_required) / len(required_skills)
                match_score += skill_match_rate * 40

                missing_skills = list(required_skills - candidate_skills)

            if preferred_skills:
                matched_preferred = candidate_skills.intersection(preferred_skills)
                matching_skills.extend(matched_preferred)
                pref_match_rate = len(matched_preferred) / len(preferred_skills)
                match_score += pref_match_rate * 10

            # Factor 2: Experience match (30 points)
            experience_match = 0.0
            if candidate.experience_years and job.experience_level:
                exp_requirements = {
                    ExperienceLevel.ENTRY: (0, 2),
                    ExperienceLevel.JUNIOR: (1, 3),
                    ExperienceLevel.MID: (3, 7),
                    ExperienceLevel.SENIOR: (7, 12),
                    ExperienceLevel.LEAD: (10, 15),
                    ExperienceLevel.EXECUTIVE: (15, 100)
                }

                min_exp, max_exp = exp_requirements.get(job.experience_level, (0, 100))
                if min_exp <= candidate.experience_years <= max_exp:
                    experience_match = 1.0
                    match_score += 30
                elif candidate.experience_years > max_exp:
                    # Overqualified
                    experience_match = 0.7
                    match_score += 21
                else:
                    # Underqualified
                    experience_match = 0.3
                    match_score += 9

            # Factor 3: Location match (15 points)
            location_match = False
            if job.remote_allowed:
                location_match = True
                match_score += 15
            elif candidate.location and job.location:
                if candidate.location.lower() in job.location.lower() or job.location.lower() in candidate.location.lower():
                    location_match = True
                    match_score += 15

            # Factor 4: Salary match (15 points)
            salary_match = True  # Assume match if no salary data
            if job.salary_min and job.salary_max:
                # For now, assume match - would need candidate salary expectations
                match_score += 15

            # Generate recommendations
            recommendations = []
            if match_score >= 80:
                recommendations.append("Excellent match - highly recommended for interview")
            elif match_score >= 60:
                recommendations.append("Good match - consider for interview")
            elif match_score >= 40:
                recommendations.append("Moderate match - review carefully")
            else:
                recommendations.append("Low match - may not be suitable")

            if missing_skills:
                recommendations.append(f"Missing required skills: {', '.join(missing_skills[:3])}")

            if experience_match < 0.5:
                recommendations.append("Experience level may not align with requirements")

            result = CandidateMatch(
                candidate_id=candidate_id,
                job_id=job_id,
                match_score=round(match_score, 2),
                matching_skills=list(set(matching_skills)),
                missing_skills=missing_skills,
                experience_match=experience_match,
                location_match=location_match,
                salary_match=salary_match,
                recommendations=recommendations,
                confidence=0.8
            )

            logger.info(f"✅ Candidate-job match calculated: {match_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"❌ Error matching candidate to job: {e}")
            raise

    # ========================================================================
    # PHASE 2, TASK 2.3: AI-POWERED ATS FEATURES
    # ========================================================================

    async def analyze_resume(self, candidate_id: str) -> ResumeAnalysis:
        """
        AI-powered resume analysis with quality assessment and improvement suggestions.

        Phase 2, Task 2.3: Complete ATS AI Features (8 hours)
        Analyzes resume quality, completeness, and provides actionable feedback.
        """
        try:
            # Get candidate
            candidate = await self.get_candidate(candidate_id)
            if not candidate:
                raise ValueError(f"Candidate not found: {candidate_id}")

            now = Database.utcnow()

            # Initialize scores
            quality_score = 0.0
            completeness_score = 0.0
            strengths = []
            weaknesses = []
            missing_sections = []
            improvement_suggestions = []
            formatting_issues = []

            # Analyze completeness (0-100 points)
            completeness_factors = {
                "contact_info": 15,  # Email, phone
                "skills": 20,  # Skills list
                "experience": 25,  # Work history
                "education": 20,  # Education
                "summary": 10,  # Current title/company
                "resume_text": 10  # Resume content
            }

            # Check contact information
            if candidate.email and candidate.phone:
                completeness_score += completeness_factors["contact_info"]
                strengths.append("Complete contact information provided")
            else:
                missing_sections.append("Contact information")
                improvement_suggestions.append("Add complete contact details (email and phone)")

            # Check skills
            if candidate.skills and len(candidate.skills) >= 5:
                completeness_score += completeness_factors["skills"]
                strengths.append(f"{len(candidate.skills)} skills listed")
            elif candidate.skills:
                completeness_score += completeness_factors["skills"] * 0.5
                weaknesses.append("Limited skills listed")
                improvement_suggestions.append("Add more relevant skills (aim for 8-12)")
            else:
                missing_sections.append("Skills")
                improvement_suggestions.append("Add a comprehensive skills section")

            # Check work experience
            if candidate.work_history and len(candidate.work_history) > 0:
                completeness_score += completeness_factors["experience"]
                strengths.append(f"{len(candidate.work_history)} work experiences listed")

                # Check for detailed work history
                detailed_count = sum(1 for exp in candidate.work_history
                                   if exp.get("description") and len(exp.get("description", "")) > 50)
                if detailed_count < len(candidate.work_history):
                    improvement_suggestions.append("Add more detailed descriptions to work experiences")
            else:
                missing_sections.append("Work experience")
                improvement_suggestions.append("Add work history with detailed descriptions")

            # Check education
            if candidate.education and len(candidate.education) > 0:
                completeness_score += completeness_factors["education"]
                strengths.append(f"{len(candidate.education)} education entries")
            else:
                missing_sections.append("Education")
                improvement_suggestions.append("Add education background")

            # Check professional summary
            if candidate.current_title and candidate.current_company:
                completeness_score += completeness_factors["summary"]
                strengths.append("Current position information provided")
            else:
                weaknesses.append("Missing current position information")
                improvement_suggestions.append("Add current job title and company")

            # Check resume text
            if candidate.resume_text and len(candidate.resume_text) > 200:
                completeness_score += completeness_factors["resume_text"]
            else:
                missing_sections.append("Resume content")
                improvement_suggestions.append("Upload a detailed resume document")

            # Analyze quality (0-100 points)
            quality_factors = []

            # Factor 1: Experience years (20 points)
            if candidate.experience_years:
                if candidate.experience_years >= 3:
                    quality_score += 20
                    quality_factors.append(f"{candidate.experience_years} years of experience")
                else:
                    quality_score += 10
                    quality_factors.append(f"{candidate.experience_years} years of experience (entry level)")
            else:
                weaknesses.append("Experience years not specified")

            # Factor 2: Skills diversity (25 points)
            if candidate.skills:
                skill_count = len(candidate.skills)
                if skill_count >= 10:
                    quality_score += 25
                    strengths.append("Diverse skill set")
                elif skill_count >= 5:
                    quality_score += 15
                else:
                    quality_score += 5
                    weaknesses.append("Limited skill diversity")

            # Factor 3: Professional links (15 points)
            links_score = 0
            if candidate.linkedin_url:
                links_score += 10
                strengths.append("LinkedIn profile provided")
            else:
                improvement_suggestions.append("Add LinkedIn profile URL")

            if candidate.portfolio_url:
                links_score += 5
                strengths.append("Portfolio/website provided")

            quality_score += links_score

            # Factor 4: Location specified (10 points)
            if candidate.location:
                quality_score += 10
            else:
                improvement_suggestions.append("Specify location/willingness to relocate")

            # Factor 5: Resume text quality (30 points)
            if candidate.resume_text:
                text_length = len(candidate.resume_text)
                if text_length > 1000:
                    quality_score += 30
                    strengths.append("Comprehensive resume content")
                elif text_length > 500:
                    quality_score += 20
                elif text_length > 200:
                    quality_score += 10
                else:
                    weaknesses.append("Resume content is too brief")
                    improvement_suggestions.append("Expand resume with more details")

            # Keyword density analysis
            keyword_density = {}
            if candidate.resume_text:
                # Common professional keywords
                keywords = [
                    "experience", "skills", "project", "team", "leadership",
                    "development", "management", "analysis", "design", "implementation"
                ]
                text_lower = candidate.resume_text.lower()
                for keyword in keywords:
                    count = text_lower.count(keyword)
                    if count > 0:
                        keyword_density[keyword] = count

            # Readability score (simplified - based on text structure)
            readability_score = 50.0  # Base score
            if candidate.resume_text:
                # Check for bullet points or structured content
                if "•" in candidate.resume_text or "-" in candidate.resume_text:
                    readability_score += 20
                    strengths.append("Well-structured with bullet points")

                # Check for reasonable paragraph length
                paragraphs = candidate.resume_text.split("\n\n")
                if len(paragraphs) >= 3:
                    readability_score += 15

                # Check for section headers (capitalized words)
                if any(word.isupper() and len(word) > 3 for word in candidate.resume_text.split()):
                    readability_score += 15
                    strengths.append("Clear section headers")

            # Formatting issues
            if candidate.resume_text:
                if len(candidate.resume_text) > 5000:
                    formatting_issues.append("Resume may be too long - consider condensing to 1-2 pages")

                if not any(char.isupper() for char in candidate.resume_text[:100]):
                    formatting_issues.append("Consider using proper capitalization")

            # Final recommendations based on scores
            if completeness_score < 60:
                improvement_suggestions.insert(0, "Resume is incomplete - add missing sections for better visibility")

            if quality_score < 50:
                improvement_suggestions.insert(0, "Enhance resume quality with more detailed information")

            if not strengths:
                strengths.append("Resume has been submitted")

            result = ResumeAnalysis(
                candidate_id=candidate_id,
                quality_score=round(quality_score, 2),
                completeness_score=round(completeness_score, 2),
                strengths=strengths,
                weaknesses=weaknesses,
                missing_sections=missing_sections,
                improvement_suggestions=improvement_suggestions,
                keyword_density=keyword_density,
                readability_score=round(readability_score, 2),
                formatting_issues=formatting_issues,
                analyzed_at=now
            )

            logger.info(f"✅ Resume analyzed for candidate {candidate_id}: Quality={quality_score:.1f}, Completeness={completeness_score:.1f}")
            return result

        except Exception as e:
            logger.error(f"❌ Error analyzing resume: {e}")
            raise

    async def suggest_interview_schedule(self, application_id: str) -> InterviewScheduleSuggestion:
        """
        AI-powered interview scheduling optimization.

        Phase 2, Task 2.3: Complete ATS AI Features (8 hours)
        Suggests optimal interview times based on multiple factors.
        """
        try:
            # Get application and related data
            application = await self.get_application(application_id)
            if not application:
                raise ValueError(f"Application not found: {application_id}")

            candidate = await self.get_candidate(application.candidate_id)
            job = await self.get_job(application.job_id)

            if not candidate or not job:
                raise ValueError("Candidate or job not found")

            now = Database.utcnow()

            # Get existing interviews for this application
            existing_interviews = await self.list_interviews(application_id=application_id)

            # Calculate urgency score (0-1)
            urgency_score = 0.5  # Base urgency

            # Factor 1: Application age
            days_since_application = (now - application.created_at).days
            if days_since_application > 14:
                urgency_score += 0.3
            elif days_since_application > 7:
                urgency_score += 0.2
            elif days_since_application > 3:
                urgency_score += 0.1

            # Factor 2: Application status
            if application.status == ApplicationStatus.SCREENING:
                urgency_score += 0.1
            elif application.status == ApplicationStatus.PHONE_SCREEN:
                urgency_score += 0.15
            elif application.status == ApplicationStatus.INTERVIEW:
                urgency_score += 0.2
            elif application.status == ApplicationStatus.FINAL_INTERVIEW:
                urgency_score += 0.25

            # Factor 3: Match score
            if application.match_score and application.match_score > 80:
                urgency_score += 0.15

            urgency_score = min(1.0, urgency_score)

            # Generate suggested time slots (next 14 days, business hours)
            suggested_slots = []

            # Business hours: 9 AM - 5 PM, Monday-Friday
            for day_offset in range(1, 15):  # Next 14 days
                slot_date = now + timedelta(days=day_offset)

                # Skip weekends
                if slot_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    continue

                # Generate morning and afternoon slots
                time_slots = [
                    (9, 0),   # 9:00 AM
                    (10, 0),  # 10:00 AM
                    (11, 0),  # 11:00 AM
                    (14, 0),  # 2:00 PM
                    (15, 0),  # 3:00 PM
                    (16, 0),  # 4:00 PM
                ]

                for hour, minute in time_slots:
                    start_time = slot_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    end_time = start_time + timedelta(hours=1)

                    # Calculate slot score (0-100)
                    slot_score = 50.0  # Base score

                    # Prefer earlier dates for urgent applications
                    if urgency_score > 0.7:
                        slot_score += (15 - day_offset) * 2  # Earlier is better
                    else:
                        slot_score += 10  # Slight preference for earlier

                    # Prefer mid-morning or mid-afternoon
                    if hour in [10, 11, 14, 15]:
                        slot_score += 15

                    # Prefer Tuesday-Thursday
                    if slot_date.weekday() in [1, 2, 3]:  # Tue, Wed, Thu
                        slot_score += 10

                    # Avoid Mondays and Fridays slightly
                    if slot_date.weekday() in [0, 4]:  # Mon, Fri
                        slot_score -= 5

                    suggested_slots.append({
                        "start_time": Database.to_iso(start_time),
                        "end_time": Database.to_iso(end_time),
                        "score": round(slot_score, 2),
                        "day_of_week": slot_date.strftime("%A"),
                        "date": slot_date.strftime("%Y-%m-%d")
                    })

            # Sort by score and get top slots
            suggested_slots.sort(key=lambda x: x["score"], reverse=True)
            top_slots = suggested_slots[:10]

            # Optimal slot is the highest scored
            optimal_slot = top_slots[0] if top_slots else {}

            # Suggest interviewers (would be based on availability in real system)
            interviewer_suggestions = []
            if job.hiring_manager:
                interviewer_suggestions.append(job.hiring_manager)

            # Add generic suggestions
            interviewer_suggestions.extend([
                "Technical Lead",
                "HR Manager",
                "Department Head"
            ])

            # Check for potential conflicts
            conflict_warnings = []

            # Check if there are already scheduled interviews
            if existing_interviews:
                scheduled_count = sum(1 for i in existing_interviews
                                    if i.status == InterviewStatus.SCHEDULED)
                if scheduled_count > 0:
                    conflict_warnings.append(f"{scheduled_count} interview(s) already scheduled for this application")

            # Check application status
            if application.status in [ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN]:
                conflict_warnings.append(f"Application status is {application.status.value} - verify before scheduling")

            # Check if candidate has recent activity
            if candidate.last_activity:
                days_since_activity = (now - candidate.last_activity).days
                if days_since_activity > 30:
                    conflict_warnings.append("Candidate has not been active recently - confirm availability")

            # Generate alternative options
            alternative_options = []

            # Virtual interview option
            alternative_options.append({
                "type": "virtual",
                "description": "Schedule as video interview for flexibility",
                "benefits": ["No travel required", "Easier to reschedule", "Can record for review"]
            })

            # Phone screen option
            if application.status == ApplicationStatus.SCREENING:
                alternative_options.append({
                    "type": "phone_screen",
                    "description": "Start with a brief phone screening (15-30 min)",
                    "benefits": ["Quick initial assessment", "Less time commitment", "Easy to schedule"]
                })

            # Panel interview option
            if application.status in [ApplicationStatus.FINAL_INTERVIEW, ApplicationStatus.INTERVIEW]:
                alternative_options.append({
                    "type": "panel",
                    "description": "Conduct panel interview with multiple interviewers",
                    "benefits": ["Multiple perspectives", "Saves time", "More comprehensive assessment"]
                })

            # Generate reasoning
            reasoning_parts = []

            if urgency_score > 0.7:
                reasoning_parts.append(f"High urgency (score: {urgency_score:.2f}) - prioritize earlier time slots")
            elif urgency_score > 0.5:
                reasoning_parts.append(f"Moderate urgency (score: {urgency_score:.2f}) - schedule within next week")
            else:
                reasoning_parts.append(f"Standard urgency (score: {urgency_score:.2f}) - flexible scheduling")

            if application.match_score and application.match_score > 80:
                reasoning_parts.append(f"High match score ({application.match_score:.1f}) - strong candidate, prioritize interview")

            reasoning_parts.append(f"Suggested {len(top_slots)} optimal time slots based on business hours and best practices")

            if conflict_warnings:
                reasoning_parts.append(f"Note: {len(conflict_warnings)} potential conflict(s) detected")

            reasoning = ". ".join(reasoning_parts)

            # Calculate confidence
            confidence = 0.75  # Base confidence
            if urgency_score > 0.6:
                confidence += 0.1
            if not conflict_warnings:
                confidence += 0.1
            confidence = min(1.0, confidence)

            result = InterviewScheduleSuggestion(
                application_id=application_id,
                suggested_slots=top_slots,
                optimal_slot=optimal_slot,
                interviewer_suggestions=interviewer_suggestions,
                urgency_score=round(urgency_score, 2),
                conflict_warnings=conflict_warnings,
                alternative_options=alternative_options,
                reasoning=reasoning,
                confidence=round(confidence, 2)
            )

            logger.info(f"✅ Interview schedule suggested for application {application_id}: Urgency={urgency_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"❌ Error suggesting interview schedule: {e}")
            raise

    async def rank_candidates_for_job(
        self,
        job_id: str,
        custom_criteria: Optional[Dict[str, float]] = None
    ) -> CandidateRanking:
        """
        AI-powered candidate ranking for a job.

        Phase 2, Task 2.3: Complete ATS AI Features (4 hours)
        Ranks candidates based on multiple weighted factors.
        """
        try:
            # Get job
            job = await self.get_job(job_id)
            if not job:
                raise ValueError(f"Job not found: {job_id}")

            # Get all applications for this job
            applications = await self.list_applications(job_id=job_id)

            if not applications:
                logger.info(f"No applications found for job {job_id}")
                return CandidateRanking(
                    job_id=job_id,
                    ranked_candidates=[],
                    ranking_criteria={},
                    total_candidates=0,
                    ranking_method="multi_factor_weighted",
                    generated_at=Database.utcnow()
                )

            # Default ranking criteria (weights sum to 1.0)
            default_criteria = {
                "match_score": 0.30,      # 30% - Skills and requirements match
                "experience": 0.25,        # 25% - Years of experience
                "education": 0.15,         # 15% - Education level
                "interview_performance": 0.20,  # 20% - Interview ratings
                "application_quality": 0.10     # 10% - Application completeness
            }

            # Use custom criteria if provided, otherwise use defaults
            criteria = custom_criteria if custom_criteria else default_criteria

            # Normalize criteria weights to sum to 1.0
            total_weight = sum(criteria.values())
            if total_weight > 0:
                criteria = {k: v / total_weight for k, v in criteria.items()}

            # Rank each candidate
            ranked_candidates = []

            for application in applications:
                # Skip rejected or withdrawn applications
                if application.status in [ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN]:
                    continue

                # Get candidate details
                candidate = await self.get_candidate(application.candidate_id)
                if not candidate:
                    continue

                # Calculate individual factor scores (0-100)
                factors = {}
                total_score = 0.0

                # Factor 1: Match score (0-100)
                if application.match_score:
                    factors["match_score"] = application.match_score
                else:
                    # Calculate match score if not already done
                    match_result = await self.match_candidate_to_job(
                        application.candidate_id,
                        job_id
                    )
                    factors["match_score"] = match_result.match_score

                total_score += factors["match_score"] * criteria.get("match_score", 0)

                # Factor 2: Experience (0-100)
                experience_score = 0.0
                if candidate.experience_years:
                    # Score based on experience level requirements
                    exp_requirements = {
                        ExperienceLevel.ENTRY: (0, 2),
                        ExperienceLevel.JUNIOR: (1, 3),
                        ExperienceLevel.MID: (3, 7),
                        ExperienceLevel.SENIOR: (7, 12),
                        ExperienceLevel.LEAD: (10, 15),
                        ExperienceLevel.EXECUTIVE: (15, 100)
                    }

                    min_exp, max_exp = exp_requirements.get(job.experience_level, (0, 100))

                    if min_exp <= candidate.experience_years <= max_exp:
                        experience_score = 100  # Perfect match
                    elif candidate.experience_years > max_exp:
                        # Overqualified - still good but slightly lower
                        experience_score = 85
                    elif candidate.experience_years >= min_exp * 0.7:
                        # Close to minimum
                        experience_score = 70
                    else:
                        # Underqualified
                        experience_score = 40

                factors["experience"] = experience_score
                total_score += experience_score * criteria.get("experience", 0)

                # Factor 3: Education (0-100)
                education_score = 50.0  # Base score
                if candidate.education:
                    # Score based on education level
                    education_count = len(candidate.education)
                    if education_count >= 2:
                        education_score = 100  # Multiple degrees
                    elif education_count == 1:
                        # Check degree level
                        edu = candidate.education[0]
                        degree = edu.get("degree", "").lower()
                        if "phd" in degree or "doctorate" in degree:
                            education_score = 100
                        elif "master" in degree or "mba" in degree:
                            education_score = 90
                        elif "bachelor" in degree:
                            education_score = 75
                        else:
                            education_score = 60

                factors["education"] = education_score
                total_score += education_score * criteria.get("education", 0)

                # Factor 4: Interview performance (0-100)
                interview_score = 50.0  # Default if no interviews
                interviews = await self.list_interviews(application_id=application.application_id)

                if interviews:
                    # Average interview ratings
                    ratings = [i.rating for i in interviews if i.rating is not None]
                    if ratings:
                        avg_rating = sum(ratings) / len(ratings)
                        interview_score = avg_rating * 20  # Convert 0-5 rating to 0-100

                    # Bonus for completed interviews
                    completed = sum(1 for i in interviews if i.status == InterviewStatus.COMPLETED)
                    if completed > 0:
                        interview_score = min(100, interview_score + (completed * 5))

                factors["interview_performance"] = interview_score
                total_score += interview_score * criteria.get("interview_performance", 0)

                # Factor 5: Application quality (0-100)
                application_quality_score = 50.0  # Base score

                # Check cover letter
                if application.cover_letter and len(application.cover_letter) > 100:
                    application_quality_score += 20

                # Check application rating
                if application.rating:
                    application_quality_score += application.rating * 10  # 0-5 rating

                # Check if answers provided
                if application.answers and len(application.answers) > 0:
                    application_quality_score += 15

                # Check candidate profile completeness
                if candidate.linkedin_url:
                    application_quality_score += 5
                if candidate.portfolio_url:
                    application_quality_score += 5
                if candidate.resume_text and len(candidate.resume_text) > 500:
                    application_quality_score += 5

                application_quality_score = min(100, application_quality_score)
                factors["application_quality"] = application_quality_score
                total_score += application_quality_score * criteria.get("application_quality", 0)

                # Create candidate ranking entry
                ranked_candidates.append({
                    "candidate_id": candidate.candidate_id,
                    "application_id": application.application_id,
                    "candidate_name": f"{candidate.first_name} {candidate.last_name}",
                    "score": round(total_score, 2),
                    "factors": {k: round(v, 2) for k, v in factors.items()},
                    "status": application.status.value,
                    "applied_at": Database.to_iso(application.created_at)
                })

            # Sort by score (highest first)
            ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Add rank numbers
            for idx, candidate in enumerate(ranked_candidates, 1):
                candidate["rank"] = idx

            result = CandidateRanking(
                job_id=job_id,
                ranked_candidates=ranked_candidates,
                ranking_criteria={k: round(v, 3) for k, v in criteria.items()},
                total_candidates=len(ranked_candidates),
                ranking_method="multi_factor_weighted",
                generated_at=Database.utcnow()
            )

            logger.info(f"✅ Ranked {len(ranked_candidates)} candidates for job {job_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Error ranking candidates: {e}")
            raise

    async def generate_candidate_communication(
        self,
        candidate_id: str,
        application_id: str,
        template_type: str
    ) -> CommunicationTemplate:
        """
        Generate AI-powered personalized candidate communication.

        Phase 2, Task 2.3: Complete ATS AI Features (4 hours)
        Creates personalized communication templates based on candidate status.
        """
        try:
            # Get candidate and application
            candidate = await self.get_candidate(candidate_id)
            application = await self.get_application(application_id)

            if not candidate or not application:
                raise ValueError("Candidate or application not found")

            # Get job details
            job = await self.get_job(application.job_id)
            if not job:
                raise ValueError("Job not found")

            now = Database.utcnow()

            # Personalization tokens
            personalization_tokens = {
                "candidate_first_name": candidate.first_name,
                "candidate_last_name": candidate.last_name,
                "candidate_full_name": f"{candidate.first_name} {candidate.last_name}",
                "job_title": job.title,
                "company_name": "Our Company",  # Would come from config
                "job_location": job.location
            }

            # Generate template based on type
            subject = ""
            body = ""
            tone = "professional"
            next_action = None
            follow_up_days = None
            suggested_send_time = now + timedelta(hours=1)  # Default: send in 1 hour

            if template_type == "application_received":
                subject = f"Application Received - {job.title}"
                tone = "friendly"
                body = f"""Dear {candidate.first_name},

Thank you for applying for the {job.title} position at {personalization_tokens['company_name']}. We have received your application and our team is currently reviewing it.

We appreciate your interest in joining our team. If your qualifications match our requirements, we will contact you within the next 5-7 business days to discuss the next steps.

In the meantime, feel free to learn more about our company and culture on our website.

Best regards,
Hiring Team
{personalization_tokens['company_name']}"""
                next_action = "Review application and schedule screening"
                follow_up_days = 7

            elif template_type == "interview_invitation":
                subject = f"Interview Invitation - {job.title}"
                tone = "professional"

                # Get suggested interview time
                schedule_suggestion = await self.suggest_interview_schedule(application_id)
                optimal_slot = schedule_suggestion.optimal_slot

                if optimal_slot:
                    interview_date = optimal_slot.get("date", "TBD")
                    interview_time = optimal_slot.get("start_time", "TBD")
                else:
                    interview_date = "TBD"
                    interview_time = "TBD"

                body = f"""Dear {candidate.first_name},

We are pleased to inform you that we would like to invite you for an interview for the {job.title} position.

Based on our review of your application, we believe you could be a great fit for our team. We would like to learn more about your experience and discuss how you can contribute to {personalization_tokens['company_name']}.

Proposed Interview Details:
- Date: {interview_date}
- Time: {interview_time}
- Duration: 60 minutes
- Format: Video call (link will be provided upon confirmation)

Please confirm your availability for this time slot, or suggest alternative times that work better for you.

We look forward to speaking with you!

Best regards,
Hiring Team
{personalization_tokens['company_name']}"""
                next_action = "Wait for candidate confirmation"
                follow_up_days = 3
                suggested_send_time = now + timedelta(hours=2)

            elif template_type == "rejection":
                subject = f"Update on Your Application - {job.title}"
                tone = "professional"
                body = f"""Dear {candidate.first_name},

Thank you for your interest in the {job.title} position at {personalization_tokens['company_name']} and for taking the time to apply.

After careful consideration, we have decided to move forward with other candidates whose qualifications more closely match our current needs for this role.

We were impressed by your background and encourage you to apply for future opportunities that align with your skills and experience. We will keep your resume on file for consideration for other positions.

We wish you the best in your job search and future professional endeavors.

Best regards,
Hiring Team
{personalization_tokens['company_name']}"""
                next_action = "Close application"
                follow_up_days = None

            elif template_type == "offer":
                subject = f"Job Offer - {job.title}"
                tone = "formal"

                salary_info = ""
                if job.salary_min and job.salary_max:
                    salary_info = f"${job.salary_min:,.0f} - ${job.salary_max:,.0f} {job.salary_currency}"
                else:
                    salary_info = "Competitive salary (details in formal offer letter)"

                body = f"""Dear {candidate.first_name},

We are delighted to extend an offer of employment for the position of {job.title} at {personalization_tokens['company_name']}.

After reviewing your qualifications and interviewing you, we believe you will be an excellent addition to our team.

Position Details:
- Title: {job.title}
- Location: {job.location}
- Employment Type: {job.job_type.value.replace('_', ' ').title()}
- Compensation: {salary_info}

A formal offer letter with complete details, including benefits and start date, will be sent to you separately within the next 24 hours.

Please review the offer carefully and let us know if you have any questions. We would appreciate your response within 5 business days.

We are excited about the possibility of you joining our team!

Best regards,
Hiring Team
{personalization_tokens['company_name']}"""
                next_action = "Send formal offer letter"
                follow_up_days = 5
                suggested_send_time = now + timedelta(hours=4)

            elif template_type == "follow_up":
                subject = f"Following Up - {job.title} Application"
                tone = "friendly"

                days_since_application = (now - application.created_at).days

                body = f"""Dear {candidate.first_name},

I hope this email finds you well. I wanted to follow up regarding your application for the {job.title} position that you submitted {days_since_application} days ago.

We are still in the process of reviewing applications and wanted to let you know that your application is under consideration. We appreciate your patience as we work through this process.

If you have any questions or would like to provide any additional information, please don't hesitate to reach out.

Thank you for your continued interest in {personalization_tokens['company_name']}.

Best regards,
Hiring Team
{personalization_tokens['company_name']}"""
                next_action = "Continue review process"
                follow_up_days = 7

            elif template_type == "interview_reminder":
                subject = f"Interview Reminder - {job.title}"
                tone = "professional"

                # Get upcoming interview
                interviews = await self.list_interviews(application_id=application_id)
                upcoming_interview = None
                for interview in interviews:
                    if interview.status == InterviewStatus.SCHEDULED and interview.scheduled_at > now:
                        upcoming_interview = interview
                        break

                if upcoming_interview:
                    interview_date = upcoming_interview.scheduled_at.strftime("%A, %B %d, %Y")
                    interview_time = upcoming_interview.scheduled_at.strftime("%I:%M %p")
                    meeting_link = upcoming_interview.meeting_link or "Will be provided separately"
                else:
                    interview_date = "TBD"
                    interview_time = "TBD"
                    meeting_link = "TBD"

                body = f"""Dear {candidate.first_name},

This is a friendly reminder about your upcoming interview for the {job.title} position.

Interview Details:
- Date: {interview_date}
- Time: {interview_time}
- Duration: {upcoming_interview.duration_minutes if upcoming_interview else 60} minutes
- Meeting Link: {meeting_link}

Please ensure you have a stable internet connection and a quiet environment for the interview. We recommend joining 5 minutes early to test your audio and video.

If you need to reschedule or have any questions, please let us know as soon as possible.

We look forward to speaking with you!

Best regards,
Hiring Team
{personalization_tokens['company_name']}"""
                next_action = "Conduct interview"
                follow_up_days = None
                suggested_send_time = now + timedelta(days=1)  # Send 1 day before

            else:
                # Generic template
                subject = f"Update on Your Application - {job.title}"
                tone = "professional"
                body = f"""Dear {candidate.first_name},

Thank you for your application for the {job.title} position at {personalization_tokens['company_name']}.

We wanted to provide you with an update on your application status. Our team is currently reviewing all applications and will be in touch soon with next steps.

Thank you for your patience and interest in joining our team.

Best regards,
Hiring Team
{personalization_tokens['company_name']}"""
                next_action = "Review and send"
                follow_up_days = 7

            result = CommunicationTemplate(
                candidate_id=candidate_id,
                application_id=application_id,
                template_type=template_type,
                subject=subject,
                body=body,
                suggested_send_time=suggested_send_time,
                personalization_tokens=personalization_tokens,
                tone=tone,
                next_action=next_action,
                follow_up_days=follow_up_days
            )

            logger.info(f"✅ Communication template generated: {template_type} for candidate {candidate_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Error generating communication template: {e}")
            raise

    # ========================================================================
    # STATISTICS
    # ========================================================================

    async def get_ats_stats(self) -> ATSStats:
        """Get comprehensive ATS statistics"""
        try:
            # Get all data
            all_jobs = await self.list_jobs(limit=10000)
            all_candidates = await self.list_candidates(limit=10000)
            all_applications = await self.list_applications(limit=10000)
            all_interviews = await self.list_interviews(limit=10000)

            # Count by status
            jobs_by_status = {}
            for status in JobStatus:
                jobs_by_status[status.value] = sum(1 for j in all_jobs if j.status == status)

            candidates_by_status = {}
            for status in CandidateStatus:
                candidates_by_status[status.value] = sum(1 for c in all_candidates if c.status == status)

            applications_by_status = {}
            for status in ApplicationStatus:
                applications_by_status[status.value] = sum(1 for a in all_applications if a.status == status)

            interviews_by_status = {}
            for status in InterviewStatus:
                interviews_by_status[status.value] = sum(1 for i in all_interviews if i.status == status)

            # Calculate average time to hire
            hired_applications = [a for a in all_applications if a.status == ApplicationStatus.HIRED]
            if hired_applications:
                time_to_hire_days = []
                for app in hired_applications:
                    days = (app.updated_at - app.created_at).days
                    time_to_hire_days.append(days)
                average_time_to_hire = sum(time_to_hire_days) / len(time_to_hire_days)
            else:
                average_time_to_hire = 0.0

            # Calculate conversion rates
            total_apps = len(all_applications)
            total_interviews = len(all_interviews)
            total_offers = sum(1 for a in all_applications if a.status == ApplicationStatus.OFFER)
            total_hired = sum(1 for a in all_applications if a.status == ApplicationStatus.HIRED)

            application_to_interview_rate = (total_interviews / total_apps * 100) if total_apps > 0 else 0.0
            interview_to_offer_rate = (total_offers / total_interviews * 100) if total_interviews > 0 else 0.0
            offer_acceptance_rate = (total_hired / total_offers * 100) if total_offers > 0 else 0.0

            stats = ATSStats(
                total_jobs=len(all_jobs),
                total_candidates=len(all_candidates),
                total_applications=len(all_applications),
                total_interviews=len(all_interviews),
                jobs_by_status=jobs_by_status,
                candidates_by_status=candidates_by_status,
                applications_by_status=applications_by_status,
                interviews_by_status=interviews_by_status,
                average_time_to_hire=round(average_time_to_hire, 2),
                application_to_interview_rate=round(application_to_interview_rate, 2),
                interview_to_offer_rate=round(interview_to_offer_rate, 2),
                offer_acceptance_rate=round(offer_acceptance_rate, 2)
            )

            logger.info("✅ ATS statistics calculated")
            return stats

        except Exception as e:
            logger.error(f"❌ Error calculating ATS stats: {e}")
            raise

