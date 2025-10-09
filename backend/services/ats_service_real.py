"""
ATS Service for CPAS4 - Real Database Integration
Comprehensive Applicant Tracking System with AI features
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
    JobType, ExperienceLevel, InterviewType
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
        """Create a new job posting"""
        try:
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            
            job = Job(
                job_id=job_id,
                **job_create.model_dump(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Set posted_at if status is OPEN
            if job.status == JobStatus.OPEN:
                job.posted_at = datetime.utcnow()
            
            # Store in database
            job_dict = job.model_dump()
            job_dict["created_at"] = job.created_at.isoformat()
            job_dict["updated_at"] = job.updated_at.isoformat()
            if job_dict.get("posted_at"):
                job_dict["posted_at"] = job.posted_at.isoformat()
            if job_dict.get("closed_at"):
                job_dict["closed_at"] = job.closed_at.isoformat()
            
            await self.db.insert_one(self.jobs_collection, job_dict)
            
            logger.info(f"✅ Job created: {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"❌ Error creating job: {e}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        try:
            job_dict = await self.db.find_one(
                self.jobs_collection,
                {"job_id": job_id}
            )
            
            if not job_dict:
                return None
            
            # Convert ISO strings back to datetime
            if "created_at" in job_dict and isinstance(job_dict["created_at"], str):
                job_dict["created_at"] = datetime.fromisoformat(job_dict["created_at"])
            if "updated_at" in job_dict and isinstance(job_dict["updated_at"], str):
                job_dict["updated_at"] = datetime.fromisoformat(job_dict["updated_at"])
            if "posted_at" in job_dict and job_dict["posted_at"]:
                job_dict["posted_at"] = datetime.fromisoformat(job_dict["posted_at"])
            if "closed_at" in job_dict and job_dict["closed_at"]:
                job_dict["closed_at"] = datetime.fromisoformat(job_dict["closed_at"])
            
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

