"""
GRETA ATS Service
Applicant Tracking System with AI-powered candidate screening
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class JobPosting:
    def __init__(self, job_data: Dict[str, Any]):
        self.id = job_data.get('id', str(uuid.uuid4()))
        self.title = job_data.get('title', '')
        self.department = job_data.get('department', '')
        self.location = job_data.get('location', '')
        self.type = job_data.get('type', 'full-time')  # full-time, part-time, contract, intern
        self.level = job_data.get('level', 'mid')  # entry, mid, senior, executive
        self.description = job_data.get('description', '')
        self.requirements = job_data.get('requirements', [])
        self.skills = job_data.get('skills', [])
        self.salary_min = job_data.get('salary_min', 0)
        self.salary_max = job_data.get('salary_max', 0)
        self.status = job_data.get('status', 'draft')  # draft, active, paused, closed
        self.created_at = job_data.get('created_at', datetime.now().isoformat())
        self.updated_at = job_data.get('updated_at', datetime.now().isoformat())
        self.hiring_manager = job_data.get('hiring_manager', '')

class Candidate:
    def __init__(self, candidate_data: Dict[str, Any]):
        self.id = candidate_data.get('id', str(uuid.uuid4()))
        self.name = candidate_data.get('name', '')
        self.email = candidate_data.get('email', '')
        self.phone = candidate_data.get('phone', '')
        self.location = candidate_data.get('location', '')
        self.resume_text = candidate_data.get('resume_text', '')
        self.skills = candidate_data.get('skills', [])
        self.experience_years = candidate_data.get('experience_years', 0)
        self.education = candidate_data.get('education', [])
        self.previous_roles = candidate_data.get('previous_roles', [])
        self.source = candidate_data.get('source', 'direct')  # direct, referral, linkedin, job_board
        self.created_at = candidate_data.get('created_at', datetime.now().isoformat())
        self.updated_at = candidate_data.get('updated_at', datetime.now().isoformat())
        self.notes = candidate_data.get('notes', [])

class Application:
    def __init__(self, application_data: Dict[str, Any]):
        self.id = application_data.get('id', str(uuid.uuid4()))
        self.job_id = application_data.get('job_id', '')
        self.candidate_id = application_data.get('candidate_id', '')
        self.status = application_data.get('status', 'applied')  # applied, screening, interview, offer, hired, rejected
        self.stage = application_data.get('stage', 'initial_review')
        self.score = application_data.get('score', 0.0)  # AI matching score
        self.applied_at = application_data.get('applied_at', datetime.now().isoformat())
        self.updated_at = application_data.get('updated_at', datetime.now().isoformat())
        self.interview_dates = application_data.get('interview_dates', [])
        self.feedback = application_data.get('feedback', [])
        self.salary_expectation = application_data.get('salary_expectation', 0)

class ATSService:
    """
    AI-powered Applicant Tracking System
    """
    
    def __init__(self):
        self.data_dir = Path("data/ats")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs_file = self.data_dir / "jobs.json"
        self.candidates_file = self.data_dir / "candidates.json"
        self.applications_file = self.data_dir / "applications.json"
        
        self.jobs = self._load_jobs()
        self.candidates = self._load_candidates()
        self.applications = self._load_applications()
        
        logger.info("✅ ATS Service initialized")
    
    def _load_jobs(self) -> Dict[str, JobPosting]:
        """Load job postings from file"""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r') as f:
                    data = json.load(f)
                return {jid: JobPosting(jdata) for jid, jdata in data.items()}
            except Exception as e:
                logger.error(f"Error loading jobs: {e}")
        return {}
    
    def _load_candidates(self) -> Dict[str, Candidate]:
        """Load candidates from file"""
        if self.candidates_file.exists():
            try:
                with open(self.candidates_file, 'r') as f:
                    data = json.load(f)
                return {cid: Candidate(cdata) for cid, cdata in data.items()}
            except Exception as e:
                logger.error(f"Error loading candidates: {e}")
        return {}
    
    def _load_applications(self) -> Dict[str, Application]:
        """Load applications from file"""
        if self.applications_file.exists():
            try:
                with open(self.applications_file, 'r') as f:
                    data = json.load(f)
                return {aid: Application(adata) for aid, adata in data.items()}
            except Exception as e:
                logger.error(f"Error loading applications: {e}")
        return {}
    
    def _save_jobs(self):
        """Save jobs to file"""
        try:
            data = {jid: job.__dict__ for jid, job in self.jobs.items()}
            with open(self.jobs_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving jobs: {e}")
    
    def _save_candidates(self):
        """Save candidates to file"""
        try:
            data = {cid: candidate.__dict__ for cid, candidate in self.candidates.items()}
            with open(self.candidates_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving candidates: {e}")
    
    def _save_applications(self):
        """Save applications to file"""
        try:
            data = {aid: application.__dict__ for aid, application in self.applications.items()}
            with open(self.applications_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving applications: {e}")
    
    async def create_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new job posting"""
        try:
            job = JobPosting(job_data)
            self.jobs[job.id] = job
            self._save_jobs()
            
            return {
                "success": True,
                "job_id": job.id,
                "message": f"Job posting '{job.title}' created successfully"
            }
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_jobs(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get job postings with optional filters"""
        try:
            jobs_list = []
            for job in self.jobs.values():
                job_dict = job.__dict__.copy()
                
                # Add application count
                job_applications = [a for a in self.applications.values() if a.job_id == job.id]
                job_dict['application_count'] = len(job_applications)
                
                # Apply filters if provided
                if filters:
                    if filters.get('status') and job.status != filters['status']:
                        continue
                    if filters.get('department') and job.department != filters['department']:
                        continue
                    if filters.get('type') and job.type != filters['type']:
                        continue
                
                jobs_list.append(job_dict)
            
            return {
                "success": True,
                "jobs": jobs_list,
                "total": len(jobs_list)
            }
        except Exception as e:
            logger.error(f"Error getting jobs: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new candidate"""
        try:
            candidate = Candidate(candidate_data)
            self.candidates[candidate.id] = candidate
            self._save_candidates()
            
            return {
                "success": True,
                "candidate_id": candidate.id,
                "message": f"Candidate {candidate.name} created successfully"
            }
        except Exception as e:
            logger.error(f"Error creating candidate: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new job application"""
        try:
            application = Application(application_data)
            
            # Calculate AI matching score
            if application.job_id in self.jobs and application.candidate_id in self.candidates:
                job = self.jobs[application.job_id]
                candidate = self.candidates[application.candidate_id]
                application.score = await self._calculate_match_score(job, candidate)
            
            self.applications[application.id] = application
            self._save_applications()
            
            return {
                "success": True,
                "application_id": application.id,
                "match_score": application.score,
                "message": "Application submitted successfully"
            }
        except Exception as e:
            logger.error(f"Error creating application: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_applications(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get applications with optional filters"""
        try:
            applications_list = []
            for application in self.applications.values():
                app_dict = application.__dict__.copy()
                
                # Add job and candidate information
                if application.job_id in self.jobs:
                    job = self.jobs[application.job_id]
                    app_dict['job_title'] = job.title
                    app_dict['job_department'] = job.department
                
                if application.candidate_id in self.candidates:
                    candidate = self.candidates[application.candidate_id]
                    app_dict['candidate_name'] = candidate.name
                    app_dict['candidate_email'] = candidate.email
                
                # Apply filters if provided
                if filters:
                    if filters.get('job_id') and application.job_id != filters['job_id']:
                        continue
                    if filters.get('status') and application.status != filters['status']:
                        continue
                    if filters.get('min_score') and application.score < filters['min_score']:
                        continue
                
                applications_list.append(app_dict)
            
            # Sort by score descending
            applications_list.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                "success": True,
                "applications": applications_list,
                "total": len(applications_list)
            }
        except Exception as e:
            logger.error(f"Error getting applications: {e}")
            return {"success": False, "error": str(e)}
    
    async def _calculate_match_score(self, job: JobPosting, candidate: Candidate) -> float:
        """Calculate AI-powered job-candidate match score"""
        try:
            score = 0.0
            max_score = 100.0
            
            # Skills matching (40% of score)
            job_skills = set(skill.lower() for skill in job.skills)
            candidate_skills = set(skill.lower() for skill in candidate.skills)
            
            if job_skills:
                skill_match = len(job_skills.intersection(candidate_skills)) / len(job_skills)
                score += skill_match * 40
            
            # Experience level matching (30% of score)
            experience_match = 0
            if job.level == 'entry' and candidate.experience_years <= 2:
                experience_match = 30
            elif job.level == 'mid' and 2 <= candidate.experience_years <= 7:
                experience_match = 30
            elif job.level == 'senior' and candidate.experience_years >= 5:
                experience_match = 30
            elif job.level == 'executive' and candidate.experience_years >= 10:
                experience_match = 30
            else:
                # Partial match based on proximity
                if job.level == 'mid' and candidate.experience_years > 0:
                    experience_match = 15
                elif job.level == 'senior' and candidate.experience_years >= 3:
                    experience_match = 20
            
            score += experience_match
            
            # Location matching (10% of score)
            if job.location.lower() in candidate.location.lower() or candidate.location.lower() in job.location.lower():
                score += 10
            
            # Resume keyword matching (20% of score)
            job_keywords = set(word.lower() for word in job.description.split() + job.requirements)
            resume_words = set(word.lower() for word in candidate.resume_text.split())
            
            if job_keywords:
                keyword_match = len(job_keywords.intersection(resume_words)) / len(job_keywords)
                score += keyword_match * 20
            
            return min(score, max_score)
            
        except Exception as e:
            logger.error(f"Error calculating match score: {e}")
            return 0.0
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get ATS analytics and insights"""
        try:
            total_jobs = len(self.jobs)
            total_candidates = len(self.candidates)
            total_applications = len(self.applications)
            
            # Job status distribution
            job_status_counts = {}
            for job in self.jobs.values():
                job_status_counts[job.status] = job_status_counts.get(job.status, 0) + 1
            
            # Application status distribution
            app_status_counts = {}
            for app in self.applications.values():
                app_status_counts[app.status] = app_status_counts.get(app.status, 0) + 1
            
            # Average time to hire
            hired_apps = [a for a in self.applications.values() if a.status == 'hired']
            avg_time_to_hire = 0
            if hired_apps:
                total_days = sum(
                    (datetime.fromisoformat(app.updated_at.replace('Z', '+00:00')) - 
                     datetime.fromisoformat(app.applied_at.replace('Z', '+00:00'))).days
                    for app in hired_apps
                )
                avg_time_to_hire = total_days / len(hired_apps)
            
            # Top performing jobs
            job_performance = []
            for job in self.jobs.values():
                job_apps = [a for a in self.applications.values() if a.job_id == job.id]
                hired_count = sum(1 for a in job_apps if a.status == 'hired')
                
                job_performance.append({
                    "job_id": job.id,
                    "title": job.title,
                    "applications": len(job_apps),
                    "hired": hired_count,
                    "conversion_rate": (hired_count / max(len(job_apps), 1)) * 100
                })
            
            job_performance.sort(key=lambda x: x['conversion_rate'], reverse=True)
            
            return {
                "success": True,
                "analytics": {
                    "total_jobs": total_jobs,
                    "total_candidates": total_candidates,
                    "total_applications": total_applications,
                    "job_statuses": job_status_counts,
                    "application_statuses": app_status_counts,
                    "avg_time_to_hire_days": avg_time_to_hire,
                    "top_performing_jobs": job_performance[:5],
                    "conversion_rate": (app_status_counts.get('hired', 0) / max(total_applications, 1)) * 100
                }
            }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"success": False, "error": str(e)}

# Global instance
ats_service = ATSService()
