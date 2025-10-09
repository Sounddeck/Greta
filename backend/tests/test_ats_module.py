"""
ATS Module Test Suite
Comprehensive tests for Applicant Tracking System
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.ats_service_real import ATSService
from models.ats_models import (
    JobCreate, JobUpdate, JobStatus, JobType, ExperienceLevel,
    CandidateCreate, CandidateUpdate, CandidateStatus,
    ApplicationCreate, ApplicationUpdate, ApplicationStatus,
    InterviewCreate, InterviewUpdate, InterviewStatus, InterviewType
)


class ATSModuleTests:
    """Test suite for ATS module"""
    
    def __init__(self):
        self.db = None
        self.service = None
        self.test_results = []
        
        # Test data IDs
        self.test_job_id = None
        self.test_candidate_id = None
        self.test_application_id = None
        self.test_interview_id = None
    
    async def setup(self):
        """Setup test environment"""
        print("\n" + "="*80)
        print("ATS MODULE TEST SUITE")
        print("="*80)
        
        self.db = Database()
        await self.db.connect()
        self.service = ATSService(self.db)
        print("✅ Test environment initialized")
    
    async def test_job_management(self):
        """Test job CRUD operations"""
        print("\n📋 Testing Job Management...")
        
        try:
            # Create job
            job_create = JobCreate(
                title="Senior Python Developer",
                description="We are looking for an experienced Python developer",
                requirements=["5+ years Python experience", "FastAPI knowledge", "MongoDB experience"],
                responsibilities=["Design and implement APIs", "Mentor junior developers"],
                skills_required=["Python", "FastAPI", "MongoDB", "Docker"],
                skills_preferred=["Kubernetes", "AWS", "React"],
                job_type=JobType.FULL_TIME,
                experience_level=ExperienceLevel.SENIOR,
                location="San Francisco, CA",
                remote_allowed=True,
                salary_min=120000.0,
                salary_max=180000.0,
                department="Engineering",
                hiring_manager="John Smith",
                tags=["backend", "python", "senior"],
                status=JobStatus.OPEN
            )
            
            job = await self.service.create_job(job_create)
            self.test_job_id = job.job_id
            assert job.title == "Senior Python Developer"
            assert job.status == JobStatus.OPEN
            assert job.posted_at is not None
            print(f"  ✅ Job created: {job.job_id}")
            
            # Get job
            retrieved_job = await self.service.get_job(self.test_job_id)
            assert retrieved_job is not None
            assert retrieved_job.job_id == self.test_job_id
            print(f"  ✅ Job retrieved: {retrieved_job.title}")
            
            # Update job
            job_update = JobUpdate(
                salary_max=200000.0,
                tags=["backend", "python", "senior", "urgent"]
            )
            updated_job = await self.service.update_job(self.test_job_id, job_update)
            assert updated_job.salary_max == 200000.0
            assert "urgent" in updated_job.tags
            print(f"  ✅ Job updated: salary_max={updated_job.salary_max}")
            
            # List jobs
            jobs = await self.service.list_jobs(status=JobStatus.OPEN, limit=10)
            assert len(jobs) > 0
            print(f"  ✅ Jobs listed: {len(jobs)} open jobs")
            
            self.test_results.append(("Job Management", True, "All CRUD operations working"))
            return True
            
        except Exception as e:
            print(f"  ❌ Job management test failed: {e}")
            self.test_results.append(("Job Management", False, str(e)))
            return False
    
    async def test_candidate_management(self):
        """Test candidate CRUD operations"""
        print("\n👤 Testing Candidate Management...")
        
        try:
            # Create candidate
            candidate_create = CandidateCreate(
                first_name="Jane",
                last_name="Doe",
                email="jane.doe@example.com",
                phone="555-0123",
                location="San Francisco, CA",
                current_title="Python Developer",
                current_company="Tech Corp",
                linkedin_url="https://linkedin.com/in/janedoe",
                skills=["Python", "FastAPI", "MongoDB", "Docker", "AWS"],
                experience_years=6,
                education=[
                    {"degree": "B.S. Computer Science", "school": "Stanford University", "year": 2017}
                ],
                work_history=[
                    {"title": "Python Developer", "company": "Tech Corp", "years": "2020-2024"},
                    {"title": "Junior Developer", "company": "StartupXYZ", "years": "2017-2020"}
                ],
                tags=["python", "backend", "experienced"]
            )
            
            candidate = await self.service.create_candidate(candidate_create)
            self.test_candidate_id = candidate.candidate_id
            assert candidate.first_name == "Jane"
            assert candidate.status == CandidateStatus.NEW
            print(f"  ✅ Candidate created: {candidate.candidate_id}")
            
            # Get candidate
            retrieved_candidate = await self.service.get_candidate(self.test_candidate_id)
            assert retrieved_candidate is not None
            assert retrieved_candidate.email == "jane.doe@example.com"
            print(f"  ✅ Candidate retrieved: {retrieved_candidate.first_name} {retrieved_candidate.last_name}")
            
            # Update candidate
            candidate_update = CandidateUpdate(
                status=CandidateStatus.ACTIVE,
                skills=["Python", "FastAPI", "MongoDB", "Docker", "AWS", "Kubernetes"]
            )
            updated_candidate = await self.service.update_candidate(self.test_candidate_id, candidate_update)
            assert updated_candidate.status == CandidateStatus.ACTIVE
            assert "Kubernetes" in updated_candidate.skills
            print(f"  ✅ Candidate updated: status={updated_candidate.status}")
            
            # List candidates
            candidates = await self.service.list_candidates(status=CandidateStatus.ACTIVE, limit=10)
            assert len(candidates) > 0
            print(f"  ✅ Candidates listed: {len(candidates)} active candidates")
            
            self.test_results.append(("Candidate Management", True, "All CRUD operations working"))
            return True
            
        except Exception as e:
            print(f"  ❌ Candidate management test failed: {e}")
            self.test_results.append(("Candidate Management", False, str(e)))
            return False
    
    async def test_application_pipeline(self):
        """Test application CRUD and pipeline"""
        print("\n📝 Testing Application Pipeline...")
        
        try:
            # Create application
            application_create = ApplicationCreate(
                job_id=self.test_job_id,
                candidate_id=self.test_candidate_id,
                cover_letter="I am very interested in this position...",
                answers={"why_join": "Great company culture", "availability": "2 weeks"},
                tags=["strong-candidate"]
            )
            
            application = await self.service.create_application(application_create)
            self.test_application_id = application.application_id
            assert application.status == ApplicationStatus.APPLIED
            print(f"  ✅ Application created: {application.application_id}")
            
            # Get application
            retrieved_app = await self.service.get_application(self.test_application_id)
            assert retrieved_app is not None
            assert retrieved_app.job_id == self.test_job_id
            print(f"  ✅ Application retrieved: {retrieved_app.application_id}")
            
            # Update application - move through pipeline
            statuses = [
                ApplicationStatus.SCREENING,
                ApplicationStatus.PHONE_SCREEN,
                ApplicationStatus.INTERVIEW
            ]
            
            for status in statuses:
                app_update = ApplicationUpdate(status=status, rating=4.5)
                updated_app = await self.service.update_application(self.test_application_id, app_update)
                assert updated_app.status == status
                print(f"  ✅ Application moved to: {status.value}")
            
            # List applications
            applications = await self.service.list_applications(job_id=self.test_job_id, limit=10)
            assert len(applications) > 0
            print(f"  ✅ Applications listed: {len(applications)} for job")
            
            self.test_results.append(("Application Pipeline", True, "Pipeline workflow working"))
            return True
            
        except Exception as e:
            print(f"  ❌ Application pipeline test failed: {e}")
            self.test_results.append(("Application Pipeline", False, str(e)))
            return False
    
    async def test_interview_scheduling(self):
        """Test interview CRUD and scheduling"""
        print("\n🗓️  Testing Interview Scheduling...")
        
        try:
            # Create interview
            scheduled_time = datetime.utcnow() + timedelta(days=3)
            interview_create = InterviewCreate(
                application_id=self.test_application_id,
                interview_type=InterviewType.VIDEO,
                scheduled_at=scheduled_time,
                duration_minutes=60,
                meeting_link="https://zoom.us/j/123456789",
                interviewer="John Smith",
                notes="Technical interview - focus on Python and system design"
            )
            
            interview = await self.service.create_interview(interview_create)
            self.test_interview_id = interview.interview_id
            assert interview.status == InterviewStatus.SCHEDULED
            print(f"  ✅ Interview created: {interview.interview_id}")
            
            # Get interview
            retrieved_interview = await self.service.get_interview(self.test_interview_id)
            assert retrieved_interview is not None
            assert retrieved_interview.interviewer == "John Smith"
            print(f"  ✅ Interview retrieved: {retrieved_interview.interview_type.value}")
            
            # Update interview - complete it
            interview_update = InterviewUpdate(
                status=InterviewStatus.COMPLETED,
                feedback="Strong technical skills, good communication",
                rating=4.5,
                recommendation="Move to final round"
            )
            updated_interview = await self.service.update_interview(self.test_interview_id, interview_update)
            assert updated_interview.status == InterviewStatus.COMPLETED
            assert updated_interview.rating == 4.5
            print(f"  ✅ Interview completed: rating={updated_interview.rating}")
            
            # List interviews
            interviews = await self.service.list_interviews(application_id=self.test_application_id, limit=10)
            assert len(interviews) > 0
            print(f"  ✅ Interviews listed: {len(interviews)} for application")
            
            self.test_results.append(("Interview Scheduling", True, "Scheduling workflow working"))
            return True
            
        except Exception as e:
            print(f"  ❌ Interview scheduling test failed: {e}")
            self.test_results.append(("Interview Scheduling", False, str(e)))
            return False

    async def test_resume_parsing(self):
        """Test AI resume parsing"""
        print("\n🤖 Testing AI Resume Parsing...")

        try:
            resume_text = """
            Jane Doe
            jane.doe@example.com
            555-0123

            EXPERIENCE
            Senior Python Developer at Tech Corp (2020-2024)
            - 6 years of experience in Python development
            - Built scalable APIs using FastAPI and Django
            - Worked with MongoDB, PostgreSQL, and Redis
            - Deployed applications on AWS using Docker and Kubernetes

            EDUCATION
            Master of Science in Computer Science - MIT (2019)
            Bachelor of Science in Computer Science - Stanford (2017)

            SKILLS
            Python, JavaScript, FastAPI, Django, React, MongoDB, PostgreSQL,
            Docker, Kubernetes, AWS, Git, Agile, Machine Learning
            """

            result = await self.service.parse_resume(self.test_candidate_id, resume_text)
            assert result.candidate_id == self.test_candidate_id
            assert len(result.skills_found) > 0
            assert "Python" in result.skills_found
            assert result.confidence > 0
            print(f"  ✅ Resume parsed: {len(result.skills_found)} skills found")
            print(f"  ✅ Skills: {', '.join(result.skills_found[:5])}")

            # Verify candidate was updated
            updated_candidate = await self.service.get_candidate(self.test_candidate_id)
            assert updated_candidate.resume_text is not None
            print(f"  ✅ Candidate updated with resume data")

            self.test_results.append(("AI Resume Parsing", True, f"{len(result.skills_found)} skills extracted"))
            return True

        except Exception as e:
            print(f"  ❌ Resume parsing test failed: {e}")
            self.test_results.append(("AI Resume Parsing", False, str(e)))
            return False

    async def test_candidate_matching(self):
        """Test AI candidate-job matching"""
        print("\n🎯 Testing AI Candidate Matching...")

        try:
            match = await self.service.match_candidate_to_job(
                self.test_candidate_id,
                self.test_job_id
            )

            assert match.candidate_id == self.test_candidate_id
            assert match.job_id == self.test_job_id
            assert 0 <= match.match_score <= 100
            assert len(match.matching_skills) > 0
            assert len(match.recommendations) > 0

            print(f"  ✅ Match score calculated: {match.match_score}/100")
            print(f"  ✅ Matching skills: {', '.join(match.matching_skills[:5])}")
            print(f"  ✅ Recommendations: {match.recommendations[0]}")

            self.test_results.append(("AI Candidate Matching", True, f"Score: {match.match_score}/100"))
            return True

        except Exception as e:
            print(f"  ❌ Candidate matching test failed: {e}")
            self.test_results.append(("AI Candidate Matching", False, str(e)))
            return False

    async def test_statistics(self):
        """Test ATS statistics"""
        print("\n📊 Testing ATS Statistics...")

        try:
            stats = await self.service.get_ats_stats()

            assert stats.total_jobs > 0
            assert stats.total_candidates > 0
            assert stats.total_applications > 0
            assert stats.total_interviews > 0
            assert len(stats.jobs_by_status) > 0
            assert len(stats.candidates_by_status) > 0
            assert len(stats.applications_by_status) > 0
            assert len(stats.interviews_by_status) > 0

            print(f"  ✅ Total jobs: {stats.total_jobs}")
            print(f"  ✅ Total candidates: {stats.total_candidates}")
            print(f"  ✅ Total applications: {stats.total_applications}")
            print(f"  ✅ Total interviews: {stats.total_interviews}")
            print(f"  ✅ Application→Interview rate: {stats.application_to_interview_rate}%")

            self.test_results.append(("ATS Statistics", True, "All metrics calculated"))
            return True

        except Exception as e:
            print(f"  ❌ Statistics test failed: {e}")
            self.test_results.append(("ATS Statistics", False, str(e)))
            return False

    async def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")

        try:
            # Delete in reverse order of creation
            if self.test_interview_id:
                await self.service.delete_interview(self.test_interview_id)
                print(f"  ✅ Deleted interview: {self.test_interview_id}")

            if self.test_application_id:
                await self.service.delete_application(self.test_application_id)
                print(f"  ✅ Deleted application: {self.test_application_id}")

            if self.test_candidate_id:
                await self.service.delete_candidate(self.test_candidate_id)
                print(f"  ✅ Deleted candidate: {self.test_candidate_id}")

            if self.test_job_id:
                await self.service.delete_job(self.test_job_id)
                print(f"  ✅ Deleted job: {self.test_job_id}")

        except Exception as e:
            print(f"  ⚠️ Cleanup warning: {e}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)

        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {test_name}: {message}")

        print("\n" + "-"*80)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print("="*80)

        # Overall status
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - ATS MODULE FULLY FUNCTIONAL")
            print("\n✅ Job Management: WORKING")
            print("✅ Candidate Management: WORKING")
            print("✅ Application Pipeline: WORKING")
            print("✅ Interview Scheduling: WORKING")
            print("✅ AI Resume Parsing: WORKING")
            print("✅ AI Candidate Matching: WORKING")
            print("✅ Statistics: WORKING")
        else:
            print("\n⚠️ SOME TESTS FAILED - REVIEW REQUIRED")

        print("="*80 + "\n")

    async def run_all_tests(self):
        """Run all tests"""
        await self.setup()

        # Run tests in sequence
        await self.test_job_management()
        await self.test_candidate_management()
        await self.test_application_pipeline()
        await self.test_interview_scheduling()
        await self.test_resume_parsing()
        await self.test_candidate_matching()
        await self.test_statistics()

        # Cleanup
        await self.cleanup()

        # Print summary
        self.print_summary()


async def main():
    """Main test runner"""
    tests = ATSModuleTests()
    await tests.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())


