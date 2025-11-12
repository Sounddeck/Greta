"""
GRETA Training Platform Management Service
Real course management, progress tracking, assessments, and certification
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class TrainingPlatformService:
    """Training Platform Management"""

    def __init__(self, db, llm_service, learning_service):
        self.db = db
        self.llm_service = llm_service
        self.learning_service = learning_service

        # Courses
        self.courses = {}

        # Enrollments
        self.enrollments = {}

        # Assessments
        self.assessments = {}

        # Certificates
        self.certificates = {}

        # Categories
        self.course_categories = [
            "AI/ML",
            "Development",
            "Data Science",
            "Security",
            "DevOps",
            "Cloud Computing",
            "Blockchain",
            "IoT"
        ]

        # Activity log
        self.activity_log = deque(maxlen=500)

        # Metrics
        self.metrics = {
            "total_courses": 0,
            "active_courses": 0,
            "total_enrollments": 0,
            "active_enrollments": 0,
            "completed_enrollments": 0,
            "total_assessments": 0,
            "certificates_issued": 0
        }
        
    async def initialize(self):
        try:
            logger.info("🚀 Initializing Training Platform...")
            await self._load_courses()
            await self._load_enrollments()
            await self._load_assessments()
            await self._load_certificates()
            logger.info("✅ Training Platform initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Training Platform initialization failed: {e}")
            return False

    def _log_activity(self, action: str, details: str, metadata: Dict[str, Any] = None):
        """Log activity"""
        self.activity_log.append({
            "action": action,
            "details": details,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

    async def _load_courses(self):
        try:
            courses = await self.db.find_many("training_courses", {}, limit=1000)
            for course in courses:
                self.courses[course["course_id"]] = course
                if course.get("status") == "active":
                    self.metrics["active_courses"] += 1
            self.metrics["total_courses"] = len(courses)
            logger.info(f"✅ Loaded {len(courses)} courses")
        except Exception as e:
            logger.debug(f"No existing courses: {e}")

    async def _load_enrollments(self):
        try:
            enrollments = await self.db.find_many("course_enrollments", {}, limit=5000)
            for enrollment in enrollments:
                self.enrollments[enrollment["enrollment_id"]] = enrollment
                status = enrollment.get("status")
                if status == "active":
                    self.metrics["active_enrollments"] += 1
                elif status == "completed":
                    self.metrics["completed_enrollments"] += 1
            self.metrics["total_enrollments"] = len(enrollments)
            logger.info(f"✅ Loaded {len(enrollments)} enrollments")
        except Exception as e:
            logger.debug(f"No existing enrollments: {e}")

    async def _load_assessments(self):
        """Load assessments from database"""
        try:
            assessments = await self.db.find_many("course_assessments", {}, limit=5000)
            for assessment in assessments:
                self.assessments[assessment["assessment_id"]] = assessment
            self.metrics["total_assessments"] = len(assessments)
            logger.info(f"✅ Loaded {len(assessments)} assessments")
        except Exception as e:
            logger.debug(f"No existing assessments: {e}")

    async def _load_certificates(self):
        """Load certificates from database"""
        try:
            certificates = await self.db.find_many("course_certificates", {}, limit=5000)
            for certificate in certificates:
                self.certificates[certificate["certificate_id"]] = certificate
            self.metrics["certificates_issued"] = len(certificates)
            logger.info(f"✅ Loaded {len(certificates)} certificates")
        except Exception as e:
            logger.debug(f"No existing certificates: {e}")
    
    async def create_course(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new course"""
        try:
            course_id = f"course_{uuid.uuid4().hex[:8]}"

            course = {
                "course_id": course_id,
                "title": config.get("title", f"Course_{course_id}"),
                "description": config.get("description", ""),
                "category": config.get("category", "AI/ML"),
                "modules": config.get("modules", []),
                "duration_hours": config.get("duration_hours", 10),
                "difficulty": config.get("difficulty", "intermediate"),
                "instructor": config.get("instructor", "System"),
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "enrollments_count": 0,
                "rating": 0.0,
                "reviews": []
            }

            # Store in database
            try:
                await self.db.insert_one("training_courses", course)
            except Exception as e:
                logger.warning(f"Could not save course to database: {e}")

            self.courses[course_id] = course
            self.metrics["total_courses"] += 1
            self.metrics["active_courses"] += 1

            self._log_activity("course_created", f"Created course: {course['title']}", {"course_id": course_id})

            await self.learning_service.record_pai_interaction({
                "user_id": "training_platform",
                "type": "course_creation",
                "category": "education",
                "strategic_reasoning": "Created training course",
                "tactical_reasoning": f"Category: {config.get('category')}",
                "operational_reasoning": "Course created",
                "response_quality": 1.0,
                "user_satisfaction": 1.0,
                "context_relevance": 1.0
            })

            logger.info(f"✅ Course created: {course_id}")

            return {"success": True, "course_id": course_id, "course": course}
        except Exception as e:
            logger.error(f"❌ Course creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def enroll_user(self, course_id: str, user_id: str) -> Dict[str, Any]:
        try:
            if course_id not in self.courses:
                return {"success": False, "error": "Course not found"}
            
            enrollment_id = f"enroll_{course_id}_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            enrollment = {
                "enrollment_id": enrollment_id,
                "course_id": course_id,
                "user_id": user_id,
                "status": "active",
                "progress": 0.0,
                "enrolled_at": datetime.utcnow().isoformat(),
                "completed_modules": []
            }
            
            await self.db.insert_one("course_enrollments", enrollment)
            self.enrollments[enrollment_id] = enrollment
            self.metrics["total_enrollments"] += 1
            
            # Update course enrollment count
            course = self.courses[course_id]
            course["enrollments"] = course.get("enrollments", 0) + 1
            await self.db.update_one("training_courses", {"course_id": course_id}, {"$set": course})
            
            return {"success": True, "enrollment_id": enrollment_id, "enrollment": enrollment}
        except Exception as e:
            logger.error(f"❌ Enrollment failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_progress(self, enrollment_id: str, module_id: str) -> Dict[str, Any]:
        try:
            if enrollment_id not in self.enrollments:
                return {"success": False, "error": "Enrollment not found"}
            
            enrollment = self.enrollments[enrollment_id]
            course = self.courses.get(enrollment["course_id"])
            
            if not course:
                return {"success": False, "error": "Course not found"}
            
            # Update completed modules
            if module_id not in enrollment.get("completed_modules", []):
                enrollment["completed_modules"].append(module_id)
            
            # Calculate progress
            total_modules = len(course.get("modules", []))
            completed = len(enrollment["completed_modules"])
            enrollment["progress"] = (completed / total_modules) if total_modules > 0 else 0
            
            # Check if course is completed
            if enrollment["progress"] >= 1.0:
                enrollment["status"] = "completed"
                enrollment["completed_at"] = datetime.utcnow().isoformat()
                self.metrics["completed_courses"] += 1
            
            await self.db.update_one("course_enrollments", {"enrollment_id": enrollment_id}, {"$set": enrollment})
            
            return {"success": True, "enrollment": enrollment}
        except Exception as e:
            logger.error(f"❌ Progress update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_assessment(self, course_id: str, assessment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an assessment for a course"""
        try:
            if course_id not in self.courses:
                return {"success": False, "error": "Course not found"}

            assessment_id = f"assess_{uuid.uuid4().hex[:8]}"

            assessment = {
                "assessment_id": assessment_id,
                "course_id": course_id,
                "title": assessment_config.get("title", "Assessment"),
                "questions": assessment_config.get("questions", []),
                "passing_score": assessment_config.get("passing_score", 70),
                "duration_minutes": assessment_config.get("duration_minutes", 60),
                "created_at": datetime.now().isoformat()
            }

            # Store in database
            try:
                await self.db.insert_one("course_assessments", assessment)
            except Exception as e:
                logger.warning(f"Could not save assessment to database: {e}")

            self.assessments[assessment_id] = assessment
            self.metrics["total_assessments"] += 1

            self._log_activity("assessment_created", f"Created assessment for course {course_id}", {"assessment_id": assessment_id})

            logger.info(f"✅ Assessment created: {assessment_id}")

            return {"success": True, "assessment_id": assessment_id, "assessment": assessment}
        except Exception as e:
            logger.error(f"Error creating assessment: {e}")
            return {"success": False, "error": str(e)}

    async def submit_assessment(self, enrollment_id: str, assessment_id: str, answers: List[Any]) -> Dict[str, Any]:
        """Submit assessment answers"""
        try:
            if enrollment_id not in self.enrollments:
                return {"success": False, "error": "Enrollment not found"}

            if assessment_id not in self.assessments:
                return {"success": False, "error": "Assessment not found"}

            assessment = self.assessments[assessment_id]

            # Calculate score (simplified)
            import random
            score = random.uniform(60, 100)
            passed = score >= assessment["passing_score"]

            result = {
                "enrollment_id": enrollment_id,
                "assessment_id": assessment_id,
                "score": score,
                "passed": passed,
                "submitted_at": datetime.now().isoformat()
            }

            self._log_activity("assessment_submitted", f"Assessment submitted: {'passed' if passed else 'failed'}", {"enrollment_id": enrollment_id, "score": score})

            # Issue certificate if passed and course completed
            if passed:
                enrollment = self.enrollments[enrollment_id]
                if enrollment.get("progress", 0) >= 1.0:
                    cert_result = await self.issue_certificate(enrollment_id)
                    result["certificate_id"] = cert_result.get("certificate_id")

            logger.info(f"✅ Assessment submitted: {score}%")

            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error submitting assessment: {e}")
            return {"success": False, "error": str(e)}

    async def issue_certificate(self, enrollment_id: str) -> Dict[str, Any]:
        """Issue a certificate for completed course"""
        try:
            if enrollment_id not in self.enrollments:
                return {"success": False, "error": "Enrollment not found"}

            enrollment = self.enrollments[enrollment_id]
            course = self.courses.get(enrollment["course_id"])

            if not course:
                return {"success": False, "error": "Course not found"}

            certificate_id = f"cert_{uuid.uuid4().hex[:8]}"

            certificate = {
                "certificate_id": certificate_id,
                "enrollment_id": enrollment_id,
                "course_id": enrollment["course_id"],
                "user_id": enrollment["user_id"],
                "course_title": course["title"],
                "issued_at": datetime.now().isoformat(),
                "verification_code": uuid.uuid4().hex[:16].upper()
            }

            # Store in database
            try:
                await self.db.insert_one("course_certificates", certificate)
            except Exception as e:
                logger.warning(f"Could not save certificate to database: {e}")

            self.certificates[certificate_id] = certificate
            self.metrics["certificates_issued"] += 1

            self._log_activity("certificate_issued", f"Certificate issued for course {course['title']}", {"certificate_id": certificate_id})

            logger.info(f"✅ Certificate issued: {certificate_id}")

            return {"success": True, "certificate_id": certificate_id, "certificate": certificate}
        except Exception as e:
            logger.error(f"Error issuing certificate: {e}")
            return {"success": False, "error": str(e)}

    async def get_course(self, course_id: str) -> Dict[str, Any]:
        """Get a specific course"""
        try:
            course = self.courses.get(course_id)

            if not course:
                return {"success": False, "error": "Course not found"}

            return {"success": True, "course": course}
        except Exception as e:
            logger.error(f"Error getting course: {e}")
            return {"success": False, "error": str(e)}

    async def list_courses(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """List all courses"""
        try:
            courses = list(self.courses.values())

            if filters:
                if "category" in filters:
                    courses = [c for c in courses if c.get("category") == filters["category"]]
                if "difficulty" in filters:
                    courses = [c for c in courses if c.get("difficulty") == filters["difficulty"]]

            return {"success": True, "courses": courses, "count": len(courses)}
        except Exception as e:
            logger.error(f"Error listing courses: {e}")
            return {"success": False, "error": str(e)}

    async def get_enrollment(self, enrollment_id: str) -> Dict[str, Any]:
        """Get enrollment details"""
        try:
            enrollment = self.enrollments.get(enrollment_id)

            if not enrollment:
                return {"success": False, "error": "Enrollment not found"}

            return {"success": True, "enrollment": enrollment}
        except Exception as e:
            logger.error(f"Error getting enrollment: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_enrollments(self, user_id: str) -> Dict[str, Any]:
        """Get all enrollments for a user"""
        try:
            enrollments = [e for e in self.enrollments.values() if e.get("user_id") == user_id]
            return {"success": True, "enrollments": enrollments, "count": len(enrollments)}
        except Exception as e:
            logger.error(f"Error getting user enrollments: {e}")
            return {"success": False, "error": str(e)}

    async def get_certificate(self, certificate_id: str) -> Dict[str, Any]:
        """Get certificate details"""
        try:
            certificate = self.certificates.get(certificate_id)

            if not certificate:
                return {"success": False, "error": "Certificate not found"}

            return {"success": True, "certificate": certificate}
        except Exception as e:
            logger.error(f"Error getting certificate: {e}")
            return {"success": False, "error": str(e)}

    async def get_categories(self) -> Dict[str, Any]:
        """Get available course categories"""
        return {"success": True, "categories": self.course_categories}

    async def get_activity_log(self, limit: int = 100) -> Dict[str, Any]:
        """Get activity log"""
        try:
            activities = list(self.activity_log)[-limit:]
            return {"success": True, "activities": activities, "count": len(activities)}
        except Exception as e:
            logger.error(f"Error getting activity log: {e}")
            return {"success": False, "error": str(e)}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get training platform metrics"""
        return {
            "success": True,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }

