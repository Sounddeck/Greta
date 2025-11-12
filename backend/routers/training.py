"""
🎓 GRETA TRAINING API ROUTER
Interactive training endpoints for Greta PAI learning platform
Complete AI agent development curriculum accessible via FastAPI
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Body
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from backend.services.interactive_training import greta_training, training_api, LessonResult
from database import Database
from utils.error_handling import GretaException
from utils.performance import performance_monitor

router = APIRouter(prefix="/api/training", tags=["training"])
db = Database()

# ===== REQUEST/RESPONSE MODELS =====

class StartLessonRequest(BaseModel):
    user_id: str
    lesson_id: str

class ExecuteCodeRequest(BaseModel):
    user_id: str
    code: str
    test_cases: Optional[List[Dict[str, Any]]] = None

class SubmitLessonRequest(BaseModel):
    user_id: str
    lesson_id: str
    score: int = Field(ge=0, le=100)
    time_taken: int = Field(gt=0)  # minutes
    mistakes_made: List[Dict[str, Any]] = []
    hints_used: int = Field(ge=0)
    confidence_level: str = "medium"

class CurriculumResponse(BaseModel):
    current_level: str
    available_lessons: int
    priority_lessons: List[Dict[str, Any]]
    completed_lessons: int
    learning_streak: int
    advanced_unlocked: List[str]
    recommended_study_time: int

class TrainingStatusResponse(BaseModel):
    system_health: str
    total_users: int
    active_lessons: int
    completion_rate: float
    popular_lessons: List[str]

# ===== API ENDPOINTS =====

@router.get("/curriculum/{user_id}")
async def get_personalized_curriculum(user_id: str):
    """
    🎓 Get personalized training curriculum for a student
    Returns adaptive content based on learning progress and preferences
    """
    try:
        with performance_monitor("training_curriculum"):
            curriculum = await training_api.get_curriculum(None, user_id)

            return CurriculumResponse(
                current_level=curriculum["current_level"],
                available_lessons=curriculum["available_lessons"],
                priority_lessons=curriculum["priority_lessons"],
                completed_lessons=curriculum["completed_lessons"],
                learning_streak=curriculum["learning_streak"],
                advanced_unlocked=curriculum["advanced_unlocked"],
                recommended_study_time=curriculum.get("recommended_study_time", 30)
            )

    except Exception as e:
        logger.error(f"Failed to get curriculum for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load curriculum: {str(e)}")

@router.post("/lesson/start")
async def start_interactive_lesson(request: StartLessonRequest):
    """
    🚀 Start an interactive lesson session
    Initializes progress tracking and returns lesson content
    """
    try:
        with performance_monitor("start_lesson"):
            result = await training_api.start_lesson(None, request.user_id, request.lesson_id)

            if result["status"] == "prerequisites_not_met":
                raise HTTPException(
                    status_code=400,
                    detail=f"Prerequisites not met. Required lessons: {', '.join(result['required_lessons'])}"
                )

            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start lesson {request.lesson_id} for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start lesson: {str(e)}")

@router.post("/code/execute")
async def execute_training_code(request: ExecuteCodeRequest):
    """
    ⚡ Execute code in safe training environment
    Runs user-provided code and returns graded results with feedback
    """
    try:
        with performance_monitor("code_execution"):
            # Default test cases if none provided
            test_cases = request.test_cases or [
                {"input": "print('Hello World')", "expected": None, "description": "Basic print test"}
            ]

            result = await training_api.execute_code(None, request.user_id, request.code, test_cases)

            # Log code execution for learning analytics
            await db.training_code_executions.insert_one({
                "user_id": request.user_id,
                "code_length": len(request.code),
                "tests_passed": result.get("tests_passed", 0),
                "score": result.get("score", 0),
                "timestamp": datetime.now()
            })

            return result

    except Exception as e:
        logger.error(f"Code execution failed for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Code execution error: {str(e)}")

@router.post("/lesson/submit")
async def submit_lesson_result(request: SubmitLessonRequest):
    """
    📊 Submit lesson completion results
    Processes scores, unlocks new content, achievements, and progress updates
    """
    try:
        with performance_monitor("submit_lesson"):
            lesson_result = LessonResult(
                lesson_id=request.lesson_id,
                user_id=request.user_id,
                score=request.score,
                time_taken=request.time_taken,
                mistakes_made=request.mistakes_made,
                hints_used=request.hints_used,
                timestamp=datetime.now(),
                feedback="",  # Generated by system
                confidence_level=request.confidence_level
            )

            result = await training_api.submit_lesson_result(request.user_id, lesson_result)

            # Add celebration sounds/effects for achievements
            if result.get("achievements"):
                result["celebration"] = {
                    "achievements": result["achievements"],
                    "sound_effect": "achievement_unlock.wav",
                    "animation": "confetti_animation"
                }

            return result

    except Exception as e:
        logger.error(f"Failed to submit lesson result for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit lesson: {str(e)}")

@router.get("/leaderboard")
async def get_training_leaderboard():
    """
    🏆 Get training leaderboard for gamification
    Shows top performers and learning achievements
    """
    try:
        # Aggregate leaderboard data
        pipeline = [
            {"$group": {
                "_id": "$user_id",
                "total_score": {"$sum": "$score"},
                "lessons_completed": {"$sum": 1},
                "avg_score": {"$avg": "$score"},
                "high_score": {"$max": "$score"}
            }},
            {"$sort": {"total_score": -1}},
            {"$limit": 20}
        ]

        leaderboard = await db.training_results.aggregate(pipeline).to_list(length=20)

        return {
            "leaderboard": [
                {
                    "rank": i + 1,
                    "user_id": entry["_id"],
                    "total_score": entry["total_score"],
                    "lessons_completed": entry["lessons_completed"],
                    "avg_score": round(entry["avg_score"], 1),
                    "high_score": entry["high_score"]
                }
                for i, entry in enumerate(leaderboard)
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to load leaderboard")

@router.get("/lessons")
async def get_available_lessons():
    """
    📚 Get catalog of all available training lessons
    Returns lesson metadata without starting sessions
    """
    try:
        lessons_data = []
        for lesson_id, lesson in greta_training.lessons.items():
            lessons_data.append({
                "id": lesson_id,
                "title": lesson.title,
                "description": lesson.description,
                "level": lesson.required_level.value,
                "duration": lesson.estimated_duration,
                "points": lesson.points_value,
                "type": lesson.lesson_type.value,
                "prerequisites": lesson.prerequisites
            })

        return {"lessons": lessons_data, "total_count": len(lessons_data)}

    except Exception as e:
        logger.error(f"Failed to get lessons catalog: {e}")
        raise HTTPException(status_code=500, detail="Failed to load lessons")

@router.get("/progress/{user_id}")
async def get_detailed_progress(user_id: str):
    """
    📊 Get detailed learning progress and analytics
    Comprehensive view of student development and learning patterns
    """
    try:
        # Get user progress
        progress = await greta_training.get_user_progress(user_id)

        # Get detailed lesson history
        lesson_history = []
        if hasattr(db, 'training_results'):
            cursor = db.training_results.find({"user_id": user_id}).sort("timestamp", -1)
            lesson_history = await cursor.to_list(length=50)

        # Get code execution history
        code_history = []
        if hasattr(db, 'training_code_executions'):
            cursor = db.training_code_executions.find({"user_id": user_id}).sort("timestamp", -1).limit(20)
            code_history = await cursor.to_list(length=20)

        # Calculate learning insights
        insights = await greta_training._calculate_learning_insights(progress, lesson_history)

        return {
            "current_level": progress.current_level.value,
            "completed_lessons": len(progress.completed_lessons),
            "lesson_scores": progress.lesson_scores,
            "learning_streak": greta_training._calculate_streak(progress),
            "last_activity": progress.last_activity.isoformat(),
            "lesson_history": lesson_history,
            "code_execution_history": code_history,
            "learning_insights": insights,
            "next_milestone": greta_training._get_next_milestone(progress)
        }

    except Exception as e:
        logger.error(f"Failed to get progress for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load progress: {str(e)}")

@router.post("/agent/create-practice")
async def create_practice_agent(request: Request, user_id: str = Body(...), personality_config: Dict[str, Any] = Body(...)):
    """
    🎭 Create a practice agent during personality customization lessons
    Allows students to test custom agent configurations in real-time
    """
    try:
        with performance_monitor("practice_agent_creation"):
            # Create temporary practice agent
            practice_result = await greta_training.create_practice_agent(user_id, personality_config)

            return practice_result

    except Exception as e:
        logger.error(f"Failed to create practice agent for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create practice agent: {str(e)}")

@router.get("/system/status")
async def get_training_system_status():
    """📈 Get training system health and usage statistics"""

    try:
        # Count active users in progress system
        active_users = len(greta_training.user_progress)

        # Count lessons delivered today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        today_lessons = await db.training_results.count_documents({
            "timestamp": {"$gte": today}
        })

        # Calculate completion rate
        pipeline = [
            {"$group": {
                "_id": "$user_id",
                "completed": {"$sum": 1},
                "avg_score": {"$avg": "$score"}
            }}
        ]
        user_stats = await db.training_results.aggregate(pipeline).to_list(length=None)
        completion_rate = sum(1 for u in user_stats if u["avg_score"] >= 70) / len(user_stats) * 100 if user_stats else 0

        # Get popular lessons
        popular_pipeline = [
            {"$group": {"_id": "$lesson_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        popular_results = await db.training_results.aggregate(popular_pipeline).to_list(length=5)
        popular_lessons = [r["_id"] for r in popular_results]

        return TrainingStatusResponse(
            system_health="operational",
            total_users=active_users,
            active_lessons=today_lessons,
            completion_rate=round(completion_rate, 1),
            popular_lessons=popular_lessons
        )

    except Exception as e:
        logger.error(f"Failed to get training system status: {e}")
        return TrainingStatusResponse(
            system_health="degraded",
            total_users=0,
            active_lessons=0,
            completion_rate=0.0,
            popular_lessons=[]
        )

@router.get("/demo/lesson/{lesson_id}")
async def get_lesson_demo(lesson_id: str):
    """
    🎮 Get demo version of a lesson for browsing
    Preview content without starting tracked progress
    """
    try:
        if lesson_id not in greta_training.lessons:
            raise HTTPException(status_code=404, detail="Lesson not found")

        lesson = greta_training.lessons[lesson_id]

        # Return demo content (first few exercises only)
        demo_content = {
            "lesson_id": lesson_id,
            "title": lesson.title,
            "description": lesson.description,
            "level": lesson.required_level.value,
            "duration": lesson.estimated_duration,
            "type": lesson.lesson_type.value,
            "sample_content": lesson.content.get("analogy", "Sample content..."),
            "sample_exercise": lesson.exercises[0] if lesson.exercises else None,
            "preview_only": True,
            "points_value": lesson.points_value
        }

        return demo_content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get lesson demo for {lesson_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load lesson demo")

# ===== HELPER METHODS =====

async def initialize_training_routes():
    """Initialize training system on startup"""
    success = await greta_training.initialize_training_system()
    if success:
        logger.info("🎓 Greta Interactive Training API ready!")
        # Pre-warm popular lessons
        await training_api.get_curriculum(None, "demo_user")
    else:
        logger.error("❌ Training system initialization failed!")

    return success

# Import logger when module loads
import logging
logger = logging.getLogger(__name__)
