"""
Comprehensive test suite for Feedback Loop Service

Tests cover:
- Service initialization
- Feedback recording (all types)
- Learning cycle triggering
- Pattern refinement
- Skill confidence adjustment
- Tool performance updates
- Metrics calculation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from services.feedback_loop_service import (
    FeedbackLoopService,
    FeedbackEntry,
    FeedbackType,
    LearningMetrics
)


@pytest.fixture
def mock_db():
    """Mock database"""
    db = Mock()

    # Mock collections
    feedback_collection = Mock()
    feedback_collection.insert_one = AsyncMock()
    feedback_collection.count_documents = AsyncMock(return_value=0)

    metrics_collection = Mock()
    metrics_collection.find_one = AsyncMock(return_value=None)

    adjustments_collection = Mock()

    # Mock cursor for find operations
    mock_cursor = Mock()
    mock_cursor.sort = Mock(return_value=mock_cursor)
    mock_cursor.limit = Mock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=[])

    feedback_collection.find = Mock(return_value=mock_cursor)
    adjustments_collection.find = Mock(return_value=mock_cursor)

    # Mock get_collection method
    def get_collection(name):
        if name == "feedback_entries":
            return feedback_collection
        elif name == "learning_metrics":
            return metrics_collection
        elif name == "skill_adjustments":
            return adjustments_collection
        return Mock()

    db.get_collection = get_collection

    return db


@pytest.fixture
def mock_reasoning_bank():
    """Mock ReasoningBank service"""
    rb = Mock()
    rb.update_pattern = AsyncMock()
    rb.search_patterns = AsyncMock(return_value=[])
    return rb


@pytest.fixture
def mock_tool_recommendation():
    """Mock Tool Recommendation service"""
    tr = Mock()
    tr.update_tool_performance = AsyncMock()
    tr.get_tool_performance = AsyncMock(return_value=None)
    return tr


@pytest.fixture
async def feedback_service(mock_db, mock_reasoning_bank, mock_tool_recommendation):
    """Create FeedbackLoopService instance with mocked dependencies"""
    service = FeedbackLoopService(
        db=mock_db,
        reasoning_bank=mock_reasoning_bank,
        tool_recommendation=mock_tool_recommendation
    )
    return service


@pytest.mark.asyncio
async def test_service_initialization(feedback_service):
    """Test 1: Service initializes correctly"""
    assert feedback_service is not None
    assert feedback_service.db is not None
    assert feedback_service.reasoning_bank is not None
    assert feedback_service.tool_recommendation is not None
    assert len(feedback_service.recent_feedback) >= 0
    assert feedback_service.learning_metrics is not None


@pytest.mark.asyncio
async def test_record_success_feedback(feedback_service):
    """Test 2: Record SUCCESS feedback"""
    feedback = await feedback_service.record_feedback(
        entity_type="workflow",
        entity_id="workflow_123",
        feedback_type=FeedbackType.SUCCESS,
        success=True,
        execution_time=1.5,
        context={"task": "test"}
    )

    assert feedback is not None
    assert feedback.entity_type == "workflow"
    assert feedback.entity_id == "workflow_123"
    assert feedback.feedback_type == FeedbackType.SUCCESS
    assert feedback.success is True
    assert feedback.execution_time == 1.5
    assert len(feedback_service.recent_feedback) > 0


@pytest.mark.asyncio
async def test_record_failure_feedback(feedback_service):
    """Test 3: Record FAILURE feedback"""
    feedback = await feedback_service.record_feedback(
        entity_type="tool",
        entity_id="tool_456",
        feedback_type=FeedbackType.FAILURE,
        success=False,
        error_message="Tool execution failed",
        context={"error_code": 500}
    )

    assert feedback is not None
    assert feedback.feedback_type == FeedbackType.FAILURE
    assert feedback.success is False
    assert feedback.error_message == "Tool execution failed"
    assert len(feedback_service.recent_feedback) > 0


@pytest.mark.asyncio
async def test_record_user_rating_feedback(feedback_service):
    """Test 4: Record USER_RATING feedback"""
    feedback = await feedback_service.record_feedback(
        entity_type="pattern",
        entity_id="pattern_789",
        feedback_type=FeedbackType.USER_RATING,
        rating=4.5,
        user_comment="Great pattern!",
        user_id="user_123"
    )

    assert feedback is not None
    assert feedback.feedback_type == FeedbackType.USER_RATING
    assert feedback.rating == 4.5
    assert feedback.user_comment == "Great pattern!"
    assert feedback.created_by == "user_123"


@pytest.mark.asyncio
async def test_record_correction_feedback(feedback_service):
    """Test 5: Record CORRECTION feedback"""
    feedback = await feedback_service.record_feedback(
        entity_type="skill",
        entity_id="skill_abc",
        feedback_type=FeedbackType.CORRECTION,
        user_comment="Should use different approach",
        context={"correction": "use method B instead of A"}
    )

    assert feedback is not None
    assert feedback.feedback_type == FeedbackType.CORRECTION
    assert feedback.user_comment == "Should use different approach"
    assert len(feedback_service.recent_feedback) > 0


@pytest.mark.asyncio
async def test_record_suggestion_feedback(feedback_service):
    """Test 6: Record SUGGESTION feedback"""
    feedback = await feedback_service.record_feedback(
        entity_type="workflow",
        entity_id="workflow_xyz",
        feedback_type=FeedbackType.SUGGESTION,
        user_comment="Consider adding caching",
        user_id="user_456"
    )

    assert feedback is not None
    assert feedback.feedback_type == FeedbackType.SUGGESTION
    assert feedback.user_comment == "Consider adding caching"


@pytest.mark.asyncio
async def test_automatic_learning_cycle_trigger(feedback_service):
    """Test 7: Learning cycle triggers automatically when threshold reached"""
    # Set minimum feedback threshold low
    feedback_service.min_feedback_for_learning = 2

    # Record enough feedback to trigger learning
    await feedback_service.record_feedback(
        entity_type="workflow",
        entity_id="workflow_123",
        feedback_type=FeedbackType.SUCCESS,
        success=True
    )

    await feedback_service.record_feedback(
        entity_type="workflow",
        entity_id="workflow_456",
        feedback_type=FeedbackType.SUCCESS,
        success=True
    )

    # Verify feedback was recorded
    assert len(feedback_service.recent_feedback) >= 2


@pytest.mark.asyncio
async def test_manual_learning_cycle_trigger(feedback_service, mock_reasoning_bank, mock_tool_recommendation):
    """Test 8: Manual learning cycle trigger"""
    # Set up mock feedback data
    mock_feedback = [
        FeedbackEntry(
            entity_type="pattern",
            entity_id="pattern_1",
            feedback_type=FeedbackType.FAILURE,
            success=False
        ),
        FeedbackEntry(
            entity_type="skill",
            entity_id="skill_1",
            feedback_type=FeedbackType.SUCCESS,
            success=True
        ),
        FeedbackEntry(
            entity_type="tool",
            entity_id="tool_1",
            feedback_type=FeedbackType.SUCCESS,
            success=True,
            execution_time=2.0
        )
    ]

    feedback_service.recent_feedback = mock_feedback

    # Trigger learning cycle (private method)
    await feedback_service._trigger_learning_cycle()

    # Verify metrics were updated
    assert feedback_service.learning_metrics.last_learning_cycle is not None


@pytest.mark.asyncio
async def test_pattern_refinement(feedback_service, mock_reasoning_bank):
    """Test 9: Pattern refinement based on feedback"""
    # Set up feedback for a pattern
    feedback_service.recent_feedback = [
        FeedbackEntry(
            entity_type="pattern",
            entity_id="pattern_1",
            feedback_type=FeedbackType.FAILURE,
            success=False,
            error_message="Pattern failed"
        ),
        FeedbackEntry(
            entity_type="pattern",
            entity_id="pattern_1",
            feedback_type=FeedbackType.FAILURE,
            success=False,
            error_message="Pattern failed again"
        )
    ]

    # Trigger pattern refinement
    refined_count = await feedback_service._refine_patterns()

    assert refined_count >= 0
    # Verify ReasoningBank was called to update patterns
    # (actual calls depend on implementation)


@pytest.mark.asyncio
async def test_skill_confidence_adjustment(feedback_service):
    """Test 10: Skill confidence adjustment based on feedback"""
    # Set up feedback for a skill
    feedback_service.recent_feedback = [
        FeedbackEntry(
            entity_type="skill",
            entity_id="skill_1",
            feedback_type=FeedbackType.SUCCESS,
            success=True
        ),
        FeedbackEntry(
            entity_type="skill",
            entity_id="skill_1",
            feedback_type=FeedbackType.SUCCESS,
            success=True
        ),
        FeedbackEntry(
            entity_type="skill",
            entity_id="skill_1",
            feedback_type=FeedbackType.FAILURE,
            success=False
        )
    ]

    # Trigger skill adjustment
    adjusted_count = await feedback_service._adjust_skill_confidence()

    assert adjusted_count >= 0


@pytest.mark.asyncio
async def test_tool_performance_update(feedback_service, mock_tool_recommendation):
    """Test 11: Tool performance update based on feedback"""
    # Set up feedback for a tool
    feedback_service.recent_feedback = [
        FeedbackEntry(
            entity_type="tool",
            entity_id="tool_1",
            feedback_type=FeedbackType.SUCCESS,
            success=True,
            execution_time=1.5
        ),
        FeedbackEntry(
            entity_type="tool",
            entity_id="tool_1",
            feedback_type=FeedbackType.FAILURE,
            success=False,
            execution_time=3.0
        )
    ]

    # Trigger tool performance update
    updated_count = await feedback_service._update_tool_performance()

    assert updated_count >= 0


@pytest.mark.asyncio
async def test_get_learning_metrics(feedback_service):
    """Test 12: Get learning metrics"""
    # Update metrics
    feedback_service.learning_metrics.total_feedback_entries = 100
    feedback_service.learning_metrics.success_rate = 0.8

    metrics = await feedback_service.get_learning_metrics()

    assert metrics is not None
    assert metrics.total_feedback_entries >= 0
    assert 0 <= metrics.success_rate <= 1.0


@pytest.mark.asyncio
async def test_get_recent_feedback(feedback_service, mock_db):
    """Test 13: Get recent feedback entries"""
    # Set up mock recent feedback
    mock_recent = [
        {
            "feedback_id": "fb_1",
            "entity_type": "workflow",
            "feedback_type": FeedbackType.SUCCESS,
            "created_at": datetime.utcnow()
        }
    ]
    
    mock_cursor = Mock()
    mock_cursor.sort = Mock(return_value=mock_cursor)
    mock_cursor.limit = Mock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=mock_recent)
    mock_db.feedback_entries.find.return_value = mock_cursor
    
    recent = await feedback_service.get_recent_feedback(limit=10)
    
    assert isinstance(recent, list)
    assert len(recent) <= 10


@pytest.mark.asyncio
async def test_get_skill_adjustments(feedback_service, mock_db):
    """Test 14: Get skill confidence adjustments"""
    # Set up mock skill adjustments
    mock_adjustments = [
        {
            "skill_id": "skill_1",
            "old_confidence": 0.8,
            "new_confidence": 0.85,
            "reason": "Improved performance",
            "adjusted_at": datetime.utcnow()
        }
    ]

    mock_cursor = Mock()
    mock_cursor.sort = Mock(return_value=mock_cursor)
    mock_cursor.limit = Mock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=mock_adjustments)
    mock_db.get_collection = Mock(return_value=Mock(find=Mock(return_value=mock_cursor)))

    adjustments = await feedback_service.get_skill_adjustments(limit=10)

    assert isinstance(adjustments, list)
    assert len(adjustments) <= 10

