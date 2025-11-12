"""
Unit Tests for ReasoningBank Service

Tests the ClaudeFlow-inspired pattern storage and retrieval system.
"""

import pytest
import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient

from models.reasoning_pattern import (
    ReasoningPattern,
    PatternCreateRequest,
    PatternSearchQuery,
    PatternUsageRecord,
    SolutionStep,
    PatternContext
)
from services.reasoning_bank_service import ReasoningBankService


# Test database configuration
TEST_DB_URL = "mongodb://localhost:27017"
TEST_DB_NAME = "greta_pai_test"


@pytest.fixture
async def db():
    """Create test database connection"""
    client = AsyncIOMotorClient(TEST_DB_URL)
    database = client[TEST_DB_NAME]
    
    # Clean up test collections
    await database.reasoning_patterns.delete_many({})
    await database.pattern_usage.delete_many({})
    
    yield database
    
    # Cleanup after tests
    await database.reasoning_patterns.delete_many({})
    await database.pattern_usage.delete_many({})
    client.close()


@pytest.fixture
async def reasoning_bank(db):
    """Create ReasoningBank service instance"""
    return ReasoningBankService(db=db, vector_search_service=None)


@pytest.fixture
def sample_pattern_request():
    """Create a sample pattern creation request"""
    return PatternCreateRequest(
        pattern_type="development",
        title="Implement REST API endpoint",
        problem_description="Need to create a new REST API endpoint for user management",
        solution_steps=[
            SolutionStep(
                step_number=1,
                description="Define Pydantic models for request/response",
                reasoning="Type safety and validation",
                tools_used=["pydantic"],
                success=True
            ),
            SolutionStep(
                step_number=2,
                description="Create FastAPI router with endpoint",
                reasoning="RESTful API design",
                tools_used=["fastapi"],
                success=True
            ),
            SolutionStep(
                step_number=3,
                description="Add database operations",
                reasoning="Data persistence",
                tools_used=["mongodb", "motor"],
                success=True
            ),
            SolutionStep(
                step_number=4,
                description="Write unit tests",
                reasoning="Quality assurance",
                tools_used=["pytest"],
                success=True
            )
        ],
        solution_summary="Created REST API endpoint with proper validation, database integration, and tests",
        tags=["api", "rest", "fastapi", "mongodb"],
        key_insights=[
            "Always define Pydantic models first for type safety",
            "Include comprehensive error handling",
            "Write tests before deploying"
        ],
        created_by="test_agent"
    )


@pytest.mark.asyncio
async def test_create_pattern(reasoning_bank, sample_pattern_request):
    """Test creating a new reasoning pattern"""
    pattern = await reasoning_bank.create_pattern(sample_pattern_request)
    
    assert pattern is not None
    assert pattern.pattern_id is not None
    assert pattern.title == sample_pattern_request.title
    assert pattern.pattern_type == sample_pattern_request.pattern_type
    assert len(pattern.solution_steps) == 4
    assert len(pattern.tags) == 4
    assert len(pattern.key_insights) == 3
    assert pattern.version == 1
    assert pattern.success_metrics.usage_count == 0
    assert pattern.success_metrics.success_rate == 1.0


@pytest.mark.asyncio
async def test_get_pattern(reasoning_bank, sample_pattern_request):
    """Test retrieving a pattern by ID"""
    # Create pattern
    created_pattern = await reasoning_bank.create_pattern(sample_pattern_request)
    
    # Retrieve pattern
    retrieved_pattern = await reasoning_bank.get_pattern(created_pattern.pattern_id)
    
    assert retrieved_pattern is not None
    assert retrieved_pattern.pattern_id == created_pattern.pattern_id
    assert retrieved_pattern.title == created_pattern.title


@pytest.mark.asyncio
async def test_search_patterns_keyword(reasoning_bank, sample_pattern_request):
    """Test keyword-based pattern search"""
    # Create pattern
    await reasoning_bank.create_pattern(sample_pattern_request)
    
    # Search for patterns
    query = PatternSearchQuery(
        query="REST API",
        limit=10
    )
    
    results = await reasoning_bank.search_patterns(query)
    
    assert len(results) > 0
    assert results[0].pattern.title == sample_pattern_request.title
    assert results[0].relevance_score > 0


@pytest.mark.asyncio
async def test_record_usage(reasoning_bank, sample_pattern_request):
    """Test recording pattern usage"""
    # Create pattern
    pattern = await reasoning_bank.create_pattern(sample_pattern_request)
    
    # Record successful usage
    usage = PatternUsageRecord(
        pattern_id=pattern.pattern_id,
        used_by="test_agent",
        problem_context="Creating user management API",
        success=True,
        execution_time_ms=1500.0,
        feedback_score=4.5
    )
    
    success = await reasoning_bank.record_usage(usage)
    assert success is True
    
    # Verify metrics updated
    updated_pattern = await reasoning_bank.get_pattern(pattern.pattern_id)
    assert updated_pattern.success_metrics.usage_count == 1
    assert updated_pattern.success_metrics.success_rate == 1.0
    assert updated_pattern.success_metrics.execution_time_ms == 1500.0
    assert updated_pattern.success_metrics.feedback_score == 4.5


@pytest.mark.asyncio
async def test_record_multiple_usages(reasoning_bank, sample_pattern_request):
    """Test recording multiple pattern usages and metric averaging"""
    # Create pattern
    pattern = await reasoning_bank.create_pattern(sample_pattern_request)
    
    # Record first usage (success)
    usage1 = PatternUsageRecord(
        pattern_id=pattern.pattern_id,
        used_by="test_agent",
        problem_context="Test 1",
        success=True,
        execution_time_ms=1000.0,
        feedback_score=5.0
    )
    await reasoning_bank.record_usage(usage1)
    
    # Record second usage (success)
    usage2 = PatternUsageRecord(
        pattern_id=pattern.pattern_id,
        used_by="test_agent",
        problem_context="Test 2",
        success=True,
        execution_time_ms=1500.0,
        feedback_score=4.0
    )
    await reasoning_bank.record_usage(usage2)
    
    # Record third usage (failure)
    usage3 = PatternUsageRecord(
        pattern_id=pattern.pattern_id,
        used_by="test_agent",
        problem_context="Test 3",
        success=False,
        execution_time_ms=2000.0,
        feedback_score=2.0
    )
    await reasoning_bank.record_usage(usage3)
    
    # Verify metrics
    updated_pattern = await reasoning_bank.get_pattern(pattern.pattern_id)
    assert updated_pattern.success_metrics.usage_count == 3
    assert updated_pattern.success_metrics.success_rate == pytest.approx(2/3, rel=0.01)  # 2 successes out of 3
    assert updated_pattern.success_metrics.execution_time_ms == pytest.approx(1500.0, rel=0.01)  # Average
    assert updated_pattern.success_metrics.feedback_score == pytest.approx(3.67, rel=0.01)  # Average


@pytest.mark.asyncio
async def test_evolve_pattern(reasoning_bank, sample_pattern_request):
    """Test creating an evolved version of a pattern"""
    from models.reasoning_pattern import PatternEvolutionRequest
    
    # Create original pattern
    original = await reasoning_bank.create_pattern(sample_pattern_request)
    
    # Evolve the pattern
    evolution_request = PatternEvolutionRequest(
        parent_pattern_id=original.pattern_id,
        improvements="Added caching layer for better performance",
        new_insights=["Use Redis for caching frequently accessed data"],
        created_by="test_agent"
    )
    
    evolved = await reasoning_bank.evolve_pattern(evolution_request)
    
    assert evolved is not None
    assert evolved.version == 2
    assert evolved.parent_pattern_id == original.pattern_id
    assert "caching" in evolved.solution_summary.lower()
    assert len(evolved.key_insights) > len(original.key_insights)


@pytest.mark.asyncio
async def test_get_stats(reasoning_bank, sample_pattern_request):
    """Test getting pattern statistics"""
    # Create multiple patterns
    pattern1 = await reasoning_bank.create_pattern(sample_pattern_request)
    
    # Create second pattern
    request2 = sample_pattern_request.copy()
    request2.title = "Different pattern"
    request2.pattern_type = "analysis"
    pattern2 = await reasoning_bank.create_pattern(request2)
    
    # Record some usage
    usage = PatternUsageRecord(
        pattern_id=pattern1.pattern_id,
        used_by="test_agent",
        problem_context="Test",
        success=True
    )
    await reasoning_bank.record_usage(usage)
    
    # Get stats
    stats = await reasoning_bank.get_stats()
    
    assert stats.total_patterns == 2
    assert "development" in stats.patterns_by_type
    assert "analysis" in stats.patterns_by_type
    assert stats.total_usage_count == 1
    assert len(stats.most_used_patterns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

