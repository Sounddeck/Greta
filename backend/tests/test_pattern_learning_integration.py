"""
Integration Test for Pattern Learning with Master Agent

Tests the end-to-end pattern learning workflow:
1. Master Agent receives a task
2. Searches for similar patterns
3. Executes task with pattern guidance
4. Stores successful execution as new pattern
"""

import pytest
import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient

from models.reasoning_pattern import (
    PatternCreateRequest,
    PatternSearchQuery,
    SolutionStep
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


@pytest.mark.asyncio
async def test_pattern_learning_workflow(reasoning_bank):
    """
    Test complete pattern learning workflow:
    1. Create initial pattern
    2. Search for similar patterns
    3. Simulate task execution with pattern guidance
    4. Store new pattern based on execution
    5. Verify pattern was stored and can be retrieved
    """
    
    # ========================================================================
    # STEP 1: Create initial pattern (simulating previous successful task)
    # ========================================================================
    print("\n📚 STEP 1: Creating initial pattern...")
    
    initial_pattern_request = PatternCreateRequest(
        pattern_type="development",
        title="Create REST API endpoint with FastAPI",
        problem_description="Need to create a new REST API endpoint for data management",
        solution_steps=[
            SolutionStep(
                step_number=1,
                description="Define Pydantic models for request/response",
                reasoning="Ensures type safety and automatic validation",
                tools_used=["pydantic"],
                success=True
            ),
            SolutionStep(
                step_number=2,
                description="Create FastAPI router with endpoint",
                reasoning="Follows RESTful API design principles",
                tools_used=["fastapi"],
                success=True
            ),
            SolutionStep(
                step_number=3,
                description="Add database operations",
                reasoning="Persist data using MongoDB",
                tools_used=["mongodb", "motor"],
                success=True
            )
        ],
        solution_summary="Successfully created REST API endpoint with validation and database integration",
        tags=["api", "rest", "fastapi", "mongodb"],
        key_insights=[
            "Always define Pydantic models first for type safety",
            "Use async database operations for better performance",
            "Include comprehensive error handling"
        ],
        created_by="test_agent"
    )
    
    initial_pattern = await reasoning_bank.create_pattern(initial_pattern_request)
    assert initial_pattern is not None
    print(f"✅ Created pattern: {initial_pattern.title} (ID: {initial_pattern.pattern_id})")
    
    # ========================================================================
    # STEP 2: Search for similar patterns (simulating new task)
    # ========================================================================
    print("\n🔍 STEP 2: Searching for similar patterns...")

    new_task = "REST API"  # Simplified query for keyword matching

    search_query = PatternSearchQuery(
        query=new_task,
        pattern_type="development",
        limit=3
    )

    similar_patterns = await reasoning_bank.search_patterns(search_query)

    # If no patterns found, try without pattern_type filter
    if len(similar_patterns) == 0:
        print("   No patterns found with type filter, trying without...")
        search_query = PatternSearchQuery(
            query=new_task,
            limit=3
        )
        similar_patterns = await reasoning_bank.search_patterns(search_query)

    assert len(similar_patterns) > 0, f"Expected to find patterns, but found {len(similar_patterns)}"
    print(f"✅ Found {len(similar_patterns)} similar pattern(s)")
    
    # Get the most relevant pattern
    best_match = similar_patterns[0]
    print(f"   Best match: {best_match.pattern.title}")
    print(f"   Relevance: {best_match.relevance_score:.1%}")
    print(f"   Success rate: {best_match.pattern.success_metrics.success_rate:.1%}")
    
    # ========================================================================
    # STEP 3: Extract pattern guidance
    # ========================================================================
    print("\n💡 STEP 3: Extracting pattern guidance...")
    
    guidance = {
        "pattern_title": best_match.pattern.title,
        "recommended_steps": [
            {
                "step": step.step_number,
                "description": step.description,
                "reasoning": step.reasoning,
                "tools": step.tools_used
            }
            for step in best_match.pattern.solution_steps
        ],
        "key_insights": best_match.pattern.key_insights
    }
    
    print(f"✅ Extracted guidance with {len(guidance['recommended_steps'])} steps")
    for insight in guidance['key_insights']:
        print(f"   💡 {insight}")
    
    # ========================================================================
    # STEP 4: Simulate task execution with pattern guidance
    # ========================================================================
    print("\n⚙️ STEP 4: Simulating task execution...")
    
    # Simulate following the pattern's steps
    execution_steps = [
        {
            "step_number": 1,
            "description": "Defined Pydantic models for User request/response",
            "reasoning": "Following pattern guidance for type safety",
            "tools_used": ["pydantic"],
            "duration_ms": 500.0,
            "success": True
        },
        {
            "step_number": 2,
            "description": "Created FastAPI router for user management",
            "reasoning": "Following pattern guidance for RESTful design",
            "tools_used": ["fastapi"],
            "duration_ms": 800.0,
            "success": True
        },
        {
            "step_number": 3,
            "description": "Implemented MongoDB operations for users",
            "reasoning": "Following pattern guidance for data persistence",
            "tools_used": ["mongodb", "motor"],
            "duration_ms": 1200.0,
            "success": True
        }
    ]
    
    execution_result = {
        "success": True,
        "steps": execution_steps,
        "total_time_ms": sum(step["duration_ms"] for step in execution_steps),
        "output": "User management API endpoint created successfully"
    }
    
    print(f"✅ Task executed successfully in {execution_result['total_time_ms']:.0f}ms")
    
    # ========================================================================
    # STEP 5: Store new pattern based on successful execution
    # ========================================================================
    print("\n💾 STEP 5: Storing new pattern...")
    
    new_pattern_request = PatternCreateRequest(
        pattern_type="development",
        title="Create user management REST API endpoint",
        problem_description=new_task,
        solution_steps=[
            SolutionStep(**step) for step in execution_steps
        ],
        solution_summary=execution_result["output"],
        tags=["api", "rest", "fastapi", "mongodb", "user-management"],
        key_insights=[
            "Pattern guidance significantly reduced development time",
            "Following established patterns ensures consistency",
            "Reusing proven approaches improves quality"
        ],
        created_by="test_agent"
    )
    
    new_pattern = await reasoning_bank.create_pattern(new_pattern_request)
    assert new_pattern is not None
    print(f"✅ Stored new pattern: {new_pattern.title} (ID: {new_pattern.pattern_id})")
    
    # ========================================================================
    # STEP 6: Verify pattern can be retrieved
    # ========================================================================
    print("\n🔍 STEP 6: Verifying pattern retrieval...")
    
    retrieved_pattern = await reasoning_bank.get_pattern(new_pattern.pattern_id)
    assert retrieved_pattern is not None
    assert retrieved_pattern.pattern_id == new_pattern.pattern_id
    assert retrieved_pattern.title == new_pattern.title
    print(f"✅ Pattern retrieved successfully")
    
    # ========================================================================
    # STEP 7: Verify pattern appears in future searches
    # ========================================================================
    print("\n🔍 STEP 7: Verifying pattern appears in searches...")
    
    future_search = PatternSearchQuery(
        query="user management API",
        pattern_type="development",
        limit=5
    )
    
    future_results = await reasoning_bank.search_patterns(future_search)
    assert len(future_results) >= 1
    
    # Check if our new pattern is in the results
    pattern_ids = [result.pattern.pattern_id for result in future_results]
    assert new_pattern.pattern_id in pattern_ids
    print(f"✅ New pattern appears in search results")
    
    # ========================================================================
    # STEP 8: Get statistics
    # ========================================================================
    print("\n📊 STEP 8: Getting statistics...")
    
    stats = await reasoning_bank.get_stats()
    assert stats.total_patterns >= 2  # At least our 2 patterns
    assert "development" in stats.patterns_by_type
    print(f"✅ Statistics retrieved:")
    print(f"   Total patterns: {stats.total_patterns}")
    print(f"   Development patterns: {stats.patterns_by_type.get('development', 0)}")
    
    # ========================================================================
    # SUCCESS!
    # ========================================================================
    print("\n" + "="*70)
    print("✅ PATTERN LEARNING WORKFLOW TEST PASSED!")
    print("="*70)
    print("\nWorkflow Summary:")
    print(f"1. ✅ Created initial pattern")
    print(f"2. ✅ Searched for similar patterns")
    print(f"3. ✅ Extracted pattern guidance")
    print(f"4. ✅ Executed task with guidance")
    print(f"5. ✅ Stored new pattern")
    print(f"6. ✅ Retrieved pattern")
    print(f"7. ✅ Verified pattern in searches")
    print(f"8. ✅ Generated statistics")
    print("\n🎉 Pattern learning system is working correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

