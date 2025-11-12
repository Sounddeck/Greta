"""
Tests for Tool Recommendation Service

Tests cover:
1. Tool recommendation based on task description
2. Tool chain recommendations
3. Auto-selection logic
4. Performance tracking
5. Keyword similarity matching
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from services.tool_recommendation_service import (
    ToolRecommendationService,
    ToolPerformanceMetrics,
    ToolChainPattern,
    ToolRecommendation,
    ToolChainRecommendation
)
from services.tool_registry import ToolDefinition, ToolCategory, PermissionLevel


# ==================== FIXTURES ====================

@pytest.fixture
def mock_db():
    """Mock database"""
    db = Mock()
    db.get_collection = Mock(return_value=Mock())
    return db


@pytest.fixture
def mock_vector_search():
    """Mock vector search service"""
    return None  # Not using vector search in basic tests


@pytest.fixture
async def tool_recommendation_service(mock_db, mock_vector_search):
    """Create tool recommendation service instance"""
    service = ToolRecommendationService(mock_db, mock_vector_search)
    
    # Mock database collections
    service.performance_collection.find = AsyncMock(return_value=AsyncMock(to_list=AsyncMock(return_value=[])))
    service.chain_patterns_collection.find = AsyncMock(return_value=AsyncMock(to_list=AsyncMock(return_value=[])))
    service.performance_collection.update_one = AsyncMock()
    service.chain_patterns_collection.update_one = AsyncMock()
    
    await service.initialize()
    return service


@pytest.fixture
def sample_tools():
    """Sample tools for testing"""
    return [
        ToolDefinition(
            tool_id="web_search",
            name="Web Search",
            description="Search the web for information using Google",
            category=ToolCategory.OSINT,
            permission_level=PermissionLevel.PUBLIC,
            version="1.0.0",
            author="system",
            tags=["search", "web", "research", "google"]
        ),
        ToolDefinition(
            tool_id="data_analyzer",
            name="Data Analyzer",
            description="Analyze and process data using statistical methods",
            category=ToolCategory.DATA_PROCESSING,
            permission_level=PermissionLevel.PUBLIC,
            version="1.0.0",
            author="system",
            tags=["analyze", "data", "statistics", "processing"]
        ),
        ToolDefinition(
            tool_id="report_generator",
            name="Report Generator",
            description="Create professional reports and documents",
            category=ToolCategory.DATA_PROCESSING,
            permission_level=PermissionLevel.PUBLIC,
            version="1.0.0",
            author="system",
            tags=["create", "report", "document", "generate"]
        )
    ]


# ==================== TESTS ====================

@pytest.mark.asyncio
async def test_service_initialization(tool_recommendation_service):
    """Test 1: Service initializes correctly"""
    assert tool_recommendation_service is not None
    assert tool_recommendation_service.db is not None
    assert isinstance(tool_recommendation_service.performance_cache, dict)
    assert isinstance(tool_recommendation_service.chain_patterns_cache, list)
    print("✅ Test 1 passed: Service initialization")


@pytest.mark.asyncio
async def test_keyword_similarity_matching(tool_recommendation_service, sample_tools):
    """Test 2: Keyword similarity matching works correctly"""
    task_description = "search the web for competitor information"
    
    # Mock tool registry
    with patch('services.tool_recommendation_service.tool_registry') as mock_registry:
        mock_registry.get_all_tools.return_value = sample_tools
        
        recommendations = await tool_recommendation_service.recommend_tools(
            task_description=task_description,
            limit=3,
            min_relevance=0.0
        )
    
    assert len(recommendations) > 0
    # Web search should be most relevant
    assert recommendations[0].tool.tool_id == "web_search"
    assert recommendations[0].relevance_score > 0
    print(f"✅ Test 2 passed: Keyword similarity - top tool: {recommendations[0].tool.name} (score: {recommendations[0].relevance_score:.2f})")


@pytest.mark.asyncio
async def test_tool_recommendations_with_min_relevance(tool_recommendation_service, sample_tools):
    """Test 3: Tool recommendations respect minimum relevance threshold"""
    task_description = "analyze sales data"
    
    with patch('services.tool_recommendation_service.tool_registry') as mock_registry:
        mock_registry.get_all_tools.return_value = sample_tools
        
        # Get recommendations with high threshold
        recommendations = await tool_recommendation_service.recommend_tools(
            task_description=task_description,
            limit=10,
            min_relevance=0.5
        )
    
    # All recommendations should meet threshold
    for rec in recommendations:
        assert rec.relevance_score >= 0.5
    
    print(f"✅ Test 3 passed: Min relevance filter - {len(recommendations)} tools above 0.5 threshold")


@pytest.mark.asyncio
async def test_performance_tracking(tool_recommendation_service):
    """Test 4: Performance tracking records execution data"""
    tool_id = "test_tool"
    
    # Record successful execution
    await tool_recommendation_service.record_tool_execution(
        tool_id=tool_id,
        success=True,
        execution_time=1.5,
        user_rating=4.5
    )
    
    # Check metrics were updated
    assert tool_id in tool_recommendation_service.performance_cache
    metrics = tool_recommendation_service.performance_cache[tool_id]
    
    assert metrics.total_executions == 1
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 0
    assert metrics.success_rate == 1.0
    assert metrics.average_execution_time == 1.5
    assert 4.5 in metrics.user_ratings
    assert metrics.average_rating == 4.5
    
    print("✅ Test 4 passed: Performance tracking")


@pytest.mark.asyncio
async def test_performance_score_calculation(tool_recommendation_service, sample_tools):
    """Test 5: Performance scores affect recommendations"""
    # Add performance data for one tool
    tool_recommendation_service.performance_cache["web_search"] = ToolPerformanceMetrics(
        tool_id="web_search",
        total_executions=100,
        successful_executions=95,
        failed_executions=5,
        success_rate=0.95,
        average_execution_time=2.0,
        user_ratings=[4.5, 4.8, 4.7],
        average_rating=4.67
    )
    
    task_description = "search for information"
    
    with patch('services.tool_recommendation_service.tool_registry') as mock_registry:
        mock_registry.get_all_tools.return_value = sample_tools
        
        recommendations = await tool_recommendation_service.recommend_tools(
            task_description=task_description,
            limit=3
        )
    
    # Find web_search recommendation
    web_search_rec = next((r for r in recommendations if r.tool.tool_id == "web_search"), None)
    assert web_search_rec is not None
    assert web_search_rec.performance_score > 0.5  # Should have good performance score
    
    print(f"✅ Test 5 passed: Performance score calculation - score: {web_search_rec.performance_score:.2f}")


@pytest.mark.asyncio
async def test_auto_selection_high_confidence(tool_recommendation_service, sample_tools):
    """Test 6: Auto-selection works with high confidence"""
    task_description = "search the web for information"
    
    with patch('services.tool_recommendation_service.tool_registry') as mock_registry:
        mock_registry.get_all_tools.return_value = sample_tools
        
        selected_tools, confidence = await tool_recommendation_service.auto_select_tools(
            task_description=task_description,
            confidence_threshold=0.1  # Low threshold for testing
        )
    
    assert len(selected_tools) > 0
    assert confidence > 0
    print(f"✅ Test 6 passed: Auto-selection - selected {len(selected_tools)} tool(s) with {confidence:.1%} confidence")


@pytest.mark.asyncio
async def test_auto_selection_low_confidence(tool_recommendation_service, sample_tools):
    """Test 7: Auto-selection returns empty with low confidence"""
    task_description = "do something completely unrelated to available tools xyz123"
    
    with patch('services.tool_recommendation_service.tool_registry') as mock_registry:
        mock_registry.get_all_tools.return_value = sample_tools
        
        selected_tools, confidence = await tool_recommendation_service.auto_select_tools(
            task_description=task_description,
            confidence_threshold=0.9  # High threshold
        )
    
    # Should return empty list if confidence is low
    assert len(selected_tools) == 0 or confidence < 0.9
    print(f"✅ Test 7 passed: Auto-selection low confidence - {len(selected_tools)} tools selected")


@pytest.mark.asyncio
async def test_tool_chain_recommendations(tool_recommendation_service, sample_tools):
    """Test 8: Tool chain recommendations work"""
    # Add a chain pattern
    pattern = ToolChainPattern(
        tool_sequence=["web_search", "data_analyzer", "report_generator"],
        task_type="research",
        usage_count=10,
        success_rate=0.85,
        average_execution_time=15.0
    )
    tool_recommendation_service.chain_patterns_cache.append(pattern)
    
    task_description = "research competitor pricing"
    
    with patch('services.tool_recommendation_service.tool_registry') as mock_registry:
        mock_registry.get_tool = Mock(side_effect=lambda tid: next((t for t in sample_tools if t.tool_id == tid), None))
        
        chains = await tool_recommendation_service.recommend_tool_chains(
            task_description=task_description,
            limit=3
        )
    
    assert len(chains) > 0
    assert chains[0].tools is not None
    print(f"✅ Test 8 passed: Tool chain recommendations - {len(chains)} chain(s) recommended")


@pytest.mark.asyncio
async def test_task_type_identification(tool_recommendation_service):
    """Test 9: Task type identification works correctly"""
    test_cases = [
        ("search for information", "research"),
        ("analyze the data", "data_processing"),
        ("create a report", "creation"),
        ("validate the results", "validation"),
        ("do something", "general")
    ]
    
    for task_desc, expected_type in test_cases:
        task_type = tool_recommendation_service._identify_task_type(task_desc)
        assert task_type == expected_type
    
    print("✅ Test 9 passed: Task type identification")


@pytest.mark.asyncio
async def test_recommendation_reason_generation(tool_recommendation_service, sample_tools):
    """Test 10: Recommendation reasons are generated"""
    tool = sample_tools[0]
    
    reason = tool_recommendation_service._generate_recommendation_reason(
        tool=tool,
        relevance_score=0.8,
        performance_score=0.9,
        task_description="search for information"
    )
    
    assert isinstance(reason, str)
    assert len(reason) > 0
    print(f"✅ Test 10 passed: Recommendation reason - '{reason}'")


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

