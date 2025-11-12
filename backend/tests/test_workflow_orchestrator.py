"""
Tests for Workflow Orchestrator

Tests intelligent workflow orchestration including:
- Task decomposition
- Tool selection
- Pattern matching
- RAG context retrieval
- Workflow execution
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from services.workflow_orchestrator import WorkflowOrchestrator, WorkflowStatus, SubtaskStatus


@pytest.fixture
def mock_tool_registry():
    """Mock Tool Registry"""
    from services.tool_registry import ToolCategory

    registry = Mock()

    # Create proper mock tools with string attributes
    web_scraper = Mock()
    web_scraper.tool_id = "osint_web_scraper"
    web_scraper.name = "Web Scraper"
    web_scraper.description = "Scrape web pages"
    web_scraper.category = ToolCategory.OSINT
    web_scraper.tags = ["web", "scraping", "research"]

    json_parser = Mock()
    json_parser.tool_id = "data_json_parser"
    json_parser.name = "JSON Parser"
    json_parser.description = "Parse JSON data"
    json_parser.category = ToolCategory.DATA_PROCESSING
    json_parser.tags = ["json", "parsing", "data"]

    text_embeddings = Mock()
    text_embeddings.tool_id = "ai_text_embeddings"
    text_embeddings.name = "Text Embeddings"
    text_embeddings.description = "Generate text embeddings"
    text_embeddings.category = ToolCategory.AI_ML
    text_embeddings.tags = ["ai", "embeddings", "text"]

    registry.tools = {
        "osint_web_scraper": web_scraper,
        "data_json_parser": json_parser,
        "ai_text_embeddings": text_embeddings
    }
    return registry


@pytest.fixture
def mock_reasoning_bank():
    """Mock ReasoningBank Service"""
    bank = AsyncMock()

    # Mock search_patterns method
    async def mock_search(query):
        from models.reasoning_pattern import PatternSearchResult, ReasoningPattern, SolutionStep, SuccessMetrics

        pattern = ReasoningPattern(
            pattern_id="test_pattern_1",
            version=1,
            pattern_type="workflow",
            title="Research and Analysis Pattern",
            problem_description="Research a topic and analyze findings",
            solution_steps=[
                SolutionStep(
                    step_number=1,
                    description="Gather information from sources"
                ),
                SolutionStep(
                    step_number=2,
                    description="Analyze and synthesize data"
                )
            ],
            solution_summary="Research and analyze data systematically",
            tags=["research", "analysis"],
            created_by="system",
            success_metrics=SuccessMetrics(
                success_rate=0.85,
                usage_count=10
            )
        )

        # Return a list of PatternSearchResult objects
        return [
            PatternSearchResult(
                pattern=pattern,
                relevance_score=0.85,
                match_type="semantic"
            )
        ]

    bank.search_patterns = mock_search
    bank.create_pattern = AsyncMock()

    return bank


@pytest.fixture
def mock_vector_search():
    """Mock Vector Search Service"""
    vs = AsyncMock()
    
    # Mock search method
    async def mock_search(namespace, query, limit):
        return [
            {
                "content": "Previous task: Research competitor pricing",
                "metadata": {"workflow_id": "prev_1", "status": "completed"}
            },
            {
                "content": "Previous task: Analyze market trends",
                "metadata": {"workflow_id": "prev_2", "status": "completed"}
            }
        ]
    
    vs.search = mock_search
    vs.add_memory = AsyncMock()
    
    return vs


@pytest.fixture
def orchestrator(mock_tool_registry, mock_reasoning_bank, mock_vector_search):
    """Create Workflow Orchestrator with mocked dependencies"""
    return WorkflowOrchestrator(
        tool_registry=mock_tool_registry,
        reasoning_bank=mock_reasoning_bank,
        vector_search=mock_vector_search,
        skills_service=None,
        db=None
    )


# ==================== TESTS ====================

@pytest.mark.asyncio
async def test_workflow_orchestrator_initialization(orchestrator):
    """Test 1: Workflow Orchestrator initializes correctly"""
    assert orchestrator is not None
    assert orchestrator.tool_registry is not None
    assert orchestrator.reasoning_bank is not None
    assert orchestrator.vector_search is not None
    assert len(orchestrator.active_workflows) == 0
    assert len(orchestrator.workflow_history) == 0
    print("✅ TEST 1: Workflow Orchestrator initialization")


@pytest.mark.asyncio
async def test_rag_context_retrieval(orchestrator):
    """Test 2: RAG context retrieval from Vector Search"""
    task = "Research competitor pricing strategies"
    
    rag_context = await orchestrator._retrieve_rag_context(task)
    
    assert rag_context is not None
    assert "memories" in rag_context
    assert len(rag_context["memories"]) > 0
    assert rag_context["count"] == 2
    print("✅ TEST 2: RAG context retrieval")


@pytest.mark.asyncio
async def test_pattern_matching(orchestrator):
    """Test 3: Pattern matching from ReasoningBank"""
    task = "Research market trends and create analysis"
    rag_context = {"memories": []}

    patterns = await orchestrator._find_matching_patterns(task, rag_context)

    assert patterns is not None
    assert len(patterns) > 0
    assert patterns[0]["pattern_id"] == "test_pattern_1"
    assert patterns[0]["success_rate"] == 0.85
    assert len(patterns[0]["solution_steps"]) >= 2
    print("✅ TEST 3: Pattern matching")


@pytest.mark.asyncio
async def test_task_decomposition_with_patterns(orchestrator):
    """Test 4: Task decomposition using patterns"""
    task = "Research and analyze competitor data"

    # Get patterns first
    rag_context = {"memories": []}
    patterns = await orchestrator._find_matching_patterns(task, rag_context)

    # Decompose task
    subtasks = await orchestrator._decompose_task(task, patterns, rag_context)

    assert subtasks is not None
    assert len(subtasks) >= 2  # From pattern (may include heuristic additions)
    assert subtasks[0]["order"] == 1
    assert subtasks[0]["status"] == SubtaskStatus.PENDING
    assert "Gather information" in subtasks[0]["description"] or "gather" in subtasks[0]["description"].lower()
    print("✅ TEST 4: Task decomposition with patterns")


@pytest.mark.asyncio
async def test_task_decomposition_without_patterns(orchestrator):
    """Test 5: Task decomposition without patterns (heuristic)"""
    task = "Research competitor pricing and create a report"
    
    # No patterns
    patterns = []
    rag_context = {"memories": []}
    
    # Decompose task
    subtasks = await orchestrator._decompose_task(task, patterns, rag_context)
    
    assert subtasks is not None
    assert len(subtasks) >= 2  # Heuristic decomposition
    assert any("research" in s["description"].lower() or "gather" in s["description"].lower() for s in subtasks)
    print("✅ TEST 5: Task decomposition without patterns (heuristic)")


@pytest.mark.asyncio
async def test_tool_selection(orchestrator):
    """Test 6: Tool selection for subtasks"""
    subtasks = [
        {
            "subtask_id": "sub1",
            "order": 1,
            "description": "Scrape web pages for competitor data",
            "status": SubtaskStatus.PENDING,
            "tools": []
        },
        {
            "subtask_id": "sub2",
            "order": 2,
            "description": "Parse JSON data from API",
            "status": SubtaskStatus.PENDING,
            "tools": []
        }
    ]
    
    await orchestrator._select_tools_for_subtasks(subtasks)
    
    # First subtask should have web scraper
    assert len(subtasks[0]["tools"]) > 0
    assert any(t["tool_id"] == "osint_web_scraper" for t in subtasks[0]["tools"])
    
    # Second subtask should have JSON parser
    assert len(subtasks[1]["tools"]) > 0
    assert any(t["tool_id"] == "data_json_parser" for t in subtasks[1]["tools"])
    
    print("✅ TEST 6: Tool selection for subtasks")


@pytest.mark.asyncio
async def test_workflow_execution(orchestrator):
    """Test 7: Complete workflow execution"""
    task = "Research competitor pricing strategies"
    
    result = await orchestrator.execute_workflow(
        task_description=task,
        context={"priority": "high"},
        user_id="test_user"
    )
    
    assert result is not None
    assert result["status"] == "completed"
    assert "workflow_id" in result
    assert result["total_subtasks"] > 0
    assert result["subtasks_completed"] == result["total_subtasks"]
    assert result["execution_time"] > 0
    assert len(result["patterns_used"]) > 0
    
    # Check workflow was stored in history
    assert len(orchestrator.workflow_history) == 1
    
    print("✅ TEST 7: Complete workflow execution")


@pytest.mark.asyncio
async def test_workflow_status_tracking(orchestrator):
    """Test 8: Workflow status tracking"""
    task = "Analyze market trends"
    
    # Execute workflow
    result = await orchestrator.execute_workflow(task_description=task)
    workflow_id = result["workflow_id"]
    
    # Get status
    status = orchestrator.get_workflow_status(workflow_id)
    
    assert status is not None
    assert status["workflow_id"] == workflow_id
    assert status["task_description"] == task
    assert status["status"] == WorkflowStatus.COMPLETED
    assert len(status["subtasks"]) > 0
    
    print("✅ TEST 8: Workflow status tracking")


@pytest.mark.asyncio
async def test_workflow_history(orchestrator):
    """Test 9: Workflow history tracking"""
    # Execute multiple workflows
    await orchestrator.execute_workflow("Task 1")
    await orchestrator.execute_workflow("Task 2")
    await orchestrator.execute_workflow("Task 3")
    
    # Get history
    history = orchestrator.get_workflow_history(limit=2)
    
    assert len(history) == 2
    assert history[0]["task_description"] == "Task 2"
    assert history[1]["task_description"] == "Task 3"
    
    print("✅ TEST 9: Workflow history tracking")


@pytest.mark.asyncio
async def test_pattern_storage_after_success(orchestrator, mock_reasoning_bank):
    """Test 10: Pattern storage after successful workflow"""
    task = "Create analysis report"
    
    # Execute workflow
    result = await orchestrator.execute_workflow(task_description=task)
    
    # Verify pattern was created
    assert mock_reasoning_bank.create_pattern.called
    call_args = mock_reasoning_bank.create_pattern.call_args[0][0]
    assert call_args.pattern_type == "workflow"
    assert task[:50] in call_args.title
    
    print("✅ TEST 10: Pattern storage after successful workflow")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

