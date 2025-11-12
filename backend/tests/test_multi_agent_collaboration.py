"""
Tests for Multi-Agent Collaboration Service

Comprehensive test suite for multi-agent team formation,
inter-agent communication, parallel execution, and consensus.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from services.multi_agent_collaboration import (
    MultiAgentCollaborationService,
    AgentRole,
    AgentTeamMember,
    AgentMessage,
    CollaborationTask,
    ConsensusVote
)


@pytest.fixture
def mock_db():
    """Mock database"""
    db = Mock()
    
    # Mock collections
    agents_collection = Mock()
    messages_collection = Mock()
    
    # Mock cursor for agents
    mock_cursor = Mock()
    mock_cursor.to_list = AsyncMock(return_value=[
        {
            "_id": "agent_1",
            "name": "Data Analyst",
            "role": "analyst",
            "capabilities": ["data_analysis", "statistics"],
            "status": "available"
        },
        {
            "_id": "agent_2",
            "name": "Report Generator",
            "role": "generator",
            "capabilities": ["report_generation", "visualization"],
            "status": "available"
        },
        {
            "_id": "agent_3",
            "name": "Research Agent",
            "role": "researcher",
            "capabilities": ["research", "data_collection"],
            "status": "available"
        }
    ])
    
    agents_collection.find = Mock(return_value=mock_cursor)
    messages_collection.insert_one = AsyncMock()
    
    # Mock get_collection
    def get_collection(name):
        if name == "agents":
            return agents_collection
        elif name == "agent_messages":
            return messages_collection
        return Mock()
    
    db.get_collection = get_collection
    
    return db


@pytest.fixture
def collaboration_service(mock_db):
    """Create collaboration service instance"""
    return MultiAgentCollaborationService(mock_db)


# Test 1: Service Initialization
@pytest.mark.asyncio
async def test_service_initialization(collaboration_service):
    """Test 1: Service initializes correctly"""
    assert collaboration_service is not None
    assert collaboration_service.teams == {}
    assert collaboration_service.message_queue == []
    assert collaboration_service.active_tasks == {}
    assert collaboration_service.consensus_votes == {}


# Test 2: Team Formation
@pytest.mark.asyncio
async def test_team_formation(collaboration_service):
    """Test 2: Form agent team based on capabilities"""
    result = await collaboration_service.form_team(
        task_description="Analyze data and create report",
        required_capabilities=["data_analysis", "report_generation"],
        max_team_size=3
    )
    
    assert result["success"] is True
    assert "team_id" in result
    assert result["team_size"] > 0
    assert len(result["members"]) > 0
    assert result["coverage"] > 0


# Test 3: Team Formation with Limited Capabilities
@pytest.mark.asyncio
async def test_team_formation_limited_capabilities(collaboration_service):
    """Test 3: Form team with limited matching capabilities"""
    result = await collaboration_service.form_team(
        task_description="Perform advanced AI research",
        required_capabilities=["ai_research", "machine_learning"],
        max_team_size=2
    )
    
    assert result["success"] is True
    # May have 0 coverage if no agents match


# Test 4: Send Message Between Agents
@pytest.mark.asyncio
async def test_send_message(collaboration_service):
    """Test 4: Send message between agents"""
    message = await collaboration_service.send_message(
        from_agent="agent_1",
        message_type="request",
        content={"action": "analyze_data", "data_id": "dataset_123"},
        to_agent="agent_2",
        requires_response=True
    )
    
    assert message is not None
    assert message.from_agent == "agent_1"
    assert message.to_agent == "agent_2"
    assert message.message_type == "request"
    assert message.requires_response is True
    assert len(collaboration_service.message_queue) > 0


# Test 5: Broadcast Message
@pytest.mark.asyncio
async def test_broadcast_message(collaboration_service):
    """Test 5: Broadcast message to all agents"""
    message = await collaboration_service.send_message(
        from_agent="agent_1",
        message_type="broadcast",
        content={"announcement": "Task completed"},
        to_agent=None,
        requires_response=False
    )
    
    assert message is not None
    assert message.to_agent is None
    assert message.message_type == "broadcast"


# Test 6: Parallel Task Execution
@pytest.mark.asyncio
async def test_parallel_execution(collaboration_service):
    """Test 6: Execute tasks in parallel across team"""
    # First form a team
    team_result = await collaboration_service.form_team(
        task_description="Process multiple datasets",
        required_capabilities=["data_analysis"],
        max_team_size=3
    )
    
    team_id = team_result["team_id"]
    
    # Execute parallel tasks
    tasks = [
        {"id": "task_1", "action": "analyze", "data": "dataset_1"},
        {"id": "task_2", "action": "analyze", "data": "dataset_2"},
        {"id": "task_3", "action": "analyze", "data": "dataset_3"}
    ]
    
    result = await collaboration_service.execute_parallel_tasks(
        team_id=team_id,
        tasks=tasks
    )
    
    assert result["success"] is True
    assert result["tasks_executed"] == len(tasks)
    assert len(result["results"]) == len(tasks)


# Test 7: Parallel Execution with Invalid Team
@pytest.mark.asyncio
async def test_parallel_execution_invalid_team(collaboration_service):
    """Test 7: Parallel execution fails with invalid team"""
    result = await collaboration_service.execute_parallel_tasks(
        team_id="invalid_team_id",
        tasks=[{"id": "task_1"}]
    )
    
    assert result["success"] is False
    assert "error" in result


# Test 8: Consensus Mechanism
@pytest.mark.asyncio
async def test_reach_consensus(collaboration_service):
    """Test 8: Reach consensus among team members"""
    # Form a team first
    team_result = await collaboration_service.form_team(
        task_description="Make decision on approach",
        required_capabilities=["decision_making"],
        max_team_size=3
    )
    
    team_id = team_result["team_id"]
    task_id = "task_consensus_123"
    
    # Reach consensus
    result = await collaboration_service.reach_consensus(
        task_id=task_id,
        team_id=team_id,
        decision_options=["option_a", "option_b", "option_c"],
        timeout_seconds=30
    )
    
    assert result["success"] is True
    assert "consensus_decision" in result
    assert "consensus_confidence" in result
    assert result["total_votes"] >= 0  # May be 0 if no matching agents
    assert 0 <= result["consensus_confidence"] <= 1


# Test 9: Consensus with Invalid Team
@pytest.mark.asyncio
async def test_consensus_invalid_team(collaboration_service):
    """Test 9: Consensus fails with invalid team"""
    result = await collaboration_service.reach_consensus(
        task_id="task_123",
        team_id="invalid_team_id",
        decision_options=["option_a", "option_b"],
        timeout_seconds=30
    )
    
    assert result["success"] is False
    assert "error" in result


# Test 10: Get Team Status
@pytest.mark.asyncio
async def test_get_team_status(collaboration_service):
    """Test 10: Get status of a team"""
    # Form a team first
    team_result = await collaboration_service.form_team(
        task_description="Test task",
        required_capabilities=["testing"],
        max_team_size=2
    )
    
    team_id = team_result["team_id"]
    
    # Get status
    status = await collaboration_service.get_team_status(team_id)
    
    assert status["success"] is True
    assert status["team_id"] == team_id
    assert "team_size" in status
    assert "members" in status
    assert "active_tasks" in status


# Test 11: Get Status of Invalid Team
@pytest.mark.asyncio
async def test_get_status_invalid_team(collaboration_service):
    """Test 11: Get status fails for invalid team"""
    status = await collaboration_service.get_team_status("invalid_team_id")
    
    assert status["success"] is False
    assert "error" in status


# Test 12: Multiple Teams
@pytest.mark.asyncio
async def test_multiple_teams(collaboration_service):
    """Test 12: Create and manage multiple teams"""
    # Form first team
    team1 = await collaboration_service.form_team(
        task_description="Task 1",
        required_capabilities=["capability_1"],
        max_team_size=2
    )
    
    # Form second team
    team2 = await collaboration_service.form_team(
        task_description="Task 2",
        required_capabilities=["capability_2"],
        max_team_size=2
    )
    
    assert team1["success"] is True
    assert team2["success"] is True
    assert team1["team_id"] != team2["team_id"]
    assert len(collaboration_service.teams) == 2


# Test 13: Message Queue Management
@pytest.mark.asyncio
async def test_message_queue(collaboration_service):
    """Test 13: Message queue accumulates messages"""
    initial_count = len(collaboration_service.message_queue)
    
    # Send multiple messages
    await collaboration_service.send_message(
        from_agent="agent_1",
        message_type="request",
        content={"test": 1}
    )
    
    await collaboration_service.send_message(
        from_agent="agent_2",
        message_type="response",
        content={"test": 2}
    )
    
    assert len(collaboration_service.message_queue) == initial_count + 2


# Test 14: Agent Role Priority
@pytest.mark.asyncio
async def test_agent_role_priority(collaboration_service):
    """Test 14: Agent roles have correct priority"""
    team_result = await collaboration_service.form_team(
        task_description="Priority test",
        required_capabilities=["data_analysis", "report_generation"],
        max_team_size=3
    )
    
    assert team_result["success"] is True
    members = team_result["members"]
    
    # Check that members have roles with priorities
    for member in members:
        assert "role" in member
        assert "priority" in member["role"]
        assert member["role"]["priority"] > 0

