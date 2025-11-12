"""
Unit Tests for Agent Collaboration Patterns

Tests all 3 collaboration patterns:
- Sequential Workflow (A → B → C)
- Parallel Execution (concurrent agents)
- Hierarchical Team (Master → Leads → Specialists)
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.agent_collaboration import (
    SequentialWorkflow,
    ParallelExecution,
    HierarchicalTeam,
    AgentCollaborationService,
    WorkflowType
)


@pytest.fixture
def mock_agent_builder():
    """Create a mock AgentBuilderService"""
    mock = AsyncMock()
    
    # Mock execute_agent to return success
    async def mock_execute(execution_create):
        return Mock(
            id="exec_123",
            status="completed",
            result={"output": f"Result from {execution_create.agent_id}"},
            execution_time=1.5
        )
    
    mock.execute_agent = mock_execute
    return mock


@pytest.fixture
async def collaboration_service(mock_agent_builder):
    """Create AgentCollaborationService for testing"""
    mock_db = Mock()
    mock_db.workflows = Mock()
    mock_db.workflow_executions = Mock()
    
    service = AgentCollaborationService(mock_db, mock_agent_builder)
    return service


class TestSequentialWorkflow:
    """Test Sequential Workflow pattern"""
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self, mock_agent_builder):
        """Test that agents execute in sequence"""
        workflow = SequentialWorkflow(mock_agent_builder)
        
        agents = ["agent1", "agent2", "agent3"]
        task = {"description": "Test task"}
        
        result = await workflow.execute(agents, task)
        
        assert result["success"] is True
        assert len(result["steps"]) == 3
        assert result["steps"][0]["agent_id"] == "agent1"
        assert result["steps"][1]["agent_id"] == "agent2"
        assert result["steps"][2]["agent_id"] == "agent3"
    
    @pytest.mark.asyncio
    async def test_sequential_data_flow(self, mock_agent_builder):
        """Test that output flows from one agent to the next"""
        workflow = SequentialWorkflow(mock_agent_builder)
        
        agents = ["agent1", "agent2"]
        task = {"description": "Test task", "input": "initial data"}
        
        result = await workflow.execute(agents, task)
        
        # Each step should have the previous step's output as input
        assert result["steps"][0]["input"]["input"] == "initial data"
        assert "output" in result["steps"][1]["input"]
    
    @pytest.mark.asyncio
    async def test_sequential_error_handling(self, mock_agent_builder):
        """Test error handling in sequential workflow"""
        # Mock one agent to fail
        async def mock_execute_with_failure(execution_create):
            if execution_create.agent_id == "agent2":
                raise Exception("Agent 2 failed")
            return Mock(
                id="exec_123",
                status="completed",
                result={"output": f"Result from {execution_create.agent_id}"}
            )
        
        mock_agent_builder.execute_agent = mock_execute_with_failure
        
        workflow = SequentialWorkflow(mock_agent_builder)
        agents = ["agent1", "agent2", "agent3"]
        task = {"description": "Test task"}
        
        result = await workflow.execute(agents, task)
        
        assert result["success"] is False
        assert "error" in result
        # Should have executed agent1 but failed at agent2
        assert len(result["steps"]) == 2


class TestParallelExecution:
    """Test Parallel Execution pattern"""
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_agent_builder):
        """Test that agents execute concurrently"""
        workflow = ParallelExecution(mock_agent_builder)
        
        agents = ["agent1", "agent2", "agent3"]
        task = {"description": "Test task"}
        
        start_time = asyncio.get_event_loop().time()
        result = await workflow.execute(agents, task)
        end_time = asyncio.get_event_loop().time()
        
        assert result["success"] is True
        assert len(result["results"]) == 3
        
        # Execution time should be close to single agent time (parallel)
        # Not 3x single agent time (sequential)
        execution_time = end_time - start_time
        assert execution_time < 5  # Should be fast for mocked agents
    
    @pytest.mark.asyncio
    async def test_parallel_result_aggregation(self, mock_agent_builder):
        """Test that results from all agents are aggregated"""
        workflow = ParallelExecution(mock_agent_builder)
        
        agents = ["agent1", "agent2", "agent3"]
        task = {"description": "Test task"}
        
        result = await workflow.execute(agents, task)
        
        assert "aggregated_output" in result
        assert len(result["results"]) == 3
        
        # Each agent should have its own result
        agent_ids = [r["agent_id"] for r in result["results"]]
        assert "agent1" in agent_ids
        assert "agent2" in agent_ids
        assert "agent3" in agent_ids
    
    @pytest.mark.asyncio
    async def test_parallel_partial_failure(self, mock_agent_builder):
        """Test handling of partial failures in parallel execution"""
        # Mock one agent to fail
        async def mock_execute_with_failure(execution_create):
            if execution_create.agent_id == "agent2":
                raise Exception("Agent 2 failed")
            return Mock(
                id="exec_123",
                status="completed",
                result={"output": f"Result from {execution_create.agent_id}"}
            )
        
        mock_agent_builder.execute_agent = mock_execute_with_failure
        
        workflow = ParallelExecution(mock_agent_builder)
        agents = ["agent1", "agent2", "agent3"]
        task = {"description": "Test task"}
        
        result = await workflow.execute(agents, task)
        
        # Should have results from agent1 and agent3
        successful_results = [r for r in result["results"] if r["status"] == "completed"]
        failed_results = [r for r in result["results"] if r["status"] == "failed"]
        
        assert len(successful_results) == 2
        assert len(failed_results) == 1


class TestHierarchicalTeam:
    """Test Hierarchical Team pattern"""
    
    @pytest.mark.asyncio
    async def test_hierarchical_execution(self, mock_agent_builder):
        """Test hierarchical team execution"""
        workflow = HierarchicalTeam(mock_agent_builder)
        
        master_agent = "master"
        team_leads = ["lead1", "lead2"]
        specialists = {
            "lead1": ["spec1", "spec2"],
            "lead2": ["spec3", "spec4"]
        }
        task = {"description": "Complex task"}
        
        result = await workflow.execute(master_agent, team_leads, specialists, task)
        
        assert result["success"] is True
        assert "master_decomposition" in result
        assert "team_results" in result
        assert "final_aggregation" in result
    
    @pytest.mark.asyncio
    async def test_hierarchical_task_decomposition(self, mock_agent_builder):
        """Test that master agent decomposes tasks"""
        # Mock master agent to return subtasks
        async def mock_execute_master(execution_create):
            if execution_create.agent_id == "master":
                return Mock(
                    id="exec_master",
                    status="completed",
                    result={
                        "output": "Decomposed tasks",
                        "subtasks": [
                            {"team": "lead1", "task": "Subtask 1"},
                            {"team": "lead2", "task": "Subtask 2"}
                        ]
                    }
                )
            return Mock(
                id="exec_123",
                status="completed",
                result={"output": f"Result from {execution_create.agent_id}"}
            )
        
        mock_agent_builder.execute_agent = mock_execute_master
        
        workflow = HierarchicalTeam(mock_agent_builder)
        master_agent = "master"
        team_leads = ["lead1", "lead2"]
        specialists = {"lead1": ["spec1"], "lead2": ["spec2"]}
        task = {"description": "Complex task"}
        
        result = await workflow.execute(master_agent, team_leads, specialists, task)
        
        assert "subtasks" in result["master_decomposition"]["result"]
    
    @pytest.mark.asyncio
    async def test_hierarchical_team_coordination(self, mock_agent_builder):
        """Test that team leads coordinate specialists"""
        workflow = HierarchicalTeam(mock_agent_builder)
        
        master_agent = "master"
        team_leads = ["lead1"]
        specialists = {"lead1": ["spec1", "spec2", "spec3"]}
        task = {"description": "Complex task"}
        
        result = await workflow.execute(master_agent, team_leads, specialists, task)
        
        # Should have results from all specialists
        team_results = result["team_results"]
        assert len(team_results) > 0
        
        # Each team should have specialist results
        for team_result in team_results:
            assert "specialist_results" in team_result


class TestAgentCollaborationService:
    """Test AgentCollaborationService"""
    
    @pytest.mark.asyncio
    async def test_create_sequential_workflow(self, collaboration_service):
        """Test creating a sequential workflow"""
        workflow = await collaboration_service.create_workflow(
            name="Test Sequential",
            description="Test workflow",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=["agent1", "agent2", "agent3"]
        )
        
        assert workflow.name == "Test Sequential"
        assert workflow.workflow_type == WorkflowType.SEQUENTIAL
        assert len(workflow.agents) == 3
    
    @pytest.mark.asyncio
    async def test_create_parallel_workflow(self, collaboration_service):
        """Test creating a parallel workflow"""
        workflow = await collaboration_service.create_workflow(
            name="Test Parallel",
            description="Test workflow",
            workflow_type=WorkflowType.PARALLEL,
            agents=["agent1", "agent2", "agent3"]
        )
        
        assert workflow.workflow_type == WorkflowType.PARALLEL
    
    @pytest.mark.asyncio
    async def test_create_hierarchical_workflow(self, collaboration_service):
        """Test creating a hierarchical workflow"""
        configuration = {
            "master_agent": "master",
            "team_leads": ["lead1", "lead2"],
            "specialists": {
                "lead1": ["spec1", "spec2"],
                "lead2": ["spec3", "spec4"]
            }
        }
        
        workflow = await collaboration_service.create_workflow(
            name="Test Hierarchical",
            description="Test workflow",
            workflow_type=WorkflowType.HIERARCHICAL,
            agents=["master", "lead1", "lead2", "spec1", "spec2", "spec3", "spec4"],
            configuration=configuration
        )
        
        assert workflow.workflow_type == WorkflowType.HIERARCHICAL
        assert workflow.configuration["master_agent"] == "master"
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, collaboration_service):
        """Test executing a workflow"""
        # Create workflow
        workflow = await collaboration_service.create_workflow(
            name="Test Workflow",
            description="Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=["agent1", "agent2"]
        )
        
        # Execute workflow
        execution = await collaboration_service.execute_workflow(
            workflow.id,
            input_data={"task": "Test task"}
        )
        
        assert execution.workflow_id == workflow.id
        assert execution.status in ["running", "completed"]
    
    @pytest.mark.asyncio
    async def test_get_workflow_metrics(self, collaboration_service):
        """Test getting workflow metrics"""
        metrics = await collaboration_service.get_workflow_metrics()
        
        assert "total_workflows" in metrics
        assert "total_executions" in metrics
        assert "by_type" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

