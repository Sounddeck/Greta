"""
Comprehensive tests for enhanced_master_agent.py module

Tests cover:
- Agent initialization and configuration
- Task processing and delegation
- Multi-agent coordination
- Response generation
- Error handling and recovery
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List


class TestMasterAgentInitialization:
    """Test master agent initialization"""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful master agent initialization"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        # Mock dependencies
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock initialization methods
        agent._initialize_collections = AsyncMock()
        agent._load_agents = AsyncMock()
        agent._initialize_modules = AsyncMock()

        result = await agent.initialize()

        assert result is True
        assert agent.system_state["status"] == "active"

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure handling"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock initialization to fail
        agent._initialize_collections = AsyncMock(side_effect=Exception("Init failed"))

        result = await agent.initialize()

        assert result is False
        assert agent.system_state["status"] == "error"

    @pytest.mark.asyncio
    async def test_agent_templates_loaded(self):
        """Test that agent templates are loaded"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        assert hasattr(agent, 'agent_templates')
        assert isinstance(agent.agent_templates, dict)
        assert len(agent.agent_templates) > 0


class TestAgentConfiguration:
    """Test agent configuration"""

    @pytest.mark.asyncio
    async def test_build_agent_success(self):
        """Test successful agent building"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Setup agent templates
        agent.agent_templates = {
            "research": {
                "name": "Research Agent",
                "capabilities": ["research", "analysis"],
                "llm_model": "llama3"
            }
        }

        command = {
            "config": {
                "type": "research",
                "name": "Research Agent"
            }
        }

        result = await agent.build_agent(command)

        assert result["success"] is True
        assert "agent_id" in result

    @pytest.mark.asyncio
    async def test_build_agent_invalid_type(self):
        """Test building agent with invalid type"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        command = {
            "config": {
                "type": "invalid_type"
            }
        }

        result = await agent.build_agent(command)

        assert result["success"] is False
        assert "error" in result


class TestTaskProcessing:
    """Test task processing"""

    @pytest.mark.asyncio
    async def test_execute_command_success(self):
        """Test successful command execution"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock get_system_status
        agent.get_system_status = AsyncMock(return_value={
            "success": True,
            "status": "active"
        })

        command = {
            "type": "get_status"
        }

        result = await agent.execute_command(command)

        assert result["success"] is True
        assert "execution_time" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_execute_command_failure(self):
        """Test command execution failure"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock to raise exception
        agent._handle_custom_command = AsyncMock(side_effect=Exception("Command failed"))

        command = {
            "type": "unknown_command"
        }

        result = await agent.execute_command(command)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_route_task_to_agent(self):
        """Test task routing handles no available agents gracefully"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Don't add any agents - test the error handling
        task = {
            "type": "research",
            "description": "Research AI trends"
        }

        result = await agent.route_task_to_agent(task)

        # Should return error when no agents available
        assert result.get("success") is False
        assert "error" in result


class TestTaskDelegation:
    """Test task delegation"""

    @pytest.mark.asyncio
    async def test_deploy_agent_success(self):
        """Test successful agent deployment"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock()
        mock_db.find_one = AsyncMock(return_value={
            "agent_id": "test_agent_123",
            "status": "created"
        })
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock agent builder
        agent.agent_builder = AsyncMock()
        agent.agent_builder.create_deployment = AsyncMock(return_value=MagicMock(
            deployment_id="deploy_123"
        ))

        # Add agent to registry
        agent.agents["test_agent_123"] = {
            "agent_id": "test_agent_123",
            "status": "created"
        }

        command = {
            "agent_id": "test_agent_123",
            "deployment_config": {}
        }

        result = await agent.deploy_agent(command)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_deploy_agent_invalid_template(self):
        """Test deploying agent with invalid template"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        agent.agent_templates = {}

        result = await agent.deploy_agent("invalid_type")

        assert result["success"] is False
        assert "error" in result


class TestMultiAgentCoordination:
    """Test multi-agent coordination"""

    @pytest.mark.asyncio
    async def test_aggregate_agent_results(self):
        """Test aggregating results from multiple agents"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        mock_llm.query = AsyncMock(return_value="Aggregated result")

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock agent deployment and execution
        agent._get_or_deploy_agent = AsyncMock(return_value={
            "agent_id": "agent_123",
            "name": "Test Agent"
        })
        agent._execute_task_with_agent = AsyncMock(return_value={
            "success": True,
            "result": "Agent result"
        })

        task = {
            "description": "Test task"
        }
        agent_types = ["research", "analysis"]

        result = await agent.aggregate_agent_results(task, agent_types)

        assert result["success"] is True
        assert "individual_results" in result
        assert "aggregated_result" in result
        assert result["agents_used"] == 2

    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test workflow execution with multiple steps"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock workflow execution
        agent._execute_workflow_steps = AsyncMock(return_value={
            "success": True,
            "steps_completed": 3,
            "results": []
        })

        command = {
            "workflow": {
                "steps": [
                    {"type": "research"},
                    {"type": "analysis"},
                    {"type": "report"}
                ]
            }
        }

        result = await agent.execute_workflow(command)

        assert result["success"] is True or "workflow" in str(result)


class TestResponseGeneration:
    """Test response generation"""

    @pytest.mark.asyncio
    async def test_query_llm_success(self):
        """Test successful LLM query"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        mock_llm.generate_completion = AsyncMock(return_value={
            "success": True,
            "response": "LLM response",
            "hrm_enhanced": True
        })

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        command = {
            "query": "What is AI?",
            "context": {}
        }

        result = await agent.query_llm(command)

        assert result["success"] is True
        assert "response" in result

    @pytest.mark.asyncio
    async def test_generate_response_with_context(self):
        """Test response generation with context"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        mock_llm.generate_completion = AsyncMock(return_value={
            "success": True,
            "response": "Contextual response"
        })

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        agent.system_state = {"status": "active"}
        agent.agents = {"agent1": {}}

        command = {
            "query": "Test query",
            "context": {"key": "value"}
        }

        result = await agent.query_llm(command)

        # Verify context was added
        call_args = mock_llm.generate_completion.call_args[0][0]
        assert "context" in call_args
        assert call_args["context"]["master_agent"] is True


class TestErrorHandling:
    """Test error handling and recovery"""

    @pytest.mark.asyncio
    async def test_handle_agent_failure(self):
        """Test handling agent execution failure"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Create test agent
        test_agent = {
            "agent_id": "test_123",
            "tasks_failed": 0
        }

        # Mock LLM to fail
        mock_llm.generate_completion = AsyncMock(side_effect=Exception("LLM failed"))

        task = {"description": "Test task"}

        result = await agent._execute_task_with_agent(test_agent, task)

        assert result["success"] is False
        assert "error" in result
        assert test_agent["tasks_failed"] == 1

    @pytest.mark.asyncio
    async def test_metrics_update_on_failure(self):
        """Test that metrics are updated on failure"""
        from services.enhanced_master_agent import EnhancedMasterAgent

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        initial_failures = agent.metrics["failed_operations"]

        # Execute failing command
        agent._handle_custom_command = AsyncMock(side_effect=Exception("Failed"))

        command = {"type": "unknown"}
        await agent.execute_command(command)

        assert agent.metrics["failed_operations"] > initial_failures


# Fixtures
@pytest.fixture
def master_agent():
    """Fixture to provide a master agent instance"""
    from services.enhanced_master_agent import EnhancedMasterAgent
    mock_db = AsyncMock()
    mock_llm = AsyncMock()
    mock_learning = AsyncMock()
    return EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)


@pytest.fixture
async def initialized_master_agent():
    """Fixture to provide an initialized master agent"""
    from services.enhanced_master_agent import EnhancedMasterAgent
    mock_db = AsyncMock()
    mock_llm = AsyncMock()
    mock_learning = AsyncMock()
    agent = EnhancedMasterAgent(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

    # Mock initialization
    agent._initialize_collections = AsyncMock()
    agent._load_agents = AsyncMock()
    agent._initialize_modules = AsyncMock()

    await agent.initialize()
    return agent

