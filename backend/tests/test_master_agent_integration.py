"""
Integration Tests for Master Agent & Agent Builder Integration

Tests the complete integration between:
- Master Agent
- Agent Builder
- Tool Registry
- MCP Hub
- Collaboration Patterns
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.enhanced_master_agent import EnhancedMasterAgent
from services.agent_builder_service import AgentBuilderService
from services.tool_registry import ToolRegistry
from services.mcp_hub_service import MCPHubService
from models.agent_builder_models import AgentType, AgentCapability


@pytest.fixture
def mock_db():
    """Create a mock database"""
    db = Mock()
    db.custom_agents = Mock()
    db.agent_executions = Mock()
    db.agent_deployments = Mock()
    db.tools = Mock()
    db.tool_permissions = Mock()
    db.mcp_servers = Mock()
    return db


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service"""
    mock = AsyncMock()
    
    async def mock_generate(prompt, **kwargs):
        return {
            "content": "Test response from LLM",
            "tool_calls": []
        }
    
    mock.generate = mock_generate
    return mock


@pytest.fixture
def mock_learning_service():
    """Create a mock learning service"""
    return Mock()


@pytest.fixture
async def master_agent(mock_db, mock_llm_service, mock_learning_service):
    """Create EnhancedMasterAgent for testing"""
    agent = EnhancedMasterAgent(mock_db, mock_llm_service, mock_learning_service)
    return agent


class TestMasterAgentAgentBuilderIntegration:
    """Test Master Agent ↔ Agent Builder integration"""
    
    @pytest.mark.asyncio
    async def test_create_custom_agent(self, master_agent):
        """Test Master Agent can create custom agents"""
        result = await master_agent.create_custom_agent(
            name="Research Agent",
            description="Specialized in web research",
            agent_type=AgentType.RESEARCH,
            capabilities=[AgentCapability.WEB_SEARCH, AgentCapability.DATA_ANALYSIS],
            tools=["osint_web_scraper", "data_json_parser"],
            system_prompt="You are a research specialist"
        )
        
        assert result["success"] is True
        assert "agent_id" in result
        assert result["agent"]["name"] == "Research Agent"
        assert result["agent"]["agent_type"] == AgentType.RESEARCH
    
    @pytest.mark.asyncio
    async def test_deploy_agent_to_autogen(self, master_agent):
        """Test Master Agent can deploy agents to AutoGen"""
        # First create an agent
        create_result = await master_agent.create_custom_agent(
            name="Test Agent",
            description="Test",
            agent_type=AgentType.ASSISTANT,
            capabilities=[AgentCapability.CONVERSATION]
        )
        
        agent_id = create_result["agent_id"]
        
        # Deploy to AutoGen
        deploy_result = await master_agent.deploy_agent_to_autogen(
            agent_id,
            runtime_config={"max_turns": 10}
        )
        
        assert deploy_result["success"] is True
        assert deploy_result["deployment"]["agent_id"] == agent_id
        assert deploy_result["deployment"]["status"] == "deployed"
    
    @pytest.mark.asyncio
    async def test_monitor_agent(self, master_agent):
        """Test Master Agent can monitor agent performance"""
        # Create and deploy an agent
        create_result = await master_agent.create_custom_agent(
            name="Monitored Agent",
            description="Test",
            agent_type=AgentType.ASSISTANT,
            capabilities=[]
        )
        
        agent_id = create_result["agent_id"]
        
        # Monitor the agent
        monitor_result = await master_agent.monitor_agent(agent_id)
        
        assert monitor_result["success"] is True
        assert "metrics" in monitor_result
        assert "health_score" in monitor_result
        assert 0 <= monitor_result["health_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_route_task_to_agent(self, master_agent):
        """Test Master Agent can intelligently route tasks"""
        # Create multiple specialized agents
        research_agent = await master_agent.create_custom_agent(
            name="Research Agent",
            description="Web research specialist",
            agent_type=AgentType.RESEARCH,
            capabilities=[AgentCapability.WEB_SEARCH]
        )
        
        code_agent = await master_agent.create_custom_agent(
            name="Code Agent",
            description="Code generation specialist",
            agent_type=AgentType.CODE,
            capabilities=[AgentCapability.CODE_GENERATION]
        )
        
        # Route a research task
        task = {
            "description": "Research AI trends in 2025",
            "type": "research",
            "priority": "high"
        }
        
        route_result = await master_agent.route_task_to_agent(task)
        
        assert route_result["success"] is True
        assert "selected_agent" in route_result
        assert "execution" in route_result


class TestMasterAgentToolRegistryIntegration:
    """Test Master Agent ↔ Tool Registry integration"""
    
    @pytest.mark.asyncio
    async def test_agent_with_tools(self, master_agent):
        """Test creating agent with specific tools"""
        result = await master_agent.create_custom_agent(
            name="OSINT Agent",
            description="OSINT specialist",
            agent_type=AgentType.RESEARCH,
            capabilities=[AgentCapability.WEB_SEARCH],
            tools=["osint_web_scraper", "osint_whois_lookup", "osint_dns_lookup"]
        )
        
        assert result["success"] is True
        assert len(result["agent"]["tools"]) == 3
        assert "osint_web_scraper" in result["agent"]["tools"]
    
    @pytest.mark.asyncio
    async def test_execute_agent_with_tools(self, master_agent, mock_db):
        """Test executing agent that uses tools"""
        # Create agent with tools
        create_result = await master_agent.create_custom_agent(
            name="Data Agent",
            description="Data processing",
            agent_type=AgentType.ASSISTANT,
            capabilities=[AgentCapability.DATA_ANALYSIS],
            tools=["data_json_parser", "data_csv_parser"]
        )
        
        agent_id = create_result["agent_id"]
        
        # Execute agent with a task that requires tools
        task = {
            "description": "Parse this JSON: {\"name\": \"test\"}",
            "input": {"json_data": '{"name": "test"}'}
        }
        
        execution_result = await master_agent._execute_agent_task(agent_id, task)
        
        assert execution_result["success"] is True


class TestMasterAgentMCPIntegration:
    """Test Master Agent ↔ MCP Hub integration"""
    
    @pytest.mark.asyncio
    async def test_agent_with_mcp_tools(self, master_agent):
        """Test creating agent with MCP tools"""
        result = await master_agent.create_custom_agent(
            name="Slack Agent",
            description="Slack integration",
            agent_type=AgentType.ASSISTANT,
            capabilities=[AgentCapability.COMMUNICATION],
            mcp_servers=["slack"]
        )
        
        assert result["success"] is True
        assert "mcp_servers" in result["agent"]
    
    @pytest.mark.asyncio
    async def test_execute_agent_with_mcp(self, master_agent):
        """Test executing agent that uses MCP tools"""
        # Create agent with MCP access
        create_result = await master_agent.create_custom_agent(
            name="Integration Agent",
            description="External integrations",
            agent_type=AgentType.ASSISTANT,
            capabilities=[AgentCapability.COMMUNICATION],
            mcp_servers=["slack", "stripe"]
        )
        
        agent_id = create_result["agent_id"]
        
        # Execute agent
        task = {
            "description": "Send a Slack message",
            "input": {"message": "Test message"}
        }
        
        execution_result = await master_agent._execute_agent_task(agent_id, task)
        
        assert execution_result["success"] is True


class TestMasterAgentCollaborationIntegration:
    """Test Master Agent ↔ Collaboration Patterns integration"""
    
    @pytest.mark.asyncio
    async def test_create_sequential_workflow(self, master_agent):
        """Test Master Agent can create sequential workflows"""
        # Create multiple agents
        agent1 = await master_agent.create_custom_agent(
            name="Agent 1",
            description="First agent",
            agent_type=AgentType.ASSISTANT,
            capabilities=[]
        )
        
        agent2 = await master_agent.create_custom_agent(
            name="Agent 2",
            description="Second agent",
            agent_type=AgentType.ASSISTANT,
            capabilities=[]
        )
        
        # Create workflow through Agent Builder
        workflow_result = await master_agent.agent_builder.collaboration_service.create_workflow(
            name="Test Sequential",
            description="Test workflow",
            workflow_type="sequential",
            agents=[agent1["agent_id"], agent2["agent_id"]]
        )
        
        assert workflow_result.name == "Test Sequential"
    
    @pytest.mark.asyncio
    async def test_execute_parallel_workflow(self, master_agent):
        """Test Master Agent can execute parallel workflows"""
        # Create multiple agents
        agents = []
        for i in range(3):
            result = await master_agent.create_custom_agent(
                name=f"Agent {i+1}",
                description=f"Agent {i+1}",
                agent_type=AgentType.ASSISTANT,
                capabilities=[]
            )
            agents.append(result["agent_id"])
        
        # Create parallel workflow
        workflow = await master_agent.agent_builder.collaboration_service.create_workflow(
            name="Test Parallel",
            description="Test",
            workflow_type="parallel",
            agents=agents
        )
        
        # Execute workflow
        execution = await master_agent.agent_builder.collaboration_service.execute_workflow(
            workflow.id,
            input_data={"task": "Test task"}
        )
        
        assert execution.workflow_id == workflow.id


class TestEndToEndIntegration:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_agent_lifecycle(self, master_agent):
        """Test complete agent lifecycle: create → deploy → execute → monitor"""
        # 1. Create agent
        create_result = await master_agent.create_custom_agent(
            name="Lifecycle Test Agent",
            description="Testing complete lifecycle",
            agent_type=AgentType.ASSISTANT,
            capabilities=[AgentCapability.CONVERSATION],
            tools=["data_json_parser"]
        )
        
        assert create_result["success"] is True
        agent_id = create_result["agent_id"]
        
        # 2. Deploy agent
        deploy_result = await master_agent.deploy_agent_to_autogen(agent_id)
        assert deploy_result["success"] is True
        
        # 3. Execute agent
        task = {"description": "Parse JSON", "input": {"data": '{"test": true}'}}
        execute_result = await master_agent._execute_agent_task(agent_id, task)
        assert execute_result["success"] is True
        
        # 4. Monitor agent
        monitor_result = await master_agent.monitor_agent(agent_id)
        assert monitor_result["success"] is True
        assert monitor_result["metrics"]["total_executions"] > 0
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, master_agent):
        """Test multi-agent collaboration workflow"""
        # Create specialized agents
        research_agent = await master_agent.create_custom_agent(
            name="Research Agent",
            description="Research specialist",
            agent_type=AgentType.RESEARCH,
            capabilities=[AgentCapability.WEB_SEARCH],
            tools=["osint_web_scraper"]
        )
        
        analysis_agent = await master_agent.create_custom_agent(
            name="Analysis Agent",
            description="Data analysis specialist",
            agent_type=AgentType.ASSISTANT,
            capabilities=[AgentCapability.DATA_ANALYSIS],
            tools=["data_json_parser", "data_transformer"]
        )
        
        report_agent = await master_agent.create_custom_agent(
            name="Report Agent",
            description="Report generation specialist",
            agent_type=AgentType.ASSISTANT,
            capabilities=[AgentCapability.CONTENT_GENERATION]
        )
        
        # Create sequential workflow: Research → Analysis → Report
        workflow = await master_agent.agent_builder.collaboration_service.create_workflow(
            name="Research Pipeline",
            description="Complete research pipeline",
            workflow_type="sequential",
            agents=[
                research_agent["agent_id"],
                analysis_agent["agent_id"],
                report_agent["agent_id"]
            ]
        )
        
        # Execute workflow
        execution = await master_agent.agent_builder.collaboration_service.execute_workflow(
            workflow.id,
            input_data={"topic": "AI trends 2025"}
        )
        
        assert execution.workflow_id == workflow.id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

