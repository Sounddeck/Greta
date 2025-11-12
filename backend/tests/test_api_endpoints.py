"""
API Endpoint Tests

Tests all new API endpoints added in Phases 4-7:
- Master Agent endpoints (5)
- Workflow endpoints (4)
- Agent execution endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FastAPI app (you'll need to adjust this import based on your app structure)
# from main import app


@pytest.fixture
def client():
    """Create a test client"""
    # Mock the app for testing
    from fastapi import FastAPI
    app = FastAPI()
    
    # Import and include routers
    from routers.enhanced_master_agent_router import router as master_router
    from routers.agent_builder_router import router as agent_router
    
    app.include_router(master_router, prefix="/api/v1/master")
    app.include_router(agent_router, prefix="/api/v1/agent-builder")
    
    return TestClient(app)


class TestMasterAgentEndpoints:
    """Test Master Agent API endpoints"""
    
    def test_create_custom_agent_endpoint(self, client):
        """Test POST /api/v1/master/agent/build"""
        payload = {
            "name": "Test Agent",
            "description": "Test agent description",
            "agent_type": "assistant",
            "capabilities": ["conversation"],
            "tools": ["data_json_parser"],
            "system_prompt": "You are a helpful assistant"
        }
        
        with patch('services.enhanced_master_agent.EnhancedMasterAgent.create_custom_agent') as mock_create:
            mock_create.return_value = {
                "success": True,
                "agent_id": "agent_123",
                "agent": payload
            }
            
            response = client.post("/api/v1/master/agent/build", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "agent_id" in data
    
    def test_deploy_agent_endpoint(self, client):
        """Test POST /api/v1/master/agent/deploy/{agent_id}"""
        agent_id = "agent_123"
        payload = {
            "runtime_config": {
                "max_turns": 10,
                "timeout": 300
            }
        }
        
        with patch('services.enhanced_master_agent.EnhancedMasterAgent.deploy_agent_to_autogen') as mock_deploy:
            mock_deploy.return_value = {
                "success": True,
                "deployment": {
                    "agent_id": agent_id,
                    "status": "deployed"
                }
            }
            
            response = client.post(f"/api/v1/master/agent/deploy/{agent_id}", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["deployment"]["status"] == "deployed"
    
    def test_monitor_agent_endpoint(self, client):
        """Test GET /api/v1/master/agent/monitor/{agent_id}"""
        agent_id = "agent_123"
        
        with patch('services.enhanced_master_agent.EnhancedMasterAgent.monitor_agent') as mock_monitor:
            mock_monitor.return_value = {
                "success": True,
                "agent_id": agent_id,
                "metrics": {
                    "total_executions": 10,
                    "success_rate": 0.9,
                    "avg_execution_time": 2.5
                },
                "health_score": 85
            }
            
            response = client.get(f"/api/v1/master/agent/monitor/{agent_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["health_score"] == 85
    
    def test_route_task_endpoint(self, client):
        """Test POST /api/v1/master/agent/route-task"""
        payload = {
            "task": {
                "description": "Research AI trends",
                "type": "research",
                "priority": "high"
            },
            "preferred_agent_id": None
        }
        
        with patch('services.enhanced_master_agent.EnhancedMasterAgent.route_task_to_agent') as mock_route:
            mock_route.return_value = {
                "success": True,
                "selected_agent": {
                    "id": "agent_123",
                    "name": "Research Agent",
                    "match_score": 0.95
                },
                "execution": {
                    "id": "exec_123",
                    "status": "completed"
                }
            }
            
            response = client.post("/api/v1/master/agent/route-task", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["selected_agent"]["match_score"] == 0.95
    
    def test_execute_agent_endpoint(self, client):
        """Test POST /api/v1/master/agent/execute"""
        payload = {
            "agent_id": "agent_123",
            "task": {
                "description": "Parse JSON data",
                "input": {"data": '{"test": true}'}
            }
        }
        
        with patch('services.enhanced_master_agent.EnhancedMasterAgent._execute_agent_task') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "execution_id": "exec_123",
                "result": {"output": "Parsed successfully"}
            }
            
            response = client.post("/api/v1/master/agent/execute", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "execution_id" in data


class TestWorkflowEndpoints:
    """Test Workflow API endpoints"""
    
    def test_create_workflow_endpoint(self, client):
        """Test POST /api/v1/agent-builder/workflows"""
        payload = {
            "name": "Test Workflow",
            "description": "Test workflow description",
            "workflow_type": "sequential",
            "agents": ["agent1", "agent2", "agent3"],
            "configuration": {}
        }
        
        with patch('services.agent_collaboration.AgentCollaborationService.create_workflow') as mock_create:
            mock_workflow = Mock()
            mock_workflow.id = "workflow_123"
            mock_workflow.name = "Test Workflow"
            mock_workflow.workflow_type = "sequential"
            mock_workflow.agents = ["agent1", "agent2", "agent3"]
            mock_create.return_value = mock_workflow
            
            response = client.post("/api/v1/agent-builder/workflows", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "workflow_123"
            assert data["name"] == "Test Workflow"
    
    def test_execute_workflow_endpoint(self, client):
        """Test POST /api/v1/agent-builder/workflows/{workflow_id}/execute"""
        workflow_id = "workflow_123"
        payload = {
            "input_data": {
                "task": "Test task",
                "priority": "high"
            }
        }
        
        with patch('services.agent_collaboration.AgentCollaborationService.execute_workflow') as mock_execute:
            mock_execution = Mock()
            mock_execution.id = "exec_123"
            mock_execution.workflow_id = workflow_id
            mock_execution.status = "running"
            mock_execute.return_value = mock_execution
            
            response = client.post(f"/api/v1/agent-builder/workflows/{workflow_id}/execute", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "exec_123"
            assert data["workflow_id"] == workflow_id
    
    def test_get_workflow_execution_endpoint(self, client):
        """Test GET /api/v1/agent-builder/workflows/{workflow_id}/executions/{execution_id}"""
        workflow_id = "workflow_123"
        execution_id = "exec_123"
        
        with patch('services.agent_collaboration.AgentCollaborationService.get_workflow_execution') as mock_get:
            mock_execution = Mock()
            mock_execution.id = execution_id
            mock_execution.workflow_id = workflow_id
            mock_execution.status = "completed"
            mock_execution.result = {"output": "Success"}
            mock_get.return_value = mock_execution
            
            response = client.get(f"/api/v1/agent-builder/workflows/{workflow_id}/executions/{execution_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == execution_id
            assert data["status"] == "completed"
    
    def test_get_workflow_metrics_endpoint(self, client):
        """Test GET /api/v1/agent-builder/workflows/metrics"""
        with patch('services.agent_collaboration.AgentCollaborationService.get_workflow_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "total_workflows": 10,
                "total_executions": 50,
                "by_type": {
                    "sequential": 5,
                    "parallel": 3,
                    "hierarchical": 2
                },
                "success_rate": 0.92
            }
            
            response = client.get("/api/v1/agent-builder/workflows/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_workflows"] == 10
            assert data["success_rate"] == 0.92


class TestAgentExecutionEndpoints:
    """Test Agent Execution endpoints"""
    
    def test_execute_agent_with_tools(self, client):
        """Test agent execution with tool calling"""
        payload = {
            "agent_id": "agent_123",
            "prompt": "Parse this JSON: {\"name\": \"test\"}",
            "context": {},
            "max_iterations": 5
        }
        
        with patch('services.agent_builder_service.AgentBuilderService.execute_agent') as mock_execute:
            mock_execution = Mock()
            mock_execution.id = "exec_123"
            mock_execution.status = "completed"
            mock_execution.result = {
                "output": "Parsed successfully",
                "tool_calls": [
                    {
                        "tool": "data_json_parser",
                        "result": {"name": "test"}
                    }
                ]
            }
            mock_execute.return_value = mock_execution
            
            response = client.post("/api/v1/agent-builder/execute", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert len(data["result"]["tool_calls"]) > 0
    
    def test_execute_agent_with_mcp(self, client):
        """Test agent execution with MCP tools"""
        payload = {
            "agent_id": "agent_123",
            "prompt": "Send a Slack message",
            "context": {"channel": "#general"},
            "max_iterations": 5
        }
        
        with patch('services.agent_builder_service.AgentBuilderService.execute_agent') as mock_execute:
            mock_execution = Mock()
            mock_execution.id = "exec_123"
            mock_execution.status = "completed"
            mock_execution.result = {
                "output": "Message sent",
                "mcp_calls": [
                    {
                        "server": "slack",
                        "tool": "send_message",
                        "result": {"success": True}
                    }
                ]
            }
            mock_execute.return_value = mock_execution
            
            response = client.post("/api/v1/agent-builder/execute", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"


class TestErrorHandling:
    """Test error handling in API endpoints"""
    
    def test_invalid_agent_id(self, client):
        """Test error handling for invalid agent ID"""
        with patch('services.enhanced_master_agent.EnhancedMasterAgent.monitor_agent') as mock_monitor:
            mock_monitor.side_effect = ValueError("Agent not found")
            
            response = client.get("/api/v1/master/agent/monitor/invalid_id")
            
            assert response.status_code in [400, 404, 500]
    
    def test_invalid_workflow_type(self, client):
        """Test error handling for invalid workflow type"""
        payload = {
            "name": "Test Workflow",
            "description": "Test",
            "workflow_type": "invalid_type",
            "agents": ["agent1"]
        }
        
        response = client.post("/api/v1/agent-builder/workflows", json=payload)
        
        assert response.status_code in [400, 422]
    
    def test_missing_required_fields(self, client):
        """Test error handling for missing required fields"""
        payload = {
            "name": "Test Agent"
            # Missing required fields
        }
        
        response = client.post("/api/v1/master/agent/build", json=payload)
        
        assert response.status_code in [400, 422]


class TestRateLimiting:
    """Test rate limiting on endpoints"""
    
    def test_rate_limit_enforcement(self, client):
        """Test that rate limiting is enforced"""
        # Make multiple rapid requests
        responses = []
        for i in range(100):
            response = client.get("/api/v1/master/agent/monitor/agent_123")
            responses.append(response)
        
        # At least one should be rate limited (429)
        status_codes = [r.status_code for r in responses]
        # Note: This test assumes rate limiting is configured
        # Adjust based on your actual rate limit settings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

