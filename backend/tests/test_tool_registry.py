"""
Unit Tests for Tool Registry

Tests all 13 built-in tools across 5 categories:
- OSINT Tools (5)
- Data Processing Tools (3)
- Web & API Tools (2)
- Code Execution Tools (1)
- AI/ML Tools (2)
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.tool_registry import ToolRegistry, ToolCategory, PermissionLevel


@pytest.fixture
async def tool_registry():
    """Create a ToolRegistry instance for testing"""
    mock_db = Mock()
    mock_db.tools = Mock()
    mock_db.tool_permissions = Mock()
    mock_db.tool_executions = Mock()
    
    registry = ToolRegistry(mock_db)
    await registry.initialize()
    return registry


@pytest.fixture
def test_agent_id():
    """Test agent ID"""
    return "test_agent_123"


class TestToolRegistryInitialization:
    """Test Tool Registry initialization"""
    
    @pytest.mark.asyncio
    async def test_registry_initialization(self, tool_registry):
        """Test that registry initializes with all tools"""
        assert len(tool_registry.tools) == 13
        assert "osint_web_scraper" in tool_registry.tools
        assert "data_json_parser" in tool_registry.tools
        assert "web_http_get" in tool_registry.tools
        assert "code_python_executor" in tool_registry.tools
        assert "ai_text_embeddings" in tool_registry.tools
    
    @pytest.mark.asyncio
    async def test_tool_categories(self, tool_registry):
        """Test that tools are properly categorized"""
        osint_tools = [t for t in tool_registry.tools.values() if t.category == ToolCategory.OSINT]
        data_tools = [t for t in tool_registry.tools.values() if t.category == ToolCategory.DATA_PROCESSING]
        web_tools = [t for t in tool_registry.tools.values() if t.category == ToolCategory.WEB_API]
        code_tools = [t for t in tool_registry.tools.values() if t.category == ToolCategory.CODE_EXECUTION]
        ai_tools = [t for t in tool_registry.tools.values() if t.category == ToolCategory.AI_ML]
        
        assert len(osint_tools) == 5
        assert len(data_tools) == 3
        assert len(web_tools) == 2
        assert len(code_tools) == 1
        assert len(ai_tools) == 2


class TestOSINTTools:
    """Test OSINT tools"""
    
    @pytest.mark.asyncio
    async def test_web_scraper(self, tool_registry, test_agent_id):
        """Test web scraper tool"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.text = AsyncMock(return_value="<html><body><h1>Test</h1><a href='/link'>Link</a></body></html>")
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await tool_registry.execute_tool(
                "osint_web_scraper",
                {"url": "https://example.com"},
                test_agent_id
            )
            
            assert result["success"] is True
            assert "text" in result["result"]
            assert "links" in result["result"]
    
    @pytest.mark.asyncio
    async def test_whois_lookup(self, tool_registry, test_agent_id):
        """Test WHOIS lookup tool"""
        with patch('whois.whois') as mock_whois:
            mock_whois.return_value = {
                "domain_name": "EXAMPLE.COM",
                "registrar": "Test Registrar",
                "creation_date": datetime.now()
            }
            
            result = await tool_registry.execute_tool(
                "osint_whois_lookup",
                {"domain": "example.com"},
                test_agent_id
            )
            
            assert result["success"] is True
            assert "domain_name" in result["result"]
    
    @pytest.mark.asyncio
    async def test_dns_lookup(self, tool_registry, test_agent_id):
        """Test DNS lookup tool"""
        with patch('dns.resolver.resolve') as mock_resolve:
            mock_resolve.return_value = [Mock(to_text=lambda: "93.184.216.34")]
            
            result = await tool_registry.execute_tool(
                "osint_dns_lookup",
                {"domain": "example.com", "record_type": "A"},
                test_agent_id
            )
            
            assert result["success"] is True
            assert len(result["result"]["records"]) > 0
    
    @pytest.mark.asyncio
    async def test_email_validator(self, tool_registry, test_agent_id):
        """Test email validator tool"""
        result = await tool_registry.execute_tool(
            "osint_email_validator",
            {"email": "test@example.com"},
            test_agent_id
        )
        
        assert result["success"] is True
        assert result["result"]["valid"] is True
        assert result["result"]["domain"] == "example.com"
    
    @pytest.mark.asyncio
    async def test_ip_geolocation(self, tool_registry, test_agent_id):
        """Test IP geolocation tool"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={
                "status": "success",
                "country": "United States",
                "city": "Los Angeles"
            })
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await tool_registry.execute_tool(
                "osint_ip_geolocation",
                {"ip": "8.8.8.8"},
                test_agent_id
            )
            
            assert result["success"] is True
            assert result["result"]["country"] == "United States"


class TestDataProcessingTools:
    """Test data processing tools"""
    
    @pytest.mark.asyncio
    async def test_json_parser(self, tool_registry, test_agent_id):
        """Test JSON parser tool"""
        result = await tool_registry.execute_tool(
            "data_json_parser",
            {"json_string": '{"name": "test", "value": 123}'},
            test_agent_id
        )
        
        assert result["success"] is True
        assert result["result"]["name"] == "test"
        assert result["result"]["value"] == 123
    
    @pytest.mark.asyncio
    async def test_csv_parser(self, tool_registry, test_agent_id):
        """Test CSV parser tool"""
        csv_data = "name,age,city\nJohn,30,NYC\nJane,25,LA"
        
        result = await tool_registry.execute_tool(
            "data_csv_parser",
            {"csv_string": csv_data, "delimiter": ",", "has_header": True},
            test_agent_id
        )
        
        assert result["success"] is True
        assert len(result["result"]) == 2
        assert result["result"][0]["name"] == "John"
    
    @pytest.mark.asyncio
    async def test_data_transformer(self, tool_registry, test_agent_id):
        """Test data transformer tool"""
        data = {"a": {"b": {"c": 1}}, "d": 2}
        
        result = await tool_registry.execute_tool(
            "data_transformer",
            {"data": data, "operation": "flatten"},
            test_agent_id
        )
        
        assert result["success"] is True
        assert "a.b.c" in result["result"]


class TestWebAPITools:
    """Test web and API tools"""
    
    @pytest.mark.asyncio
    async def test_http_get(self, tool_registry, test_agent_id):
        """Test HTTP GET tool"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"status": "ok"})
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await tool_registry.execute_tool(
                "web_http_get",
                {"url": "https://api.example.com/data"},
                test_agent_id
            )
            
            assert result["success"] is True
            assert result["result"]["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_http_post(self, tool_registry, test_agent_id):
        """Test HTTP POST tool"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"id": 123})
            mock_response.status = 201
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await tool_registry.execute_tool(
                "web_http_post",
                {"url": "https://api.example.com/create", "data": {"name": "test"}},
                test_agent_id
            )
            
            assert result["success"] is True
            assert result["result"]["status_code"] == 201


class TestCodeExecutionTools:
    """Test code execution tools"""
    
    @pytest.mark.asyncio
    async def test_python_executor(self, tool_registry, test_agent_id):
        """Test Python code executor"""
        code = "result = 2 + 2"
        
        result = await tool_registry.execute_tool(
            "code_python_executor",
            {"code": code},
            test_agent_id
        )
        
        assert result["success"] is True
        assert result["result"]["result"] == 4
    
    @pytest.mark.asyncio
    async def test_python_executor_security(self, tool_registry, test_agent_id):
        """Test Python executor security restrictions"""
        malicious_code = "import os; os.system('ls')"
        
        result = await tool_registry.execute_tool(
            "code_python_executor",
            {"code": malicious_code},
            test_agent_id
        )
        
        # Should fail due to restricted imports
        assert result["success"] is False


class TestAIMLTools:
    """Test AI/ML tools"""
    
    @pytest.mark.asyncio
    async def test_text_embeddings(self, tool_registry, test_agent_id):
        """Test text embeddings tool"""
        result = await tool_registry.execute_tool(
            "ai_text_embeddings",
            {"text": "This is a test sentence"},
            test_agent_id
        )
        
        assert result["success"] is True
        assert "embedding" in result["result"]
        assert len(result["result"]["embedding"]) > 0
    
    @pytest.mark.asyncio
    async def test_text_similarity(self, tool_registry, test_agent_id):
        """Test text similarity tool"""
        result = await tool_registry.execute_tool(
            "ai_text_similarity",
            {"text1": "hello world", "text2": "hello earth"},
            test_agent_id
        )
        
        assert result["success"] is True
        assert 0 <= result["result"]["similarity"] <= 1


class TestPermissions:
    """Test permission system"""
    
    @pytest.mark.asyncio
    async def test_public_tool_access(self, tool_registry, test_agent_id):
        """Test that public tools are accessible to all agents"""
        has_permission = await tool_registry.check_permission(test_agent_id, "data_json_parser")
        assert has_permission is True
    
    @pytest.mark.asyncio
    async def test_restricted_tool_access(self, tool_registry, test_agent_id):
        """Test that restricted tools require explicit permission"""
        # By default, should not have permission
        has_permission = await tool_registry.check_permission(test_agent_id, "code_python_executor")
        assert has_permission is False
        
        # Grant permission
        await tool_registry.grant_permission(test_agent_id, "code_python_executor")
        
        # Now should have permission
        has_permission = await tool_registry.check_permission(test_agent_id, "code_python_executor")
        assert has_permission is True


class TestMetrics:
    """Test metrics tracking"""
    
    @pytest.mark.asyncio
    async def test_execution_metrics(self, tool_registry, test_agent_id):
        """Test that execution metrics are tracked"""
        # Execute a tool
        await tool_registry.execute_tool(
            "data_json_parser",
            {"json_string": '{"test": true}'},
            test_agent_id
        )
        
        # Check metrics
        metrics = tool_registry.get_metrics()
        assert metrics["total_executions"] > 0
        assert "data_json_parser" in metrics["by_tool"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

