"""
MCP Hub Module Test Suite
Comprehensive tests for MCP Server Discovery and Management
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from database import Database
from services.mcp_service import MCPHubService
from models.mcp_models import *

class TestMCPHubModule:
    def __init__(self):
        self.db = None
        self.service = None
        self.test_results = []
        
    async def setup(self):
        """Initialize database and service"""
        self.db = Database()
        await self.db.connect()
        self.service = MCPHubService(self.db)
        print("✅ MCP Hub module test setup complete")
    
    async def cleanup(self):
        """Clean up test data"""
        if self.db is not None:
            for collection in ["mcp_servers", "mcp_tools", "mcp_discovery_configs", "mcp_discovery_events", "mcp_health_checks", "mcp_integrations", "mcp_notifications"]:
                try:
                    docs = await self.db.find_many(collection, {}, limit=10000)
                    for doc in docs:
                        if "_id" in doc:
                            await self.db.delete_one(collection, {"_id": doc["_id"]})
                except:
                    pass
        print("✅ Test cleanup complete")
    
    def record_result(self, test_name: str, passed: bool, message: str = ""):
        """Record test result"""
        self.test_results.append({"test": test_name, "passed": passed, "message": message})
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}" + (f" - {message}" if message else ""))
    
    async def test_server_crud(self):
        """Test MCP server CRUD operations"""
        try:
            # Create server
            server_create = MCPServerCreate(
                name="Test MCP Server",
                description="A test MCP server",
                version="1.0.0",
                author="Test Author",
                repository="https://github.com/test/mcp-server",
                command="node",
                args=["index.js"],
                capabilities=["tools", "resources"],
                tags=["test", "demo"],
                discovery_source=DiscoverySource.MANUAL
            )
            server = await self.service.create_server(server_create)
            assert server.server_id.startswith("mcp_"), "Server ID format incorrect"
            assert server.name == "Test MCP Server", "Server name mismatch"
            
            # Get server
            retrieved = await self.service.get_server(server.server_id)
            assert retrieved is not None, "Server not found"
            assert retrieved.server_id == server.server_id, "Server ID mismatch"
            
            # Update server
            update = MCPServerUpdate(name="Updated MCP Server", status=MCPServerStatus.ACTIVE)
            updated = await self.service.update_server(server.server_id, update)
            assert updated is not None, "Server update failed"
            assert updated.name == "Updated MCP Server", "Server name not updated"
            
            # List servers
            servers = await self.service.list_servers()
            assert len(servers) >= 1, "No servers found"
            
            # Delete server
            deleted = await self.service.delete_server(server.server_id)
            assert deleted, "Server deletion failed"
            
            self.record_result("Server CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Server CRUD", False, str(e))
    
    async def test_tool_crud(self):
        """Test MCP tool CRUD operations"""
        try:
            # Create server first
            server_create = MCPServerCreate(name="Tool Test Server", command="node", discovery_source=DiscoverySource.MANUAL)
            server = await self.service.create_server(server_create)
            
            # Create tool
            tool_create = MCPToolBase(
                name="test_function",
                description="A test function",
                tool_type=MCPToolType.FUNCTION,
                parameters={"param1": {"type": "string", "description": "Test parameter"}},
                returns={"type": "string"}
            )
            tool = await self.service.create_tool(tool_create, server.server_id)
            assert tool.tool_id.startswith("tool_"), "Tool ID format incorrect"
            assert tool.name == "test_function", "Tool name mismatch"
            
            # Get tool
            retrieved = await self.service.get_tool(tool.tool_id)
            assert retrieved is not None, "Tool not found"
            
            # List tools
            tools = await self.service.list_tools(server_id=server.server_id)
            assert len(tools) >= 1, "No tools found"
            
            # Verify server tool count updated
            updated_server = await self.service.get_server(server.server_id)
            assert updated_server.tool_count == 1, "Server tool count not updated"
            
            self.record_result("Tool CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Tool CRUD", False, str(e))
    
    async def test_discovery_config_crud(self):
        """Test discovery config CRUD operations"""
        try:
            # Create discovery config
            config_create = DiscoveryConfigCreate(
                source=DiscoverySource.OFFICIAL_REGISTRY,
                enabled=True,
                check_interval_seconds=3600,
                registry_url="https://registry.example.com",
                search_keywords=["mcp", "server"]
            )
            config = await self.service.create_discovery_config(config_create)
            assert config.config_id.startswith("cfg_"), "Config ID format incorrect"
            assert config.source == DiscoverySource.OFFICIAL_REGISTRY, "Source mismatch"
            
            # Get config
            retrieved = await self.service.get_discovery_config(config.config_id)
            assert retrieved is not None, "Config not found"
            
            # Update config
            update = DiscoveryConfigUpdate(enabled=False, check_interval_seconds=7200)
            updated = await self.service.update_discovery_config(config.config_id, update)
            assert updated is not None, "Config update failed"
            assert updated.enabled == False, "Enabled status not updated"
            
            # List configs
            configs = await self.service.list_discovery_configs()
            assert len(configs) >= 1, "No configs found"
            
            self.record_result("Discovery Config CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Discovery Config CRUD", False, str(e))
    
    async def test_discovery_events(self):
        """Test discovery event creation"""
        try:
            # Create discovery event
            event_create = DiscoveryEventBase(
                source=DiscoverySource.GITHUB,
                event_type="new_server",
                server_name="New Test Server",
                server_data={"version": "1.0.0", "author": "Test"}
            )
            event = await self.service.create_discovery_event(event_create)
            assert event.event_id.startswith("evt_"), "Event ID format incorrect"
            assert event.processed == False, "Event should not be processed initially"
            
            # List events
            events = await self.service.list_discovery_events(processed=False)
            assert len(events) >= 1, "No events found"
            
            self.record_result("Discovery Events", True, "Event creation successful")
        except Exception as e:
            self.record_result("Discovery Events", False, str(e))
    
    async def test_health_checks(self):
        """Test health check functionality"""
        try:
            # Create server
            server_create = MCPServerCreate(name="Health Test Server", command="node", discovery_source=DiscoverySource.MANUAL)
            server = await self.service.create_server(server_create)
            
            # Create health check
            check_create = HealthCheckBase(
                server_id=server.server_id,
                status=HealthStatus.HEALTHY,
                response_time_ms=150.5,
                details={"uptime": 3600}
            )
            check = await self.service.create_health_check(check_create)
            assert check.check_id.startswith("chk_"), "Check ID format incorrect"
            assert check.status == HealthStatus.HEALTHY, "Status mismatch"
            
            # Verify server health status updated
            updated_server = await self.service.get_server(server.server_id)
            assert updated_server.health_status == HealthStatus.HEALTHY, "Server health status not updated"
            
            # List health checks
            checks = await self.service.list_health_checks(server_id=server.server_id)
            assert len(checks) >= 1, "No health checks found"
            
            self.record_result("Health Checks", True, "Health check successful")
        except Exception as e:
            self.record_result("Health Checks", False, str(e))
    
    async def test_agent_integrations(self):
        """Test agent MCP integration"""
        try:
            # Create server
            server_create = MCPServerCreate(name="Integration Test Server", command="node", discovery_source=DiscoverySource.MANUAL)
            server = await self.service.create_server(server_create)
            
            # Create integration
            integration_create = AgentMCPIntegrationCreate(
                agent_id="agent_001",
                server_id=server.server_id,
                enabled=True,
                tool_whitelist=["tool1", "tool2"],
                priority=50
            )
            integration = await self.service.create_integration(integration_create)
            assert integration.integration_id.startswith("int_"), "Integration ID format incorrect"
            assert integration.agent_id == "agent_001", "Agent ID mismatch"
            
            # Get integration
            retrieved = await self.service.get_integration(integration.integration_id)
            assert retrieved is not None, "Integration not found"
            
            # Update integration
            update = AgentMCPIntegrationUpdate(enabled=False, priority=75)
            updated = await self.service.update_integration(integration.integration_id, update)
            assert updated is not None, "Integration update failed"
            assert updated.priority == 75, "Priority not updated"
            
            # List integrations
            integrations = await self.service.list_integrations(agent_id="agent_001")
            assert len(integrations) >= 1, "No integrations found"
            
            self.record_result("Agent Integrations", True, "All operations successful")
        except Exception as e:
            self.record_result("Agent Integrations", False, str(e))
    
    async def test_notifications(self):
        """Test notification system"""
        try:
            # Notifications are created automatically by other operations
            # List notifications
            notifications = await self.service.list_notifications(read=False)
            # Should have notifications from server creation, health checks, etc.
            assert notifications is not None, "Notifications not returned"
            
            self.record_result("Notifications", True, f"{len(notifications)} notifications found")
        except Exception as e:
            self.record_result("Notifications", False, str(e))
    
    async def test_server_catalog(self):
        """Test server catalog"""
        try:
            catalog = await self.service.get_server_catalog()
            assert catalog is not None, "Catalog not returned"
            assert isinstance(catalog, list), "Catalog should be a list"
            
            self.record_result("Server Catalog", True, f"{len(catalog)} servers in catalog")
        except Exception as e:
            self.record_result("Server Catalog", False, str(e))
    
    async def test_statistics(self):
        """Test MCP Hub statistics"""
        try:
            stats = await self.service.get_stats()
            assert stats is not None, "Stats not returned"
            assert hasattr(stats, 'total_servers'), "Missing total_servers"
            assert hasattr(stats, 'total_tools'), "Missing total_tools"
            assert hasattr(stats, 'total_integrations'), "Missing total_integrations"
            
            self.record_result("Statistics", True, f"Stats: {stats.total_servers} servers, {stats.total_tools} tools, {stats.total_integrations} integrations")
        except Exception as e:
            self.record_result("Statistics", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 MCP HUB MODULE TEST SUITE")
        print("="*80 + "\n")
        
        await self.setup()
        
        # Run tests
        await self.test_server_crud()
        await self.test_tool_crud()
        await self.test_discovery_config_crud()
        await self.test_discovery_events()
        await self.test_health_checks()
        await self.test_agent_integrations()
        await self.test_notifications()
        await self.test_server_catalog()
        await self.test_statistics()
        
        await self.cleanup()
        
        # Print summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass Rate: {(passed/total*100):.1f}%")
        print("="*80 + "\n")
        
        return passed == total

async def main():
    tester = TestMCPHubModule()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())

