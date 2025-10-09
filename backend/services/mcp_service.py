"""
MCP Hub Service for CPAS4 - Real Database Integration
Intelligent MCP server discovery, registration, and monitoring
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from database import Database
from models.mcp_models import *

logger = logging.getLogger(__name__)

class MCPHubService:
    def __init__(self, database: Database):
        self.db = database
        self.servers_collection = "mcp_servers"
        self.tools_collection = "mcp_tools"
        self.discovery_configs_collection = "mcp_discovery_configs"
        self.discovery_events_collection = "mcp_discovery_events"
        self.health_checks_collection = "mcp_health_checks"
        self.integrations_collection = "mcp_integrations"
        self.notifications_collection = "mcp_notifications"
        logger.info("✅ MCP Hub Service initialized with real database")
    
    # MCP Server CRUD
    async def create_server(self, server_create: MCPServerCreate) -> MCPServer:
        server_id = f"mcp_{uuid.uuid4().hex[:12]}"
        server = MCPServer(server_id=server_id, **server_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow(), discovered_at=datetime.utcnow() if server_create.discovery_source != DiscoverySource.MANUAL else None)
        sd = server.model_dump()
        for k in ["created_at", "updated_at", "discovered_at", "last_health_check", "last_discovery_check"]:
            if sd.get(k): sd[k] = sd[k].isoformat()
        await self.db.insert_one(self.servers_collection, sd)
        
        # Create notification for new server
        await self.create_notification("new_server", "info", f"New MCP Server: {server.name}", f"Server {server.name} has been registered", server_id)
        
        logger.info(f"✅ MCP Server created: {server_id}")
        return server
    
    async def get_server(self, server_id: str) -> Optional[MCPServer]:
        sd = await self.db.find_one(self.servers_collection, {"server_id": server_id})
        if not sd: return None
        for k in ["created_at", "updated_at", "discovered_at", "last_health_check", "last_discovery_check"]:
            if k in sd and sd[k] and isinstance(sd[k], str): sd[k] = datetime.fromisoformat(sd[k])
        return MCPServer(**sd)
    
    async def update_server(self, server_id: str, server_update: MCPServerUpdate) -> Optional[MCPServer]:
        ud = server_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_server(server_id) if await self.db.update_one(self.servers_collection, {"server_id": server_id}, {"$set": ud}) else None
    
    async def delete_server(self, server_id: str) -> bool:
        # Delete associated tools
        tools = await self.list_tools(server_id=server_id, limit=10000)
        for tool in tools:
            await self.delete_tool(tool.tool_id)
        
        # Delete integrations
        integrations = await self.list_integrations(server_id=server_id, limit=10000)
        for integration in integrations:
            await self.delete_integration(integration.integration_id)
        
        return await self.db.delete_one(self.servers_collection, {"server_id": server_id})
    
    async def list_servers(self, status: Optional[MCPServerStatus] = None, discovery_source: Optional[DiscoverySource] = None, limit: int = 100) -> List[MCPServer]:
        f = {}
        if status: f["status"] = status
        if discovery_source: f["discovery_source"] = discovery_source
        sds = await self.db.find_many(self.servers_collection, f, limit=limit)
        servers = []
        for sd in sds:
            for k in ["created_at", "updated_at", "discovered_at", "last_health_check", "last_discovery_check"]:
                if k in sd and sd[k] and isinstance(sd[k], str): sd[k] = datetime.fromisoformat(sd[k])
            # Count tools
            tools = await self.list_tools(server_id=sd["server_id"], limit=10000)
            sd["tool_count"] = len(tools)
            servers.append(MCPServer(**sd))
        return servers
    
    # MCP Tool CRUD
    async def create_tool(self, tool_create: MCPToolBase, server_id: str) -> MCPTool:
        tool_id = f"tool_{uuid.uuid4().hex[:12]}"
        tool = MCPTool(tool_id=tool_id, server_id=server_id, **tool_create.model_dump(), created_at=datetime.utcnow())
        td = tool.model_dump()
        for k in ["created_at", "last_used"]:
            if td.get(k): td[k] = td[k].isoformat()
        await self.db.insert_one(self.tools_collection, td)
        
        # Update server tool count
        server = await self.get_server(server_id)
        if server:
            await self.db.update_one(self.servers_collection, {"server_id": server_id}, {"$set": {"tool_count": server.tool_count + 1}})
        
        logger.info(f"✅ MCP Tool created: {tool_id}")
        return tool
    
    async def get_tool(self, tool_id: str) -> Optional[MCPTool]:
        td = await self.db.find_one(self.tools_collection, {"tool_id": tool_id})
        if not td: return None
        for k in ["created_at", "last_used"]:
            if k in td and td[k] and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
        return MCPTool(**td)
    
    async def delete_tool(self, tool_id: str) -> bool:
        tool = await self.get_tool(tool_id)
        if tool and await self.db.delete_one(self.tools_collection, {"tool_id": tool_id}):
            # Update server tool count
            server = await self.get_server(tool.server_id)
            if server:
                await self.db.update_one(self.servers_collection, {"server_id": tool.server_id}, {"$set": {"tool_count": max(0, server.tool_count - 1)}})
            return True
        return False
    
    async def list_tools(self, server_id: Optional[str] = None, tool_type: Optional[MCPToolType] = None, limit: int = 100) -> List[MCPTool]:
        f = {}
        if server_id: f["server_id"] = server_id
        if tool_type: f["tool_type"] = tool_type
        tds = await self.db.find_many(self.tools_collection, f, limit=limit)
        tools = []
        for td in tds:
            for k in ["created_at", "last_used"]:
                if k in td and td[k] and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
            tools.append(MCPTool(**td))
        return tools
    
    # Discovery Config CRUD
    async def create_discovery_config(self, config_create: DiscoveryConfigCreate) -> DiscoveryConfig:
        config_id = f"cfg_{uuid.uuid4().hex[:12]}"
        config = DiscoveryConfig(config_id=config_id, **config_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        cd = config.model_dump()
        for k in ["created_at", "updated_at", "last_check"]:
            if cd.get(k): cd[k] = cd[k].isoformat()
        await self.db.insert_one(self.discovery_configs_collection, cd)
        logger.info(f"✅ Discovery Config created: {config_id}")
        return config
    
    async def get_discovery_config(self, config_id: str) -> Optional[DiscoveryConfig]:
        cd = await self.db.find_one(self.discovery_configs_collection, {"config_id": config_id})
        if not cd: return None
        for k in ["created_at", "updated_at", "last_check"]:
            if k in cd and cd[k] and isinstance(cd[k], str): cd[k] = datetime.fromisoformat(cd[k])
        return DiscoveryConfig(**cd)
    
    async def update_discovery_config(self, config_id: str, config_update: DiscoveryConfigUpdate) -> Optional[DiscoveryConfig]:
        ud = config_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_discovery_config(config_id) if await self.db.update_one(self.discovery_configs_collection, {"config_id": config_id}, {"$set": ud}) else None
    
    async def delete_discovery_config(self, config_id: str) -> bool:
        return await self.db.delete_one(self.discovery_configs_collection, {"config_id": config_id})
    
    async def list_discovery_configs(self, enabled: Optional[bool] = None, limit: int = 100) -> List[DiscoveryConfig]:
        f = {}
        if enabled is not None: f["enabled"] = enabled
        cds = await self.db.find_many(self.discovery_configs_collection, f, limit=limit)
        configs = []
        for cd in cds:
            for k in ["created_at", "updated_at", "last_check"]:
                if k in cd and cd[k] and isinstance(cd[k], str): cd[k] = datetime.fromisoformat(cd[k])
            configs.append(DiscoveryConfig(**cd))
        return configs
    
    # Discovery Events
    async def create_discovery_event(self, event_create: DiscoveryEventBase) -> DiscoveryEvent:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        event = DiscoveryEvent(event_id=event_id, **event_create.model_dump(), created_at=datetime.utcnow())
        ed = event.model_dump()
        for k in ["created_at", "processed_at"]:
            if ed.get(k): ed[k] = ed[k].isoformat()
        await self.db.insert_one(self.discovery_events_collection, ed)
        logger.info(f"✅ Discovery Event created: {event_id}")
        return event
    
    async def list_discovery_events(self, processed: Optional[bool] = None, limit: int = 100) -> List[DiscoveryEvent]:
        f = {}
        if processed is not None: f["processed"] = processed
        eds = await self.db.find_many(self.discovery_events_collection, f, limit=limit)
        events = []
        for ed in eds:
            for k in ["created_at", "processed_at"]:
                if k in ed and ed[k] and isinstance(ed[k], str): ed[k] = datetime.fromisoformat(ed[k])
            events.append(DiscoveryEvent(**ed))
        return events
    
    # Health Checks
    async def create_health_check(self, check_create: HealthCheckBase) -> HealthCheck:
        check_id = f"chk_{uuid.uuid4().hex[:12]}"
        check = HealthCheck(check_id=check_id, **check_create.model_dump(), checked_at=datetime.utcnow())
        cd = check.model_dump()
        cd["checked_at"] = cd["checked_at"].isoformat()
        await self.db.insert_one(self.health_checks_collection, cd)
        
        # Update server health status
        await self.db.update_one(self.servers_collection, {"server_id": check.server_id}, {"$set": {"health_status": check.status, "last_health_check": datetime.utcnow().isoformat()}})
        
        # Create notification if unhealthy
        if check.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
            server = await self.get_server(check.server_id)
            if server:
                await self.create_notification("health_degraded", "warning" if check.status == HealthStatus.DEGRADED else "error", f"Server Health Issue: {server.name}", f"Server {server.name} is {check.status}", check.server_id)
        
        return check
    
    async def list_health_checks(self, server_id: Optional[str] = None, limit: int = 100) -> List[HealthCheck]:
        f = {}
        if server_id: f["server_id"] = server_id
        cds = await self.db.find_many(self.health_checks_collection, f, limit=limit)
        checks = []
        for cd in cds:
            if "checked_at" in cd and isinstance(cd["checked_at"], str): cd["checked_at"] = datetime.fromisoformat(cd["checked_at"])
            checks.append(HealthCheck(**cd))
        return checks
    
    # Agent Integrations
    async def create_integration(self, integration_create: AgentMCPIntegrationCreate) -> AgentMCPIntegration:
        integration_id = f"int_{uuid.uuid4().hex[:12]}"
        integration = AgentMCPIntegration(integration_id=integration_id, **integration_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        id = integration.model_dump()
        for k in ["created_at", "updated_at", "last_used"]:
            if id.get(k): id[k] = id[k].isoformat()
        await self.db.insert_one(self.integrations_collection, id)
        logger.info(f"✅ Agent MCP Integration created: {integration_id}")
        return integration
    
    async def get_integration(self, integration_id: str) -> Optional[AgentMCPIntegration]:
        id = await self.db.find_one(self.integrations_collection, {"integration_id": integration_id})
        if not id: return None
        for k in ["created_at", "updated_at", "last_used"]:
            if k in id and id[k] and isinstance(id[k], str): id[k] = datetime.fromisoformat(id[k])
        return AgentMCPIntegration(**id)
    
    async def update_integration(self, integration_id: str, integration_update: AgentMCPIntegrationUpdate) -> Optional[AgentMCPIntegration]:
        ud = integration_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_integration(integration_id) if await self.db.update_one(self.integrations_collection, {"integration_id": integration_id}, {"$set": ud}) else None
    
    async def delete_integration(self, integration_id: str) -> bool:
        return await self.db.delete_one(self.integrations_collection, {"integration_id": integration_id})
    
    async def list_integrations(self, agent_id: Optional[str] = None, server_id: Optional[str] = None, limit: int = 100) -> List[AgentMCPIntegration]:
        f = {}
        if agent_id: f["agent_id"] = agent_id
        if server_id: f["server_id"] = server_id
        ids = await self.db.find_many(self.integrations_collection, f, limit=limit)
        integrations = []
        for id in ids:
            for k in ["created_at", "updated_at", "last_used"]:
                if k in id and id[k] and isinstance(id[k], str): id[k] = datetime.fromisoformat(id[k])
            integrations.append(AgentMCPIntegration(**id))
        return integrations
    
    # Notifications
    async def create_notification(self, notification_type: str, severity: str, title: str, message: str, server_id: Optional[str] = None):
        notification_id = f"ntf_{uuid.uuid4().hex[:12]}"
        notification = MCPNotification(notification_id=notification_id, notification_type=notification_type, severity=severity, title=title, message=message, server_id=server_id, created_at=datetime.utcnow())
        nd = notification.model_dump()
        for k in ["created_at", "read_at"]:
            if nd.get(k): nd[k] = nd[k].isoformat()
        await self.db.insert_one(self.notifications_collection, nd)
    
    async def list_notifications(self, read: Optional[bool] = None, limit: int = 100) -> List[MCPNotification]:
        f = {}
        if read is not None: f["read"] = read
        nds = await self.db.find_many(self.notifications_collection, f, limit=limit)
        notifications = []
        for nd in nds:
            for k in ["created_at", "read_at"]:
                if k in nd and nd[k] and isinstance(nd[k], str): nd[k] = datetime.fromisoformat(nd[k])
            notifications.append(MCPNotification(**nd))
        return notifications
    
    # Server Catalog
    async def get_server_catalog(self, limit: int = 100) -> List[ServerCatalogEntry]:
        servers = await self.list_servers(limit=limit)
        catalog = []
        for server in servers:
            entry = ServerCatalogEntry(
                server_id=server.server_id, name=server.name, description=server.description,
                version=server.version, status=server.status, health_status=server.health_status,
                tool_count=server.tool_count, capabilities=server.capabilities, tags=server.tags,
                discovery_source=server.discovery_source, last_health_check=server.last_health_check,
                created_at=server.created_at
            )
            catalog.append(entry)
        return catalog
    
    # Statistics
    async def get_stats(self) -> MCPHubStats:
        all_servers = await self.list_servers(limit=10000)
        all_tools = await self.list_tools(limit=10000)
        all_integrations = await self.list_integrations(limit=10000)
        all_configs = await self.list_discovery_configs(limit=10000)
        all_checks = await self.list_health_checks(limit=10000)
        all_notifications = await self.list_notifications(limit=10000)
        
        servers_by_source = {s.value: sum(1 for srv in all_servers if srv.discovery_source == s) for s in DiscoverySource}
        servers_by_status = {s.value: sum(1 for srv in all_servers if srv.status == s) for s in MCPServerStatus}
        servers_by_health = {h.value: sum(1 for srv in all_servers if srv.health_status == h) for h in HealthStatus}
        tools_by_type = {t.value: sum(1 for tool in all_tools if tool.tool_type == t) for t in MCPToolType}
        
        recent = sorted(all_servers, key=lambda s: s.discovered_at or s.created_at, reverse=True)[:5]
        recent_data = [{"server_id": s.server_id, "name": s.name, "discovered_at": (s.discovered_at or s.created_at).isoformat()} for s in recent]
        
        top_used = sorted(all_servers, key=lambda s: s.tool_count, reverse=True)[:5]
        top_used_data = [{"server_id": s.server_id, "name": s.name, "tool_count": s.tool_count} for s in top_used]
        
        recent_checks = [c for c in all_checks if c.checked_at > datetime.utcnow() - timedelta(hours=24)]
        health_summary = {"total_checks_24h": len(recent_checks), "healthy": sum(1 for c in recent_checks if c.status == HealthStatus.HEALTHY), "degraded": sum(1 for c in recent_checks if c.status == HealthStatus.DEGRADED), "unhealthy": sum(1 for c in recent_checks if c.status == HealthStatus.UNHEALTHY)}
        
        return MCPHubStats(
            total_servers=len(all_servers), active_servers=sum(1 for s in all_servers if s.status == MCPServerStatus.ACTIVE),
            inactive_servers=sum(1 for s in all_servers if s.status == MCPServerStatus.INACTIVE),
            discovering_servers=sum(1 for s in all_servers if s.status == MCPServerStatus.DISCOVERING),
            error_servers=sum(1 for s in all_servers if s.status == MCPServerStatus.ERROR),
            total_tools=len(all_tools), total_integrations=len(all_integrations),
            total_discovery_configs=len(all_configs), total_health_checks=len(all_checks),
            total_notifications=len(all_notifications), unread_notifications=sum(1 for n in all_notifications if not n.read),
            servers_by_source=servers_by_source, servers_by_status=servers_by_status,
            servers_by_health=servers_by_health, tools_by_type=tools_by_type,
            recent_discoveries=recent_data, top_used_servers=top_used_data,
            health_check_summary=health_summary
        )

