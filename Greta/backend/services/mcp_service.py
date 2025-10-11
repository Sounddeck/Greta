"""
MCP Service - Week 17-18 Implementation

Model Context Protocol (MCP) integration for exposing Greta's capabilities
and connecting to external MCP-compatible tools and services.

MCP enables standardized context sharing and tool interoperability.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Schemas
# ============================================================================


class MCPTool(BaseModel):
    """MCP tool definition"""

    tool_id: str
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    returns: Dict[str, Any] = Field(default_factory=dict)
    category: str = "general"
    source: str = "local"  # local, remote
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPRequest(BaseModel):
    """MCP request"""

    request_id: str
    method: str  # tool.call, context.get, context.set, tool.list
    params: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPResponse(BaseModel):
    """MCP response"""

    request_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPContext(BaseModel):
    """MCP context for sharing between clients/servers"""

    context_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    """MCP server configuration"""

    server_id: str
    name: str
    description: str
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8765
    auth_enabled: bool = True
    rate_limit: int = 100  # requests per minute
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPClientConfig(BaseModel):
    """MCP client configuration"""

    client_id: str
    server_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPStats(BaseModel):
    """MCP statistics"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tools: int = 0
    active_contexts: int = 0
    avg_response_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# MCP Service
# ============================================================================


class MCPService:
    """
    Model Context Protocol (MCP) Service

    Features:
    - MCP server (expose Greta tools via MCP)
    - MCP client (connect to external MCP servers)
    - Tool registry and discovery
    - Context sharing via MCP
    - Authentication and rate limiting
    - Request/response and streaming
    - Tool execution framework
    - Monitoring and logging
    """

    def __init__(
        self,
        server_config: Optional[MCPServerConfig] = None,
    ):
        # Server configuration
        self.server_config = server_config or MCPServerConfig(
            server_id="greta_mcp_server",
            name="Greta MCP Server",
            description="MCP server exposing Greta's capabilities",
        )

        # Tool registry
        self.tools: Dict[str, MCPTool] = {}
        self.tool_handlers: Dict[str, Callable] = {}

        # Context storage
        self.contexts: Dict[str, MCPContext] = {}

        # Client connections
        self.clients: Dict[str, MCPClientConfig] = {}

        # Authentication
        self.api_keys: Dict[str, str] = {}  # api_key -> client_id
        self.rate_limits: Dict[str, List[float]] = {}  # client_id -> timestamps

        # Statistics
        self.stats = MCPStats()
        self.request_times: List[float] = []

        logger.info(f"MCP Service initialized")
        logger.info(f"  Server: {self.server_config.name}")
        logger.info(f"  Server ID: {self.server_config.server_id}")
        logger.info(f"  Auth: {'enabled' if self.server_config.auth_enabled else 'disabled'}")

        # Default tools will be registered on first use
        self._default_tools_registered = False

    # ========================================================================
    # Tool Registry
    # ========================================================================

    async def _ensure_default_tools(self):
        """Ensure default tools are registered"""
        if not self._default_tools_registered:
            await self._register_default_tools()
            self._default_tools_registered = True

    async def register_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "general",
        source: str = "local",
    ) -> bool:
        """Register a tool in the MCP registry"""
        tool = MCPTool(
            tool_id=tool_id,
            name=name,
            description=description,
            parameters=parameters or {},
            category=category,
            source=source,
        )

        self.tools[tool_id] = tool
        self.tool_handlers[tool_id] = handler

        logger.info(f"Registered MCP tool: {tool_id} ({name})")
        return True

    async def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool from the MCP registry"""
        if tool_id in self.tools:
            del self.tools[tool_id]
            del self.tool_handlers[tool_id]
            logger.info(f"Unregistered MCP tool: {tool_id}")
            return True
        return False

    async def list_tools(
        self, category: Optional[str] = None, source: Optional[str] = None
    ) -> List[MCPTool]:
        """List all registered tools"""
        await self._ensure_default_tools()
        tools = list(self.tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if source:
            tools = [t for t in tools if t.source == source]

        return tools

    async def get_tool(self, tool_id: str) -> Optional[MCPTool]:
        """Get tool by ID"""
        return self.tools.get(tool_id)

    async def discover_tools(self, server_url: str) -> List[MCPTool]:
        """
        Discover tools from external MCP server

        In production, implement actual HTTP/WebSocket discovery
        """
        logger.info(f"Discovering tools from: {server_url}")

        # TODO: Implement actual tool discovery via MCP protocol
        # For now, return empty list
        return []

    # ========================================================================
    # Default Tool Registration
    # ========================================================================

    async def _register_default_tools(self):
        """Register default tools from Greta services"""
        # LLM Service tools
        await self.register_tool(
            tool_id="llm.generate",
            name="Generate Text",
            description="Generate text using LLM",
            handler=self._handle_llm_generate,
            parameters={
                "prompt": {"type": "string", "required": True},
                "max_tokens": {"type": "integer", "default": 100},
            },
            category="llm",
        )

        # Conversation Service tools
        await self.register_tool(
            tool_id="conversation.create",
            name="Create Conversation",
            description="Create a new conversation",
            handler=self._handle_conversation_create,
            parameters={
                "title": {"type": "string", "required": True},
                "agent_id": {"type": "string", "required": False},
            },
            category="conversation",
        )

        # RAG Service tools
        await self.register_tool(
            tool_id="rag.search",
            name="Search Documents",
            description="Search RAG documents",
            handler=self._handle_rag_search,
            parameters={
                "query": {"type": "string", "required": True},
                "collection": {"type": "string", "required": True},
                "limit": {"type": "integer", "default": 5},
            },
            category="rag",
        )

        # Agent Template tools
        await self.register_tool(
            tool_id="template.list",
            name="List Agent Templates",
            description="List all agent templates",
            handler=self._handle_template_list,
            parameters={
                "category": {"type": "string", "required": False},
            },
            category="template",
        )

        # Context Management tools
        await self.register_tool(
            tool_id="context.create",
            name="Create Context Window",
            description="Create a new context window",
            handler=self._handle_context_create,
            parameters={
                "window_id": {"type": "string", "required": True},
                "max_tokens": {"type": "integer", "default": 4096},
            },
            category="context",
        )

        # Portability tools
        await self.register_tool(
            tool_id="export.all",
            name="Export All Data",
            description="Export all Greta data",
            handler=self._handle_export_all,
            parameters={
                "encrypt": {"type": "boolean", "default": False},
            },
            category="portability",
        )

        logger.info(f"Registered {len(self.tools)} default MCP tools")

    # ========================================================================
    # Tool Handlers (Default Tools)
    # ========================================================================

    async def _handle_llm_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM generation tool"""
        try:
            from backend.services.llm_service import llm_service

            prompt = params.get("prompt", "")
            max_tokens = params.get("max_tokens", 100)

            # Generate text
            response = await llm_service.generate(
                prompt=prompt, max_tokens=max_tokens
            )

            return {"text": response, "tokens": max_tokens}

        except Exception as e:
            logger.error(f"Error in LLM generate: {e}")
            return {"error": str(e)}

    async def _handle_conversation_create(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle conversation creation tool"""
        try:
            from backend.services.conversation_service import conversation_service

            title = params.get("title", "New Conversation")
            agent_id = params.get("agent_id")

            # Create conversation
            conversation = await conversation_service.create_conversation(
                title=title, agent_id=agent_id
            )

            return {
                "conversation_id": conversation.conversation_id,
                "title": conversation.title,
            }

        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            return {"error": str(e)}

    async def _handle_rag_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG search tool"""
        try:
            from backend.services.rag_service import rag_service

            query = params.get("query", "")
            collection = params.get("collection", "default")
            limit = params.get("limit", 5)

            # Search documents
            results = await rag_service.search(
                query=query, collection_name=collection, limit=limit
            )

            return {"results": results, "count": len(results)}

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return {"error": str(e)}

    async def _handle_template_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle template listing tool"""
        try:
            from backend.services.agent_template_service import agent_template_service

            category = params.get("category")

            # List templates
            templates = await agent_template_service.list_templates(
                category=category, limit=100
            )

            return {
                "templates": [t.model_dump() for t in templates],
                "count": len(templates),
            }

        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return {"error": str(e)}

    async def _handle_context_create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context creation tool"""
        try:
            from backend.services.context_management_service import (
                context_management_service,
            )

            window_id = params.get("window_id", "default")
            max_tokens = params.get("max_tokens", 4096)

            # Create context window
            window = await context_management_service.create_context_window(
                window_id=window_id, max_tokens=max_tokens
            )

            return {
                "window_id": window.window_id,
                "max_tokens": window.max_tokens,
            }

        except Exception as e:
            logger.error(f"Error creating context: {e}")
            return {"error": str(e)}

    async def _handle_export_all(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle export all tool"""
        try:
            from backend.services.portability_service import portability_service

            encrypt = params.get("encrypt", False)

            # Export all data
            export_path = await portability_service.export_all_data(encrypt=encrypt)

            return {"export_path": export_path, "encrypted": encrypt}

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {"error": str(e)}

    # ========================================================================
    # MCP Request Processing
    # ========================================================================

    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process MCP request"""
        await self._ensure_default_tools()

        start_time = time.time()

        logger.info(f"Processing MCP request: {request.method} ({request.request_id})")

        try:
            # Update stats
            self.stats.total_requests += 1

            # Route request based on method
            if request.method == "tool.call":
                result = await self._handle_tool_call(request)
            elif request.method == "tool.list":
                result = await self._handle_tool_list(request)
            elif request.method == "context.get":
                result = await self._handle_context_get(request)
            elif request.method == "context.set":
                result = await self._handle_context_set(request)
            else:
                result = None
                error = f"Unknown method: {request.method}"
                logger.error(error)
                return MCPResponse(
                    request_id=request.request_id, success=False, error=error
                )

            # Update stats
            self.stats.successful_requests += 1
            response_time = time.time() - start_time
            self.request_times.append(response_time)
            self.stats.avg_response_time = sum(self.request_times) / len(
                self.request_times
            )

            return MCPResponse(
                request_id=request.request_id,
                success=True,
                result=result,
                metadata={"response_time": response_time},
            )

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.stats.failed_requests += 1

            return MCPResponse(
                request_id=request.request_id, success=False, error=str(e)
            )

    async def _handle_tool_call(self, request: MCPRequest) -> Any:
        """Handle tool.call method"""
        tool_id = request.params.get("tool_id")
        tool_params = request.params.get("params", {})

        if not tool_id:
            raise ValueError("tool_id is required")

        if tool_id not in self.tool_handlers:
            raise ValueError(f"Tool not found: {tool_id}")

        # Execute tool
        handler = self.tool_handlers[tool_id]
        result = await handler(tool_params)

        return result

    async def _handle_tool_list(self, request: MCPRequest) -> List[Dict[str, Any]]:
        """Handle tool.list method"""
        category = request.params.get("category")
        source = request.params.get("source")

        tools = await self.list_tools(category=category, source=source)
        return [t.model_dump() for t in tools]

    async def _handle_context_get(self, request: MCPRequest) -> Optional[Dict[str, Any]]:
        """Handle context.get method"""
        context_id = request.params.get("context_id")

        if not context_id:
            raise ValueError("context_id is required")

        context = self.contexts.get(context_id)
        return context.model_dump() if context else None

    async def _handle_context_set(self, request: MCPRequest) -> bool:
        """Handle context.set method"""
        context_id = request.params.get("context_id")
        data = request.params.get("data", {})

        if not context_id:
            raise ValueError("context_id is required")

        # Create or update context
        if context_id in self.contexts:
            context = self.contexts[context_id]
            context.data = data
            context.updated_at = datetime.utcnow().isoformat()
        else:
            context = MCPContext(context_id=context_id, data=data)
            self.contexts[context_id] = context

        return True

    # ========================================================================
    # Authentication & Rate Limiting
    # ========================================================================

    def generate_api_key(self, client_id: str) -> str:
        """Generate API key for client"""
        # Generate random API key
        key_data = f"{client_id}_{datetime.utcnow().isoformat()}_{time.time()}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()

        self.api_keys[api_key] = client_id
        logger.info(f"Generated API key for client: {client_id}")

        return api_key

    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key and return client_id"""
        return self.api_keys.get(api_key)

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit"""
        if not self.server_config.auth_enabled:
            return True

        now = time.time()
        window = 60  # 1 minute

        # Get client's request timestamps
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []

        timestamps = self.rate_limits[client_id]

        # Remove old timestamps
        timestamps = [t for t in timestamps if now - t < window]
        self.rate_limits[client_id] = timestamps

        # Check limit
        if len(timestamps) >= self.server_config.rate_limit:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False

        # Add current timestamp
        timestamps.append(now)
        return True

    # ========================================================================
    # MCP Client
    # ========================================================================

    async def connect_to_server(
        self, server_url: str, api_key: Optional[str] = None
    ) -> str:
        """
        Connect to external MCP server

        Returns client_id
        """
        client_id = f"client_{hashlib.md5(server_url.encode()).hexdigest()[:8]}"

        client_config = MCPClientConfig(
            client_id=client_id, server_url=server_url, api_key=api_key
        )

        self.clients[client_id] = client_config

        logger.info(f"Connected to MCP server: {server_url} (client_id={client_id})")
        return client_id

    async def disconnect_from_server(self, client_id: str) -> bool:
        """Disconnect from external MCP server"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Disconnected from MCP server: {client_id}")
            return True
        return False

    async def call_remote_tool(
        self, client_id: str, tool_id: str, params: Dict[str, Any]
    ) -> Any:
        """
        Call tool on remote MCP server

        In production, implement actual HTTP/WebSocket communication
        """
        if client_id not in self.clients:
            raise ValueError(f"Client not found: {client_id}")

        client = self.clients[client_id]

        logger.info(f"Calling remote tool: {tool_id} on {client.server_url}")

        # TODO: Implement actual remote tool call via MCP protocol
        # For now, return placeholder
        return {"status": "not_implemented", "message": "Remote tool call not yet implemented"}

    # ========================================================================
    # Context Sharing
    # ========================================================================

    async def share_context(
        self, context_id: str, target_client: Optional[str] = None
    ) -> bool:
        """
        Share context with MCP client/server

        In production, implement actual context sharing via MCP protocol
        """
        if context_id not in self.contexts:
            logger.error(f"Context not found: {context_id}")
            return False

        context = self.contexts[context_id]

        logger.info(f"Sharing context: {context_id}")

        # TODO: Implement actual context sharing via MCP protocol
        return True

    async def receive_context(
        self, context_id: str, data: Dict[str, Any]
    ) -> bool:
        """Receive context from MCP client/server"""
        context = MCPContext(context_id=context_id, data=data)
        self.contexts[context_id] = context

        logger.info(f"Received context: {context_id}")
        return True

    # ========================================================================
    # Monitoring & Statistics
    # ========================================================================

    async def get_stats(self) -> MCPStats:
        """Get MCP statistics"""
        self.stats.total_tools = len(self.tools)
        self.stats.active_contexts = len(self.contexts)
        return self.stats

    async def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server information"""
        return {
            "server_id": self.server_config.server_id,
            "name": self.server_config.name,
            "description": self.server_config.description,
            "version": self.server_config.version,
            "auth_enabled": self.server_config.auth_enabled,
            "rate_limit": self.server_config.rate_limit,
            "total_tools": len(self.tools),
            "active_contexts": len(self.contexts),
            "total_requests": self.stats.total_requests,
            "success_rate": (
                self.stats.successful_requests / self.stats.total_requests
                if self.stats.total_requests > 0
                else 0.0
            ),
        }

    async def reset_stats(self) -> bool:
        """Reset statistics"""
        self.stats = MCPStats()
        self.request_times = []
        logger.info("MCP statistics reset")
        return True

    # ========================================================================
    # Agent Communication via MCP
    # ========================================================================

    async def send_agent_message(
        self, from_agent: str, to_agent: str, message: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send message from one agent to another via MCP

        Enables agent-to-agent communication using MCP protocol
        """
        logger.info(f"Agent message: {from_agent} -> {to_agent}")

        # Create MCP request for agent communication
        request = MCPRequest(
            request_id=f"agent_msg_{time.time()}",
            method="agent.message",
            params={
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message": message,
            },
            context=context,
        )

        # Process request
        response = await self.process_request(request)

        return response.success

    async def broadcast_to_agents(
        self, from_agent: str, message: str, agent_ids: List[str]
    ) -> Dict[str, bool]:
        """Broadcast message to multiple agents"""
        results = {}

        for agent_id in agent_ids:
            success = await self.send_agent_message(from_agent, agent_id, message)
            results[agent_id] = success

        logger.info(f"Broadcast from {from_agent} to {len(agent_ids)} agents")
        return results

    # ========================================================================
    # Tool Execution Framework
    # ========================================================================

    async def execute_tool(
        self, tool_id: str, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute tool with context

        Unified tool execution interface
        """
        request = MCPRequest(
            request_id=f"exec_{time.time()}",
            method="tool.call",
            params={"tool_id": tool_id, "params": params},
            context=context,
        )

        response = await self.process_request(request)

        if response.success:
            return response.result
        else:
            raise Exception(f"Tool execution failed: {response.error}")

    async def execute_tool_chain(
        self, tools: List[Tuple[str, Dict[str, Any]]], context: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Execute chain of tools sequentially

        Each tool receives output from previous tool in context
        """
        results = []
        current_context = context or {}

        for tool_id, params in tools:
            # Execute tool
            result = await self.execute_tool(tool_id, params, current_context)
            results.append(result)

            # Update context with result
            current_context["previous_result"] = result

        logger.info(f"Executed tool chain: {len(tools)} tools")
        return results


# ============================================================================
# Singleton Instance
# ============================================================================

mcp_service = MCPService()

