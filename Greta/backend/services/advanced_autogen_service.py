"""
Advanced AutoGen Service - Week 7-8 Implementation

Extends Week 3 AutoGen implementation with:
1. Advanced Multi-Agent Orchestration
2. Custom Agent Types
3. Tool Integration for Agents
4. Agent Memory and Context Management
5. RAG Integration with Agents

Builds on AutoGen 0.1.14 from Week 3
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

# AutoGen imports
try:
    from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

    AUTOGEN_AVAILABLE = True
except ImportError:
    logger.warning("AutoGen not available")
    AUTOGEN_AVAILABLE = False

# Import existing services
try:
    from backend.services.autogen_service import AutoGenService, AgentConfig
    from backend.services.conversation_service import conversation_service
    from backend.services.rag_service import rag_service
except ImportError:
    logger.warning("Could not import existing services")
    AutoGenService = None
    conversation_service = None
    rag_service = None


# ============================================================================
# Advanced Agent Types and Schemas
# ============================================================================

class ToolDefinition(BaseModel):
    """Tool definition for agent use"""

    name: str
    description: str
    function: str  # Function name to call
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_capabilities: List[str] = Field(default_factory=list)


class AgentMemory(BaseModel):
    """Agent memory schema"""

    agent_name: str
    memory_type: str  # short_term, long_term, episodic
    content: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    importance: float = 0.5  # 0-1 scale
    tags: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Agent state schema"""

    agent_name: str
    current_task: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    memory_ids: List[str] = Field(default_factory=list)
    tool_usage_count: Dict[str, int] = Field(default_factory=dict)
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CustomAgentConfig(AgentConfig):
    """Extended agent configuration with advanced features"""

    tools: List[str] = Field(default_factory=list)  # Tool names this agent can use
    memory_enabled: bool = True
    rag_enabled: bool = False
    rag_collection: Optional[str] = None
    personality_traits: Dict[str, float] = Field(default_factory=dict)
    learning_rate: float = 0.1  # For adaptive behavior


# ============================================================================
# Tool Registry and Execution
# ============================================================================

class ToolRegistry:
    """Registry for agent tools"""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        # Web search tool
        self.register_tool(
            ToolDefinition(
                name="web_search",
                description="Search the web for information",
                function="web_search_function",
                parameters={"query": "string", "num_results": "int"},
                required_capabilities=["search"],
            ),
            self._web_search_stub,
        )

        # Code execution tool
        self.register_tool(
            ToolDefinition(
                name="execute_code",
                description="Execute Python code safely",
                function="execute_code_function",
                parameters={"code": "string", "language": "string"},
                required_capabilities=["coding"],
            ),
            self._execute_code_stub,
        )

        # File operations tool
        self.register_tool(
            ToolDefinition(
                name="file_operations",
                description="Read, write, or list files",
                function="file_operations_function",
                parameters={"operation": "string", "path": "string", "content": "string"},
                required_capabilities=["file_access"],
            ),
            self._file_operations_stub,
        )

        # RAG query tool
        self.register_tool(
            ToolDefinition(
                name="query_knowledge",
                description="Query the RAG knowledge base",
                function="query_knowledge_function",
                parameters={"query": "string", "n_results": "int"},
                required_capabilities=["rag"],
            ),
            self._query_knowledge,
        )

    def register_tool(self, tool: ToolDefinition, function: Callable):
        """Register a new tool"""
        self.tools[tool.name] = tool
        self.tool_functions[tool.name] = function
        logger.info(f"Registered tool: {tool.name}")

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        try:
            function = self.tool_functions[tool_name]
            result = await function(**parameters)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"success": False, "error": str(e)}

    # Tool stub implementations
    async def _web_search_stub(self, query: str, num_results: int = 5) -> List[Dict]:
        """Web search stub (to be implemented with actual search API)"""
        logger.info(f"Web search: {query} (stub)")
        return [
            {
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a search result snippet for {query}",
            }
            for i in range(num_results)
        ]

    async def _execute_code_stub(self, code: str, language: str = "python") -> Dict:
        """Code execution stub (to be implemented with safe sandbox)"""
        logger.info(f"Code execution: {language} (stub)")
        return {
            "output": f"Code execution result (stub): {code[:50]}...",
            "success": True,
        }

    async def _file_operations_stub(
        self, operation: str, path: str, content: str = ""
    ) -> Dict:
        """File operations stub"""
        logger.info(f"File operation: {operation} on {path} (stub)")
        return {"success": True, "message": f"File operation {operation} completed (stub)"}

    async def _query_knowledge(self, query: str, n_results: int = 3) -> Dict:
        """Query RAG knowledge base"""
        if rag_service:
            try:
                results = await rag_service.search(query, n_results=n_results)
                return {
                    "results": [
                        {
                            "content": r.content,
                            "score": r.score,
                            "metadata": r.metadata,
                        }
                        for r in results
                    ]
                }
            except Exception as e:
                logger.error(f"RAG query error: {e}")
                return {"error": str(e)}
        else:
            return {"error": "RAG service not available"}


# ============================================================================
# Agent Memory Manager
# ============================================================================

class AgentMemoryManager:
    """Manages agent memory across conversations"""

    def __init__(self):
        self.memories: Dict[str, List[AgentMemory]] = {}
        self.agent_states: Dict[str, AgentState] = {}

    async def store_memory(
        self,
        agent_name: str,
        memory_type: str,
        content: Dict[str, Any],
        importance: float = 0.5,
        tags: List[str] = None,
    ) -> str:
        """Store a memory for an agent"""
        memory = AgentMemory(
            agent_name=agent_name,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags or [],
        )

        if agent_name not in self.memories:
            self.memories[agent_name] = []

        self.memories[agent_name].append(memory)

        # Update agent state
        if agent_name in self.agent_states:
            self.agent_states[agent_name].memory_ids.append(memory.timestamp)

        logger.info(f"Stored {memory_type} memory for {agent_name}")
        return memory.timestamp

    async def retrieve_memories(
        self,
        agent_name: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[AgentMemory]:
        """Retrieve memories for an agent"""
        if agent_name not in self.memories:
            return []

        memories = self.memories[agent_name]

        # Filter by type
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]

        # Filter by importance
        memories = [m for m in memories if m.importance >= min_importance]

        # Sort by importance and timestamp
        memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)

        return memories[:limit]

    async def update_agent_state(
        self, agent_name: str, updates: Dict[str, Any]
    ) -> AgentState:
        """Update agent state"""
        if agent_name not in self.agent_states:
            self.agent_states[agent_name] = AgentState(agent_name=agent_name)

        state = self.agent_states[agent_name]

        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

        state.last_updated = datetime.utcnow().isoformat()

        logger.info(f"Updated state for {agent_name}")
        return state

    async def get_agent_state(self, agent_name: str) -> Optional[AgentState]:
        """Get agent state"""
        return self.agent_states.get(agent_name)


# ============================================================================
# Advanced AutoGen Service
# ============================================================================

class AdvancedAutoGenService:
    """
    Advanced AutoGen Service - Week 7-8

    Features:
    - Advanced multi-agent orchestration
    - Custom agent types with specialized capabilities
    - Tool integration and execution
    - Agent memory and state management
    - RAG integration for knowledge-aware agents
    """

    def __init__(self, base_service: Optional[AutoGenService] = None):
        if not AUTOGEN_AVAILABLE:
            raise ImportError("AutoGen not installed")

        self.base_service = base_service
        self.tool_registry = ToolRegistry()
        self.memory_manager = AgentMemoryManager()
        self.custom_agents: Dict[str, Any] = {}
        self.agent_configs: Dict[str, CustomAgentConfig] = {}

        logger.info("Advanced AutoGen Service initialized (Week 7-8)")

    async def initialize(self):
        """Initialize service"""
        await self._create_specialized_agents()
        logger.info(
            f"Advanced AutoGen Service initialized with {len(self.agent_configs)} specialized agents"
        )

    async def _create_specialized_agents(self):
        """Create specialized agent configurations"""
        # Research Agent - with RAG capabilities
        self.agent_configs["researcher"] = CustomAgentConfig(
            name="researcher",
            role="assistant",
            system_message=(
                "You are a research specialist. You excel at finding information, "
                "analyzing data, and synthesizing knowledge from multiple sources. "
                "You can query knowledge bases and search the web."
            ),
            llm_config=self._get_llm_config(),
            capabilities=["research", "analysis", "rag", "search"],
            tools=["web_search", "query_knowledge"],
            rag_enabled=True,
            rag_collection="greta_documents",
            personality_traits={"curiosity": 0.9, "thoroughness": 0.8},
        )

        # Code Architect - advanced coding with file operations
        self.agent_configs["architect"] = CustomAgentConfig(
            name="architect",
            role="assistant",
            system_message=(
                "You are a software architect. You design systems, write code, "
                "and can execute and test code. You have access to file operations "
                "and code execution tools."
            ),
            llm_config=self._get_llm_config(),
            capabilities=["coding", "architecture", "file_access", "execution"],
            tools=["execute_code", "file_operations"],
            personality_traits={"precision": 0.9, "creativity": 0.7},
        )

        # Knowledge Curator - RAG-focused agent
        self.agent_configs["curator"] = CustomAgentConfig(
            name="curator",
            role="assistant",
            system_message=(
                "You are a knowledge curator. You organize, categorize, and retrieve "
                "information from the knowledge base. You help maintain and query "
                "the document repository."
            ),
            llm_config=self._get_llm_config(),
            capabilities=["rag", "organization", "search"],
            tools=["query_knowledge"],
            rag_enabled=True,
            rag_collection="greta_documents",
            personality_traits={"organization": 0.9, "detail_oriented": 0.8},
        )

        # Orchestrator - manages multi-agent workflows
        self.agent_configs["orchestrator"] = CustomAgentConfig(
            name="orchestrator",
            role="assistant",
            system_message=(
                "You are a workflow orchestrator. You coordinate multiple agents, "
                "delegate tasks, and ensure efficient collaboration. You understand "
                "each agent's capabilities and assign work accordingly."
            ),
            llm_config=self._get_llm_config(),
            capabilities=["coordination", "planning", "delegation"],
            tools=[],
            personality_traits={"leadership": 0.9, "efficiency": 0.8},
        )

        # Critic - quality assurance and validation
        self.agent_configs["critic"] = CustomAgentConfig(
            name="critic",
            role="assistant",
            system_message=(
                "You are a quality assurance specialist. You review work, identify "
                "issues, suggest improvements, and ensure high standards. You provide "
                "constructive feedback."
            ),
            llm_config=self._get_llm_config(),
            capabilities=["review", "validation", "quality_assurance"],
            tools=[],
            personality_traits={"critical_thinking": 0.9, "constructiveness": 0.8},
        )

        logger.info(f"Created {len(self.agent_configs)} specialized agent configs")

    def _get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        if self.base_service:
            return self.base_service.default_llm_config
        else:
            return {
                "config_list": [
                    {
                        "model": "llama-3-8b",
                        "base_url": "http://localhost:8000/v1",
                        "api_key": "not-needed",
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
            }

    async def create_agent(
        self, config: CustomAgentConfig, enable_tools: bool = True
    ) -> Any:
        """Create an AutoGen agent with advanced features"""
        # Create base agent
        if config.role == "user_proxy":
            agent = UserProxyAgent(
                name=config.name,
                system_message=config.system_message,
                human_input_mode=config.human_input_mode,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
                code_execution_config=config.code_execution_config or False,
            )
        else:
            agent = AssistantAgent(
                name=config.name,
                system_message=config.system_message,
                llm_config=config.llm_config,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
            )

        # Enhance agent with tools if enabled
        if enable_tools and config.tools:
            agent = await self._enhance_agent_with_tools(agent, config)

        # Initialize agent memory
        if config.memory_enabled:
            await self.memory_manager.update_agent_state(
                config.name, {"current_task": None, "context": {}}
            )

        self.custom_agents[config.name] = agent
        logger.info(f"Created advanced agent: {config.name} with {len(config.tools)} tools")

        return agent

    async def _enhance_agent_with_tools(self, agent: Any, config: CustomAgentConfig) -> Any:
        """Enhance agent with tool capabilities"""
        # Store original reply function
        original_generate_reply = agent.generate_reply

        # Create enhanced reply function
        async def enhanced_generate_reply(messages, sender, config_override=None):
            """Enhanced reply with tool support"""
            # Check if message requests tool use
            last_message = messages[-1] if messages else {}
            content = last_message.get("content", "")

            # Simple tool detection (can be enhanced with better parsing)
            tool_result = None
            for tool_name in config.tools:
                if tool_name in content.lower():
                    # Extract parameters (simplified - can be enhanced)
                    params = self._extract_tool_params(content, tool_name)
                    tool_result = await self.tool_registry.execute_tool(
                        tool_name, params
                    )

                    # Update tool usage count
                    state = await self.memory_manager.get_agent_state(config.name)
                    if state:
                        if tool_name not in state.tool_usage_count:
                            state.tool_usage_count[tool_name] = 0
                        state.tool_usage_count[tool_name] += 1
                        await self.memory_manager.update_agent_state(
                            config.name, {"tool_usage_count": state.tool_usage_count}
                        )

            # If tool was used, add result to context
            if tool_result:
                # Store in memory
                await self.memory_manager.store_memory(
                    config.name,
                    "episodic",
                    {"tool_use": tool_name, "result": tool_result},
                    importance=0.7,
                )

                # Add tool result to messages
                tool_message = {
                    "role": "system",
                    "content": f"Tool result: {json.dumps(tool_result)}",
                }
                messages.append(tool_message)

            # Call original reply function
            return original_generate_reply(messages, sender, config_override)

        # Replace reply function
        agent.generate_reply = enhanced_generate_reply

        return agent

    def _extract_tool_params(self, content: str, tool_name: str) -> Dict[str, Any]:
        """Extract tool parameters from message content (simplified)"""
        # This is a simplified implementation
        # In production, use proper NLP or structured format
        params = {}

        if tool_name == "web_search":
            # Extract query
            if "search for" in content.lower():
                query_start = content.lower().find("search for") + len("search for")
                query = content[query_start:].strip().split(".")[0]
                params["query"] = query
                params["num_results"] = 5

        elif tool_name == "query_knowledge":
            # Extract query
            params["query"] = content
            params["n_results"] = 3

        elif tool_name == "execute_code":
            # Extract code (look for code blocks)
            if "```" in content:
                code_start = content.find("```") + 3
                code_end = content.find("```", code_start)
                code = content[code_start:code_end].strip()
                params["code"] = code
                params["language"] = "python"

        return params

    async def create_rag_enhanced_agent(
        self, config: CustomAgentConfig, conversation_id: Optional[str] = None
    ) -> Any:
        """Create an agent with RAG capabilities"""
        if not config.rag_enabled:
            return await self.create_agent(config)

        # Create base agent
        agent = await self.create_agent(config, enable_tools=True)

        # Store original reply function
        original_generate_reply = agent.generate_reply

        # Create RAG-enhanced reply function
        async def rag_enhanced_reply(messages, sender, config_override=None):
            """Reply with RAG context"""
            last_message = messages[-1] if messages else {}
            query = last_message.get("content", "")

            # Query RAG system
            if rag_service and query:
                try:
                    # Search for relevant documents
                    search_results = await rag_service.search(query, n_results=3)

                    if search_results:
                        # Build context from search results
                        context_parts = []
                        for i, result in enumerate(search_results, 1):
                            context_parts.append(
                                f"[Source {i}] {result.content[:200]}..."
                            )

                        context = "\n\n".join(context_parts)

                        # Add RAG context to messages
                        rag_message = {
                            "role": "system",
                            "content": f"Relevant knowledge:\n{context}",
                        }
                        messages.insert(-1, rag_message)

                        # Store RAG usage in memory
                        await self.memory_manager.store_memory(
                            config.name,
                            "episodic",
                            {
                                "rag_query": query,
                                "sources_found": len(search_results),
                            },
                            importance=0.6,
                        )

                        logger.info(
                            f"Agent {config.name} used RAG: {len(search_results)} sources"
                        )

                except Exception as e:
                    logger.error(f"RAG enhancement error: {e}")

            # Call original reply function
            return original_generate_reply(messages, sender, config_override)

        # Replace reply function
        agent.generate_reply = rag_enhanced_reply

        return agent

    async def create_advanced_group_chat(
        self,
        agents: List[Any],
        task: str,
        max_rounds: int = 10,
        speaker_selection: str = "auto",
        enable_memory: bool = True,
    ) -> Dict[str, Any]:
        """Create an advanced group chat with enhanced orchestration"""
        # Create GroupChat
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=speaker_selection,
        )

        # Create manager
        manager = GroupChatManager(groupchat=group_chat)

        # Store initial state
        if enable_memory:
            for agent in agents:
                await self.memory_manager.update_agent_state(
                    agent.name, {"current_task": task}
                )

        logger.info(
            f"Created advanced group chat with {len(agents)} agents, task: {task}"
        )

        return {
            "group_chat": group_chat,
            "manager": manager,
            "task": task,
            "agents": [a.name for a in agents],
        }

    async def run_collaborative_task(
        self,
        task: str,
        agent_names: List[str],
        max_rounds: int = 10,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a collaborative task with multiple agents"""
        # Create agents
        agents = []
        for name in agent_names:
            if name in self.agent_configs:
                config = self.agent_configs[name]
                if config.rag_enabled:
                    agent = await self.create_rag_enhanced_agent(config, conversation_id)
                else:
                    agent = await self.create_agent(config)
                agents.append(agent)

        if not agents:
            return {"error": "No valid agents created"}

        # Create group chat
        chat_info = await self.create_advanced_group_chat(
            agents, task, max_rounds=max_rounds
        )

        # Initiate conversation
        user_proxy = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
        )

        # Start the conversation
        try:
            await user_proxy.initiate_chat(chat_info["manager"], message=task)

            # Collect results
            messages = chat_info["group_chat"].messages

            # Store conversation in memory
            for agent in agents:
                await self.memory_manager.store_memory(
                    agent.name,
                    "long_term",
                    {
                        "task": task,
                        "collaboration": agent_names,
                        "message_count": len(messages),
                    },
                    importance=0.8,
                    tags=["collaboration", "task_completion"],
                )

            # Save to conversation service if available
            if conversation_service and conversation_id:
                for msg in messages:
                    await conversation_service.add_message(
                        conversation_id,
                        role=msg.get("role", "assistant"),
                        content=msg.get("content", ""),
                        metadata={"agent": msg.get("name", "unknown")},
                    )

            return {
                "success": True,
                "task": task,
                "agents": agent_names,
                "messages": messages,
                "rounds": len(messages),
            }

        except Exception as e:
            logger.error(f"Collaborative task error: {e}")
            return {"success": False, "error": str(e)}

    async def get_agent_insights(self, agent_name: str) -> Dict[str, Any]:
        """Get insights about an agent's performance and memory"""
        state = await self.memory_manager.get_agent_state(agent_name)
        memories = await self.memory_manager.retrieve_memories(agent_name, limit=20)

        if not state:
            return {"error": f"Agent {agent_name} not found"}

        return {
            "agent_name": agent_name,
            "current_task": state.current_task,
            "tool_usage": state.tool_usage_count,
            "memory_count": len(memories),
            "recent_memories": [
                {
                    "type": m.memory_type,
                    "content": m.content,
                    "importance": m.importance,
                    "timestamp": m.timestamp,
                }
                for m in memories[:5]
            ],
            "last_updated": state.last_updated,
        }

    async def optimize_agent_context(
        self, agent_name: str, max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Optimize agent context by retrieving relevant memories"""
        # Get agent state
        state = await self.memory_manager.get_agent_state(agent_name)
        if not state:
            return {"error": f"Agent {agent_name} not found"}

        # Retrieve important memories
        memories = await self.memory_manager.retrieve_memories(
            agent_name, min_importance=0.5, limit=10
        )

        # Build optimized context
        context = {
            "current_task": state.current_task,
            "recent_tools": list(state.tool_usage_count.keys())[-5:],
            "key_memories": [
                {"type": m.memory_type, "content": m.content} for m in memories[:5]
            ],
        }

        return {
            "agent_name": agent_name,
            "optimized_context": context,
            "memory_count": len(memories),
        }


# ============================================================================
# Singleton Instance
# ============================================================================

# Create singleton instance (will be initialized in main app)
advanced_autogen_service: Optional[AdvancedAutoGenService] = None


def get_advanced_autogen_service() -> AdvancedAutoGenService:
    """Get or create the advanced AutoGen service instance"""
    global advanced_autogen_service
    if advanced_autogen_service is None:
        advanced_autogen_service = AdvancedAutoGenService()
    return advanced_autogen_service

