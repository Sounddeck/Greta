"""
AutoGen Integration Service
Week 3 - Component 1: AutoGen Framework Integration

Integrates Microsoft AutoGen for multi-agent conversations.
Connects to local LLM via OpenAI-compatible API from Week 1.
Uses AutoGen 0.1.14 API.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field

# AutoGen imports (0.1.14 API)
try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    logger.warning("AutoGen not available, install with: pip install pyautogen==0.1.14")
    AUTOGEN_AVAILABLE = False


class AgentConfig(BaseModel):
    """Agent configuration schema"""
    name: str
    role: str  # e.g., "assistant", "user_proxy", "code_executor", "planner"
    system_message: str
    llm_config: Union[Dict[str, Any], bool] = Field(default_factory=dict)  # Can be dict or False for user_proxy
    max_consecutive_auto_reply: int = 10
    human_input_mode: str = "NEVER"  # NEVER, TERMINATE, ALWAYS
    code_execution_config: Union[Dict[str, Any], bool, None] = None
    capabilities: List[str] = Field(default_factory=list)


class ConversationConfig(BaseModel):
    """Multi-agent conversation configuration"""
    name: str
    agents: List[str]  # List of agent names
    max_rounds: int = 10
    speaker_selection_method: str = "round_robin"  # round_robin, auto, manual
    allow_repeat_speaker: bool = False


class AutoGenService:
    """
    AutoGen Integration Service
    
    Manages AutoGen agents and multi-agent conversations.
    Integrates with local LLM via OpenAI-compatible API.
    """
    
    def __init__(self, llm_base_url: str = "http://localhost:8000/v1"):
        if not AUTOGEN_AVAILABLE:
            raise ImportError("AutoGen not installed. Install with: pip install pyautogen==0.1.14")
        
        self.llm_base_url = llm_base_url
        self.agents: Dict[str, Any] = {}
        self.conversations: Dict[str, Any] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # Default LLM config for local llama.cpp
        self.default_llm_config = {
            "config_list": [{
                "model": "llama-3-8b",  # Model name from Week 1
                "base_url": llm_base_url,
                "api_key": "not-needed",  # Local LLM doesn't need API key
            }],
            "temperature": 0.7,
            "max_tokens": 2000,
        }
        
        logger.info(f"AutoGen Service initialized (v0.1.14)")
        logger.info(f"LLM Base URL: {llm_base_url}")
    
    async def initialize(self):
        """Initialize service and create default agents"""
        await self._create_default_agents()
        logger.info(f"AutoGen Service initialized with {len(self.agent_configs)} agent configs")
    
    async def _create_default_agents(self):
        """Create default agent configurations"""
        default_agents = [
            AgentConfig(
                name="assistant",
                role="assistant",
                system_message="You are a helpful AI assistant. You can help with various tasks including coding, analysis, and problem-solving.",
                llm_config=self.default_llm_config,
                capabilities=["general", "coding", "analysis"]
            ),
            AgentConfig(
                name="coder",
                role="assistant",
                system_message="You are an expert programmer. You write clean, efficient, and well-documented code. You can work with multiple programming languages.",
                llm_config=self.default_llm_config,
                capabilities=["coding", "debugging", "code_review"]
            ),
            AgentConfig(
                name="planner",
                role="assistant",
                system_message="You are a strategic planner. You break down complex tasks into manageable steps and create detailed execution plans.",
                llm_config=self.default_llm_config,
                capabilities=["planning", "task_breakdown", "strategy"]
            ),
            AgentConfig(
                name="critic",
                role="assistant",
                system_message="You are a critical reviewer. You analyze solutions, identify potential issues, and suggest improvements.",
                llm_config=self.default_llm_config,
                capabilities=["review", "critique", "quality_assurance"]
            ),
            AgentConfig(
                name="user_proxy",
                role="user_proxy",
                system_message="You are a user proxy that can execute code and provide feedback.",
                llm_config=False,  # User proxy doesn't need LLM
                human_input_mode="NEVER",
                code_execution_config={"work_dir": "coding", "use_docker": False},
                capabilities=["code_execution", "user_interaction"]
            )
        ]
        
        for config in default_agents:
            self.agent_configs[config.name] = config
            logger.info(f"Registered agent config: {config.name} ({config.role})")
    
    async def create_agent(self, config: AgentConfig) -> Any:
        """Create an AutoGen agent from configuration"""
        try:
            if config.role == "user_proxy":
                agent = UserProxyAgent(
                    name=config.name,
                    system_message=config.system_message,
                    human_input_mode=config.human_input_mode,
                    max_consecutive_auto_reply=config.max_consecutive_auto_reply,
                    code_execution_config=config.code_execution_config or False
                )
            else:
                agent = AssistantAgent(
                    name=config.name,
                    system_message=config.system_message,
                    llm_config=config.llm_config,
                    max_consecutive_auto_reply=config.max_consecutive_auto_reply
                )
            
            self.agents[config.name] = agent
            logger.success(f"Created agent: {config.name}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent {config.name}: {e}")
            return None
    
    async def get_agent(self, name: str) -> Optional[Any]:
        """Get an existing agent by name"""
        if name not in self.agents:
            # Try to create from config
            if name in self.agent_configs:
                return await self.create_agent(self.agent_configs[name])
        return self.agents.get(name)
    
    async def list_agents(self) -> List[str]:
        """List all available agents"""
        return list(self.agent_configs.keys())
    
    async def register_agent_config(self, config: AgentConfig):
        """Register a new agent configuration"""
        self.agent_configs[config.name] = config
        logger.info(f"Registered agent config: {config.name}")
    
    async def start_conversation(
        self,
        conversation_id: str,
        agent_names: List[str],
        initial_message: str,
        max_rounds: int = 10
    ) -> Dict[str, Any]:
        """Start a multi-agent conversation"""
        try:
            # Get or create agents
            agents = []
            for name in agent_names:
                agent = await self.get_agent(name)
                if agent:
                    agents.append(agent)
                else:
                    logger.warning(f"Agent not found: {name}")
            
            if len(agents) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 agents for conversation"
                }
            
            # Create group chat (0.1.14 API)
            group_chat = GroupChat(
                agents=agents,
                messages=[],
                max_round=max_rounds
            )
            
            # Create manager
            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=self.default_llm_config
            )
            
            # Store conversation
            self.conversations[conversation_id] = {
                "group_chat": group_chat,
                "manager": manager,
                "agents": agent_names,
                "status": "active",
                "messages": []
            }
            
            logger.info(f"Started conversation {conversation_id} with {len(agents)} agents")
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "agents": agent_names,
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get status of a conversation"""
        if conversation_id not in self.conversations:
            return {"error": "Conversation not found"}
        
        conv = self.conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "agents": conv["agents"],
            "status": conv["status"],
            "message_count": len(conv.get("messages", []))
        }
    
    async def stop_conversation(self, conversation_id: str) -> bool:
        """Stop a conversation"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["status"] = "stopped"
            logger.info(f"Stopped conversation: {conversation_id}")
            return True
        return False
    
    async def clear_agents(self):
        """Clear all agents"""
        self.agents.clear()
        logger.info("Cleared all agents")
    
    async def clear_conversations(self):
        """Clear all conversations"""
        self.conversations.clear()
        logger.info("Cleared all conversations")


# Global instance
autogen_service = AutoGenService()
