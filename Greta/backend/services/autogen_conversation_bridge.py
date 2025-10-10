"""
AutoGen Conversation Bridge
Week 4 - Component 2: AutoGen Integration

Bridges AutoGen multi-agent conversations with conversation storage.
Captures GroupChat sessions and stores them in MongoDB.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
from pydantic import BaseModel

# Import from Week 3
try:
    from backend.services.autogen_service import autogen_service, AgentConfig
    AUTOGEN_AVAILABLE = True
except ImportError:
    logger.warning("AutoGen service not available")
    AUTOGEN_AVAILABLE = False

# Import from Week 4
from backend.services.conversation_service import conversation_service, Message


class AutoGenConversationBridge:
    """
    Bridge between AutoGen and Conversation Storage
    
    Captures multi-agent conversations and stores them in MongoDB.
    """
    
    def __init__(self):
        self.active_bridges: Dict[str, Dict[str, Any]] = {}
        logger.info("AutoGen Conversation Bridge initialized")
    
    async def start_tracked_conversation(
        self,
        conversation_id: str,
        agent_names: List[str],
        initial_message: str,
        title: Optional[str] = None,
        max_rounds: int = 10,
        context_window: int = 4096
    ) -> Dict[str, Any]:
        """
        Start an AutoGen conversation with tracking.
        All messages are automatically saved to conversation storage.
        """
        if not AUTOGEN_AVAILABLE:
            return {
                "success": False,
                "error": "AutoGen not available"
            }
        
        # Create conversation in storage
        await conversation_service.create_conversation(
            conversation_id=conversation_id,
            title=title or f"AutoGen: {', '.join(agent_names)}",
            participants=agent_names,
            context_window=context_window,
            metadata={
                "type": "autogen_groupchat",
                "max_rounds": max_rounds,
                "initial_message": initial_message
            }
        )
        
        # Add initial message
        await conversation_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=initial_message
        )
        
        # Start AutoGen conversation
        result = await autogen_service.start_conversation(
            conversation_id=conversation_id,
            agent_names=agent_names,
            initial_message=initial_message,
            max_rounds=max_rounds
        )
        
        if result.get("success"):
            # Track this conversation
            self.active_bridges[conversation_id] = {
                "agent_names": agent_names,
                "max_rounds": max_rounds,
                "started_at": result.get("started_at")
            }
            
            logger.success(f"Started tracked AutoGen conversation: {conversation_id}")
        
        return result
    
    async def capture_agent_message(
        self,
        conversation_id: str,
        agent_name: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Capture a message from an AutoGen agent"""
        return await conversation_service.add_message(
            conversation_id=conversation_id,
            role=agent_name,
            content=message,
            metadata=metadata or {"source": "autogen"}
        )
    
    async def get_conversation_for_autogen(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history in AutoGen-compatible format.
        Returns list of {"role": "...", "content": "..."} dicts.
        """
        messages = await conversation_service.get_context_window(
            conversation_id=conversation_id,
            max_messages=max_messages
        )
        
        # Convert to AutoGen format
        autogen_messages = []
        for msg in messages:
            autogen_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return autogen_messages
    
    async def stop_tracked_conversation(
        self,
        conversation_id: str
    ) -> bool:
        """Stop tracking an AutoGen conversation"""
        # Stop AutoGen conversation
        if AUTOGEN_AVAILABLE:
            await autogen_service.stop_conversation(conversation_id)
        
        # Update conversation status
        await conversation_service.update_conversation(
            conversation_id=conversation_id,
            status="completed"
        )
        
        # Remove from active bridges
        self.active_bridges.pop(conversation_id, None)
        
        logger.info(f"Stopped tracked conversation: {conversation_id}")
        return True
    
    async def get_active_conversations(self) -> List[str]:
        """Get list of active tracked conversations"""
        return list(self.active_bridges.keys())


# Global instance
autogen_bridge = AutoGenConversationBridge()
