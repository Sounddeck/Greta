"""
GRETA WebSocket Manager
Real-time communication system for live updates across all GRETA capabilities
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid

logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: str
    source: str
    target: Optional[str] = None

class ConnectionInfo(BaseModel):
    connection_id: str
    client_ip: str
    connected_at: str
    subscriptions: Set[str]
    last_activity: str

class WebSocketManager:
    """Manages WebSocket connections and real-time updates"""
    
    def __init__(self):
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        
        # Subscription management
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self.connection_subscriptions: Dict[str, Set[str]] = {}  # connection_id -> topics
        
        # Message queues for offline clients
        self.message_queues: Dict[str, List[WebSocketMessage]] = {}
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.start_time = datetime.now()
        
    async def connect(self, websocket: WebSocket, client_ip: str = "unknown") -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        # Store connection info
        self.connection_info[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            client_ip=client_ip,
            connected_at=datetime.now().isoformat(),
            subscriptions=set(),
            last_activity=datetime.now().isoformat()
        )
        
        self.connection_subscriptions[connection_id] = set()
        self.total_connections += 1
        
        logger.info(f"✅ WebSocket connected: {connection_id} from {client_ip}")
        
        # Send welcome message
        await self.send_to_connection(connection_id, {
            "type": "connection_established",
            "data": {
                "connection_id": connection_id,
                "server_time": datetime.now().isoformat(),
                "available_topics": self._get_available_topics()
            },
            "source": "websocket_manager"
        })
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id in self.active_connections:
            # Remove from all subscriptions
            if connection_id in self.connection_subscriptions:
                for topic in self.connection_subscriptions[connection_id]:
                    if topic in self.subscriptions:
                        self.subscriptions[topic].discard(connection_id)
                        if not self.subscriptions[topic]:
                            del self.subscriptions[topic]
                
                del self.connection_subscriptions[connection_id]
            
            # Clean up connection
            del self.active_connections[connection_id]
            if connection_id in self.connection_info:
                del self.connection_info[connection_id]
            
            logger.info(f"🔌 WebSocket disconnected: {connection_id}")
    
    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to a topic"""
        if connection_id not in self.active_connections:
            return False
        
        # Add to subscriptions
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        
        self.subscriptions[topic].add(connection_id)
        self.connection_subscriptions[connection_id].add(topic)
        
        # Update connection info
        if connection_id in self.connection_info:
            self.connection_info[connection_id].subscriptions.add(topic)
        
        logger.info(f"📡 Connection {connection_id} subscribed to {topic}")
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "subscription_confirmed",
            "data": {
                "topic": topic,
                "subscriber_count": len(self.subscriptions[topic])
            },
            "source": "websocket_manager"
        })
        
        return True
    
    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe connection from a topic"""
        if connection_id not in self.active_connections:
            return False
        
        # Remove from subscriptions
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(connection_id)
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
        
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].discard(topic)
        
        # Update connection info
        if connection_id in self.connection_info:
            self.connection_info[connection_id].subscriptions.discard(topic)
        
        logger.info(f"📡 Connection {connection_id} unsubscribed from {topic}")
        return True
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection"""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
            
            await websocket.send_text(json.dumps(message))
            
            # Update activity
            if connection_id in self.connection_info:
                self.connection_info[connection_id].last_activity = datetime.now().isoformat()
            
            self.total_messages_sent += 1
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all subscribers of a topic"""
        if topic not in self.subscriptions:
            return 0
        
        subscribers = self.subscriptions[topic].copy()
        successful_sends = 0
        
        # Add topic to message
        message["topic"] = topic
        
        for connection_id in subscribers:
            success = await self.send_to_connection(connection_id, message)
            if success:
                successful_sends += 1
        
        logger.info(f"📢 Broadcasted to topic '{topic}': {successful_sends}/{len(subscribers)} successful")
        return successful_sends
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all active connections"""
        connections = list(self.active_connections.keys())
        successful_sends = 0
        
        for connection_id in connections:
            success = await self.send_to_connection(connection_id, message)
            if success:
                successful_sends += 1
        
        logger.info(f"📢 Broadcasted to all: {successful_sends}/{len(connections)} successful")
        return successful_sends
    
    def _get_available_topics(self) -> List[str]:
        """Get list of available topics"""
        return [
            "system_status",
            "agent_updates",
            "llm_responses",
            "voice_synthesis",
            "memory_updates",
            "training_progress",
            "osint_results",
            "reasoning_updates",
            "workflow_status",
            "orchestrator_commands",
            "performance_metrics",
            "error_notifications",
            # Lead Generation Topics (Phase 3)
            "lead_updates",
            "campaign_updates",
        ]
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "active_topics": len(self.subscriptions),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "topics": {
                topic: len(subscribers) 
                for topic, subscribers in self.subscriptions.items()
            }
        }
    
    async def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about specific connection"""
        return self.connection_info.get(connection_id)
    
    async def handle_client_message(self, connection_id: str, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                topic = data.get("topic")
                if topic:
                    await self.subscribe(connection_id, topic)
            
            elif message_type == "unsubscribe":
                topic = data.get("topic")
                if topic:
                    await self.unsubscribe(connection_id, topic)
            
            elif message_type == "ping":
                await self.send_to_connection(connection_id, {
                    "type": "pong",
                    "data": {"server_time": datetime.now().isoformat()},
                    "source": "websocket_manager"
                })
            
            elif message_type == "get_stats":
                stats = await self.get_connection_stats()
                await self.send_to_connection(connection_id, {
                    "type": "stats_response",
                    "data": stats,
                    "source": "websocket_manager"
                })
            
            else:
                logger.warning(f"Unknown message type from {connection_id}: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {connection_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Convenience functions for other services
async def notify_system_update(data: Dict[str, Any]):
    """Notify all clients of system updates"""
    await websocket_manager.broadcast_to_topic("system_status", {
        "type": "system_update",
        "data": data,
        "source": "system"
    })

async def notify_agent_update(agent_name: str, status: str, details: Dict[str, Any]):
    """Notify clients of agent updates"""
    await websocket_manager.broadcast_to_topic("agent_updates", {
        "type": "agent_update",
        "data": {
            "agent_name": agent_name,
            "status": status,
            "details": details
        },
        "source": "agent_system"
    })

async def notify_llm_response(model: str, response_time: float, tokens: int):
    """Notify clients of LLM responses"""
    await websocket_manager.broadcast_to_topic("llm_responses", {
        "type": "llm_response",
        "data": {
            "model": model,
            "response_time": response_time,
            "tokens_generated": tokens
        },
        "source": "llm_service"
    })

async def notify_training_progress(lesson_id: str, progress: float, user_id: str):
    """Notify clients of training progress"""
    await websocket_manager.broadcast_to_topic("training_progress", {
        "type": "training_progress",
        "data": {
            "lesson_id": lesson_id,
            "progress": progress,
            "user_id": user_id
        },
        "source": "training_service"
    })

async def notify_error(error_type: str, message: str, source: str):
    """Notify clients of errors"""
    await websocket_manager.broadcast_to_topic("error_notifications", {
        "type": "error_notification",
        "data": {
            "error_type": error_type,
            "message": message,
            "source": source
        },
        "source": "error_handler"
    })
