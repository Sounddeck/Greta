"""
Event Bus to WebSocket Bridge
Connects the internal event bus to WebSocket for real-time client updates
"""

import asyncio
from typing import Dict, Any, Optional
from loguru import logger

from services.event_bus import event_bus
from services.websocket_manager import websocket_manager


class EventWebSocketBridge:
    """
    Bridges event bus events to WebSocket connections
    
    Features:
    - Automatic event-to-WebSocket forwarding
    - Topic mapping (event types -> WebSocket topics)
    - Filtering and transformation
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the bridge"""
        self.active = False
        self.events_forwarded = 0
        self.topic_mappings: Dict[str, str] = {}
        self._setup_default_mappings()
        
        logger.info("🌉 Event-WebSocket Bridge initialized")
    
    def _setup_default_mappings(self):
        """Setup default event type to WebSocket topic mappings"""
        self.topic_mappings = {
            # Lead Generation Events
            "lead.generated": "lead_updates",
            "lead.scored": "lead_updates",
            "lead.status_changed": "lead_updates",
            "lead.research_completed": "lead_updates",
            "lead.monitoring_started": "lead_updates",
            "lead.monitoring_stopped": "lead_updates",
            
            # Campaign Events
            "campaign.created": "campaign_updates",
            "campaign.updated": "campaign_updates",
            "campaign.stats_updated": "campaign_updates",
            
            # System Events
            "system.status_changed": "system_status",
            "system.error": "error_notifications",
            "system.warning": "error_notifications",
            
            # Agent Events
            "agent.started": "agent_updates",
            "agent.stopped": "agent_updates",
            "agent.error": "agent_updates",
            
            # Workflow Events
            "workflow.started": "workflow_status",
            "workflow.completed": "workflow_status",
            "workflow.failed": "workflow_status",
        }
    
    def add_mapping(self, event_type: str, websocket_topic: str):
        """Add custom event type to WebSocket topic mapping"""
        self.topic_mappings[event_type] = websocket_topic
        logger.info(f"📝 Added mapping: {event_type} -> {websocket_topic}")
    
    async def start(self):
        """Start the bridge by subscribing to all mapped events"""
        if self.active:
            logger.warning("⚠️ Bridge already active")
            return
        
        # Subscribe to all mapped event types
        for event_type in self.topic_mappings.keys():
            event_bus.subscribe(event_type, self._forward_event)
        
        self.active = True
        logger.info("✅ Event-WebSocket Bridge started")
    
    async def stop(self):
        """Stop the bridge by unsubscribing from all events"""
        if not self.active:
            return
        
        # Unsubscribe from all mapped event types
        for event_type in self.topic_mappings.keys():
            event_bus.unsubscribe(event_type, self._forward_event)
        
        self.active = False
        logger.info("🛑 Event-WebSocket Bridge stopped")
    
    async def _forward_event(self, event: Dict[str, Any]):
        """
        Forward event from event bus to WebSocket
        
        Args:
            event: Event data from event bus
        """
        try:
            event_type = event.get("type")
            
            # Get WebSocket topic for this event type
            websocket_topic = self.topic_mappings.get(event_type)
            
            if not websocket_topic:
                # No mapping found, skip
                return
            
            # Transform event for WebSocket
            websocket_message = self._transform_event(event)
            
            # Broadcast to WebSocket topic
            await websocket_manager.broadcast_to_topic(
                websocket_topic,
                websocket_message
            )
            
            self.events_forwarded += 1
            
            logger.debug(f"🌉 Forwarded event: {event_type} -> {websocket_topic}")
            
        except Exception as e:
            logger.error(f"❌ Failed to forward event: {e}", exc_info=True)
    
    def _transform_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform event bus event to WebSocket message format
        
        Args:
            event: Event from event bus
            
        Returns:
            Transformed message for WebSocket
        """
        return {
            "type": event.get("type"),
            "data": event.get("data", {}),
            "source": event.get("source", "event_bus"),
            "event_id": event.get("event_id"),
            "timestamp": event.get("timestamp"),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            "active": self.active,
            "events_forwarded": self.events_forwarded,
            "mapped_event_types": len(self.topic_mappings),
            "mappings": self.topic_mappings,
        }


# Global bridge instance
event_websocket_bridge = EventWebSocketBridge()


# Convenience functions
async def start_bridge():
    """Start the event-WebSocket bridge"""
    await event_websocket_bridge.start()


async def stop_bridge():
    """Stop the event-WebSocket bridge"""
    await event_websocket_bridge.stop()


def add_event_mapping(event_type: str, websocket_topic: str):
    """Add custom event type to WebSocket topic mapping"""
    event_websocket_bridge.add_mapping(event_type, websocket_topic)


# Example usage:
"""
# In main.py startup:
from services.event_websocket_bridge import start_bridge

@app.on_event("startup")
async def startup():
    await start_bridge()

# In a service:
from services.event_bus import publish_event

# Publish event - it will automatically be forwarded to WebSocket
await publish_event(
    "lead.scored",
    {
        "lead_id": "123",
        "score": 85.0,
        "confidence": 0.9
    },
    source="lead_scoring_service"
)

# WebSocket clients subscribed to "lead_updates" will receive:
{
    "type": "lead.scored",
    "data": {
        "lead_id": "123",
        "score": 85.0,
        "confidence": 0.9
    },
    "source": "lead_scoring_service",
    "event_id": "...",
    "timestamp": "2025-10-26T12:00:00",
    "topic": "lead_updates"
}
"""

