"""
Event Bus Service - Central event-driven communication system
Enables loose coupling between CPAS5 modules through publish/subscribe pattern
"""

from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
import asyncio
import logging
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventBus:
    """
    Central event bus for component communication
    
    Features:
    - Publish/subscribe pattern
    - Async event handling
    - Event history tracking
    - Error isolation (one handler failure doesn't affect others)
    - Type-safe event routing
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize event bus

        Args:
            max_history: Maximum number of events to keep in history
        """
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self._lock: Optional[asyncio.Lock] = None

        logger.info("📢 Event Bus initialized")

    async def _get_lock(self) -> asyncio.Lock:
        """Get or create the async lock (lazy initialization)"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> str:
        """
        Publish event to all subscribers
        
        Args:
            event_type: Type of event (e.g., "lead.generated")
            data: Event payload
            source: Optional source identifier
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Store in history
        lock = await self._get_lock()
        async with lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
        
        # Notify subscribers asynchronously
        if event_type in self.subscribers:
            tasks = []
            for callback in self.subscribers[event_type]:
                tasks.append(self._safe_callback(callback, event))
            
            # Execute all callbacks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"📢 Event published: {event_type} (ID: {event_id[:8]}...)")
        return event_id
    
    async def _safe_callback(self, callback: Callable, event: Dict[str, Any]):
        """
        Execute callback with error handling
        
        Args:
            callback: Async callback function
            event: Event data
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                # Run sync callbacks in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, event)
        except Exception as e:
            logger.error(
                f"❌ Event handler error for {event['type']}: {e}",
                exc_info=True
            )
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to event type
        
        Args:
            event_type: Type of event to subscribe to
            callback: Async or sync callback function
        """
        self.subscribers[event_type].append(callback)
        logger.info(f"📝 Subscribed to: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from event type
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
                logger.info(f"📝 Unsubscribed from: {event_type}")
            except ValueError:
                logger.warning(f"⚠️ Callback not found for: {event_type}")
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get event history
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            since: Only return events after this timestamp (optional)
            
        Returns:
            List of events
        """
        events = self.event_history.copy()
        
        # Filter by type
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        
        # Filter by timestamp
        if since:
            since_iso = since.isoformat()
            events = [e for e in events if e["timestamp"] > since_iso]
        
        # Return most recent events
        return events[-limit:]
    
    def get_subscribers_count(self, event_type: Optional[str] = None) -> int:
        """
        Get number of subscribers
        
        Args:
            event_type: Specific event type (optional)
            
        Returns:
            Number of subscribers
        """
        if event_type:
            return len(self.subscribers.get(event_type, []))
        return sum(len(subs) for subs in self.subscribers.values())
    
    def clear_history(self) -> None:
        """Clear event history"""
        self.event_history.clear()
        logger.info("🗑️ Event history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics
        
        Returns:
            Statistics dictionary
        """
        event_types = defaultdict(int)
        for event in self.event_history:
            event_types[event["type"]] += 1
        
        return {
            "total_events": len(self.event_history),
            "total_subscribers": self.get_subscribers_count(),
            "event_types": dict(event_types),
            "subscriber_counts": {
                event_type: len(subs)
                for event_type, subs in self.subscribers.items()
            }
        }


# Global event bus instance
event_bus = EventBus()


# Convenience functions for common operations
async def publish_event(event_type: str, data: Dict[str, Any], source: Optional[str] = None) -> str:
    """Publish an event (convenience function)"""
    return await event_bus.publish(event_type, data, source)


def subscribe_to_event(event_type: str, callback: Callable) -> None:
    """Subscribe to an event (convenience function)"""
    event_bus.subscribe(event_type, callback)


def unsubscribe_from_event(event_type: str, callback: Callable) -> None:
    """Unsubscribe from an event (convenience function)"""
    event_bus.unsubscribe(event_type, callback)


# Example usage:
"""
# In a service:
from services.event_bus import event_bus, publish_event, subscribe_to_event

# Subscribe to events
async def on_lead_generated(event: Dict[str, Any]):
    lead_data = event["data"]
    print(f"New lead: {lead_data}")

subscribe_to_event("lead.generated", on_lead_generated)

# Publish events
await publish_event(
    "lead.generated",
    {
        "company_name": "Acme Corp",
        "email": "contact@acme.com",
        "score": 85
    },
    source="lead_generation_service"
)
"""

