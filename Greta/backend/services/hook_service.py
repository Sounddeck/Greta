"""
Hook Service - Week 19-20 Implementation

Extensible event-driven hook system for Greta.

Enables users and plugins to hook into any operation with pre/post hooks,
custom event handlers, middleware, and conditional execution.
"""

import asyncio
import inspect
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# Event Types
# ============================================================================


class EventType(str, Enum):
    """Event types for hook system"""

    # LLM Events (Week 1)
    LLM_GENERATE_PRE = "llm.generate.pre"
    LLM_GENERATE_POST = "llm.generate.post"
    LLM_STREAM_PRE = "llm.stream.pre"
    LLM_STREAM_POST = "llm.stream.post"

    # Model Events (Week 2)
    MODEL_LOAD_PRE = "model.load.pre"
    MODEL_LOAD_POST = "model.load.post"
    MODEL_UNLOAD_PRE = "model.unload.pre"
    MODEL_UNLOAD_POST = "model.unload.post"

    # AutoGen Events (Week 3)
    AGENT_CREATE_PRE = "agent.create.pre"
    AGENT_CREATE_POST = "agent.create.post"
    AGENT_EXECUTE_PRE = "agent.execute.pre"
    AGENT_EXECUTE_POST = "agent.execute.post"

    # Conversation Events (Week 4)
    CONVERSATION_CREATE_PRE = "conversation.create.pre"
    CONVERSATION_CREATE_POST = "conversation.create.post"
    MESSAGE_SEND_PRE = "message.send.pre"
    MESSAGE_SEND_POST = "message.send.post"

    # RAG Events (Week 5-6)
    RAG_INGEST_PRE = "rag.ingest.pre"
    RAG_INGEST_POST = "rag.ingest.post"
    RAG_SEARCH_PRE = "rag.search.pre"
    RAG_SEARCH_POST = "rag.search.post"

    # LoRA Events (Week 9-10)
    LORA_TRAIN_PRE = "lora.train.pre"
    LORA_TRAIN_POST = "lora.train.post"
    LORA_LOAD_PRE = "lora.load.pre"
    LORA_LOAD_POST = "lora.load.post"

    # Template Events (Week 11-12)
    TEMPLATE_CREATE_PRE = "template.create.pre"
    TEMPLATE_CREATE_POST = "template.create.post"
    TEMPLATE_INSTANTIATE_PRE = "template.instantiate.pre"
    TEMPLATE_INSTANTIATE_POST = "template.instantiate.post"

    # Context Events (Week 13-14)
    CONTEXT_CREATE_PRE = "context.create.pre"
    CONTEXT_CREATE_POST = "context.create.post"
    CONTEXT_COMPRESS_PRE = "context.compress.pre"
    CONTEXT_COMPRESS_POST = "context.compress.post"

    # Portability Events (Week 15-16)
    EXPORT_PRE = "export.pre"
    EXPORT_POST = "export.post"
    IMPORT_PRE = "import.pre"
    IMPORT_POST = "import.post"

    # MCP Events (Week 17-18)
    MCP_REQUEST_PRE = "mcp.request.pre"
    MCP_REQUEST_POST = "mcp.request.post"
    MCP_TOOL_CALL_PRE = "mcp.tool.call.pre"
    MCP_TOOL_CALL_POST = "mcp.tool.call.post"

    # Custom Events
    CUSTOM = "custom"


# ============================================================================
# Pydantic Schemas
# ============================================================================


class HookEvent(BaseModel):
    """Event data for hooks"""

    event_type: EventType
    event_id: str = Field(default_factory=lambda: f"evt_{time.time()}")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Hook(BaseModel):
    """Hook definition"""

    hook_id: str
    name: str
    event_type: EventType
    handler: Any  # Callable - can't serialize, so Any
    priority: int = 50  # 0-100, higher = earlier execution
    enabled: bool = True
    condition: Optional[Any] = None  # Optional condition function
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class HookResult(BaseModel):
    """Result of hook execution"""

    hook_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Plugin(BaseModel):
    """Plugin definition"""

    plugin_id: str
    name: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True
    hooks: List[str] = Field(default_factory=list)  # Hook IDs
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HookStats(BaseModel):
    """Hook execution statistics"""

    total_hooks: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    events_by_type: Dict[str, int] = Field(default_factory=dict)


# ============================================================================
# Hook Service
# ============================================================================


class HookService:
    """
    Extensible Event-Driven Hook System

    Features:
    - Event-driven architecture with typed events
    - Hook registration with priorities
    - Pre/post hooks for all operations
    - Hook chaining and composition
    - Plugin system for extensibility
    - Custom event handlers
    - Conditional hook execution
    - Hook middleware
    - Async execution
    - Monitoring and debugging
    """

    def __init__(self):
        # Hook registry
        self.hooks: Dict[str, Hook] = {}  # hook_id -> Hook
        self.hooks_by_event: Dict[EventType, List[str]] = {}  # event_type -> [hook_ids]

        # Plugin registry
        self.plugins: Dict[str, Plugin] = {}

        # Middleware registry
        self.middleware: List[Callable] = []

        # Statistics
        self.stats = HookStats()
        self.execution_times: List[float] = []

        # Event history (for debugging)
        self.event_history: List[HookEvent] = []
        self.max_history = 100

        logger.info("Hook Service initialized")

    # ========================================================================
    # Hook Registration
    # ========================================================================

    async def register_hook(
        self,
        hook_id: str,
        name: str,
        event_type: EventType,
        handler: Callable,
        priority: int = 50,
        condition: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a hook

        Args:
            hook_id: Unique hook identifier
            name: Human-readable hook name
            event_type: Event type to hook into
            handler: Async function to execute (receives HookEvent, returns Any)
            priority: Execution priority (0-100, higher = earlier)
            condition: Optional condition function (receives HookEvent, returns bool)
            metadata: Optional metadata

        Returns:
            True if registered successfully
        """
        hook = Hook(
            hook_id=hook_id,
            name=name,
            event_type=event_type,
            handler=handler,
            priority=priority,
            condition=condition,
            metadata=metadata or {},
        )

        self.hooks[hook_id] = hook

        # Add to event index
        if event_type not in self.hooks_by_event:
            self.hooks_by_event[event_type] = []
        self.hooks_by_event[event_type].append(hook_id)

        # Sort by priority (descending)
        self.hooks_by_event[event_type].sort(
            key=lambda hid: self.hooks[hid].priority, reverse=True
        )

        logger.info(f"Registered hook: {hook_id} ({name}) for {event_type.value}")
        return True

    async def unregister_hook(self, hook_id: str) -> bool:
        """Unregister a hook"""
        if hook_id not in self.hooks:
            return False

        hook = self.hooks[hook_id]
        event_type = hook.event_type

        # Remove from registry
        del self.hooks[hook_id]

        # Remove from event index
        if event_type in self.hooks_by_event:
            self.hooks_by_event[event_type] = [
                hid for hid in self.hooks_by_event[event_type] if hid != hook_id
            ]

        logger.info(f"Unregistered hook: {hook_id}")
        return True

    async def list_hooks(
        self, event_type: Optional[EventType] = None, enabled_only: bool = False
    ) -> List[Hook]:
        """List all hooks"""
        hooks = list(self.hooks.values())

        if event_type:
            hooks = [h for h in hooks if h.event_type == event_type]

        if enabled_only:
            hooks = [h for h in hooks if h.enabled]

        return hooks

    async def get_hook(self, hook_id: str) -> Optional[Hook]:
        """Get hook by ID"""
        return self.hooks.get(hook_id)

    async def enable_hook(self, hook_id: str) -> bool:
        """Enable a hook"""
        if hook_id in self.hooks:
            self.hooks[hook_id].enabled = True
            logger.info(f"Enabled hook: {hook_id}")
            return True
        return False

    async def disable_hook(self, hook_id: str) -> bool:
        """Disable a hook"""
        if hook_id in self.hooks:
            self.hooks[hook_id].enabled = False
            logger.info(f"Disabled hook: {hook_id}")
            return True
        return False

    # ========================================================================
    # Event Emission & Hook Execution
    # ========================================================================

    async def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HookResult]:
        """
        Emit an event and execute all registered hooks

        Args:
            event_type: Type of event
            data: Event data
            context: Event context (shared across hooks)

        Returns:
            List of hook results
        """
        # Create event
        event = HookEvent(
            event_type=event_type,
            data=data or {},
            context=context or {},
        )

        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        # Update stats
        event_type_str = event_type.value
        if event_type_str not in self.stats.events_by_type:
            self.stats.events_by_type[event_type_str] = 0
        self.stats.events_by_type[event_type_str] += 1

        # Get hooks for this event
        hook_ids = self.hooks_by_event.get(event_type, [])
        if not hook_ids:
            logger.debug(f"No hooks registered for event: {event_type.value}")
            return []

        logger.info(f"Emitting event: {event_type.value} ({len(hook_ids)} hooks)")

        # Execute hooks
        results = []
        for hook_id in hook_ids:
            hook = self.hooks[hook_id]

            # Skip disabled hooks
            if not hook.enabled:
                continue

            # Check condition
            if hook.condition:
                try:
                    if inspect.iscoroutinefunction(hook.condition):
                        should_execute = await hook.condition(event)
                    else:
                        should_execute = hook.condition(event)

                    if not should_execute:
                        logger.debug(f"Hook condition not met: {hook_id}")
                        continue
                except Exception as e:
                    logger.error(f"Error checking hook condition: {e}")
                    continue

            # Execute hook
            result = await self._execute_hook(hook, event)
            results.append(result)

            # Update event context with result (for next hooks in chain)
            if result.success and result.result is not None:
                event.context[f"hook_{hook_id}_result"] = result.result

        return results

    async def _execute_hook(self, hook: Hook, event: HookEvent) -> HookResult:
        """Execute a single hook"""
        start_time = time.time()

        try:
            self.stats.total_executions += 1

            # Apply middleware (pre-processing)
            processed_event = event
            for middleware in self.middleware:
                if inspect.iscoroutinefunction(middleware):
                    processed_event = await middleware(processed_event)
                else:
                    processed_event = middleware(processed_event)

            # Execute handler
            if inspect.iscoroutinefunction(hook.handler):
                result = await hook.handler(processed_event)
            else:
                result = hook.handler(processed_event)

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self.stats.avg_execution_time = sum(self.execution_times) / len(
                self.execution_times
            )
            self.stats.successful_executions += 1

            logger.debug(
                f"Hook executed: {hook.hook_id} ({execution_time:.3f}s)"
            )

            return HookResult(
                hook_id=hook.hook_id,
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.stats.failed_executions += 1

            logger.error(f"Hook execution failed: {hook.hook_id} - {e}")

            return HookResult(
                hook_id=hook.hook_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    # ========================================================================
    # Plugin System
    # ========================================================================

    async def register_plugin(
        self,
        plugin_id: str,
        name: str,
        description: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a plugin"""
        plugin = Plugin(
            plugin_id=plugin_id,
            name=name,
            description=description,
            version=version,
            metadata=metadata or {},
        )

        self.plugins[plugin_id] = plugin

        logger.info(f"Registered plugin: {plugin_id} ({name})")
        return True

    async def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin and all its hooks"""
        if plugin_id not in self.plugins:
            return False

        plugin = self.plugins[plugin_id]

        # Unregister all plugin hooks
        for hook_id in plugin.hooks:
            await self.unregister_hook(hook_id)

        # Remove plugin
        del self.plugins[plugin_id]

        logger.info(f"Unregistered plugin: {plugin_id}")
        return True

    async def add_hook_to_plugin(self, plugin_id: str, hook_id: str) -> bool:
        """Add a hook to a plugin"""
        if plugin_id not in self.plugins:
            return False

        if hook_id not in self.hooks:
            return False

        self.plugins[plugin_id].hooks.append(hook_id)
        return True

    async def list_plugins(self, enabled_only: bool = False) -> List[Plugin]:
        """List all plugins"""
        plugins = list(self.plugins.values())

        if enabled_only:
            plugins = [p for p in plugins if p.enabled]

        return plugins

    async def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get plugin by ID"""
        return self.plugins.get(plugin_id)

    async def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin and all its hooks"""
        if plugin_id not in self.plugins:
            return False

        plugin = self.plugins[plugin_id]
        plugin.enabled = True

        # Enable all plugin hooks
        for hook_id in plugin.hooks:
            await self.enable_hook(hook_id)

        logger.info(f"Enabled plugin: {plugin_id}")
        return True

    async def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin and all its hooks"""
        if plugin_id not in self.plugins:
            return False

        plugin = self.plugins[plugin_id]
        plugin.enabled = False

        # Disable all plugin hooks
        for hook_id in plugin.hooks:
            await self.disable_hook(hook_id)

        logger.info(f"Disabled plugin: {plugin_id}")
        return True

    # ========================================================================
    # Middleware System
    # ========================================================================

    def add_middleware(self, middleware: Callable) -> bool:
        """
        Add middleware to process events before hooks

        Middleware receives HookEvent and returns modified HookEvent
        """
        self.middleware.append(middleware)
        logger.info(f"Added middleware: {middleware.__name__}")
        return True

    def remove_middleware(self, middleware: Callable) -> bool:
        """Remove middleware"""
        if middleware in self.middleware:
            self.middleware.remove(middleware)
            logger.info(f"Removed middleware: {middleware.__name__}")
            return True
        return False

    def clear_middleware(self) -> bool:
        """Clear all middleware"""
        self.middleware.clear()
        logger.info("Cleared all middleware")
        return True

    # ========================================================================
    # Decorators
    # ========================================================================

    def pre_hook(self, event_type: EventType, priority: int = 50):
        """
        Decorator to register a pre-hook

        Usage:
            @hook_service.pre_hook(EventType.LLM_GENERATE_PRE)
            async def my_pre_hook(event: HookEvent):
                # Do something before LLM generation
                pass
        """

        def decorator(func: Callable):
            hook_id = f"pre_{func.__name__}_{time.time()}"
            asyncio.create_task(
                self.register_hook(
                    hook_id=hook_id,
                    name=func.__name__,
                    event_type=event_type,
                    handler=func,
                    priority=priority,
                )
            )
            return func

        return decorator

    def post_hook(self, event_type: EventType, priority: int = 50):
        """
        Decorator to register a post-hook

        Usage:
            @hook_service.post_hook(EventType.LLM_GENERATE_POST)
            async def my_post_hook(event: HookEvent):
                # Do something after LLM generation
                pass
        """

        def decorator(func: Callable):
            hook_id = f"post_{func.__name__}_{time.time()}"
            asyncio.create_task(
                self.register_hook(
                    hook_id=hook_id,
                    name=func.__name__,
                    event_type=event_type,
                    handler=func,
                    priority=priority,
                )
            )
            return func

        return decorator

    # ========================================================================
    # Monitoring & Debugging
    # ========================================================================

    async def get_stats(self) -> HookStats:
        """Get hook execution statistics"""
        self.stats.total_hooks = len(self.hooks)
        return self.stats

    async def get_event_history(
        self, event_type: Optional[EventType] = None, limit: int = 100
    ) -> List[HookEvent]:
        """Get event history"""
        history = self.event_history

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        return history[-limit:]

    async def clear_event_history(self) -> bool:
        """Clear event history"""
        self.event_history.clear()
        logger.info("Cleared event history")
        return True

    async def reset_stats(self) -> bool:
        """Reset statistics"""
        self.stats = HookStats()
        self.execution_times = []
        logger.info("Reset hook statistics")
        return True

    async def get_hook_performance(
        self, hook_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific hook"""
        if hook_id not in self.hooks:
            return None

        # Calculate metrics from event history
        executions = 0
        total_time = 0.0

        # This is a simplified version - in production, track per-hook metrics
        return {
            "hook_id": hook_id,
            "executions": executions,
            "avg_time": total_time / executions if executions > 0 else 0.0,
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def emit_pre(
        self,
        base_event: str,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HookResult]:
        """
        Emit a pre-event (convenience method)

        Example: emit_pre("llm.generate") -> emits LLM_GENERATE_PRE
        """
        event_type_str = f"{base_event}.pre"
        try:
            event_type = EventType(event_type_str)
            return await self.emit(event_type, data, context)
        except ValueError:
            logger.warning(f"Unknown event type: {event_type_str}")
            return []

    async def emit_post(
        self,
        base_event: str,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HookResult]:
        """
        Emit a post-event (convenience method)

        Example: emit_post("llm.generate") -> emits LLM_GENERATE_POST
        """
        event_type_str = f"{base_event}.post"
        try:
            event_type = EventType(event_type_str)
            return await self.emit(event_type, data, context)
        except ValueError:
            logger.warning(f"Unknown event type: {event_type_str}")
            return []

    async def create_hook_chain(
        self,
        chain_id: str,
        event_type: EventType,
        handlers: List[Callable],
        base_priority: int = 50,
    ) -> List[str]:
        """
        Create a chain of hooks for the same event

        Hooks are registered with decreasing priority to ensure execution order
        """
        hook_ids = []

        for i, handler in enumerate(handlers):
            hook_id = f"{chain_id}_{i}"
            priority = base_priority - i  # Decreasing priority

            await self.register_hook(
                hook_id=hook_id,
                name=f"{chain_id}_step_{i}",
                event_type=event_type,
                handler=handler,
                priority=priority,
            )

            hook_ids.append(hook_id)

        logger.info(f"Created hook chain: {chain_id} ({len(hook_ids)} hooks)")
        return hook_ids

    async def remove_hook_chain(self, hook_ids: List[str]) -> bool:
        """Remove a chain of hooks"""
        for hook_id in hook_ids:
            await self.unregister_hook(hook_id)

        logger.info(f"Removed hook chain: {len(hook_ids)} hooks")
        return True


# ============================================================================
# Singleton Instance
# ============================================================================

hook_service = HookService()

