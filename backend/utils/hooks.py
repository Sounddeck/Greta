"""
GRETA PAI - Dynamic Hook System
Core PAI Feature: Extensible pre/post command execution hooks
"""
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
import asyncio
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import inspect

from utils.error_handling import GretaException, handle_errors, logger

logger = logging.getLogger(__name__)


class HookError(GretaException):
    """Hook system errors"""


class HookContext:
    """Context passed to hooks during execution"""

    def __init__(self, hook_type: str, command: Optional[str] = None,
                 user_query: Optional[str] = None, **kwargs):
        self.hook_type = hook_type
        self.command = command
        self.user_query = user_query
        self.timestamp = datetime.utcnow()
        self.metadata = kwargs
        self.execution_time = None
        self.success = None
        self.result = None
        self.error = None

    def mark_success(self, result: Any = None):
        """Mark hook execution as successful"""
        self.success = True
        self.result = result
        self.execution_time = datetime.utcnow() - self.timestamp

    def mark_failure(self, error: Exception):
        """Mark hook execution as failed"""
        self.success = False
        self.error = str(error)
        self.execution_time = datetime.utcnow() - self.timestamp


class HookManager:
    """
    Dynamic Hook Manager
    PAI's extensible hook system for command lifecycle management
    """

    HOOK_TYPES = {
        # Context Loading Hooks
        'submit-user-query': 'Fired when user submits a query - loads relevant context',
        'load-dynamic-requirements': 'Smart routing based on query analysis',
        'context-loaded': 'Fired after context is loaded and available',

        # Command Execution Hooks
        'pre-command': 'Fired before any command execution',
        'post-command': 'Fired after command execution (success or failure)',
        'command-success': 'Fired only on successful command execution',
        'command-failure': 'Fired only on command execution failure',

        # Agent Hooks
        'agent-selected': 'Fired when an agent is selected for a task',
        'agent-started': 'Fired when agent begins execution',
        'agent-completed': 'Fired when agent completes execution',
        'agent-error': 'Fired when agent encounters an error',

        # Session Management Hooks
        'session-started': 'Fired when new session begins',
        'session-ended': 'Fired when session ends',
        'session-saved': 'Fired when session state is saved',

        # Integration Hooks
        'external-api-call': 'Fired before external API calls',
        'external-api-response': 'Fired after external API responses',

        # System Hooks
        'system-startup': 'Fired during system initialization',
        'system-shutdown': 'Fired during system shutdown',
        'system-health-check': 'Fired during health checks'
    }

    def __init__(self):
        self.hooks: Dict[str, List[Dict[str, Any]]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hook-executor")

        # Initialize hook storage for each type
        for hook_type in self.HOOK_TYPES:
            self.hooks[hook_type] = []

        # Global hook statistics
        self.statistics = {
            'executions': {},
            'errors': {},
            'performance': {}
        }

        logger.info("🪝 Dynamic Hook System initialized with {} hook types".format(len(self.HOOK_TYPES)))

    def register_hook(self, hook_type: str, hook_function: Callable,
                     priority: int = 50, name: Optional[str] = None,
                     async_hook: bool = False) -> bool:
        """
        Register a new hook function

        Args:
            hook_type: Type of hook (from HOOK_TYPES)
            hook_function: Function to execute
            priority: Execution priority (0-100, lower = higher priority)
            name: Optional name for the hook
            async_hook: Whether this is an async function

        Returns:
            True if successfully registered
        """
        if hook_type not in self.HOOK_TYPES:
            available_types = ", ".join(self.HOOK_TYPES.keys())
            raise HookError(f"Unknown hook type '{hook_type}'. Available: {available_types}")

        if not callable(hook_function):
            raise HookError(f"Hook function {hook_function} is not callable")

        hook_info = {
            'function': hook_function,
            'priority': priority,
            'name': name or f"{hook_function.__module__}.{hook_function.__qualname__}",
            'async': async_hook,
            'registered_at': datetime.utcnow(),
            'execution_count': 0,
            'error_count': 0
        }

        # Insert hook in priority order (lower priority first)
        insert_index = 0
        for i, existing_hook in enumerate(self.hooks[hook_type]):
            if priority < existing_hook['priority']:
                break
            insert_index = i + 1

        self.hooks[hook_type].insert(insert_index, hook_info)

        logger.info(f"🪝 Registered {hook_type} hook: {hook_info['name']} (priority: {priority})")
        return True

    def unregister_hook(self, hook_type: str, hook_name: str) -> bool:
        """Unregister a hook by name"""
        if hook_type not in self.hooks:
            return False

        for i, hook_info in enumerate(self.hooks[hook_type]):
            if hook_info['name'] == hook_name:
                self.hooks[hook_type].pop(i)
                logger.info(f"🪝 Unregistered {hook_type} hook: {hook_name}")
                return True

        return False

    async def execute_hooks(self, hook_type: str, context: HookContext) -> List[Any]:
        """
        Execute all hooks of a specific type with the given context

        Args:
            hook_type: Type of hooks to execute
            context: Hook execution context

        Returns:
            List of hook execution results
        """
        if hook_type not in self.hooks:
            logger.warning(f"No hooks registered for type: {hook_type}")
            return []

        if not self.hooks[hook_type]:
            logger.debug(f"No hooks to execute for type: {hook_type}")
            return []

        logger.info(f"🔥 Executing {len(self.hooks[hook_type])} {hook_type} hooks")

        results = []
        start_time = datetime.utcnow()

        for hook_info in self.hooks[hook_type]:
            hook_start = datetime.utcnow()

            try:
                # Execute hook synchronously or asynchronously
                if hook_info['async'] or inspect.iscoroutinefunction(hook_info['function']):
                    if asyncio.iscoroutinefunction(hook_info['function']):
                        result = await hook_info['function'](context)
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            self._executor, hook_info['function'], context
                        )
                else:
                    # Run sync function in thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        self._executor, hook_info['function'], context
                    )

                execution_time = datetime.utcnow() - hook_start
                hook_info['execution_count'] += 1

                logger.debug(f"✅ Hook {hook_info['name']} executed in {execution_time.total_seconds():.3f}s")

                results.append({
                    'hook_name': hook_info['name'],
                    'success': True,
                    'result': result,
                    'execution_time': execution_time.total_seconds()
                })

            except Exception as e:
                hook_info['error_count'] += 1

                if hook_type not in self.statistics['errors']:
                    self.statistics['errors'][hook_type] = 0
                self.statistics['errors'][hook_type] += 1

                logger.error(f"❌ Hook {hook_info['name']} failed: {str(e)}")

                results.append({
                    'hook_name': hook_info['name'],
                    'success': False,
                    'error': str(e),
                    'execution_time': (datetime.utcnow() - hook_start).total_seconds()
                })

        total_time = datetime.utcnow() - start_time

        # Update performance statistics
        self._update_performance_stats(hook_type, len(results), total_time.total_seconds())

        logger.info(f"🔚 Completed {hook_type} hooks execution in {total_time.total_seconds():.3f}s")

        return results

    def _update_performance_stats(self, hook_type: str, hook_count: int, total_time: float):
        """Update performance statistics"""
        if hook_type not in self.statistics['performance']:
            self.statistics['performance'][hook_type] = []

        self.statistics['performance'][hook_type].append({
            'timestamp': datetime.utcnow(),
            'hook_count': hook_count,
            'total_time': total_time,
            'avg_time_per_hook': total_time / hook_count if hook_count > 0 else 0
        })

        # Keep only last 100 performance measurements per hook type
        if len(self.statistics['performance'][hook_type]) > 100:
            self.statistics['performance'][hook_type] = self.statistics['performance'][hook_type][-100:]

    def get_hook_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hook system statistics"""
        stats = {
            'total_hook_types': len(self.HOOK_TYPES),
            'hooks_registered': {},
            'executions': {},
            'errors': {},
            'performance': {}
        }

        # Count registered hooks
        for hook_type, hooks in self.hooks.items():
            stats['hooks_registered'][hook_type] = len(hooks)

            # Get execution stats per hook
            for hook_info in hooks:
                hook_name = hook_info['name']
                if hook_name not in stats['executions']:
                    stats['executions'][hook_name] = hook_info['execution_count']
                if hook_name not in stats['errors']:
                    stats['errors'][hook_name] = hook_info['error_count']

        # Copy performance stats
        stats['performance'] = self.statistics['performance'].copy()

        return stats

    def list_hooks(self, hook_type: Optional[str] = None) -> Dict[str, Any]:
        """List all registered hooks"""
        if hook_type:
            if hook_type not in self.hooks:
                return {}

            return {
                hook_type: [
                    {
                        'name': hook_info['name'],
                        'priority': hook_info['priority'],
                        'async': hook_info['async'],
                        'executions': hook_info['execution_count'],
                        'errors': hook_info['error_count']
                    }
                    for hook_info in self.hooks[hook_type]
                ]
            }

        # Return all hooks
        return {
            hook_type: [
                {
                    'name': hook_info['name'],
                    'priority': hook_info['priority'],
                    'async': hook_info['async'],
                    'executions': hook_info['execution_count'],
                    'errors': hook_info['error_count']
                }
                for hook_info in self.hooks[hook_type]
            ]
            for hook_type in self.hooks.keys()
        }


# Global hook manager instance
hook_manager = HookManager()


def hook(hook_type: str, priority: int = 50, name: Optional[str] = None):
    """
    Decorator to register a function as a hook

    Usage:
        @hook('pre-command', priority=10)
        async def my_pre_command_hook(context):
            # Do something before command execution
            pass
    """
    def decorator(func):
        # Determine if function is async
        is_async = asyncio.iscoroutinefunction(func)

        # Register the hook
        hook_manager.register_hook(
            hook_type=hook_type,
            hook_function=func,
            priority=priority,
            name=name or f"{func.__module__}.{func.__qualname__}",
            async_hook=is_async
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # For direct calls, just call the function
            if is_async:
                return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper
    return decorator


async def execute_hooks(hook_type: str, **context_kwargs) -> List[Any]:
    """
    Convenience function to execute hooks

    Usage:
        await execute_hooks('pre-command', command='analyze-code', user='john')
    """
    context = HookContext(hook_type, **context_kwargs)
    return await hook_manager.execute_hooks(hook_type, context)


# PAI Built-in Hooks
@hook('system-startup', priority=1)
async def pai_startup_initialization(context: HookContext):
    """PAI system startup hook - runs first"""
    logger.info("🚀 PAI System startup initialization")
    # Initialize core systems, load configurations, etc.
    return "PAI System ready"

@hook('system-shutdown', priority=99)
async def pai_shutdown_cleanup(context: HookContext):
    """PAI system shutdown hook - runs last"""
    logger.info("🔄 PAI System shutdown cleanup")
    # Save state, cleanup resources, etc.
    return "PAI System shutdown complete"

@hook('submit-user-query', priority=5)
async def uf_context_loader(context: HookContext):
    """Load UFC context based on user query"""
    from utils.ufc_context import ufc_manager

    if not context.user_query:
        return {"context_loaded": False, "reason": "No user query"}

    # Classify intent and load context
    intent = await ufc_manager.classify_intent(context.user_query)
    context_data = await ufc_manager.load_context_by_intent(intent, context.user_query)

    return {
        "context_loaded": True,
        "intent": intent,
        "files_loaded": len(context_data.get("relevant_files", [])),
        "context_items": len(context_data.get("context", {}))
    }

@hook('session-started', priority=10)
async def session_context_initialization(context: HookContext):
    """Initialize session with user preferences and history"""
    logger.info(f"📋 Session started - initializing user context")

    # Could load user preferences, recent history, etc.
    return {
        "session_initialized": True,
        "preferences_loaded": True,
        "history_loaded": True
    }


class HookAPI:
    """Hook System API Endpoints"""
    pass  # Will be implemented in the API router


__all__ = [
    'HookManager',
    'HookContext',
    'HookError',
    'hook_manager',
    'hook',
    'execute_hooks',
    'HookAPI'
]
