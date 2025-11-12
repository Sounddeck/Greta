"""
Hook Service for CPAS5
Wraps the hook system from utils.hooks for service-level integration
"""
import logging
from typing import Dict, Any, List
from utils.hooks import hook_manager, HookContext, execute_hooks

logger = logging.getLogger(__name__)


class HookService:
    """
    Hook Service - Provides service-level access to the hook system
    
    This service wraps the hook manager from utils.hooks and provides
    initialization, shutdown, and health check methods for integration
    with the main application lifecycle.
    """
    
    def __init__(self):
        self.hook_manager = hook_manager
        self.initialized = False
        self.metrics = {
            "total_hooks_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "registered_hooks": 0,
        }
        logger.info("✅ Hook Service created")
    
    async def initialize(self):
        """Initialize the hook service"""
        try:
            # Count registered hooks
            self.metrics["registered_hooks"] = len(self.hook_manager.hooks)
            
            # Execute system startup hooks
            await execute_hooks('system-startup')
            
            self.initialized = True
            logger.info(f"✅ Hook Service initialized with {self.metrics['registered_hooks']} registered hooks")
            
        except Exception as e:
            logger.error(f"❌ Hook Service initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the hook service"""
        try:
            # Execute system shutdown hooks
            await execute_hooks('system-shutdown')
            
            self.initialized = False
            logger.info("✅ Hook Service shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Hook Service shutdown error: {e}")
            raise
    
    async def execute_hook(self, hook_type: str, **context_kwargs) -> List[Any]:
        """
        Execute hooks of a specific type
        
        Args:
            hook_type: Type of hook to execute (e.g., 'pre-command', 'post-command')
            **context_kwargs: Context data to pass to hooks
            
        Returns:
            List of results from all executed hooks
        """
        try:
            results = await execute_hooks(hook_type, **context_kwargs)
            self.metrics["total_hooks_executed"] += 1
            self.metrics["successful_executions"] += 1
            return results
            
        except Exception as e:
            self.metrics["total_hooks_executed"] += 1
            self.metrics["failed_executions"] += 1
            logger.error(f"❌ Hook execution failed for type '{hook_type}': {e}")
            raise
    
    def get_registered_hooks(self) -> Dict[str, List[str]]:
        """
        Get all registered hooks grouped by type
        
        Returns:
            Dictionary mapping hook types to lists of hook function names
        """
        hooks_by_type = {}
        for hook_type, hooks in self.hook_manager.hooks.items():
            hooks_by_type[hook_type] = [
                hook.func.__name__ for hook in hooks
            ]
        return hooks_by_type
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get hook service metrics"""
        return {
            "service": "hook_service",
            "initialized": self.initialized,
            "metrics": self.metrics,
            "registered_hooks_by_type": {
                hook_type: len(hooks)
                for hook_type, hooks in self.hook_manager.hooks.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Hook service health check"""
        return {
            "status": "healthy" if self.initialized else "not_initialized",
            "service": "hook_service",
            "initialized": self.initialized,
            "registered_hooks": self.metrics["registered_hooks"],
            "total_executions": self.metrics["total_hooks_executed"],
            "success_rate": (
                (self.metrics["successful_executions"] / 
                 max(self.metrics["total_hooks_executed"], 1)) * 100
            ),
            "hooks_by_type": {
                hook_type: len(hooks)
                for hook_type, hooks in self.hook_manager.hooks.items()
            }
        }


# Global hook service instance
hook_service = HookService()


__all__ = ['HookService', 'hook_service']

