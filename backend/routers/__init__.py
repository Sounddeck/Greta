"""Enhanced GRETA Router Package - Complete PAI System"""

from fastapi import APIRouter
import os
import logging
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)


def get_routers() -> List[APIRouter]:
    """Get all available routers safely"""
    routers = []

    # Define router modules to try loading
    router_modules = [
        'web_commands',
        'llm',
        'memory',
        'system',
        'agent',
        'voice',
        'reasoning',
        'master_agent',
        'training',
        'osint',
        'autonomous',
        'multimodal',
        'predictive',
        'workflows',
        'integrations'
    ]

    for module_name in router_modules:
        try:
            module = __import__(f'routers.{module_name}', fromlist=[module_name])
            router = getattr(module, 'router', None)
            if router:
                routers.append((module_name, router))
                logger.info(f"✅ Router {module_name} loaded successfully")
            else:
                logger.debug(f"Router {module_name} not found in module")
        except ImportError as e:
            logger.debug(f"Router {module_name} not available: {e}")
        except Exception as e:
            logger.warning(f"Error loading router {module_name}: {e}")

    return [(name, router) for name, router in routers]


def import_router_safe(router_name: str) -> Optional[APIRouter]:
    """Safely import a specific router"""
    try:
        module = __import__(f'routers.{router_name}', fromlist=[router_name])
        router = getattr(module, 'router', None)
        return router
    except (ImportError, AttributeError):
        return None
    except Exception as e:
        logger.warning(f"Error importing router {router_name}: {e}")
        return None


# Legacy compatibility - will be removed in future versions
def get_all_routers():
    """Legacy function for backward compatibility"""
    return get_routers()
