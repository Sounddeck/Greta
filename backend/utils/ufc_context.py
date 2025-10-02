"""
GRETA PAI - Universal File-based Context (UFC) System
Core PAI Feature: Plain-text hierarchical context management
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from filelock import FileLock, Timeout
import asyncio
from functools import wraps
import hashlib
import logging

from utils.error_handling import GretaException, handle_errors, error_context
from utils.performance import performance_monitor

logger = logging.getLogger(__name__)


class UFCContextError(GretaException):
    """UFC context management errors"""


def require_context_lock(func):
    """Decorator to ensure file locking for context operations"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_context_lock') or self._context_lock is None:
            raise UFCContextError("Context manager not properly initialized")

        try:
            with self._context_lock:
                return await func(self, *args, **kwargs)
        except Timeout:
            raise UFCContextError("Context file locked by another process")
        except Exception as e:
            raise UFCContextError(f"Context operation failed: {e}")

    return wrapper


class UFCManager:
    """
    Universal File-based Context (UFC) Manager
    PAI's core context management system
    """

    def __init__(self, context_root: Optional[Path] = None):
        # Use PAI-style directory structure
        self.context_root = context_root or Path("~/.greta/context").expanduser()

        # PAI's hierarchical context structure
        self.context_hierarchy = {
            "projects": self.context_root / "projects",
            "life": self.context_root / "life",
            "work": self.context_root / "work",
            "benefits": self.context_root / "benefits",
            "tools": self.context_root / "tools",
            "communication": self.context_root / "communication"
        }

        # Initialize directories
        self._initialize_directories()

        # Context lock for thread safety
        self._context_lock = FileLock(self.context_root / ".context.lock")

        # Context cache for performance
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info(f"🎯 UFC Context Manager initialized at {self.context_root}")

    def _initialize_directories(self):
        """Create PAI directory structure if it doesn't exist"""
        self.context_root.mkdir(parents=True, exist_ok=True)

        for category, path in self.context_hierarchy.items():
            path.mkdir(exist_ok=True)

            # Create subdirectory structure
            if category == "projects":
                (path / "active").mkdir(exist_ok=True)
                (path / "completed").mkdir(exist_ok=True)
                (path / "templates").mkdir(exist_ok=True)

            elif category == "life":
                (path / "finances").mkdir(exist_ok=True)
                (path / "health").mkdir(exist_ok=True)
                (path / "personal").mkdir(exist_ok=True)

            elif category == "work":
                (path / "consulting").mkdir(exist_ok=True)
                (path / "research").mkdir(exist_ok=True)
                (path / "meetings").mkdir(exist_ok=True)

        # Create metadata file
        metadata_file = self.context_root / "metadata.json"
        if not metadata_file.exists():
            metadata = {
                "version": "2.0.0",
                "created": datetime.utcnow().isoformat(),
                "categories": list(self.context_hierarchy.keys()),
                "description": "GRETA PAI - Universal File-based Context System"
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

    async def classify_intent(self, query: str) -> str:
        """
        Classify user intent to determine relevant context
        PAI's dynamic context routing
        """
        query_lower = query.lower()

        # Intent classification logic
        if any(keyword in query_lower for keyword in ['code', 'programming', 'debug', 'fix', 'build']):
            return 'engineering'

        elif any(keyword in query_lower for keyword in ['write', 'blog', 'article', 'content', 'draft']):
            return 'writing'

        elif any(keyword in query_lower for keyword in ['analyze', 'review', 'assess', 'evaluate']):
            return 'research'

        elif any(keyword in query_lower for keyword in ['finance', 'money', 'budget', 'expense']):
            return 'finance'

        elif any(keyword in query_lower for keyword in ['health', 'medical', 'wellness', 'fitness']):
            return 'health'

        elif any(keyword in query_lower for keyword in ['business', 'meeting', 'email', 'communication']):
            return 'business'

        else:
            return 'general'

    @require_context_lock
    @handle_errors("load_context_by_intent")
    async def load_context_by_intent(self, intent: str,
                                   user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Load relevant context based on user intent
        Core PAI functionality - dynamic context injection
        """
        context_data = {
            "intent": intent,
            "timestamp": datetime.utcnow().isoformat(),
            "context": {},
            "relevant_files": [],
            "metadata": {}
        }

        # Load context based on intent
        if intent == 'engineering':
            context_data["context"] = await self._load_engineering_context()
            context_data["project"] = "engineering"
            context_data["expertise"] = "technical"

        elif intent == 'writing':
            context_data["context"] = await self._load_writing_context()
            context_data["style"] = "professional"
            context_data["expertise"] = "communication"

        elif intent == 'research':
            context_data["context"] = await self._load_research_context()
            context_data["methodology"] = "analytical"
            context_data["expertise"] = "research"

        elif intent == 'finance':
            context_data["context"] = await self._load_finance_context()
            context_data["expertise"] = "financial"

        elif intent == 'health':
            context_data["context"] = await self._load_health_context()
            context_data["expertise"] = "healthcare"

        elif intent == 'business':
            context_data["context"] = await self._load_business_context()
            context_data["expertise"] = "business"

        else:
            context_data["context"] = await self._load_general_context()

        # Find relevant context files
        context_data["relevant_files"] = await self._find_relevant_files(intent, user_query)

        logger.info(f"🎯 Loaded context for intent '{intent}': {len(context_data['context'])} items, {len(context_data['relevant_files'])} files")

        return context_data

    async def _load_engineering_context(self) -> Dict[str, Any]:
        """Load engineering project context"""
        projects_path = self.context_hierarchy["projects"] / "active"
        context = {}

        # Load all active engineering projects
        for project_file in projects_path.glob("*.md"):
            if "engineering" in project_file.name.lower() or "code" in project_file.name.lower():
                try:
                    content = await asyncio.get_event_loop().run_in_executor(
                        None, project_file.read_text, 'utf-8'
                    )
                    context[f"project_{project_file.stem}"] = content[:2000]  # Truncate for context
                except Exception as e:
                    logger.warning(f"Failed to load project context {project_file}: {e}")

        return context

    async def _load_writing_context(self) -> Dict[str, Any]:
        """Load writing and content creation context"""
        context = {}

        # Load writing samples, style guides, etc.
        writing_file = self.context_hierarchy["work"] / "writing_style.md"
        if writing_file.exists():
            try:
                context["writing_style"] = await asyncio.get_event_loop().run_in_executor(
                    None, writing_file.read_text, 'utf-8'
                )
            except Exception as e:
                logger.warning(f"Failed to load writing context: {e}")

        return context

    async def _load_research_context(self) -> Dict[str, Any]:
        """Load research methodology and frameworks"""
        context = {}

        # Load research templates, methodologies
        research_dir = self.context_hierarchy["work"] / "research"
        if research_dir.exists():
            for template_file in research_dir.glob("*.md"):
                try:
                    content = await asyncio.get_event_loop().run_in_executor(
                        None, template_file.read_text, 'utf-8'
                    )
                    context[f"research_{template_file.stem}"] = content
                except Exception as e:
                    logger.warning(f"Failed to load research template {template_file}: {e}")

        return context

    async def _load_finance_context(self) -> Dict[str, Any]:
        """Load financial planning and tracking context"""
        context = {}

        finance_dir = self.context_hierarchy["life"] / "finances"
        if finance_dir.exists():
            for data_file in finance_dir.glob("*.json"):
                try:
                    content = await asyncio.get_event_loop().run_in_executor(
                        None, data_file.read_text, 'utf-8'
                    )
                    data = json.loads(content)
                    context[data_file.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load finance data {data_file}: {e}")

        return context

    async def _load_health_context(self) -> Dict[str, Any]:
        """Load health monitoring and wellness context"""
        context = {}

        health_dir = self.context_hierarchy["life"] / "health"
        if health_dir.exists():
            for tracking_file in health_dir.glob("*.md"):
                try:
                    content = await asyncio.get_event_loop().run_in_executor(
                        None, tracking_file.read_text, 'utf-8'
                    )
                    context[f"health_{tracking_file.stem}"] = content[:1000]  # Summary
                except Exception as e:
                    logger.warning(f"Failed to load health context {tracking_file}: {e}")

        return context

    async def _load_business_context(self) -> Dict[str, Any]:
        """Load business operations and communication context"""
        context = {}

        work_dir = self.context_hierarchy["work"]
        for category in ["consulting", "meetings"]:
            category_dir = work_dir / category
            if category_dir.exists():
                for file in category_dir.glob("*.md"):
                    try:
                        content = await asyncio.get_event_loop().run_in_executor(
                            None, file.read_text, 'utf-8'
                        )
                        context[f"{category}_{file.stem}"] = content[:1500]
                    except Exception as e:
                        logger.warning(f"Failed to load business context {file}: {e}")

        return context

    async def _load_general_context(self) -> Dict[str, Any]:
        """Load general personal context"""
        context = {}

        # Load personal preferences, habits, etc.
        personal_file = self.context_hierarchy["life"] / "personal" / "profile.md"
        if personal_file.exists():
            try:
                context["personal_profile"] = await asyncio.get_event_loop().run_in_executor(
                    None, personal_file.read_text, 'utf-8'
                )
            except Exception as e:
                logger.warning(f"Failed to load personal context: {e}")

        return context

    async def _find_relevant_files(self, intent: str, query: Optional[str] = None) -> List[str]:
        """Find files relevant to the intent and query"""
        relevant_files = []

        # Search patterns based on intent
        search_paths = {
            'engineering': ['projects/active', 'tools'],
            'writing': ['work', 'communication'],
            'research': ['work/research', 'projects'],
            'finance': ['life/finances', 'benefits'],
            'health': ['life/health'],
            'business': ['work', 'communication'],
            'general': ['life/personal']
        }

        paths_to_search = search_paths.get(intent, ['life/personal'])

        for path_part in paths_to_search:
            full_path = self.context_hierarchy.get(path_part.split('/')[0])
            if full_path and path_part != path_part.split('/')[0]:
                full_path = full_path / '/'.join(path_part.split('/')[1:])

            if full_path and full_path.exists():
                for file_path in full_path.rglob("*.md"):
                    relevant_files.append(str(file_path.relative_to(self.context_root)))

        # Limit to 10 most relevant files
        return relevant_files[:10]

    @require_context_lock
    @handle_errors("save_context")
    async def save_context(self, category: str, name: str, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save context to the UFC system
        """
        if category not in self.context_hierarchy:
            raise UFCContextError(f"Unknown context category: {category}")

        category_path = self.context_hierarchy[category]
        file_path = category_path / f"{name}.md"

        # Create full content with metadata
        full_content = content
        if metadata:
            metadata_str = f"---\n{json.dumps(metadata, indent=2)}\n---\n\n"
            full_content = metadata_str + content

        try:
            # Write file atomically
            temp_path = file_path.with_suffix('.tmp')
            await asyncio.get_event_loop().run_in_executor(
                None, temp_path.write_text, full_content, 'utf-8'
            )
            await asyncio.get_event_loop().run_in_executor(
                None, temp_path.replace, file_path
            )

            # Update cache
            cache_key = f"{category}/{name}"
            self._context_cache[cache_key] = {
                "content": content,
                "metadata": metadata or {},
                "last_modified": datetime.utcnow().isoformat()
            }

            logger.info(f"💾 Saved context: {category}/{name}")
            return True

        except Exception as e:
            raise UFCContextError(f"Failed to save context {category}/{name}: {e}")

    @handle_errors("get_context_statistics")
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get comprehensive context system statistics"""
        stats = {
            "total_files": 0,
            "categories": {},
            "last_modified": {},
            "file_sizes": {},
            "total_size_mb": 0
        }

        for category, path in self.context_hierarchy.items():
            if path.exists():
                files = list(path.rglob("*.md"))
                stats["categories"][category] = len(files)
                stats["total_files"] += len(files)

                if files:
                    # Get last modified time
                    latest_file = max(files, key=lambda f: f.stat().st_mtime)
                    stats["last_modified"][category] = datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ).isoformat()

                    # Calculate sizes
                    category_size = sum(f.stat().st_size for f in files)
                    stats["file_sizes"][category] = category_size
                    stats["total_size_mb"] += category_size / (1024 * 1024)

        return stats

    async def create_template_context(self, template_name: str, template_data: Dict[str, Any]) -> str:
        """Create context from template (PAI pattern expansion)"""
        templates_dir = self.context_hierarchy["projects"] / "templates"

        # Read template
        template_file = templates_dir / f"{template_name}.md"
        if not template_file.exists():
            raise UFCContextError(f"Template {template_name} not found")

        try:
            template_content = await asyncio.get_event_loop().run_in_executor(
                None, template_file.read_text, 'utf-8'
            )

            # Simple template substitution
            for key, value in template_data.items():
                template_content = template_content.replace(f"{{{{{key}}}}}", str(value))

            return template_content

        except Exception as e:
            raise UFCContextError(f"Failed to create template context: {e}")


# Global UFC manager instance
ufc_manager = UFCManager()


class UFCAPI:
    """UFC Context System API Endpoints"""
    pass  # Will be implemented in the API router


__all__ = [
    'UFCManager',
    'UFCContextError',
    'ufc_manager',
    'UFCAPI'
]
