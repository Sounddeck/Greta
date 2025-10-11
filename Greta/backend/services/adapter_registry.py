"""
Adapter Registry Service - Week 9-10 Component 3

Manages LoRA adapters: storage, versioning, metadata, loading/unloading.
Integrates with Week 2 Model Management and Week 1 LLM Service.
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Schemas
# ============================================================================


class AdapterInfo(BaseModel):
    """Adapter metadata"""

    adapter_id: str
    name: str
    description: str
    base_model: str
    task: str  # e.g., "general", "coding", "writing", "analysis"
    version: str = "1.0.0"
    metrics: Dict[str, float] = Field(default_factory=dict)
    lora_config: Dict[str, Any] = Field(default_factory=dict)
    training_info: Dict[str, Any] = Field(default_factory=dict)
    file_path: str
    file_size: int = 0  # bytes
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used: Optional[str] = None
    usage_count: int = 0
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdapterStats(BaseModel):
    """Adapter registry statistics"""

    total_adapters: int
    adapters_by_task: Dict[str, int]
    adapters_by_base_model: Dict[str, int]
    total_size: int  # bytes
    most_used: List[str]


# ============================================================================
# Adapter Registry
# ============================================================================


class AdapterRegistry:
    """
    Adapter Registry Service

    Features:
    - Adapter registration and metadata storage
    - Adapter versioning
    - Adapter loading/unloading
    - Usage tracking
    - Search and filtering
    - Storage management
    """

    def __init__(
        self,
        adapters_dir: str = "./data/adapters",
        registry_file: str = "./data/adapters/registry.json",
    ):
        self.adapters_dir = Path(adapters_dir)
        self.registry_file = Path(registry_file)

        # Create directories
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

        # Adapter registry (in-memory)
        self.adapters: Dict[str, AdapterInfo] = {}

        # Load existing registry
        self._load_registry()

        logger.info(f"Adapter Registry initialized")
        logger.info(f"  Directory: {self.adapters_dir}")
        logger.info(f"  Adapters: {len(self.adapters)}")

    def _load_registry(self):
        """Load adapter registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)

                for adapter_id, adapter_data in data.items():
                    self.adapters[adapter_id] = AdapterInfo(**adapter_data)

                logger.info(f"Loaded {len(self.adapters)} adapters from registry")

            except Exception as e:
                logger.error(f"Error loading registry: {e}")
        else:
            logger.info("No existing registry found, starting fresh")

    def _save_registry(self):
        """Save adapter registry to disk"""
        try:
            data = {
                adapter_id: adapter.model_dump()
                for adapter_id, adapter in self.adapters.items()
            }

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved registry with {len(self.adapters)} adapters")

        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    async def register_adapter(
        self,
        adapter_id: str,
        name: str,
        description: str,
        base_model: str,
        task: str,
        file_path: str,
        metrics: Optional[Dict[str, float]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdapterInfo:
        """
        Register a new adapter

        Args:
            adapter_id: Unique adapter ID
            name: Human-readable name
            description: Adapter description
            base_model: Base model name
            task: Task type (general, coding, writing, etc.)
            file_path: Path to adapter files
            metrics: Training/evaluation metrics
            lora_config: LoRA configuration used
            training_info: Training information
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            AdapterInfo
        """
        # Get file size
        adapter_path = Path(file_path)
        file_size = 0
        if adapter_path.exists():
            if adapter_path.is_dir():
                file_size = sum(
                    f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()
                )
            else:
                file_size = adapter_path.stat().st_size

        # Create adapter info
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            name=name,
            description=description,
            base_model=base_model,
            task=task,
            file_path=str(file_path),
            file_size=file_size,
            metrics=metrics or {},
            lora_config=lora_config or {},
            training_info=training_info or {},
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store in registry
        self.adapters[adapter_id] = adapter_info

        # Save to disk
        self._save_registry()

        logger.info(f"Registered adapter: {adapter_id} ({name})")
        return adapter_info

    async def get_adapter(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get adapter by ID"""
        adapter = self.adapters.get(adapter_id)

        if adapter:
            # Update usage stats
            adapter.last_used = datetime.utcnow().isoformat()
            adapter.usage_count += 1
            self._save_registry()

        return adapter

    async def list_adapters(
        self,
        base_model: Optional[str] = None,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[AdapterInfo]:
        """
        List adapters with optional filtering

        Args:
            base_model: Filter by base model
            task: Filter by task
            tags: Filter by tags (any match)
            limit: Maximum results

        Returns:
            List of AdapterInfo
        """
        adapters = list(self.adapters.values())

        # Apply filters
        if base_model:
            adapters = [a for a in adapters if a.base_model == base_model]

        if task:
            adapters = [a for a in adapters if a.task == task]

        if tags:
            adapters = [a for a in adapters if any(tag in a.tags for tag in tags)]

        # Sort by usage count (most used first)
        adapters.sort(key=lambda a: a.usage_count, reverse=True)

        return adapters[:limit]

    async def delete_adapter(self, adapter_id: str, delete_files: bool = True) -> bool:
        """
        Delete adapter from registry

        Args:
            adapter_id: Adapter ID
            delete_files: Also delete adapter files

        Returns:
            True if deleted, False if not found
        """
        if adapter_id not in self.adapters:
            logger.warning(f"Adapter not found: {adapter_id}")
            return False

        adapter = self.adapters[adapter_id]

        # Delete files if requested
        if delete_files:
            adapter_path = Path(adapter.file_path)
            if adapter_path.exists():
                try:
                    if adapter_path.is_dir():
                        shutil.rmtree(adapter_path)
                    else:
                        adapter_path.unlink()
                    logger.info(f"Deleted adapter files: {adapter_path}")
                except Exception as e:
                    logger.error(f"Error deleting adapter files: {e}")

        # Remove from registry
        del self.adapters[adapter_id]
        self._save_registry()

        logger.info(f"Deleted adapter: {adapter_id}")
        return True

    async def update_adapter_metadata(
        self, adapter_id: str, updates: Dict[str, Any]
    ) -> Optional[AdapterInfo]:
        """Update adapter metadata"""
        if adapter_id not in self.adapters:
            return None

        adapter = self.adapters[adapter_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(adapter, key):
                setattr(adapter, key, value)

        self._save_registry()

        logger.info(f"Updated adapter metadata: {adapter_id}")
        return adapter

    async def get_stats(self) -> AdapterStats:
        """Get adapter registry statistics"""
        adapters = list(self.adapters.values())

        # Count by task
        by_task = {}
        for adapter in adapters:
            by_task[adapter.task] = by_task.get(adapter.task, 0) + 1

        # Count by base model
        by_model = {}
        for adapter in adapters:
            by_model[adapter.base_model] = by_model.get(adapter.base_model, 0) + 1

        # Total size
        total_size = sum(a.file_size for a in adapters)

        # Most used
        sorted_adapters = sorted(adapters, key=lambda a: a.usage_count, reverse=True)
        most_used = [a.adapter_id for a in sorted_adapters[:5]]

        return AdapterStats(
            total_adapters=len(adapters),
            adapters_by_task=by_task,
            adapters_by_base_model=by_model,
            total_size=total_size,
            most_used=most_used,
        )

    async def search_adapters(self, query: str, limit: int = 10) -> List[AdapterInfo]:
        """
        Search adapters by name, description, or tags

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching adapters
        """
        query_lower = query.lower()
        results = []

        for adapter in self.adapters.values():
            # Search in name, description, tags
            if (
                query_lower in adapter.name.lower()
                or query_lower in adapter.description.lower()
                or any(query_lower in tag.lower() for tag in adapter.tags)
            ):
                results.append(adapter)

        # Sort by usage count
        results.sort(key=lambda a: a.usage_count, reverse=True)

        return results[:limit]


# ============================================================================
# Singleton Instance
# ============================================================================

adapter_registry = AdapterRegistry()

