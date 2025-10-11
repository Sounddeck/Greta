"""
Context Management Service - Week 13-14 Implementation

Intelligent context optimization, compression, caching, and multi-modal support.
Integrates with all existing services for optimal context usage.
"""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Schemas
# ============================================================================


class ContextItem(BaseModel):
    """Individual context item"""

    item_id: str
    content: str
    content_type: str = "text"  # text, code, image, structured_data
    tokens: int = 0
    relevance_score: float = 1.0
    priority: int = 0  # Higher priority = more important
    source: str = "user"  # user, system, rag, agent, conversation
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ContextWindow(BaseModel):
    """Context window with items and metadata"""

    window_id: str
    items: List[ContextItem] = Field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 4096
    compression_ratio: float = 1.0
    is_compressed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ContextCache(BaseModel):
    """Cached context entry"""

    cache_key: str
    context_window: ContextWindow
    hit_count: int = 0
    last_accessed: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None


class ContextStats(BaseModel):
    """Context usage statistics"""

    total_contexts: int
    total_tokens: int
    avg_tokens_per_context: float
    compression_ratio: float
    cache_hit_rate: float
    contexts_by_type: Dict[str, int]
    contexts_by_source: Dict[str, int]


# ============================================================================
# Context Management Service
# ============================================================================


class ContextManagementService:
    """
    Context Management Service

    Features:
    - Context window management
    - Context compression and optimization
    - Context caching
    - Context persistence
    - Multi-modal context support
    - Context relevance scoring
    - Adaptive context windows
    - Context analytics
    """

    def __init__(
        self,
        contexts_dir: str = "./data/contexts",
        cache_dir: str = "./data/contexts/cache",
        max_cache_size: int = 100,
    ):
        self.contexts_dir = Path(contexts_dir)
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size

        # Create directories
        self.contexts_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.contexts: Dict[str, ContextWindow] = {}
        self.cache: Dict[str, ContextCache] = {}

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Context Management Service initialized")
        logger.info(f"  Contexts directory: {self.contexts_dir}")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Max cache size: {self.max_cache_size}")

    # ========================================================================
    # Token Counting
    # ========================================================================

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Simple estimation: ~4 characters per token
        In production, use tiktoken or similar
        """
        return len(text) // 4

    def count_context_tokens(self, context_window: ContextWindow) -> int:
        """Count total tokens in context window"""
        total = sum(item.tokens for item in context_window.items)
        context_window.total_tokens = total
        return total

    # ========================================================================
    # Context Window Management
    # ========================================================================

    async def create_context_window(
        self,
        window_id: str,
        max_tokens: int = 4096,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextWindow:
        """Create a new context window"""
        window = ContextWindow(
            window_id=window_id,
            max_tokens=max_tokens,
            metadata=metadata or {},
        )

        self.contexts[window_id] = window
        logger.info(f"Created context window: {window_id} (max_tokens={max_tokens})")
        return window

    async def add_context_item(
        self,
        window_id: str,
        content: str,
        content_type: str = "text",
        priority: int = 0,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ContextItem]:
        """Add item to context window"""
        if window_id not in self.contexts:
            logger.error(f"Context window not found: {window_id}")
            return None

        window = self.contexts[window_id]

        # Create context item
        item_id = f"{window_id}_item_{len(window.items)}"
        tokens = self.count_tokens(content)

        item = ContextItem(
            item_id=item_id,
            content=content,
            content_type=content_type,
            tokens=tokens,
            priority=priority,
            source=source,
            metadata=metadata or {},
        )

        # Add to window
        window.items.append(item)
        window.total_tokens = self.count_context_tokens(window)
        window.updated_at = datetime.utcnow().isoformat()

        logger.debug(f"Added item to {window_id}: {tokens} tokens")
        return item

    async def get_context_window(self, window_id: str) -> Optional[ContextWindow]:
        """Get context window by ID"""
        return self.contexts.get(window_id)

    async def remove_context_item(
        self, window_id: str, item_id: str
    ) -> bool:
        """Remove item from context window"""
        if window_id not in self.contexts:
            return False

        window = self.contexts[window_id]
        window.items = [item for item in window.items if item.item_id != item_id]
        window.total_tokens = self.count_context_tokens(window)
        window.updated_at = datetime.utcnow().isoformat()

        logger.debug(f"Removed item {item_id} from {window_id}")
        return True

    # ========================================================================
    # Context Compression
    # ========================================================================

    async def compress_context(
        self, window_id: str, target_tokens: Optional[int] = None
    ) -> Optional[ContextWindow]:
        """
        Compress context window to fit within token limit

        Strategies:
        1. Remove low-priority items
        2. Summarize long items
        3. Remove redundant items
        """
        if window_id not in self.contexts:
            return None

        window = self.contexts[window_id]
        target = target_tokens or window.max_tokens
        original_tokens = window.total_tokens

        if window.total_tokens <= target:
            return window  # Already within limit

        logger.info(f"Compressing context {window_id}: {window.total_tokens} -> {target} tokens")

        # Strategy 1: Remove low-priority items
        sorted_items = sorted(window.items, key=lambda x: (x.priority, x.relevance_score), reverse=True)

        compressed_items = []
        current_tokens = 0

        for item in sorted_items:
            if current_tokens + item.tokens <= target:
                compressed_items.append(item)
                current_tokens += item.tokens
            else:
                # Try to summarize if it's a long text item
                if item.content_type == "text" and item.tokens > 100:
                    summary = await self._summarize_text(item.content, max_tokens=50)
                    item.content = summary
                    item.tokens = self.count_tokens(summary)
                    item.metadata["summarized"] = True

                    if current_tokens + item.tokens <= target:
                        compressed_items.append(item)
                        current_tokens += item.tokens

        window.items = compressed_items
        window.total_tokens = current_tokens
        window.is_compressed = True
        window.compression_ratio = current_tokens / original_tokens if original_tokens > 0 else 1.0
        window.updated_at = datetime.utcnow().isoformat()

        logger.info(f"Compressed to {current_tokens} tokens ({len(compressed_items)} items)")
        return window

    async def _summarize_text(self, text: str, max_tokens: int = 50) -> str:
        """
        Summarize text to fit within token limit

        Simple implementation: take first sentences
        In production, use LLM for summarization
        """
        sentences = text.split(". ")
        summary = ""
        tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            if tokens + sentence_tokens <= max_tokens:
                summary += sentence + ". "
                tokens += sentence_tokens
            else:
                break

        return summary.strip() or text[:max_tokens * 4]  # Fallback to character limit

    # ========================================================================
    # Context Relevance Scoring
    # ========================================================================

    async def score_relevance(
        self, window_id: str, query: str
    ) -> Optional[ContextWindow]:
        """
        Score context items by relevance to query

        Simple implementation: keyword matching
        In production, use embeddings and semantic similarity
        """
        if window_id not in self.contexts:
            return None

        window = self.contexts[window_id]
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for item in window.items:
            content_lower = item.content.lower()
            content_words = set(content_lower.split())

            # Calculate overlap
            overlap = len(query_words & content_words)
            total = len(query_words | content_words)

            item.relevance_score = overlap / total if total > 0 else 0.0

        # Sort by relevance
        window.items.sort(key=lambda x: x.relevance_score, reverse=True)
        window.updated_at = datetime.utcnow().isoformat()

        logger.debug(f"Scored relevance for {window_id} with query: {query[:50]}")
        return window

    # ========================================================================
    # Context Caching
    # ========================================================================

    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def cache_context(
        self, cache_key: str, context_window: ContextWindow, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache context window"""
        # Check cache size limit
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used
            lru_key = min(self.cache.items(), key=lambda x: x[1].last_accessed)[0]
            del self.cache[lru_key]
            logger.debug(f"Evicted LRU cache entry: {lru_key}")

        # Calculate expiration
        expires_at = None
        if ttl_seconds:
            from datetime import timedelta
            expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()

        # Create cache entry
        cache_entry = ContextCache(
            cache_key=cache_key,
            context_window=context_window,
            expires_at=expires_at,
        )

        self.cache[cache_key] = cache_entry
        logger.debug(f"Cached context: {cache_key}")
        return True

    async def get_cached_context(self, cache_key: str) -> Optional[ContextWindow]:
        """Get cached context window"""
        if cache_key not in self.cache:
            self.cache_misses += 1
            return None

        cache_entry = self.cache[cache_key]

        # Check expiration
        if cache_entry.expires_at:
            expires = datetime.fromisoformat(cache_entry.expires_at)
            if datetime.utcnow() > expires:
                del self.cache[cache_key]
                self.cache_misses += 1
                logger.debug(f"Cache expired: {cache_key}")
                return None

        # Update stats
        cache_entry.hit_count += 1
        cache_entry.last_accessed = datetime.utcnow().isoformat()
        self.cache_hits += 1

        logger.debug(f"Cache hit: {cache_key} (hits={cache_entry.hit_count})")
        return cache_entry.context_window

    async def clear_cache(self) -> int:
        """Clear all cached contexts"""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} cached contexts")
        return count

    # ========================================================================
    # Context Persistence
    # ========================================================================

    async def save_context(self, window_id: str) -> Optional[str]:
        """Save context window to disk"""
        if window_id not in self.contexts:
            return None

        window = self.contexts[window_id]
        file_path = self.contexts_dir / f"{window_id}.json"

        try:
            with open(file_path, "w") as f:
                json.dump(window.model_dump(), f, indent=2)

            logger.info(f"Saved context to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error saving context: {e}")
            return None

    async def load_context(self, window_id: str) -> Optional[ContextWindow]:
        """Load context window from disk"""
        file_path = self.contexts_dir / f"{window_id}.json"

        if not file_path.exists():
            logger.warning(f"Context file not found: {file_path}")
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            window = ContextWindow(**data)
            self.contexts[window_id] = window

            logger.info(f"Loaded context from {file_path}")
            return window

        except Exception as e:
            logger.error(f"Error loading context: {e}")
            return None

    async def export_context(
        self, window_id: str, format: str = "json"
    ) -> Optional[str]:
        """Export context window to file"""
        if window_id not in self.contexts:
            return None

        window = self.contexts[window_id]
        export_dir = self.contexts_dir / "exports"
        export_dir.mkdir(exist_ok=True)

        if format == "json":
            file_path = export_dir / f"{window_id}.json"
            with open(file_path, "w") as f:
                json.dump(window.model_dump(), f, indent=2)
        elif format == "text":
            file_path = export_dir / f"{window_id}.txt"
            with open(file_path, "w") as f:
                for item in window.items:
                    f.write(f"[{item.content_type}] {item.content}\n\n")
        else:
            logger.error(f"Unsupported format: {format}")
            return None

        logger.info(f"Exported context to {file_path}")
        return str(file_path)

    async def import_context(self, file_path: str) -> Optional[ContextWindow]:
        """Import context window from file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            window = ContextWindow(**data)
            self.contexts[window.window_id] = window

            logger.info(f"Imported context: {window.window_id}")
            return window

        except Exception as e:
            logger.error(f"Error importing context: {e}")
            return None

    # ========================================================================
    # Multi-Modal Context Support
    # ========================================================================

    async def add_code_context(
        self,
        window_id: str,
        code: str,
        language: str = "python",
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ContextItem]:
        """Add code to context window"""
        meta = metadata or {}
        meta["language"] = language

        return await self.add_context_item(
            window_id=window_id,
            content=code,
            content_type="code",
            priority=priority,
            source="code",
            metadata=meta,
        )

    async def add_structured_data_context(
        self,
        window_id: str,
        data: Dict[str, Any],
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ContextItem]:
        """Add structured data to context window"""
        # Convert to JSON string
        content = json.dumps(data, indent=2)

        return await self.add_context_item(
            window_id=window_id,
            content=content,
            content_type="structured_data",
            priority=priority,
            source="data",
            metadata=metadata,
        )

    async def add_image_context(
        self,
        window_id: str,
        image_path: str,
        description: str = "",
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ContextItem]:
        """Add image reference to context window"""
        meta = metadata or {}
        meta["image_path"] = image_path

        content = f"[Image: {image_path}]"
        if description:
            content += f" {description}"

        return await self.add_context_item(
            window_id=window_id,
            content=content,
            content_type="image",
            priority=priority,
            source="image",
            metadata=meta,
        )

    # ========================================================================
    # Context Merging
    # ========================================================================

    async def merge_contexts(
        self,
        window_ids: List[str],
        target_window_id: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[ContextWindow]:
        """
        Merge multiple context windows into one

        Strategies:
        1. Combine all items
        2. Remove duplicates
        3. Sort by priority and relevance
        4. Compress if needed
        """
        # Get all windows
        windows = []
        for wid in window_ids:
            window = self.contexts.get(wid)
            if window:
                windows.append(window)

        if not windows:
            logger.error("No valid windows to merge")
            return None

        # Create target window
        target_max_tokens = max_tokens or max(w.max_tokens for w in windows)
        target_window = await self.create_context_window(
            target_window_id, max_tokens=target_max_tokens
        )

        # Collect all items
        all_items = []
        seen_content = set()

        for window in windows:
            for item in window.items:
                # Remove duplicates based on content hash
                content_hash = hashlib.md5(item.content.encode()).hexdigest()
                if content_hash not in seen_content:
                    all_items.append(item)
                    seen_content.add(content_hash)

        # Sort by priority and relevance
        all_items.sort(key=lambda x: (x.priority, x.relevance_score), reverse=True)

        # Add items to target window
        target_window.items = all_items
        target_window.total_tokens = sum(item.tokens for item in all_items)

        # Compress if needed
        if target_window.total_tokens > target_window.max_tokens:
            await self.compress_context(target_window_id, target_window.max_tokens)

        logger.info(f"Merged {len(window_ids)} contexts into {target_window_id}")
        return target_window

    # ========================================================================
    # Adaptive Context Windows
    # ========================================================================

    async def adjust_context_window(
        self, window_id: str, task_complexity: str = "medium"
    ) -> Optional[ContextWindow]:
        """
        Adjust context window size based on task complexity

        Complexity levels:
        - simple: 2048 tokens
        - medium: 4096 tokens
        - complex: 8192 tokens
        - very_complex: 16384 tokens
        """
        if window_id not in self.contexts:
            return None

        window = self.contexts[window_id]

        complexity_map = {
            "simple": 2048,
            "medium": 4096,
            "complex": 8192,
            "very_complex": 16384,
        }

        new_max_tokens = complexity_map.get(task_complexity, 4096)
        window.max_tokens = new_max_tokens

        # Compress if current tokens exceed new limit
        if window.total_tokens > new_max_tokens:
            await self.compress_context(window_id, new_max_tokens)

        logger.info(f"Adjusted {window_id} to {task_complexity} ({new_max_tokens} tokens)")
        return window

    # ========================================================================
    # Context Analytics
    # ========================================================================

    async def get_context_stats(self) -> ContextStats:
        """Get context usage statistics"""
        total_contexts = len(self.contexts)
        total_tokens = sum(w.total_tokens for w in self.contexts.values())
        avg_tokens = total_tokens / total_contexts if total_contexts > 0 else 0

        # Compression ratio
        compressed_windows = [w for w in self.contexts.values() if w.is_compressed]
        avg_compression = (
            sum(w.compression_ratio for w in compressed_windows) / len(compressed_windows)
            if compressed_windows
            else 1.0
        )

        # Cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        )

        # Contexts by type
        contexts_by_type = {}
        contexts_by_source = {}

        for window in self.contexts.values():
            for item in window.items:
                contexts_by_type[item.content_type] = (
                    contexts_by_type.get(item.content_type, 0) + 1
                )
                contexts_by_source[item.source] = (
                    contexts_by_source.get(item.source, 0) + 1
                )

        return ContextStats(
            total_contexts=total_contexts,
            total_tokens=total_tokens,
            avg_tokens_per_context=avg_tokens,
            compression_ratio=avg_compression,
            cache_hit_rate=cache_hit_rate,
            contexts_by_type=contexts_by_type,
            contexts_by_source=contexts_by_source,
        )

    async def analyze_context_usage(self, window_id: str) -> Optional[Dict[str, Any]]:
        """Analyze context window usage"""
        if window_id not in self.contexts:
            return None

        window = self.contexts[window_id]

        analysis = {
            "window_id": window_id,
            "total_items": len(window.items),
            "total_tokens": window.total_tokens,
            "max_tokens": window.max_tokens,
            "utilization": window.total_tokens / window.max_tokens if window.max_tokens > 0 else 0,
            "is_compressed": window.is_compressed,
            "compression_ratio": window.compression_ratio,
            "items_by_type": {},
            "items_by_source": {},
            "items_by_priority": {},
            "avg_relevance_score": 0.0,
        }

        # Analyze items
        for item in window.items:
            # By type
            analysis["items_by_type"][item.content_type] = (
                analysis["items_by_type"].get(item.content_type, 0) + 1
            )

            # By source
            analysis["items_by_source"][item.source] = (
                analysis["items_by_source"].get(item.source, 0) + 1
            )

            # By priority
            analysis["items_by_priority"][item.priority] = (
                analysis["items_by_priority"].get(item.priority, 0) + 1
            )

        # Average relevance
        if window.items:
            analysis["avg_relevance_score"] = sum(
                item.relevance_score for item in window.items
            ) / len(window.items)

        return analysis

    async def get_optimization_suggestions(
        self, window_id: str
    ) -> Optional[List[str]]:
        """Get suggestions for optimizing context window"""
        analysis = await self.analyze_context_usage(window_id)
        if not analysis:
            return None

        suggestions = []

        # Check utilization
        if analysis["utilization"] > 0.9:
            suggestions.append("Context window is near capacity. Consider compression or increasing max_tokens.")

        if analysis["utilization"] < 0.3:
            suggestions.append("Context window is underutilized. Consider reducing max_tokens to save resources.")

        # Check compression
        if not analysis["is_compressed"] and analysis["total_tokens"] > 2048:
            suggestions.append("Context could benefit from compression to reduce token usage.")

        # Check relevance
        if analysis["avg_relevance_score"] < 0.5:
            suggestions.append("Average relevance score is low. Consider re-scoring or pruning irrelevant items.")

        # Check diversity
        if len(analysis["items_by_type"]) == 1:
            suggestions.append("Context contains only one content type. Consider adding multi-modal content.")

        return suggestions


# ============================================================================
# Singleton Instance
# ============================================================================

context_management_service = ContextManagementService()

