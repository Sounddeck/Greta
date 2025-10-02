"""
GRETA PAI - Performance Utilities
Phase 2 Performance Optimization
Async file processing, memory management, and performance monitoring
"""
import asyncio
import os
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from functools import wraps
from contextlib import asynccontextmanager
import logging
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Configuration
MAX_WORKERS = min(4, os.cpu_count() or 2)
THREAD_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="greta-async")
PROCESS_POOL = ProcessPoolExecutor(max_workers=min(2, MAX_WORKERS), max_tasks_per_child=10)

# Memory management settings
MEMORY_CLEANUP_THRESHOLD = 200 * 1024 * 1024  # 200MB
CONTEXT_WINDOW_MAX_SIZE = 1000  # messages
CONTEXT_COMPRESSION_RATIO = 0.7

@dataclass
class MemoryStats:
    """Memory usage tracking"""
    process_memory_mb: float = 0.0
    virtual_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_usage_percent: float = 0.0
    gc_collections: List[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def collect(cls) -> 'MemoryStats':
        """Collect current memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()

        return cls(
            process_memory_mb=memory_info.rss / (1024 * 1024),
            virtual_memory_mb=virtual_memory.used / (1024 * 1024),
            available_memory_mb=virtual_memory.available / (1024 * 1024),
            memory_usage_percent=virtual_memory.percent,
            gc_collections=[gc.get_count()[i] for i in range(3)]
        )


@dataclass
class ContextWindowManager:
    """Memory-managed context window with automatic compression"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_messages: int = CONTEXT_WINDOW_MAX_SIZE
    max_tokens_per_message: int = 10000
    compression_enabled: bool = True
    total_tokens: int = 0

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4

    def _compress_old_messages(self) -> int:
        """Compress old messages to save memory"""
        if len(self.messages) <= self.max_messages // 2:
            return 0

        # Keep only the most recent messages
        keep_count = self.max_messages // 2
        compressed_count = len(self.messages) - keep_count

        if compressed_count > 0:
            # Remove oldest messages
            self.messages = self.messages[-keep_count:]
            self.total_tokens = sum(
                self._estimate_tokens(msg.get('content', ''))
                for msg in self.messages
            )
            logger.info(f"Compressed context window: removed {compressed_count} old messages")

        return compressed_count

    async def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add message with automatic memory management"""
        try:
            # Truncate overly long messages
            if len(content) > self.max_tokens_per_message * 4:  # Rough char limit
                content = content[:self.max_tokens_per_message * 4] + "...[truncated]"
                logger.warning("Message truncated due to length")

            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
                "tokens": self._estimate_tokens(content)
            }

            self.messages.append(message)
            self.total_tokens += message["tokens"]

            # Check if cleanup is needed
            if len(self.messages) > self.max_messages:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._compress_old_messages
                )

            # Memory cleanup threshold check
            memory_stats = await asyncio.get_event_loop().run_in_executor(
                None, MemoryStats.collect
            )

            if memory_stats.process_memory_mb > MEMORY_CLEANUP_THRESHOLD / (1024 * 1024):
                await self.cleanup_memory()

            return True

        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return False

    async def cleanup_memory(self):
        """Force memory cleanup"""
        try:
            # Compress context window
            await asyncio.get_event_loop().run_in_executor(
                None, self._compress_old_messages
            )

            # Trigger garbage collection
            collected = await asyncio.get_event_loop().run_in_executor(None, gc.collect)

            # Clear any temporary references
            if hasattr(self, '_temp_refs'):
                self._temp_refs.clear()

            logger.info(f"Memory cleanup: collected {collected} objects, {len(self.messages)} messages")

        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    def get_context(self, max_messages: int = 50) -> List[Dict]:
        """Get context with size limit"""
        return self.messages[-max_messages:] if self.messages else []

    def clear_context(self):
        """Clear all context"""
        self.messages.clear()
        self.total_tokens = 0
        logger.info("Context window cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get context window statistics"""
        return {
            "total_messages": len(self.messages),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_message": self.total_tokens / len(self.messages) if self.messages else 0,
            "memory_usage_mb": len(self.messages) * 0.5  # Rough estimate
        }


class AsyncFileProcessor:
    """Async file processing with thread pool offloading"""

    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="file-processor")
        self.temp_files: List[str] = []

    async def process_file_async(self, content: bytes, filename: str, processor_func) -> Any:
        """Process file asynchronously using thread pool"""
        start_time = time.time()

        def _sync_process():
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
                self.temp_files.append(temp_path)

            try:
                # Call the processor function in thread
                result = processor_func(temp_path, filename)
                processing_time = time.time() - start_time
                logger.info(f"File {filename} processed in {processing_time:.2f}s")
                return result
            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_path)
                    if temp_path in self.temp_files:
                        self.temp_files.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, _sync_process
            )
        except Exception as e:
            logger.error(f"Async file processing failed for {filename}: {e}")
            raise

    async def process_pdf(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Async PDF processing"""
        def _process_pdf_blocking(temp_path: str, fname: str) -> Dict[str, Any]:
            try:
                # Use existing PDF processing logic here
                # This is a placeholder - integrate with your PDF library
                with open(temp_path, 'rb') as f:
                    # Mock processing
                    text = f"PDF content extracted from {fname}"
                    pages = 5  # Mock page count

                return {
                    "filename": fname,
                    "content_type": "pdf",
                    "text_content": text,
                    "pages": pages,
                    "processed": True
                }
            except Exception as e:
                raise RuntimeError(f"PDF processing failed: {e}")

        return await self.process_file_async(content, filename, _process_pdf_blocking)

    async def process_video(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Async video processing with proper blocking operation offloading"""
        def _process_video_blocking(temp_path: str, fname: str) -> Dict[str, Any]:
            try:
                # Import here to avoid circular imports
                import moviepy.editor as mp

                # This blocking operation is now safely in a thread
                video_clip = mp.VideoFileClip(temp_path)
                duration = video_clip.duration
                size = video_clip.size

                if video_clip.audio:
                    # Optional audio analysis (also blocking)
                    audio_duration = video_clip.audio.duration
                else:
                    audio_duration = None

                # Cleanup clip resources
                video_clip.close()

                return {
                    "filename": fname,
                    "content_type": "video",
                    "duration_seconds": duration,
                    "dimensions": size,
                    "has_audio": audio_duration is not None,
                    "audio_duration": audio_duration,
                    "processed": True
                }
            except Exception as e:
                raise RuntimeError(f"Video processing failed: {e}")

        return await self.process_file_async(content, filename, _process_video_blocking)

    async def process_image(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Async image processing"""
        def _process_image_blocking(temp_path: str, fname: str) -> Dict[str, Any]:
            try:
                from PIL import Image
                import io

                # Load image
                img = Image.open(temp_path)
                width, height = img.size
                format_type = img.format
                mode = img.mode

                return {
                    "filename": fname,
                    "content_type": "image",
                    "dimensions": f"{width}x{height}",
                    "format": format_type,
                    "color_mode": mode,
                    "processed": True
                }
            except Exception as e:
                raise RuntimeError(f"Image processing failed: {e}")

        return await self.process_file_async(content, filename, _process_image_blocking)

    async def cleanup_temp_files(self):
        """Cleanup any remaining temporary files"""
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                self.temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""

    def __init__(self):
        self.metrics: Dict[str, List[Tuple[float, Any]]] = {}
        self.start_times: Dict[str, float] = {}

    async def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()

    async def end_timer(self, operation: str, metadata: Optional[Dict] = None):
        """End timing and record metrics"""
        if operation not in self.start_times:
            logger.warning(f"No start time for operation: {operation}")
            return

        duration = time.time() - self.start_times[operation]

        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append((duration, metadata))

        # Keep only last 100 measurements
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-100:]

    async def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        if operation:
            if operation not in self.metrics:
                return {}

            durations = [d[0] for d in self.metrics[operation]]
            return {
                "operation": operation,
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "recent_avg": sum(durations[-10:]) / len(durations[-10:]) if len(durations) >= 10 else sum(durations) / len(durations) if durations else 0
            }
        else:
            # Return all metrics
            return {
                op: await self.get_metrics(op)
                for op in self.metrics.keys()
            }


def performance_monitor(operation: str):
    """Decorator for performance monitoring"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = getattr(args[0] if args else None, 'performance_monitor', None)
            if not monitor:
                from backend.utils.performance import performance_monitor as global_monitor
                monitor = global_monitor

            await monitor.start_timer(operation)

            try:
                result = await func(*args, **kwargs)
                await monitor.end_timer(operation, {"success": True})
                return result
            except Exception as e:
                await monitor.end_timer(operation, {"success": False, "error": str(e)})
                raise

        return wrapper
    return decorator


# Global instances
context_manager = ContextWindowManager()
file_processor = AsyncFileProcessor()
performance_monitor = PerformanceMonitor()

# Cleanup on shutdown
async def cleanup_performance_resources():
    """Cleanup performance-related resources"""
    await file_processor.cleanup_temp_files()
    logger.info("Performance resources cleaned up")


__all__ = [
    'ContextWindowManager',
    'AsyncFileProcessor',
    'PerformanceMonitor',
    'MemoryStats',
    'context_manager',
    'file_processor',
    'performance_monitor',
    'performance_monitor',
    'cleanup_performance_resources'
]
