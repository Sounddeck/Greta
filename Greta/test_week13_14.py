"""
Test Suite for Week 13-14: Context Management

Tests:
1. Context Window Management
2. Token Counting and Tracking
3. Context Compression
4. Context Relevance Scoring
5. Context Caching
6. Context Persistence
7. Multi-Modal Context Support
8. Context Merging
9. Adaptive Context Windows
10. Context Analytics

Target: 100% pass rate
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.context_management_service import (
    ContextManagementService,
    ContextWindow,
    ContextItem,
)

# Test results tracking
test_results = {"passed": 0, "failed": 0, "total": 0}


def print_test_header(component: str):
    """Print test component header"""
    print("\n" + "=" * 80)
    print(f"Component {component}")
    print("=" * 80 + "\n")


def print_test(test_num: int, total: int, description: str):
    """Print test description"""
    print(f"[Test {test_num}/{total}] {description}...")


def test_passed():
    """Mark test as passed"""
    test_results["passed"] += 1
    test_results["total"] += 1
    print("✅ PASSED\n")


def test_failed(error: str):
    """Mark test as failed"""
    test_results["failed"] += 1
    test_results["total"] += 1
    print(f"❌ FAILED: {error}\n")


# ============================================================================
# Component 1: Context Window Management
# ============================================================================


async def test_context_window_management():
    """Test context window creation and management"""
    print_test_header("1: Context Window Management")

    service = ContextManagementService()

    # Test 1: Create context window
    print_test(1, 5, "Create context window")
    try:
        window = await service.create_context_window(
            window_id="test_window_1", max_tokens=4096
        )
        assert window.window_id == "test_window_1"
        assert window.max_tokens == 4096
        print(f"   Created window: {window.window_id}, max_tokens={window.max_tokens}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Add context item
    print_test(2, 5, "Add context item")
    try:
        item = await service.add_context_item(
            window_id="test_window_1",
            content="This is a test context item.",
            content_type="text",
            priority=1,
        )
        assert item is not None
        assert item.content_type == "text"
        print(f"   Added item: {item.tokens} tokens")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Get context window
    print_test(3, 5, "Get context window")
    try:
        window = await service.get_context_window("test_window_1")
        assert window is not None
        assert len(window.items) == 1
        print(f"   Retrieved window with {len(window.items)} items")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Add multiple items
    print_test(4, 5, "Add multiple context items")
    try:
        for i in range(3):
            await service.add_context_item(
                window_id="test_window_1",
                content=f"Additional context item {i}",
                priority=i,
            )
        window = await service.get_context_window("test_window_1")
        assert len(window.items) == 4
        print(f"   Total items: {len(window.items)}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Remove context item
    print_test(5, 5, "Remove context item")
    try:
        window = await service.get_context_window("test_window_1")
        item_id = window.items[0].item_id
        result = await service.remove_context_item("test_window_1", item_id)
        assert result is True
        window = await service.get_context_window("test_window_1")
        assert len(window.items) == 3
        print(f"   Removed item, remaining: {len(window.items)}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 2: Token Counting
# ============================================================================


async def test_token_counting():
    """Test token counting and tracking"""
    print_test_header("2: Token Counting and Tracking")

    service = ContextManagementService()

    # Test 1: Count tokens in text
    print_test(1, 3, "Count tokens in text")
    try:
        text = "This is a test sentence with multiple words."
        tokens = service.count_tokens(text)
        assert tokens > 0
        print(f"   Text: '{text[:30]}...' -> {tokens} tokens")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Track total tokens in window
    print_test(2, 3, "Track total tokens in context window")
    try:
        window = await service.create_context_window("test_tokens", max_tokens=1000)
        await service.add_context_item("test_tokens", "Short text", priority=1)
        await service.add_context_item("test_tokens", "Another short text", priority=1)

        window = await service.get_context_window("test_tokens")
        assert window.total_tokens > 0
        print(f"   Total tokens: {window.total_tokens}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Verify token counting accuracy
    print_test(3, 3, "Verify token counting")
    try:
        window = await service.get_context_window("test_tokens")
        manual_count = sum(item.tokens for item in window.items)
        assert window.total_tokens == manual_count
        print(f"   Verified: {window.total_tokens} == {manual_count}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 3: Context Compression
# ============================================================================


async def test_context_compression():
    """Test context compression"""
    print_test_header("3: Context Compression")

    service = ContextManagementService()

    # Test 1: Create window with many items
    print_test(1, 3, "Create window exceeding token limit")
    try:
        window = await service.create_context_window("test_compress", max_tokens=200)

        # Add items that exceed limit
        for i in range(10):
            await service.add_context_item(
                "test_compress",
                f"This is a longer context item number {i} with more content to increase token count.",
                priority=i % 3,  # Varying priorities
            )

        window = await service.get_context_window("test_compress")
        initial_tokens = window.total_tokens
        print(f"   Initial tokens: {initial_tokens}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Compress context
    print_test(2, 3, "Compress context to fit limit")
    try:
        window = await service.compress_context("test_compress", target_tokens=100)
        assert window is not None
        assert window.total_tokens <= 100
        assert window.is_compressed is True
        print(f"   Compressed to: {window.total_tokens} tokens ({len(window.items)} items)")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Verify compression ratio
    print_test(3, 3, "Verify compression ratio")
    try:
        window = await service.get_context_window("test_compress")
        assert window.compression_ratio > 0
        print(f"   Compression ratio: {window.compression_ratio:.2f}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 4: Context Relevance Scoring
# ============================================================================


async def test_relevance_scoring():
    """Test context relevance scoring"""
    print_test_header("4: Context Relevance Scoring")

    service = ContextManagementService()

    # Test 1: Create window with varied content
    print_test(1, 2, "Create window with varied content")
    try:
        window = await service.create_context_window("test_relevance", max_tokens=2000)

        await service.add_context_item(
            "test_relevance", "Python programming language tutorial"
        )
        await service.add_context_item(
            "test_relevance", "JavaScript web development guide"
        )
        await service.add_context_item(
            "test_relevance", "Python data science with pandas"
        )

        window = await service.get_context_window("test_relevance")
        print(f"   Created window with {len(window.items)} items")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Score relevance to query
    print_test(2, 2, "Score relevance to query")
    try:
        window = await service.score_relevance("test_relevance", "Python programming")
        assert window is not None

        # Check that items are sorted by relevance
        relevance_scores = [item.relevance_score for item in window.items]
        print(f"   Relevance scores: {[f'{s:.2f}' for s in relevance_scores]}")

        # First item should have highest relevance
        assert relevance_scores[0] >= relevance_scores[-1]
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 5: Context Caching
# ============================================================================


async def test_context_caching():
    """Test context caching"""
    print_test_header("5: Context Caching")

    service = ContextManagementService()

    # Test 1: Cache context
    print_test(1, 4, "Cache context window")
    try:
        window = await service.create_context_window("test_cache", max_tokens=1000)
        await service.add_context_item("test_cache", "Cached content")

        cache_key = "test_cache_key"
        result = await service.cache_context(cache_key, window)
        assert result is True
        print(f"   Cached context: {cache_key}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Retrieve cached context
    print_test(2, 4, "Retrieve cached context")
    try:
        cached_window = await service.get_cached_context("test_cache_key")
        assert cached_window is not None
        assert cached_window.window_id == "test_cache"
        print(f"   Retrieved cached window: {cached_window.window_id}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Cache miss
    print_test(3, 4, "Test cache miss")
    try:
        cached_window = await service.get_cached_context("nonexistent_key")
        assert cached_window is None
        print(f"   Cache miss handled correctly")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Clear cache
    print_test(4, 4, "Clear cache")
    try:
        count = await service.clear_cache()
        assert count >= 0
        print(f"   Cleared {count} cached contexts")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 6: Context Persistence
# ============================================================================


async def test_context_persistence():
    """Test context persistence"""
    print_test_header("6: Context Persistence")

    service = ContextManagementService()

    # Test 1: Save context
    print_test(1, 3, "Save context to disk")
    try:
        window = await service.create_context_window("test_persist", max_tokens=1000)
        await service.add_context_item("test_persist", "Persistent content")

        file_path = await service.save_context("test_persist")
        assert file_path is not None
        assert Path(file_path).exists()
        print(f"   Saved to: {file_path}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Load context
    print_test(2, 3, "Load context from disk")
    try:
        # Create new service instance to test loading
        service2 = ContextManagementService()
        loaded_window = await service2.load_context("test_persist")
        assert loaded_window is not None
        assert loaded_window.window_id == "test_persist"
        print(f"   Loaded window: {loaded_window.window_id}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Export context
    print_test(3, 3, "Export context to file")
    try:
        export_path = await service.export_context("test_persist", format="json")
        assert export_path is not None
        assert Path(export_path).exists()
        print(f"   Exported to: {export_path}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 7: Multi-Modal Context Support
# ============================================================================


async def test_multimodal_context():
    """Test multi-modal context support"""
    print_test_header("7: Multi-Modal Context Support")

    service = ContextManagementService()

    # Test 1: Add code context
    print_test(1, 3, "Add code context")
    try:
        window = await service.create_context_window("test_multimodal", max_tokens=2000)

        code_item = await service.add_code_context(
            "test_multimodal",
            "def hello():\n    print('Hello, World!')",
            language="python",
        )
        assert code_item is not None
        assert code_item.content_type == "code"
        print(f"   Added code context: {code_item.metadata.get('language')}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Add structured data context
    print_test(2, 3, "Add structured data context")
    try:
        data_item = await service.add_structured_data_context(
            "test_multimodal", {"name": "Test", "value": 123}
        )
        assert data_item is not None
        assert data_item.content_type == "structured_data"
        print(f"   Added structured data context")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Add image context
    print_test(3, 3, "Add image context")
    try:
        image_item = await service.add_image_context(
            "test_multimodal", "/path/to/image.png", description="Test image"
        )
        assert image_item is not None
        assert image_item.content_type == "image"
        print(f"   Added image context: {image_item.metadata.get('image_path')}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 8: Context Merging
# ============================================================================


async def test_context_merging():
    """Test context merging"""
    print_test_header("8: Context Merging")

    service = ContextManagementService()

    # Test 1: Create multiple windows
    print_test(1, 2, "Create multiple context windows")
    try:
        window1 = await service.create_context_window("merge_1", max_tokens=1000)
        await service.add_context_item("merge_1", "Content from window 1")

        window2 = await service.create_context_window("merge_2", max_tokens=1000)
        await service.add_context_item("merge_2", "Content from window 2")

        window3 = await service.create_context_window("merge_3", max_tokens=1000)
        await service.add_context_item("merge_3", "Content from window 3")

        print(f"   Created 3 windows for merging")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Merge contexts
    print_test(2, 2, "Merge multiple contexts")
    try:
        merged = await service.merge_contexts(
            ["merge_1", "merge_2", "merge_3"], "merged_window", max_tokens=3000
        )
        assert merged is not None
        assert len(merged.items) >= 3
        print(f"   Merged into window with {len(merged.items)} items")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 9: Adaptive Context Windows
# ============================================================================


async def test_adaptive_windows():
    """Test adaptive context windows"""
    print_test_header("9: Adaptive Context Windows")

    service = ContextManagementService()

    # Test 1: Create window and adjust for simple task
    print_test(1, 3, "Adjust window for simple task")
    try:
        window = await service.create_context_window("adaptive_1", max_tokens=4096)
        adjusted = await service.adjust_context_window("adaptive_1", "simple")
        assert adjusted is not None
        assert adjusted.max_tokens == 2048
        print(f"   Adjusted to simple: {adjusted.max_tokens} tokens")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Adjust for complex task
    print_test(2, 3, "Adjust window for complex task")
    try:
        adjusted = await service.adjust_context_window("adaptive_1", "complex")
        assert adjusted is not None
        assert adjusted.max_tokens == 8192
        print(f"   Adjusted to complex: {adjusted.max_tokens} tokens")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Adjust for very complex task
    print_test(3, 3, "Adjust window for very complex task")
    try:
        adjusted = await service.adjust_context_window("adaptive_1", "very_complex")
        assert adjusted is not None
        assert adjusted.max_tokens == 16384
        print(f"   Adjusted to very_complex: {adjusted.max_tokens} tokens")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 10: Context Analytics
# ============================================================================


async def test_context_analytics():
    """Test context analytics"""
    print_test_header("10: Context Analytics")

    service = ContextManagementService()

    # Test 1: Get context statistics
    print_test(1, 3, "Get context statistics")
    try:
        stats = await service.get_context_stats()
        assert stats.total_contexts >= 0
        print(f"   Total contexts: {stats.total_contexts}")
        print(f"   Total tokens: {stats.total_tokens}")
        print(f"   Cache hit rate: {stats.cache_hit_rate:.2%}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Analyze context usage
    print_test(2, 3, "Analyze context usage")
    try:
        # Create a window with varied content
        window = await service.create_context_window("analytics_test", max_tokens=2000)
        await service.add_context_item("analytics_test", "Text content", priority=1)
        await service.add_code_context("analytics_test", "print('hello')", priority=2)

        analysis = await service.analyze_context_usage("analytics_test")
        assert analysis is not None
        assert "total_items" in analysis
        assert "utilization" in analysis
        print(f"   Items: {analysis['total_items']}, Utilization: {analysis['utilization']:.2%}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Get optimization suggestions
    print_test(3, 3, "Get optimization suggestions")
    try:
        suggestions = await service.get_optimization_suggestions("analytics_test")
        assert suggestions is not None
        assert isinstance(suggestions, list)
        print(f"   Suggestions: {len(suggestions)}")
        if suggestions:
            print(f"   Example: {suggestions[0][:60]}...")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Main Test Runner
# ============================================================================


async def run_all_tests():
    """Run all test components"""
    print("\n" + "=" * 80)
    print("WEEK 13-14 TEST SUITE: Context Management")
    print("=" * 80)
    print("Testing all context management components...\n")

    # Run test components
    await test_context_window_management()
    await test_token_counting()
    await test_context_compression()
    await test_relevance_scoring()
    await test_context_caching()
    await test_context_persistence()
    await test_multimodal_context()
    await test_context_merging()
    await test_adaptive_windows()
    await test_context_analytics()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")

    if test_results["failed"] == 0:
        print("\n🎉 ALL TESTS PASSED! 100% Success Rate")
    else:
        success_rate = (
            test_results["passed"] / test_results["total"] * 100
            if test_results["total"] > 0
            else 0
        )
        print(f"\n⚠️  Success Rate: {success_rate:.1f}%")

    print("=" * 80 + "\n")

    return test_results["failed"] == 0


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)

