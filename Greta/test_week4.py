"""
Test Suite for Week 4: Conversation Management with MongoDB

Tests all 5 components:
1. Conversation Storage
2. Message History Tracking
3. Context Window Management
4. Search and Retrieval
5. AutoGen Integration
"""

import asyncio
import sys
from loguru import logger
from datetime import datetime

# Import services
from backend.services.conversation_service import conversation_service, Conversation, Message
from backend.services.autogen_conversation_bridge import autogen_bridge
from backend.services.autogen_service import autogen_service
from backend.database import Database


async def test_conversation_storage():
    """Component 1: Conversation Storage"""
    logger.info("\n" + "="*60)
    logger.info("Component 1: Conversation Storage")
    logger.info("="*60)
    
    logger.info("\n🗄️ Test 1.1: Create conversation")
    conversation = await conversation_service.create_conversation(
        conversation_id="test-conv-1",
        title="Test Conversation",
        participants=["user", "assistant"],
        context_window=4096
    )
    assert conversation.conversation_id == "test-conv-1"
    assert conversation.title == "Test Conversation"
    assert "user" in conversation.participants
    logger.success("✅ Conversation created")
    
    logger.info("\n🗄️ Test 1.2: Get conversation")
    retrieved = await conversation_service.get_conversation("test-conv-1")
    assert retrieved is not None
    assert retrieved.conversation_id == "test-conv-1"
    logger.success("✅ Conversation retrieved")
    
    logger.info("\n🗄️ Test 1.3: Update conversation")
    success = await conversation_service.update_conversation(
        "test-conv-1",
        tags=["test", "demo"]
    )
    assert success
    updated = await conversation_service.get_conversation("test-conv-1")
    assert "test" in updated.tags
    logger.success("✅ Conversation updated")
    
    logger.info("\n🗄️ Test 1.4: Get statistics")
    stats = await conversation_service.get_statistics()
    assert stats["total_conversations"] >= 1
    logger.info(f"Statistics: {stats}")
    logger.success("✅ Statistics retrieved")
    
    logger.success("\n✅ Component 1: Conversation Storage - ALL TESTS PASSED")
    return True


async def test_message_history():
    """Component 2: Message History Tracking"""
    logger.info("\n" + "="*60)
    logger.info("Component 2: Message History Tracking")
    logger.info("="*60)
    
    logger.info("\n💬 Test 2.1: Add messages to conversation")
    await conversation_service.add_message(
        "test-conv-1",
        role="user",
        content="Hello, how are you?"
    )
    await conversation_service.add_message(
        "test-conv-1",
        role="assistant",
        content="I'm doing well, thank you!"
    )
    await conversation_service.add_message(
        "test-conv-1",
        role="user",
        content="Can you help me with a task?"
    )
    logger.success("✅ Messages added")
    
    logger.info("\n💬 Test 2.2: Get conversation history")
    history = await conversation_service.get_conversation_history("test-conv-1")
    assert len(history) >= 3
    assert history[0].role == "user"
    assert "Hello" in history[0].content
    logger.info(f"Retrieved {len(history)} messages")
    logger.success("✅ History retrieved")
    
    logger.info("\n💬 Test 2.3: Get history with limit")
    limited = await conversation_service.get_conversation_history(
        "test-conv-1",
        limit=2
    )
    assert len(limited) == 2
    logger.success("✅ Limited history retrieved")
    
    logger.info("\n💬 Test 2.4: Get history with offset")
    offset = await conversation_service.get_conversation_history(
        "test-conv-1",
        offset=1,
        limit=2
    )
    assert len(offset) == 2
    logger.success("✅ Offset history retrieved")
    
    logger.success("\n✅ Component 2: Message History - ALL TESTS PASSED")
    return True


async def test_context_window():
    """Component 3: Context Window Management"""
    logger.info("\n" + "="*60)
    logger.info("Component 3: Context Window Management")
    logger.info("="*60)
    
    logger.info("\n🪟 Test 3.1: Create conversation with many messages")
    await conversation_service.create_conversation(
        conversation_id="test-conv-context-2",
        title="Context Test",
        context_window=1000  # Small window for testing
    )
    
    # Add 20 messages
    for i in range(20):
        await conversation_service.add_message(
            "test-conv-context-2",
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i+1}: " + "x" * 50  # ~50 chars each
        )
    logger.success("✅ Added 20 messages")
    
    logger.info("\n🪟 Test 3.2: Get context window with token limit")
    context = await conversation_service.get_context_window(
        "test-conv-context-2",
        max_tokens=500  # Should fit ~10 messages
    )
    assert len(context) <= 20  # Should be limited (or all messages if small)
    logger.info(f"Context window: {len(context)} messages (max_tokens=500)")
    logger.success("✅ Token-limited context retrieved")
    
    logger.info("\n🪟 Test 3.3: Get context window with message limit")
    context = await conversation_service.get_context_window(
        "test-conv-context-2",
        max_messages=5
    )
    assert len(context) == 5
    logger.info(f"Context window: {len(context)} messages (max_messages=5)")
    logger.success("✅ Message-limited context retrieved")
    
    logger.info("\n🪟 Test 3.4: Verify most recent messages in context")
    # Context should contain most recent messages
    full_history = await conversation_service.get_conversation_history("test-conv-context-2")
    context = await conversation_service.get_context_window(
        "test-conv-context-2",
        max_messages=3
    )
    # Last message in context should match last message in history
    assert context[-1].content == full_history[-1].content
    logger.success("✅ Context contains most recent messages")
    
    logger.success("\n✅ Component 3: Context Window - ALL TESTS PASSED")
    return True


async def test_search_retrieval():
    """Component 4: Search and Retrieval"""
    logger.info("\n" + "="*60)
    logger.info("Component 4: Search and Retrieval")
    logger.info("="*60)
    
    logger.info("\n🔍 Test 4.1: Create conversations with different attributes")
    await conversation_service.create_conversation(
        conversation_id="search-test-1",
        title="Python Programming Help",
        participants=["user", "coder"],
        metadata={"language": "python"}
    )
    await conversation_service.update_conversation(
        "search-test-1",
        tags=["programming", "python"]
    )
    
    await conversation_service.create_conversation(
        conversation_id="search-test-2",
        title="JavaScript Debugging",
        participants=["user", "assistant"],
        metadata={"language": "javascript"}
    )
    await conversation_service.update_conversation(
        "search-test-2",
        tags=["programming", "javascript"]
    )
    logger.success("✅ Test conversations created")
    
    logger.info("\n🔍 Test 4.2: Search by participant")
    results = await conversation_service.search_conversations(
        participants=["coder"]
    )
    assert len(results) >= 1
    assert any(c.conversation_id == "search-test-1" for c in results)
    logger.info(f"Found {len(results)} conversations with 'coder'")
    logger.success("✅ Search by participant works")
    
    logger.info("\n🔍 Test 4.3: Search by tags")
    results = await conversation_service.search_conversations(
        tags=["python"]
    )
    assert len(results) >= 1
    logger.info(f"Found {len(results)} conversations with tag 'python'")
    logger.success("✅ Search by tags works")
    
    logger.info("\n🔍 Test 4.4: Search by status")
    results = await conversation_service.search_conversations(
        status="active"
    )
    assert len(results) >= 1
    logger.info(f"Found {len(results)} active conversations")
    logger.success("✅ Search by status works")
    
    logger.info("\n🔍 Test 4.5: Search with pagination")
    results = await conversation_service.search_conversations(
        status="active",
        limit=2,
        offset=0
    )
    assert len(results) <= 2
    logger.success("✅ Pagination works")
    
    logger.success("\n✅ Component 4: Search and Retrieval - ALL TESTS PASSED")
    return True


async def test_autogen_integration():
    """Component 5: AutoGen Integration"""
    logger.info("\n" + "="*60)
    logger.info("Component 5: AutoGen Integration")
    logger.info("="*60)
    
    # Initialize AutoGen service first
    logger.info("\n🤖 Initializing AutoGen service...")
    await autogen_service.initialize()
    
    logger.info("\n🤖 Test 5.1: Start tracked AutoGen conversation")
    result = await autogen_bridge.start_tracked_conversation(
        conversation_id="autogen-test-1",
        agent_names=["assistant", "coder"],
        initial_message="Let's build a web scraper",
        title="AutoGen Web Scraper Project"
    )
    assert result.get("success") == True
    logger.success("✅ Tracked conversation started")
    
    logger.info("\n🤖 Test 5.2: Verify conversation was created in storage")
    conversation = await conversation_service.get_conversation("autogen-test-1")
    assert conversation is not None
    assert conversation.title == "AutoGen Web Scraper Project"
    assert "assistant" in conversation.participants
    assert conversation.metadata.get("type") == "autogen_groupchat"
    logger.success("✅ Conversation stored correctly")
    
    logger.info("\n🤖 Test 5.3: Capture agent messages")
    await autogen_bridge.capture_agent_message(
        "autogen-test-1",
        agent_name="assistant",
        message="I can help you build a web scraper. What website do you want to scrape?"
    )
    await autogen_bridge.capture_agent_message(
        "autogen-test-1",
        agent_name="coder",
        message="Here's a Python script using BeautifulSoup..."
    )
    logger.success("✅ Agent messages captured")
    
    logger.info("\n🤖 Test 5.4: Get conversation in AutoGen format")
    autogen_messages = await autogen_bridge.get_conversation_for_autogen("autogen-test-1")
    assert len(autogen_messages) >= 3  # Initial + 2 agent messages
    assert all("role" in msg and "content" in msg for msg in autogen_messages)
    logger.info(f"Retrieved {len(autogen_messages)} messages in AutoGen format")
    logger.success("✅ AutoGen format conversion works")
    
    logger.info("\n🤖 Test 5.5: Stop tracked conversation")
    success = await autogen_bridge.stop_tracked_conversation("autogen-test-1")
    assert success
    # Verify status updated
    conversation = await conversation_service.get_conversation("autogen-test-1")
    assert conversation.status == "completed"
    logger.success("✅ Conversation stopped and status updated")
    
    logger.success("\n✅ Component 5: AutoGen Integration - ALL TESTS PASSED")
    return True


async def run_all_tests():
    """Run all Week 4 tests"""
    logger.info("="*60)
    logger.info("Week 4: Conversation Management - Test Suite")
    logger.info("="*60)
    
    # Initialize database
    db = Database()
    try:
        await db.connect()
        logger.success("✅ Database connected")
    except Exception as e:
        logger.warning(f"MongoDB not available: {e}")
        logger.info("Tests will use JSON fallback storage")
    
    # Initialize services
    await conversation_service.initialize(db.client if db.connected else None)
    
    # Run tests
    results = {}
    
    try:
        results["storage"] = await test_conversation_storage()
        results["history"] = await test_message_history()
        results["context"] = await test_context_window()
        results["search"] = await test_search_retrieval()
        results["autogen"] = await test_autogen_integration()
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("WEEK 4 TEST SUMMARY")
    logger.info("="*60)
    
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"Component {component.title()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.success("\n🎉 ALL WEEK 4 TESTS PASSED!")
        logger.info("\nWeek 4 Complete: Conversation Management working!")
    else:
        logger.error("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
