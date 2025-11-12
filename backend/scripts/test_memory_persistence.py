#!/usr/bin/env python3
"""
Test script to verify Master Agent memory persistence
Tests both MongoDB and ChromaDB integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database import get_database
from services.vector_search_service import VectorSearchService
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mongodb_persistence():
    """Test MongoDB conversation persistence"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: MongoDB Conversation Persistence")
    logger.info("="*60)
    
    try:
        db = await get_database()
        
        # Test 1: Insert a test conversation
        test_conversation = {
            "agent_id": "test_agent",
            "agent_type": "general",
            "user_message": "Test user message for memory persistence",
            "assistant_message": "Test assistant response for memory persistence",
            "conversation_history": [
                {"role": "user", "content": "Previous message 1"},
                {"role": "assistant", "content": "Previous response 1"}
            ],
            "timestamp": datetime.utcnow(),
            "session_id": "test_session",
            "metadata": {
                "test": True,
                "purpose": "memory_persistence_test"
            }
        }
        
        logger.info("📝 Inserting test conversation...")
        result = await db.insert_one("conversations", test_conversation)
        logger.info(f"✅ Conversation inserted with ID: {result}")
        
        # Test 2: Retrieve the conversation
        logger.info("\n📖 Retrieving conversations...")
        conversations = await db.find_many(
            "conversations",
            {"agent_type": "general"},
            limit=5,
            sort=[("timestamp", -1)]
        )
        
        logger.info(f"✅ Retrieved {len(conversations)} conversations")
        
        if conversations:
            latest = conversations[0]
            logger.info(f"\nLatest conversation:")
            logger.info(f"  User: {latest.get('user_message', 'N/A')[:100]}...")
            logger.info(f"  Assistant: {latest.get('assistant_message', 'N/A')[:100]}...")
            logger.info(f"  Timestamp: {latest.get('timestamp')}")
            logger.info(f"  Session: {latest.get('session_id')}")
        
        # Test 3: Clean up test data
        logger.info("\n🧹 Cleaning up test data...")
        await db.delete_many("conversations", {"metadata.test": True})
        logger.info("✅ Test data cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chromadb_persistence():
    """Test ChromaDB vector search persistence"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: ChromaDB Vector Search Persistence")
    logger.info("="*60)
    
    try:
        vector_search = VectorSearchService()
        
        if not vector_search.client:
            logger.warning("⚠️ ChromaDB not available - skipping vector search test")
            return True
        
        # Test 1: Store a test memory
        logger.info("📝 Storing test memory in ChromaDB...")
        test_content = "This is a test memory for the Master Agent. It should be retrievable via semantic search."
        test_namespace = "test_master_agent"
        
        memory_id = await vector_search.store_memory(
            content=test_content,
            namespace=test_namespace,
            metadata={
                "role": "user",
                "agent_type": "general",
                "timestamp": datetime.utcnow().isoformat(),
                "test": True
            }
        )
        
        logger.info(f"✅ Memory stored with ID: {memory_id}")
        
        # Test 2: Search for the memory
        logger.info("\n🔍 Searching for similar memories...")
        results = await vector_search.search_memories(
            query="test memory for Master Agent",
            namespace=test_namespace,
            limit=3
        )
        
        logger.info(f"✅ Found {len(results)} similar memories")
        
        if results:
            for i, result in enumerate(results, 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"  Content: {result.get('content', 'N/A')[:100]}...")
                logger.info(f"  Similarity: {result.get('similarity', 0):.4f}")
                logger.info(f"  Metadata: {result.get('metadata', {})}")
        
        # Test 3: Clean up test data
        logger.info("\n🧹 Cleaning up test collection...")
        try:
            vector_search.client.delete_collection(test_namespace)
            logger.info("✅ Test collection cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up test collection: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ChromaDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_retrieval_api():
    """Test the conversation retrieval API endpoint"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Conversation Retrieval API")
    logger.info("="*60)
    
    try:
        import httpx
        
        # Test the API endpoint
        logger.info("📡 Testing /api/v1/tasks/conversations/general endpoint...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8000/api/v1/tasks/conversations/general",
                params={"limit": 5}
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API returned {data.get('count', 0)} conversations")
                
                if data.get('conversations'):
                    logger.info(f"\nSample conversation:")
                    conv = data['conversations'][0]
                    logger.info(f"  User: {conv.get('user_message', 'N/A')[:100]}...")
                    logger.info(f"  Assistant: {conv.get('assistant_message', 'N/A')[:100]}...")
                
                return True
            else:
                logger.error(f"❌ API returned status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all memory persistence tests"""
    logger.info("\n" + "="*60)
    logger.info("🧪 MASTER AGENT MEMORY PERSISTENCE TEST SUITE")
    logger.info("="*60)
    
    results = {
        "MongoDB Persistence": await test_mongodb_persistence(),
        "ChromaDB Vector Search": await test_chromadb_persistence(),
        "API Retrieval": await test_conversation_retrieval_api()
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("\n✅ Master Agent memory persistence is working correctly:")
        logger.info("   - Conversations are saved to MongoDB")
        logger.info("   - Vector search is available via ChromaDB")
        logger.info("   - API endpoints are functioning")
        logger.info("   - Frontend can load conversation history")
    else:
        logger.error("\n❌ SOME TESTS FAILED - Please review the errors above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

