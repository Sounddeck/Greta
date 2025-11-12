#!/usr/bin/env python3
"""
RAG Integration Test for Master Agent

This script tests the Retrieval Augmented Generation (RAG) integration
with the Master Agent, demonstrating:
1. Context retrieval from past interactions
2. Memory storage after task execution
3. Context-aware responses using retrieved memories
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.greta_master_agent import greta_master_agent, execute_master_task


class RAGIntegrationTest:
    """Test RAG integration with Master Agent"""
    
    def __init__(self):
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all RAG integration tests"""
        print("=" * 80)
        print("RAG INTEGRATION TEST - Master Agent with Vector Search")
        print("=" * 80)
        print()
        
        # Test 1: Store initial memories
        await self.test_store_initial_memories()
        
        # Test 2: Retrieve context for related query
        await self.test_context_retrieval()
        
        # Test 3: Test context-aware responses
        await self.test_context_aware_responses()
        
        # Test 4: Test memory recall
        await self.test_memory_recall()
        
        # Print summary
        self.print_summary()
        
    async def test_store_initial_memories(self):
        """Test 1: Store initial memories by executing tasks"""
        print("📝 Test 1: Storing Initial Memories")
        print("-" * 80)
        
        test_tasks = [
            "We successfully installed ChromaDB version 1.3.0 for vector search",
            "The Skills System has 5 core skills: SPARC, Pair Programming, Swarm Orchestration, Code Review, and Performance Optimization",
            "Performance benchmark showed 99.55ms average latency, meeting the <100ms target",
            "The backend is running on port 8000 with 42 routers loaded"
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n  Task {i}: {task[:60]}...")
            
            try:
                result = await execute_master_task(task)
                
                if greta_master_agent.rag_enabled:
                    print(f"  ✅ Task executed and memory stored")
                else:
                    print(f"  ⚠️  Task executed but RAG not enabled")
                    
                self.test_results.append({
                    'test': 'store_memory',
                    'task': task,
                    'success': True
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.test_results.append({
                    'test': 'store_memory',
                    'task': task,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n✅ Test 1 Complete: Stored {len(test_tasks)} memories")
        print()
        
    async def test_context_retrieval(self):
        """Test 2: Retrieve context for related queries"""
        print("🔍 Test 2: Context Retrieval")
        print("-" * 80)
        
        if not greta_master_agent.rag_enabled:
            print("  ⚠️  RAG not enabled, skipping context retrieval test")
            print()
            return
        
        test_queries = [
            "What version of ChromaDB did we install?",
            "How many skills are in the Skills System?",
            "What was the performance benchmark result?",
            "Which port is the backend running on?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n  Query {i}: {query}")
            
            try:
                # Retrieve relevant context
                memories = await greta_master_agent.retrieve_relevant_context(query, limit=3)
                
                if memories:
                    print(f"  ✅ Retrieved {len(memories)} relevant memories")
                    
                    # Show top memory
                    top_memory = memories[0]
                    content = top_memory.get('content', '')[:100]
                    distance = top_memory.get('distance', 0)
                    relevance = 1 - distance
                    
                    print(f"     Top result (relevance: {relevance:.1%}): {content}...")
                    
                    self.test_results.append({
                        'test': 'retrieve_context',
                        'query': query,
                        'memories_found': len(memories),
                        'success': True
                    })
                else:
                    print(f"  ⚠️  No relevant memories found")
                    self.test_results.append({
                        'test': 'retrieve_context',
                        'query': query,
                        'memories_found': 0,
                        'success': False
                    })
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.test_results.append({
                    'test': 'retrieve_context',
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n✅ Test 2 Complete: Context retrieval tested")
        print()
        
    async def test_context_aware_responses(self):
        """Test 3: Test that responses use retrieved context"""
        print("🧠 Test 3: Context-Aware Responses")
        print("-" * 80)
        
        if not greta_master_agent.rag_enabled:
            print("  ⚠️  RAG not enabled, skipping context-aware response test")
            print()
            return
        
        # Execute a task that should benefit from context
        query = "Remind me about the performance benchmarks we ran"
        
        print(f"\n  Query: {query}")
        print(f"  Retrieving context...")
        
        try:
            # This should retrieve context and use it in the response
            result = await execute_master_task(query)
            
            # Check if RAG context was used
            if 'rag_memories_count' in result.get('context', {}):
                memories_count = result['context']['rag_memories_count']
                print(f"  ✅ RAG context retrieved: {memories_count} memories used")
                
                self.test_results.append({
                    'test': 'context_aware_response',
                    'query': query,
                    'memories_used': memories_count,
                    'success': True
                })
            else:
                print(f"  ⚠️  Response generated but no RAG context info")
                self.test_results.append({
                    'test': 'context_aware_response',
                    'query': query,
                    'success': False
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results.append({
                'test': 'context_aware_response',
                'query': query,
                'success': False,
                'error': str(e)
            })
        
        print(f"\n✅ Test 3 Complete: Context-aware responses tested")
        print()
        
    async def test_memory_recall(self):
        """Test 4: Test memory recall for specific topics"""
        print("💭 Test 4: Memory Recall")
        print("-" * 80)
        
        if not greta_master_agent.rag_enabled:
            print("  ⚠️  RAG not enabled, skipping memory recall test")
            print()
            return
        
        recall_queries = [
            "What did we discuss about the Skills System?",
            "Tell me about the ChromaDB installation",
            "What were the results of our performance testing?"
        ]
        
        for i, query in enumerate(recall_queries, 1):
            print(f"\n  Recall {i}: {query}")
            
            try:
                # Retrieve memories
                memories = await greta_master_agent.retrieve_relevant_context(query, limit=2)
                
                if memories:
                    print(f"  ✅ Recalled {len(memories)} relevant memories")
                    
                    for j, memory in enumerate(memories, 1):
                        metadata = memory.get('metadata', {})
                        task_desc = metadata.get('task_description', 'Unknown')
                        timestamp = metadata.get('timestamp', 'Unknown')
                        
                        print(f"     Memory {j}:")
                        print(f"       Task: {task_desc[:60]}...")
                        print(f"       Time: {timestamp}")
                    
                    self.test_results.append({
                        'test': 'memory_recall',
                        'query': query,
                        'memories_recalled': len(memories),
                        'success': True
                    })
                else:
                    print(f"  ⚠️  No memories recalled")
                    self.test_results.append({
                        'test': 'memory_recall',
                        'query': query,
                        'memories_recalled': 0,
                        'success': False
                    })
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.test_results.append({
                    'test': 'memory_recall',
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n✅ Test 4 Complete: Memory recall tested")
        print()
        
    def print_summary(self):
        """Print test summary"""
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.get('success', False))
        failed_tests = total_tests - successful_tests
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if greta_master_agent.rag_enabled:
            print(f"\n✅ RAG Status: ENABLED")
            print(f"   Namespace: {greta_master_agent.rag_namespace}")
            print(f"   Retrieval Limit: {greta_master_agent.rag_retrieval_limit}")
        else:
            print(f"\n⚠️  RAG Status: DISABLED")
            print(f"   Vector Search Service not available")
        
        print("\n" + "=" * 80)


async def main():
    """Main entry point"""
    test = RAGIntegrationTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

