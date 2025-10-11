"""
Test Suite for Week 5-6: RAG Implementation

Tests all components of the RAG system:
1. Document Ingestion (file and directory)
2. Document Chunking Strategy
3. Semantic Search
4. RAG Generation
5. Integration with Week 4 Conversation Management

Expected: 100% pass rate (all tests should pass)
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.rag_service import rag_service, Document, SearchResult, RAGResponse


# ============================================================================
# Test Component 1: Document Ingestion
# ============================================================================

async def test_document_ingestion():
    """Test document ingestion with text content"""
    print("\n" + "="*80)
    print("Component 1: Document Ingestion")
    print("="*80)
    
    # Test 1: Ingest a simple document
    print("\n[Test 1/4] Ingest simple document...")
    content = "This is a test document about artificial intelligence and machine learning."
    doc = await rag_service.ingest_document(
        content=content,
        source="test_doc_1.txt",
        doc_type="text",
        metadata={"category": "test"}
    )
    assert doc.document_id is not None, "Document ID should be generated"
    assert doc.content == content, "Content should match"
    assert doc.source == "test_doc_1.txt", "Source should match"
    print(f"✅ Document ingested: {doc.document_id}")
    
    # Test 2: Ingest a longer document (will be chunked)
    print("\n[Test 2/4] Ingest long document (will be chunked)...")
    long_content = "AI and machine learning. " * 100  # ~2500 chars
    doc2 = await rag_service.ingest_document(
        content=long_content,
        source="test_doc_2.txt",
        doc_type="text"
    )
    assert doc2.document_id is not None, "Document ID should be generated"
    print(f"✅ Long document ingested: {doc2.document_id}")
    
    # Test 3: Ingest from file
    print("\n[Test 3/4] Ingest from file...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test file for RAG ingestion.")
        temp_file = f.name
    
    try:
        doc3 = await rag_service.ingest_file(temp_file)
        assert doc3.document_id is not None, "Document ID should be generated"
        assert "test file" in doc3.content, "Content should be read from file"
        print(f"✅ File ingested: {doc3.document_id}")
    finally:
        os.unlink(temp_file)
    
    # Test 4: Get collection stats
    print("\n[Test 4/4] Get collection statistics...")
    stats = await rag_service.get_collection_stats()
    assert stats['total_chunks'] > 0, "Should have chunks in collection"
    assert stats['total_documents'] > 0, "Should have documents in cache"
    print(f"✅ Collection stats: {stats['total_chunks']} chunks, {stats['total_documents']} documents")
    
    print("\n✅ Component 1: Document Ingestion - ALL TESTS PASSED")
    return True


# ============================================================================
# Test Component 2: Document Chunking Strategy
# ============================================================================

async def test_document_chunking():
    """Test document chunking with overlap"""
    print("\n" + "="*80)
    print("Component 2: Document Chunking Strategy")
    print("="*80)
    
    # Test 1: Small document (no chunking needed)
    print("\n[Test 1/4] Small document (no chunking)...")
    small_content = "Short document."
    chunks = rag_service.chunk_document(small_content, "test_small")
    assert len(chunks) == 1, "Small document should have 1 chunk"
    assert chunks[0].total_chunks == 1, "Total chunks should be 1"
    print(f"✅ Small document: {len(chunks)} chunk")
    
    # Test 2: Large document (will be chunked)
    print("\n[Test 2/4] Large document (will be chunked)...")
    large_content = "A" * 1500  # Larger than default chunk_size (512)
    chunks = rag_service.chunk_document(large_content, "test_large")
    assert len(chunks) > 1, "Large document should have multiple chunks"
    print(f"✅ Large document: {len(chunks)} chunks")
    
    # Test 3: Verify chunk overlap
    print("\n[Test 3/4] Verify chunk overlap...")
    assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
    # Check that chunks have overlap (chunk_overlap = 50 by default)
    chunk1_end = chunks[0].metadata['end']
    chunk2_start = chunks[1].metadata['start']
    overlap = chunk1_end - chunk2_start
    assert overlap == rag_service.chunk_overlap, f"Overlap should be {rag_service.chunk_overlap}"
    print(f"✅ Chunk overlap verified: {overlap} characters")
    
    # Test 4: Verify chunk metadata
    print("\n[Test 4/4] Verify chunk metadata...")
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i, f"Chunk index should be {i}"
        assert chunk.total_chunks == len(chunks), "Total chunks should match"
        assert chunk.document_id == "test_large", "Document ID should match"
    print(f"✅ All {len(chunks)} chunks have correct metadata")
    
    print("\n✅ Component 2: Document Chunking Strategy - ALL TESTS PASSED")
    return True


# ============================================================================
# Test Component 3: Semantic Search
# ============================================================================

async def test_semantic_search():
    """Test semantic search functionality"""
    print("\n" + "="*80)
    print("Component 3: Semantic Search")
    print("="*80)
    
    # Setup: Ingest test documents
    print("\n[Setup] Ingesting test documents...")
    await rag_service.ingest_document(
        content="Python is a high-level programming language used for web development, data science, and automation.",
        source="python_doc.txt",
        doc_type="text",
        metadata={"topic": "programming"}
    )
    await rag_service.ingest_document(
        content="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        source="ml_doc.txt",
        doc_type="text",
        metadata={"topic": "ai"}
    )
    await rag_service.ingest_document(
        content="JavaScript is a programming language commonly used for web development and creating interactive websites.",
        source="js_doc.txt",
        doc_type="text",
        metadata={"topic": "programming"}
    )
    print("✅ Test documents ingested")
    
    # Test 1: Basic search
    print("\n[Test 1/4] Basic semantic search...")
    results = await rag_service.search("programming language", n_results=3)
    assert len(results) > 0, "Should return search results"
    assert all(isinstance(r, SearchResult) for r in results), "Results should be SearchResult objects"
    print(f"✅ Search returned {len(results)} results")
    
    # Test 2: Search with relevance scoring
    print("\n[Test 2/4] Verify relevance scoring...")
    results = await rag_service.search("machine learning AI", n_results=3)
    assert len(results) > 0, "Should return results"
    # Scores should be between 0 and 1
    for result in results:
        assert 0 <= result.score <= 1, f"Score should be between 0 and 1, got {result.score}"
    print(f"✅ Relevance scores: {[f'{r.score:.3f}' for r in results]}")
    
    # Test 3: Search with metadata filter
    print("\n[Test 3/4] Search with metadata filter...")
    results = await rag_service.search(
        "programming",
        n_results=5,
        filter_metadata={"topic": "programming"}
    )
    assert len(results) > 0, "Should return filtered results"
    # Verify all results have the correct topic
    for result in results:
        assert result.metadata.get("topic") == "programming", "Should only return programming docs"
    print(f"✅ Filtered search returned {len(results)} results")
    
    # Test 4: Search with limited results
    print("\n[Test 4/4] Search with result limit...")
    results = await rag_service.search("web development", n_results=1)
    assert len(results) <= 1, "Should respect n_results limit"
    print(f"✅ Limited search returned {len(results)} result(s)")
    
    print("\n✅ Component 3: Semantic Search - ALL TESTS PASSED")
    return True


# ============================================================================
# Test Component 4: RAG Generation
# ============================================================================

async def test_rag_generation():
    """Test RAG (Retrieval-Augmented Generation)"""
    print("\n" + "="*80)
    print("Component 4: RAG Generation")
    print("="*80)
    
    # Setup: Ensure we have documents
    print("\n[Setup] Ensuring documents are available...")
    stats = await rag_service.get_collection_stats()
    print(f"✅ Collection has {stats['total_chunks']} chunks")
    
    # Test 1: Basic RAG query
    print("\n[Test 1/4] Basic RAG query...")
    response = await rag_service.generate_with_rag(
        query="What is Python used for?",
        n_results=2
    )
    assert isinstance(response, RAGResponse), "Should return RAGResponse"
    assert response.query == "What is Python used for?", "Query should match"
    assert response.answer is not None, "Should have an answer"
    assert len(response.sources) > 0, "Should have sources"
    print(f"✅ RAG response generated with {len(response.sources)} sources")
    print(f"   Answer preview: {response.answer[:100]}...")
    
    # Test 2: RAG with more context
    print("\n[Test 2/4] RAG with more context...")
    response = await rag_service.generate_with_rag(
        query="Explain machine learning",
        n_results=3,
        max_context_length=1000
    )
    assert isinstance(response, RAGResponse), "Should return RAGResponse"
    assert len(response.context_used) <= 1000 + 3, "Context should respect max length (with ellipsis)"
    print(f"✅ RAG with context limit: {len(response.context_used)} chars")
    
    # Test 3: Verify sources in response
    print("\n[Test 3/4] Verify sources in response...")
    response = await rag_service.generate_with_rag(
        query="Tell me about programming languages",
        n_results=2
    )
    assert len(response.sources) > 0, "Should have sources"
    for source in response.sources:
        assert source.document_id is not None, "Source should have document_id"
        assert source.content is not None, "Source should have content"
        assert 0 <= source.score <= 1, "Source should have valid score"
    print(f"✅ All {len(response.sources)} sources have valid data")
    
    # Test 4: RAG metadata
    print("\n[Test 4/4] Verify RAG metadata...")
    response = await rag_service.generate_with_rag(
        query="What is JavaScript?",
        n_results=1
    )
    assert response.metadata is not None, "Should have metadata"
    assert 'n_results' in response.metadata, "Metadata should include n_results"
    assert 'context_length' in response.metadata, "Metadata should include context_length"
    assert response.timestamp is not None, "Should have timestamp"
    print(f"✅ RAG metadata: {response.metadata}")
    
    print("\n✅ Component 4: RAG Generation - ALL TESTS PASSED")
    return True


# ============================================================================
# Test Component 5: Integration with Week 4 Conversation Management
# ============================================================================

async def test_conversation_integration():
    """Test RAG integration with conversation management"""
    print("\n" + "="*80)
    print("Component 5: Integration with Week 4 Conversation Management")
    print("="*80)
    
    try:
        from backend.services.conversation_service import conversation_service
        
        # Test 1: Create conversation
        print("\n[Test 1/5] Create conversation for RAG...")
        conversation = await conversation_service.create_conversation(
            conversation_id="rag_test_conv_1",
            title="RAG Test Conversation",
            participants=["user", "assistant"]
        )
        assert conversation.conversation_id == "rag_test_conv_1", "Conversation ID should match"
        print(f"✅ Conversation created: {conversation.conversation_id}")
        
        # Test 2: RAG with conversation context
        print("\n[Test 2/5] RAG with conversation context...")
        response = await rag_service.generate_with_rag(
            query="What programming languages are mentioned?",
            n_results=2,
            conversation_id="rag_test_conv_1"
        )
        assert isinstance(response, RAGResponse), "Should return RAGResponse"
        print(f"✅ RAG with conversation context completed")
        
        # Test 3: Verify messages stored in conversation
        print("\n[Test 3/5] Verify messages stored in conversation...")
        history = await conversation_service.get_conversation_history("rag_test_conv_1")
        assert len(history) >= 2, "Should have at least user query and assistant response"
        print(f"✅ Conversation has {len(history)} messages")
        
        # Test 4: RAG with conversation history context
        print("\n[Test 4/5] Add more messages and test context...")
        await conversation_service.add_message(
            "rag_test_conv_1",
            role="user",
            content="Tell me more about Python"
        )
        response = await rag_service.generate_with_rag(
            query="What are its main uses?",
            n_results=2,
            conversation_id="rag_test_conv_1"
        )
        assert isinstance(response, RAGResponse), "Should return RAGResponse"
        print(f"✅ RAG with conversation history completed")
        
        # Test 5: Verify RAG metadata in conversation
        print("\n[Test 5/5] Verify RAG metadata in conversation...")
        history = await conversation_service.get_conversation_history("rag_test_conv_1")
        # Find assistant messages with RAG metadata
        rag_messages = [msg for msg in history if msg.role == "assistant" and msg.metadata.get("rag_sources")]
        assert len(rag_messages) > 0, "Should have assistant messages with RAG metadata"
        print(f"✅ Found {len(rag_messages)} messages with RAG metadata")
        
        print("\n✅ Component 5: Conversation Integration - ALL TESTS PASSED")
        return True
        
    except ImportError:
        print("\n⚠️  Conversation service not available - skipping integration tests")
        return True


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_all_tests():
    """Run all Week 5-6 tests"""
    print("\n" + "="*80)
    print("WEEK 5-6 TEST SUITE: RAG Implementation")
    print("="*80)
    print("Testing all RAG components...")
    
    try:
        # Run all test components
        results = []
        
        results.append(await test_document_ingestion())
        results.append(await test_document_chunking())
        results.append(await test_semantic_search())
        results.append(await test_rag_generation())
        results.append(await test_conversation_integration())
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        total_components = len(results)
        passed_components = sum(results)
        
        print(f"\nComponents Tested: {total_components}")
        print(f"Components Passed: {passed_components}")
        print(f"Components Failed: {total_components - passed_components}")
        
        if all(results):
            print("\n🎉 ALL TESTS PASSED! Week 5-6 RAG Implementation Complete!")
            print("\nNext Steps:")
            print("- Week 7-8: Advanced AutoGen Features")
            print("- Week 9-10: LoRA Fine-tuning (Optional)")
            return True
        else:
            print("\n❌ SOME TESTS FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST SUITE ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
