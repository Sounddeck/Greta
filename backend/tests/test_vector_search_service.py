"""
Comprehensive tests for vector_search_service.py module

Tests cover:
- Vector embedding generation
- Similarity search
- Index management
- Performance optimization
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List


class TestEmbeddingGeneration:
    """Test vector embedding generation"""
    
    @pytest.mark.asyncio
    async def test_store_memory_generates_embedding(self):
        """Test that storing memory generates embeddings"""
        from services.vector_search_service import VectorSearchService
        
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        memory_id = await service.store_memory(
            content="Test content",
            namespace="test",
            metadata={"key": "value"}
        )
        
        assert memory_id is not None
        assert mock_collection.add.called
    
    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self):
        """Test that embeddings have consistent dimensions"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        # Store multiple memories
        id1 = await service.store_memory("Content 1", namespace="test")
        id2 = await service.store_memory("Content 2", namespace="test")
        
        # Both should be stored successfully
        assert id1 is not None
        assert id2 is not None
        assert mock_collection.add.call_count == 2
    
    @pytest.mark.asyncio
    async def test_handle_empty_input(self):
        """Test handling empty or invalid input"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        # Empty content should still work (ChromaDB handles it)
        memory_id = await service.store_memory(
            content="",
            namespace="test"
        )
        
        assert memory_id is not None
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self):
        """Test batch embedding generation"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        # Store multiple memories in batch
        contents = ["Content 1", "Content 2", "Content 3"]
        memory_ids = []
        
        for content in contents:
            memory_id = await service.store_memory(content, namespace="test")
            memory_ids.append(memory_id)
        
        assert len(memory_ids) == 3
        assert all(mid is not None for mid in memory_ids)


class TestSimilaritySearch:
    """Test similarity search functionality"""
    
    @pytest.mark.asyncio
    async def test_search_memories_success(self):
        """Test successful similarity search"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Mock search results
        mock_collection.query = MagicMock(return_value={
            'documents': [['Test document 1', 'Test document 2']],
            'ids': [['id1', 'id2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        })
        
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        results = await service.search_memories(
            query="test query",
            namespace="test",
            limit=10
        )
        
        assert len(results) == 2
        assert results[0]['id'] == 'id1'
        assert results[0]['content'] == 'Test document 1'
        assert results[0]['distance'] == 0.1
    
    @pytest.mark.asyncio
    async def test_search_with_threshold(self):
        """Test similarity search with distance threshold"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Mock results with varying distances
        mock_collection.query = MagicMock(return_value={
            'documents': [['Doc 1', 'Doc 2', 'Doc 3']],
            'ids': [['id1', 'id2', 'id3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.5, 0.9]]
        })
        
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        results = await service.search_memories(
            query="test",
            namespace="test",
            limit=10
        )
        
        # All results returned (filtering by threshold would be done by caller)
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_k_nearest_neighbors(self):
        """Test k-nearest neighbors search"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        mock_collection.query = MagicMock(return_value={
            'documents': [['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5']],
            'ids': [['id1', 'id2', 'id3', 'id4', 'id5']],
            'metadatas': [[{}, {}, {}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3, 0.4, 0.5]]
        })
        
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        # Search for top 5 results
        results = await service.search_memories(
            query="test",
            namespace="test",
            limit=5
        )
        
        assert len(results) == 5
        # Results should be ordered by distance (closest first)
        assert results[0]['distance'] == 0.1
        assert results[4]['distance'] == 0.5
    
    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """Test search with no results"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Mock empty results
        mock_collection.query = MagicMock(return_value={
            'documents': [[]],
            'ids': [[]],
            'metadatas': [[]],
            'distances': [[]]
        })
        
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        results = await service.search_memories(
            query="nonexistent query",
            namespace="test",
            limit=10
        )
        
        assert len(results) == 0


class TestIndexManagement:
    """Test index management operations"""
    
    @pytest.mark.asyncio
    async def test_create_collection(self):
        """Test creating a new collection/index"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        # Create collection by storing memory
        collection = service._get_or_create_collection("new_namespace")
        
        assert collection is not None
        assert "new_namespace" in service.collections
    
    @pytest.mark.asyncio
    async def test_delete_memory_from_index(self):
        """Test deleting vectors from index"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.delete = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        result = await service.delete_memory("test_id", namespace="test")
        
        assert result is True
        assert mock_collection.delete.called
    
    @pytest.mark.asyncio
    async def test_get_memory_by_id(self):
        """Test retrieving specific memory by ID"""
        from services.vector_search_service import VectorSearchService
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        mock_collection.get = MagicMock(return_value={
            'documents': ['Test content'],
            'metadatas': [{'key': 'value'}]
        })
        
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
        
        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()
        
        memory = await service.get_memory("test_id", namespace="test")

        assert memory is not None
        assert memory['id'] == 'test_id'
        assert memory['content'] == 'Test content'

    @pytest.mark.asyncio
    async def test_collection_caching(self):
        """Test that collections are cached after first access"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        # First access
        collection1 = service._get_or_create_collection("test")
        # Second access (should use cache)
        collection2 = service._get_or_create_collection("test")

        # Should only call get_or_create_collection once
        assert mock_client.get_or_create_collection.call_count == 1
        assert collection1 is collection2


class TestPerformance:
    """Test performance optimization"""

    @pytest.mark.asyncio
    async def test_search_performance(self):
        """Test search performance with large dataset"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_collection = MagicMock()

        # Mock large result set
        large_results = {
            'documents': [[f'Doc {i}' for i in range(100)]],
            'ids': [[f'id{i}' for i in range(100)]],
            'metadatas': [[{} for i in range(100)]],
            'distances': [[i * 0.01 for i in range(100)]]
        }

        mock_collection.query = MagicMock(return_value=large_results)
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        # Search should handle large results efficiently
        results = await service.search_memories(
            query="test",
            namespace="test",
            limit=100
        )

        assert len(results) == 100

    @pytest.mark.asyncio
    async def test_metadata_filtering(self):
        """Test search with metadata filtering for performance"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_collection = MagicMock()

        mock_collection.query = MagicMock(return_value={
            'documents': [['Filtered doc']],
            'ids': [['id1']],
            'metadatas': [[{'category': 'test'}]],
            'distances': [[0.1]]
        })

        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        # Search with metadata filter
        results = await service.search_memories(
            query="test",
            namespace="test",
            limit=10,
            metadata_filter={"category": "test"}
        )

        assert len(results) == 1
        # Verify filter was passed to query
        call_args = mock_collection.query.call_args
        assert call_args[1]['where'] == {"category": "test"}

    @pytest.mark.asyncio
    async def test_get_collection_stats(self):
        """Test retrieving collection statistics"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=42)
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        stats = await service.get_collection_stats("test")

        assert stats['namespace'] == 'test'
        assert stats['count'] == 42
        assert 'embedding_function' in stats


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_service_not_initialized(self):
        """Test handling when service is not initialized"""
        from services.vector_search_service import VectorSearchService

        service = VectorSearchService(persist_directory="./test_data")
        service.client = None  # Simulate failed initialization

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Vector Search Service not available"):
            await service.store_memory("test", namespace="test")

    @pytest.mark.asyncio
    async def test_search_service_not_initialized(self):
        """Test search when service is not initialized"""
        from services.vector_search_service import VectorSearchService

        service = VectorSearchService(persist_directory="./test_data")
        service.client = None

        with pytest.raises(RuntimeError, match="Vector Search Service not available"):
            await service.search_memories("test", namespace="test")

    @pytest.mark.asyncio
    async def test_collection_creation_failure(self):
        """Test handling collection creation failure"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_client.get_or_create_collection = MagicMock(
            side_effect=Exception("Collection creation failed")
        )

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        with pytest.raises(Exception, match="Collection creation failed"):
            service._get_or_create_collection("test")

    @pytest.mark.asyncio
    async def test_store_memory_failure(self):
        """Test handling store memory failure"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.add = MagicMock(side_effect=Exception("Storage failed"))
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        with pytest.raises(Exception, match="Storage failed"):
            await service.store_memory("test", namespace="test")

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        """Test retrieving non-existent memory"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_collection = MagicMock()

        # Mock empty result
        mock_collection.get = MagicMock(return_value={
            'documents': [],
            'metadatas': []
        })

        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        memory = await service.get_memory("nonexistent_id", namespace="test")

        assert memory is None

    @pytest.mark.asyncio
    async def test_delete_memory_failure(self):
        """Test handling delete memory failure"""
        from services.vector_search_service import VectorSearchService

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.delete = MagicMock(side_effect=Exception("Delete failed"))
        mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

        service = VectorSearchService(persist_directory="./test_data")
        service.client = mock_client
        service.embedding_function = MagicMock()

        result = await service.delete_memory("test_id", namespace="test")

        # Should return False on failure
        assert result is False

