"""
Comprehensive tests for knowledge_base_service.py module

Tests cover:
- Knowledge storage and retrieval
- Vector search functionality
- Knowledge graph operations
- Indexing and search
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List
import numpy as np


class TestKnowledgeStorage:
    """Test knowledge storage and retrieval"""
    
    @pytest.mark.asyncio
    async def test_add_document_success(self):
        """Test successfully adding a document"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        # Mock embedding generation
        service._generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        service._auto_categorize = AsyncMock(return_value="technical")
        
        config = {
            "title": "Test Document",
            "content": "This is test content",
            "category": "technical"
        }
        
        result = await service.add_document(config)
        
        assert result["success"] is True
        assert "document_id" in result
        assert result["document"]["title"] == "Test Document"
    
    @pytest.mark.asyncio
    async def test_get_document_by_id(self):
        """Test retrieving document by ID"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        # Add a test document
        test_doc = {
            "document_id": "doc_123",
            "title": "Test Doc",
            "content": "Content"
        }
        service.documents["doc_123"] = test_doc
        
        result = await service.get_document("doc_123")
        
        assert result["success"] is True
        assert result["document"]["document_id"] == "doc_123"
    
    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """Test retrieving non-existent document"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        result = await service.get_document("nonexistent_id")
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_update_document_success(self):
        """Test updating a document"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        # Add a test document
        test_doc = {
            "document_id": "doc_123",
            "title": "Original Title",
            "content": "Original Content"
        }
        service.documents["doc_123"] = test_doc
        
        # Mock embedding generation
        service._generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        updates = {
            "title": "Updated Title",
            "content": "Updated Content"
        }
        
        result = await service.update_document("doc_123", updates)
        
        assert result["success"] is True
        assert result["document"]["title"] == "Updated Title"


class TestVectorSearch:
    """Test vector search functionality"""
    
    @pytest.mark.asyncio
    async def test_semantic_search_success(self):
        """Test semantic search with embeddings"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        # Add test documents with embeddings
        service.documents["doc_1"] = {
            "document_id": "doc_1",
            "title": "AI Document",
            "content": "About AI",
            "category": "technical",
            "access_count": 0
        }
        service.document_embeddings["doc_1"] = [0.9, 0.1, 0.1]
        
        service.documents["doc_2"] = {
            "document_id": "doc_2",
            "title": "Business Document",
            "content": "About business",
            "category": "business",
            "access_count": 0
        }
        service.document_embeddings["doc_2"] = [0.1, 0.9, 0.1]
        
        # Mock query embedding (similar to doc_1)
        service._generate_embedding = AsyncMock(return_value=[0.8, 0.2, 0.1])
        
        result = await service.semantic_search("AI query", limit=10)
        
        assert result["success"] is True
        assert len(result["results"]) > 0
        assert result["results"][0]["document_id"] == "doc_1"
    
    @pytest.mark.asyncio
    async def test_semantic_search_with_category_filter(self):
        """Test semantic search with category filtering"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        # Add test documents
        service.documents["doc_1"] = {
            "document_id": "doc_1",
            "title": "Tech Doc",
            "category": "technical",
            "access_count": 0
        }
        service.document_embeddings["doc_1"] = [0.9, 0.1, 0.1]
        
        service.documents["doc_2"] = {
            "document_id": "doc_2",
            "title": "Business Doc",
            "category": "business",
            "access_count": 0
        }
        service.document_embeddings["doc_2"] = [0.8, 0.2, 0.1]
        
        service._generate_embedding = AsyncMock(return_value=[0.9, 0.1, 0.1])
        
        result = await service.semantic_search("query", limit=10, category="technical")
        
        assert result["success"] is True
        assert all(r["document"]["category"] == "technical" for r in result["results"])
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self):
        """Test embedding generation"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        # Test the actual embedding generation (uses simple hash-based approach)
        embedding = await service._generate_embedding("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


class TestSearchFunctionality:
    """Test search and indexing"""
    
    @pytest.mark.asyncio
    async def test_search_documents_by_keyword(self):
        """Test keyword search"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock _log_activity
        service._log_activity = Mock()

        # Add test documents
        service.documents["doc_1"] = {
            "document_id": "doc_1",
            "title": "Python Programming",
            "content": "Learn Python",
            "category": "technical",
            "tags": ["python", "programming"],
            "access_count": 0
        }

        service.documents["doc_2"] = {
            "document_id": "doc_2",
            "title": "Java Programming",
            "content": "Learn Java",
            "category": "technical",
            "tags": ["java", "programming"],
            "access_count": 0
        }

        query = {"search_term": "python"}
        result = await service.search_documents(query)

        assert result["success"] is True
        assert len(result["documents"]) == 1
        assert result["documents"][0]["title"] == "Python Programming"
    
    @pytest.mark.asyncio
    async def test_search_documents_with_filters(self):
        """Test search with category and tag filters"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock _log_activity
        service._log_activity = Mock()

        # Add test documents
        service.documents["doc_1"] = {
            "document_id": "doc_1",
            "title": "Doc 1",
            "content": "Content",
            "category": "technical",
            "tags": ["python"],
            "access_count": 0
        }

        service.documents["doc_2"] = {
            "document_id": "doc_2",
            "title": "Doc 2",
            "content": "Content",
            "category": "business",
            "tags": ["finance"],
            "access_count": 0
        }

        query = {
            "category": "technical",
            "tags": ["python"]
        }
        result = await service.search_documents(query)

        assert result["success"] is True
        assert len(result["documents"]) == 1
        assert result["documents"][0]["category"] == "technical"
    
    @pytest.mark.asyncio
    async def test_get_all_documents(self):
        """Test retrieving all documents"""
        from services.knowledge_base_service import KnowledgeBaseService
        
        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()
        
        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)
        
        # Add test documents
        service.documents["doc_1"] = {"document_id": "doc_1", "category": "technical"}
        service.documents["doc_2"] = {"document_id": "doc_2", "category": "business"}
        
        result = await service.get_all_documents()
        
        assert result["success"] is True
        assert result["count"] == 2


class TestKnowledgeGraph:
    """Test knowledge graph operations"""

    @pytest.mark.asyncio
    async def test_add_to_knowledge_graph(self):
        """Test adding document to knowledge graph"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Add a document to create relationships
        service.documents["doc_1"] = {
            "document_id": "doc_1",
            "title": "Doc 1",
            "category": "technical"
        }

        # Add another document to knowledge graph
        await service._add_to_knowledge_graph("doc_2", "Doc 2", "technical")

        # Verify node was added
        assert "doc_2" in service.knowledge_graph["nodes"]
        assert service.knowledge_graph["nodes"]["doc_2"]["title"] == "Doc 2"

    @pytest.mark.asyncio
    async def test_get_knowledge_graph(self):
        """Test retrieving knowledge graph"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Add test nodes and edges
        service.knowledge_graph["nodes"]["doc_1"] = {"document_id": "doc_1", "title": "Doc 1"}
        service.knowledge_graph["nodes"]["doc_2"] = {"document_id": "doc_2", "title": "Doc 2"}
        service.knowledge_graph["edges"].append({
            "from": "doc_1",
            "to": "doc_2",
            "relationship": "related_category"
        })

        result = await service.get_knowledge_graph()

        assert result["success"] is True
        assert result["nodes_count"] == 2
        assert result["edges_count"] == 1


class TestCategories:
    """Test category management"""

    @pytest.mark.asyncio
    async def test_get_categories(self):
        """Test retrieving all categories"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Add test documents
        service.documents["doc_1"] = {"document_id": "doc_1", "category": "technical"}
        service.documents["doc_2"] = {"document_id": "doc_2", "category": "technical"}
        service.documents["doc_3"] = {"document_id": "doc_3", "category": "business"}

        result = await service.get_categories()

        assert result["success"] is True
        assert len(result["categories"]) > 0

    @pytest.mark.asyncio
    async def test_auto_categorize(self):
        """Test automatic document categorization"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.generate_completion = AsyncMock(return_value={
            "success": True,
            "response": "technical"
        })
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        category = await service._auto_categorize("Python Tutorial", "Learn Python programming")

        assert category in service.categories


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_add_document_database_failure(self):
        """Test handling database failure when adding document"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(side_effect=Exception("DB Error"))
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock embedding generation
        service._generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        service._auto_categorize = AsyncMock(return_value="technical")

        config = {
            "title": "Test",
            "content": "Content"
        }

        # Should still succeed even if DB fails (stores in memory)
        result = await service.add_document(config)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test search error handling"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Force an error by making documents None
        service.documents = None

        result = await service.search_documents({"search_term": "test"})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        mock_db.find = AsyncMock(return_value=AsyncMock(__aiter__=AsyncMock(return_value=iter([]))))
        mock_db.find_one = AsyncMock(return_value=None)
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        result = await service.initialize()

        assert result is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure handling"""
        from services.knowledge_base_service import KnowledgeBaseService

        mock_db = AsyncMock()
        # Make _load_documents fail
        mock_db.find = AsyncMock(side_effect=Exception("DB Connection Error"))
        mock_llm = AsyncMock()
        mock_learning = AsyncMock()

        service = KnowledgeBaseService(db=mock_db, llm_service=mock_llm, learning_service=mock_learning)

        # Mock _load_knowledge_graph to also fail
        service._load_knowledge_graph = AsyncMock(side_effect=Exception("Graph load error"))

        result = await service.initialize()

        # The service catches exceptions and returns False
        assert result is False

