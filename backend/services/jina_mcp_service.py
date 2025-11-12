"""
Jina AI MCP Service
Provides AI-powered search and embedding capabilities via Jina AI API

Features:
- AI-powered web search
- Text embeddings generation
- Content analysis and summarization
- Mock mode fallback when API key is not configured

API Documentation: https://jina.ai/docs
"""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


class JinaMCPService:
    """
    Jina AI MCP Service
    
    Provides access to Jina AI's search and embedding capabilities.
    Falls back to mock mode when API key is not configured.
    """
    
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY", "")
        self.base_url = "https://api.jina.ai/v1"
        self.mock_mode = not self.api_key or self.api_key == "your-jina-api-key-here"
        
        # Configuration
        self.config = {
            "timeout": 30,
            "max_retries": 3,
            "default_model": "jina-embeddings-v2-base-en",
            "search_limit": 10,
        }
        
        # Metrics
        self.metrics = {
            "searches": 0,
            "embeddings": 0,
            "errors": 0,
            "mock_responses": 0,
        }
        
        if self.mock_mode:
            logger.warning("⚠️ Jina MCP Service running in MOCK MODE - No API key configured")
        else:
            logger.info("✅ Jina MCP Service initialized with API key")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        include_content: bool = True
    ) -> Dict[str, Any]:
        """
        Perform AI-powered web search using Jina AI
        
        Args:
            query: Search query
            limit: Maximum number of results (default: 10)
            include_content: Whether to include full content (default: True)
        
        Returns:
            Dict containing search results
        """
        try:
            if self.mock_mode:
                return self._mock_search(query, limit)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "limit": limit,
                        "include_content": include_content,
                    },
                    timeout=self.config["timeout"],
                )
                
                response.raise_for_status()
                self.metrics["searches"] += 1
                
                return {
                    "success": True,
                    "query": query,
                    "results": response.json(),
                    "timestamp": datetime.utcnow().isoformat(),
                }
        
        except Exception as e:
            logger.error(f"Jina search error: {e}")
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "query": query,
            }
    
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for text using Jina AI
        
        Args:
            texts: List of texts to embed
            model: Model to use (default: jina-embeddings-v2-base-en)
        
        Returns:
            Dict containing embeddings
        """
        try:
            if self.mock_mode:
                return self._mock_embed(texts, model)
            
            model = model or self.config["default_model"]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": texts,
                        "model": model,
                    },
                    timeout=self.config["timeout"],
                )
                
                response.raise_for_status()
                self.metrics["embeddings"] += 1
                
                return {
                    "success": True,
                    "model": model,
                    "embeddings": response.json(),
                    "count": len(texts),
                    "timestamp": datetime.utcnow().isoformat(),
                }
        
        except Exception as e:
            logger.error(f"Jina embedding error: {e}")
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "count": len(texts),
            }
    
    async def analyze_content(
        self,
        url: str,
        extract_images: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze and extract content from a URL using Jina AI
        
        Args:
            url: URL to analyze
            extract_images: Whether to extract images (default: False)
        
        Returns:
            Dict containing analyzed content
        """
        try:
            if self.mock_mode:
                return self._mock_analyze(url)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/reader",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "url": url,
                        "extract_images": extract_images,
                    },
                    timeout=self.config["timeout"],
                )
                
                response.raise_for_status()
                
                return {
                    "success": True,
                    "url": url,
                    "content": response.json(),
                    "timestamp": datetime.utcnow().isoformat(),
                }
        
        except Exception as e:
            logger.error(f"Jina content analysis error: {e}")
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            "service": "jina_mcp",
            "mock_mode": self.mock_mode,
            "metrics": self.metrics,
            "config": self.config,
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "service": "jina_mcp",
            "status": "healthy",
            "mock_mode": self.mock_mode,
            "api_key_configured": bool(self.api_key and self.api_key != "your-jina-api-key-here"),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ==================== Mock Mode Methods ====================
    
    def _mock_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Mock search results for testing"""
        self.metrics["mock_responses"] += 1
        
        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "title": f"Mock Result {i+1} for: {query}",
                    "url": f"https://example.com/result-{i+1}",
                    "snippet": f"This is a mock search result {i+1} for the query '{query}'. "
                              f"Configure JINA_API_KEY to get real results.",
                    "score": 0.9 - (i * 0.1),
                }
                for i in range(min(limit, 5))
            ],
            "mock_mode": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _mock_embed(self, texts: List[str], model: Optional[str]) -> Dict[str, Any]:
        """Mock embeddings for testing"""
        self.metrics["mock_responses"] += 1
        
        # Generate mock embeddings (768-dimensional vectors)
        import random
        random.seed(hash(texts[0]) if texts else 0)
        
        return {
            "success": True,
            "model": model or self.config["default_model"],
            "embeddings": [
                {
                    "embedding": [random.random() for _ in range(768)],
                    "index": i,
                }
                for i in range(len(texts))
            ],
            "count": len(texts),
            "mock_mode": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _mock_analyze(self, url: str) -> Dict[str, Any]:
        """Mock content analysis for testing"""
        self.metrics["mock_responses"] += 1
        
        return {
            "success": True,
            "url": url,
            "content": {
                "title": f"Mock Content Analysis for {url}",
                "text": f"This is mock analyzed content from {url}. "
                       f"Configure JINA_API_KEY to get real content analysis.",
                "author": "Mock Author",
                "published_date": datetime.utcnow().isoformat(),
                "word_count": 150,
            },
            "mock_mode": True,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Singleton instance
_jina_service = None


def get_jina_service() -> JinaMCPService:
    """Get or create Jina MCP service instance"""
    global _jina_service
    if _jina_service is None:
        _jina_service = JinaMCPService()
    return _jina_service

