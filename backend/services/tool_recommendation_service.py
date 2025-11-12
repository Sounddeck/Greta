"""
Tool Recommendation & Auto-Selection Service

Provides intelligent tool recommendations based on:
- Semantic similarity to task descriptions
- Historical performance data
- Tool chaining patterns
- User preferences
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field

from services.tool_registry import tool_registry, ToolDefinition
from services.vector_search_service import VectorSearchService
from database import Database

logger = logging.getLogger(__name__)


# ==================== MODELS ====================

class ToolPerformanceMetrics(BaseModel):
    """Performance metrics for a tool"""
    tool_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    user_ratings: List[float] = Field(default_factory=list)
    average_rating: float = 0.0


class ToolChainPattern(BaseModel):
    """Pattern of tools used together"""
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_sequence: List[str]  # List of tool_ids in order
    task_type: str  # e.g., "data_processing", "research", "analysis"
    usage_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None


class ToolRecommendation(BaseModel):
    """A single tool recommendation"""
    tool: ToolDefinition
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    performance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    combined_score: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_parameters: Optional[Dict[str, Any]] = None


class ToolChainRecommendation(BaseModel):
    """A recommended sequence of tools"""
    chain_id: str = Field(default_factory=lambda: str(uuid4()))
    tools: List[ToolDefinition]
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    estimated_execution_time: float = 0.0
    success_probability: float = Field(default=0.0, ge=0.0, le=1.0)


# ==================== SERVICE ====================

class ToolRecommendationService:
    """
    Intelligent tool recommendation and auto-selection service
    """
    
    def __init__(self, db: Database, vector_search: Optional[VectorSearchService] = None):
        self.db = db
        self.vector_search = vector_search
        self.performance_collection = db.get_collection("tool_performance")
        self.chain_patterns_collection = db.get_collection("tool_chain_patterns")
        
        # In-memory cache for performance metrics
        self.performance_cache: Dict[str, ToolPerformanceMetrics] = {}
        self.chain_patterns_cache: List[ToolChainPattern] = []
        
        logger.info("✅ Tool Recommendation Service initialized")
    
    
    async def initialize(self):
        """Initialize service and load cached data"""
        try:
            # Load performance metrics
            metrics_docs = await self.performance_collection.find({}).to_list(length=1000)
            for doc in metrics_docs:
                metrics = ToolPerformanceMetrics(**doc)
                self.performance_cache[metrics.tool_id] = metrics
            
            # Load chain patterns
            chain_docs = await self.chain_patterns_collection.find({}).to_list(length=1000)
            self.chain_patterns_cache = [ToolChainPattern(**doc) for doc in chain_docs]
            
            logger.info(f"✅ Loaded {len(self.performance_cache)} tool metrics and {len(self.chain_patterns_cache)} chain patterns")
            
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
    
    
    async def recommend_tools(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        min_relevance: float = 0.3
    ) -> List[ToolRecommendation]:
        """
        Recommend tools for a given task
        
        Args:
            task_description: Natural language description of the task
            context: Additional context (user preferences, constraints, etc.)
            limit: Maximum number of recommendations
            min_relevance: Minimum relevance score threshold
        
        Returns:
            List of tool recommendations sorted by combined score
        """
        try:
            recommendations = []
            
            # Step 1: Get all available tools
            all_tools = tool_registry.get_all_tools()
            
            # Step 2: Calculate semantic similarity if Vector Search available
            if self.vector_search:
                semantic_scores = await self._calculate_semantic_similarity(task_description, all_tools)
            else:
                # Fallback to keyword matching
                semantic_scores = self._calculate_keyword_similarity(task_description, all_tools)
            
            # Step 3: Get performance scores
            performance_scores = self._get_performance_scores(all_tools)
            
            # Step 4: Combine scores and create recommendations
            for tool in all_tools:
                relevance_score = semantic_scores.get(tool.tool_id, 0.0)
                performance_score = performance_scores.get(tool.tool_id, 0.5)  # Default to 0.5 if no data
                
                # Skip if below minimum relevance
                if relevance_score < min_relevance:
                    continue
                
                # Combined score: 70% relevance, 30% performance
                combined_score = (relevance_score * 0.7) + (performance_score * 0.3)
                
                # Generate reason
                reason = self._generate_recommendation_reason(
                    tool, relevance_score, performance_score, task_description
                )
                
                recommendations.append(ToolRecommendation(
                    tool=tool,
                    relevance_score=relevance_score,
                    performance_score=performance_score,
                    combined_score=combined_score,
                    reason=reason
                ))
            
            # Sort by combined score and limit
            recommendations.sort(key=lambda x: x.combined_score, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Tool recommendation failed: {e}")
            return []
    
    
    async def recommend_tool_chains(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 3
    ) -> List[ToolChainRecommendation]:
        """
        Recommend sequences of tools (tool chains) for complex tasks
        
        Args:
            task_description: Natural language description of the task
            context: Additional context
            limit: Maximum number of chain recommendations
        
        Returns:
            List of tool chain recommendations
        """
        try:
            recommendations = []
            
            # Step 1: Identify task type from description
            task_type = self._identify_task_type(task_description)
            
            # Step 2: Find matching chain patterns
            matching_patterns = [
                pattern for pattern in self.chain_patterns_cache
                if pattern.task_type == task_type and pattern.usage_count > 0
            ]
            
            # Step 3: Create recommendations from patterns
            for pattern in matching_patterns[:limit]:
                tools = [tool_registry.get_tool(tool_id) for tool_id in pattern.tool_sequence]
                tools = [t for t in tools if t is not None]  # Filter out None values
                
                if not tools:
                    continue
                
                recommendations.append(ToolChainRecommendation(
                    tools=tools,
                    relevance_score=min(1.0, pattern.success_rate + 0.2),
                    reason=f"This tool chain has been successfully used {pattern.usage_count} times for {task_type} tasks",
                    estimated_execution_time=pattern.average_execution_time,
                    success_probability=pattern.success_rate
                ))
            
            # Step 4: If no patterns found, create heuristic chains
            if not recommendations:
                recommendations = await self._create_heuristic_chains(task_description, task_type, limit)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Tool chain recommendation failed: {e}")
            return []
    
    
    async def auto_select_tools(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.7
    ) -> Tuple[List[ToolDefinition], float]:
        """
        Automatically select the best tools for a task
        
        Args:
            task_description: Natural language description of the task
            context: Additional context
            confidence_threshold: Minimum confidence to auto-select
        
        Returns:
            Tuple of (selected tools, confidence score)
        """
        try:
            # Get recommendations
            recommendations = await self.recommend_tools(task_description, context, limit=3)
            
            if not recommendations:
                return [], 0.0
            
            # Check if top recommendation meets confidence threshold
            top_recommendation = recommendations[0]
            
            if top_recommendation.combined_score >= confidence_threshold:
                # High confidence - auto-select top tool
                return [top_recommendation.tool], top_recommendation.combined_score
            elif len(recommendations) >= 2 and recommendations[1].combined_score >= confidence_threshold * 0.9:
                # Multiple good options - return top 2
                return [r.tool for r in recommendations[:2]], recommendations[0].combined_score
            else:
                # Low confidence - return empty list (manual selection needed)
                return [], top_recommendation.combined_score
            
        except Exception as e:
            logger.error(f"Auto-selection failed: {e}")
            return [], 0.0
    
    
    async def record_tool_execution(
        self,
        tool_id: str,
        success: bool,
        execution_time: float,
        user_rating: Optional[float] = None
    ):
        """Record tool execution for performance tracking"""
        try:
            # Get or create metrics
            if tool_id not in self.performance_cache:
                self.performance_cache[tool_id] = ToolPerformanceMetrics(tool_id=tool_id)
            
            metrics = self.performance_cache[tool_id]
            
            # Update metrics
            metrics.total_executions += 1
            if success:
                metrics.successful_executions += 1
            else:
                metrics.failed_executions += 1
            
            # Update average execution time (running average)
            metrics.average_execution_time = (
                (metrics.average_execution_time * (metrics.total_executions - 1) + execution_time)
                / metrics.total_executions
            )
            
            # Update success rate
            metrics.success_rate = metrics.successful_executions / metrics.total_executions
            
            # Update last used
            metrics.last_used = datetime.utcnow()
            
            # Add user rating if provided
            if user_rating is not None:
                metrics.user_ratings.append(user_rating)
                metrics.average_rating = sum(metrics.user_ratings) / len(metrics.user_ratings)
            
            # Save to database
            await self.performance_collection.update_one(
                {"tool_id": tool_id},
                {"$set": metrics.dict()},
                upsert=True
            )
            
            logger.info(f"✅ Recorded execution for tool {tool_id}: success={success}, time={execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to record tool execution: {e}")
    
    
    # ==================== HELPER METHODS ====================
    
    async def _calculate_semantic_similarity(
        self,
        task_description: str,
        tools: List[ToolDefinition]
    ) -> Dict[str, float]:
        """Calculate semantic similarity using Vector Search"""
        # TODO: Implement semantic similarity using embeddings
        # For now, fallback to keyword matching
        return self._calculate_keyword_similarity(task_description, tools)
    
    
    def _calculate_keyword_similarity(
        self,
        task_description: str,
        tools: List[ToolDefinition]
    ) -> Dict[str, float]:
        """Calculate similarity based on keyword matching"""
        scores = {}
        task_lower = task_description.lower()
        task_words = set(task_lower.split())
        
        for tool in tools:
            # Combine tool name, description, and tags
            tool_text = f"{tool.name} {tool.description} {' '.join(tool.tags)}".lower()
            tool_words = set(tool_text.split())
            
            # Calculate Jaccard similarity
            intersection = task_words & tool_words
            union = task_words | tool_words
            
            if union:
                scores[tool.tool_id] = len(intersection) / len(union)
            else:
                scores[tool.tool_id] = 0.0
        
        return scores
    
    
    def _get_performance_scores(self, tools: List[ToolDefinition]) -> Dict[str, float]:
        """Get performance scores for tools"""
        scores = {}
        
        for tool in tools:
            if tool.tool_id in self.performance_cache:
                metrics = self.performance_cache[tool.tool_id]
                # Combine success rate and user ratings
                scores[tool.tool_id] = (metrics.success_rate * 0.7) + (metrics.average_rating / 5.0 * 0.3)
            else:
                scores[tool.tool_id] = 0.5  # Default score for new tools
        
        return scores
    
    
    def _generate_recommendation_reason(
        self,
        tool: ToolDefinition,
        relevance_score: float,
        performance_score: float,
        task_description: str
    ) -> str:
        """Generate human-readable reason for recommendation"""
        reasons = []
        
        if relevance_score > 0.7:
            reasons.append(f"Highly relevant to '{task_description[:50]}...'")
        elif relevance_score > 0.5:
            reasons.append(f"Relevant to the task")
        
        if performance_score > 0.8:
            reasons.append("excellent track record")
        elif performance_score > 0.6:
            reasons.append("good performance history")
        
        if tool.tool_id in self.performance_cache:
            metrics = self.performance_cache[tool.tool_id]
            if metrics.total_executions > 10:
                reasons.append(f"used {metrics.total_executions} times")
        
        return " - ".join(reasons) if reasons else "Matches task requirements"
    
    
    def _identify_task_type(self, task_description: str) -> str:
        """Identify task type from description"""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["research", "find", "search", "lookup"]):
            return "research"
        elif any(word in task_lower for word in ["analyze", "process", "transform"]):
            return "data_processing"
        elif any(word in task_lower for word in ["create", "generate", "build"]):
            return "creation"
        elif any(word in task_lower for word in ["test", "validate", "verify"]):
            return "validation"
        else:
            return "general"
    
    
    async def _create_heuristic_chains(
        self,
        task_description: str,
        task_type: str,
        limit: int
    ) -> List[ToolChainRecommendation]:
        """Create heuristic tool chains when no patterns exist"""
        # Get individual tool recommendations
        tool_recommendations = await self.recommend_tools(task_description, limit=5)
        
        if len(tool_recommendations) < 2:
            return []
        
        # Create a simple chain from top recommendations
        chain = ToolChainRecommendation(
            tools=[r.tool for r in tool_recommendations[:3]],
            relevance_score=tool_recommendations[0].relevance_score,
            reason=f"Suggested tool sequence for {task_type} tasks",
            estimated_execution_time=5.0,  # Default estimate
            success_probability=0.7  # Default probability
        )
        
        return [chain]

