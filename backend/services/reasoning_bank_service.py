"""
Reasoning Bank Service - ClaudeFlow-inspired Pattern Storage

This service manages the storage, retrieval, and matching of reasoning patterns.
It combines MongoDB for structured storage with Vector Search for semantic discovery.

Key Features:
- Store successful reasoning patterns
- Semantic pattern matching using Vector Search
- Hybrid search (keyword + semantic)
- Pattern evolution and versioning
- Usage tracking and effectiveness metrics
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING

from models.reasoning_pattern import (
    ReasoningPattern,
    PatternSearchQuery,
    PatternSearchResult,
    PatternCreateRequest,
    PatternUpdateRequest,
    PatternUsageRecord,
    PatternStats,
    PatternEvolutionRequest,
    SuccessMetrics,
    SolutionStep
)

logger = logging.getLogger(__name__)


class ReasoningBankService:
    """
    Service for managing reasoning patterns (ClaudeFlow-inspired)
    
    Combines MongoDB for structured storage with Vector Search for semantic discovery.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase, vector_search_service=None):
        """
        Initialize ReasoningBank Service

        Args:
            db: MongoDB database instance
            vector_search_service: Optional Vector Search Service for semantic matching
        """
        self.db = db
        self.collection = db.reasoning_patterns
        self.usage_collection = db.pattern_usage
        self.vector_search = vector_search_service
        self.vector_namespace = "reasoning_patterns"
        self._indexes_created = False

        logger.info("✅ ReasoningBank Service initialized")

    async def _ensure_indexes(self):
        """Ensure required indexes exist"""
        if self._indexes_created:
            return

        try:
            # Create text index for keyword search
            await self.collection.create_index([
                ("title", "text"),
                ("problem_description", "text"),
                ("solution_summary", "text"),
                ("tags", "text")
            ])

            # Create index on pattern_id for fast lookups
            await self.collection.create_index("pattern_id", unique=True)

            # Create index on pattern_type for filtering
            await self.collection.create_index("pattern_type")

            self._indexes_created = True
            logger.debug("✅ Database indexes created")
        except Exception as e:
            logger.warning(f"⚠️ Error creating indexes: {e}")
    
    async def create_pattern(
        self,
        request: PatternCreateRequest
    ) -> ReasoningPattern:
        """
        Create a new reasoning pattern

        Args:
            request: Pattern creation request

        Returns:
            Created pattern
        """
        try:
            # Ensure indexes exist
            await self._ensure_indexes()
            # Create pattern object
            from models.reasoning_pattern import PatternContext

            pattern = ReasoningPattern(
                pattern_type=request.pattern_type,
                title=request.title,
                problem_description=request.problem_description,
                solution_steps=request.solution_steps,
                solution_summary=request.solution_summary,
                tags=request.tags or [],
                key_insights=request.key_insights or [],
                context=request.context if request.context else PatternContext(),
                created_by=request.created_by
            )
            
            # Store in MongoDB
            pattern_dict = pattern.dict()
            await self.collection.insert_one(pattern_dict)
            
            # Store in Vector Search for semantic discovery
            if self.vector_search:
                try:
                    # Create searchable content
                    content = self._create_searchable_content(pattern)
                    
                    await self.vector_search.store_memory(
                        content=content,
                        namespace=self.vector_namespace,
                        metadata={
                            "pattern_id": pattern.pattern_id,
                            "pattern_type": pattern.pattern_type,
                            "title": pattern.title,
                            "tags": ",".join(pattern.tags),
                            "created_at": pattern.created_at.isoformat()
                        }
                    )
                    logger.info(f"📚 Pattern stored in vector database: {pattern.pattern_id}")
                except Exception as e:
                    logger.warning(f"Failed to store pattern in vector database: {e}")
            
            logger.info(f"✅ Created reasoning pattern: {pattern.title} ({pattern.pattern_id})")
            return pattern
            
        except Exception as e:
            logger.error(f"Error creating pattern: {e}")
            raise
    
    async def get_pattern(self, pattern_id: str) -> Optional[ReasoningPattern]:
        """Get a pattern by ID"""
        try:
            pattern_dict = await self.collection.find_one({"pattern_id": pattern_id})
            if pattern_dict:
                pattern_dict.pop("_id", None)
                return ReasoningPattern(**pattern_dict)
            return None
        except Exception as e:
            logger.error(f"Error getting pattern {pattern_id}: {e}")
            return None
    
    async def update_pattern(
        self,
        pattern_id: str,
        request: PatternUpdateRequest
    ) -> Optional[ReasoningPattern]:
        """Update an existing pattern"""
        try:
            # Get existing pattern
            pattern = await self.get_pattern(pattern_id)
            if not pattern:
                return None
            
            # Update fields
            update_data = request.dict(exclude_unset=True)
            update_data["updated_at"] = datetime.utcnow()
            
            await self.collection.update_one(
                {"pattern_id": pattern_id},
                {"$set": update_data}
            )
            
            # Get updated pattern
            updated_pattern = await self.get_pattern(pattern_id)
            
            # Update vector search if content changed
            if self.vector_search and any(k in update_data for k in ["problem_description", "solution_summary", "tags"]):
                try:
                    content = self._create_searchable_content(updated_pattern)
                    # Note: Vector search doesn't support updates, so we'd need to delete and re-add
                    # For now, we'll just log this
                    logger.info(f"📚 Pattern updated in database: {pattern_id}")
                except Exception as e:
                    logger.warning(f"Failed to update pattern in vector database: {e}")
            
            logger.info(f"✅ Updated pattern: {pattern_id}")
            return updated_pattern
            
        except Exception as e:
            logger.error(f"Error updating pattern {pattern_id}: {e}")
            return None
    
    async def search_patterns(
        self,
        query: PatternSearchQuery
    ) -> List[PatternSearchResult]:
        """
        Search for patterns using hybrid approach (keyword + semantic)

        Args:
            query: Search query

        Returns:
            List of matching patterns with relevance scores
        """
        try:
            # Ensure indexes exist
            await self._ensure_indexes()

            results = []
            
            # 1. Semantic search using Vector Search
            semantic_results = []
            if self.vector_search:
                try:
                    memories = await self.vector_search.search_memories(
                        query=query.query,
                        namespace=self.vector_namespace,
                        limit=query.limit * 2  # Get more for filtering
                    )
                    semantic_results = memories
                    logger.info(f"🔍 Found {len(semantic_results)} patterns via semantic search")
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")
            
            # 2. Keyword search using MongoDB
            keyword_filter = {}
            if query.pattern_type:
                keyword_filter["pattern_type"] = query.pattern_type
            if query.tags:
                keyword_filter["tags"] = {"$in": query.tags}
            if query.min_success_rate is not None:
                keyword_filter["success_metrics.success_rate"] = {"$gte": query.min_success_rate}
            
            # Text search on problem description and solution summary
            if query.query:
                keyword_filter["$or"] = [
                    {"problem_description": {"$regex": query.query, "$options": "i"}},
                    {"solution_summary": {"$regex": query.query, "$options": "i"}},
                    {"title": {"$regex": query.query, "$options": "i"}},
                    {"tags": {"$regex": query.query, "$options": "i"}}
                ]
            
            keyword_patterns = await self.collection.find(keyword_filter).limit(query.limit).to_list(length=query.limit)
            logger.info(f"🔍 Found {len(keyword_patterns)} patterns via keyword search")
            
            # 3. Combine and rank results
            pattern_scores = {}
            
            # Add semantic results
            for memory in semantic_results:
                pattern_id = memory.get("metadata", {}).get("pattern_id")
                if pattern_id:
                    # Distance to relevance score (lower distance = higher relevance)
                    distance = memory.get("distance", 1.0)
                    relevance = max(0.0, 1.0 - distance)
                    pattern_scores[pattern_id] = {
                        "semantic_score": relevance,
                        "keyword_score": 0.0,
                        "match_type": "semantic"
                    }
            
            # Add keyword results
            for pattern_dict in keyword_patterns:
                pattern_id = pattern_dict.get("pattern_id")
                if pattern_id:
                    if pattern_id in pattern_scores:
                        pattern_scores[pattern_id]["keyword_score"] = 0.5
                        pattern_scores[pattern_id]["match_type"] = "hybrid"
                    else:
                        pattern_scores[pattern_id] = {
                            "semantic_score": 0.0,
                            "keyword_score": 0.5,
                            "match_type": "keyword"
                        }
            
            # 4. Fetch full patterns and create results
            for pattern_id, scores in pattern_scores.items():
                pattern = await self.get_pattern(pattern_id)
                if pattern:
                    # Calculate combined relevance score
                    relevance = (scores["semantic_score"] * 0.7) + (scores["keyword_score"] * 0.3)
                    
                    # Boost by success rate
                    relevance *= (0.5 + (pattern.success_metrics.success_rate * 0.5))
                    
                    results.append(PatternSearchResult(
                        pattern=pattern,
                        relevance_score=min(1.0, relevance),
                        match_type=scores["match_type"],
                        match_explanation=f"Matched via {scores['match_type']} search"
                    ))
            
            # Sort by relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit results
            results = results[:query.limit]
            
            logger.info(f"✅ Returning {len(results)} pattern search results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching patterns: {e}")
            return []
    
    async def record_usage(self, usage: PatternUsageRecord) -> bool:
        """Record pattern usage for tracking effectiveness"""
        try:
            # Store usage record
            await self.usage_collection.insert_one(usage.dict())

            # Update pattern metrics
            pattern = await self.get_pattern(usage.pattern_id)
            if pattern:
                # Update usage count
                new_usage_count = pattern.success_metrics.usage_count + 1

                # Update success rate (running average)
                old_rate = pattern.success_metrics.success_rate
                old_count = pattern.success_metrics.usage_count
                new_rate = ((old_rate * old_count) + (1.0 if usage.success else 0.0)) / new_usage_count

                # Update execution time (running average)
                new_exec_time = None
                if usage.execution_time_ms and pattern.success_metrics.execution_time_ms:
                    new_exec_time = ((pattern.success_metrics.execution_time_ms * old_count) + usage.execution_time_ms) / new_usage_count
                elif usage.execution_time_ms:
                    new_exec_time = usage.execution_time_ms

                # Update feedback score (running average)
                new_feedback = None
                if usage.feedback_score and pattern.success_metrics.feedback_score:
                    new_feedback = ((pattern.success_metrics.feedback_score * old_count) + usage.feedback_score) / new_usage_count
                elif usage.feedback_score:
                    new_feedback = usage.feedback_score

                # Update pattern
                await self.collection.update_one(
                    {"pattern_id": usage.pattern_id},
                    {
                        "$set": {
                            "success_metrics.usage_count": new_usage_count,
                            "success_metrics.success_rate": new_rate,
                            "success_metrics.last_used_at": usage.used_at,
                            "success_metrics.execution_time_ms": new_exec_time,
                            "success_metrics.feedback_score": new_feedback,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )

                logger.info(f"📊 Recorded usage for pattern {usage.pattern_id} (success: {usage.success})")
                return True

            return False

        except Exception as e:
            logger.error(f"Error recording pattern usage: {e}")
            return False

    async def evolve_pattern(
        self,
        request: PatternEvolutionRequest
    ) -> Optional[ReasoningPattern]:
        """Create an evolved version of an existing pattern"""
        try:
            # Get parent pattern
            parent = await self.get_pattern(request.parent_pattern_id)
            if not parent:
                logger.error(f"Parent pattern not found: {request.parent_pattern_id}")
                return None

            # Create new pattern based on parent
            evolved_pattern = ReasoningPattern(
                pattern_type=parent.pattern_type,
                title=f"{parent.title} (v{parent.version + 1})",
                problem_description=parent.problem_description,
                solution_steps=request.new_solution_steps or parent.solution_steps,
                solution_summary=f"{parent.solution_summary}\n\nImprovements: {request.improvements}",
                tags=parent.tags,
                key_insights=(parent.key_insights + (request.new_insights or [])),
                context=parent.context,
                created_by=request.created_by,
                version=parent.version + 1,
                parent_pattern_id=request.parent_pattern_id
            )

            # Store evolved pattern
            pattern_dict = evolved_pattern.dict()
            await self.collection.insert_one(pattern_dict)

            # Update parent to mark it as superseded
            await self.collection.update_one(
                {"pattern_id": request.parent_pattern_id},
                {"$set": {"supersedes": evolved_pattern.pattern_id}}
            )

            logger.info(f"🧬 Evolved pattern {request.parent_pattern_id} → {evolved_pattern.pattern_id}")
            return evolved_pattern

        except Exception as e:
            logger.error(f"Error evolving pattern: {e}")
            return None

    async def get_stats(self) -> PatternStats:
        """Get statistics about pattern usage"""
        try:
            # Total patterns
            total = await self.collection.count_documents({})

            # Patterns by type
            pipeline = [
                {"$group": {"_id": "$pattern_type", "count": {"$sum": 1}}}
            ]
            by_type_cursor = self.collection.aggregate(pipeline)
            by_type = {doc["_id"]: doc["count"] async for doc in by_type_cursor}

            # Most used patterns
            most_used = await self.collection.find().sort("success_metrics.usage_count", DESCENDING).limit(10).to_list(length=10)
            most_used_list = [
                {
                    "pattern_id": p["pattern_id"],
                    "title": p["title"],
                    "usage_count": p["success_metrics"]["usage_count"]
                }
                for p in most_used
            ]

            # Highest rated patterns
            highest_rated = await self.collection.find(
                {"success_metrics.feedback_score": {"$ne": None}}
            ).sort("success_metrics.feedback_score", DESCENDING).limit(10).to_list(length=10)
            highest_rated_list = [
                {
                    "pattern_id": p["pattern_id"],
                    "title": p["title"],
                    "feedback_score": p["success_metrics"].get("feedback_score")
                }
                for p in highest_rated
            ]

            # Recent patterns
            recent = await self.collection.find().sort("created_at", DESCENDING).limit(10).to_list(length=10)
            recent_list = [
                {
                    "pattern_id": p["pattern_id"],
                    "title": p["title"],
                    "created_at": p["created_at"].isoformat()
                }
                for p in recent
            ]

            # Average success rate
            pipeline = [
                {"$group": {"_id": None, "avg_success": {"$avg": "$success_metrics.success_rate"}, "total_usage": {"$sum": "$success_metrics.usage_count"}}}
            ]
            avg_cursor = self.collection.aggregate(pipeline)
            avg_doc = await avg_cursor.to_list(length=1)
            avg_success = avg_doc[0]["avg_success"] if avg_doc else 0.0
            total_usage = avg_doc[0]["total_usage"] if avg_doc else 0

            return PatternStats(
                total_patterns=total,
                patterns_by_type=by_type,
                most_used_patterns=most_used_list,
                highest_rated_patterns=highest_rated_list,
                recent_patterns=recent_list,
                average_success_rate=avg_success,
                total_usage_count=total_usage
            )

        except Exception as e:
            logger.error(f"Error getting pattern stats: {e}")
            return PatternStats(
                total_patterns=0,
                patterns_by_type={},
                most_used_patterns=[],
                highest_rated_patterns=[],
                recent_patterns=[],
                average_success_rate=0.0,
                total_usage_count=0
            )

    def _create_searchable_content(self, pattern: ReasoningPattern) -> str:
        """Create searchable content from pattern for vector storage"""
        parts = [
            f"Title: {pattern.title}",
            f"Type: {pattern.pattern_type}",
            f"Problem: {pattern.problem_description}",
            f"Solution: {pattern.solution_summary}",
        ]

        if pattern.key_insights:
            parts.append(f"Insights: {', '.join(pattern.key_insights)}")

        if pattern.tags:
            parts.append(f"Tags: {', '.join(pattern.tags)}")

        return "\n\n".join(parts)

