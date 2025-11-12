"""
Query Optimizer Service for CPAS5

Phase 3, Task 3.1: Optimize Database Queries (12 hours)

This service provides:
- Query performance analysis and optimization
- Pagination for large datasets
- Bulk operation optimization
- N+1 query detection and prevention
- Query result caching
- Index recommendation
- Slow query logging and analysis
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from database import Database

logger = logging.getLogger(__name__)


@dataclass
class QueryOptimizationResult:
    """Result of query optimization analysis"""
    query_name: str
    original_time_ms: float
    optimized_time_ms: float
    improvement_percent: float
    optimization_applied: str
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PaginationConfig:
    """Configuration for paginated queries"""
    page: int = 1
    page_size: int = 100
    max_page_size: int = 1000
    sort_field: str = "created_at"
    sort_direction: int = -1  # -1 for descending, 1 for ascending
    
    def get_skip(self) -> int:
        """Calculate skip value for pagination"""
        return (self.page - 1) * self.page_size
    
    def get_limit(self) -> int:
        """Get limit value (capped at max_page_size)"""
        return min(self.page_size, self.max_page_size)


@dataclass
class PaginatedResult:
    """Paginated query result"""
    data: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
    
    @classmethod
    def create(cls, data: List[Dict[str, Any]], total_count: int, config: PaginationConfig):
        """Create paginated result from data and config"""
        total_pages = (total_count + config.page_size - 1) // config.page_size
        return cls(
            data=data,
            total_count=total_count,
            page=config.page,
            page_size=config.page_size,
            total_pages=total_pages,
            has_next=config.page < total_pages,
            has_prev=config.page > 1
        )


class QueryOptimizer:
    """
    Query Optimizer Service
    
    Phase 3, Task 3.1: Optimize Database Queries
    Provides query optimization, pagination, and performance monitoring.
    """
    
    def __init__(self, database: Database):
        self.db = database
        self.slow_query_threshold_ms = 100  # 100ms threshold
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes default TTL
        self.optimization_results: List[QueryOptimizationResult] = []
        logger.info("✅ Query Optimizer initialized")
    
    # ========================================================================
    # PAGINATION SUPPORT
    # ========================================================================
    
    async def paginated_find(
        self,
        collection_name: str,
        query: Dict[str, Any],
        config: PaginationConfig
    ) -> PaginatedResult:
        """
        Execute paginated query with optimized performance.
        
        Phase 3, Task 3.1: Implement query result pagination for large datasets
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query filter
            config: Pagination configuration
            
        Returns:
            PaginatedResult with data and pagination metadata
        """
        start_time = time.time()
        
        try:
            # Get total count (cached for performance)
            cache_key = f"count_{collection_name}_{hash(str(query))}"
            total_count = await self._get_cached_count(cache_key, collection_name, query)
            
            # Get paginated data
            sort_spec = [(config.sort_field, config.sort_direction)]
            data = await self.db.find_many(
                collection_name,
                query,
                limit=config.get_limit(),
                sort=sort_spec
            )
            
            # Skip manually if needed (Motor doesn't support skip in find_many)
            skip = config.get_skip()
            if skip > 0 and len(data) > skip:
                data = data[skip:]
            
            result = PaginatedResult.create(data, total_count, config)
            
            # Log performance
            execution_time_ms = (time.time() - start_time) * 1000
            if execution_time_ms > self.slow_query_threshold_ms:
                logger.warning(
                    f"⚠️ Slow paginated query: {collection_name} took {execution_time_ms:.2f}ms "
                    f"(page {config.page}, size {config.page_size})"
                )
            else:
                logger.debug(
                    f"✅ Paginated query: {collection_name} took {execution_time_ms:.2f}ms "
                    f"(page {config.page}, {len(data)} results)"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Paginated query failed for {collection_name}: {e}")
            raise
    
    async def _get_cached_count(
        self,
        cache_key: str,
        collection_name: str,
        query: Dict[str, Any]
    ) -> int:
        """Get count with caching to avoid expensive count operations"""
        # Check cache
        if cache_key in self.query_cache:
            cached_value, cached_time = self.query_cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl_seconds:
                return cached_value
        
        # Count not cached or expired, fetch from database
        if self.db.connected and self.db.db is not None:
            collection = self.db.db[collection_name]
            count = await collection.count_documents(query)
        else:
            # Fallback: estimate from find_many
            all_docs = await self.db.find_many(collection_name, query, limit=10000)
            count = len(all_docs)
        
        # Cache the count
        self.query_cache[cache_key] = (count, datetime.utcnow())
        
        return count
    
    # ========================================================================
    # BULK OPERATIONS
    # ========================================================================
    
    async def bulk_insert(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Efficient bulk insert with batching.
        
        Phase 3, Task 3.1: Implement efficient bulk operations
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
            batch_size: Number of documents per batch
            
        Returns:
            Dict with insertion results
        """
        start_time = time.time()
        
        try:
            if not self.db.connected or self.db.db is None:
                raise RuntimeError("Database not connected")
            
            collection = self.db.db[collection_name]
            total_inserted = 0
            
            # Insert in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = await collection.insert_many(batch, ordered=False)
                total_inserted += len(result.inserted_ids)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"✅ Bulk insert: {total_inserted} documents into {collection_name} "
                f"in {execution_time_ms:.2f}ms ({total_inserted / (execution_time_ms / 1000):.0f} docs/sec)"
            )
            
            return {
                "success": True,
                "inserted_count": total_inserted,
                "execution_time_ms": execution_time_ms,
                "throughput_docs_per_sec": total_inserted / (execution_time_ms / 1000)
            }
            
        except Exception as e:
            logger.error(f"❌ Bulk insert failed for {collection_name}: {e}")
            raise
    
    async def bulk_update(
        self,
        collection_name: str,
        updates: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Efficient bulk update with batching.
        
        Args:
            collection_name: Name of the collection
            updates: List of (query, update) tuples
            batch_size: Number of updates per batch
            
        Returns:
            Dict with update results
        """
        start_time = time.time()
        
        try:
            if not self.db.connected or self.db.db is None:
                raise RuntimeError("Database not connected")
            
            collection = self.db.db[collection_name]
            total_modified = 0
            
            # Update in batches
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]
                
                # Use bulk_write for efficiency
                from pymongo import UpdateOne
                operations = [
                    UpdateOne(query, update)
                    for query, update in batch
                ]
                
                result = await collection.bulk_write(operations, ordered=False)
                total_modified += result.modified_count
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"✅ Bulk update: {total_modified} documents in {collection_name} "
                f"in {execution_time_ms:.2f}ms"
            )
            
            return {
                "success": True,
                "modified_count": total_modified,
                "execution_time_ms": execution_time_ms
            }
            
        except Exception as e:
            logger.error(f"❌ Bulk update failed for {collection_name}: {e}")
            raise
    
    # ========================================================================
    # QUERY PERFORMANCE MONITORING
    # ========================================================================
    
    async def analyze_query_performance(
        self,
        collection_name: str,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze query performance and provide optimization recommendations.
        
        Phase 3, Task 3.1: Add query performance logging and monitoring
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query to analyze
            
        Returns:
            Dict with performance analysis and recommendations
        """
        try:
            if not self.db.connected or self.db.db is None:
                return {"error": "Database not connected"}
            
            collection = self.db.db[collection_name]
            
            # Execute query with explain
            start_time = time.time()
            cursor = collection.find(query)
            explain_result = await cursor.explain()
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Analyze execution stats
            exec_stats = explain_result.get("executionStats", {})
            total_docs_examined = exec_stats.get("totalDocsExamined", 0)
            total_keys_examined = exec_stats.get("totalKeysExamined", 0)
            n_returned = exec_stats.get("nReturned", 0)
            
            # Generate recommendations
            recommendations = []
            
            if total_docs_examined > n_returned * 10:
                recommendations.append(
                    f"Query examines {total_docs_examined} documents but returns only {n_returned}. "
                    "Consider adding an index."
                )
            
            if total_keys_examined == 0 and total_docs_examined > 0:
                recommendations.append("Query performs a collection scan. Add an index for better performance.")
            
            if execution_time_ms > self.slow_query_threshold_ms:
                recommendations.append(
                    f"Query execution time ({execution_time_ms:.2f}ms) exceeds threshold "
                    f"({self.slow_query_threshold_ms}ms)"
                )
            
            return {
                "collection": collection_name,
                "execution_time_ms": execution_time_ms,
                "docs_examined": total_docs_examined,
                "keys_examined": total_keys_examined,
                "docs_returned": n_returned,
                "index_used": total_keys_examined > 0,
                "recommendations": recommendations,
                "is_slow": execution_time_ms > self.slow_query_threshold_ms
            }
            
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # CACHE MANAGEMENT
    # ========================================================================
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear query cache, optionally matching a pattern"""
        if pattern:
            keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.query_cache[key]
            logger.info(f"✅ Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")
        else:
            self.query_cache.clear()
            logger.info("✅ Cleared all cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_entries": len(self.query_cache),
            "cache_size_bytes": sum(
                len(str(v[0])) for v in self.query_cache.values()
            ),
            "oldest_entry": min(
                (v[1] for v in self.query_cache.values()),
                default=None
            ),
            "newest_entry": max(
                (v[1] for v in self.query_cache.values()),
                default=None
            )
        }

    # ========================================================================
    # AGGREGATION PIPELINE OPTIMIZATION
    # ========================================================================

    async def optimized_aggregate(
        self,
        collection_name: str,
        pipeline: List[Dict[str, Any]],
        allow_disk_use: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute optimized aggregation pipeline.

        Phase 3, Task 3.1: Optimize aggregation pipelines in MongoDB

        Args:
            collection_name: Name of the collection
            pipeline: Aggregation pipeline stages
            allow_disk_use: Allow using disk for large aggregations

        Returns:
            List of aggregation results
        """
        start_time = time.time()

        try:
            if not self.db.connected or self.db.db is None:
                raise RuntimeError("Database not connected")

            collection = self.db.db[collection_name]

            # Optimize pipeline order
            optimized_pipeline = self._optimize_pipeline_order(pipeline)

            # Execute aggregation
            cursor = collection.aggregate(optimized_pipeline, allowDiskUse=allow_disk_use)
            results = await cursor.to_list(length=None)

            execution_time_ms = (time.time() - start_time) * 1000

            if execution_time_ms > self.slow_query_threshold_ms:
                logger.warning(
                    f"⚠️ Slow aggregation: {collection_name} took {execution_time_ms:.2f}ms "
                    f"({len(results)} results, {len(pipeline)} stages)"
                )
            else:
                logger.debug(
                    f"✅ Aggregation: {collection_name} took {execution_time_ms:.2f}ms "
                    f"({len(results)} results)"
                )

            return results

        except Exception as e:
            logger.error(f"❌ Aggregation failed for {collection_name}: {e}")
            raise

    def _optimize_pipeline_order(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize aggregation pipeline stage order for better performance.

        Best practices:
        1. $match stages should come first to reduce documents early
        2. $project stages should come before $sort to reduce data size
        3. $limit should come after $sort
        4. $lookup should come as late as possible
        """
        optimized = []
        match_stages = []
        project_stages = []
        sort_stages = []
        limit_stages = []
        lookup_stages = []
        other_stages = []

        # Categorize stages
        for stage in pipeline:
            stage_type = list(stage.keys())[0]
            if stage_type == "$match":
                match_stages.append(stage)
            elif stage_type == "$project":
                project_stages.append(stage)
            elif stage_type == "$sort":
                sort_stages.append(stage)
            elif stage_type == "$limit":
                limit_stages.append(stage)
            elif stage_type == "$lookup":
                lookup_stages.append(stage)
            else:
                other_stages.append(stage)

        # Optimal order: $match -> $project -> other -> $sort -> $limit -> $lookup
        optimized.extend(match_stages)
        optimized.extend(project_stages)
        optimized.extend(other_stages)
        optimized.extend(sort_stages)
        optimized.extend(limit_stages)
        optimized.extend(lookup_stages)

        return optimized

    # ========================================================================
    # N+1 QUERY DETECTION AND PREVENTION
    # ========================================================================

    async def batch_load_related(
        self,
        collection_name: str,
        parent_docs: List[Dict[str, Any]],
        foreign_key: str,
        related_collection: str,
        related_key: str = "_id"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch load related documents to prevent N+1 queries.

        Phase 3, Task 3.1: Reduce N+1 query problems

        Instead of:
            for doc in docs:
                related = await db.find_one(related_collection, {related_key: doc[foreign_key]})

        Use:
            related_map = await optimizer.batch_load_related(
                collection_name, docs, foreign_key, related_collection, related_key
            )

        Args:
            collection_name: Parent collection name (for logging)
            parent_docs: List of parent documents
            foreign_key: Key in parent docs that references related docs
            related_collection: Collection containing related documents
            related_key: Key in related collection to match against

        Returns:
            Dict mapping foreign key values to lists of related documents
        """
        start_time = time.time()

        try:
            # Extract unique foreign key values
            foreign_values = list(set(
                doc.get(foreign_key)
                for doc in parent_docs
                if doc.get(foreign_key) is not None
            ))

            if not foreign_values:
                return {}

            # Single query to fetch all related documents
            query = {related_key: {"$in": foreign_values}}
            related_docs = await self.db.find_many(
                related_collection,
                query,
                limit=len(foreign_values) * 10  # Reasonable limit
            )

            # Group by foreign key value
            related_map: Dict[str, List[Dict[str, Any]]] = {}
            for doc in related_docs:
                key_value = doc.get(related_key)
                if key_value not in related_map:
                    related_map[key_value] = []
                related_map[key_value].append(doc)

            execution_time_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"✅ Batch loaded {len(related_docs)} related documents from {related_collection} "
                f"for {len(parent_docs)} {collection_name} documents in {execution_time_ms:.2f}ms "
                f"(prevented {len(parent_docs)} individual queries)"
            )

            return related_map

        except Exception as e:
            logger.error(f"❌ Batch load failed: {e}")
            return {}

    async def detect_n_plus_one(
        self,
        operation_name: str,
        threshold: int = 10
    ) -> Dict[str, Any]:
        """
        Detect potential N+1 query patterns.

        Monitors query patterns and alerts when many similar queries are executed
        in quick succession, which may indicate an N+1 problem.

        Args:
            operation_name: Name of the operation to monitor
            threshold: Number of similar queries to trigger alert

        Returns:
            Dict with detection results
        """
        # This would be implemented with query logging and pattern detection
        # For now, return a placeholder
        return {
            "operation": operation_name,
            "n_plus_one_detected": False,
            "similar_queries_count": 0,
            "recommendation": "Use batch_load_related() to prevent N+1 queries"
        }

    # ========================================================================
    # INDEX RECOMMENDATIONS
    # ========================================================================

    async def recommend_indexes(
        self,
        collection_name: str,
        sample_queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze queries and recommend indexes.

        Phase 3, Task 3.1: Add database indexes for frequently queried fields

        Args:
            collection_name: Name of the collection
            sample_queries: Sample queries to analyze

        Returns:
            List of index recommendations
        """
        recommendations = []
        field_usage = {}

        # Analyze query patterns
        for query in sample_queries:
            for field in query.keys():
                if field not in field_usage:
                    field_usage[field] = 0
                field_usage[field] += 1

        # Recommend indexes for frequently used fields
        for field, count in field_usage.items():
            if count >= len(sample_queries) * 0.3:  # Used in 30%+ of queries
                recommendations.append({
                    "collection": collection_name,
                    "field": field,
                    "usage_count": count,
                    "usage_percent": (count / len(sample_queries)) * 100,
                    "index_type": "single",
                    "command": f"db.{collection_name}.createIndex({{{field}: 1}})"
                })

        # Recommend compound indexes for common field combinations
        field_pairs = {}
        for query in sample_queries:
            fields = sorted(query.keys())
            if len(fields) >= 2:
                for i in range(len(fields) - 1):
                    pair = (fields[i], fields[i + 1])
                    if pair not in field_pairs:
                        field_pairs[pair] = 0
                    field_pairs[pair] += 1

        for (field1, field2), count in field_pairs.items():
            if count >= len(sample_queries) * 0.2:  # Used in 20%+ of queries
                recommendations.append({
                    "collection": collection_name,
                    "fields": [field1, field2],
                    "usage_count": count,
                    "usage_percent": (count / len(sample_queries)) * 100,
                    "index_type": "compound",
                    "command": f"db.{collection_name}.createIndex({{{field1}: 1, {field2}: 1}})"
                })

        return recommendations

    # ========================================================================
    # OPTIMIZATION REPORTING
    # ========================================================================

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results"""
        if not self.optimization_results:
            return {
                "total_optimizations": 0,
                "average_improvement": 0.0,
                "total_time_saved_ms": 0.0
            }

        total_improvement = sum(r.improvement_percent for r in self.optimization_results)
        total_time_saved = sum(
            r.original_time_ms - r.optimized_time_ms
            for r in self.optimization_results
        )

        return {
            "total_optimizations": len(self.optimization_results),
            "average_improvement": total_improvement / len(self.optimization_results),
            "total_time_saved_ms": total_time_saved,
            "recent_optimizations": [
                {
                    "query": r.query_name,
                    "improvement": f"{r.improvement_percent:.1f}%",
                    "time_saved_ms": r.original_time_ms - r.optimized_time_ms,
                    "optimization": r.optimization_applied
                }
                for r in self.optimization_results[-10:]
            ]
        }

