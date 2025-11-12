"""
Query Optimizer Router for CPAS5

Phase 3, Task 3.1: Optimize Database Queries
REST API endpoints for query optimization and performance monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
import logging
from services.query_optimizer import QueryOptimizer, PaginationConfig, PaginatedResult
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/query-optimizer", tags=["Query Optimizer"])


async def get_optimizer() -> QueryOptimizer:
    """Get query optimizer service instance"""
    db = Database()
    await db.connect()
    return QueryOptimizer(db)


# ========================================================================
# PAGINATION ENDPOINTS
# ========================================================================

@router.get("/paginated/{collection_name}")
async def get_paginated_data(
    collection_name: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    sort_field: str = Query("created_at", description="Field to sort by"),
    sort_direction: int = Query(-1, ge=-1, le=1, description="Sort direction (-1 desc, 1 asc)"),
    optimizer: QueryOptimizer = Depends(get_optimizer)
):
    """
    Get paginated data from a collection.
    
    Phase 3, Task 3.1: Implement query result pagination for large datasets
    
    Example:
        GET /api/v1/query-optimizer/paginated/crm_contacts?page=2&page_size=50
    """
    try:
        config = PaginationConfig(
            page=page,
            page_size=page_size,
            sort_field=sort_field,
            sort_direction=sort_direction
        )
        
        result = await optimizer.paginated_find(collection_name, {}, config)
        
        return {
            "data": result.data,
            "pagination": {
                "total_count": result.total_count,
                "page": result.page,
                "page_size": result.page_size,
                "total_pages": result.total_pages,
                "has_next": result.has_next,
                "has_prev": result.has_prev
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Pagination error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# QUERY ANALYSIS ENDPOINTS
# ========================================================================

@router.post("/analyze/{collection_name}")
async def analyze_query(
    collection_name: str,
    query: Dict[str, Any],
    optimizer: QueryOptimizer = Depends(get_optimizer)
):
    """
    Analyze query performance and get optimization recommendations.
    
    Phase 3, Task 3.1: Add query performance logging and monitoring
    
    Example:
        POST /api/v1/query-optimizer/analyze/crm_contacts
        Body: {"status": "active"}
    """
    try:
        analysis = await optimizer.analyze_query_performance(collection_name, query)
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Query analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-indexes/{collection_name}")
async def recommend_indexes(
    collection_name: str,
    sample_queries: List[Dict[str, Any]],
    optimizer: QueryOptimizer = Depends(get_optimizer)
):
    """
    Get index recommendations based on sample queries.
    
    Phase 3, Task 3.1: Add database indexes for frequently queried fields
    
    Example:
        POST /api/v1/query-optimizer/recommend-indexes/crm_contacts
        Body: [{"status": "active"}, {"email": "test@example.com"}]
    """
    try:
        recommendations = await optimizer.recommend_indexes(collection_name, sample_queries)
        return {
            "collection": collection_name,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"❌ Index recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# BULK OPERATIONS ENDPOINTS
# ========================================================================

@router.post("/bulk-insert/{collection_name}")
async def bulk_insert(
    collection_name: str,
    documents: List[Dict[str, Any]],
    batch_size: int = Query(1000, ge=1, le=10000, description="Batch size"),
    optimizer: QueryOptimizer = Depends(get_optimizer)
):
    """
    Perform efficient bulk insert operation.
    
    Phase 3, Task 3.1: Implement efficient bulk operations
    
    Example:
        POST /api/v1/query-optimizer/bulk-insert/crm_contacts?batch_size=500
        Body: [{"name": "Contact 1"}, {"name": "Contact 2"}, ...]
    """
    try:
        result = await optimizer.bulk_insert(collection_name, documents, batch_size)
        return result
        
    except Exception as e:
        logger.error(f"❌ Bulk insert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# AGGREGATION ENDPOINTS
# ========================================================================

@router.post("/aggregate/{collection_name}")
async def execute_aggregation(
    collection_name: str,
    pipeline: List[Dict[str, Any]],
    allow_disk_use: bool = Query(True, description="Allow disk use for large aggregations"),
    optimizer: QueryOptimizer = Depends(get_optimizer)
):
    """
    Execute optimized aggregation pipeline.
    
    Phase 3, Task 3.1: Optimize aggregation pipelines in MongoDB
    
    Example:
        POST /api/v1/query-optimizer/aggregate/crm_deals?allow_disk_use=true
        Body: [{"$match": {"stage": "negotiation"}}, {"$group": {"_id": "$company_id", "total": {"$sum": "$value"}}}]
    """
    try:
        results = await optimizer.optimized_aggregate(collection_name, pipeline, allow_disk_use)
        return {
            "collection": collection_name,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"❌ Aggregation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# CACHE MANAGEMENT ENDPOINTS
# ========================================================================

@router.delete("/cache")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Pattern to match cache keys"),
    optimizer: QueryOptimizer = Depends(get_optimizer)
):
    """
    Clear query cache.
    
    Example:
        DELETE /api/v1/query-optimizer/cache?pattern=count_crm
    """
    try:
        optimizer.clear_cache(pattern)
        return {
            "success": True,
            "message": f"Cache cleared{' for pattern: ' + pattern if pattern else ''}"
        }
        
    except Exception as e:
        logger.error(f"❌ Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats(optimizer: QueryOptimizer = Depends(get_optimizer)):
    """
    Get cache statistics.
    
    Example:
        GET /api/v1/query-optimizer/cache/stats
    """
    try:
        stats = optimizer.get_cache_stats()
        return stats
        
    except Exception as e:
        logger.error(f"❌ Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# OPTIMIZATION REPORTING ENDPOINTS
# ========================================================================

@router.get("/summary")
async def get_optimization_summary(optimizer: QueryOptimizer = Depends(get_optimizer)):
    """
    Get summary of all query optimizations.
    
    Example:
        GET /api/v1/query-optimizer/summary
    """
    try:
        summary = optimizer.get_optimization_summary()
        return summary
        
    except Exception as e:
        logger.error(f"❌ Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# HEALTH CHECK
# ========================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "module": "Query Optimizer",
        "features": [
            "Paginated Queries",
            "Query Performance Analysis",
            "Index Recommendations",
            "Bulk Operations",
            "Optimized Aggregations",
            "Query Caching",
            "N+1 Query Prevention",
            "Performance Monitoring"
        ],
        "endpoints": 10,
        "phase": "Phase 3, Task 3.1"
    }

