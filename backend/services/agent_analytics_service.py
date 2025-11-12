"""
Agent Analytics Service
Tracks and analyzes agent performance metrics

GitHub Issue #3: Agent Analytics
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from database import Database
from services.event_bus import subscribe_to_event
from models.event_models import EventType

logger = logging.getLogger(__name__)


class AgentAnalyticsService:
    """
    Agent Analytics Service
    
    Tracks:
    - Agent execution metrics (count, duration, success rate)
    - Performance trends over time
    - Error patterns
    - Resource usage
    - Agent comparison metrics
    """
    
    def __init__(self, db: Optional[Database] = None):
        """Initialize agent analytics service"""
        self.db = db or Database()
        self.analytics_collection = "agent_analytics"
        self.executions_collection = "agent_executions"
        
        # Subscribe to agent events
        subscribe_to_event(EventType.AGENT_EXECUTION_STARTED, self._on_execution_started)
        subscribe_to_event(EventType.AGENT_EXECUTION_COMPLETED, self._on_execution_completed)
        subscribe_to_event(EventType.AGENT_EXECUTION_FAILED, self._on_execution_failed)
        
        logger.info("📊 Agent Analytics Service initialized")
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    async def _on_execution_started(self, event: Dict[str, Any]):
        """Handle agent execution started event"""
        try:
            data = event["data"]
            execution_record = {
                "execution_id": data.get("execution_id"),
                "agent_id": data.get("agent_id"),
                "agent_name": data.get("agent_name"),
                "started_at": event["timestamp"],
                "status": "running",
                "created_at": datetime.utcnow().isoformat(),
            }
            
            await self.db.insert_one(self.executions_collection, execution_record)
            logger.info(f"📊 Tracking execution: {execution_record['execution_id']}")
            
        except Exception as e:
            logger.error(f"❌ Error tracking execution start: {e}")
    
    async def _on_execution_completed(self, event: Dict[str, Any]):
        """Handle agent execution completed event"""
        try:
            data = event["data"]
            execution_id = data.get("execution_id")
            
            # Update execution record
            await self.db.update_one(
                self.executions_collection,
                {"execution_id": execution_id},
                {
                    "$set": {
                        "completed_at": event["timestamp"],
                        "status": "completed",
                        "duration_seconds": data.get("duration_seconds", 0),
                        "result": data.get("result"),
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                }
            )
            
            # Update agent analytics
            await self._update_agent_analytics(
                data.get("agent_id"),
                success=True,
                duration=data.get("duration_seconds", 0)
            )
            
            logger.info(f"✅ Execution completed: {execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Error tracking execution completion: {e}")
    
    async def _on_execution_failed(self, event: Dict[str, Any]):
        """Handle agent execution failed event"""
        try:
            data = event["data"]
            execution_id = data.get("execution_id")
            
            # Update execution record
            await self.db.update_one(
                self.executions_collection,
                {"execution_id": execution_id},
                {
                    "$set": {
                        "completed_at": event["timestamp"],
                        "status": "failed",
                        "duration_seconds": data.get("duration_seconds", 0),
                        "error": data.get("error"),
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                }
            )
            
            # Update agent analytics
            await self._update_agent_analytics(
                data.get("agent_id"),
                success=False,
                duration=data.get("duration_seconds", 0),
                error=data.get("error")
            )
            
            logger.info(f"❌ Execution failed: {execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Error tracking execution failure: {e}")
    
    # ========================================================================
    # Analytics Methods
    # ========================================================================
    
    async def _update_agent_analytics(
        self,
        agent_id: str,
        success: bool,
        duration: float,
        error: Optional[str] = None
    ):
        """Update agent analytics with execution data"""
        try:
            # Get or create analytics record
            analytics = await self.db.find_one(
                self.analytics_collection,
                {"agent_id": agent_id}
            )
            
            if not analytics:
                analytics = {
                    "agent_id": agent_id,
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "total_duration_seconds": 0.0,
                    "average_duration_seconds": 0.0,
                    "min_duration_seconds": float('inf'),
                    "max_duration_seconds": 0.0,
                    "success_rate": 0.0,
                    "error_patterns": {},
                    "created_at": datetime.utcnow().isoformat(),
                }
            
            # Update metrics
            analytics["total_executions"] += 1
            if success:
                analytics["successful_executions"] += 1
            else:
                analytics["failed_executions"] += 1
                
                # Track error patterns
                if error:
                    error_patterns = analytics.get("error_patterns", {})
                    error_type = error.split(":")[0] if ":" in error else "Unknown"
                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
                    analytics["error_patterns"] = error_patterns
            
            # Update duration metrics
            analytics["total_duration_seconds"] += duration
            analytics["average_duration_seconds"] = (
                analytics["total_duration_seconds"] / analytics["total_executions"]
            )
            analytics["min_duration_seconds"] = min(
                analytics.get("min_duration_seconds", float('inf')),
                duration
            )
            analytics["max_duration_seconds"] = max(
                analytics.get("max_duration_seconds", 0.0),
                duration
            )
            
            # Update success rate
            analytics["success_rate"] = (
                analytics["successful_executions"] / analytics["total_executions"] * 100
            )
            
            analytics["updated_at"] = datetime.utcnow().isoformat()
            
            # Upsert analytics record
            await self.db.update_one(
                self.analytics_collection,
                {"agent_id": agent_id},
                {"$set": analytics},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating agent analytics: {e}")
    
    async def get_agent_analytics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analytics for a specific agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Analytics data or None
        """
        try:
            analytics = await self.db.find_one(
                self.analytics_collection,
                {"agent_id": agent_id}
            )
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Error getting agent analytics: {e}")
            return None
    
    async def get_all_agent_analytics(self) -> List[Dict[str, Any]]:
        """
        Get analytics for all agents
        
        Returns:
            List of analytics data
        """
        try:
            analytics_list = await self.db.find(
                self.analytics_collection,
                {},
                sort=[("total_executions", -1)]
            )
            return analytics_list
            
        except Exception as e:
            logger.error(f"❌ Error getting all agent analytics: {e}")
            return []
    
    async def get_execution_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution history
        
        Args:
            agent_id: Filter by agent ID (optional)
            limit: Maximum number of records
            status: Filter by status (optional)
            
        Returns:
            List of execution records
        """
        try:
            query = {}
            if agent_id:
                query["agent_id"] = agent_id
            if status:
                query["status"] = status
            
            executions = await self.db.find(
                self.executions_collection,
                query,
                sort=[("created_at", -1)],
                limit=limit
            )
            return executions
            
        except Exception as e:
            logger.error(f"❌ Error getting execution history: {e}")
            return []
    
    async def get_performance_trends(
        self,
        agent_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance trends over time
        
        Args:
            agent_id: Agent ID
            days: Number of days to analyze
            
        Returns:
            Trend data
        """
        try:
            since = datetime.utcnow() - timedelta(days=days)
            
            executions = await self.db.find(
                self.executions_collection,
                {
                    "agent_id": agent_id,
                    "created_at": {"$gte": since.isoformat()}
                },
                sort=[("created_at", 1)]
            )
            
            # Group by day
            daily_stats = defaultdict(lambda: {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "total_duration": 0.0,
            })
            
            for execution in executions:
                created_at = datetime.fromisoformat(execution["created_at"])
                day_key = created_at.strftime("%Y-%m-%d")
                
                daily_stats[day_key]["executions"] += 1
                if execution.get("status") == "completed":
                    daily_stats[day_key]["successes"] += 1
                elif execution.get("status") == "failed":
                    daily_stats[day_key]["failures"] += 1
                
                daily_stats[day_key]["total_duration"] += execution.get("duration_seconds", 0)
            
            # Calculate daily metrics
            trends = []
            for day, stats in sorted(daily_stats.items()):
                trends.append({
                    "date": day,
                    "executions": stats["executions"],
                    "success_rate": (stats["successes"] / stats["executions"] * 100) if stats["executions"] > 0 else 0,
                    "average_duration": (stats["total_duration"] / stats["executions"]) if stats["executions"] > 0 else 0,
                })
            
            return {
                "agent_id": agent_id,
                "period_days": days,
                "trends": trends,
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance trends: {e}")
            return {"agent_id": agent_id, "period_days": days, "trends": []}

