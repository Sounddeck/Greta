"""
GRETA Agent Management Framework Service
Complete agent lifecycle management with versioning, performance monitoring, and A/B testing
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class AgentManagementService:
    """
    Agent Management Framework - Complete agent lifecycle management
    """

    def __init__(self, db, llm_service, learning_service):
        self.db = db
        self.llm_service = llm_service
        self.learning_service = learning_service

        # Agent registry
        self.agents = {}
        self.agent_versions = {}  # agent_id -> [versions]
        self.agent_templates = {
            "autonomous": {"capabilities": ["decision_making", "task_execution", "learning"], "priority": "high"},
            "reactive": {"capabilities": ["event_response", "monitoring"], "priority": "medium"},
            "proactive": {"capabilities": ["prediction", "planning", "optimization"], "priority": "high"},
            "collaborative": {"capabilities": ["communication", "coordination", "teamwork"], "priority": "medium"},
            "specialized": {"capabilities": ["domain_expertise", "precision_tasks"], "priority": "high"}
        }

        # Performance monitoring
        self.performance_data = defaultdict(lambda: {
            "response_times": deque(maxlen=100),
            "success_count": 0,
            "failure_count": 0,
            "avg_response_time": 0.0
        })

        # A/B testing
        self.ab_tests = {}

        # Activity log
        self.activity_log = deque(maxlen=500)

        # Metrics
        self.metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "inactive_agents": 0,
            "total_tasks_executed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_versions": 0,
            "active_ab_tests": 0
        }
        
    async def initialize(self):
        """Initialize Agent Management Service"""
        try:
            logger.info("🚀 Initializing Agent Management Framework...")
            await self._load_agents()
            await self._load_versions()
            await self._load_ab_tests()
            logger.info("✅ Agent Management Framework initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Agent Management initialization failed: {e}")
            return False

    def _log_activity(self, action: str, details: str, metadata: Dict[str, Any] = None):
        """Log activity"""
        self.activity_log.append({
            "action": action,
            "details": details,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

    async def _load_agents(self):
        """Load existing agents from database"""
        try:
            agents = await self.db.find_many("managed_agents", {}, limit=1000)
            for agent in agents:
                self.agents[agent["agent_id"]] = agent
                if agent.get("status") == "active":
                    self.metrics["active_agents"] += 1
                else:
                    self.metrics["inactive_agents"] += 1
            self.metrics["total_agents"] = len(agents)
            logger.info(f"✅ Loaded {len(agents)} agents")
        except Exception as e:
            logger.debug(f"No existing agents to load: {e}")

    async def _load_versions(self):
        """Load agent versions from database"""
        try:
            versions = await self.db.find_many("agent_versions", {}, limit=5000)
            for version in versions:
                agent_id = version.get("agent_id")
                if agent_id not in self.agent_versions:
                    self.agent_versions[agent_id] = []
                self.agent_versions[agent_id].append(version)
            self.metrics["total_versions"] = len(versions)
            logger.info(f"✅ Loaded {len(versions)} agent versions")
        except Exception as e:
            logger.debug(f"No existing versions to load: {e}")

    async def _load_ab_tests(self):
        """Load A/B tests from database"""
        try:
            tests = await self.db.find_many("ab_tests", {}, limit=1000)
            for test in tests:
                self.ab_tests[test["test_id"]] = test
                if test.get("status") == "active":
                    self.metrics["active_ab_tests"] += 1
            logger.info(f"✅ Loaded {len(tests)} A/B tests")
        except Exception as e:
            logger.debug(f"No existing A/B tests to load: {e}")
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent"""
        try:
            agent_type = agent_config.get("type", "autonomous")
            agent_name = agent_config.get("name", f"{agent_type}_agent_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            
            logger.info(f"🤖 Creating agent: {agent_name} ({agent_type})")
            
            template = self.agent_templates.get(agent_type, self.agent_templates["autonomous"])
            
            agent_id = f"agent_{agent_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            agent = {
                "agent_id": agent_id,
                "name": agent_name,
                "type": agent_type,
                "config": agent_config,
                "capabilities": template["capabilities"],
                "priority": template["priority"],
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat(),
                "tasks_completed": 0,
                "success_rate": 1.0
            }
            
            await self.db.insert_one("managed_agents", agent)
            self.agents[agent_id] = agent
            
            self.metrics["total_agents"] += 1
            self.metrics["active_agents"] += 1
            
            await self.learning_service.record_pai_interaction({
                "user_id": "agent_management",
                "type": "agent_creation",
                "category": "agent_lifecycle",
                "strategic_reasoning": f"Created {agent_type} agent",
                "tactical_reasoning": f"Agent: {agent_name}",
                "operational_reasoning": "Agent created successfully",
                "response_quality": 1.0,
                "user_satisfaction": 1.0,
                "context_relevance": 1.0
            })
            
            logger.info(f"✅ Agent created: {agent_id}")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "agent": agent
            }
            
        except Exception as e:
            logger.error(f"❌ Agent creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing agent"""
        try:
            if agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}
            
            agent = self.agents[agent_id]
            
            # Update fields
            for key, value in updates.items():
                if key not in ["agent_id", "created_at"]:
                    agent[key] = value
            
            agent["updated_at"] = datetime.utcnow().isoformat()
            
            await self.db.update_one("managed_agents", {"agent_id": agent_id}, {"$set": agent})
            
            logger.info(f"✅ Agent updated: {agent_id}")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "agent": agent
            }
            
        except Exception as e:
            logger.error(f"❌ Agent update failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """Delete an agent"""
        try:
            if agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}
            
            agent = self.agents[agent_id]
            
            await self.db.delete_one("managed_agents", {"agent_id": agent_id})
            del self.agents[agent_id]
            
            self.metrics["total_agents"] -= 1
            if agent.get("status") == "active":
                self.metrics["active_agents"] -= 1
            else:
                self.metrics["inactive_agents"] -= 1
            
            logger.info(f"✅ Agent deleted: {agent_id}")
            
            return {
                "success": True,
                "message": f"Agent {agent_id} deleted"
            }
            
        except Exception as e:
            logger.error(f"❌ Agent deletion failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an agent"""
        try:
            if agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}
            
            agent = self.agents[agent_id]
            
            if agent.get("status") != "active":
                return {"success": False, "error": "Agent is not active"}
            
            task_id = f"task_{agent_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Simulate task execution
            task_result = {
                "task_id": task_id,
                "agent_id": agent_id,
                "task": task,
                "status": "completed",
                "result": {"message": "Task executed successfully"},
                "executed_at": datetime.utcnow().isoformat()
            }
            
            # Update agent stats
            agent["tasks_completed"] = agent.get("tasks_completed", 0) + 1
            agent["last_active"] = datetime.utcnow().isoformat()
            
            await self.db.update_one("managed_agents", {"agent_id": agent_id}, {"$set": agent})
            
            self.metrics["total_tasks_executed"] += 1
            self.metrics["successful_tasks"] += 1
            
            logger.info(f"✅ Task executed: {task_id}")
            
            return {
                "success": True,
                "task_id": task_id,
                "result": task_result
            }
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            self.metrics["failed_tasks"] += 1
            return {"success": False, "error": str(e)}
    
    async def list_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List all agents"""
        try:
            agents = list(self.agents.values())
            
            if filters:
                if "status" in filters:
                    agents = [a for a in agents if a.get("status") == filters["status"]]
                if "type" in filters:
                    agents = [a for a in agents if a.get("type") == filters["type"]]
            
            return agents
        except Exception as e:
            logger.error(f"❌ List agents failed: {e}")
            return []
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent"""
        return self.agents.get(agent_id)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get agent management metrics"""
        return {
            "success": True,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }

    # ==================== VERSIONING ====================

    async def create_version(self, agent_id: str, version_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new version of an agent"""
        try:
            if agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}

            agent = self.agents[agent_id]

            # Get current version number
            versions = self.agent_versions.get(agent_id, [])
            version_number = len(versions) + 1

            version_id = f"{agent_id}_v{version_number}"

            version = {
                "version_id": version_id,
                "agent_id": agent_id,
                "version_number": version_number,
                "config": version_config,
                "changes": version_config.get("changes", ""),
                "created_at": datetime.now().isoformat(),
                "created_by": version_config.get("created_by", "system"),
                "status": "active"
            }

            # Store in database
            try:
                await self.db.insert_one("agent_versions", version)
            except Exception as e:
                logger.warning(f"Could not save version to database: {e}")

            # Add to memory
            if agent_id not in self.agent_versions:
                self.agent_versions[agent_id] = []
            self.agent_versions[agent_id].append(version)

            self.metrics["total_versions"] += 1
            self._log_activity("version_created", f"Created version {version_number} for agent {agent_id}", {"version_id": version_id})

            logger.info(f"✅ Version created: {version_id}")

            return {
                "success": True,
                "version_id": version_id,
                "version": version
            }

        except Exception as e:
            logger.error(f"Version creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_versions(self, agent_id: str) -> Dict[str, Any]:
        """Get all versions of an agent"""
        try:
            versions = self.agent_versions.get(agent_id, [])
            return {
                "success": True,
                "agent_id": agent_id,
                "versions": versions,
                "count": len(versions)
            }
        except Exception as e:
            logger.error(f"Error getting versions: {e}")
            return {"success": False, "error": str(e)}

    async def rollback_version(self, agent_id: str, version_number: int) -> Dict[str, Any]:
        """Rollback agent to a specific version"""
        try:
            if agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}

            versions = self.agent_versions.get(agent_id, [])
            target_version = None

            for version in versions:
                if version.get("version_number") == version_number:
                    target_version = version
                    break

            if not target_version:
                return {"success": False, "error": f"Version {version_number} not found"}

            # Update agent with version config
            agent = self.agents[agent_id]
            agent["config"] = target_version["config"]
            agent["current_version"] = version_number
            agent["updated_at"] = datetime.now().isoformat()

            # Update in database
            try:
                await self.db.update_one("managed_agents", {"agent_id": agent_id}, {"$set": agent})
            except Exception as e:
                logger.warning(f"Could not update agent in database: {e}")

            self._log_activity("version_rollback", f"Rolled back agent {agent_id} to version {version_number}", {"agent_id": agent_id, "version": version_number})

            logger.info(f"✅ Agent rolled back to version {version_number}")

            return {
                "success": True,
                "agent_id": agent_id,
                "version": version_number,
                "agent": agent
            }

        except Exception as e:
            logger.error(f"Version rollback failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== PERFORMANCE MONITORING ====================

    async def record_performance(self, agent_id: str, response_time: float, success: bool) -> Dict[str, Any]:
        """Record performance metrics for an agent"""
        try:
            perf = self.performance_data[agent_id]

            # Record response time
            perf["response_times"].append(response_time)

            # Update counts
            if success:
                perf["success_count"] += 1
            else:
                perf["failure_count"] += 1

            # Calculate average response time
            if perf["response_times"]:
                perf["avg_response_time"] = sum(perf["response_times"]) / len(perf["response_times"])

            return {
                "success": True,
                "agent_id": agent_id,
                "performance": {
                    "avg_response_time": perf["avg_response_time"],
                    "success_rate": perf["success_count"] / (perf["success_count"] + perf["failure_count"]) if (perf["success_count"] + perf["failure_count"]) > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error recording performance: {e}")
            return {"success": False, "error": str(e)}

    async def get_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent"""
        try:
            if agent_id not in self.performance_data:
                return {"success": False, "error": "No performance data available"}

            perf = self.performance_data[agent_id]
            total_requests = perf["success_count"] + perf["failure_count"]

            return {
                "success": True,
                "agent_id": agent_id,
                "performance": {
                    "avg_response_time": perf["avg_response_time"],
                    "success_count": perf["success_count"],
                    "failure_count": perf["failure_count"],
                    "total_requests": total_requests,
                    "success_rate": perf["success_count"] / total_requests if total_requests > 0 else 0,
                    "recent_response_times": list(perf["response_times"])[-10:]
                }
            }

        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {"success": False, "error": str(e)}

    # ==================== A/B TESTING ====================

    async def create_ab_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an A/B test for agents"""
        try:
            test_id = f"ab_test_{uuid.uuid4().hex[:8]}"

            test = {
                "test_id": test_id,
                "name": test_config.get("name", f"Test {test_id}"),
                "variant_a": test_config.get("variant_a"),  # agent_id or config
                "variant_b": test_config.get("variant_b"),  # agent_id or config
                "metric": test_config.get("metric", "success_rate"),  # What to measure
                "traffic_split": test_config.get("traffic_split", 0.5),  # 50/50 by default
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "results": {
                    "variant_a": {"requests": 0, "successes": 0, "total_time": 0},
                    "variant_b": {"requests": 0, "successes": 0, "total_time": 0}
                }
            }

            # Store in database
            try:
                await self.db.insert_one("ab_tests", test)
            except Exception as e:
                logger.warning(f"Could not save A/B test to database: {e}")

            self.ab_tests[test_id] = test
            self.metrics["active_ab_tests"] += 1

            self._log_activity("ab_test_created", f"Created A/B test: {test['name']}", {"test_id": test_id})

            logger.info(f"✅ A/B test created: {test_id}")

            return {
                "success": True,
                "test_id": test_id,
                "test": test
            }

        except Exception as e:
            logger.error(f"A/B test creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def record_ab_result(self, test_id: str, variant: str, success: bool, response_time: float) -> Dict[str, Any]:
        """Record result for an A/B test"""
        try:
            if test_id not in self.ab_tests:
                return {"success": False, "error": "A/B test not found"}

            test = self.ab_tests[test_id]

            if variant not in ["variant_a", "variant_b"]:
                return {"success": False, "error": "Invalid variant"}

            # Update results
            results = test["results"][variant]
            results["requests"] += 1
            if success:
                results["successes"] += 1
            results["total_time"] += response_time

            # Update in database
            try:
                await self.db.update_one("ab_tests", {"test_id": test_id}, {"$set": {"results": test["results"]}})
            except Exception as e:
                logger.warning(f"Could not update A/B test in database: {e}")

            return {
                "success": True,
                "test_id": test_id,
                "variant": variant
            }

        except Exception as e:
            logger.error(f"Error recording A/B result: {e}")
            return {"success": False, "error": str(e)}

    async def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get results of an A/B test"""
        try:
            if test_id not in self.ab_tests:
                return {"success": False, "error": "A/B test not found"}

            test = self.ab_tests[test_id]
            results = test["results"]

            # Calculate metrics for each variant
            analysis = {}
            for variant in ["variant_a", "variant_b"]:
                data = results[variant]
                requests = data["requests"]

                if requests > 0:
                    analysis[variant] = {
                        "requests": requests,
                        "success_rate": data["successes"] / requests,
                        "avg_response_time": data["total_time"] / requests
                    }
                else:
                    analysis[variant] = {
                        "requests": 0,
                        "success_rate": 0,
                        "avg_response_time": 0
                    }

            # Determine winner
            winner = None
            if analysis["variant_a"]["success_rate"] > analysis["variant_b"]["success_rate"]:
                winner = "variant_a"
            elif analysis["variant_b"]["success_rate"] > analysis["variant_a"]["success_rate"]:
                winner = "variant_b"

            return {
                "success": True,
                "test_id": test_id,
                "test": test,
                "analysis": analysis,
                "winner": winner
            }

        except Exception as e:
            logger.error(f"Error getting A/B test results: {e}")
            return {"success": False, "error": str(e)}

    async def stop_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Stop an A/B test"""
        try:
            if test_id not in self.ab_tests:
                return {"success": False, "error": "A/B test not found"}

            test = self.ab_tests[test_id]
            test["status"] = "stopped"
            test["stopped_at"] = datetime.now().isoformat()

            # Update in database
            try:
                await self.db.update_one("ab_tests", {"test_id": test_id}, {"$set": {"status": "stopped", "stopped_at": test["stopped_at"]}})
            except Exception as e:
                logger.warning(f"Could not update A/B test in database: {e}")

            self.metrics["active_ab_tests"] -= 1
            self._log_activity("ab_test_stopped", f"Stopped A/B test: {test['name']}", {"test_id": test_id})

            logger.info(f"✅ A/B test stopped: {test_id}")

            return {
                "success": True,
                "test_id": test_id,
                "test": test
            }

        except Exception as e:
            logger.error(f"Error stopping A/B test: {e}")
            return {"success": False, "error": str(e)}

    async def get_all_ab_tests(self) -> Dict[str, Any]:
        """Get all A/B tests"""
        try:
            tests = list(self.ab_tests.values())
            return {
                "success": True,
                "tests": tests,
                "count": len(tests)
            }
        except Exception as e:
            logger.error(f"Error getting A/B tests: {e}")
            return {"success": False, "error": str(e)}

    # ==================== ACTIVITY LOG ====================

    async def get_activity_log(self, limit: int = 100) -> Dict[str, Any]:
        """Get activity log"""
        try:
            activities = list(self.activity_log)[-limit:]
            return {
                "success": True,
                "activities": activities,
                "count": len(activities)
            }
        except Exception as e:
            logger.error(f"Error getting activity log: {e}")
            return {"success": False, "error": str(e)}

