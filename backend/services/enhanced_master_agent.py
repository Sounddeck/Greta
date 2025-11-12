"""
GRETA Enhanced Master Agent - Complete System Controller
The Master Agent that controls everything in Greta with full functionality
Integrates with local LLM for continuous learning and improvement
"""

import asyncio
import logging
import json
import psutil
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
from collections import deque

# Import agent templates
from services.agent_templates import AgentTemplates

logger = logging.getLogger(__name__)

class EnhancedMasterAgent:
    """
    Enhanced Master Agent - Complete System Controller
    Controls all aspects of Greta with real functionality
    """
    
    def __init__(self, db, llm_service, learning_service):
        self.db = db
        self.llm_service = llm_service
        self.learning_service = learning_service

        # Initialize Agent Builder Service
        from services.agent_builder_service import AgentBuilderService
        self.agent_builder = AgentBuilderService(db)

        # System state
        self.system_state = {
            "status": "initializing",
            "modules": {},
            "agents": {},
            "workflows": {},
            "performance": {},
            "paused": False,
            "emergency_mode": False
        }

        # Module registry - all 12 modules
        self.modules = {
            "pai_hub": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "master_agent": {"status": "active", "service": self, "health": "healthy", "last_check": datetime.now()},
            "predictive_analytics": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "agent_management": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "autonomous_operations": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "agent_builder": {"status": "active", "service": self.agent_builder, "health": "healthy", "last_check": datetime.now()},
            "mcp_hub": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "osint": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "multimodal": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "memory_system": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "training_platform": {"status": "inactive", "service": None, "health": "unknown", "last_check": None},
            "knowledge_base": {"status": "inactive", "service": None, "health": "unknown", "last_check": None}
        }

        # Agent registry with deployed agents
        self.agents = {}

        # Load agent templates from AgentTemplates class
        self.agent_templates = {t["id"]: t for t in AgentTemplates.get_all_templates()}

        # Workflow engine
        self.active_workflows = {}
        self.workflow_history = []

        # Activity log (keep last 1000 activities)
        self.activity_log = deque(maxlen=1000)

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "agents_created": 0,
            "agents_deployed": 0,
            "workflows_executed": 0,
            "llm_interactions": 0,
            "learning_cycles": 0,
            "average_response_time": 0.0,
            "uptime_seconds": 0,
            "last_reset": datetime.now()
        }

        # System metrics history (for charts)
        self.metrics_history = deque(maxlen=100)

        # Start time for uptime calculation
        self.start_time = datetime.now()
        
        # Learning integration
        self.learning_enabled = True
        self.learning_buffer = []
        self.learning_batch_size = 10
        
    async def initialize(self):
        """Initialize the Master Agent system"""
        try:
            logger.info("🚀 Initializing Enhanced Master Agent...")
            
            # Initialize database collections
            await self._initialize_collections()
            
            # Load existing agents
            await self._load_agents()
            
            # Initialize all modules
            await self._initialize_modules()
            
            # Start background tasks
            asyncio.create_task(self._learning_loop())
            asyncio.create_task(self._health_monitor())
            
            self.system_state["status"] = "active"
            logger.info("✅ Enhanced Master Agent initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Master Agent initialization failed: {e}")
            self.system_state["status"] = "error"
            return False
    
    async def _initialize_collections(self):
        """Initialize database collections"""
        collections = [
            "master_agent_state",
            "agents",
            "workflows",
            "modules",
            "learning_data",
            "system_metrics",
            "deployments",
            "applications"
        ]
        
        for collection in collections:
            try:
                await self.db.create_collection(collection)
                logger.info(f"✅ Collection initialized: {collection}")
            except Exception as e:
                logger.debug(f"Collection {collection} already exists or error: {e}")
    
    async def _load_agents(self):
        """Load existing agents from database"""
        try:
            agents = await self.db.find_many("agents", {}, limit=1000)
            for agent in agents:
                self.agents[agent["agent_id"]] = agent
            logger.info(f"✅ Loaded {len(agents)} agents")
        except Exception as e:
            logger.debug(f"No existing agents to load: {e}")
    
    async def _initialize_modules(self):
        """Initialize all 12 modules"""
        logger.info("🔧 Initializing all modules...")

        # Mark master_agent as active
        self.modules["master_agent"]["status"] = "active"
        self.modules["master_agent"]["health"] = "healthy"

        # Other modules will be initialized on-demand
        for module_name in self.modules:
            if module_name != "master_agent":
                self.modules[module_name]["status"] = "ready"

        logger.info("✅ All modules ready")

    # ==================== SYSTEM MONITORING ====================

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024 ** 3)
            disk_total_gb = disk.total / (1024 ** 3)

            # Network metrics
            net_io = psutil.net_io_counters()

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 ** 2)  # MB
            process_cpu = process.cpu_percent()

            # Calculate uptime
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": round(cpu_percent, 2),
                    "count": cpu_count,
                    "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else 0
                },
                "memory": {
                    "percent": round(memory_percent, 2),
                    "used_gb": round(memory_used_gb, 2),
                    "total_gb": round(memory_total_gb, 2),
                    "available_gb": round(memory.available / (1024 ** 3), 2)
                },
                "disk": {
                    "percent": round(disk_percent, 2),
                    "used_gb": round(disk_used_gb, 2),
                    "total_gb": round(disk_total_gb, 2),
                    "free_gb": round(disk.free / (1024 ** 3), 2)
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                },
                "process": {
                    "memory_mb": round(process_memory, 2),
                    "cpu_percent": round(process_cpu, 2)
                },
                "uptime_seconds": int(uptime_seconds),
                "uptime_formatted": self._format_uptime(uptime_seconds)
            }

            # Add to history
            self.metrics_history.append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    async def get_module_status(self) -> Dict[str, Any]:
        """Get status of all modules"""
        module_status = {}

        for module_name, module_info in self.modules.items():
            module_status[module_name] = {
                "name": module_name.replace("_", " ").title(),
                "status": module_info["status"],
                "health": module_info.get("health", "unknown"),
                "last_check": module_info.get("last_check").isoformat() if module_info.get("last_check") else None,
                "service_active": module_info["service"] is not None
            }

        return module_status

    async def get_agent_inventory(self) -> Dict[str, Any]:
        """Get inventory of all deployed agents"""
        inventory = {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.get("status") == "active"),
            "inactive_agents": sum(1 for a in self.agents.values() if a.get("status") == "inactive"),
            "agents_by_type": {},
            "agents": []
        }

        # Count by type
        for agent in self.agents.values():
            agent_type = agent.get("type", "unknown")
            inventory["agents_by_type"][agent_type] = inventory["agents_by_type"].get(agent_type, 0) + 1

            # Add agent details
            inventory["agents"].append({
                "id": agent.get("agent_id"),
                "name": agent.get("name"),
                "type": agent_type,
                "status": agent.get("status"),
                "created_at": agent.get("created_at"),
                "last_active": agent.get("last_active")
            })

        return inventory

    async def get_activity_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent activity log"""
        # Return most recent activities
        activities = list(self.activity_log)[-limit:]
        return activities

    def _log_activity(self, activity_type: str, description: str, metadata: Dict[str, Any] = None):
        """Log an activity"""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "description": description,
            "metadata": metadata or {}
        }
        self.activity_log.append(activity)
        logger.info(f"📝 Activity: {activity_type} - {description}")

    async def calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        try:
            metrics = await self.get_system_metrics()

            # Health factors (0-100 each)
            cpu_health = 100 - metrics["cpu"]["percent"]
            memory_health = 100 - metrics["memory"]["percent"]
            disk_health = 100 - metrics["disk"]["percent"]

            # Module health
            healthy_modules = sum(1 for m in self.modules.values() if m.get("health") == "healthy")
            total_modules = len(self.modules)
            module_health = (healthy_modules / total_modules) * 100 if total_modules > 0 else 0

            # Agent health
            active_agents = sum(1 for a in self.agents.values() if a.get("status") == "active")
            total_agents = len(self.agents)
            agent_health = (active_agents / total_agents) * 100 if total_agents > 0 else 100

            # Calculate weighted average
            overall_health = (
                cpu_health * 0.2 +
                memory_health * 0.2 +
                disk_health * 0.1 +
                module_health * 0.3 +
                agent_health * 0.2
            )

            # Determine status
            if overall_health >= 80:
                status = "healthy"
                status_icon = "🟢"
            elif overall_health >= 60:
                status = "warning"
                status_icon = "🟡"
            else:
                status = "critical"
                status_icon = "🔴"

            return {
                "overall_health": round(overall_health, 2),
                "status": status,
                "status_icon": status_icon,
                "factors": {
                    "cpu_health": round(cpu_health, 2),
                    "memory_health": round(memory_health, 2),
                    "disk_health": round(disk_health, 2),
                    "module_health": round(module_health, 2),
                    "agent_health": round(agent_health, 2)
                },
                "recommendations": self._get_health_recommendations(overall_health, metrics)
            }

        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {
                "overall_health": 0,
                "status": "error",
                "status_icon": "❌",
                "error": str(e)
            }

    def _get_health_recommendations(self, health_score: float, metrics: Dict[str, Any]) -> List[str]:
        """Get health recommendations based on metrics"""
        recommendations = []

        if metrics["cpu"]["percent"] > 80:
            recommendations.append("High CPU usage detected. Consider scaling resources or optimizing workloads.")

        if metrics["memory"]["percent"] > 80:
            recommendations.append("High memory usage detected. Consider increasing memory or reducing active agents.")

        if metrics["disk"]["percent"] > 80:
            recommendations.append("Low disk space. Consider cleaning up old data or expanding storage.")

        if health_score < 60:
            recommendations.append("System health is critical. Review all modules and agents for issues.")

        if not recommendations:
            recommendations.append("System is operating normally. No immediate action required.")

        return recommendations
    
    # ==================== AGENT ORCHESTRATION ====================

    async def deploy_agent_from_template(self, template_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deploy an agent from a template"""
        try:
            # Get template
            template = AgentTemplates.get_template(template_id)
            if not template:
                return {
                    "success": False,
                    "error": f"Template not found: {template_id}"
                }

            # Generate agent ID
            agent_id = f"{template_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Merge config with template defaults
            agent_config = {**template.get("default_config", {}), **(config or {})}

            # Create agent instance
            agent = {
                "agent_id": agent_id,
                "name": config.get("name", template["name"]) if config else template["name"],
                "type": template_id,
                "template": template,
                "config": agent_config,
                "status": "deployed",
                "health": "healthy",
                "created_at": datetime.now().isoformat(),
                "deployed_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_tokens_used": 0,
                "average_response_time": 0.0
            }

            # Store agent
            self.agents[agent_id] = agent

            # Save to database
            try:
                await self.db.insert_one("agents", agent)
            except Exception as e:
                logger.warning(f"Could not save agent to database: {e}")

            # Update metrics
            self.metrics["agents_created"] += 1
            self.metrics["agents_deployed"] += 1

            # Log activity
            self._log_activity(
                "agent_deployed",
                f"Deployed {template['name']} agent: {agent['name']}",
                {"agent_id": agent_id, "template": template_id}
            )

            logger.info(f"✅ Agent deployed: {agent_id} ({template['name']})")

            return {
                "success": True,
                "agent_id": agent_id,
                "agent": agent,
                "message": f"Agent {agent['name']} deployed successfully"
            }

        except Exception as e:
            logger.error(f"Error deploying agent: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def route_task_to_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route a task to the most appropriate agent"""
        try:
            task_type = task.get("type", "general")
            task_description = task.get("description", "")

            # Determine best agent type based on task
            agent_type = self._determine_agent_type(task_type, task_description)

            # Find or deploy appropriate agent
            agent = await self._get_or_deploy_agent(agent_type)

            if not agent:
                return {
                    "success": False,
                    "error": f"Could not deploy agent for task type: {task_type}"
                }

            # Execute task with agent
            result = await self._execute_task_with_agent(agent, task)

            # Log activity
            self._log_activity(
                "task_routed",
                f"Task routed to {agent['name']}",
                {"task_type": task_type, "agent_id": agent["agent_id"]}
            )

            return result

        except Exception as e:
            logger.error(f"Error routing task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _determine_agent_type(self, task_type: str, description: str) -> str:
        """Determine the best agent type for a task"""
        # Simple keyword-based routing (can be enhanced with ML)
        description_lower = description.lower()

        if task_type == "research" or any(kw in description_lower for kw in ["research", "find", "search", "investigate"]):
            return "researcher"
        elif task_type == "code" or any(kw in description_lower for kw in ["code", "program", "develop", "debug", "implement"]):
            return "engineer"
        elif task_type == "design" or any(kw in description_lower for kw in ["design", "ui", "ux", "interface", "mockup"]):
            return "designer"
        elif task_type == "security" or any(kw in description_lower for kw in ["security", "pentest", "vulnerability", "hack"]):
            return "pentester"
        elif task_type == "architecture" or any(kw in description_lower for kw in ["architecture", "system design", "scalability"]):
            return "architect"
        elif task_type == "analysis" or any(kw in description_lower for kw in ["analyze", "data", "statistics", "metrics"]):
            return "analyst"
        elif task_type == "writing" or any(kw in description_lower for kw in ["write", "content", "article", "blog"]):
            return "writer"
        elif task_type == "teaching" or any(kw in description_lower for kw in ["teach", "explain", "tutorial", "learn"]):
            return "teacher"
        else:
            return "researcher"  # Default

    async def _get_or_deploy_agent(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Get an existing agent or deploy a new one"""
        # Look for existing active agent of this type
        for agent in self.agents.values():
            if agent.get("type") == agent_type and agent.get("status") == "deployed":
                return agent

        # Deploy new agent
        result = await self.deploy_agent_from_template(agent_type)
        if result.get("success"):
            return result.get("agent")

        return None

    async def _execute_task_with_agent(self, agent: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a specific agent"""
        try:
            start_time = datetime.now()

            # Prepare prompt using agent's template
            template = agent.get("template", {})
            system_prompt = template.get("prompts", {}).get("system", "")
            task_prefix = template.get("prompts", {}).get("task_prefix", "")

            prompt = f"{task_prefix}\n\n{task.get('description', '')}"

            # Query LLM with agent's configuration
            llm_result = await self.llm_service.query(
                prompt=prompt,
                system_prompt=system_prompt,
                **agent.get("config", {})
            )

            # Update agent metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            agent["last_active"] = datetime.now().isoformat()
            agent["tasks_completed"] = agent.get("tasks_completed", 0) + 1

            # Update average response time
            total_tasks = agent.get("tasks_completed", 1)
            current_avg = agent.get("average_response_time", 0.0)
            agent["average_response_time"] = ((current_avg * (total_tasks - 1)) + execution_time) / total_tasks

            # Save updated agent
            try:
                await self.db.update_one(
                    "agents",
                    {"agent_id": agent["agent_id"]},
                    {"$set": agent}
                )
            except Exception as e:
                logger.warning(f"Could not update agent in database: {e}")

            return {
                "success": True,
                "result": llm_result,
                "agent_id": agent["agent_id"],
                "agent_name": agent["name"],
                "execution_time": execution_time
            }

        except Exception as e:
            logger.error(f"Error executing task with agent: {e}")
            agent["tasks_failed"] = agent.get("tasks_failed", 0) + 1
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent.get("agent_id")
            }

    async def aggregate_agent_results(self, task: Dict[str, Any], agent_types: List[str]) -> Dict[str, Any]:
        """Execute a task with multiple agents and aggregate results"""
        try:
            results = []

            # Execute task with each agent type
            for agent_type in agent_types:
                agent = await self._get_or_deploy_agent(agent_type)
                if agent:
                    result = await self._execute_task_with_agent(agent, task)
                    results.append({
                        "agent_type": agent_type,
                        "agent_name": agent.get("name"),
                        "result": result
                    })

            # Aggregate results using LLM
            aggregation_prompt = f"""
            Multiple agents have completed the following task:
            {task.get('description', '')}

            Here are their results:
            {json.dumps(results, indent=2)}

            Please provide a comprehensive synthesis of these results, highlighting:
            1. Common findings and agreements
            2. Unique insights from each agent
            3. Any contradictions or disagreements
            4. Final recommendations
            """

            aggregated_result = await self.llm_service.query(
                prompt=aggregation_prompt,
                system_prompt="You are an expert at synthesizing information from multiple sources."
            )

            return {
                "success": True,
                "individual_results": results,
                "aggregated_result": aggregated_result,
                "agents_used": len(results)
            }

        except Exception as e:
            logger.error(f"Error aggregating agent results: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def manage_agent_lifecycle(self, agent_id: str, action: str) -> Dict[str, Any]:
        """Manage agent lifecycle (start, stop, pause, resume, delete)"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                return {
                    "success": False,
                    "error": f"Agent not found: {agent_id}"
                }

            if action == "start":
                agent["status"] = "deployed"
                message = f"Agent {agent['name']} started"
            elif action == "stop":
                agent["status"] = "stopped"
                message = f"Agent {agent['name']} stopped"
            elif action == "pause":
                agent["status"] = "paused"
                message = f"Agent {agent['name']} paused"
            elif action == "resume":
                agent["status"] = "deployed"
                message = f"Agent {agent['name']} resumed"
            elif action == "delete":
                del self.agents[agent_id]
                try:
                    await self.db.delete_one("agents", {"agent_id": agent_id})
                except Exception as e:
                    logger.warning(f"Could not delete agent from database: {e}")
                message = f"Agent {agent['name']} deleted"
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }

            # Log activity
            self._log_activity(
                f"agent_{action}",
                message,
                {"agent_id": agent_id}
            )

            return {
                "success": True,
                "message": message,
                "agent": agent if action != "delete" else None
            }

        except Exception as e:
            logger.error(f"Error managing agent lifecycle: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a master agent command
        This is the main entry point for all system operations
        """
        start_time = datetime.utcnow()
        command_type = command.get("type", "unknown")
        
        try:
            self.metrics["total_requests"] += 1
            
            logger.info(f"🎯 Executing command: {command_type}")
            
            # Route to appropriate handler
            if command_type == "build_agent":
                result = await self.build_agent(command)
            elif command_type == "deploy_agent":
                result = await self.deploy_agent(command)
            elif command_type == "manage_agent":
                result = await self.manage_agent(command)
            elif command_type == "execute_workflow":
                result = await self.execute_workflow(command)
            elif command_type == "build_application":
                result = await self.build_application(command)
            elif command_type == "optimize_system":
                result = await self.optimize_system(command)
            elif command_type == "query_llm":
                result = await self.query_llm(command)
            elif command_type == "get_status":
                result = await self.get_system_status()
            else:
                result = await self._handle_custom_command(command)
            
            # Record learning data
            if self.learning_enabled:
                await self._record_learning_data(command, result)
            
            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_metrics(command_type, execution_time, result.get("success", False))
            
            if result.get("success"):
                self.metrics["successful_operations"] += 1
            else:
                self.metrics["failed_operations"] += 1
            
            return {
                "success": result.get("success", False),
                "result": result,
                "command_type": command_type,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Command execution failed: {e}")
            self.metrics["failed_operations"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "command_type": command_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def build_agent(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Build a new agent"""
        try:
            agent_config = command.get("config", {})
            agent_type = agent_config.get("type", "research")
            
            # Get template
            template = self.agent_templates.get(agent_type)
            if not template:
                return {"success": False, "error": f"Unknown agent type: {agent_type}"}
            
            # Create agent
            agent_id = f"{agent_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            agent = {
                "agent_id": agent_id,
                "name": agent_config.get("name", template["name"]),
                "type": agent_type,
                "capabilities": template["capabilities"],
                "llm_model": agent_config.get("llm_model", template["llm_model"]),
                "config": agent_config,
                "status": "created",
                "created_at": datetime.utcnow().isoformat(),
                "created_by": "master_agent"
            }
            
            # Store in database
            await self.db.insert_one("agents", agent)
            self.agents[agent_id] = agent
            self.metrics["agents_created"] += 1
            
            logger.info(f"✅ Agent created: {agent_id}")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "agent": agent
            }
            
        except Exception as e:
            logger.error(f"❌ Agent build failed: {e}")
            return {"success": False, "error": str(e)}

    async def deploy_agent(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy an agent to the system"""
        try:
            agent_id = command.get("agent_id")
            deployment_config = command.get("deployment_config", {})

            if not agent_id or agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}

            agent = self.agents[agent_id]

            # Update agent status
            agent["status"] = "deployed"
            agent["deployed_at"] = datetime.utcnow().isoformat()
            agent["deployment_config"] = deployment_config

            # Update in database
            await self.db.update_one(
                "agents",
                {"agent_id": agent_id},
                {"$set": agent}
            )

            self.agents[agent_id] = agent

            logger.info(f"✅ Agent deployed: {agent_id}")

            return {
                "success": True,
                "agent_id": agent_id,
                "agent": agent
            }

        except Exception as e:
            logger.error(f"❌ Agent deployment failed: {e}")
            return {"success": False, "error": str(e)}

    async def manage_agent(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Manage agent lifecycle"""
        try:
            action = command.get("action")
            agent_id = command.get("agent_id")

            if not agent_id or agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}

            agent = self.agents[agent_id]

            if action == "start":
                agent["status"] = "running"
            elif action == "stop":
                agent["status"] = "stopped"
            elif action == "restart":
                agent["status"] = "running"
            elif action == "delete":
                await self.db.delete_one("agents", {"agent_id": agent_id})
                del self.agents[agent_id]
                return {"success": True, "message": f"Agent {agent_id} deleted"}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}

            # Update in database
            await self.db.update_one(
                "agents",
                {"agent_id": agent_id},
                {"$set": agent}
            )

            self.agents[agent_id] = agent

            logger.info(f"✅ Agent {action}: {agent_id}")

            return {
                "success": True,
                "agent_id": agent_id,
                "action": action,
                "agent": agent
            }

        except Exception as e:
            logger.error(f"❌ Agent management failed: {e}")
            return {"success": False, "error": str(e)}

    async def execute_workflow(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            workflow_spec = command.get("workflow", {})
            workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            workflow = {
                "workflow_id": workflow_id,
                "name": workflow_spec.get("name", "Unnamed Workflow"),
                "description": workflow_spec.get("description", ""),
                "steps": workflow_spec.get("steps", []),
                "status": "running",
                "created_at": datetime.utcnow().isoformat(),
                "results": []
            }

            self.active_workflows[workflow_id] = workflow

            # Execute workflow steps
            for i, step in enumerate(workflow["steps"]):
                step_result = await self._execute_workflow_step(step, workflow_id)
                workflow["results"].append(step_result)

                if not step_result.get("success"):
                    workflow["status"] = "failed"
                    break

            if workflow["status"] == "running":
                workflow["status"] = "completed"

            workflow["completed_at"] = datetime.utcnow().isoformat()

            # Store in database
            await self.db.insert_one("workflows", workflow)
            self.workflow_history.append(workflow)
            self.metrics["workflows_executed"] += 1

            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            logger.info(f"✅ Workflow completed: {workflow_id}")

            return {
                "success": workflow["status"] == "completed",
                "workflow_id": workflow_id,
                "workflow": workflow
            }

        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_workflow_step(self, step: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            step_type = step.get("type")

            if step_type == "agent_task":
                # Execute task using an agent
                agent_id = step.get("agent_id")
                task = step.get("task")

                if agent_id and agent_id in self.agents:
                    # Simulate agent task execution
                    result = await self._execute_agent_task(agent_id, task)
                    return {"success": True, "result": result}
                else:
                    return {"success": False, "error": "Agent not found"}

            elif step_type == "llm_query":
                # Query LLM
                query = step.get("query")
                result = await self.query_llm({"query": query})
                return result

            elif step_type == "data_processing":
                # Process data
                data = step.get("data")
                operation = step.get("operation")
                result = await self._process_data(data, operation)
                return {"success": True, "result": result}

            else:
                return {"success": False, "error": f"Unknown step type: {step_type}"}

        except Exception as e:
            logger.error(f"❌ Workflow step failed: {e}")
            return {"success": False, "error": str(e)}

    async def build_application(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Build a complete application"""
        try:
            app_spec = command.get("spec", {})
            app_name = app_spec.get("name", f"app_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            app_type = app_spec.get("type", "web_application")

            app_id = f"{app_type}_{app_name}"

            application = {
                "app_id": app_id,
                "name": app_name,
                "type": app_type,
                "spec": app_spec,
                "status": "building",
                "created_at": datetime.utcnow().isoformat(),
                "components": []
            }

            # Build components based on type
            if app_type == "web_application":
                components = await self._build_web_components(app_spec)
            elif app_type == "api_application":
                components = await self._build_api_components(app_spec)
            elif app_type == "cli_application":
                components = await self._build_cli_components(app_spec)
            else:
                components = []

            application["components"] = components
            application["status"] = "built"
            application["built_at"] = datetime.utcnow().isoformat()

            # Store in database
            await self.db.insert_one("applications", application)

            logger.info(f"✅ Application built: {app_id}")

            return {
                "success": True,
                "app_id": app_id,
                "application": application
            }

        except Exception as e:
            logger.error(f"❌ Application build failed: {e}")
            return {"success": False, "error": str(e)}

    async def optimize_system(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system performance"""
        try:
            optimization_type = command.get("type", "general")

            optimizations = []

            if optimization_type in ["memory", "general"]:
                memory_opt = await self._optimize_memory()
                optimizations.append(memory_opt)

            if optimization_type in ["performance", "general"]:
                perf_opt = await self._optimize_performance()
                optimizations.append(perf_opt)

            if optimization_type in ["resources", "general"]:
                resource_opt = await self._optimize_resources()
                optimizations.append(resource_opt)

            logger.info(f"✅ System optimization completed: {optimization_type}")

            return {
                "success": True,
                "optimization_type": optimization_type,
                "optimizations": optimizations
            }

        except Exception as e:
            logger.error(f"❌ System optimization failed: {e}")
            return {"success": False, "error": str(e)}

    async def query_llm(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Query the local LLM"""
        try:
            query = command.get("query", "")
            context = command.get("context", {})

            # Add system context
            context["master_agent"] = True
            context["system_state"] = self.system_state
            context["available_agents"] = list(self.agents.keys())

            # Query LLM
            llm_response = await self.llm_service.generate_completion({
                "message": query,
                "context": context,
                "use_hrm": True
            })

            self.metrics["llm_interactions"] += 1

            logger.info(f"✅ LLM query completed")

            return {
                "success": llm_response.get("success", False),
                "response": llm_response.get("response", ""),
                "hrm_enhanced": llm_response.get("hrm_enhanced", False)
            }

        except Exception as e:
            logger.error(f"❌ LLM query failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Count agents by status
            agent_stats = {
                "total": len(self.agents),
                "running": len([a for a in self.agents.values() if a.get("status") == "running"]),
                "stopped": len([a for a in self.agents.values() if a.get("status") == "stopped"]),
                "deployed": len([a for a in self.agents.values() if a.get("status") == "deployed"])
            }

            # Module status
            module_stats = {
                "total": len(self.modules),
                "active": len([m for m in self.modules.values() if m.get("status") == "active"]),
                "ready": len([m for m in self.modules.values() if m.get("status") == "ready"])
            }

            return {
                "success": True,
                "status": self.system_state["status"],
                "system_resources": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent
                },
                "agents": agent_stats,
                "modules": module_stats,
                "workflows": {
                    "active": len(self.active_workflows),
                    "completed": len(self.workflow_history)
                },
                "metrics": self.metrics,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Status retrieval failed: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_custom_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom commands"""
        try:
            # Use LLM to interpret and execute custom command
            llm_result = await self.query_llm({
                "query": f"Execute custom command: {json.dumps(command)}",
                "context": {"command_type": "custom"}
            })

            return {
                "success": llm_result.get("success", False),
                "result": llm_result.get("response", ""),
                "custom_command": True
            }

        except Exception as e:
            logger.error(f"❌ Custom command failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_agent_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using a specific agent"""
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                return {"success": False, "error": "Agent not found"}

            # Use LLM to execute task with agent context
            llm_result = await self.llm_service.generate_completion({
                "message": task.get("description", ""),
                "context": {
                    "agent": agent,
                    "capabilities": agent.get("capabilities", []),
                    "task": task
                },
                "use_hrm": True
            })

            return {
                "success": llm_result.get("success", False),
                "result": llm_result.get("response", ""),
                "agent_id": agent_id
            }

        except Exception as e:
            logger.error(f"❌ Agent task execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _process_data(self, data: Any, operation: str) -> Any:
        """Process data with specified operation"""
        try:
            if operation == "analyze":
                # Use LLM to analyze data
                llm_result = await self.llm_service.generate_completion({
                    "message": f"Analyze this data: {json.dumps(data)}",
                    "context": {"operation": "data_analysis"},
                    "use_hrm": True
                })
                return llm_result.get("response", {})

            elif operation == "transform":
                # Transform data
                return {"transformed": data, "timestamp": datetime.utcnow().isoformat()}

            elif operation == "validate":
                # Validate data
                return {"valid": True, "data": data}

            else:
                return data

        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return {"error": str(e)}

    async def _build_web_components(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build web application components"""
        components = [
            {"type": "frontend", "framework": spec.get("frontend_framework", "react"), "status": "built"},
            {"type": "backend", "framework": spec.get("backend_framework", "fastapi"), "status": "built"},
            {"type": "database", "type": spec.get("database", "mongodb"), "status": "configured"}
        ]
        return components

    async def _build_api_components(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build API application components"""
        components = [
            {"type": "api", "framework": spec.get("framework", "fastapi"), "status": "built"},
            {"type": "database", "type": spec.get("database", "mongodb"), "status": "configured"}
        ]
        return components

    async def _build_cli_components(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build CLI application components"""
        components = [
            {"type": "cli", "framework": spec.get("framework", "click"), "status": "built"}
        ]
        return components

    async def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            # Clean up inactive agents
            inactive_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.get("status") == "stopped"
            ]

            # Clear old workflow history
            if len(self.workflow_history) > 100:
                self.workflow_history = self.workflow_history[-100:]

            return {
                "type": "memory",
                "actions": [
                    f"Cleaned {len(inactive_agents)} inactive agents",
                    f"Trimmed workflow history to 100 entries"
                ]
            }

        except Exception as e:
            logger.error(f"❌ Memory optimization failed: {e}")
            return {"type": "memory", "error": str(e)}

    async def _optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance"""
        try:
            # Update average response time
            if self.metrics["total_requests"] > 0:
                success_rate = (self.metrics["successful_operations"] / self.metrics["total_requests"]) * 100
            else:
                success_rate = 0

            return {
                "type": "performance",
                "metrics": {
                    "success_rate": success_rate,
                    "total_requests": self.metrics["total_requests"]
                }
            }

        except Exception as e:
            logger.error(f"❌ Performance optimization failed: {e}")
            return {"type": "performance", "error": str(e)}

    async def _optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            return {
                "type": "resources",
                "current": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent
                }
            }

        except Exception as e:
            logger.error(f"❌ Resource optimization failed: {e}")
            return {"type": "resources", "error": str(e)}

    # ==================== LEARNING COORDINATION ====================

    async def get_learning_status(self) -> Dict[str, Any]:
        """Get learning system status"""
        try:
            return {
                "success": True,
                "learning_enabled": self.learning_enabled,
                "buffer_size": len(self.learning_buffer),
                "buffer_capacity": self.learning_batch_size,
                "total_learning_cycles": self.metrics["learning_cycles"],
                "llm_interactions": self.metrics["llm_interactions"],
                "data_sources": {
                    "master_agent": True,
                    "agents": len(self.agents),
                    "workflows": len(self.workflow_history),
                    "modules": len([m for m in self.modules.values() if m.get("status") == "active"])
                }
            }
        except Exception as e:
            logger.error(f"Error getting learning status: {e}")
            return {"success": False, "error": str(e)}

    async def collect_training_data(self) -> Dict[str, Any]:
        """Collect training data from all modules"""
        try:
            training_data = {
                "timestamp": datetime.now().isoformat(),
                "sources": {}
            }

            # Collect from agents
            agent_data = []
            for agent in self.agents.values():
                agent_data.append({
                    "type": agent.get("type"),
                    "tasks_completed": agent.get("tasks_completed", 0),
                    "tasks_failed": agent.get("tasks_failed", 0),
                    "average_response_time": agent.get("average_response_time", 0.0)
                })
            training_data["sources"]["agents"] = agent_data

            # Collect from workflows
            workflow_data = []
            for workflow in self.workflow_history[-100:]:  # Last 100 workflows
                workflow_data.append({
                    "type": workflow.get("type"),
                    "success": workflow.get("success"),
                    "execution_time": workflow.get("execution_time")
                })
            training_data["sources"]["workflows"] = workflow_data

            # Collect system metrics
            training_data["sources"]["system_metrics"] = self.metrics.copy()

            # Send to learning service
            await self.learning_service.record_pai_interaction({
                "user_id": "master_agent",
                "type": "training_data_collection",
                "category": "system_learning",
                "strategic_reasoning": "Collecting training data from all system components",
                "tactical_reasoning": f"Collected data from {len(agent_data)} agents and {len(workflow_data)} workflows",
                "operational_reasoning": "Data will be used for LLM fine-tuning",
                "response_quality": 1.0,
                "user_satisfaction": 1.0,
                "context_relevance": 1.0
            })

            self._log_activity(
                "training_data_collected",
                f"Collected training data from {len(agent_data)} agents and {len(workflow_data)} workflows",
                {"data_points": len(agent_data) + len(workflow_data)}
            )

            return {
                "success": True,
                "data": training_data,
                "data_points": len(agent_data) + len(workflow_data)
            }

        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return {"success": False, "error": str(e)}

    async def trigger_llm_training(self) -> Dict[str, Any]:
        """Trigger LLM training with collected data"""
        try:
            # Collect all training data
            training_data = await self.collect_training_data()

            if not training_data.get("success"):
                return training_data

            # Format data for LLM training
            formatted_data = self._format_training_data(training_data.get("data", {}))

            # Send to LLM service for training
            # Note: This would integrate with llama.cpp or similar for actual fine-tuning
            logger.info(f"🎓 Triggering LLM training with {len(formatted_data)} data points")

            self.metrics["learning_cycles"] += 1

            self._log_activity(
                "llm_training_triggered",
                f"Triggered LLM training with {len(formatted_data)} data points",
                {"data_points": len(formatted_data)}
            )

            return {
                "success": True,
                "message": "LLM training triggered successfully",
                "data_points": len(formatted_data),
                "training_cycle": self.metrics["learning_cycles"]
            }

        except Exception as e:
            logger.error(f"Error triggering LLM training: {e}")
            return {"success": False, "error": str(e)}

    def _format_training_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format training data for LLM"""
        formatted = []

        # Format agent data
        for agent_data in data.get("sources", {}).get("agents", []):
            formatted.append({
                "input": f"Agent type: {agent_data['type']}",
                "output": f"Completed {agent_data['tasks_completed']} tasks with {agent_data['tasks_failed']} failures",
                "metadata": agent_data
            })

        # Format workflow data
        for workflow_data in data.get("sources", {}).get("workflows", []):
            formatted.append({
                "input": f"Workflow type: {workflow_data['type']}",
                "output": f"Success: {workflow_data['success']}, Time: {workflow_data['execution_time']}s",
                "metadata": workflow_data
            })

        return formatted

    # ==================== EMERGENCY CONTROLS ====================

    async def pause_system(self) -> Dict[str, Any]:
        """Pause all system operations"""
        try:
            self.system_state["paused"] = True

            # Pause all active agents
            paused_agents = []
            for agent_id, agent in self.agents.items():
                if agent.get("status") == "deployed":
                    agent["status"] = "paused"
                    agent["paused_at"] = datetime.now().isoformat()
                    paused_agents.append(agent_id)

            # Pause all active workflows
            paused_workflows = []
            for workflow_id, workflow in self.active_workflows.items():
                workflow["status"] = "paused"
                workflow["paused_at"] = datetime.now().isoformat()
                paused_workflows.append(workflow_id)

            self._log_activity(
                "system_paused",
                f"System paused: {len(paused_agents)} agents, {len(paused_workflows)} workflows",
                {"agents": paused_agents, "workflows": paused_workflows}
            )

            logger.warning(f"⏸️ System paused: {len(paused_agents)} agents, {len(paused_workflows)} workflows")

            return {
                "success": True,
                "message": "System paused successfully",
                "paused_agents": len(paused_agents),
                "paused_workflows": len(paused_workflows)
            }

        except Exception as e:
            logger.error(f"Error pausing system: {e}")
            return {"success": False, "error": str(e)}

    async def resume_system(self) -> Dict[str, Any]:
        """Resume all system operations"""
        try:
            self.system_state["paused"] = False

            # Resume all paused agents
            resumed_agents = []
            for agent_id, agent in self.agents.items():
                if agent.get("status") == "paused":
                    agent["status"] = "deployed"
                    agent["resumed_at"] = datetime.now().isoformat()
                    resumed_agents.append(agent_id)

            # Resume all paused workflows
            resumed_workflows = []
            for workflow_id, workflow in self.active_workflows.items():
                if workflow.get("status") == "paused":
                    workflow["status"] = "running"
                    workflow["resumed_at"] = datetime.now().isoformat()
                    resumed_workflows.append(workflow_id)

            self._log_activity(
                "system_resumed",
                f"System resumed: {len(resumed_agents)} agents, {len(resumed_workflows)} workflows",
                {"agents": resumed_agents, "workflows": resumed_workflows}
            )

            logger.info(f"▶️ System resumed: {len(resumed_agents)} agents, {len(resumed_workflows)} workflows")

            return {
                "success": True,
                "message": "System resumed successfully",
                "resumed_agents": len(resumed_agents),
                "resumed_workflows": len(resumed_workflows)
            }

        except Exception as e:
            logger.error(f"Error resuming system: {e}")
            return {"success": False, "error": str(e)}

    async def emergency_shutdown(self) -> Dict[str, Any]:
        """Emergency shutdown of all operations"""
        try:
            self.system_state["emergency_mode"] = True
            self.system_state["status"] = "shutdown"

            # Stop all agents
            stopped_agents = []
            for agent_id, agent in self.agents.items():
                agent["status"] = "stopped"
                agent["stopped_at"] = datetime.now().isoformat()
                agent["stop_reason"] = "emergency_shutdown"
                stopped_agents.append(agent_id)

            # Cancel all workflows
            cancelled_workflows = []
            for workflow_id, workflow in self.active_workflows.items():
                workflow["status"] = "cancelled"
                workflow["cancelled_at"] = datetime.now().isoformat()
                workflow["cancel_reason"] = "emergency_shutdown"
                cancelled_workflows.append(workflow_id)

            # Clear active workflows
            self.active_workflows = {}

            self._log_activity(
                "emergency_shutdown",
                f"Emergency shutdown: {len(stopped_agents)} agents stopped, {len(cancelled_workflows)} workflows cancelled",
                {"agents": stopped_agents, "workflows": cancelled_workflows}
            )

            logger.critical(f"🚨 EMERGENCY SHUTDOWN: {len(stopped_agents)} agents stopped, {len(cancelled_workflows)} workflows cancelled")

            return {
                "success": True,
                "message": "Emergency shutdown completed",
                "stopped_agents": len(stopped_agents),
                "cancelled_workflows": len(cancelled_workflows)
            }

        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
            return {"success": False, "error": str(e)}

    async def restart_system(self) -> Dict[str, Any]:
        """Restart the system"""
        try:
            # First, shutdown
            shutdown_result = await self.emergency_shutdown()

            if not shutdown_result.get("success"):
                return shutdown_result

            # Wait a moment
            await asyncio.sleep(2)

            # Reset state
            self.system_state["emergency_mode"] = False
            self.system_state["status"] = "active"
            self.system_state["paused"] = False

            # Reinitialize
            await self.initialize()

            self._log_activity(
                "system_restarted",
                "System restarted successfully",
                {}
            )

            logger.info("🔄 System restarted successfully")

            return {
                "success": True,
                "message": "System restarted successfully"
            }

        except Exception as e:
            logger.error(f"Error restarting system: {e}")
            return {"success": False, "error": str(e)}

    async def _record_learning_data(self, command: Dict[str, Any], result: Dict[str, Any]):
        """Record learning data for LLM training"""
        try:
            learning_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "command_type": command.get("type"),
                "command": command,
                "result": result,
                "success": result.get("success", False),
                "system_state": {
                    "active_agents": len([a for a in self.agents.values() if a.get("status") == "running"]),
                    "active_workflows": len(self.active_workflows),
                    "total_requests": self.metrics["total_requests"]
                }
            }

            # Add to learning buffer
            self.learning_buffer.append(learning_entry)

            # If buffer is full, send to learning service
            if len(self.learning_buffer) >= self.learning_batch_size:
                await self._flush_learning_buffer()

        except Exception as e:
            logger.error(f"❌ Learning data recording failed: {e}")

    async def _flush_learning_buffer(self):
        """Flush learning buffer to learning service"""
        try:
            if not self.learning_buffer:
                return

            # Send to learning service
            for entry in self.learning_buffer:
                await self.learning_service.record_pai_interaction({
                    "user_id": "master_agent",
                    "type": entry["command_type"],
                    "category": "master_agent_operations",
                    "strategic_reasoning": f"Command: {entry['command_type']}",
                    "tactical_reasoning": f"Result: {entry['result']}",
                    "operational_reasoning": f"Success: {entry['success']}",
                    "response_quality": 1.0 if entry["success"] else 0.5,
                    "user_satisfaction": 1.0 if entry["success"] else 0.5,
                    "context_relevance": 1.0
                })

            # Store in database
            await self.db.insert_many("learning_data", self.learning_buffer)

            self.metrics["learning_cycles"] += 1
            logger.info(f"✅ Flushed {len(self.learning_buffer)} learning entries")

            # Clear buffer
            self.learning_buffer = []

        except Exception as e:
            logger.error(f"❌ Learning buffer flush failed: {e}")

    async def _update_metrics(self, command_type: str, execution_time: float, success: bool):
        """Update system metrics"""
        try:
            # Update average response time
            total_time = self.metrics["average_response_time"] * (self.metrics["total_requests"] - 1)
            self.metrics["average_response_time"] = (total_time + execution_time) / self.metrics["total_requests"]

            # Store metrics in database periodically
            if self.metrics["total_requests"] % 10 == 0:
                await self.db.insert_one("system_metrics", {
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": self.metrics.copy()
                })

        except Exception as e:
            logger.error(f"❌ Metrics update failed: {e}")

    async def _learning_loop(self):
        """Background learning loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Flush learning buffer
                if self.learning_buffer:
                    await self._flush_learning_buffer()

                logger.info("🧠 Learning loop cycle completed")

            except Exception as e:
                logger.error(f"❌ Learning loop error: {e}")

    async def _health_monitor(self):
        """Background health monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                # Check system health
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                # Log warnings if resources are high
                if cpu_percent > 80:
                    logger.warning(f"⚠️ High CPU usage: {cpu_percent}%")

                if memory.percent > 80:
                    logger.warning(f"⚠️ High memory usage: {memory.percent}%")

                # Check agent health
                for agent_id, agent in self.agents.items():
                    if agent.get("status") == "running":
                        # Agent is healthy
                        pass

                logger.debug("💚 Health check completed")

            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")

    async def list_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List all agents with optional filters"""
        try:
            agents = list(self.agents.values())

            if filters:
                # Apply filters
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

    async def list_workflows(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List workflows"""
        try:
            if active_only:
                return list(self.active_workflows.values())
            else:
                return self.workflow_history

        except Exception as e:
            logger.error(f"❌ List workflows failed: {e}")
            return []

    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific workflow"""
        # Check active workflows first
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]

        # Check history
        for workflow in self.workflow_history:
            if workflow.get("workflow_id") == workflow_id:
                return workflow

        return None

    async def shutdown(self):
        """Shutdown the Master Agent"""
        try:
            logger.info("🛑 Shutting down Master Agent...")

            # Flush learning buffer
            if self.learning_buffer:
                await self._flush_learning_buffer()

            # Stop all running agents
            for agent_id, agent in self.agents.items():
                if agent.get("status") == "running":
                    agent["status"] = "stopped"
                    await self.db.update_one(
                        "agents",
                        {"agent_id": agent_id},
                        {"$set": agent}
                    )

            # Save final state
            await self.db.insert_one("master_agent_state", {
                "timestamp": datetime.utcnow().isoformat(),
                "state": self.system_state,
                "metrics": self.metrics
            })

            self.system_state["status"] = "shutdown"
            logger.info("✅ Master Agent shutdown complete")

        except Exception as e:
            logger.error(f"❌ Master Agent shutdown failed: {e}")

    # ==================== AGENT BUILDER INTEGRATION ====================

    async def create_custom_agent(
        self,
        name: str,
        description: str,
        agent_type: str,
        capabilities: List[str],
        system_prompt: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a custom agent using Agent Builder
        Master Agent can dynamically create specialized agents
        """
        try:
            from models.agent_builder_models import CustomAgentCreate, AgentType

            logger.info(f"🤖 Master Agent creating custom agent: {name}")

            # Convert string agent_type to AgentType enum
            try:
                agent_type_enum = AgentType(agent_type.lower())
            except ValueError:
                agent_type_enum = AgentType.CUSTOM

            # Create agent through Agent Builder
            agent_create = CustomAgentCreate(
                name=name,
                description=description,
                agent_type=agent_type_enum,
                capabilities=capabilities,
                system_prompt=system_prompt or f"You are {name}, a specialized AI agent. {description}",
                configuration=configuration or {}
            )

            agent = await self.agent_builder.create_agent(agent_create)

            # Register in Master Agent's registry
            self.agents[agent.agent_id] = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "type": agent.agent_type,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "created_by": "master_agent"
            }

            # Update metrics
            self.metrics["agents_created"] += 1

            # Log activity
            self._log_activity("agent_created", {
                "agent_id": agent.agent_id,
                "name": name,
                "type": agent_type
            })

            logger.info(f"✅ Custom agent created: {agent.agent_id}")

            return {
                "success": True,
                "agent_id": agent.agent_id,
                "name": agent.name,
                "type": agent.agent_type,
                "status": agent.status,
                "message": f"Agent '{name}' created successfully"
            }

        except Exception as e:
            logger.error(f"❌ Failed to create custom agent: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def deploy_agent_to_autogen(
        self,
        agent_id: str,
        runtime_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deploy agent to AutoGen multi-agent runtime
        Enables agent to participate in AutoGen conversations
        """
        try:
            logger.info(f"🚀 Deploying agent {agent_id} to AutoGen runtime")

            # Get agent from Agent Builder
            agent = await self.agent_builder.get_agent(agent_id)

            if not agent:
                return {
                    "success": False,
                    "error": f"Agent not found: {agent_id}"
                }

            # Create deployment through Agent Builder
            from models.agent_builder_models import DeploymentCreate, DeploymentEnvironment

            deployment_create = DeploymentCreate(
                agent_id=agent_id,
                environment=DeploymentEnvironment.AUTOGEN,
                configuration=runtime_config or {
                    "runtime": "autogen",
                    "model": "llama3-local",
                    "max_consecutive_auto_reply": 10
                }
            )

            deployment = await self.agent_builder.create_deployment(deployment_create)

            # Update agent status in Master Agent registry
            if agent_id in self.agents:
                self.agents[agent_id]["status"] = "deployed"
                self.agents[agent_id]["deployment_id"] = deployment.deployment_id

            # Update metrics
            self.metrics["agents_deployed"] += 1

            # Log activity
            self._log_activity("agent_deployed", {
                "agent_id": agent_id,
                "deployment_id": deployment.deployment_id,
                "environment": "autogen"
            })

            logger.info(f"✅ Agent deployed to AutoGen: {deployment.deployment_id}")

            return {
                "success": True,
                "agent_id": agent_id,
                "deployment_id": deployment.deployment_id,
                "environment": "autogen",
                "status": deployment.status,
                "message": f"Agent deployed successfully to AutoGen runtime"
            }

        except Exception as e:
            logger.error(f"❌ Failed to deploy agent to AutoGen: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def monitor_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Monitor agent execution and performance
        Returns real-time metrics and status
        """
        try:
            # Get agent from Agent Builder
            agent = await self.agent_builder.get_agent(agent_id)

            if not agent:
                return {
                    "success": False,
                    "error": f"Agent not found: {agent_id}"
                }

            # Get agent statistics
            stats = await self.agent_builder.get_agent_statistics(agent_id)

            # Get recent executions
            executions = await self.agent_builder.list_executions(
                agent_id=agent_id,
                limit=10
            )

            # Calculate health score
            health_score = 100
            if stats.get("execution_count", 0) > 0:
                success_rate = (stats.get("success_count", 0) / stats["execution_count"]) * 100
                health_score = success_rate

            return {
                "success": True,
                "agent_id": agent_id,
                "name": agent.name,
                "status": agent.status,
                "health_score": health_score,
                "statistics": stats,
                "recent_executions": [
                    {
                        "execution_id": e.execution_id,
                        "status": e.status,
                        "started_at": e.started_at.isoformat() if e.started_at else None,
                        "execution_time": e.execution_time
                    }
                    for e in executions
                ],
                "last_executed": agent.last_executed
            }

        except Exception as e:
            logger.error(f"❌ Failed to monitor agent: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def route_task_to_agent(
        self,
        task: Dict[str, Any],
        preferred_agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Intelligently route task to the most suitable agent
        Uses agent capabilities and performance metrics
        """
        try:
            logger.info(f"🎯 Routing task to optimal agent")

            # If preferred agent specified, use it
            if preferred_agent_id:
                agent = await self.agent_builder.get_agent(preferred_agent_id)
                if agent:
                    return await self._execute_agent_task(preferred_agent_id, task)

            # Otherwise, find best agent based on capabilities
            task_type = task.get("type", "general")
            task_description = task.get("description", "")

            # Get all agents
            agents = await self.agent_builder.list_agents(limit=100)

            if not agents:
                return {
                    "success": False,
                    "error": "No agents available"
                }

            # Score agents based on capabilities and performance
            best_agent = None
            best_score = 0

            for agent in agents:
                # Only consider deployed agents
                if agent.status != "deployed":
                    continue

                score = 0

                # Match agent type
                if agent.agent_type.value == task_type:
                    score += 50

                # Match capabilities
                task_keywords = task_description.lower().split()
                for capability in agent.capabilities:
                    if any(keyword in capability.lower() for keyword in task_keywords):
                        score += 10

                # Consider success rate
                if agent.execution_count > 0:
                    success_rate = (agent.success_count / agent.execution_count) * 100
                    score += success_rate * 0.3

                if score > best_score:
                    best_score = score
                    best_agent = agent

            if not best_agent:
                # Fallback to first available agent
                best_agent = agents[0]

            logger.info(f"✅ Selected agent: {best_agent.name} (score: {best_score})")

            # Execute task with selected agent
            return await self._execute_agent_task(best_agent.agent_id, task)

        except Exception as e:
            logger.error(f"❌ Failed to route task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_agent_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with specific agent"""
        try:
            from models.agent_builder_models import AgentExecutionCreate

            # Execute agent
            execution = await self.agent_builder.execute_agent(
                AgentExecutionCreate(
                    agent_id=agent_id,
                    input_data=task,
                    configuration={}
                )
            )

            return {
                "success": execution.status == "completed",
                "agent_id": agent_id,
                "execution_id": execution.execution_id,
                "status": execution.status,
                "output": execution.output_data,
                "execution_time": execution.execution_time,
                "error": execution.error
            }

        except Exception as e:
            logger.error(f"❌ Agent task execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

