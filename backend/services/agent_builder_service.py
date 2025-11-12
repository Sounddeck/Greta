"""
Agent Builder Service for CPAS4 - Real Database Integration
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from database import Database
from models.agent_builder_models import *

logger = logging.getLogger(__name__)

class AgentBuilderService:
    def __init__(self, database: Database):
        self.db = database
        self.templates_collection = "agent_templates"
        self.agents_collection = "custom_agents"
        self.versions_collection = "agent_versions"
        self.deployments_collection = "agent_deployments"
        self.executions_collection = "agent_executions"
        self.drafts_collection = "agent_drafts"  # NEW: For autosave
        logger.info("✅ Agent Builder Service initialized")

    # Template CRUD
    async def create_template(self, t: AgentTemplateCreate) -> AgentTemplate:
        tid = f"template_{uuid.uuid4().hex[:12]}"
        template = AgentTemplate(template_id=tid, **t.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        td = template.model_dump()
        td["created_at"], td["updated_at"] = td["created_at"].isoformat(), td["updated_at"].isoformat()
        await self.db.insert_one(self.templates_collection, td)
        return template
    
    async def get_template(self, tid: str) -> Optional[AgentTemplate]:
        td = await self.db.find_one(self.templates_collection, {"template_id": tid})
        if not td: return None
        for k in ["created_at", "updated_at"]:
            if k in td and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
        return AgentTemplate(**td)
    
    async def update_template(self, tid: str, tu: AgentTemplateUpdate) -> Optional[AgentTemplate]:
        ud = tu.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_template(tid) if await self.db.update_one(self.templates_collection, {"template_id": tid}, {"$set": ud}) else None
    
    async def delete_template(self, tid: str) -> bool:
        return await self.db.delete_one(self.templates_collection, {"template_id": tid})
    
    async def list_templates(self, agent_type: Optional[AgentType] = None, tags: Optional[List[str]] = None, limit: int = 100) -> List[AgentTemplate]:
        f = {}
        if agent_type: f["agent_type"] = agent_type
        if tags: f["tags"] = {"$in": tags}
        tds = await self.db.find_many(self.templates_collection, f, limit=limit)
        templates = []
        for td in tds:
            for k in ["created_at", "updated_at"]:
                if k in td and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
            templates.append(AgentTemplate(**td))
        return templates
    
    # Agent CRUD
    async def create_agent(self, ac: CustomAgentCreate) -> CustomAgent:
        aid = f"agent_{uuid.uuid4().hex[:12]}"
        if ac.template_id:
            t = await self.get_template(ac.template_id)
            if t: await self.db.update_one(self.templates_collection, {"template_id": ac.template_id}, {"$inc": {"usage_count": 1}})
        agent = CustomAgent(agent_id=aid, **ac.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        ad = agent.model_dump()
        for k in ["created_at", "updated_at", "deployed_at", "last_executed"]:
            if ad.get(k): ad[k] = ad[k].isoformat()
        await self.db.insert_one(self.agents_collection, ad)
        return agent
    
    async def get_agent(self, aid: str) -> Optional[CustomAgent]:
        ad = await self.db.find_one(self.agents_collection, {"agent_id": aid})
        if not ad: return None
        for k in ["created_at", "updated_at", "deployed_at", "last_executed"]:
            if k in ad and ad[k] and isinstance(ad[k], str): ad[k] = datetime.fromisoformat(ad[k])
        return CustomAgent(**ad)
    
    async def update_agent(self, aid: str, au: CustomAgentUpdate, auto_version: bool = True, version_changes: str = "Agent updated") -> Optional[CustomAgent]:
        """Update an agent and optionally create a new version"""
        # Get current agent state before update
        current_agent = await self.get_agent(aid)
        if not current_agent:
            return None

        # Prepare update data
        ud = au.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()

        if "status" in ud and ud["status"] == AgentStatus.DEPLOYED:
            if not current_agent.deployed_at:
                ud["deployed_at"] = datetime.utcnow().isoformat()

        # Create version snapshot before update (if auto_version is enabled)
        if auto_version:
            try:
                # Create version with current state
                version_number = current_agent.version + 1
                vc = AgentVersionCreate(
                    agent_id=aid,
                    version_number=version_number,
                    changes=version_changes,
                    configuration_snapshot=current_agent.model_dump(
                        exclude={"agent_id", "version", "created_at", "updated_at", "current_version_id"}
                    ),
                    tags=["auto-version"]
                )
                await self.create_version(vc)
                ud["version"] = version_number
                logger.info(f"✅ Auto-created version {version_number} for agent {aid}")
            except Exception as e:
                logger.warning(f"Failed to auto-create version: {e}")

        # Perform update
        success = await self.db.update_one(self.agents_collection, {"agent_id": aid}, {"$set": ud})
        return await self.get_agent(aid) if success else None
    
    async def delete_agent(self, aid: str) -> bool:
        return await self.db.delete_one(self.agents_collection, {"agent_id": aid})
    
    async def list_agents(self, status: Optional[AgentStatus] = None, agent_type: Optional[AgentType] = None, tags: Optional[List[str]] = None, limit: int = 100) -> List[CustomAgent]:
        f = {}
        if status: f["status"] = status
        if agent_type: f["agent_type"] = agent_type
        if tags: f["tags"] = {"$in": tags}
        ads = await self.db.find_many(self.agents_collection, f, limit=limit)
        agents = []
        for ad in ads:
            for k in ["created_at", "updated_at", "deployed_at", "last_executed"]:
                if k in ad and ad[k] and isinstance(ad[k], str): ad[k] = datetime.fromisoformat(ad[k])
            agents.append(CustomAgent(**ad))
        return agents
    
    async def clone_agent(self, aid: str, new_name: str) -> Optional[CustomAgent]:
        a = await self.get_agent(aid)
        if not a: return None
        ac = CustomAgentCreate(name=new_name, description=a.description, agent_type=a.agent_type, capabilities=a.capabilities, system_prompt=a.system_prompt, tools=a.tools, configuration=a.configuration, tags=a.tags, metadata=a.metadata, status=AgentStatus.DRAFT)
        return await self.create_agent(ac)
    
    # Version management
    async def create_version(self, vc: AgentVersionCreate) -> AgentVersion:
        """Create a new version of an agent"""
        vid = f"version_{uuid.uuid4().hex[:12]}"
        version = AgentVersion(version_id=vid, **vc.model_dump(), created_at=datetime.utcnow())
        vd = version.model_dump()
        vd["created_at"] = vd["created_at"].isoformat()
        await self.db.insert_one(self.versions_collection, vd)
        await self.db.update_one(self.agents_collection, {"agent_id": vc.agent_id}, {"$set": {"current_version_id": vid, "version": vc.version_number}})
        logger.info(f"✅ Created version {vid} for agent {vc.agent_id}")
        return version

    async def get_version(self, vid: str) -> Optional[AgentVersion]:
        """Get a specific version by ID"""
        vd = await self.db.find_one(self.versions_collection, {"version_id": vid})
        if not vd:
            return None
        if "created_at" in vd and isinstance(vd["created_at"], str):
            vd["created_at"] = datetime.fromisoformat(vd["created_at"])
        return AgentVersion(**vd)

    async def list_versions(self, aid: str, limit: int = 50) -> List[AgentVersion]:
        """List all versions for an agent, sorted by version number descending"""
        vds = await self.db.find_many(
            self.versions_collection,
            {"agent_id": aid},
            limit=limit,
            sort=[("version_number", -1)]  # Most recent first
        )
        versions = []
        for vd in vds:
            if "created_at" in vd and isinstance(vd["created_at"], str):
                vd["created_at"] = datetime.fromisoformat(vd["created_at"])
            versions.append(AgentVersion(**vd))
        return versions

    async def rollback_version(self, aid: str, vid: str) -> Optional[CustomAgent]:
        """Rollback agent to a specific version"""
        vd = await self.db.find_one(self.versions_collection, {"version_id": vid, "agent_id": aid})
        if not vd:
            return None

        # Get current agent to create a backup version before rollback
        agent = await self.get_agent(aid)
        if agent:
            # Create a backup version before rollback
            backup_vc = AgentVersionCreate(
                agent_id=aid,
                version_number=agent.version + 1,
                changes=f"Backup before rollback to version {vd.get('version_number')}",
                configuration_snapshot=agent.model_dump(exclude={"agent_id", "version", "created_at", "updated_at"}),
                tags=["backup", "pre-rollback"]
            )
            await self.create_version(backup_vc)

        # Perform rollback
        config = vd.get("configuration_snapshot", {})
        au = CustomAgentUpdate(**config)
        rolled_back_agent = await self.update_agent(aid, au)

        logger.info(f"✅ Rolled back agent {aid} to version {vid}")
        return rolled_back_agent

    async def compare_versions(self, vid_a: str, vid_b: str) -> Optional[VersionDiff]:
        """Compare two versions and return differences"""
        version_a = await self.get_version(vid_a)
        version_b = await self.get_version(vid_b)

        if not version_a or not version_b:
            return None

        if version_a.agent_id != version_b.agent_id:
            logger.warning(f"Cannot compare versions from different agents")
            return None

        # Compare configurations
        config_a = version_a.configuration_snapshot
        config_b = version_b.configuration_snapshot

        changes = self._compute_diff(config_a, config_b)
        summary = self._generate_diff_summary(changes)

        return VersionDiff(
            version_a_id=vid_a,
            version_a_number=version_a.version_number,
            version_b_id=vid_b,
            version_b_number=version_b.version_number,
            changes=changes,
            summary=summary
        )

    def _compute_diff(self, config_a: Dict[str, Any], config_b: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Recursively compute differences between two configurations"""
        changes = {}

        # Check all keys in both configs
        all_keys = set(config_a.keys()) | set(config_b.keys())

        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key

            if key not in config_a:
                changes[full_key] = {"added": config_b[key]}
            elif key not in config_b:
                changes[full_key] = {"removed": config_a[key]}
            elif config_a[key] != config_b[key]:
                # Handle nested dicts
                if isinstance(config_a[key], dict) and isinstance(config_b[key], dict):
                    nested_changes = self._compute_diff(config_a[key], config_b[key], full_key)
                    changes.update(nested_changes)
                # Handle lists
                elif isinstance(config_a[key], list) and isinstance(config_b[key], list):
                    added = [item for item in config_b[key] if item not in config_a[key]]
                    removed = [item for item in config_a[key] if item not in config_b[key]]
                    if added or removed:
                        changes[full_key] = {"added": added, "removed": removed}
                # Handle simple values
                else:
                    changes[full_key] = {"old": config_a[key], "new": config_b[key]}

        return changes

    def _generate_diff_summary(self, changes: Dict[str, Any]) -> str:
        """Generate a human-readable summary of changes"""
        if not changes:
            return "No changes"

        summary_parts = []
        for key, change in changes.items():
            if "added" in change and "removed" in change:
                summary_parts.append(f"Modified {key}")
            elif "added" in change:
                summary_parts.append(f"Added {key}")
            elif "removed" in change:
                summary_parts.append(f"Removed {key}")
            elif "old" in change and "new" in change:
                summary_parts.append(f"Changed {key}")

        return "; ".join(summary_parts[:5])  # Limit to 5 changes in summary
    
    # Deployment
    async def deploy_agent(self, dc: DeploymentCreate) -> Deployment:
        did = f"deploy_{uuid.uuid4().hex[:12]}"
        deployment = Deployment(deployment_id=did, **dc.model_dump(), started_at=datetime.utcnow())
        dd = deployment.model_dump()
        dd["started_at"] = dd["started_at"].isoformat()
        if dd.get("completed_at"): dd["completed_at"] = dd["completed_at"].isoformat()
        await self.db.insert_one(self.deployments_collection, dd)
        await self.db.update_one(self.deployments_collection, {"deployment_id": did}, {"$set": {"status": DeploymentStatus.DEPLOYED, "completed_at": datetime.utcnow().isoformat()}})
        await self.db.update_one(self.agents_collection, {"agent_id": dc.agent_id}, {"$set": {"status": AgentStatus.DEPLOYED, "deployed_at": datetime.utcnow().isoformat()}})
        return await self.get_deployment(did)
    
    async def get_deployment(self, did: str) -> Optional[Deployment]:
        dd = await self.db.find_one(self.deployments_collection, {"deployment_id": did})
        if not dd: return None
        for k in ["started_at", "completed_at"]:
            if k in dd and dd[k] and isinstance(dd[k], str): dd[k] = datetime.fromisoformat(dd[k])
        return Deployment(**dd)
    
    async def list_deployments(self, agent_id: Optional[str] = None, status: Optional[DeploymentStatus] = None, limit: int = 100) -> List[Deployment]:
        f = {}
        if agent_id: f["agent_id"] = agent_id
        if status: f["status"] = status
        dds = await self.db.find_many(self.deployments_collection, f, limit=limit)
        deployments = []
        for dd in dds:
            for k in ["started_at", "completed_at"]:
                if k in dd and dd[k] and isinstance(dd[k], str): dd[k] = datetime.fromisoformat(dd[k])
            deployments.append(Deployment(**dd))
        return deployments

    # Execution
    async def execute_agent(self, ec: AgentExecutionCreate) -> AgentExecution:
        """
        Execute agent with full LLM integration, tool calling, and MCP support
        This is the core method that brings agents to life
        """
        import time
        from services.llm_service import LLMService
        from services.tool_registry import tool_registry
        from services.mcp_hub_service import MCPHubService

        eid = f"exec_{uuid.uuid4().hex[:12]}"
        start_time = datetime.utcnow()

        try:
            # 1. Get agent configuration
            agent = await self.get_agent(ec.agent_id)

            if not agent:
                raise ValueError(f"Agent not found: {ec.agent_id}")

            # 2. Create execution record with RUNNING status
            execution = AgentExecution(
                execution_id=eid,
                agent_id=ec.agent_id,
                input_data=ec.input_data,
                configuration=ec.configuration,
                status=ExecutionStatus.RUNNING,
                started_at=start_time
            )

            ed = execution.model_dump()
            ed["started_at"] = ed["started_at"].isoformat()
            await self.db.insert_one(self.executions_collection, ed)

            # 3. Initialize services
            llm_service = LLMService()
            mcp_service = MCPHubService(db=self.db)

            # Ensure tool registry is initialized
            if not tool_registry.db:
                await tool_registry.initialize()

            # 4. Get tools from Tool Registry
            registry_tools = await tool_registry.get_tools_for_agent(ec.agent_id)

            # 5. Get MCP tools based on agent configuration
            mcp_tools = await self._get_mcp_tools_for_agent(agent, mcp_service)

            # 6. Get skills based on agent configuration
            skills = await self._get_skills_for_agent(agent)

            # 7. Combine all tools
            all_tools = registry_tools + mcp_tools

            # 8. Initialize memory systems
            memory_context = await self._initialize_memory_for_agent(agent, eid)

            # 9. Build system prompt with agent capabilities, skills, and memory
            system_prompt = self._build_system_prompt(agent, all_tools, skills, memory_context)

            # 10. Get user prompt from input_data
            user_prompt = ec.input_data.get("prompt", ec.input_data.get("message", ""))

            if not user_prompt:
                raise ValueError("No prompt provided in input_data")

            # 11. Inject MCP context
            enhanced_prompt = await mcp_service.inject_mcp_context(ec.agent_id, user_prompt)

            # 12. Inject memory context (retrieve relevant memories)
            if memory_context.get("reasoning_bank"):
                enhanced_prompt = await self._inject_memory_context(
                    agent,
                    enhanced_prompt,
                    memory_context
                )

            # 13. Execute with LLM
            logger.info(f"🤖 Executing agent {agent.name} with {len(all_tools)} tools, {len(skills)} skills, {len(memory_context['enabled_systems'])} memory systems")

            llm_result = await llm_service.generate_completion({
                "message": enhanced_prompt,
                "system_prompt": system_prompt,
                "context": {
                    "agent_id": ec.agent_id,
                    "agent_name": agent.name,
                    "agent_type": agent.agent_type,
                    "capabilities": agent.capabilities,
                    "tools_available": len(all_tools),
                    "skills_available": len(skills),
                    "memory_systems": memory_context["enabled_systems"]
                },
                "use_hrm": True,
                "model": ec.configuration.get("model", "llama3-local")
            })

            # 11. Handle tool calls if present
            tool_results = []
            if llm_result.get("tool_calls"):
                logger.info(f"🔧 Processing {len(llm_result['tool_calls'])} tool calls")

                for tool_call in llm_result["tool_calls"]:
                    tool_result = await self._execute_tool_call(
                        tool_call,
                        ec.agent_id,
                        tool_registry,
                        mcp_service
                    )
                    tool_results.append(tool_result)

            # 12. Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # 13. Calculate tool performance metrics
            tool_performance = {
                "total_tools_called": len(tool_results),
                "successful_tools": sum(1 for t in tool_results if t.get("success")),
                "failed_tools": sum(1 for t in tool_results if not t.get("success")),
                "total_tool_time": sum(t.get("execution_time", 0) for t in tool_results),
                "tools_by_type": {}
            }

            # Group tools by type (MCP vs Registry)
            for tool_result in tool_results:
                tool_name = tool_result.get("tool", "unknown")
                tool_type = "mcp" if tool_name.startswith("mcp_") else "registry"

                if tool_type not in tool_performance["tools_by_type"]:
                    tool_performance["tools_by_type"][tool_type] = {
                        "count": 0,
                        "successful": 0,
                        "failed": 0,
                        "total_time": 0
                    }

                tool_performance["tools_by_type"][tool_type]["count"] += 1
                if tool_result.get("success"):
                    tool_performance["tools_by_type"][tool_type]["successful"] += 1
                else:
                    tool_performance["tools_by_type"][tool_type]["failed"] += 1
                tool_performance["tools_by_type"][tool_type]["total_time"] += tool_result.get("execution_time", 0)

            # 14. Prepare output
            output_data = {
                "response": llm_result.get("response", ""),
                "tool_calls": len(llm_result.get("tool_calls", [])),
                "tool_results": tool_results,
                "tool_performance": tool_performance,
                "skills_used": [s["skill_name"] for s in skills] if skills else [],
                "memory_systems_used": memory_context.get("enabled_systems", []),
                "model_used": llm_result.get("model", "unknown"),
                "tokens_used": llm_result.get("tokens", 0)
            }

            # 15. Store successful execution pattern in ReasoningBank if enabled
            if memory_context.get("reasoning_bank") and agent.memory_config and agent.memory_config.auto_store_patterns:
                try:
                    await self._store_execution_pattern(
                        agent,
                        ec.input_data,
                        output_data,
                        memory_context["reasoning_bank"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to store execution pattern: {e}")

            # 16. Update execution record with SUCCESS
            await self.db.update_one(
                self.executions_collection,
                {"execution_id": eid},
                {
                    "$set": {
                        "status": ExecutionStatus.COMPLETED,
                        "output_data": output_data,
                        "completed_at": datetime.utcnow().isoformat(),
                        "execution_time": execution_time
                    }
                }
            )

            # 15. Update agent statistics
            await self.db.update_one(
                self.agents_collection,
                {"agent_id": ec.agent_id},
                {
                    "$inc": {
                        "execution_count": 1,
                        "success_count": 1
                    },
                    "$set": {
                        "last_executed": datetime.utcnow().isoformat(),
                        "average_execution_time": execution_time  # Simplified, should be running average
                    }
                }
            )

            logger.info(f"✅ Agent execution completed in {execution_time:.2f}s")

            return await self.get_execution(eid)

        except Exception as e:
            # Handle execution failure
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_message = str(e)

            logger.error(f"❌ Agent execution failed: {error_message}")

            # Update execution record with FAILED status
            await self.db.update_one(
                self.executions_collection,
                {"execution_id": eid},
                {
                    "$set": {
                        "status": ExecutionStatus.FAILED,
                        "error": error_message,
                        "completed_at": datetime.utcnow().isoformat(),
                        "execution_time": execution_time
                    }
                }
            )

            # Update agent failure count
            await self.db.update_one(
                self.agents_collection,
                {"agent_id": ec.agent_id},
                {
                    "$inc": {
                        "execution_count": 1,
                        "failure_count": 1
                    },
                    "$set": {
                        "last_executed": datetime.utcnow().isoformat()
                    }
                }
            )

            return await self.get_execution(eid)

    def _build_system_prompt(
        self,
        agent,
        tools: List[Dict[str, Any]],
        skills: List[Dict[str, Any]] = None,
        memory_context: Dict[str, Any] = None
    ) -> str:
        """Build system prompt with agent configuration, tools, skills, and memory"""

        prompt_parts = [
            f"You are {agent.name}, a specialized AI agent.",
            f"\nAgent Type: {agent.agent_type}",
            f"\nDescription: {agent.description}",
            f"\nCapabilities: {', '.join(agent.capabilities)}",
        ]

        if agent.system_prompt:
            prompt_parts.append(f"\n{agent.system_prompt}")

        if tools:
            prompt_parts.append(f"\n\nYou have access to {len(tools)} tools to help you complete tasks.")
            prompt_parts.append("Use these tools when appropriate to gather information, process data, or take actions.")

        if skills and len(skills) > 0:
            prompt_parts.append(f"\n\nYou have {len(skills)} specialized skills:")
            for skill in skills:
                prompt_parts.append(f"- {skill['display_name']}: {skill['description']}")
            prompt_parts.append("Activate these skills when appropriate for the task at hand.")

        if memory_context and len(memory_context.get("enabled_systems", [])) > 0:
            prompt_parts.append(f"\n\nYou have access to {len(memory_context['enabled_systems'])} memory systems:")
            if "vector_search" in memory_context["enabled_systems"]:
                prompt_parts.append("- Vector Search: Semantic memory storage and retrieval")
            if "reasoning_bank" in memory_context["enabled_systems"]:
                prompt_parts.append("- ReasoningBank: Pattern storage and retrieval for similar problems")
            if "functional_memory" in memory_context["enabled_systems"]:
                prompt_parts.append("- Functional Memory: Episodic, semantic, procedural, and working memory")
            if "memory_orchestrator" in memory_context["enabled_systems"]:
                prompt_parts.append("- Memory Orchestrator: Intelligent memory management and synthesis")
            prompt_parts.append("Use these memory systems to learn from past experiences and improve over time.")

        return "\n".join(prompt_parts)

    async def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        agent_id: str,
        tool_registry,
        mcp_service
    ) -> Dict[str, Any]:
        """Execute a single tool call"""
        try:
            tool_name = tool_call.get("name", tool_call.get("function", {}).get("name"))
            parameters = tool_call.get("parameters", tool_call.get("arguments", {}))

            logger.info(f"🔧 Executing tool: {tool_name}")

            # Determine if this is a Tool Registry tool or MCP tool
            if tool_name.startswith("mcp_"):
                # MCP tool
                result = await mcp_service.execute_mcp_tool(tool_name, parameters, agent_id)
            else:
                # Tool Registry tool
                result = await tool_registry.execute_tool(tool_name, parameters, agent_id)

            return {
                "tool": tool_name,
                "success": result.get("success", False),
                "result": result.get("result"),
                "error": result.get("error"),
                "execution_time": result.get("execution_time", 0)
            }

        except Exception as e:
            logger.error(f"❌ Tool call failed: {e}")
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e)
            }

    async def get_execution(self, eid: str) -> Optional[AgentExecution]:
        ed = await self.db.find_one(self.executions_collection, {"execution_id": eid})
        if not ed: return None
        for k in ["started_at", "completed_at"]:
            if k in ed and ed[k] and isinstance(ed[k], str): ed[k] = datetime.fromisoformat(ed[k])
        return AgentExecution(**ed)

    async def list_executions(self, agent_id: Optional[str] = None, status: Optional[ExecutionStatus] = None, limit: int = 100) -> List[AgentExecution]:
        f = {}
        if agent_id: f["agent_id"] = agent_id
        if status: f["status"] = status
        eds = await self.db.find_many(self.executions_collection, f, limit=limit)
        executions = []
        for ed in eds:
            for k in ["started_at", "completed_at"]:
                if k in ed and ed[k] and isinstance(ed[k], str): ed[k] = datetime.fromisoformat(ed[k])
            executions.append(AgentExecution(**ed))
        return executions
    
    # Code generation
    async def generate_code(self, req: CodeGenerationRequest) -> CodeGenerationResult:
        agent = await self.get_agent(req.agent_id)
        if not agent: raise ValueError("Agent not found")
        code = f"""# Generated Agent: {agent.name}
class {agent.name.replace(' ', '')}Agent:
    def __init__(self):
        self.name = "{agent.name}"
        self.type = "{agent.agent_type}"
        self.capabilities = {agent.capabilities}
    
    def execute(self, input_data):
        # Agent execution logic
        return {{"status": "success", "result": "Agent executed"}}
"""
        result = CodeGenerationResult(agent_id=req.agent_id, generated_code=code, is_valid=True, generated_at=datetime.utcnow())
        await self.db.update_one(self.agents_collection, {"agent_id": req.agent_id}, {"$set": {"generated_code": code}})
        return result
    
    # Validation
    async def validate_agent(self, aid: str) -> ValidationResult:
        agent = await self.get_agent(aid)
        if not agent: return ValidationResult(agent_id=aid, is_valid=False, errors=["Agent not found"])
        errors, warnings = [], []
        if not agent.name: errors.append("Agent name is required")
        if not agent.system_prompt: warnings.append("System prompt is empty")
        if not agent.capabilities: warnings.append("No capabilities defined")
        return ValidationResult(agent_id=aid, is_valid=len(errors)==0, errors=errors, warnings=warnings, validated_at=datetime.utcnow())
    
    # Statistics
    async def get_stats(self) -> AgentBuilderStats:
        all_agents = await self.list_agents(limit=10000)
        all_templates = await self.list_templates(limit=10000)
        all_executions = await self.list_executions(limit=10000)
        all_deployments = await self.list_deployments(limit=10000)
        agents_by_status = {s.value: sum(1 for a in all_agents if a.status == s) for s in AgentStatus}
        agents_by_type = {t.value: sum(1 for a in all_agents if a.agent_type == t) for t in AgentType}
        executions_by_status = {s.value: sum(1 for e in all_executions if e.status == s) for s in ExecutionStatus}
        success_count = sum(a.success_count for a in all_agents)
        total_count = sum(a.execution_count for a in all_agents)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0.0
        avg_time = sum(a.average_execution_time for a in all_agents) / len(all_agents) if all_agents else 0.0
        return AgentBuilderStats(
            total_agents=len(all_agents), total_templates=len(all_templates), total_executions=len(all_executions),
            agents_by_status=agents_by_status, agents_by_type=agents_by_type, executions_by_status=executions_by_status,
            most_used_tools=[], most_used_templates=[], average_execution_time=round(avg_time, 2), success_rate=round(success_rate, 2),
            total_deployments=len(all_deployments), active_deployments=sum(1 for d in all_deployments if d.status == DeploymentStatus.DEPLOYED)
        )

    # ============================================================================
    # DRAFT MANAGEMENT (for autosave functionality)
    # ============================================================================

    async def create_draft(self, dc: AgentDraftCreate) -> AgentDraft:
        """Create a new agent draft for autosave"""
        from datetime import timedelta

        did = f"draft_{uuid.uuid4().hex[:12]}"
        expires_at = datetime.utcnow() + timedelta(days=7)  # Drafts expire after 7 days

        draft = AgentDraft(
            draft_id=did,
            **dc.model_dump(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            expires_at=expires_at
        )

        dd = draft.model_dump()
        for k in ["created_at", "updated_at", "expires_at"]:
            if dd.get(k):
                dd[k] = dd[k].isoformat()

        await self.db.insert_one(self.drafts_collection, dd)
        logger.info(f"✅ Created draft {did}")
        return draft

    async def get_draft(self, did: str) -> Optional[AgentDraft]:
        """Get a draft by ID"""
        dd = await self.db.find_one(self.drafts_collection, {"draft_id": did})
        if not dd:
            return None

        for k in ["created_at", "updated_at", "expires_at"]:
            if k in dd and dd[k] and isinstance(dd[k], str):
                dd[k] = datetime.fromisoformat(dd[k])

        return AgentDraft(**dd)

    async def update_draft(self, did: str, du: AgentDraftUpdate) -> Optional[AgentDraft]:
        """Update a draft (autosave)"""
        ud = du.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()

        success = await self.db.update_one(
            self.drafts_collection,
            {"draft_id": did},
            {"$set": ud}
        )

        if success:
            logger.info(f"✅ Updated draft {did}")
            return await self.get_draft(did)
        return None

    async def delete_draft(self, did: str) -> bool:
        """Delete a draft"""
        success = await self.db.delete_one(self.drafts_collection, {"draft_id": did})
        if success:
            logger.info(f"✅ Deleted draft {did}")
        return success

    async def list_drafts(self, limit: int = 100) -> List[AgentDraft]:
        """List all drafts (not expired)"""
        # Filter out expired drafts
        current_time = datetime.utcnow().isoformat()
        dds = await self.db.find_many(
            self.drafts_collection,
            {"$or": [
                {"expires_at": {"$gt": current_time}},
                {"expires_at": None}
            ]},
            limit=limit
        )

        drafts = []
        for dd in dds:
            for k in ["created_at", "updated_at", "expires_at"]:
                if k in dd and dd[k] and isinstance(dd[k], str):
                    dd[k] = datetime.fromisoformat(dd[k])
            drafts.append(AgentDraft(**dd))

        return drafts

    async def convert_draft_to_agent(self, did: str) -> Optional[CustomAgent]:
        """Convert a draft to a full agent"""
        draft = await self.get_draft(did)
        if not draft:
            return None

        # Create agent from draft
        ac = CustomAgentCreate(
            name=draft.name,
            description=draft.description,
            agent_type=draft.agent_type or AgentType.GENERAL,
            capabilities=draft.capabilities,
            system_prompt=draft.system_prompt,
            tools=draft.tools,
            configuration=draft.configuration,
            tags=draft.tags,
            metadata=draft.metadata,
            template_id=draft.template_id,
            status=AgentStatus.DRAFT
        )

        agent = await self.create_agent(ac)

        # Delete the draft after conversion
        await self.delete_draft(did)

        logger.info(f"✅ Converted draft {did} to agent {agent.agent_id}")
        return agent

    async def cleanup_expired_drafts(self) -> int:
        """Clean up expired drafts (can be run periodically)"""
        current_time = datetime.utcnow().isoformat()

        # Find expired drafts
        expired_drafts = await self.db.find_many(
            self.drafts_collection,
            {"expires_at": {"$lt": current_time}},
            limit=1000
        )

        count = 0
        for draft in expired_drafts:
            await self.db.delete_one(self.drafts_collection, {"draft_id": draft["draft_id"]})
            count += 1

        if count > 0:
            logger.info(f"✅ Cleaned up {count} expired drafts")

        return count

    # ============================================================================
    # ENHANCED VALIDATION
    # ============================================================================

    async def validate_agent_enhanced(self, aid: Optional[str] = None, draft_id: Optional[str] = None) -> EnhancedValidationResult:
        """Enhanced validation with detailed field-level feedback"""

        # Get agent or draft
        if aid:
            agent = await self.get_agent(aid)
            if not agent:
                return EnhancedValidationResult(
                    agent_id=aid,
                    is_valid=False,
                    errors=[{"field": "agent", "message": "Agent not found"}]
                )
            data = agent.model_dump()
        elif draft_id:
            draft = await self.get_draft(draft_id)
            if not draft:
                return EnhancedValidationResult(
                    draft_id=draft_id,
                    is_valid=False,
                    errors=[{"field": "draft", "message": "Draft not found"}]
                )
            data = draft.model_dump()
        else:
            return EnhancedValidationResult(
                is_valid=False,
                errors=[{"field": "input", "message": "Either agent_id or draft_id is required"}]
            )

        errors = []
        warnings = []
        suggestions = []
        field_validations = {}

        # Validate name
        if not data.get("name") or len(data["name"].strip()) == 0:
            errors.append({"field": "name", "message": "Agent name is required"})
            field_validations["name"] = False
        elif len(data["name"]) < 3:
            warnings.append({"field": "name", "message": "Agent name should be at least 3 characters"})
            field_validations["name"] = True
        else:
            field_validations["name"] = True

        # Validate description
        if not data.get("description") or len(data["description"].strip()) == 0:
            warnings.append({"field": "description", "message": "Description is recommended"})
            field_validations["description"] = False
        elif len(data["description"]) < 10:
            suggestions.append({"field": "description", "message": "Consider adding a more detailed description"})
            field_validations["description"] = True
        else:
            field_validations["description"] = True

        # Validate agent type
        if not data.get("agent_type"):
            errors.append({"field": "agent_type", "message": "Agent type is required"})
            field_validations["agent_type"] = False
        else:
            field_validations["agent_type"] = True

        # Validate system prompt
        if not data.get("system_prompt") or len(data["system_prompt"].strip()) == 0:
            warnings.append({"field": "system_prompt", "message": "System prompt is recommended for better agent behavior"})
            field_validations["system_prompt"] = False
        elif len(data["system_prompt"]) < 20:
            suggestions.append({"field": "system_prompt", "message": "Consider adding a more detailed system prompt"})
            field_validations["system_prompt"] = True
        else:
            field_validations["system_prompt"] = True

        # Validate capabilities
        if not data.get("capabilities") or len(data["capabilities"]) == 0:
            warnings.append({"field": "capabilities", "message": "Adding capabilities helps define agent behavior"})
            field_validations["capabilities"] = False
        else:
            field_validations["capabilities"] = True

        # Validate tools
        if not data.get("tools") or len(data["tools"]) == 0:
            suggestions.append({"field": "tools", "message": "Consider adding tools to enhance agent capabilities"})
            field_validations["tools"] = False
        else:
            field_validations["tools"] = True

        # Validate configuration (LLM settings)
        config = data.get("configuration", {})
        if not config.get("llm"):
            suggestions.append({"field": "configuration.llm", "message": "Consider configuring LLM settings"})
            field_validations["configuration"] = False
        else:
            field_validations["configuration"] = True

        # Calculate completion percentage
        total_fields = len(field_validations)
        completed_fields = sum(1 for v in field_validations.values() if v)
        completion_percentage = (completed_fields / total_fields * 100) if total_fields > 0 else 0.0

        is_valid = len(errors) == 0

        return EnhancedValidationResult(
            agent_id=aid,
            draft_id=draft_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            field_validations=field_validations,
            completion_percentage=round(completion_percentage, 1),
            validated_at=datetime.utcnow()
        )

    # ============================================================================
    # MCP TOOLS, SKILLS, AND MEMORY INTEGRATION
    # ============================================================================

    async def _get_mcp_tools_for_agent(
        self,
        agent: CustomAgent,
        mcp_service
    ) -> List[Dict[str, Any]]:
        """
        Get MCP tools for agent based on agent's mcp_tools configuration

        Args:
            agent: Agent configuration
            mcp_service: MCP Hub Service instance

        Returns:
            List of MCP tools in LLM-compatible format
        """
        try:
            tools = []

            # Check if agent has MCP tools configured
            if not agent.mcp_tools or len(agent.mcp_tools) == 0:
                logger.debug(f"Agent {agent.agent_id} has no MCP tools configured")
                return tools

            # Get each configured MCP tool
            for mcp_tool_assignment in agent.mcp_tools:
                if not mcp_tool_assignment.is_enabled:
                    continue

                # Get tool from MCP service
                tool_id = mcp_tool_assignment.tool_id
                server_id = mcp_tool_assignment.server_id

                # Get tool details from database
                tool_doc = await self.db.find_one("mcp_tools", {"tool_id": tool_id})

                if not tool_doc:
                    logger.warning(f"MCP tool not found: {tool_id}")
                    continue

                # Format tool for LLM
                tool = {
                    "type": "function",
                    "function": {
                        "name": f"mcp_{server_id}_{tool_doc.get('name', tool_id)}",
                        "description": tool_doc.get("description", ""),
                        "parameters": tool_doc.get("input_schema", {})
                    },
                    "source": "mcp_server",
                    "server_id": server_id,
                    "tool_id": tool_id,
                    "configuration": mcp_tool_assignment.configuration
                }

                tools.append(tool)

            logger.info(f"✅ Loaded {len(tools)} MCP tools for agent {agent.agent_id}")
            return tools

        except Exception as e:
            logger.error(f"❌ Failed to get MCP tools for agent: {e}")
            return []

    async def _get_skills_for_agent(
        self,
        agent: CustomAgent
    ) -> List[Dict[str, Any]]:
        """
        Get skills for agent based on agent's skills configuration

        Args:
            agent: Agent configuration

        Returns:
            List of skills with metadata
        """
        try:
            skills = []

            # Check if agent has skills configured
            if not agent.skills or len(agent.skills) == 0:
                logger.debug(f"Agent {agent.agent_id} has no skills configured")
                return skills

            # Import skills service
            from services.skills_service import SkillsService
            skills_service = SkillsService()

            # Get each configured skill
            for skill_assignment in agent.skills:
                if not skill_assignment.is_enabled:
                    continue

                # Get skill from skills service
                skill = skills_service.skills.get(skill_assignment.skill_name)

                if not skill:
                    logger.warning(f"Skill not found: {skill_assignment.skill_name}")
                    continue

                # Add skill with configuration
                skills.append({
                    "skill_name": skill_assignment.skill_name,
                    "display_name": skill.display_name,
                    "description": skill.description,
                    "category": skill.category,
                    "priority": skill_assignment.priority,
                    "configuration": skill_assignment.configuration,
                    "handler_function": skill.handler_function
                })

            logger.info(f"✅ Loaded {len(skills)} skills for agent {agent.agent_id}")
            return skills

        except Exception as e:
            logger.error(f"❌ Failed to get skills for agent: {e}")
            return []

    async def _initialize_memory_for_agent(
        self,
        agent: CustomAgent,
        execution_id: str
    ) -> Dict[str, Any]:
        """
        Initialize memory systems for agent based on agent's memory_config

        Args:
            agent: Agent configuration
            execution_id: Current execution ID

        Returns:
            Dictionary with memory service instances and configuration
        """
        try:
            memory_context = {
                "vector_search": None,
                "reasoning_bank": None,
                "functional_memory": None,
                "memory_orchestrator": None,
                "enabled_systems": []
            }

            # Check if agent has memory configuration
            if not agent.memory_config:
                logger.debug(f"Agent {agent.agent_id} has no memory configuration")
                return memory_context

            config = agent.memory_config

            # Initialize Vector Search if enabled
            if config.enable_vector_search:
                from services.vector_search_service import VectorSearchService
                vector_search = VectorSearchService()
                memory_context["vector_search"] = vector_search
                memory_context["enabled_systems"].append("vector_search")
                logger.info(f"✅ Vector Search enabled for agent {agent.agent_id}")

            # Initialize ReasoningBank if enabled
            if config.enable_reasoning_bank:
                from services.reasoning_bank_service import ReasoningBankService
                reasoning_bank = ReasoningBankService(
                    db=self.db.db,  # Use the MongoDB database instance
                    vector_search_service=memory_context.get("vector_search")
                )
                memory_context["reasoning_bank"] = reasoning_bank
                memory_context["enabled_systems"].append("reasoning_bank")
                logger.info(f"✅ ReasoningBank enabled for agent {agent.agent_id}")

            # Initialize Functional Memory if enabled
            if config.enable_functional_memory:
                from services.functional_memory_service import FunctionalMemoryService
                from services.llm_service import LLMService
                functional_memory = FunctionalMemoryService(
                    db=self.db,
                    llm_service=LLMService(),
                    learning_service=None
                )
                memory_context["functional_memory"] = functional_memory
                memory_context["enabled_systems"].append("functional_memory")
                logger.info(f"✅ Functional Memory enabled for agent {agent.agent_id}")

            # Initialize Memory Orchestrator if enabled
            if config.enable_memory_orchestrator:
                from services.memory_orchestrator import PAIMemoryOrchestrator
                memory_orchestrator = PAIMemoryOrchestrator()
                memory_context["memory_orchestrator"] = memory_orchestrator
                memory_context["enabled_systems"].append("memory_orchestrator")
                logger.info(f"✅ Memory Orchestrator enabled for agent {agent.agent_id}")

            logger.info(f"✅ Initialized {len(memory_context['enabled_systems'])} memory systems for agent {agent.agent_id}")
            return memory_context

        except Exception as e:
            logger.error(f"❌ Failed to initialize memory for agent: {e}")
            return {
                "vector_search": None,
                "reasoning_bank": None,
                "functional_memory": None,
                "memory_orchestrator": None,
                "enabled_systems": []
            }

    async def _inject_memory_context(
        self,
        agent: CustomAgent,
        prompt: str,
        memory_context: Dict[str, Any]
    ) -> str:
        """
        Inject relevant memories into the prompt

        Args:
            agent: Agent configuration
            prompt: User prompt
            memory_context: Memory context with initialized services

        Returns:
            Enhanced prompt with memory context
        """
        try:
            enhanced_parts = [prompt]

            # Retrieve from ReasoningBank if enabled
            if memory_context.get("reasoning_bank") and agent.memory_config:
                reasoning_bank = memory_context["reasoning_bank"]

                # Search for similar patterns
                from models.reasoning_pattern import PatternSearchQuery
                search_query = PatternSearchQuery(
                    query=prompt,
                    pattern_types=agent.memory_config.reasoning_bank_pattern_types or [],
                    limit=agent.memory_config.reasoning_bank_limit
                )

                patterns = await reasoning_bank.search_patterns(search_query)

                if patterns and len(patterns) > 0:
                    enhanced_parts.append("\n\n## Relevant Past Patterns")
                    enhanced_parts.append("Here are similar problems you've solved before:\n")

                    for i, pattern_result in enumerate(patterns[:3], 1):  # Top 3 patterns
                        pattern = pattern_result.pattern
                        enhanced_parts.append(f"\n{i}. **{pattern.title}** (Relevance: {pattern_result.relevance_score:.2f})")
                        enhanced_parts.append(f"   Problem: {pattern.problem_description}")
                        enhanced_parts.append(f"   Solution: {pattern.solution_summary}")
                        if pattern.key_insights:
                            enhanced_parts.append(f"   Key Insights: {', '.join(pattern.key_insights[:2])}")

                    enhanced_parts.append("\nConsider applying similar approaches to this task.")
                    logger.info(f"✅ Injected {len(patterns)} patterns from ReasoningBank")

            # Retrieve from Vector Search if enabled
            if memory_context.get("vector_search") and agent.memory_config:
                vector_search = memory_context["vector_search"]
                namespace = agent.memory_config.vector_search_namespace or f"agent_{agent.agent_id}"

                # Search for similar memories
                memories = await vector_search.search_memories(
                    query=prompt,
                    namespace=namespace,
                    limit=agent.memory_config.vector_search_limit
                )

                if memories and len(memories) > 0:
                    enhanced_parts.append("\n\n## Relevant Memories")
                    enhanced_parts.append("Here are relevant memories from past interactions:\n")

                    for i, memory in enumerate(memories[:3], 1):  # Top 3 memories
                        enhanced_parts.append(f"\n{i}. {memory['content'][:200]}...")
                        if memory.get('metadata'):
                            enhanced_parts.append(f"   Context: {memory['metadata']}")

                    logger.info(f"✅ Injected {len(memories)} memories from Vector Search")

            return "\n".join(enhanced_parts)

        except Exception as e:
            logger.error(f"❌ Failed to inject memory context: {e}")
            return prompt  # Return original prompt on error

    async def _store_execution_pattern(
        self,
        agent: CustomAgent,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        reasoning_bank
    ):
        """
        Store successful execution as a pattern in ReasoningBank

        Args:
            agent: Agent configuration
            input_data: Execution input
            output_data: Execution output
            reasoning_bank: ReasoningBank service instance
        """
        try:
            from models.reasoning_pattern import PatternCreateRequest, PatternContext

            # Only store if execution was successful
            if not output_data.get("response"):
                return

            # Create pattern
            pattern_request = PatternCreateRequest(
                pattern_type="agent_execution",
                title=f"Agent Execution: {agent.name}",
                problem_description=input_data.get("prompt", "No prompt provided"),
                solution_steps=[
                    {
                        "step_number": 1,
                        "description": "Agent executed task",
                        "reasoning": f"Used {output_data.get('tool_calls', 0)} tools",
                        "outcome": "Success"
                    }
                ],
                solution_summary=output_data.get("response", "")[:500],  # First 500 chars
                tags=[
                    agent.agent_type,
                    f"agent_{agent.agent_id}",
                    *agent.tags
                ],
                key_insights=[
                    f"Tools used: {output_data.get('tool_calls', 0)}",
                    f"Execution time: {output_data.get('execution_time', 0):.2f}s",
                    f"Skills: {', '.join(output_data.get('skills_used', []))}"
                ],
                context=PatternContext(
                    domain=agent.agent_type,
                    complexity="medium",
                    tools_used=output_data.get("tool_performance", {}).get("total_tools_called", 0),
                    execution_time=output_data.get("execution_time", 0)
                ),
                created_by=f"agent_{agent.agent_id}"
            )

            # Store pattern
            pattern = await reasoning_bank.create_pattern(pattern_request)
            logger.info(f"✅ Stored execution pattern: {pattern.pattern_id}")

        except Exception as e:
            logger.error(f"❌ Failed to store execution pattern: {e}")
            raise

    # ============================================================================
    # PHASE 2, TASK 2.4: COMPREHENSIVE VALIDATION METHODS
    # ============================================================================

    async def validate_dependencies(
        self,
        agent_id: Optional[str] = None,
        draft_id: Optional[str] = None
    ) -> DependencyValidationResult:
        """
        Validate all dependencies for an agent configuration.

        Phase 2, Task 2.4: Add Agent Builder Validation (4 hours)
        Checks for required tools, models, API keys, and integrations.
        """
        try:
            # Get agent or draft data
            if agent_id:
                agent = await self.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent not found: {agent_id}")
                data = agent.model_dump()
            elif draft_id:
                draft = await self.get_draft(draft_id)
                if not draft:
                    raise ValueError(f"Draft not found: {draft_id}")
                data = draft.model_dump()
            else:
                raise ValueError("Either agent_id or draft_id must be provided")

            now = Database.utcnow()

            # Track all dependencies
            all_checks: List[DependencyCheck] = []
            missing_dependencies: List[DependencyCheck] = []
            incompatible_dependencies: List[DependencyCheck] = []
            available_dependencies: List[DependencyCheck] = []
            circular_dependencies: List[str] = []
            resolution_steps: List[str] = []

            # 1. Check tool dependencies
            tools = data.get("tools", [])
            for tool_assignment in tools:
                tool_id = tool_assignment.get("tool_id") if isinstance(tool_assignment, dict) else tool_assignment.tool_id

                # Check if tool exists in tool registry
                from services.tool_registry import tool_registry
                tool_available = await tool_registry.get_tool(tool_id)

                check = DependencyCheck(
                    dependency_id=tool_id,
                    dependency_name=tool_id,
                    dependency_type=DependencyType.TOOL,
                    status=DependencyStatus.AVAILABLE if tool_available else DependencyStatus.MISSING,
                    required=True
                )

                if not tool_available:
                    check.error_message = f"Tool '{tool_id}' is not available in the tool registry"
                    check.resolution_suggestion = f"Register tool '{tool_id}' or remove it from agent configuration"
                    missing_dependencies.append(check)
                else:
                    available_dependencies.append(check)

                all_checks.append(check)

            # 2. Check MCP tool dependencies
            mcp_tools = data.get("mcp_tools", [])
            for mcp_tool_assignment in mcp_tools:
                tool_id = mcp_tool_assignment.get("tool_id") if isinstance(mcp_tool_assignment, dict) else mcp_tool_assignment.tool_id
                server_id = mcp_tool_assignment.get("server_id") if isinstance(mcp_tool_assignment, dict) else mcp_tool_assignment.server_id

                # Check if MCP server is configured
                from services.mcp_hub_service import MCPHubService
                mcp_service = MCPHubService(self.db)
                server = await mcp_service.get_server(server_id)

                check = DependencyCheck(
                    dependency_id=f"mcp_{tool_id}",
                    dependency_name=f"MCP Tool: {tool_id}",
                    dependency_type=DependencyType.TOOL,
                    status=DependencyStatus.AVAILABLE if server else DependencyStatus.MISSING,
                    required=True
                )

                if not server:
                    check.error_message = f"MCP server '{server_id}' is not configured"
                    check.resolution_suggestion = f"Configure MCP server '{server_id}' or remove MCP tool from agent"
                    missing_dependencies.append(check)
                else:
                    available_dependencies.append(check)

                all_checks.append(check)

            # 3. Check model dependencies
            configuration = data.get("configuration", {})
            llm_config = configuration.get("llm_config", {})

            if llm_config:
                provider = llm_config.get("provider", "llama_cpp")
                model_id = llm_config.get("model_id", "")

                check = DependencyCheck(
                    dependency_id=model_id,
                    dependency_name=f"LLM Model: {model_id}",
                    dependency_type=DependencyType.MODEL,
                    status=DependencyStatus.AVAILABLE,  # Assume available for now
                    required=True
                )

                # Check if API key is required for cloud providers
                if provider in ["openai", "anthropic", "google"]:
                    api_key = llm_config.get("api_key")
                    if not api_key:
                        check.status = DependencyStatus.MISSING
                        check.error_message = f"API key required for {provider} provider"
                        check.resolution_suggestion = f"Add API key for {provider} in agent configuration"
                        missing_dependencies.append(check)
                    else:
                        available_dependencies.append(check)
                else:
                    available_dependencies.append(check)

                all_checks.append(check)

            # 4. Check integration dependencies
            agent_type = data.get("agent_type")

            # CRM agent needs CRM service
            if agent_type in ["sales", "customer_support"]:
                check = DependencyCheck(
                    dependency_id="crm_service",
                    dependency_name="CRM Service",
                    dependency_type=DependencyType.INTEGRATION,
                    status=DependencyStatus.AVAILABLE,  # Assume available
                    required=True,
                    resolution_suggestion="Ensure CRM service is running and accessible"
                )
                available_dependencies.append(check)
                all_checks.append(check)

            # ATS agent needs ATS service
            if agent_type in ["research"]:  # Could be used for candidate research
                check = DependencyCheck(
                    dependency_id="ats_service",
                    dependency_name="ATS Service",
                    dependency_type=DependencyType.INTEGRATION,
                    status=DependencyStatus.AVAILABLE,  # Assume available
                    required=False,
                    resolution_suggestion="ATS service can enhance research capabilities"
                )
                available_dependencies.append(check)
                all_checks.append(check)

            # 5. Check for circular dependencies in tool chains
            # (Simplified - would need more complex graph analysis for real circular dependency detection)
            tool_ids = [t.get("tool_id") if isinstance(t, dict) else t.tool_id for t in tools]
            if len(tool_ids) != len(set(tool_ids)):
                circular_dependencies.append("Duplicate tools detected in configuration")

            # 6. Generate resolution steps
            if missing_dependencies:
                resolution_steps.append(f"Resolve {len(missing_dependencies)} missing dependencies")
                for dep in missing_dependencies[:3]:  # Show first 3
                    if dep.resolution_suggestion:
                        resolution_steps.append(f"  - {dep.resolution_suggestion}")

            if incompatible_dependencies:
                resolution_steps.append(f"Fix {len(incompatible_dependencies)} incompatible dependencies")

            if not missing_dependencies and not incompatible_dependencies:
                resolution_steps.append("All dependencies are satisfied - agent is ready")

            # Determine if all dependencies are met
            all_dependencies_met = (
                len(missing_dependencies) == 0 and
                len(incompatible_dependencies) == 0 and
                len(circular_dependencies) == 0
            )

            result = DependencyValidationResult(
                agent_id=agent_id,
                draft_id=draft_id,
                all_dependencies_met=all_dependencies_met,
                missing_dependencies=missing_dependencies,
                incompatible_dependencies=incompatible_dependencies,
                available_dependencies=available_dependencies,
                circular_dependencies=circular_dependencies,
                resolution_steps=resolution_steps,
                validated_at=now
            )

            logger.info(f"✅ Dependency validation complete: {len(available_dependencies)} available, {len(missing_dependencies)} missing")
            return result

        except Exception as e:
            logger.error(f"❌ Error validating dependencies: {e}")
            raise

    async def validate_compatibility(
        self,
        agent_id: Optional[str] = None,
        draft_id: Optional[str] = None
    ) -> CompatibilityValidationResult:
        """
        Validate compatibility between agent components.

        Phase 2, Task 2.4: Add Agent Builder Validation (4 hours)
        Checks for incompatible tool/agent type combinations, model compatibility, etc.
        """
        try:
            # Get agent or draft data
            if agent_id:
                agent = await self.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent not found: {agent_id}")
                data = agent.model_dump()
            elif draft_id:
                draft = await self.get_draft(draft_id)
                if not draft:
                    raise ValueError(f"Draft not found: {draft_id}")
                data = draft.model_dump()
            else:
                raise ValueError("Either agent_id or draft_id must be provided")

            now = Database.utcnow()

            # Track compatibility issues
            critical_issues: List[CompatibilityIssue] = []
            warnings: List[CompatibilityIssue] = []
            recommendations: List[str] = []
            compatibility_score = 100.0  # Start at perfect, deduct for issues

            agent_type = data.get("agent_type")
            capabilities = data.get("capabilities", [])
            tools = data.get("tools", [])
            configuration = data.get("configuration", {})

            # 1. Validate tool compatibility with agent type
            incompatible_tools = {
                AgentType.CODING: ["web_scraping", "social_media"],  # Coding agents don't need these
                AgentType.RESEARCH: ["code_execution", "deployment"],  # Research agents don't deploy
                AgentType.CUSTOMER_SUPPORT: ["code_generation", "data_analysis"],  # Support doesn't code
                AgentType.FINANCE: ["content_creation", "social_media"],  # Finance is analytical
            }

            if agent_type in incompatible_tools:
                for tool_assignment in tools:
                    tool_id = tool_assignment.get("tool_id") if isinstance(tool_assignment, dict) else tool_assignment.tool_id

                    for incompatible_tool in incompatible_tools[agent_type]:
                        if incompatible_tool in tool_id.lower():
                            issue = CompatibilityIssue(
                                component_a=f"Agent Type: {agent_type}",
                                component_b=f"Tool: {tool_id}",
                                issue_type="tool_agent_type",
                                severity="warning",
                                description=f"Tool '{tool_id}' may not be suitable for {agent_type} agents",
                                recommendation=f"Consider removing '{tool_id}' or changing agent type"
                            )
                            warnings.append(issue)
                            compatibility_score -= 5

            # 2. Validate model compatibility with capabilities
            llm_config = configuration.get("llm_config", {})
            if llm_config:
                model_id = llm_config.get("model_id", "")
                max_tokens = llm_config.get("max_tokens", 2000)
                context_window = llm_config.get("context_window", 8192)

                # Check if model can handle required capabilities
                if "long_context" in capabilities:
                    if context_window < 16000:
                        issue = CompatibilityIssue(
                            component_a="Capability: long_context",
                            component_b=f"Model: {model_id} (context: {context_window})",
                            issue_type="model_capability",
                            severity="error",
                            description=f"Model context window ({context_window}) is too small for long_context capability",
                            recommendation="Use a model with at least 16K context window or remove long_context capability"
                        )
                        critical_issues.append(issue)
                        compatibility_score -= 20

                if "code_generation" in capabilities:
                    if "code" not in model_id.lower() and "gpt-4" not in model_id.lower():
                        issue = CompatibilityIssue(
                            component_a="Capability: code_generation",
                            component_b=f"Model: {model_id}",
                            issue_type="model_capability",
                            severity="warning",
                            description=f"Model '{model_id}' may not be optimized for code generation",
                            recommendation="Consider using a code-specialized model like CodeLlama or GPT-4"
                        )
                        warnings.append(issue)
                        compatibility_score -= 10

                # Check token limits
                if max_tokens > context_window:
                    issue = CompatibilityIssue(
                        component_a=f"max_tokens: {max_tokens}",
                        component_b=f"context_window: {context_window}",
                        issue_type="model_configuration",
                        severity="error",
                        description="max_tokens exceeds context_window",
                        recommendation=f"Reduce max_tokens to at most {context_window}"
                    )
                    critical_issues.append(issue)
                    compatibility_score -= 15

            # 3. Validate integration compatibility
            if agent_type == AgentType.SALES:
                # Sales agents should have CRM capabilities
                if "crm_integration" not in capabilities:
                    issue = CompatibilityIssue(
                        component_a=f"Agent Type: {agent_type}",
                        component_b="Capabilities",
                        issue_type="integration_capability",
                        severity="warning",
                        description="Sales agents typically need CRM integration capability",
                        recommendation="Add 'crm_integration' to capabilities"
                    )
                    warnings.append(issue)
                    compatibility_score -= 5

            if agent_type == AgentType.CUSTOMER_SUPPORT:
                # Support agents should have communication capabilities
                if "communication" not in capabilities and "email" not in capabilities:
                    issue = CompatibilityIssue(
                        component_a=f"Agent Type: {agent_type}",
                        component_b="Capabilities",
                        issue_type="integration_capability",
                        severity="warning",
                        description="Customer support agents typically need communication capabilities",
                        recommendation="Add 'communication' or 'email' to capabilities"
                    )
                    warnings.append(issue)
                    compatibility_score -= 5

            # 4. Validate memory configuration compatibility
            memory_config = data.get("memory_config")
            if memory_config:
                if isinstance(memory_config, dict):
                    enable_vector = memory_config.get("enable_vector_search", False)
                    enable_reasoning = memory_config.get("enable_reasoning_bank", False)

                    # If both are enabled, ensure they're compatible
                    if enable_vector and enable_reasoning:
                        recommendations.append("Both vector search and reasoning bank are enabled - ensure sufficient resources")
                        compatibility_score -= 2

            # 5. Check for conflicting capabilities
            conflicting_pairs = [
                ("real_time", "batch_processing"),
                ("stateless", "memory_enabled"),
                ("simple", "complex_reasoning")
            ]

            for cap_a, cap_b in conflicting_pairs:
                if cap_a in capabilities and cap_b in capabilities:
                    issue = CompatibilityIssue(
                        component_a=f"Capability: {cap_a}",
                        component_b=f"Capability: {cap_b}",
                        issue_type="capability_conflict",
                        severity="warning",
                        description=f"Capabilities '{cap_a}' and '{cap_b}' may conflict",
                        recommendation=f"Review if both '{cap_a}' and '{cap_b}' are needed"
                    )
                    warnings.append(issue)
                    compatibility_score -= 5

            # 6. Generate recommendations
            if not critical_issues and not warnings:
                recommendations.append("All components are compatible - no issues detected")
            else:
                if critical_issues:
                    recommendations.append(f"Fix {len(critical_issues)} critical compatibility issues before deployment")
                if warnings:
                    recommendations.append(f"Review {len(warnings)} compatibility warnings for optimal performance")

            # Ensure score doesn't go below 0
            compatibility_score = max(0.0, compatibility_score)

            # Determine if compatible (no critical issues)
            is_compatible = len(critical_issues) == 0

            result = CompatibilityValidationResult(
                agent_id=agent_id,
                draft_id=draft_id,
                is_compatible=is_compatible,
                compatibility_score=round(compatibility_score, 2),
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                validated_at=now
            )

            logger.info(f"✅ Compatibility validation complete: Score={compatibility_score:.1f}, Issues={len(critical_issues)}, Warnings={len(warnings)}")
            return result

        except Exception as e:
            logger.error(f"❌ Error validating compatibility: {e}")
            raise

    async def estimate_performance(
        self,
        agent_id: Optional[str] = None,
        draft_id: Optional[str] = None
    ) -> PerformanceEstimate:
        """
        Estimate performance metrics for an agent configuration.

        Phase 2, Task 2.4: Add Agent Builder Validation (4 hours)
        Predicts token usage, response time, cost, and resource requirements.
        """
        try:
            # Get agent or draft data
            if agent_id:
                agent = await self.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent not found: {agent_id}")
                data = agent.model_dump()
            elif draft_id:
                draft = await self.get_draft(draft_id)
                if not draft:
                    raise ValueError(f"Draft not found: {draft_id}")
                data = draft.model_dump()
            else:
                raise ValueError("Either agent_id or draft_id must be provided")

            now = Database.utcnow()

            # Extract configuration
            system_prompt = data.get("system_prompt", "")
            capabilities = data.get("capabilities", [])
            tools = data.get("tools", [])
            mcp_tools = data.get("mcp_tools", [])
            configuration = data.get("configuration", {})
            llm_config = configuration.get("llm_config", {})
            memory_config = data.get("memory_config")

            # 1. Estimate token usage
            # Base tokens from system prompt
            system_prompt_tokens = len(system_prompt.split()) * 1.3  # Rough estimate: 1.3 tokens per word

            # Add tokens for tools (tool descriptions in context)
            tool_tokens = len(tools) * 100  # ~100 tokens per tool description
            mcp_tool_tokens = len(mcp_tools) * 100

            # Add tokens for capabilities
            capability_tokens = len(capabilities) * 20  # ~20 tokens per capability

            # Add tokens for memory context
            memory_tokens = 0
            if memory_config:
                if isinstance(memory_config, dict):
                    if memory_config.get("enable_vector_search"):
                        memory_tokens += memory_config.get("vector_search_limit", 10) * 200  # ~200 tokens per memory
                    if memory_config.get("enable_reasoning_bank"):
                        memory_tokens += memory_config.get("reasoning_bank_limit", 5) * 300  # ~300 tokens per pattern

            # Estimated context tokens (system + tools + capabilities + memory)
            estimated_context_tokens = int(
                system_prompt_tokens +
                tool_tokens +
                mcp_tool_tokens +
                capability_tokens +
                memory_tokens
            )

            # Estimated completion tokens (response)
            max_tokens = llm_config.get("max_tokens", 2000)
            estimated_completion_tokens = int(max_tokens * 0.7)  # Assume 70% of max is used on average

            # Total tokens per request
            estimated_tokens_per_request = estimated_context_tokens + estimated_completion_tokens
            max_tokens_per_request = estimated_context_tokens + max_tokens

            # 2. Estimate response time
            # Base response time depends on model and complexity
            provider = llm_config.get("provider", "llama_cpp")
            model_id = llm_config.get("model_id", "")

            # Base latency by provider
            base_latency_ms = {
                "llama_cpp": 2000,  # Local models are slower but no network latency
                "ollama": 1500,
                "openai": 800,
                "anthropic": 900,
                "google": 1000
            }.get(provider, 1500)

            # Add latency for token generation (rough estimate: 50ms per 100 tokens)
            generation_latency_ms = int((estimated_completion_tokens / 100) * 50)

            # Add latency for tool calls
            tool_call_latency_ms = (len(tools) + len(mcp_tools)) * 200  # ~200ms per tool

            # Add latency for memory retrieval
            memory_latency_ms = 0
            if memory_config and isinstance(memory_config, dict):
                if memory_config.get("enable_vector_search"):
                    memory_latency_ms += 300  # Vector search overhead
                if memory_config.get("enable_reasoning_bank"):
                    memory_latency_ms += 200  # Reasoning bank overhead

            estimated_response_time_ms = (
                base_latency_ms +
                generation_latency_ms +
                tool_call_latency_ms +
                memory_latency_ms
            )

            min_response_time_ms = int(estimated_response_time_ms * 0.6)  # Best case
            max_response_time_ms = int(estimated_response_time_ms * 2.0)  # Worst case

            # 3. Estimate cost (in USD)
            # Pricing per 1M tokens (approximate)
            pricing = {
                "llama_cpp": 0.0,  # Local, no API cost
                "ollama": 0.0,  # Local, no API cost
                "openai": 30.0,  # GPT-4 pricing
                "anthropic": 25.0,  # Claude pricing
                "google": 20.0  # Gemini pricing
            }

            cost_per_million_tokens = pricing.get(provider, 0.0)
            estimated_cost_per_request = (estimated_tokens_per_request / 1_000_000) * cost_per_million_tokens
            estimated_cost_per_1k_requests = estimated_cost_per_request * 1000

            # Estimate monthly cost (assume 10k requests/month for active agent)
            estimated_monthly_cost = estimated_cost_per_request * 10000

            # 4. Estimate resource requirements
            # Memory requirement based on model and context
            context_window = llm_config.get("context_window", 8192)

            # Base memory for model
            model_memory_mb = {
                "llama_cpp": 4000,  # 4GB for 7B model
                "ollama": 4000,
                "openai": 0,  # Cloud, no local memory
                "anthropic": 0,
                "google": 0
            }.get(provider, 2000)

            # Add memory for context
            context_memory_mb = int((context_window / 1000) * 10)  # ~10MB per 1K tokens

            # Add memory for tools and memory systems
            tool_memory_mb = (len(tools) + len(mcp_tools)) * 50  # ~50MB per tool
            memory_system_mb = 500 if memory_config else 0

            memory_requirement_mb = (
                model_memory_mb +
                context_memory_mb +
                tool_memory_mb +
                memory_system_mb
            )

            # CPU requirement
            if provider in ["llama_cpp", "ollama"]:
                cpu_requirement = "high"  # Local inference needs CPU
            else:
                cpu_requirement = "low"  # Cloud API, minimal CPU

            # GPU requirement
            gpu_required = provider in ["llama_cpp", "ollama"] and "large" in model_id.lower()

            # 5. Calculate complexity score
            complexity_score = 0.0

            # Factor 1: Number of tools (0-25 points)
            total_tools = len(tools) + len(mcp_tools)
            complexity_score += min(25, total_tools * 5)

            # Factor 2: Number of capabilities (0-20 points)
            complexity_score += min(20, len(capabilities) * 4)

            # Factor 3: Memory systems (0-20 points)
            if memory_config and isinstance(memory_config, dict):
                if memory_config.get("enable_vector_search"):
                    complexity_score += 10
                if memory_config.get("enable_reasoning_bank"):
                    complexity_score += 10

            # Factor 4: Context size (0-20 points)
            complexity_score += min(20, (context_window / 1000) * 2)

            # Factor 5: System prompt complexity (0-15 points)
            prompt_length = len(system_prompt)
            complexity_score += min(15, (prompt_length / 100) * 3)

            complexity_score = min(100.0, complexity_score)

            # 6. Generate optimization suggestions
            optimization_suggestions = []
            bottlenecks = []

            if estimated_tokens_per_request > 8000:
                bottlenecks.append("High token usage may slow response time")
                optimization_suggestions.append("Reduce system prompt length or limit tool descriptions")

            if total_tools > 10:
                bottlenecks.append("Large number of tools increases complexity")
                optimization_suggestions.append("Consider grouping related tools or reducing tool count")

            if provider in ["llama_cpp", "ollama"] and context_window > 16000:
                bottlenecks.append("Large context window with local model may be slow")
                optimization_suggestions.append("Consider using a cloud provider for large context windows")

            if memory_config and isinstance(memory_config, dict):
                if memory_config.get("enable_vector_search") and memory_config.get("enable_reasoning_bank"):
                    bottlenecks.append("Multiple memory systems add overhead")
                    optimization_suggestions.append("Consider using only one memory system for better performance")

            if estimated_cost_per_1k_requests > 50:
                optimization_suggestions.append(f"High cost per 1K requests (${estimated_cost_per_1k_requests:.2f}) - consider local models")

            if not optimization_suggestions:
                optimization_suggestions.append("Configuration is well-optimized")

            result = PerformanceEstimate(
                agent_id=agent_id,
                draft_id=draft_id,
                estimated_tokens_per_request=estimated_tokens_per_request,
                estimated_context_tokens=estimated_context_tokens,
                estimated_completion_tokens=estimated_completion_tokens,
                max_tokens_per_request=max_tokens_per_request,
                estimated_response_time_ms=estimated_response_time_ms,
                min_response_time_ms=min_response_time_ms,
                max_response_time_ms=max_response_time_ms,
                estimated_cost_per_request=round(estimated_cost_per_request, 6),
                estimated_cost_per_1k_requests=round(estimated_cost_per_1k_requests, 2),
                estimated_monthly_cost=round(estimated_monthly_cost, 2),
                memory_requirement_mb=memory_requirement_mb,
                cpu_requirement=cpu_requirement,
                gpu_required=gpu_required,
                complexity_score=round(complexity_score, 2),
                optimization_suggestions=optimization_suggestions,
                bottlenecks=bottlenecks,
                estimated_at=now
            )

            logger.info(f"✅ Performance estimation complete: {estimated_tokens_per_request} tokens, {estimated_response_time_ms}ms, ${estimated_cost_per_request:.6f}/request")
            return result

        except Exception as e:
            logger.error(f"❌ Error estimating performance: {e}")
            raise

    async def validate_agent_comprehensive(
        self,
        agent_id: Optional[str] = None,
        draft_id: Optional[str] = None
    ) -> ComprehensiveValidationResult:
        """
        Perform comprehensive validation combining all validation types.

        Phase 2, Task 2.4: Add Agent Builder Validation
        Aggregates configuration, dependency, compatibility, and performance validation.
        """
        try:
            now = Database.utcnow()

            logger.info(f"🔍 Starting comprehensive validation for agent_id={agent_id}, draft_id={draft_id}")

            # 1. Run configuration validation (enhanced validation)
            configuration_validation = await self.validate_agent_enhanced(
                aid=agent_id,
                draft_id=draft_id
            )

            # 2. Run dependency validation
            dependency_validation = await self.validate_dependencies(
                agent_id=agent_id,
                draft_id=draft_id
            )

            # 3. Run compatibility validation
            compatibility_validation = await self.validate_compatibility(
                agent_id=agent_id,
                draft_id=draft_id
            )

            # 4. Run performance estimation
            performance_estimate = await self.estimate_performance(
                agent_id=agent_id,
                draft_id=draft_id
            )

            # 5. Calculate overall metrics
            # Count issues
            critical_issues_count = (
                len([e for e in configuration_validation.errors if e]) +
                len(dependency_validation.missing_dependencies) +
                len(dependency_validation.incompatible_dependencies) +
                len(compatibility_validation.critical_issues)
            )

            warnings_count = (
                len(configuration_validation.warnings) +
                len(compatibility_validation.warnings)
            )

            suggestions_count = (
                len(configuration_validation.suggestions) +
                len(dependency_validation.resolution_steps) +
                len(compatibility_validation.recommendations) +
                len(performance_estimate.optimization_suggestions)
            )

            # 6. Calculate overall score (0-100)
            # Weighted average of different validation scores
            config_score = configuration_validation.completion_percentage  # 0-100
            dependency_score = 100.0 if dependency_validation.all_dependencies_met else 50.0
            compatibility_score = compatibility_validation.compatibility_score  # 0-100
            performance_score = 100.0 - performance_estimate.complexity_score  # Inverse complexity

            overall_score = (
                config_score * 0.30 +  # 30% weight
                dependency_score * 0.25 +  # 25% weight
                compatibility_score * 0.25 +  # 25% weight
                performance_score * 0.20  # 20% weight
            )

            # 7. Determine if valid and deployable
            is_valid = configuration_validation.is_valid and len(configuration_validation.errors) == 0
            is_deployable = (
                is_valid and
                dependency_validation.all_dependencies_met and
                compatibility_validation.is_compatible and
                critical_issues_count == 0
            )

            # 8. Determine deployment readiness
            if is_deployable:
                deployment_readiness = "ready"
            elif critical_issues_count > 0:
                deployment_readiness = "not_ready"
            else:
                deployment_readiness = "needs_fixes"

            # 9. Generate next steps
            next_steps = []

            if not is_valid:
                next_steps.append("Fix configuration errors before proceeding")
                for error in configuration_validation.errors[:3]:  # Show first 3
                    if isinstance(error, dict) and error.get("message"):
                        next_steps.append(f"  - {error['message']}")

            if not dependency_validation.all_dependencies_met:
                next_steps.append("Resolve missing dependencies")
                for step in dependency_validation.resolution_steps[:3]:
                    next_steps.append(f"  - {step}")

            if not compatibility_validation.is_compatible:
                next_steps.append("Fix compatibility issues")
                for issue in compatibility_validation.critical_issues[:2]:
                    next_steps.append(f"  - {issue.description}")

            if performance_estimate.bottlenecks:
                next_steps.append("Address performance bottlenecks")
                for bottleneck in performance_estimate.bottlenecks[:2]:
                    next_steps.append(f"  - {bottleneck}")

            if is_deployable:
                next_steps.append("✅ Agent is ready for deployment")
                next_steps.append("Consider testing in a staging environment first")

            # 10. Create comprehensive result
            result = ComprehensiveValidationResult(
                agent_id=agent_id,
                draft_id=draft_id,
                is_valid=is_valid,
                is_deployable=is_deployable,
                overall_score=round(overall_score, 2),
                configuration_validation=configuration_validation,
                dependency_validation=dependency_validation,
                compatibility_validation=compatibility_validation,
                performance_estimate=performance_estimate,
                critical_issues_count=critical_issues_count,
                warnings_count=warnings_count,
                suggestions_count=suggestions_count,
                deployment_readiness=deployment_readiness,
                next_steps=next_steps,
                validated_at=now
            )

            logger.info(f"✅ Comprehensive validation complete: Score={overall_score:.1f}, Deployable={is_deployable}, Issues={critical_issues_count}")
            return result

        except Exception as e:
            logger.error(f"❌ Error in comprehensive validation: {e}")
            raise
