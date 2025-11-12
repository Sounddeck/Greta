"""
Intelligent Workflow Orchestration Engine

This service orchestrates complex workflows by:
1. Automatic task decomposition
2. Dynamic tool selection from Tool Registry
3. Pattern-guided execution using ReasoningBank
4. RAG-enhanced context from Vector Search

Integrates: RAG, ReasoningBank, Tool Registry, Skills System
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubtaskStatus(str, Enum):
    """Subtask execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowOrchestrator:
    """
    Intelligent Workflow Orchestration Engine
    
    Coordinates task decomposition, tool selection, pattern matching,
    and execution with RAG context.
    """
    
    def __init__(
        self,
        tool_registry=None,
        reasoning_bank=None,
        vector_search=None,
        skills_service=None,
        db=None
    ):
        """
        Initialize Workflow Orchestrator
        
        Args:
            tool_registry: Tool Registry Service
            reasoning_bank: ReasoningBank Service
            vector_search: Vector Search Service
            skills_service: Skills Service
            db: MongoDB database
        """
        self.tool_registry = tool_registry
        self.reasoning_bank = reasoning_bank
        self.vector_search = vector_search
        self.skills_service = skills_service
        self.db = db
        
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        logger.info("✅ Workflow Orchestrator initialized")
    
    async def execute_workflow(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Execute an intelligent workflow
        
        Steps:
        1. Retrieve RAG context (similar past executions)
        2. Find matching patterns from ReasoningBank
        3. Decompose task into subtasks
        4. Select appropriate tools for each subtask
        5. Execute workflow with monitoring
        6. Store results and patterns
        
        Args:
            task_description: High-level task description
            context: Additional context
            user_id: User identifier
            
        Returns:
            Workflow execution result
        """
        workflow_id = str(uuid4())
        start_time = datetime.utcnow()
        
        workflow = {
            "workflow_id": workflow_id,
            "task_description": task_description,
            "context": context or {},
            "user_id": user_id,
            "status": WorkflowStatus.PENDING,
            "start_time": start_time,
            "subtasks": [],
            "tools_used": [],
            "patterns_used": [],
            "rag_context": None,
            "result": None,
            "error": None,
            "execution_time": 0.0
        }
        
        self.active_workflows[workflow_id] = workflow
        
        try:
            # Step 1: Retrieve RAG context
            logger.info(f"🔍 Step 1: Retrieving RAG context for workflow {workflow_id}")
            workflow["status"] = WorkflowStatus.ANALYZING
            rag_context = await self._retrieve_rag_context(task_description)
            workflow["rag_context"] = rag_context
            
            # Step 2: Find matching patterns
            logger.info(f"🧠 Step 2: Finding matching patterns")
            matching_patterns = await self._find_matching_patterns(task_description, rag_context)
            workflow["patterns_used"] = matching_patterns
            
            # Step 3: Decompose task
            logger.info(f"📋 Step 3: Decomposing task into subtasks")
            workflow["status"] = WorkflowStatus.PLANNING
            subtasks = await self._decompose_task(task_description, matching_patterns, rag_context)
            workflow["subtasks"] = subtasks
            
            # Step 4: Select tools for each subtask
            logger.info(f"🔧 Step 4: Selecting tools for {len(subtasks)} subtasks")
            await self._select_tools_for_subtasks(subtasks)
            
            # Step 5: Execute workflow
            logger.info(f"▶️  Step 5: Executing workflow")
            workflow["status"] = WorkflowStatus.EXECUTING
            result = await self._execute_subtasks(workflow_id, subtasks)
            
            # Step 6: Store results
            logger.info(f"💾 Step 6: Storing workflow results")
            workflow["status"] = WorkflowStatus.COMPLETED
            workflow["result"] = result
            workflow["execution_time"] = (datetime.utcnow() - start_time).total_seconds()
            
            await self._store_workflow_results(workflow)
            
            logger.info(f"✅ Workflow {workflow_id} completed in {workflow['execution_time']:.2f}s")
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "result": result,
                "subtasks_completed": len([s for s in subtasks if s["status"] == SubtaskStatus.COMPLETED]),
                "total_subtasks": len(subtasks),
                "tools_used": workflow["tools_used"],
                "patterns_used": [p["pattern_id"] for p in matching_patterns],
                "execution_time": workflow["execution_time"]
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow {workflow_id} failed: {e}")
            workflow["status"] = WorkflowStatus.FAILED
            workflow["error"] = str(e)
            workflow["execution_time"] = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_time": workflow["execution_time"]
            }
        
        finally:
            # Move to history
            self.workflow_history.append(workflow)
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    async def _retrieve_rag_context(self, task_description: str) -> Dict[str, Any]:
        """Retrieve relevant context from Vector Search (RAG)"""
        if not self.vector_search:
            return {"memories": [], "message": "Vector Search not available"}
        
        try:
            # Search for similar past tasks
            memories = await self.vector_search.search(
                namespace="master_agent_context",
                query=task_description,
                limit=5
            )
            
            return {
                "memories": memories,
                "count": len(memories),
                "message": f"Found {len(memories)} similar past executions"
            }
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
            return {"memories": [], "error": str(e)}
    
    async def _find_matching_patterns(
        self,
        task_description: str,
        rag_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find matching reasoning patterns from ReasoningBank"""
        if not self.reasoning_bank:
            return []

        try:
            # Search for relevant patterns
            from models.reasoning_pattern import PatternSearchQuery

            search_query = PatternSearchQuery(
                query=task_description,
                limit=3,
                min_success_rate=0.7
            )

            results = await self.reasoning_bank.search_patterns(search_query)

            logger.info(f"Found {len(results)} matching patterns")

            return [
                {
                    "pattern_id": result.pattern.pattern_id,
                    "title": result.pattern.title,
                    "pattern_type": result.pattern.pattern_type,
                    "success_rate": result.pattern.success_metrics.success_rate if result.pattern.success_metrics else 1.0,
                    "solution_steps": [
                        {
                            "description": step.description,
                            "step_number": step.step_number
                        }
                        for step in result.pattern.solution_steps
                    ]
                }
                for result in results
            ]
        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
            return []

    async def _decompose_task(
        self,
        task_description: str,
        matching_patterns: List[Dict[str, Any]],
        rag_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Decompose task into executable subtasks

        Uses patterns and RAG context to guide decomposition
        """
        subtasks = []

        # If we have matching patterns, use their solution steps
        if matching_patterns:
            best_pattern = matching_patterns[0]  # Highest success rate
            logger.info(f"Using pattern '{best_pattern['title']}' for decomposition")

            for idx, step in enumerate(best_pattern.get("solution_steps", [])):
                subtasks.append({
                    "subtask_id": str(uuid4()),
                    "order": idx + 1,
                    "description": step.get("description", step) if isinstance(step, dict) else step,
                    "status": SubtaskStatus.PENDING,
                    "tools": [],
                    "result": None,
                    "error": None,
                    "execution_time": 0.0
                })
        else:
            # Fallback: Simple decomposition based on task type
            logger.info("No patterns found, using heuristic decomposition")

            # Analyze task for common patterns
            task_lower = task_description.lower()

            if "research" in task_lower or "find" in task_lower or "search" in task_lower:
                subtasks.extend([
                    {
                        "subtask_id": str(uuid4()),
                        "order": 1,
                        "description": "Gather information from available sources",
                        "status": SubtaskStatus.PENDING,
                        "tools": [],
                        "result": None
                    },
                    {
                        "subtask_id": str(uuid4()),
                        "order": 2,
                        "description": "Analyze and synthesize findings",
                        "status": SubtaskStatus.PENDING,
                        "tools": [],
                        "result": None
                    }
                ])

            if "create" in task_lower or "generate" in task_lower or "build" in task_lower:
                subtasks.append({
                    "subtask_id": str(uuid4()),
                    "order": len(subtasks) + 1,
                    "description": "Create/generate the requested output",
                    "status": SubtaskStatus.PENDING,
                    "tools": [],
                    "result": None
                })

            if "analyze" in task_lower or "process" in task_lower:
                subtasks.append({
                    "subtask_id": str(uuid4()),
                    "order": len(subtasks) + 1,
                    "description": "Analyze and process data",
                    "status": SubtaskStatus.PENDING,
                    "tools": [],
                    "result": None
                })

            # Default fallback
            if not subtasks:
                subtasks.append({
                    "subtask_id": str(uuid4()),
                    "order": 1,
                    "description": task_description,
                    "status": SubtaskStatus.PENDING,
                    "tools": [],
                    "result": None
                })

        logger.info(f"Decomposed into {len(subtasks)} subtasks")
        return subtasks

    async def _select_tools_for_subtasks(self, subtasks: List[Dict[str, Any]]):
        """Select appropriate tools for each subtask"""
        if not self.tool_registry:
            logger.warning("Tool Registry not available, skipping tool selection")
            return

        for subtask in subtasks:
            try:
                # Get all available tools
                tools = list(self.tool_registry.tools.values())

                # Simple keyword matching for tool selection
                description_lower = subtask["description"].lower()
                selected_tools = []

                for tool in tools:
                    tool_keywords = [
                        tool.name.lower(),
                        tool.description.lower(),
                        tool.category.value.lower()
                    ] + [tag.lower() for tag in tool.tags]

                    # Check if any keyword matches
                    if any(keyword in description_lower for keyword in tool_keywords):
                        selected_tools.append({
                            "tool_id": tool.tool_id,
                            "name": tool.name,
                            "category": tool.category.value
                        })

                subtask["tools"] = selected_tools[:3]  # Limit to top 3 tools
                logger.debug(f"Selected {len(subtask['tools'])} tools for subtask: {subtask['description'][:50]}")

            except Exception as e:
                logger.warning(f"Tool selection failed for subtask: {e}")
                subtask["tools"] = []

    async def _execute_subtasks(
        self,
        workflow_id: str,
        subtasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute all subtasks in order"""
        results = []

        for subtask in subtasks:
            try:
                subtask["status"] = SubtaskStatus.RUNNING
                start_time = datetime.utcnow()

                # Execute subtask (simplified - in real implementation would call tools)
                result = await self._execute_single_subtask(subtask)

                subtask["status"] = SubtaskStatus.COMPLETED
                subtask["result"] = result
                subtask["execution_time"] = (datetime.utcnow() - start_time).total_seconds()

                results.append(result)

                logger.info(f"✅ Subtask {subtask['order']} completed: {subtask['description'][:50]}")

            except Exception as e:
                logger.error(f"❌ Subtask {subtask['order']} failed: {e}")
                subtask["status"] = SubtaskStatus.FAILED
                subtask["error"] = str(e)

                # Continue with other subtasks
                results.append({"error": str(e)})

        # Synthesize final result
        return {
            "summary": f"Completed {len([s for s in subtasks if s['status'] == SubtaskStatus.COMPLETED])}/{len(subtasks)} subtasks",
            "subtask_results": results,
            "success": all(s["status"] == SubtaskStatus.COMPLETED for s in subtasks)
        }

    async def _execute_single_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask"""
        # Simplified execution - in real implementation would:
        # 1. Call selected tools
        # 2. Process tool outputs
        # 3. Combine results

        await asyncio.sleep(0.1)  # Simulate work

        return {
            "subtask_id": subtask["subtask_id"],
            "description": subtask["description"],
            "tools_used": [t["tool_id"] for t in subtask["tools"]],
            "status": "completed",
            "output": f"Executed: {subtask['description']}"
        }

    async def _store_workflow_results(self, workflow: Dict[str, Any]):
        """Store workflow results in Vector Search and ReasoningBank"""
        try:
            # Store in Vector Search for RAG
            if self.vector_search:
                await self.vector_search.add_memory(
                    namespace="master_agent_context",
                    content=f"Task: {workflow['task_description']}\nResult: {workflow['result'].get('summary', 'N/A')}",
                    metadata={
                        "workflow_id": workflow["workflow_id"],
                        "status": workflow["status"],
                        "execution_time": workflow["execution_time"],
                        "subtasks_count": len(workflow["subtasks"])
                    }
                )

            # Create pattern if successful
            if self.reasoning_bank and workflow["status"] == WorkflowStatus.COMPLETED:
                from models.reasoning_pattern import PatternCreateRequest, SolutionStep

                # Extract solution steps from subtasks
                solution_steps = [
                    SolutionStep(
                        step_number=s["order"],
                        description=s["description"]
                    )
                    for s in workflow["subtasks"]
                    if s["status"] == SubtaskStatus.COMPLETED
                ]

                pattern_request = PatternCreateRequest(
                    pattern_type="workflow",
                    title=f"Workflow: {workflow['task_description'][:50]}",
                    problem_description=workflow["task_description"],
                    solution_steps=solution_steps,
                    solution_summary=f"Automated workflow with {len(solution_steps)} steps",
                    tags=["auto-generated", "workflow"],
                    created_by="workflow_orchestrator"
                )

                await self.reasoning_bank.create_pattern(pattern_request)
                logger.info(f"Created new pattern from workflow {workflow['workflow_id']}")

        except Exception as e:
            logger.warning(f"Failed to store workflow results: {e}")

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]

        # Check history
        for workflow in self.workflow_history:
            if workflow["workflow_id"] == workflow_id:
                return workflow

        return None

    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows"""
        return list(self.active_workflows.values())

    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow history"""
        return self.workflow_history[-limit:]

