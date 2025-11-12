"""
Workflow Execution Service

Manages workflow execution and state
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

from backend.services.event_bus import EventBus
from backend.database import get_database


class WorkflowExecutionService:
    """Service for executing workflows"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.db = get_database()
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    async def create_workflow(
        self,
        name: str,
        description: Optional[str],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> str:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow_doc = {
            "workflow_id": workflow_id,
            "name": name,
            "description": description,
            "nodes": [node.dict() if hasattr(node, 'dict') else node for node in nodes],
            "edges": [edge.dict() if hasattr(edge, 'dict') else edge for edge in edges],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": "active"
        }
        
        await self.db.workflows.insert_one(workflow_doc)
        
        # Publish workflow created event
        await self.event_bus.publish(
            event_type="workflow.created",
            data={
                "workflow_id": workflow_id,
                "name": name,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        return workflow_id
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        workflow = await self.db.workflows.find_one({"workflow_id": workflow_id})
        if workflow:
            workflow.pop("_id", None)
        return workflow
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        cursor = self.db.workflows.find({})
        workflows = await cursor.to_list(length=100)
        for workflow in workflows:
            workflow.pop("_id", None)
        return workflows
    
    async def delete_workflow(self, workflow_id: str):
        """Delete a workflow"""
        await self.db.workflows.delete_one({"workflow_id": workflow_id})
        
        # Publish workflow deleted event
        await self.event_bus.publish(
            event_type="workflow.deleted",
            data={
                "workflow_id": workflow_id,
                "deleted_at": datetime.utcnow().isoformat()
            }
        )
    
    async def start_execution(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> str:
        """Start workflow execution"""
        execution_id = str(uuid.uuid4())
        
        # Get workflow definition
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Create execution record
        execution_doc = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "running",
            "started_at": datetime.utcnow(),
            "input_data": input_data,
            "current_node": None,
            "node_statuses": {},
            "result": None,
            "error": None
        }
        
        await self.db.workflow_executions.insert_one(execution_doc)
        
        # Start execution task
        task = asyncio.create_task(
            self._execute_workflow(execution_id, workflow, input_data)
        )
        self.active_executions[execution_id] = task
        
        return execution_id
    
    async def _execute_workflow(
        self,
        execution_id: str,
        workflow: Dict[str, Any],
        input_data: Dict[str, Any]
    ):
        """Execute workflow (background task)"""
        try:
            nodes = workflow["nodes"]
            edges = workflow["edges"]
            
            # Find trigger node (starting point)
            trigger_nodes = [n for n in nodes if n["type"] == "trigger"]
            if not trigger_nodes:
                raise ValueError("No trigger node found in workflow")
            
            current_node = trigger_nodes[0]
            context = {"input": input_data, "results": {}}
            
            # Execute nodes in sequence
            while current_node:
                node_id = current_node["id"]
                
                # Update execution status
                await self._update_execution_status(
                    execution_id,
                    current_node=node_id,
                    node_status="running"
                )
                
                # Publish node execution started event
                await self.event_bus.publish(
                    event_type="workflow.node.started",
                    data={
                        "execution_id": execution_id,
                        "node_id": node_id,
                        "node_type": current_node["type"],
                        "started_at": datetime.utcnow().isoformat()
                    }
                )
                
                # Execute node
                try:
                    result = await self._execute_node(current_node, context)
                    context["results"][node_id] = result
                    
                    # Update node status
                    await self._update_execution_status(
                        execution_id,
                        node_status="completed",
                        node_id=node_id
                    )
                    
                    # Publish node execution completed event
                    await self.event_bus.publish(
                        event_type="workflow.node.completed",
                        data={
                            "execution_id": execution_id,
                            "node_id": node_id,
                            "result": result,
                            "completed_at": datetime.utcnow().isoformat()
                        }
                    )
                    
                except Exception as e:
                    # Node execution failed
                    await self._update_execution_status(
                        execution_id,
                        node_status="failed",
                        node_id=node_id,
                        error=str(e)
                    )
                    
                    # Publish node execution failed event
                    await self.event_bus.publish(
                        event_type="workflow.node.failed",
                        data={
                            "execution_id": execution_id,
                            "node_id": node_id,
                            "error": str(e),
                            "failed_at": datetime.utcnow().isoformat()
                        }
                    )
                    
                    raise
                
                # Find next node
                current_node = self._get_next_node(current_node, edges, nodes, context)
            
            # Workflow completed successfully
            await self._complete_execution(execution_id, context["results"])
            
        except Exception as e:
            # Workflow execution failed
            await self._fail_execution(execution_id, str(e))
        finally:
            # Clean up
            self.active_executions.pop(execution_id, None)
    
    async def _execute_node(
        self,
        node: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single node"""
        node_type = node["type"]
        node_data = node["data"]
        
        # Simulate node execution (replace with actual logic)
        await asyncio.sleep(1)  # Simulate work
        
        if node_type == "trigger":
            return {"triggered": True, "input": context["input"]}
        elif node_type == "action":
            return {"action": node_data.get("actionType", "unknown"), "executed": True}
        elif node_type == "condition":
            # Simple condition evaluation (replace with actual logic)
            return {"condition_met": True}
        elif node_type == "agent":
            return {"agent_response": f"Processed by {node_data.get('agentType', 'unknown')} agent"}
        else:
            return {"executed": True}
    
    def _get_next_node(
        self,
        current_node: Dict[str, Any],
        edges: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get the next node to execute"""
        # Find outgoing edges from current node
        outgoing_edges = [e for e in edges if e["source"] == current_node["id"]]
        
        if not outgoing_edges:
            return None  # End of workflow
        
        # For condition nodes, choose edge based on result
        if current_node["type"] == "condition":
            result = context["results"].get(current_node["id"], {})
            condition_met = result.get("condition_met", False)
            
            # Find the appropriate edge (true or false)
            for edge in outgoing_edges:
                if condition_met and edge.get("sourceHandle") == "true":
                    target_id = edge["target"]
                    return next((n for n in nodes if n["id"] == target_id), None)
                elif not condition_met and edge.get("sourceHandle") == "false":
                    target_id = edge["target"]
                    return next((n for n in nodes if n["id"] == target_id), None)
        
        # For other nodes, take the first outgoing edge
        if outgoing_edges:
            target_id = outgoing_edges[0]["target"]
            return next((n for n in nodes if n["id"] == target_id), None)
        
        return None
    
    async def _update_execution_status(
        self,
        execution_id: str,
        current_node: Optional[str] = None,
        node_status: Optional[str] = None,
        node_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update execution status"""
        update_data = {}
        
        if current_node:
            update_data["current_node"] = current_node
        
        if node_status and node_id:
            update_data[f"node_statuses.{node_id}"] = node_status
        
        if error:
            update_data["error"] = error
        
        if update_data:
            await self.db.workflow_executions.update_one(
                {"execution_id": execution_id},
                {"$set": update_data}
            )
    
    async def _complete_execution(self, execution_id: str, result: Dict[str, Any]):
        """Mark execution as completed"""
        await self.db.workflow_executions.update_one(
            {"execution_id": execution_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "result": result
                }
            }
        )
        
        # Publish workflow completed event
        await self.event_bus.publish(
            event_type="workflow.completed",
            data={
                "execution_id": execution_id,
                "completed_at": datetime.utcnow().isoformat(),
                "result": result
            }
        )
    
    async def _fail_execution(self, execution_id: str, error: str):
        """Mark execution as failed"""
        await self.db.workflow_executions.update_one(
            {"execution_id": execution_id},
            {
                "$set": {
                    "status": "failed",
                    "completed_at": datetime.utcnow(),
                    "error": error
                }
            }
        )
        
        # Publish workflow failed event
        await self.event_bus.publish(
            event_type="workflow.failed",
            data={
                "execution_id": execution_id,
                "failed_at": datetime.utcnow().isoformat(),
                "error": error
            }
        )
    
    async def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution by ID"""
        execution = await self.db.workflow_executions.find_one({"execution_id": execution_id})
        if execution:
            execution.pop("_id", None)
        return execution
    
    async def stop_execution(self, execution_id: str):
        """Stop a running execution"""
        task = self.active_executions.get(execution_id)
        if task:
            task.cancel()
            await self.db.workflow_executions.update_one(
                {"execution_id": execution_id},
                {
                    "$set": {
                        "status": "stopped",
                        "completed_at": datetime.utcnow()
                    }
                }
            )

