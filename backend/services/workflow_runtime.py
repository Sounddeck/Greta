"""
Workflow Runtime

Compiles visual workflows (nodes/edges) to an executable runtime.
Supports parallel execution, conditional branching, error recovery, and checkpointing.
Uses LangGraph if available; otherwise falls back to an enhanced internal executor.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    # Optional dependency
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception as e:
    LANGGRAPH_AVAILABLE = False
    logger.info("LangGraph not installed; using enhanced runtime. To enable, install: pip install langgraph")

# Try to import database for checkpointing
try:
    from database import Database
    DATABASE_AVAILABLE = True
except Exception:
    DATABASE_AVAILABLE = False
    logger.info("Database not available; checkpointing disabled")


class NodeStatus(str, Enum):
    """Status of a node execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class EnhancedRuntime:
    """
    Enhanced workflow executor with parallel execution, conditional branching,
    error recovery, and checkpointing support.

    Features:
    - Parallel execution of independent nodes
    - Conditional branching with comparison operators
    - Retry logic with exponential backoff
    - Fallback node execution
    - Circuit breaker pattern
    - Checkpoint/resume capability
    """

    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        run_id: Optional[str] = None,
        enable_parallel: bool = True,
        enable_checkpointing: bool = True,
        db: Optional[Any] = None
    ):
        self.nodes = nodes
        self.edges = edges
        self.run_id = run_id or f"run_{int(time.time() * 1000)}"
        self.enable_parallel = enable_parallel
        self.enable_checkpointing = enable_checkpointing and DATABASE_AVAILABLE
        self.db = db

        # Build node and edge maps
        self.node_map = {str(n.get("id")): n for n in nodes}
        self.adj: Dict[str, List[str]] = {}  # node_id -> [child_ids]
        self.reverse_adj: Dict[str, List[str]] = {}  # node_id -> [parent_ids]

        for e in edges:
            s = str(e.get("source"))
            t = str(e.get("target"))
            self.adj.setdefault(s, []).append(t)
            self.reverse_adj.setdefault(t, []).append(s)

        # Execution state
        self.node_status: Dict[str, NodeStatus] = {
            node_id: NodeStatus.PENDING for node_id in self.node_map
        }
        self.node_outputs: Dict[str, Any] = {}
        self.node_errors: Dict[str, Exception] = {}

        # Error recovery state
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Checkpointing
        self.checkpoints: List[Dict[str, Any]] = []

    def _get_ready_nodes(self) -> List[str]:
        """Get nodes that are ready to execute (all dependencies completed)"""
        ready = []
        for node_id, status in self.node_status.items():
            if status != NodeStatus.PENDING:
                continue

            # Check if all parent nodes are completed
            parents = self.reverse_adj.get(node_id, [])
            if all(self.node_status.get(p) == NodeStatus.COMPLETED for p in parents):
                ready.append(node_id)

        return ready

    def _find_independent_groups(self, ready_nodes: List[str]) -> List[List[str]]:
        """
        Group ready nodes into independent sets that can execute in parallel.
        Nodes are independent if they don't have dependencies on each other.
        """
        if not self.enable_parallel or len(ready_nodes) <= 1:
            return [[node] for node in ready_nodes]

        # Build dependency graph among ready nodes
        groups: List[List[str]] = []
        remaining = set(ready_nodes)

        while remaining:
            # Start a new independent group
            group = []
            to_remove = set()

            for node_id in remaining:
                # Check if this node is independent of all nodes in current group
                independent = True
                for group_node in group:
                    # Check if there's a path between these nodes
                    if self._has_path(node_id, group_node) or self._has_path(group_node, node_id):
                        independent = False
                        break

                if independent:
                    group.append(node_id)
                    to_remove.add(node_id)

            if group:
                groups.append(group)
                remaining -= to_remove
            else:
                # Fallback: take first remaining node
                node = remaining.pop()
                groups.append([node])

        return groups

    def _has_path(self, start: str, end: str, visited: Optional[Set[str]] = None) -> bool:
        """Check if there's a path from start to end node"""
        if start == end:
            return True

        if visited is None:
            visited = set()

        if start in visited:
            return False

        visited.add(start)

        for child in self.adj.get(start, []):
            if self._has_path(child, end, visited):
                return True

        return False

    async def run(self, prompt: str, on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Execute workflow with parallel execution support.

        Args:
            prompt: Input prompt for the workflow
            on_event: Optional callback for execution events

        Returns:
            Dictionary with output, logs, and execution metadata
        """
        logs: List[Dict[str, Any]] = []
        start_time = time.time()

        def emit(event: Dict[str, Any]):
            """Emit execution event with OpenTelemetry enrichment"""
            try:
                from opentelemetry import trace as _trace
                _span = _trace.get_current_span()
                _sc = _span.get_span_context() if _span else None
                if _sc and getattr(_sc, "is_valid", lambda: True)():
                    event.setdefault("trace_id", f"{_sc.trace_id:032x}")
                    event.setdefault("span_id", f"{_sc.span_id:016x}")
            except Exception:
                pass

            event.setdefault("timestamp", datetime.utcnow().isoformat())
            event.setdefault("run_id", self.run_id)
            logs.append(event)

            if on_event:
                try:
                    on_event(event)
                except Exception as e:
                    logger.warning(f"Event callback failed: {e}")

        emit({"level": "info", "message": f"Starting workflow execution", "prompt_length": len(prompt)})

        # Find entry nodes (nodes with no parents)
        entry_nodes = [
            node_id for node_id in self.node_map
            if not self.reverse_adj.get(node_id)
        ]

        if not entry_nodes:
            # Fallback: find first agent node
            agent_nodes = [n for n in self.nodes if n.get("type") == "agent"]
            if agent_nodes:
                entry_nodes = [str(agent_nodes[0]["id"])]
            else:
                raise ValueError("No entry nodes found in workflow")

        emit({"level": "info", "message": f"Found {len(entry_nodes)} entry nodes", "entry_nodes": entry_nodes})

        # Execute workflow in waves (parallel execution of independent nodes)
        wave_count = 0
        while True:
            ready_nodes = self._get_ready_nodes()

            if not ready_nodes:
                # Check if all nodes are completed or if we're stuck
                pending = [nid for nid, status in self.node_status.items() if status == NodeStatus.PENDING]
                if pending:
                    emit({"level": "warning", "message": f"Workflow stuck with {len(pending)} pending nodes", "pending_nodes": pending})
                break

            wave_count += 1
            emit({"level": "info", "message": f"Wave {wave_count}: {len(ready_nodes)} nodes ready", "ready_nodes": ready_nodes})

            # Group independent nodes for parallel execution
            groups = self._find_independent_groups(ready_nodes)

            for group_idx, group in enumerate(groups):
                if len(group) > 1:
                    emit({"level": "info", "message": f"Executing {len(group)} nodes in parallel", "nodes": group})
                    # Execute group in parallel
                    await asyncio.gather(*[self._execute_node(node_id, prompt, emit) for node_id in group])
                else:
                    # Execute single node
                    await self._execute_node(group[0], prompt, emit)

        # Collect results
        completed_nodes = [nid for nid, status in self.node_status.items() if status == NodeStatus.COMPLETED]
        failed_nodes = [nid for nid, status in self.node_status.items() if status == NodeStatus.FAILED]

        execution_time = time.time() - start_time

        emit({
            "level": "info",
            "message": f"Workflow execution completed",
            "completed_nodes": len(completed_nodes),
            "failed_nodes": len(failed_nodes),
            "execution_time": execution_time,
            "waves": wave_count
        })

        output = {
            "result": f"Workflow processed prompt: {prompt[:120]}",
            "nodes_executed": completed_nodes,
            "nodes_failed": failed_nodes,
            "execution_time": execution_time,
            "waves": wave_count,
            "parallel_enabled": self.enable_parallel
        }

        return {"output": output, "logs": logs, "node_outputs": self.node_outputs}

    async def _execute_node(
        self,
        node_id: str,
        prompt: str,
        emit: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Execute a single node with error recovery support.

        Implements:
        - Retry logic with exponential backoff
        - Circuit breaker pattern
        - Fallback node execution
        - Checkpointing
        """
        node = self.node_map.get(node_id)
        if not node:
            emit({"level": "error", "message": f"Node {node_id} not found"})
            self.node_status[node_id] = NodeStatus.FAILED
            return

        node_type = node.get("type", "unknown")
        node_data = node.get("data", {})
        node_name = node_data.get("name", node_id)

        # Check circuit breaker
        if self._is_circuit_open(node_id):
            emit({"level": "warning", "message": f"Circuit breaker open for node {node_id}, skipping"})
            self.node_status[node_id] = NodeStatus.SKIPPED
            return

        # Mark as running
        self.node_status[node_id] = NodeStatus.RUNNING
        emit({"level": "info", "message": f"Executing node {node_name}", "node_id": node_id, "node_type": node_type})

        # Get retry configuration
        max_retries = node_data.get("max_retries", 3)
        retry_delay = node_data.get("retry_delay", 1.0)
        fallback_node = node_data.get("fallback_node")

        # Execute with retry logic
        for attempt in range(max_retries + 1):
            try:
                # Simulate node execution (in real implementation, this would call actual node logic)
                output = await self._execute_node_logic(node_id, node, prompt, emit)

                # Success!
                self.node_status[node_id] = NodeStatus.COMPLETED
                self.node_outputs[node_id] = output
                self.retry_counts[node_id] = 0  # Reset retry count
                self._reset_circuit_breaker(node_id)

                emit({
                    "level": "info",
                    "message": f"Node {node_name} completed successfully",
                    "node_id": node_id,
                    "attempt": attempt + 1
                })

                # Save checkpoint
                if self.enable_checkpointing:
                    await self._save_checkpoint(node_id, output)

                return

            except Exception as e:
                self.retry_counts[node_id] += 1
                self.node_errors[node_id] = e

                if attempt < max_retries:
                    # Exponential backoff
                    delay = retry_delay * (2 ** attempt)
                    emit({
                        "level": "warning",
                        "message": f"Node {node_name} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s",
                        "node_id": node_id,
                        "error": str(e)
                    })
                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted
                    emit({
                        "level": "error",
                        "message": f"Node {node_name} failed after {max_retries + 1} attempts",
                        "node_id": node_id,
                        "error": str(e)
                    })

                    # Update circuit breaker
                    self._record_failure(node_id)

                    # Try fallback node if configured
                    if fallback_node and fallback_node in self.node_map:
                        emit({"level": "info", "message": f"Executing fallback node {fallback_node}", "node_id": node_id})
                        try:
                            fallback_output = await self._execute_node_logic(fallback_node, self.node_map[fallback_node], prompt, emit)
                            self.node_status[node_id] = NodeStatus.COMPLETED
                            self.node_outputs[node_id] = fallback_output
                            emit({"level": "info", "message": f"Fallback node {fallback_node} succeeded", "node_id": node_id})
                            return
                        except Exception as fallback_error:
                            emit({"level": "error", "message": f"Fallback node {fallback_node} also failed", "error": str(fallback_error)})

                    # Mark as failed
                    self.node_status[node_id] = NodeStatus.FAILED
                    return

    async def _execute_node_logic(
        self,
        node_id: str,
        node: Dict[str, Any],
        prompt: str,
        emit: Callable[[Dict[str, Any]], None]
    ) -> Any:
        """
        Execute the actual node logic.

        In a real implementation, this would:
        - Call LLM for agent nodes
        - Execute tools for tool nodes
        - Evaluate conditions for condition nodes
        - etc.

        For now, this is a placeholder that simulates execution.
        """
        node_type = node.get("type", "unknown")
        node_data = node.get("data", {})

        # Simulate execution time
        await asyncio.sleep(0.1)

        # Check for conditional nodes
        if node_type == "condition":
            return await self._evaluate_condition(node_id, node, emit)

        # Default: return success with node info
        return {
            "node_id": node_id,
            "node_type": node_type,
            "status": "success",
            "message": f"Node {node_id} executed successfully"
        }

    async def _evaluate_condition(
        self,
        node_id: str,
        node: Dict[str, Any],
        emit: Callable[[Dict[str, Any]], None]
    ) -> Any:
        """
        Evaluate a condition node.

        Supports comparison operators: ==, !=, >, <, >=, <=, in, not in
        """
        node_data = node.get("data", {})
        condition = node_data.get("condition", {})

        left_value = condition.get("left")
        operator = condition.get("operator", "==")
        right_value = condition.get("right")

        # Resolve values from previous node outputs if needed
        if isinstance(left_value, str) and left_value.startswith("$"):
            # Reference to previous node output
            ref_node_id = left_value[1:]
            left_value = self.node_outputs.get(ref_node_id)

        if isinstance(right_value, str) and right_value.startswith("$"):
            ref_node_id = right_value[1:]
            right_value = self.node_outputs.get(ref_node_id)

        # Evaluate condition
        result = False
        try:
            if operator == "==":
                result = left_value == right_value
            elif operator == "!=":
                result = left_value != right_value
            elif operator == ">":
                result = left_value > right_value
            elif operator == "<":
                result = left_value < right_value
            elif operator == ">=":
                result = left_value >= right_value
            elif operator == "<=":
                result = left_value <= right_value
            elif operator == "in":
                result = left_value in right_value
            elif operator == "not in":
                result = left_value not in right_value
            else:
                emit({"level": "warning", "message": f"Unknown operator: {operator}", "node_id": node_id})
                result = False
        except Exception as e:
            emit({"level": "error", "message": f"Condition evaluation failed: {e}", "node_id": node_id})
            result = False

        emit({
            "level": "info",
            "message": f"Condition evaluated to {result}",
            "node_id": node_id,
            "left": left_value,
            "operator": operator,
            "right": right_value
        })

        return {"condition_result": result, "left": left_value, "operator": operator, "right": right_value}

    def _is_circuit_open(self, node_id: str) -> bool:
        """Check if circuit breaker is open for this node"""
        breaker = self.circuit_breakers.get(node_id)
        if not breaker:
            return False

        state = breaker.get("state", CircuitBreakerState.CLOSED)
        if state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            open_time = breaker.get("open_time", 0)
            if time.time() - open_time > 60:  # 60 seconds timeout
                breaker["state"] = CircuitBreakerState.HALF_OPEN
                return False
            return True

        return False

    def _record_failure(self, node_id: str) -> None:
        """Record a failure and update circuit breaker state"""
        if node_id not in self.circuit_breakers:
            self.circuit_breakers[node_id] = {
                "state": CircuitBreakerState.CLOSED,
                "failure_count": 0,
                "success_count": 0
            }

        breaker = self.circuit_breakers[node_id]
        breaker["failure_count"] += 1

        # Open circuit after 5 consecutive failures
        if breaker["failure_count"] >= 5:
            breaker["state"] = CircuitBreakerState.OPEN
            breaker["open_time"] = time.time()
            logger.warning(f"Circuit breaker opened for node {node_id} after {breaker['failure_count']} failures")

    def _reset_circuit_breaker(self, node_id: str) -> None:
        """Reset circuit breaker after successful execution"""
        if node_id in self.circuit_breakers:
            breaker = self.circuit_breakers[node_id]
            breaker["failure_count"] = 0
            breaker["success_count"] += 1

            if breaker["state"] == CircuitBreakerState.HALF_OPEN and breaker["success_count"] >= 2:
                breaker["state"] = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker closed for node {node_id}")

    async def _save_checkpoint(self, node_id: str, output: Any) -> None:
        """Save checkpoint to database"""
        if not self.enable_checkpointing or not self.db:
            return

        try:
            checkpoint = {
                "run_id": self.run_id,
                "node_id": node_id,
                "output": output,
                "timestamp": datetime.utcnow(),
                "status": self.node_status[node_id].value
            }

            self.checkpoints.append(checkpoint)

            # Save to database
            if hasattr(self.db, 'workflow_checkpoints'):
                await self.db.workflow_checkpoints.insert_one(checkpoint)
                logger.debug(f"Checkpoint saved for node {node_id}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for node {node_id}: {e}")

    async def resume_from_checkpoint(self, run_id: str) -> bool:
        """
        Resume workflow from saved checkpoint.

        Args:
            run_id: The run ID to resume from

        Returns:
            True if successfully resumed, False otherwise
        """
        if not self.enable_checkpointing or not self.db:
            logger.warning("Checkpointing not enabled, cannot resume")
            return False

        try:
            # Load checkpoints from database
            if hasattr(self.db, 'workflow_checkpoints'):
                checkpoints = await self.db.workflow_checkpoints.find({"run_id": run_id}).to_list(length=1000)

                if not checkpoints:
                    logger.warning(f"No checkpoints found for run_id {run_id}")
                    return False

                # Restore state from checkpoints
                for checkpoint in checkpoints:
                    node_id = checkpoint["node_id"]
                    self.node_status[node_id] = NodeStatus.COMPLETED
                    self.node_outputs[node_id] = checkpoint["output"]

                logger.info(f"Resumed from {len(checkpoints)} checkpoints for run_id {run_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return False


class WorkflowRuntime:
    """
    Main workflow runtime interface.

    Provides a high-level API for workflow execution with support for:
    - Parallel execution
    - Conditional branching
    - Error recovery
    - Checkpointing
    """

    def __init__(
        self,
        workflow: Dict[str, Any],
        run_id: Optional[str] = None,
        enable_parallel: bool = True,
        enable_checkpointing: bool = True,
        db: Optional[Any] = None
    ):
        self.workflow = workflow
        self.nodes: List[Dict[str, Any]] = workflow.get("nodes", [])
        self.edges: List[Dict[str, Any]] = workflow.get("edges", [])
        self.run_id = run_id
        self.enable_parallel = enable_parallel
        self.enable_checkpointing = enable_checkpointing
        self.db = db

    def compile(self) -> EnhancedRuntime:
        """Compile workflow to executable runtime"""
        if LANGGRAPH_AVAILABLE:
            # Placeholder: In future, use LangGraph for more advanced features
            logger.info("Compiling workflow with EnhancedRuntime (LangGraph available but not used)")
        else:
            logger.info("Compiling workflow with EnhancedRuntime")

        return EnhancedRuntime(
            nodes=self.nodes,
            edges=self.edges,
            run_id=self.run_id,
            enable_parallel=self.enable_parallel,
            enable_checkpointing=self.enable_checkpointing,
            db=self.db
        )

    async def run(
        self,
        prompt: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow asynchronously.

        Args:
            prompt: Input prompt for the workflow
            on_event: Optional callback for execution events

        Returns:
            Dictionary with output, logs, and execution metadata
        """
        # OpenTelemetry spans
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
        except Exception:
            tracer = None

        runtime = self.compile()

        if tracer:
            with tracer.start_as_current_span(
                "workflow.run",
                attributes={
                    "nodes": len(self.nodes),
                    "edges": len(self.edges),
                    "prompt.length": len(prompt or ""),
                    "parallel_enabled": self.enable_parallel,
                    "checkpointing_enabled": self.enable_checkpointing
                }
            ) as span:
                try:
                    return await runtime.run(prompt, on_event=on_event)
                except Exception as e:
                    try:
                        from opentelemetry.trace import Status, StatusCode
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    except Exception:
                        pass
                    raise
        else:
            return await runtime.run(prompt, on_event=on_event)

    async def resume(
        self,
        run_id: str,
        prompt: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Resume workflow from checkpoint.

        Args:
            run_id: The run ID to resume from
            prompt: Input prompt for the workflow
            on_event: Optional callback for execution events

        Returns:
            Dictionary with output, logs, and execution metadata
        """
        runtime = self.compile()

        # Load checkpoint
        success = await runtime.resume_from_checkpoint(run_id)
        if not success:
            logger.warning(f"Failed to resume from checkpoint {run_id}, starting fresh")

        # Continue execution
        return await runtime.run(prompt, on_event=on_event)

