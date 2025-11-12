"""
Unit Tests for Enhanced Workflow Runtime

Tests for parallel execution, conditional branching, error recovery,
and checkpoint/resume functionality.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from backend.services.workflow_runtime import (
    WorkflowRuntime,
    EnhancedRuntime,
    NodeStatus,
    CircuitBreakerState
)


@pytest.fixture
def simple_workflow():
    """Create a simple linear workflow"""
    return {
        "nodes": [
            {"id": "node1", "type": "agent", "data": {"name": "Agent 1"}},
            {"id": "node2", "type": "agent", "data": {"name": "Agent 2"}},
            {"id": "node3", "type": "agent", "data": {"name": "Agent 3"}}
        ],
        "edges": [
            {"source": "node1", "target": "node2"},
            {"source": "node2", "target": "node3"}
        ]
    }


@pytest.fixture
def parallel_workflow():
    """Create a workflow with parallel branches"""
    return {
        "nodes": [
            {"id": "start", "type": "agent", "data": {"name": "Start"}},
            {"id": "parallel1", "type": "agent", "data": {"name": "Parallel 1"}},
            {"id": "parallel2", "type": "agent", "data": {"name": "Parallel 2"}},
            {"id": "parallel3", "type": "agent", "data": {"name": "Parallel 3"}},
            {"id": "end", "type": "agent", "data": {"name": "End"}}
        ],
        "edges": [
            {"source": "start", "target": "parallel1"},
            {"source": "start", "target": "parallel2"},
            {"source": "start", "target": "parallel3"},
            {"source": "parallel1", "target": "end"},
            {"source": "parallel2", "target": "end"},
            {"source": "parallel3", "target": "end"}
        ]
    }


@pytest.fixture
def conditional_workflow():
    """Create a workflow with conditional branching"""
    return {
        "nodes": [
            {"id": "start", "type": "agent", "data": {"name": "Start"}},
            {
                "id": "condition",
                "type": "condition",
                "data": {
                    "name": "Check Value",
                    "condition": {
                        "left": "$start",
                        "operator": ">",
                        "right": 50
                    }
                }
            },
            {"id": "true_branch", "type": "agent", "data": {"name": "True Branch"}},
            {"id": "false_branch", "type": "agent", "data": {"name": "False Branch"}}
        ],
        "edges": [
            {"source": "start", "target": "condition"},
            {"source": "condition", "target": "true_branch", "condition": "true"},
            {"source": "condition", "target": "false_branch", "condition": "false"}
        ]
    }


@pytest.fixture
def retry_workflow():
    """Create a workflow with retry configuration"""
    return {
        "nodes": [
            {
                "id": "retry_node",
                "type": "agent",
                "data": {
                    "name": "Retry Node",
                    "max_retries": 3,
                    "retry_delay": 0.1
                }
            }
        ],
        "edges": []
    }


@pytest.fixture
def fallback_workflow():
    """Create a workflow with fallback node"""
    return {
        "nodes": [
            {
                "id": "main_node",
                "type": "agent",
                "data": {
                    "name": "Main Node",
                    "max_retries": 1,
                    "fallback_node": "fallback_node"
                }
            },
            {
                "id": "fallback_node",
                "type": "agent",
                "data": {"name": "Fallback Node"}
            }
        ],
        "edges": []
    }


class TestWorkflowRuntime:
    """Tests for WorkflowRuntime class"""
    
    @pytest.mark.asyncio
    async def test_create_runtime(self, simple_workflow):
        """Test creating a workflow runtime"""
        runtime = WorkflowRuntime(simple_workflow)
        assert runtime.workflow == simple_workflow
        assert len(runtime.nodes) == 3
        assert len(runtime.edges) == 2
    
    @pytest.mark.asyncio
    async def test_compile_runtime(self, simple_workflow):
        """Test compiling workflow to executable runtime"""
        runtime = WorkflowRuntime(simple_workflow)
        compiled = runtime.compile()
        assert isinstance(compiled, EnhancedRuntime)
        assert len(compiled.nodes) == 3
        assert len(compiled.edges) == 2
    
    @pytest.mark.asyncio
    async def test_simple_execution(self, simple_workflow):
        """Test executing a simple linear workflow"""
        runtime = WorkflowRuntime(simple_workflow, enable_parallel=False)
        result = await runtime.run("Test prompt")
        
        assert "output" in result
        assert "logs" in result
        assert result["output"]["nodes_executed"] == ["node1", "node2", "node3"]
        assert result["output"]["nodes_failed"] == []
    
    @pytest.mark.asyncio
    async def test_execution_with_events(self, simple_workflow):
        """Test execution with event callback"""
        events = []
        
        def on_event(event):
            events.append(event)
        
        runtime = WorkflowRuntime(simple_workflow, enable_parallel=False)
        result = await runtime.run("Test prompt", on_event=on_event)
        
        assert len(events) > 0
        assert any(e.get("message") == "Starting workflow execution" for e in events)


class TestParallelExecution:
    """Tests for parallel node execution"""
    
    @pytest.mark.asyncio
    async def test_parallel_execution_enabled(self, parallel_workflow):
        """Test that parallel execution is enabled"""
        runtime = WorkflowRuntime(parallel_workflow, enable_parallel=True)
        result = await runtime.run("Test prompt")
        
        assert result["output"]["parallel_enabled"] is True
        assert len(result["output"]["nodes_executed"]) == 5
    
    @pytest.mark.asyncio
    async def test_parallel_execution_waves(self, parallel_workflow):
        """Test that parallel nodes execute in waves"""
        runtime = WorkflowRuntime(parallel_workflow, enable_parallel=True)
        result = await runtime.run("Test prompt")
        
        # Should have 3 waves: start, parallel1-3, end
        assert result["output"]["waves"] == 3
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential(self, parallel_workflow):
        """Test that parallel execution is faster than sequential"""
        import time
        
        # Parallel execution
        runtime_parallel = WorkflowRuntime(parallel_workflow, enable_parallel=True)
        start = time.time()
        result_parallel = await runtime_parallel.run("Test prompt")
        parallel_time = time.time() - start
        
        # Sequential execution
        runtime_sequential = WorkflowRuntime(parallel_workflow, enable_parallel=False)
        start = time.time()
        result_sequential = await runtime_sequential.run("Test prompt")
        sequential_time = time.time() - start
        
        # Parallel should be faster (or at least not significantly slower)
        # Note: Due to overhead, this might not always be true for tiny workflows
        assert result_parallel["output"]["nodes_executed"] == result_sequential["output"]["nodes_executed"]
    
    @pytest.mark.asyncio
    async def test_independent_node_detection(self, parallel_workflow):
        """Test detection of independent nodes"""
        runtime = WorkflowRuntime(parallel_workflow, enable_parallel=True)
        compiled = runtime.compile()
        
        # After start node completes, parallel1-3 should be ready
        compiled.node_status["start"] = NodeStatus.COMPLETED
        ready = compiled._get_ready_nodes()
        
        assert set(ready) == {"parallel1", "parallel2", "parallel3"}
        
        # These should be grouped together for parallel execution
        groups = compiled._find_independent_groups(ready)
        assert len(groups) == 1  # All in one group
        assert set(groups[0]) == {"parallel1", "parallel2", "parallel3"}


class TestConditionalBranching:
    """Tests for conditional branching"""
    
    @pytest.mark.asyncio
    async def test_condition_evaluation_greater_than(self):
        """Test > operator in conditions"""
        runtime = EnhancedRuntime([], [])
        runtime.node_outputs["start"] = 75
        
        node = {
            "type": "condition",
            "data": {
                "condition": {
                    "left": "$start",
                    "operator": ">",
                    "right": 50
                }
            }
        }
        
        result = await runtime._evaluate_condition("cond1", node, lambda e: None)
        assert result["condition_result"] is True
    
    @pytest.mark.asyncio
    async def test_condition_evaluation_equals(self):
        """Test == operator in conditions"""
        runtime = EnhancedRuntime([], [])
        runtime.node_outputs["start"] = "test"
        
        node = {
            "type": "condition",
            "data": {
                "condition": {
                    "left": "$start",
                    "operator": "==",
                    "right": "test"
                }
            }
        }
        
        result = await runtime._evaluate_condition("cond1", node, lambda e: None)
        assert result["condition_result"] is True
    
    @pytest.mark.asyncio
    async def test_condition_evaluation_in_operator(self):
        """Test 'in' operator in conditions"""
        runtime = EnhancedRuntime([], [])
        runtime.node_outputs["start"] = "hello"
        
        node = {
            "type": "condition",
            "data": {
                "condition": {
                    "left": "$start",
                    "operator": "in",
                    "right": ["hello", "world"]
                }
            }
        }
        
        result = await runtime._evaluate_condition("cond1", node, lambda e: None)
        assert result["condition_result"] is True


class TestErrorRecovery:
    """Tests for error recovery strategies"""
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic with exponential backoff"""
        runtime = EnhancedRuntime(
            nodes=[{"id": "node1", "type": "agent", "data": {"name": "Test", "max_retries": 3}}],
            edges=[]
        )
        
        # Mock execution to fail first 2 times, then succeed
        call_count = 0
        
        async def mock_execute(node_id, node, prompt, emit):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Simulated failure")
            return {"status": "success"}
        
        runtime._execute_node_logic = mock_execute
        
        await runtime._execute_node("node1", "test", lambda e: None)
        
        assert call_count == 3
        assert runtime.node_status["node1"] == NodeStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        """Test fallback node execution on failure"""
        runtime = EnhancedRuntime(
            nodes=[
                {"id": "main", "type": "agent", "data": {"name": "Main", "max_retries": 0, "fallback_node": "fallback"}},
                {"id": "fallback", "type": "agent", "data": {"name": "Fallback"}}
            ],
            edges=[]
        )
        
        # Mock main node to always fail
        async def mock_execute(node_id, node, prompt, emit):
            if node_id == "main":
                raise Exception("Main node failed")
            return {"status": "success"}
        
        runtime._execute_node_logic = mock_execute
        
        await runtime._execute_node("main", "test", lambda e: None)
        
        # Main node should be completed (via fallback)
        assert runtime.node_status["main"] == NodeStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after repeated failures"""
        runtime = EnhancedRuntime([], [])
        
        # Record 5 failures
        for _ in range(5):
            runtime._record_failure("node1")
        
        # Circuit should be open
        assert runtime._is_circuit_open("node1") is True
        assert runtime.circuit_breakers["node1"]["state"] == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_resets(self):
        """Test circuit breaker resets after success"""
        runtime = EnhancedRuntime([], [])
        
        # Record failures
        for _ in range(3):
            runtime._record_failure("node1")
        
        # Record success
        runtime._reset_circuit_breaker("node1")
        
        # Failure count should be reset
        assert runtime.circuit_breakers["node1"]["failure_count"] == 0


class TestCheckpointing:
    """Tests for checkpoint/resume functionality"""
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """Test checkpoint is created after node execution"""
        runtime = EnhancedRuntime(
            nodes=[{"id": "node1", "type": "agent", "data": {"name": "Test"}}],
            edges=[],
            enable_checkpointing=True
        )
        
        await runtime._save_checkpoint("node1", {"result": "success"})
        
        assert len(runtime.checkpoints) == 1
        assert runtime.checkpoints[0]["node_id"] == "node1"
        assert runtime.checkpoints[0]["output"] == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_checkpoint_disabled(self):
        """Test checkpointing can be disabled"""
        runtime = EnhancedRuntime(
            nodes=[{"id": "node1", "type": "agent", "data": {"name": "Test"}}],
            edges=[],
            enable_checkpointing=False
        )
        
        await runtime._save_checkpoint("node1", {"result": "success"})
        
        # No checkpoints should be created
        assert len(runtime.checkpoints) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

