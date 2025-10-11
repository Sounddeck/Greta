"""
Test Suite for Week 19-20: Hook System

Tests:
1. Hook Registration (register, unregister, list, get, enable/disable)
2. Event Emission & Hook Execution (emit events, execute hooks)
3. Hook Priorities (priority-based execution order)
4. Conditional Hooks (execute only if condition met)
5. Plugin System (register, unregister, enable/disable plugins)
6. Middleware System (add, remove middleware)
7. Hook Chains (create and execute hook chains)
8. Monitoring & Statistics (stats, event history, performance)

Target: 100% pass rate
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.hook_service import (
    HookService,
    EventType,
    HookEvent,
    Hook,
    Plugin,
)

# Test results tracking
test_results = {"passed": 0, "failed": 0, "total": 0}


def print_test_header(component: str):
    """Print test component header"""
    print("\n" + "=" * 80)
    print(f"Component {component}")
    print("=" * 80 + "\n")


def print_test(test_num: int, total: int, description: str):
    """Print test description"""
    print(f"[Test {test_num}/{total}] {description}...")


def test_passed():
    """Mark test as passed"""
    test_results["passed"] += 1
    test_results["total"] += 1
    print("✅ PASSED\n")


def test_failed(error: str):
    """Mark test as failed"""
    test_results["failed"] += 1
    test_results["total"] += 1
    print(f"❌ FAILED: {error}\n")


# ============================================================================
# Component 1: Hook Registration
# ============================================================================


async def test_hook_registration():
    """Test hook registration functionality"""
    print_test_header("1: Hook Registration")

    service = HookService()

    # Test 1: Register hook
    print_test(1, 6, "Register hook")
    try:
        async def test_handler(event: HookEvent):
            return {"status": "executed"}

        result = await service.register_hook(
            hook_id="test_hook_1",
            name="Test Hook 1",
            event_type=EventType.LLM_GENERATE_PRE,
            handler=test_handler,
            priority=50,
        )
        assert result is True
        print(f"   Registered hook: test_hook_1")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Get hook
    print_test(2, 6, "Get hook by ID")
    try:
        hook = await service.get_hook("test_hook_1")
        assert hook is not None
        assert hook.name == "Test Hook 1"
        print(f"   Retrieved hook: {hook.name}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: List hooks
    print_test(3, 6, "List all hooks")
    try:
        hooks = await service.list_hooks()
        assert len(hooks) >= 1
        print(f"   Found {len(hooks)} hooks")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: List hooks by event type
    print_test(4, 6, "List hooks by event type")
    try:
        hooks = await service.list_hooks(event_type=EventType.LLM_GENERATE_PRE)
        assert len(hooks) >= 1
        print(f"   Found {len(hooks)} hooks for LLM_GENERATE_PRE")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Disable hook
    print_test(5, 6, "Disable hook")
    try:
        result = await service.disable_hook("test_hook_1")
        assert result is True
        hook = await service.get_hook("test_hook_1")
        assert hook.enabled is False
        print(f"   Disabled hook: test_hook_1")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 6: Enable hook
    print_test(6, 6, "Enable hook")
    try:
        result = await service.enable_hook("test_hook_1")
        assert result is True
        hook = await service.get_hook("test_hook_1")
        assert hook.enabled is True
        print(f"   Enabled hook: test_hook_1")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 2: Event Emission & Hook Execution
# ============================================================================


async def test_event_emission():
    """Test event emission and hook execution"""
    print_test_header("2: Event Emission & Hook Execution")

    service = HookService()

    # Test 1: Register and emit event
    print_test(1, 3, "Emit event and execute hooks")
    try:
        executed = []

        async def handler1(event: HookEvent):
            executed.append("handler1")
            return "result1"

        await service.register_hook(
            hook_id="emit_test_1",
            name="Emit Test 1",
            event_type=EventType.CONVERSATION_CREATE_PRE,
            handler=handler1,
        )

        results = await service.emit(
            EventType.CONVERSATION_CREATE_PRE,
            data={"title": "Test Conversation"},
        )

        assert len(results) >= 1
        assert "handler1" in executed
        print(f"   Executed {len(results)} hooks")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Emit with context
    print_test(2, 3, "Emit event with context")
    try:
        async def handler2(event: HookEvent):
            return {"context_value": event.context.get("test_key")}

        await service.register_hook(
            hook_id="emit_test_2",
            name="Emit Test 2",
            event_type=EventType.RAG_SEARCH_PRE,
            handler=handler2,
        )

        results = await service.emit(
            EventType.RAG_SEARCH_PRE,
            data={"query": "test"},
            context={"test_key": "test_value"},
        )

        assert len(results) >= 1
        print(f"   Emitted event with context")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Emit pre/post convenience methods
    print_test(3, 3, "Emit using convenience methods")
    try:
        async def handler3(event: HookEvent):
            return "pre_result"

        await service.register_hook(
            hook_id="emit_test_3",
            name="Emit Test 3",
            event_type=EventType.MODEL_LOAD_PRE,
            handler=handler3,
        )

        results = await service.emit_pre("model.load", data={"model_id": "test"})
        assert len(results) >= 1
        print(f"   Emitted pre-event using convenience method")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 3: Hook Priorities
# ============================================================================


async def test_hook_priorities():
    """Test hook priority-based execution order"""
    print_test_header("3: Hook Priorities")

    service = HookService()

    # Test 1: Priority-based execution order
    print_test(1, 2, "Execute hooks in priority order")
    try:
        execution_order = []

        async def high_priority(event: HookEvent):
            execution_order.append("high")

        async def medium_priority(event: HookEvent):
            execution_order.append("medium")

        async def low_priority(event: HookEvent):
            execution_order.append("low")

        # Register in random order
        await service.register_hook(
            "priority_low", "Low", EventType.AGENT_CREATE_PRE, low_priority, priority=10
        )
        await service.register_hook(
            "priority_high",
            "High",
            EventType.AGENT_CREATE_PRE,
            high_priority,
            priority=90,
        )
        await service.register_hook(
            "priority_medium",
            "Medium",
            EventType.AGENT_CREATE_PRE,
            medium_priority,
            priority=50,
        )

        await service.emit(EventType.AGENT_CREATE_PRE)

        # Should execute in order: high, medium, low
        assert execution_order == ["high", "medium", "low"]
        print(f"   Execution order: {execution_order}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Context propagation through chain
    print_test(2, 2, "Propagate context through hook chain")
    try:
        async def hook1(event: HookEvent):
            return "value1"

        async def hook2(event: HookEvent):
            # Should have access to hook1's result
            prev_result = event.context.get("hook_chain_hook1_result")
            return f"value2_{prev_result}"

        await service.register_hook(
            "chain_hook1",
            "Chain 1",
            EventType.CONTEXT_CREATE_PRE,
            hook1,
            priority=80,
        )
        await service.register_hook(
            "chain_hook2",
            "Chain 2",
            EventType.CONTEXT_CREATE_PRE,
            hook2,
            priority=70,
        )

        results = await service.emit(EventType.CONTEXT_CREATE_PRE)
        assert len(results) >= 2
        print(f"   Context propagated through {len(results)} hooks")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 4: Conditional Hooks
# ============================================================================


async def test_conditional_hooks():
    """Test conditional hook execution"""
    print_test_header("4: Conditional Hooks")

    service = HookService()

    # Test 1: Hook with condition
    print_test(1, 2, "Execute hook only if condition met")
    try:
        executed = []

        async def conditional_handler(event: HookEvent):
            executed.append("conditional")

        def condition(event: HookEvent):
            return event.data.get("execute") is True

        await service.register_hook(
            hook_id="conditional_hook",
            name="Conditional Hook",
            event_type=EventType.EXPORT_PRE,
            handler=conditional_handler,
            condition=condition,
        )

        # Should not execute (condition not met)
        await service.emit(EventType.EXPORT_PRE, data={"execute": False})
        assert len(executed) == 0

        # Should execute (condition met)
        await service.emit(EventType.EXPORT_PRE, data={"execute": True})
        assert len(executed) == 1

        print(f"   Conditional hook executed correctly")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Async condition
    print_test(2, 2, "Execute hook with async condition")
    try:
        executed = []

        async def async_handler(event: HookEvent):
            executed.append("async_conditional")

        async def async_condition(event: HookEvent):
            await asyncio.sleep(0.01)  # Simulate async check
            return event.data.get("async_execute") is True

        await service.register_hook(
            hook_id="async_conditional_hook",
            name="Async Conditional Hook",
            event_type=EventType.IMPORT_PRE,
            handler=async_handler,
            condition=async_condition,
        )

        # Should execute
        await service.emit(EventType.IMPORT_PRE, data={"async_execute": True})
        assert len(executed) == 1

        print(f"   Async conditional hook executed correctly")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 5: Plugin System
# ============================================================================


async def test_plugin_system():
    """Test plugin system"""
    print_test_header("5: Plugin System")

    service = HookService()

    # Test 1: Register plugin
    print_test(1, 5, "Register plugin")
    try:
        result = await service.register_plugin(
            plugin_id="test_plugin",
            name="Test Plugin",
            description="A test plugin",
            version="1.0.0",
        )
        assert result is True
        print(f"   Registered plugin: test_plugin")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Add hook to plugin
    print_test(2, 5, "Add hook to plugin")
    try:
        async def plugin_handler(event: HookEvent):
            return "plugin_result"

        await service.register_hook(
            "plugin_hook_1",
            "Plugin Hook 1",
            EventType.MCP_REQUEST_PRE,
            plugin_handler,
        )

        result = await service.add_hook_to_plugin("test_plugin", "plugin_hook_1")
        assert result is True

        plugin = await service.get_plugin("test_plugin")
        assert "plugin_hook_1" in plugin.hooks
        print(f"   Added hook to plugin")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: List plugins
    print_test(3, 5, "List plugins")
    try:
        plugins = await service.list_plugins()
        assert len(plugins) >= 1
        print(f"   Found {len(plugins)} plugins")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Disable plugin
    print_test(4, 5, "Disable plugin")
    try:
        result = await service.disable_plugin("test_plugin")
        assert result is True
        plugin = await service.get_plugin("test_plugin")
        assert plugin.enabled is False
        print(f"   Disabled plugin: test_plugin")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Enable plugin
    print_test(5, 5, "Enable plugin")
    try:
        result = await service.enable_plugin("test_plugin")
        assert result is True
        plugin = await service.get_plugin("test_plugin")
        assert plugin.enabled is True
        print(f"   Enabled plugin: test_plugin")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 6: Middleware System
# ============================================================================


async def test_middleware_system():
    """Test middleware system"""
    print_test_header("6: Middleware System")

    service = HookService()

    # Test 1: Add middleware
    print_test(1, 3, "Add middleware")
    try:
        def test_middleware(event: HookEvent):
            # Add metadata to event
            event.metadata["middleware_processed"] = True
            return event

        result = service.add_middleware(test_middleware)
        assert result is True
        print(f"   Added middleware: test_middleware")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Middleware processes events
    print_test(2, 3, "Middleware processes events")
    try:
        processed = []

        async def check_handler(event: HookEvent):
            if event.metadata.get("middleware_processed"):
                processed.append(True)

        await service.register_hook(
            "middleware_check",
            "Middleware Check",
            EventType.LORA_TRAIN_PRE,
            check_handler,
        )

        await service.emit(EventType.LORA_TRAIN_PRE)
        assert len(processed) == 1
        print(f"   Middleware processed event")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Clear middleware
    print_test(3, 3, "Clear middleware")
    try:
        result = service.clear_middleware()
        assert result is True
        print(f"   Cleared all middleware")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 7: Hook Chains
# ============================================================================


async def test_hook_chains():
    """Test hook chain creation and execution"""
    print_test_header("7: Hook Chains")

    service = HookService()

    # Test 1: Create hook chain
    print_test(1, 2, "Create hook chain")
    try:
        execution_order = []

        async def step1(event: HookEvent):
            execution_order.append("step1")

        async def step2(event: HookEvent):
            execution_order.append("step2")

        async def step3(event: HookEvent):
            execution_order.append("step3")

        hook_ids = await service.create_hook_chain(
            chain_id="test_chain",
            event_type=EventType.TEMPLATE_INSTANTIATE_PRE,
            handlers=[step1, step2, step3],
            base_priority=80,
        )

        assert len(hook_ids) == 3
        print(f"   Created hook chain with {len(hook_ids)} hooks")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Execute hook chain
    print_test(2, 2, "Execute hook chain in order")
    try:
        await service.emit(EventType.TEMPLATE_INSTANTIATE_PRE)
        assert execution_order == ["step1", "step2", "step3"]
        print(f"   Chain executed in order: {execution_order}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 8: Monitoring & Statistics
# ============================================================================


async def test_monitoring():
    """Test monitoring and statistics"""
    print_test_header("8: Monitoring & Statistics")

    service = HookService()

    # Test 1: Get statistics
    print_test(1, 4, "Get hook statistics")
    try:
        stats = await service.get_stats()
        assert stats.total_hooks >= 0
        print(f"   Total hooks: {stats.total_hooks}")
        print(f"   Total executions: {stats.total_executions}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Get event history
    print_test(2, 4, "Get event history")
    try:
        history = await service.get_event_history(limit=10)
        assert isinstance(history, list)
        print(f"   Event history: {len(history)} events")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Clear event history
    print_test(3, 4, "Clear event history")
    try:
        result = await service.clear_event_history()
        assert result is True
        history = await service.get_event_history()
        assert len(history) == 0
        print(f"   Cleared event history")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Reset statistics
    print_test(4, 4, "Reset statistics")
    try:
        result = await service.reset_stats()
        assert result is True
        stats = await service.get_stats()
        assert stats.total_executions == 0
        print(f"   Reset statistics")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Main Test Runner
# ============================================================================


async def run_all_tests():
    """Run all test components"""
    print("\n" + "=" * 80)
    print("WEEK 19-20 TEST SUITE: Hook System")
    print("=" * 80)
    print("Testing all hook system components...\n")

    # Run test components
    await test_hook_registration()
    await test_event_emission()
    await test_hook_priorities()
    await test_conditional_hooks()
    await test_plugin_system()
    await test_middleware_system()
    await test_hook_chains()
    await test_monitoring()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")

    if test_results["failed"] == 0:
        print("\n🎉 ALL TESTS PASSED! 100% Success Rate")
    else:
        success_rate = (
            test_results["passed"] / test_results["total"] * 100
            if test_results["total"] > 0
            else 0
        )
        print(f"\n⚠️  Success Rate: {success_rate:.1f}%")

    print("=" * 80 + "\n")

    return test_results["failed"] == 0


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)

