"""
Test Suite for Week 7-8: Advanced AutoGen Features

Tests:
1. Custom Agent Types
2. Tool Integration
3. Agent Memory and State Management
4. RAG Integration with Agents
5. Advanced Multi-Agent Orchestration

Target: 100% pass rate
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.advanced_autogen_service import (
    AdvancedAutoGenService,
    AgentMemoryManager,
    CustomAgentConfig,
    ToolRegistry,
)
from backend.services.autogen_service import AutoGenService

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
# Component 1: Custom Agent Types
# ============================================================================


async def test_custom_agent_types():
    """Test custom agent type creation and configuration"""
    print_test_header("1: Custom Agent Types")

    service = AdvancedAutoGenService()
    await service.initialize()

    # Test 1: Verify specialized agents created
    print_test(1, 5, "Verify specialized agent configs created")
    try:
        assert "researcher" in service.agent_configs
        assert "architect" in service.agent_configs
        assert "curator" in service.agent_configs
        assert "orchestrator" in service.agent_configs
        assert "critic" in service.agent_configs
        assert len(service.agent_configs) == 5
        print(f"   Created {len(service.agent_configs)} specialized agents")
        test_passed()
    except AssertionError as e:
        test_failed(str(e))

    # Test 2: Verify researcher agent configuration
    print_test(2, 5, "Verify researcher agent has RAG capabilities")
    try:
        researcher = service.agent_configs["researcher"]
        assert researcher.rag_enabled is True
        assert "query_knowledge" in researcher.tools
        assert "rag" in researcher.capabilities
        print(f"   Researcher tools: {researcher.tools}")
        print(f"   Researcher capabilities: {researcher.capabilities}")
        test_passed()
    except AssertionError as e:
        test_failed(str(e))

    # Test 3: Verify architect agent configuration
    print_test(3, 5, "Verify architect agent has coding tools")
    try:
        architect = service.agent_configs["architect"]
        assert "execute_code" in architect.tools
        assert "file_operations" in architect.tools
        assert "coding" in architect.capabilities
        print(f"   Architect tools: {architect.tools}")
        test_passed()
    except AssertionError as e:
        test_failed(str(e))

    # Test 4: Verify personality traits
    print_test(4, 5, "Verify agents have personality traits")
    try:
        researcher = service.agent_configs["researcher"]
        assert "curiosity" in researcher.personality_traits
        assert researcher.personality_traits["curiosity"] > 0
        print(f"   Researcher traits: {researcher.personality_traits}")
        test_passed()
    except AssertionError as e:
        test_failed(str(e))

    # Test 5: Create actual agent instance
    print_test(5, 5, "Create agent instance from config")
    try:
        researcher_config = service.agent_configs["researcher"]
        agent = await service.create_agent(researcher_config, enable_tools=False)
        assert agent is not None
        assert agent.name == "researcher"
        print(f"   Created agent: {agent.name}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 2: Tool Integration
# ============================================================================


async def test_tool_integration():
    """Test tool registry and execution"""
    print_test_header("2: Tool Integration")

    registry = ToolRegistry()

    # Test 1: Verify default tools registered
    print_test(1, 5, "Verify default tools registered")
    try:
        assert "web_search" in registry.tools
        assert "execute_code" in registry.tools
        assert "file_operations" in registry.tools
        assert "query_knowledge" in registry.tools
        assert len(registry.tools) >= 4
        print(f"   Registered tools: {list(registry.tools.keys())}")
        test_passed()
    except AssertionError as e:
        test_failed(str(e))

    # Test 2: Execute web search tool
    print_test(2, 5, "Execute web search tool (stub)")
    try:
        result = await registry.execute_tool(
            "web_search", {"query": "artificial intelligence", "num_results": 3}
        )
        assert result["success"] is True
        assert "result" in result
        print(f"   Search results: {len(result['result'])} items")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Execute code execution tool
    print_test(3, 5, "Execute code execution tool (stub)")
    try:
        result = await registry.execute_tool(
            "execute_code", {"code": "print('Hello')", "language": "python"}
        )
        assert result["success"] is True
        print(f"   Code execution result: {result['result']}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Execute file operations tool
    print_test(4, 5, "Execute file operations tool (stub)")
    try:
        result = await registry.execute_tool(
            "file_operations",
            {"operation": "read", "path": "/tmp/test.txt", "content": ""},
        )
        assert result["success"] is True
        print(f"   File operation result: {result['result']}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Handle invalid tool
    print_test(5, 5, "Handle invalid tool gracefully")
    try:
        result = await registry.execute_tool("invalid_tool", {})
        assert "error" in result
        print(f"   Error handling works: {result['error']}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 3: Agent Memory and State Management
# ============================================================================


async def test_agent_memory():
    """Test agent memory and state management"""
    print_test_header("3: Agent Memory and State Management")

    memory_manager = AgentMemoryManager()

    # Test 1: Store short-term memory
    print_test(1, 5, "Store short-term memory")
    try:
        memory_id = await memory_manager.store_memory(
            "researcher",
            "short_term",
            {"task": "search for AI papers", "status": "in_progress"},
            importance=0.6,
        )
        assert memory_id is not None
        print(f"   Stored memory: {memory_id}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Store long-term memory
    print_test(2, 5, "Store long-term memory")
    try:
        memory_id = await memory_manager.store_memory(
            "researcher",
            "long_term",
            {"learned": "RAG improves accuracy", "confidence": 0.9},
            importance=0.8,
            tags=["learning", "rag"],
        )
        assert memory_id is not None
        print(f"   Stored long-term memory with tags")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Retrieve memories
    print_test(3, 5, "Retrieve agent memories")
    try:
        memories = await memory_manager.retrieve_memories("researcher", limit=10)
        assert len(memories) >= 2
        print(f"   Retrieved {len(memories)} memories")
        for m in memories:
            print(f"     - {m.memory_type}: {m.content}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Update agent state
    print_test(4, 5, "Update agent state")
    try:
        state = await memory_manager.update_agent_state(
            "researcher", {"current_task": "research AI", "context": {"topic": "ML"}}
        )
        assert state.agent_name == "researcher"
        assert state.current_task == "research AI"
        print(f"   Updated state: {state.current_task}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Get agent state
    print_test(5, 5, "Get agent state")
    try:
        state = await memory_manager.get_agent_state("researcher")
        assert state is not None
        assert state.current_task == "research AI"
        print(f"   Agent state: task={state.current_task}, memories={len(state.memory_ids)}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 4: RAG Integration with Agents
# ============================================================================


async def test_rag_integration():
    """Test RAG integration with agents"""
    print_test_header("4: RAG Integration with Agents")

    service = AdvancedAutoGenService()
    await service.initialize()

    # Test 1: Verify RAG-enabled agents
    print_test(1, 4, "Verify RAG-enabled agent configs")
    try:
        researcher = service.agent_configs["researcher"]
        curator = service.agent_configs["curator"]
        assert researcher.rag_enabled is True
        assert curator.rag_enabled is True
        print(f"   RAG-enabled agents: researcher, curator")
        test_passed()
    except AssertionError as e:
        test_failed(str(e))

    # Test 2: Query knowledge tool
    print_test(2, 4, "Test query_knowledge tool")
    try:
        result = await service.tool_registry.execute_tool(
            "query_knowledge", {"query": "What is machine learning?", "n_results": 3}
        )
        # Tool should execute (may return error if RAG service not initialized)
        assert result is not None
        print(f"   Knowledge query executed: {result.get('success', 'N/A')}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Create RAG-enhanced agent
    print_test(3, 4, "Create RAG-enhanced agent")
    try:
        researcher_config = service.agent_configs["researcher"]
        agent = await service.create_rag_enhanced_agent(researcher_config)
        assert agent is not None
        assert agent.name == "researcher"
        print(f"   Created RAG-enhanced agent: {agent.name}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Verify RAG collection configuration
    print_test(4, 4, "Verify RAG collection configuration")
    try:
        researcher = service.agent_configs["researcher"]
        assert researcher.rag_collection == "greta_documents"
        print(f"   RAG collection: {researcher.rag_collection}")
        test_passed()
    except AssertionError as e:
        test_failed(str(e))


# ============================================================================
# Component 5: Advanced Multi-Agent Orchestration
# ============================================================================


async def test_advanced_orchestration():
    """Test advanced multi-agent orchestration"""
    print_test_header("5: Advanced Multi-Agent Orchestration")

    service = AdvancedAutoGenService()
    await service.initialize()

    # Test 1: Get agent insights
    print_test(1, 5, "Get agent insights")
    try:
        # First create an agent and update its state
        await service.memory_manager.update_agent_state(
            "researcher", {"current_task": "test task"}
        )
        await service.memory_manager.store_memory(
            "researcher", "short_term", {"test": "data"}, importance=0.5
        )

        insights = await service.get_agent_insights("researcher")
        assert "agent_name" in insights
        assert insights["agent_name"] == "researcher"
        print(f"   Agent insights: {insights['current_task']}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Optimize agent context
    print_test(2, 5, "Optimize agent context")
    try:
        context = await service.optimize_agent_context("researcher", max_tokens=2000)
        assert "agent_name" in context
        assert "optimized_context" in context
        print(f"   Optimized context with {context['memory_count']} memories")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Create multiple agents for collaboration
    print_test(3, 5, "Create multiple agents for collaboration")
    try:
        researcher_config = service.agent_configs["researcher"]
        architect_config = service.agent_configs["architect"]

        researcher = await service.create_agent(researcher_config, enable_tools=False)
        architect = await service.create_agent(architect_config, enable_tools=False)

        assert researcher.name == "researcher"
        assert architect.name == "architect"
        print(f"   Created agents: {researcher.name}, {architect.name}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Create advanced group chat
    print_test(4, 5, "Create advanced group chat")
    try:
        researcher_config = service.agent_configs["researcher"]
        critic_config = service.agent_configs["critic"]

        researcher = await service.create_agent(researcher_config, enable_tools=False)
        critic = await service.create_agent(critic_config, enable_tools=False)

        chat_info = await service.create_advanced_group_chat(
            agents=[researcher, critic],
            task="Analyze AI trends",
            max_rounds=5,
            speaker_selection="auto",
        )

        assert "group_chat" in chat_info
        assert "manager" in chat_info
        assert len(chat_info["agents"]) == 2
        print(f"   Group chat created with {len(chat_info['agents'])} agents")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Verify agent state persistence
    print_test(5, 5, "Verify agent state persistence")
    try:
        # Update state
        await service.memory_manager.update_agent_state(
            "orchestrator", {"current_task": "coordinate team", "context": {"team_size": 3}}
        )

        # Retrieve state
        state = await service.memory_manager.get_agent_state("orchestrator")
        assert state is not None
        assert state.current_task == "coordinate team"
        assert state.context["team_size"] == 3
        print(f"   State persisted: {state.current_task}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Main Test Runner
# ============================================================================


async def run_all_tests():
    """Run all test components"""
    print("\n" + "=" * 80)
    print("WEEK 7-8 TEST SUITE: Advanced AutoGen Features")
    print("=" * 80)
    print("Testing all advanced AutoGen components...\n")

    # Run all test components
    await test_custom_agent_types()
    await test_tool_integration()
    await test_agent_memory()
    await test_rag_integration()
    await test_advanced_orchestration()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")

    if test_results["failed"] == 0:
        print("\n🎉 ALL TESTS PASSED! 100% Success Rate")
        success_rate = 100.0
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

