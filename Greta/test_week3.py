"""
Test script for Week 3: AutoGen Integration
Tests AutoGen framework integration with local LLM
"""

import asyncio
import sys
sys.path.insert(0, '/Users/macone/Desktop/Greta/Greta')

from backend.services.autogen_service import autogen_service, AgentConfig
from loguru import logger


async def test_autogen_initialization():
    """Test Component 1: AutoGen Service Initialization"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 1: AutoGen Service Initialization")
    logger.info("=" * 60)
    
    # Test 1: Initialize service
    logger.info("\n🤖 Test 1.1: Initialize AutoGen Service")
    await autogen_service.initialize()
    logger.success("✅ Service initialized")
    
    # Test 2: List default agents
    logger.info("\n🤖 Test 1.2: List default agent configurations")
    agents = await autogen_service.list_agents()
    logger.info(f"Found {len(agents)} agent configurations")
    for agent_name in agents:
        config = autogen_service.agent_configs[agent_name]
        logger.info(f"  - {agent_name}: {config.role} - {config.capabilities}")
    logger.success("✅ Agent configurations listed")
    
    if len(agents) < 5:
        logger.error("❌ Expected at least 5 default agents")
        return False
    
    logger.success("\n✅ Component 1: AutoGen Initialization - ALL TESTS PASSED")
    return True


async def test_agent_creation():
    """Test Component 2: Agent Creation"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 2: Agent Creation")
    logger.info("=" * 60)
    
    # Test 1: Create assistant agent
    logger.info("\n🤖 Test 2.1: Create assistant agent")
    config = autogen_service.agent_configs.get("assistant")
    if not config:
        logger.error("❌ Assistant config not found")
        return False
    
    agent = await autogen_service.create_agent(config)
    if agent:
        logger.info(f"Agent created: {agent.name}")
        logger.success("✅ Assistant agent created")
    else:
        logger.error("❌ Failed to create assistant agent")
        return False
    
    # Test 2: Create coder agent
    logger.info("\n🤖 Test 2.2: Create coder agent")
    config = autogen_service.agent_configs.get("coder")
    agent = await autogen_service.create_agent(config)
    if agent:
        logger.info(f"Agent created: {agent.name}")
        logger.success("✅ Coder agent created")
    else:
        logger.error("❌ Failed to create coder agent")
        return False
    
    # Test 3: Create user proxy agent
    logger.info("\n🤖 Test 2.3: Create user proxy agent")
    config = autogen_service.agent_configs.get("user_proxy")
    agent = await autogen_service.create_agent(config)
    if agent:
        logger.info(f"Agent created: {agent.name}")
        logger.success("✅ User proxy agent created")
    else:
        logger.error("❌ Failed to create user proxy agent")
        return False
    
    # Test 4: Get existing agent
    logger.info("\n🤖 Test 2.4: Get existing agent")
    agent = await autogen_service.get_agent("assistant")
    if agent:
        logger.info(f"Retrieved agent: {agent.name}")
        logger.success("✅ Agent retrieved")
    else:
        logger.error("❌ Failed to retrieve agent")
        return False
    
    logger.success("\n✅ Component 2: Agent Creation - ALL TESTS PASSED")
    return True


async def test_custom_agent():
    """Test Component 3: Custom Agent Configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 3: Custom Agent Configuration")
    logger.info("=" * 60)
    
    # Test 1: Register custom agent config
    logger.info("\n🤖 Test 3.1: Register custom agent configuration")
    custom_config = AgentConfig(
        name="researcher",
        role="assistant",
        system_message="You are a research assistant specialized in gathering and analyzing information.",
        llm_config=autogen_service.default_llm_config,
        capabilities=["research", "analysis", "summarization"]
    )
    
    await autogen_service.register_agent_config(custom_config)
    logger.success("✅ Custom agent config registered")
    
    # Test 2: Create custom agent
    logger.info("\n🤖 Test 3.2: Create custom agent")
    agent = await autogen_service.create_agent(custom_config)
    if agent:
        logger.info(f"Custom agent created: {agent.name}")
        logger.success("✅ Custom agent created")
    else:
        logger.error("❌ Failed to create custom agent")
        return False
    
    # Test 3: Verify agent in list
    logger.info("\n🤖 Test 3.3: Verify custom agent in list")
    agents = await autogen_service.list_agents()
    if "researcher" in agents:
        logger.success("✅ Custom agent found in list")
    else:
        logger.error("❌ Custom agent not in list")
        return False
    
    logger.success("\n✅ Component 3: Custom Agent Configuration - ALL TESTS PASSED")
    return True


async def test_conversation_setup():
    """Test Component 4: Multi-Agent Conversation Setup"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 4: Multi-Agent Conversation Setup")
    logger.info("=" * 60)
    
    # Test 1: Start conversation with 2 agents
    logger.info("\n🤖 Test 4.1: Start conversation with 2 agents")
    result = await autogen_service.start_conversation(
        conversation_id="test-conv-1",
        agent_names=["assistant", "coder"],
        initial_message="Let's discuss how to implement a sorting algorithm.",
        max_rounds=5
    )
    
    if result.get("success"):
        logger.info(f"Conversation started: {result.get('conversation_id')}")
        logger.info(f"Agents: {result.get('agents')}")
        logger.success("✅ Conversation started")
    else:
        logger.error(f"❌ Failed to start conversation: {result.get('error')}")
        return False
    
    # Test 2: Get conversation status
    logger.info("\n🤖 Test 4.2: Get conversation status")
    status = await autogen_service.get_conversation_status("test-conv-1")
    if "error" not in status:
        logger.info(f"Status: {status.get('status')}")
        logger.info(f"Agents: {status.get('agents')}")
        logger.success("✅ Conversation status retrieved")
    else:
        logger.error("❌ Failed to get conversation status")
        return False
    
    # Test 3: Start conversation with 3 agents
    logger.info("\n🤖 Test 4.3: Start conversation with 3 agents")
    result = await autogen_service.start_conversation(
        conversation_id="test-conv-2",
        agent_names=["planner", "coder", "critic"],
        initial_message="Plan and implement a web scraper.",
        max_rounds=10
    )
    
    if result.get("success"):
        logger.success("✅ Multi-agent conversation started")
    else:
        logger.error(f"❌ Failed: {result.get('error')}")
        return False
    
    # Test 4: Stop conversation
    logger.info("\n🤖 Test 4.4: Stop conversation")
    stopped = await autogen_service.stop_conversation("test-conv-1")
    if stopped:
        logger.success("✅ Conversation stopped")
    else:
        logger.error("❌ Failed to stop conversation")
        return False
    
    logger.success("\n✅ Component 4: Multi-Agent Conversation - ALL TESTS PASSED")
    return True


async def test_cleanup():
    """Test Component 5: Cleanup"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 5: Cleanup")
    logger.info("=" * 60)
    
    # Test 1: Clear conversations
    logger.info("\n🤖 Test 5.1: Clear conversations")
    await autogen_service.clear_conversations()
    logger.success("✅ Conversations cleared")
    
    # Test 2: Clear agents
    logger.info("\n🤖 Test 5.2: Clear agents")
    await autogen_service.clear_agents()
    logger.success("✅ Agents cleared")
    
    logger.success("\n✅ Component 5: Cleanup - ALL TESTS PASSED")
    return True


async def run_all_tests():
    """Run all Week 3 tests"""
    logger.info("=" * 60)
    logger.info("Week 3: AutoGen Integration - Test Suite")
    logger.info("=" * 60)
    
    results = {
        'Component 1: AutoGen Initialization': False,
        'Component 2: Agent Creation': False,
        'Component 3: Custom Agent Configuration': False,
        'Component 4: Multi-Agent Conversation': False,
        'Component 5: Cleanup': False
    }
    
    try:
        results['Component 1: AutoGen Initialization'] = await test_autogen_initialization()
    except Exception as e:
        logger.error(f"Component 1 failed: {e}")
    
    try:
        results['Component 2: Agent Creation'] = await test_agent_creation()
    except Exception as e:
        logger.error(f"Component 2 failed: {e}")
    
    try:
        results['Component 3: Custom Agent Configuration'] = await test_custom_agent()
    except Exception as e:
        logger.error(f"Component 3 failed: {e}")
    
    try:
        results['Component 4: Multi-Agent Conversation'] = await test_conversation_setup()
    except Exception as e:
        logger.error(f"Component 4 failed: {e}")
    
    try:
        results['Component 5: Cleanup'] = await test_cleanup()
    except Exception as e:
        logger.error(f"Component 5 failed: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("WEEK 3 TEST SUMMARY")
    logger.info("=" * 60)
    
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{component}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.success("\n🎉 ALL WEEK 3 TESTS PASSED!")
        logger.info("\nWeek 3 Complete: AutoGen Integration working!")
    else:
        logger.error("\n❌ SOME TESTS FAILED")
        logger.info("Please review the errors above")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
