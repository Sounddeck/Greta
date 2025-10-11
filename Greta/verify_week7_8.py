"""
Quick verification script for Week 7-8 implementation
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.services.advanced_autogen_service import (
    AdvancedAutoGenService,
    AgentMemoryManager,
    ToolRegistry,
)


async def verify():
    """Quick verification of Week 7-8 features"""
    print("=" * 80)
    print("WEEK 7-8 VERIFICATION")
    print("=" * 80 + "\n")

    # Test 1: Tool Registry
    print("[1/5] Tool Registry...")
    registry = ToolRegistry()
    assert len(registry.tools) >= 4
    print(f"   ✅ {len(registry.tools)} tools registered: {list(registry.tools.keys())}\n")

    # Test 2: Memory Manager
    print("[2/5] Memory Manager...")
    memory = AgentMemoryManager()
    await memory.store_memory("test_agent", "short_term", {"test": "data"})
    memories = await memory.retrieve_memories("test_agent")
    assert len(memories) >= 1
    print(f"   ✅ Memory system working: {len(memories)} memories stored\n")

    # Test 3: Advanced Service
    print("[3/5] Advanced AutoGen Service...")
    service = AdvancedAutoGenService()
    await service.initialize()
    assert len(service.agent_configs) == 5
    print(f"   ✅ Service initialized with {len(service.agent_configs)} agents\n")

    # Test 4: Agent Configs
    print("[4/5] Specialized Agents...")
    agents = list(service.agent_configs.keys())
    print(f"   ✅ Agents: {agents}\n")

    # Test 5: RAG Integration
    print("[5/5] RAG Integration...")
    researcher = service.agent_configs["researcher"]
    curator = service.agent_configs["curator"]
    assert researcher.rag_enabled is True
    assert curator.rag_enabled is True
    print(f"   ✅ RAG-enabled agents: researcher, curator\n")

    print("=" * 80)
    print("✅ ALL VERIFICATIONS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(verify())

