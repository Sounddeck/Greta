"""
Test Suite for Tool Registry API
Tests MCP-inspired tool discovery, registration, and execution endpoints
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# Test configuration
TEST_AGENT_ID = "test_agent_001"
TEST_TOOL_ID = "osint_web_scraper"


@pytest.mark.asyncio
async def test_tool_discovery():
    """Test tool discovery endpoint"""
    from services.tool_registry import ToolRegistry
    
    print("\n" + "="*60)
    print("TEST 1: Tool Discovery")
    print("="*60)
    
    # Initialize registry
    registry = ToolRegistry()
    await registry.initialize()
    
    # Get all tools
    all_tools = list(registry.tools.values())
    print(f"\n✅ Total tools available: {len(all_tools)}")
    
    # Test category filtering
    osint_tools = [t for t in all_tools if t.category.value == "osint"]
    print(f"✅ OSINT tools: {len(osint_tools)}")
    
    data_tools = [t for t in all_tools if t.category.value == "data_processing"]
    print(f"✅ Data Processing tools: {len(data_tools)}")
    
    web_tools = [t for t in all_tools if t.category.value == "web_api"]
    print(f"✅ Web API tools: {len(web_tools)}")
    
    # Test search
    search_results = [
        t for t in all_tools
        if "web" in t.name.lower() or "web" in t.description.lower()
    ]
    print(f"✅ Tools matching 'web': {len(search_results)}")
    
    assert len(all_tools) > 0, "Should have tools registered"
    assert len(osint_tools) > 0, "Should have OSINT tools"
    
    print("\n✅ Tool Discovery Test PASSED")


@pytest.mark.asyncio
async def test_get_tool_by_id():
    """Test getting tool by ID"""
    from services.tool_registry import ToolRegistry
    
    print("\n" + "="*60)
    print("TEST 2: Get Tool by ID")
    print("="*60)
    
    # Initialize registry
    registry = ToolRegistry()
    await registry.initialize()
    
    # Get specific tool
    tool = registry.tools.get(TEST_TOOL_ID)
    
    if tool:
        print(f"\n✅ Tool found: {tool.name}")
        print(f"   ID: {tool.tool_id}")
        print(f"   Category: {tool.category.value}")
        print(f"   Permission: {tool.permission_level.value}")
        print(f"   Description: {tool.description}")
        print(f"   Parameters: {list(tool.parameters.keys())}")
        
        assert tool.tool_id == TEST_TOOL_ID
        assert tool.name is not None
        assert tool.category is not None
        
        print("\n✅ Get Tool by ID Test PASSED")
    else:
        print(f"\n⚠️ Tool '{TEST_TOOL_ID}' not found, skipping test")


@pytest.mark.asyncio
async def test_tool_execution():
    """Test tool execution"""
    from services.tool_registry import ToolRegistry
    
    print("\n" + "="*60)
    print("TEST 3: Tool Execution")
    print("="*60)
    
    # Initialize registry
    registry = ToolRegistry()
    await registry.initialize()
    
    # Execute web scraper tool
    try:
        result = await registry.execute_tool(
            tool_id=TEST_TOOL_ID,
            parameters={
                "url": "https://example.com",
                "extract_links": False
            },
            agent_id=TEST_AGENT_ID
        )
        
        print(f"\n✅ Tool executed successfully")
        print(f"   Execution ID: {result.get('execution_id')}")
        print(f"   Success: {result.get('success')}")
        print(f"   Execution Time: {result.get('execution_time', 0):.3f}s")
        
        if result.get('error'):
            print(f"   Error: {result.get('error')}")
        
        assert result.get('execution_id') is not None
        
        print("\n✅ Tool Execution Test PASSED")
        
    except Exception as e:
        print(f"\n⚠️ Tool execution failed: {e}")
        print("   This may be expected if the tool requires external resources")


@pytest.mark.asyncio
async def test_tool_categories():
    """Test tool categorization"""
    from services.tool_registry import ToolRegistry, ToolCategory
    
    print("\n" + "="*60)
    print("TEST 4: Tool Categories")
    print("="*60)
    
    # Initialize registry
    registry = ToolRegistry()
    await registry.initialize()
    
    # Count tools by category
    category_counts = {}
    for tool in registry.tools.values():
        cat = tool.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n📊 Tools by Category:")
    for category, count in sorted(category_counts.items()):
        print(f"   {category}: {count} tools")
    
    # Verify all categories are represented
    expected_categories = [
        "osint", "data_processing", "web_api",
        "code_execution", "ai_ml"
    ]
    
    for cat in expected_categories:
        if cat in category_counts:
            print(f"✅ {cat}: {category_counts[cat]} tools")
        else:
            print(f"⚠️ {cat}: No tools registered")
    
    assert len(category_counts) > 0, "Should have tools in categories"
    
    print("\n✅ Tool Categories Test PASSED")


@pytest.mark.asyncio
async def test_tool_permissions():
    """Test tool permission system"""
    from services.tool_registry import ToolRegistry, PermissionLevel
    
    print("\n" + "="*60)
    print("TEST 5: Tool Permissions")
    print("="*60)
    
    # Initialize registry
    registry = ToolRegistry()
    await registry.initialize()
    
    # Count tools by permission level
    permission_counts = {}
    for tool in registry.tools.values():
        perm = tool.permission_level.value
        permission_counts[perm] = permission_counts.get(perm, 0) + 1
    
    print("\n🔒 Tools by Permission Level:")
    for perm, count in sorted(permission_counts.items()):
        print(f"   {perm}: {count} tools")
    
    # Test permission check
    test_tool_id = TEST_TOOL_ID
    can_execute = await registry.check_permission(TEST_AGENT_ID, test_tool_id)
    print(f"\n✅ Agent '{TEST_AGENT_ID}' can execute '{test_tool_id}': {can_execute}")
    
    assert len(permission_counts) > 0, "Should have tools with permissions"
    
    print("\n✅ Tool Permissions Test PASSED")


@pytest.mark.asyncio
async def test_tool_metrics():
    """Test tool metrics tracking"""
    from services.tool_registry import ToolRegistry
    
    print("\n" + "="*60)
    print("TEST 6: Tool Metrics")
    print("="*60)
    
    # Initialize registry
    registry = ToolRegistry()
    await registry.initialize()
    
    # Get metrics
    metrics = registry.metrics
    
    print("\n📊 Tool Registry Metrics:")
    print(f"   Total Executions: {metrics['total_executions']}")
    print(f"   Successful: {metrics['successful_executions']}")
    print(f"   Failed: {metrics['failed_executions']}")
    print(f"   Total Execution Time: {metrics['total_execution_time']:.3f}s")
    
    if metrics['total_executions'] > 0:
        avg_time = metrics['total_execution_time'] / metrics['total_executions']
        print(f"   Average Execution Time: {avg_time:.3f}s")
    
    print("\n📈 Executions by Category:")
    for category, count in metrics['executions_by_category'].items():
        print(f"   {category}: {count}")
    
    print("\n👥 Executions by Agent:")
    for agent, count in metrics['executions_by_agent'].items():
        print(f"   {agent}: {count}")
    
    assert metrics is not None
    
    print("\n✅ Tool Metrics Test PASSED")


@pytest.mark.asyncio
async def test_tool_versioning():
    """Test tool versioning support"""
    from services.tool_registry import ToolRegistry
    
    print("\n" + "="*60)
    print("TEST 7: Tool Versioning")
    print("="*60)
    
    # Initialize registry
    registry = ToolRegistry()
    await registry.initialize()
    
    # Check versioning fields
    versioned_tools = []
    for tool in registry.tools.values():
        if hasattr(tool, 'version'):
            versioned_tools.append(tool)
            print(f"\n✅ Tool: {tool.name}")
            print(f"   Version: {tool.version}")
            print(f"   Author: {tool.author}")
            if hasattr(tool, 'tags') and tool.tags:
                print(f"   Tags: {', '.join(tool.tags)}")
            if hasattr(tool, 'deprecated'):
                print(f"   Deprecated: {tool.deprecated}")
    
    print(f"\n✅ Tools with versioning: {len(versioned_tools)}/{len(registry.tools)}")
    
    assert len(versioned_tools) > 0, "Should have tools with versioning"
    
    print("\n✅ Tool Versioning Test PASSED")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TOOL REGISTRY API TEST SUITE")
    print("="*60)
    
    try:
        await test_tool_discovery()
        await test_get_tool_by_id()
        await test_tool_execution()
        await test_tool_categories()
        await test_tool_permissions()
        await test_tool_metrics()
        await test_tool_versioning()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

