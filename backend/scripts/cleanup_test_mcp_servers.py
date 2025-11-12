#!/usr/bin/env python3
"""
Cleanup Test MCP Servers
Removes test/duplicate MCP servers from the database
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import Database
from services.mcp_service import MCPHubService

async def cleanup_test_servers():
    """Remove test MCP servers from the database"""
    print("🧹 Cleaning up test MCP servers...")
    print("=" * 60)
    
    # Connect to database
    db = Database()
    await db.connect()
    service = MCPHubService(db)
    
    # List all servers
    all_servers = await service.list_servers(limit=1000)
    print(f"\n📊 Total MCP servers before cleanup: {len(all_servers)}")
    
    # Identify test servers to remove
    test_server_patterns = [
        "Tool Test Server",
        "Health Test Server",
        "Integration Test Server",
        "Test Slack Workspace",
        "TestSlack",
        "Test Stripe Account",
        "TestStripe",
        "EXA"  # Assuming this is a test server
    ]
    
    servers_to_remove = []
    for server in all_servers:
        for pattern in test_server_patterns:
            if pattern in server.name:
                servers_to_remove.append(server)
                break
    
    print(f"\n🔍 Found {len(servers_to_remove)} test servers to remove:")
    for server in servers_to_remove:
        print(f"   • {server.name} (ID: {server.server_id}, {server.tool_count} tools)")
    
    if len(servers_to_remove) == 0:
        print("\n✅ No test servers found. Database is clean!")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  This will delete {len(servers_to_remove)} test servers.")
    response = input("Do you want to proceed? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    # Delete test servers
    deleted_count = 0
    for server in servers_to_remove:
        try:
            await service.delete_server(server.server_id)
            print(f"✅ Deleted: {server.name}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Error deleting {server.name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully deleted {deleted_count} test servers")
    print("=" * 60)
    
    # List remaining servers
    remaining_servers = await service.list_servers(limit=1000)
    print(f"\n📊 Total MCP servers after cleanup: {len(remaining_servers)}")
    print("\n🎯 Remaining useful servers:")
    for server in remaining_servers:
        print(f"   • {server.name} ({server.tool_count} tools)")

if __name__ == "__main__":
    asyncio.run(cleanup_test_servers())

