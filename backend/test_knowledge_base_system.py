"""
Test script for Master Agent Knowledge Base System
Tests all CRUD operations and Master Agent integration
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

async def test_knowledge_base():
    """Test knowledge base endpoints"""
    async with httpx.AsyncClient() as client:
        print("\n" + "="*60)
        print("🧪 MASTER AGENT KNOWLEDGE BASE SYSTEM TEST")
        print("="*60)

        # Test 1: Get system knowledge
        print("\n1️⃣  Testing GET /api/v1/knowledge/system")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/knowledge/system")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {data['total']} knowledge entries")
                print(f"   Entries: {len(data['entries'])}")
                for entry in data['entries'][:2]:
                    print(f"   - {entry['title']} (Priority: {entry['priority']})")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Exception: {e}")

        # Test 2: Search knowledge
        print("\n2️⃣  Testing GET /api/v1/knowledge/system/search")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/knowledge/system/search",
                params={"query": "architecture"}
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {data['total']} results for 'architecture'")
                for result in data['results'][:2]:
                    print(f"   - {result['title']}")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Exception: {e}")

        # Test 3: Get Master Agent context
        print("\n3️⃣  Testing POST /api/v1/knowledge/system/context")
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/knowledge/system/context",
                json={
                    "include_categories": ["architecture", "features"],
                    "max_entries": 5,
                    "min_priority": 7
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved Master Agent context")
                print(f"   System overview length: {len(data['system_overview'])} chars")
                print(f"   Available endpoints: {len(data['available_endpoints'])}")
                print(f"   Capabilities: {len(data['capabilities'])}")
                print(f"   Knowledge entries: {len(data['knowledge_entries'])}")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Exception: {e}")

        # Test 4: Create new knowledge entry
        print("\n4️⃣  Testing POST /api/v1/knowledge/system (Create)")
        try:
            new_entry = {
                "category": "capabilities",
                "title": "Test Knowledge Entry",
                "content": "This is a test knowledge entry for the Master Agent.",
                "tags": ["test", "knowledge-base"],
                "priority": 5,
                "version": "1.0.0"
            }
            response = await client.post(
                f"{BASE_URL}/api/v1/knowledge/system",
                json=new_entry
            )
            if response.status_code == 200:
                data = response.json()
                entry_id = data['id']
                print(f"✅ Created knowledge entry: {entry_id}")
                print(f"   Title: {data['title']}")
                print(f"   Category: {data['category']}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Exception: {e}")

        # Test 5: Execute task with Master Agent (uses knowledge base)
        print("\n5️⃣  Testing POST /api/v1/tasks/execute (with knowledge base)")
        try:
            task_request = {
                "agent_id": "master_agent",
                "description": "What are the main features of CPAS5?",
                "input_data": {
                    "agent_type": "general",
                    "conversation_history": [],
                    "uploaded_files": []
                },
                "max_iterations": 1
            }
            response = await client.post(
                f"{BASE_URL}/api/v1/tasks/execute",
                json=task_request,
                timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Task executed successfully")
                print(f"   Status: {data['status']}")
                if data.get('result'):
                    result_preview = data['result'][:200]
                    print(f"   Result preview: {result_preview}...")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"❌ Exception: {e}")

        # Test 6: Filter by category
        print("\n6️⃣  Testing GET /api/v1/knowledge/system (with category filter)")
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/knowledge/system",
                params={"category": "api_endpoints"}
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {data['total']} API endpoint entries")
                for entry in data['entries'][:2]:
                    print(f"   - {entry['title']}")
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Exception: {e}")

        print("\n" + "="*60)
        print("✅ KNOWLEDGE BASE SYSTEM TEST COMPLETE")
        print("="*60 + "\n")


async def main():
    """Run all tests"""
    await test_knowledge_base()


if __name__ == "__main__":
    asyncio.run(main())

