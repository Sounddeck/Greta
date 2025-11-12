"""
Test AutoGen Orchestration Functionality
Verifies that AutoGen multi-agent conversation works (not using fallback)
"""
import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*70)
print("AUTOGEN ORCHESTRATION TEST")
print("="*70)

# Load environment
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Import Master Agent
from utils.greta_master_agent import GretaMasterAgent, AUTOGEN_AVAILABLE

print(f"\n[1] AutoGen Availability: {AUTOGEN_AVAILABLE}")

if not AUTOGEN_AVAILABLE:
    print("❌ AutoGen is not available - cannot test orchestration")
    sys.exit(1)

print("✅ AutoGen is available")

# Initialize Master Agent
print("\n[2] Initializing Master Agent...")
master_agent = GretaMasterAgent()
print(f"✅ Master Agent initialized: {master_agent.agent_id}")

# Test AutoGen initialization
print("\n[3] Testing AutoGen Initialization...")

async def test_autogen():
    """Test AutoGen initialization"""
    try:
        # Call the initialization method
        autogen_manager = await master_agent._initialize_autogen()
        
        if autogen_manager is None:
            print("❌ AutoGen manager is None (using fallback)")
            return False
        
        print("✅ AutoGen manager initialized successfully")
        print(f"   Type: {type(autogen_manager).__name__}")
        
        # Check if it's the real AutoGen or fallback
        if hasattr(autogen_manager, 'analyze_task'):
            print("✅ AutoGen has analyze_task method")
            
            # Try to analyze a simple task
            print("\n[4] Testing Task Analysis...")
            task_analysis = await autogen_manager.analyze_task(
                "Create a simple marketing email for a new product launch"
            )
            
            print("✅ Task analysis completed")
            print(f"   Complexity: {task_analysis.get('complexity', 'unknown')}")
            print(f"   Required agents: {len(task_analysis.get('required_agents', []))}")
            print(f"   Subtasks: {len(task_analysis.get('subtasks', []))}")
            
            return True
        else:
            print("⚠️  AutoGen manager doesn't have analyze_task method (might be fallback)")
            return False
            
    except Exception as e:
        print(f"❌ AutoGen initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the async test
print("\nRunning async test...")
result = asyncio.run(test_autogen())

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

if result:
    print("✅ AUTOGEN ORCHESTRATION TEST PASSED")
    print("\nAutoGen Status:")
    print("  ✅ AutoGen package installed and available")
    print("  ✅ AutoGen manager initializes successfully")
    print("  ✅ Task analysis works (NOT using fallback)")
    print("\n🎉 Master Agent can use full AutoGen multi-agent orchestration!")
    sys.exit(0)
else:
    print("❌ AUTOGEN ORCHESTRATION TEST FAILED")
    print("\nAutoGen may be using fallback implementation")
    sys.exit(1)

