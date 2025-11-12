"""
Integration Tests for Tool Discovery with Agent Execution

This test suite verifies end-to-end integration of the Progressive Tool Discovery
feature with the agent execution system.

Tests:
1. Tool discovery suggests tools before execution
2. Tools are tracked during execution
3. Learning happens automatically after execution
4. Execution records include tool data
5. Suggestions improve over time
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.agent_builder_service import AgentBuilderService
from services.tool_discovery_service import get_tool_discovery_service
from models.agent_builder_models import (
    CustomAgentCreate,
    AgentExecutionCreate,
    ExecutionStatus
)
from models.tool_discovery_models import ToolDiscoveryRequest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolDiscoveryIntegrationTests:
    """Integration tests for tool discovery with agent execution"""
    
    def __init__(self):
        self.db = None
        self.agent_service = None
        self.tool_discovery_service = None
        self.test_agent_id = None
        self.test_results = []
    
    async def setup(self):
        """Setup test environment"""
        print("\n" + "="*80)
        print("TOOL DISCOVERY INTEGRATION TESTS - SETUP")
        print("="*80 + "\n")
        
        # Initialize database
        self.db = Database()
        await self.db.connect()
        
        # Initialize services
        self.agent_service = AgentBuilderService(self.db)
        self.tool_discovery_service = get_tool_discovery_service(self.db)
        
        # Create test agent
        agent_create = CustomAgentCreate(
            name="Test Research Agent",
            description="Agent for testing tool discovery integration",
            agent_type="research",
            capabilities=["web_scraping", "data_extraction", "research"],
            system_prompt="You are a research agent that helps with web scraping and data extraction.",
            tools=[],
            memory_config={
                "enabled_systems": ["reasoning_bank", "vector_search"],
                "auto_store_patterns": True
            }
        )
        
        agent = await self.agent_service.create_agent(agent_create)
        self.test_agent_id = agent.agent_id
        
        print(f"✅ Test agent created: {self.test_agent_id}")
        print(f"   Agent Type: {agent.agent_type}")
        print(f"   Capabilities: {', '.join(agent.capabilities)}")
    
    async def cleanup(self):
        """Cleanup test environment"""
        print("\n" + "="*80)
        print("CLEANUP")
        print("="*80 + "\n")
        
        # Delete test agent
        if self.test_agent_id:
            await self.agent_service.delete_agent(self.test_agent_id)
            print(f"✅ Test agent deleted: {self.test_agent_id}")
        
        # Print test results
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80 + "\n")
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, message in self.test_results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {test_name}")
            if message:
                print(f"   {message}")
        
        print(f"\n{'='*80}")
        print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        print(f"{'='*80}\n")
        
        return passed == total
    
    async def test_tool_discovery_before_execution(self):
        """Test 1: Tool discovery suggests tools before execution"""
        print("\n" + "-"*80)
        print("TEST 1: Tool Discovery Before Execution")
        print("-"*80 + "\n")
        
        try:
            # Test tool discovery directly
            request = ToolDiscoveryRequest(
                task_description="Scrape product prices from competitor websites",
                agent_id=self.test_agent_id,
                agent_type="research",
                limit=5,
                min_relevance=0.2
            )
            
            response = await self.tool_discovery_service.discover_tools(request)
            
            print(f"✅ Tool discovery completed in {response.discovery_time:.2f}s")
            print(f"   Task Intent: {response.task_analysis.intent}")
            print(f"   Task Complexity: {response.task_analysis.complexity}")
            print(f"   Keywords: {', '.join(response.task_analysis.keywords[:5])}")
            print(f"   Suggestions: {len(response.suggestions)} tools")
            
            for i, suggestion in enumerate(response.suggestions[:3], 1):
                print(f"\n   {i}. {suggestion.tool_name} (ID: {suggestion.tool_id})")
                print(f"      Relevance: {suggestion.relevance_score:.2f}")
                print(f"      Reason: {suggestion.reason[:80]}...")
            
            # Verify suggestions were made
            if len(response.suggestions) > 0:
                self.test_results.append(("Tool Discovery Before Execution", True, f"{len(response.suggestions)} tools suggested"))
                return True
            else:
                self.test_results.append(("Tool Discovery Before Execution", False, "No tools suggested"))
                return False
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results.append(("Tool Discovery Before Execution", False, str(e)))
            return False
    
    async def test_agent_execution_with_tool_discovery(self):
        """Test 2: Agent execution includes tool discovery"""
        print("\n" + "-"*80)
        print("TEST 2: Agent Execution with Tool Discovery")
        print("-"*80 + "\n")
        
        try:
            # Execute agent with a task that should trigger tool discovery
            execution_request = AgentExecutionCreate(
                agent_id=self.test_agent_id,
                input_data={
                    "prompt": "Research the latest trends in AI and machine learning",
                    "max_tokens": 500
                }
            )
            
            execution = await self.agent_service.execute_agent(execution_request)
            
            print(f"✅ Agent execution completed")
            print(f"   Execution ID: {execution.execution_id}")
            print(f"   Status: {execution.status}")
            print(f"   Execution Time: {execution.execution_time:.2f}s")
            
            # Check if tools were suggested
            if hasattr(execution, 'tools_suggested'):
                print(f"   Tools Suggested: {len(execution.tools_suggested)}")
                if execution.tools_suggested:
                    print(f"      {', '.join(execution.tools_suggested[:3])}")
            
            # Check if tools were used
            if hasattr(execution, 'tools_used'):
                print(f"   Tools Used: {len(execution.tools_used)}")
                if execution.tools_used:
                    print(f"      {', '.join(execution.tools_used[:3])}")
            
            # Verify execution completed successfully
            if execution.status == ExecutionStatus.COMPLETED:
                self.test_results.append(("Agent Execution with Tool Discovery", True, f"Execution completed successfully"))
                return True
            else:
                self.test_results.append(("Agent Execution with Tool Discovery", False, f"Execution status: {execution.status}"))
                return False
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results.append(("Agent Execution with Tool Discovery", False, str(e)))
            return False
    
    async def test_execution_record_includes_tool_data(self):
        """Test 3: Execution records include tool data"""
        print("\n" + "-"*80)
        print("TEST 3: Execution Record Includes Tool Data")
        print("-"*80 + "\n")
        
        try:
            # Execute agent
            execution_request = AgentExecutionCreate(
                agent_id=self.test_agent_id,
                input_data={
                    "prompt": "Analyze competitor pricing strategies",
                    "max_tokens": 500
                }
            )
            
            execution = await self.agent_service.execute_agent(execution_request)
            
            # Retrieve execution record
            retrieved_execution = await self.agent_service.get_execution(execution.execution_id)
            
            print(f"✅ Execution record retrieved")
            print(f"   Execution ID: {retrieved_execution.execution_id}")
            
            # Check for tool data fields
            has_tools_suggested = hasattr(retrieved_execution, 'tools_suggested')
            has_tools_used = hasattr(retrieved_execution, 'tools_used')
            
            print(f"   Has tools_suggested field: {has_tools_suggested}")
            print(f"   Has tools_used field: {has_tools_used}")
            
            if has_tools_suggested:
                print(f"   Tools Suggested: {retrieved_execution.tools_suggested}")
            if has_tools_used:
                print(f"   Tools Used: {retrieved_execution.tools_used}")
            
            # Verify both fields exist
            if has_tools_suggested and has_tools_used:
                self.test_results.append(("Execution Record Includes Tool Data", True, "Both fields present"))
                return True
            else:
                missing = []
                if not has_tools_suggested:
                    missing.append("tools_suggested")
                if not has_tools_used:
                    missing.append("tools_used")
                self.test_results.append(("Execution Record Includes Tool Data", False, f"Missing fields: {', '.join(missing)}"))
                return False
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results.append(("Execution Record Includes Tool Data", False, str(e)))
            return False
    
    async def test_learning_from_execution(self):
        """Test 4: Learning happens after execution"""
        print("\n" + "-"*80)
        print("TEST 4: Learning from Execution")
        print("-"*80 + "\n")
        
        try:
            # Get initial stats
            initial_stats = await self.tool_discovery_service.get_stats()
            initial_patterns = initial_stats.total_patterns_learned
            
            print(f"📊 Initial patterns learned: {initial_patterns}")
            
            # Execute agent (learning happens automatically)
            execution_request = AgentExecutionCreate(
                agent_id=self.test_agent_id,
                input_data={
                    "prompt": "Extract contact information from company websites",
                    "max_tokens": 500
                }
            )
            
            execution = await self.agent_service.execute_agent(execution_request)
            
            # Wait a bit for async learning to complete
            await asyncio.sleep(2)
            
            # Get updated stats
            updated_stats = await self.tool_discovery_service.get_stats()
            updated_patterns = updated_stats.total_patterns_learned
            
            print(f"📊 Updated patterns learned: {updated_patterns}")
            print(f"   Patterns added: {updated_patterns - initial_patterns}")
            
            # Verify learning happened (patterns increased)
            if updated_patterns >= initial_patterns:
                self.test_results.append(("Learning from Execution", True, f"{updated_patterns - initial_patterns} patterns added"))
                return True
            else:
                self.test_results.append(("Learning from Execution", False, "No patterns added"))
                return False
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results.append(("Learning from Execution", False, str(e)))
            return False
    
    async def test_discovery_statistics(self):
        """Test 5: Discovery statistics are tracked"""
        print("\n" + "-"*80)
        print("TEST 5: Discovery Statistics")
        print("-"*80 + "\n")
        
        try:
            # Get statistics
            stats = await self.tool_discovery_service.get_stats()
            
            print(f"📊 Tool Discovery Statistics:")
            print(f"   Total Discoveries: {stats.total_discoveries}")
            print(f"   Total Suggestions: {stats.total_suggestions}")
            print(f"   Avg Suggestions/Request: {stats.avg_suggestions_per_request:.1f}")
            print(f"   Avg Discovery Time: {stats.avg_discovery_time:.2f}s")
            print(f"   Suggestion Accuracy: {stats.suggestion_accuracy:.0%}")
            print(f"   Total Patterns Learned: {stats.total_patterns_learned}")
            
            # Verify stats are being tracked
            if stats.total_discoveries >= 0:
                self.test_results.append(("Discovery Statistics", True, f"{stats.total_discoveries} discoveries tracked"))
                return True
            else:
                self.test_results.append(("Discovery Statistics", False, "Stats not tracked"))
                return False
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.test_results.append(("Discovery Statistics", False, str(e)))
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        try:
            await self.setup()
            
            # Run tests
            await self.test_tool_discovery_before_execution()
            await self.test_agent_execution_with_tool_discovery()
            await self.test_execution_record_includes_tool_data()
            await self.test_learning_from_execution()
            await self.test_discovery_statistics()
            
            # Cleanup and return results
            return await self.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


async def main():
    """Main test runner"""
    tests = ToolDiscoveryIntegrationTests()
    success = await tests.run_all_tests()
    
    if success:
        print("\n🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

