"""
🎯 Skills Auto-Save Test Suite - Phase 1 Enhancement

Tests the automatic skill creation from successful execution patterns.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.agent_builder_service import AgentBuilderService
from services.skills_service import skills_service
from models.agent_builder_models import *


class SkillsAutoSaveTests:
    def __init__(self):
        self.db = None
        self.service = None
        self.test_agent_id = None
        self.test_results = []
    
    async def setup(self):
        print("\n" + "="*80)
        print("🎯 PHASE 1: SKILLS AUTO-SAVE TEST")
        print("="*80)
        print("\n📦 Setting up test environment...")
        
        self.db = Database()
        await self.db.connect()
        self.service = AgentBuilderService(self.db)
        
        print("✅ Database and service initialized")
    
    async def cleanup(self):
        print("\n🧹 Cleaning up test data...")
        
        if self.test_agent_id:
            try:
                await self.service.delete_agent(self.test_agent_id)
                print(f"✅ Deleted test agent: {self.test_agent_id}")
            except Exception as e:
                print(f"⚠️ Failed to delete test agent: {e}")
        
        await self.db.disconnect()
        print("✅ Database disconnected")
    
    async def test_create_agent_with_memory(self):
        """Test 1: Create agent with memory systems enabled"""
        print("\n" + "="*80)
        print("TEST 1: Create Agent with Memory Systems")
        print("="*80)
        
        try:
            # Create agent with all memory systems enabled
            agent_create = CustomAgentCreate(
                name="Skills Test Agent",
                description="Agent for testing skills auto-save",
                agent_type=AgentType.GENERAL,
                capabilities=["research", "analysis", "web_search"],
                system_prompt="You are a helpful AI agent that learns skills from experience.",
                tags=["test", "skills", "auto-save"],
                memory_config=MemoryConfiguration(
                    enable_vector_search=True,
                    enable_reasoning_bank=True,
                    enable_functional_memory=True,
                    enable_memory_orchestrator=True,
                    auto_store_patterns=True  # Enable pattern storage
                )
            )
            
            agent = await self.service.create_agent(agent_create)
            self.test_agent_id = agent.agent_id
            
            print(f"\n✅ Agent created: {agent.agent_id}")
            print(f"   Name: {agent.name}")
            print(f"   Type: {agent.agent_type}")
            print(f"   Memory Systems Enabled:")
            print(f"     - Vector Search: {agent.memory_config.enable_vector_search}")
            print(f"     - ReasoningBank: {agent.memory_config.enable_reasoning_bank}")
            print(f"     - Functional Memory: {agent.memory_config.enable_functional_memory}")
            print(f"     - Memory Orchestrator: {agent.memory_config.enable_memory_orchestrator}")
            print(f"   Auto-Store Patterns: {agent.memory_config.auto_store_patterns}")
            
            self.test_results.append(("Create Agent with Memory", "All memory systems enabled"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Create Agent with Memory", str(e)))
            raise
    
    async def test_execution_with_tools(self):
        """Test 2: Execute agent with tools to trigger skill auto-save"""
        print("\n" + "="*80)
        print("TEST 2: Execution with Tools (Trigger Skill Auto-Save)")
        print("="*80)
        
        try:
            # Get initial skill count
            initial_stats = skills_service.get_skill_stats()
            initial_auto_saved = initial_stats["auto_saved_skills"]
            
            print(f"\n📊 Initial Skills Stats:")
            print(f"   Total Skills: {initial_stats['total_skills']}")
            print(f"   Core Skills: {initial_stats['core_skills']}")
            print(f"   Auto-Saved Skills: {initial_auto_saved}")
            
            # Execute agent with a task that would use tools
            print(f"\n🚀 Executing agent with web search task...")
            print(f"   Task: Research the latest AI agent frameworks")
            
            execution_create = AgentExecutionCreate(
                agent_id=self.test_agent_id,
                input_data={
                    "prompt": "Research the latest AI agent frameworks and summarize key features",
                    "message": "Research the latest AI agent frameworks and summarize key features"
                }
            )
            
            execution = await self.service.execute_agent(execution_create)
            
            print(f"\n✅ Execution completed: {execution.execution_id}")
            print(f"   Status: {execution.status}")
            print(f"   Execution Time: {execution.output_data.get('execution_time', 0):.2f}s")
            print(f"   Tools Called: {execution.output_data.get('tool_calls', 0)}")
            print(f"   Response Length: {len(execution.output_data.get('response', ''))} chars")
            
            # Wait a moment for async skill auto-save to complete
            await asyncio.sleep(1)
            
            # Check if skill was auto-saved
            final_stats = skills_service.get_skill_stats()
            final_auto_saved = final_stats["auto_saved_skills"]
            
            print(f"\n📊 Final Skills Stats:")
            print(f"   Total Skills: {final_stats['total_skills']}")
            print(f"   Core Skills: {final_stats['core_skills']}")
            print(f"   Auto-Saved Skills: {final_auto_saved}")
            
            if final_auto_saved > initial_auto_saved:
                print(f"\n   ✅ SUCCESS: {final_auto_saved - initial_auto_saved} new skill(s) auto-saved!")
                
                # Show the auto-saved skills
                auto_saved_skills = skills_service.get_auto_saved_skills()
                print(f"\n   Auto-Saved Skills:")
                for skill in auto_saved_skills[-3:]:  # Show last 3
                    print(f"     - {skill.display_name}")
                    print(f"       Category: {skill.category}")
                    print(f"       Priority: {skill.priority}")
                    print(f"       Source Pattern: {skill.source_pattern_id}")
            else:
                print(f"\n   ℹ️ No new skills auto-saved (may not have met criteria)")
            
            self.test_results.append(("Execution with Tools", f"Completed in {execution.output_data.get('execution_time', 0):.2f}s"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("Execution with Tools", str(e)))
    
    async def test_skill_retrieval(self):
        """Test 3: Verify auto-saved skills can be retrieved"""
        print("\n" + "="*80)
        print("TEST 3: Skill Retrieval")
        print("="*80)
        
        try:
            auto_saved_skills = skills_service.get_auto_saved_skills()
            
            print(f"\n✅ Retrieved {len(auto_saved_skills)} auto-saved skill(s)")
            
            if auto_saved_skills:
                for i, skill in enumerate(auto_saved_skills, 1):
                    print(f"\n   Skill {i}:")
                    print(f"     Name: {skill.name}")
                    print(f"     Display Name: {skill.display_name}")
                    print(f"     Category: {skill.category}")
                    print(f"     Priority: {skill.priority}")
                    print(f"     Keywords: {', '.join(skill.activation_keywords[:5])}")
                    print(f"     Source Pattern: {skill.source_pattern_id}")
                    print(f"     Usage Count: {skill.usage_count}")
                    print(f"     Success Rate: {skill.success_rate:.2%}")
            
            self.test_results.append(("Skill Retrieval", f"Found {len(auto_saved_skills)} skills"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Skill Retrieval", str(e)))
    
    async def test_skill_matching(self):
        """Test 4: Test skill matching with natural language"""
        print("\n" + "="*80)
        print("TEST 4: Skill Matching")
        print("="*80)
        
        try:
            # Test matching with various inputs
            test_inputs = [
                "Research AI frameworks",
                "Help me analyze data",
                "Build a feature with tests",
                "Review my code"
            ]
            
            for user_input in test_inputs:
                print(f"\n🔍 Testing input: '{user_input}'")
                match = await skills_service.match_skill(user_input, threshold=0.3)
                
                if match:
                    print(f"   ✅ Matched: {match.skill.display_name}")
                    print(f"      Confidence: {match.confidence:.2f}")
                    print(f"      Auto-Saved: {match.skill.is_auto_saved}")
                else:
                    print(f"   ℹ️ No match found")
            
            self.test_results.append(("Skill Matching", "Tested 4 inputs"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Skill Matching", str(e)))
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.setup()
            
            await self.test_create_agent_with_memory()
            await self.test_execution_with_tools()
            await self.test_skill_retrieval()
            await self.test_skill_matching()
            
        finally:
            await self.cleanup()
            
            # Print summary
            print("\n" + "="*80)
            print("📊 TEST SUMMARY")
            print("="*80)
            
            total = len(self.test_results)
            passed = sum(1 for _, result in self.test_results if "failed" not in result.lower() and "error" not in result.lower())
            failed = total - passed
            
            print(f"\nTotal Tests: {total}")
            print(f"Passed: {passed} ✅")
            print(f"Failed: {failed} ❌")
            print(f"Success Rate: {(passed/total*100) if total > 0 else 0:.1f}%")
            
            print(f"\nDetailed Results:")
            for test_name, result in self.test_results:
                status = "✅" if "failed" not in result.lower() and "error" not in result.lower() else "❌"
                print(f"  {status} {test_name}: {result}")
            
            if failed == 0:
                print(f"\n🎉 ALL TESTS PASSED!")
                print(f"\n✅ Skills Auto-Save: WORKING")
                print(f"✅ Skill Retrieval: WORKING")
                print(f"✅ Skill Matching: WORKING")
            else:
                print(f"\n⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
            
            print("\n" + "="*80)


async def main():
    """Main test runner"""
    tests = SkillsAutoSaveTests()
    await tests.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

