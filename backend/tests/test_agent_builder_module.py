"""
Agent Builder Module Test Suite
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.agent_builder_service import AgentBuilderService
from models.agent_builder_models import *

class AgentBuilderTests:
    def __init__(self):
        self.db = None
        self.service = None
        self.test_results = []
        self.test_template_id = None
        self.test_agent_id = None
        self.test_version_id = None
        self.test_deployment_id = None
        self.test_execution_id = None
    
    async def setup(self):
        print("\n" + "="*80)
        print("AGENT BUILDER MODULE TEST SUITE")
        print("="*80)
        self.db = Database()
        await self.db.connect()
        self.service = AgentBuilderService(self.db)
        print("✅ Test environment initialized")
    
    async def test_template_management(self):
        print("\n📋 Testing Template Management...")
        try:
            tc = AgentTemplateCreate(name="Research Agent Template", description="Template for research agents", agent_type=AgentType.RESEARCH, capabilities=["web_search", "data_analysis"], default_tools=["search", "scraper"], default_prompt="You are a research agent", tags=["research", "template"])
            template = await self.service.create_template(tc)
            self.test_template_id = template.template_id
            assert template.name == "Research Agent Template"
            print(f"  ✅ Template created: {template.template_id}")
            
            retrieved = await self.service.get_template(self.test_template_id)
            assert retrieved is not None
            print(f"  ✅ Template retrieved: {retrieved.name}")
            
            tu = AgentTemplateUpdate(description="Updated template description")
            updated = await self.service.update_template(self.test_template_id, tu)
            assert updated.description == "Updated template description"
            print(f"  ✅ Template updated")
            
            templates = await self.service.list_templates(agent_type=AgentType.RESEARCH)
            assert len(templates) > 0
            print(f"  ✅ Templates listed: {len(templates)}")
            
            self.test_results.append(("Template Management", True, "All CRUD operations working"))
            return True
        except Exception as e:
            print(f"  ❌ Template management test failed: {e}")
            self.test_results.append(("Template Management", False, str(e)))
            return False
    
    async def test_agent_creation(self):
        print("\n🤖 Testing Agent Creation...")
        try:
            ac = CustomAgentCreate(name="My Research Agent", description="Custom research agent", agent_type=AgentType.RESEARCH, capabilities=["web_search", "summarization"], system_prompt="You are a helpful research assistant", tools=[], configuration={"max_results": 10}, tags=["research", "custom"], template_id=self.test_template_id, status=AgentStatus.DRAFT)
            agent = await self.service.create_agent(ac)
            self.test_agent_id = agent.agent_id
            assert agent.name == "My Research Agent"
            assert agent.status == AgentStatus.DRAFT
            print(f"  ✅ Agent created: {agent.agent_id}")
            
            retrieved = await self.service.get_agent(self.test_agent_id)
            assert retrieved is not None
            print(f"  ✅ Agent retrieved: {retrieved.name}")
            
            au = CustomAgentUpdate(status=AgentStatus.TESTING)
            updated = await self.service.update_agent(self.test_agent_id, au)
            assert updated.status == AgentStatus.TESTING
            print(f"  ✅ Agent updated: status={updated.status}")
            
            agents = await self.service.list_agents(status=AgentStatus.TESTING)
            assert len(agents) > 0
            print(f"  ✅ Agents listed: {len(agents)}")
            
            self.test_results.append(("Agent Creation", True, "All CRUD operations working"))
            return True
        except Exception as e:
            print(f"  ❌ Agent creation test failed: {e}")
            self.test_results.append(("Agent Creation", False, str(e)))
            return False
    
    async def test_agent_cloning(self):
        print("\n📋 Testing Agent Cloning...")
        try:
            cloned = await self.service.clone_agent(self.test_agent_id, "Cloned Research Agent")
            assert cloned is not None
            assert cloned.name == "Cloned Research Agent"
            assert cloned.agent_id != self.test_agent_id
            print(f"  ✅ Agent cloned: {cloned.agent_id}")
            await self.service.delete_agent(cloned.agent_id)
            print(f"  ✅ Cloned agent deleted")
            self.test_results.append(("Agent Cloning", True, "Cloning working"))
            return True
        except Exception as e:
            print(f"  ❌ Agent cloning test failed: {e}")
            self.test_results.append(("Agent Cloning", False, str(e)))
            return False
    
    async def test_version_management(self):
        print("\n🔄 Testing Version Management...")
        try:
            vc = AgentVersionCreate(agent_id=self.test_agent_id, version_number=2, changes="Updated capabilities", configuration_snapshot={"capabilities": ["web_search", "summarization", "translation"]})
            version = await self.service.create_version(vc)
            self.test_version_id = version.version_id
            assert version.version_number == 2
            print(f"  ✅ Version created: {version.version_id}")
            
            versions = await self.service.list_versions(self.test_agent_id)
            assert len(versions) > 0
            print(f"  ✅ Versions listed: {len(versions)}")
            
            rolled_back = await self.service.rollback_version(self.test_agent_id, self.test_version_id)
            assert rolled_back is not None
            print(f"  ✅ Version rolled back")
            
            self.test_results.append(("Version Management", True, "Versioning working"))
            return True
        except Exception as e:
            print(f"  ❌ Version management test failed: {e}")
            self.test_results.append(("Version Management", False, str(e)))
            return False
    
    async def test_deployment(self):
        print("\n🚀 Testing Deployment...")
        try:
            dc = DeploymentCreate(agent_id=self.test_agent_id, version_id=self.test_version_id, environment="production", configuration={"replicas": 1})
            deployment = await self.service.deploy_agent(dc)
            self.test_deployment_id = deployment.deployment_id
            assert deployment.status == DeploymentStatus.DEPLOYED
            print(f"  ✅ Agent deployed: {deployment.deployment_id}")
            
            retrieved = await self.service.get_deployment(self.test_deployment_id)
            assert retrieved is not None
            print(f"  ✅ Deployment retrieved")
            
            deployments = await self.service.list_deployments(agent_id=self.test_agent_id)
            assert len(deployments) > 0
            print(f"  ✅ Deployments listed: {len(deployments)}")
            
            self.test_results.append(("Deployment", True, "Deployment working"))
            return True
        except Exception as e:
            print(f"  ❌ Deployment test failed: {e}")
            self.test_results.append(("Deployment", False, str(e)))
            return False
    
    async def test_execution(self):
        print("\n⚡ Testing Execution...")
        try:
            ec = AgentExecutionCreate(agent_id=self.test_agent_id, input_data={"query": "test query"}, configuration={})
            execution = await self.service.execute_agent(ec)
            self.test_execution_id = execution.execution_id
            assert execution.status == ExecutionStatus.COMPLETED
            print(f"  ✅ Agent executed: {execution.execution_id}")
            
            retrieved = await self.service.get_execution(self.test_execution_id)
            assert retrieved is not None
            print(f"  ✅ Execution retrieved")
            
            executions = await self.service.list_executions(agent_id=self.test_agent_id)
            assert len(executions) > 0
            print(f"  ✅ Executions listed: {len(executions)}")
            
            self.test_results.append(("Execution", True, "Execution working"))
            return True
        except Exception as e:
            print(f"  ❌ Execution test failed: {e}")
            self.test_results.append(("Execution", False, str(e)))
            return False
    
    async def test_code_generation(self):
        print("\n💻 Testing Code Generation...")
        try:
            req = CodeGenerationRequest(agent_id=self.test_agent_id, include_tools=True, include_tests=False)
            result = await self.service.generate_code(req)
            assert result.generated_code is not None
            assert len(result.generated_code) > 0
            assert "class" in result.generated_code
            print(f"  ✅ Code generated: {len(result.generated_code)} characters")
            print(f"  ✅ Code is valid: {result.is_valid}")
            self.test_results.append(("Code Generation", True, f"{len(result.generated_code)} chars generated"))
            return True
        except Exception as e:
            print(f"  ❌ Code generation test failed: {e}")
            self.test_results.append(("Code Generation", False, str(e)))
            return False
    
    async def test_validation(self):
        print("\n✅ Testing Validation...")
        try:
            result = await self.service.validate_agent(self.test_agent_id)
            assert result is not None
            print(f"  ✅ Agent validated: is_valid={result.is_valid}")
            print(f"  ✅ Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
            self.test_results.append(("Validation", True, f"Valid: {result.is_valid}"))
            return True
        except Exception as e:
            print(f"  ❌ Validation test failed: {e}")
            self.test_results.append(("Validation", False, str(e)))
            return False
    
    async def test_statistics(self):
        print("\n📊 Testing Statistics...")
        try:
            stats = await self.service.get_stats()
            assert stats.total_agents > 0
            assert stats.total_templates > 0
            print(f"  ✅ Total agents: {stats.total_agents}")
            print(f"  ✅ Total templates: {stats.total_templates}")
            print(f"  ✅ Total executions: {stats.total_executions}")
            print(f"  ✅ Success rate: {stats.success_rate}%")
            self.test_results.append(("Statistics", True, "All metrics calculated"))
            return True
        except Exception as e:
            print(f"  ❌ Statistics test failed: {e}")
            self.test_results.append(("Statistics", False, str(e)))
            return False
    
    async def cleanup(self):
        print("\n🧹 Cleaning up test data...")
        try:
            if self.test_agent_id:
                await self.service.delete_agent(self.test_agent_id)
                print(f"  ✅ Deleted agent: {self.test_agent_id}")
            if self.test_template_id:
                await self.service.delete_template(self.test_template_id)
                print(f"  ✅ Deleted template: {self.test_template_id}")
        except Exception as e:
            print(f"  ⚠️ Cleanup warning: {e}")
    
    def print_summary(self):
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {test_name}: {message}")
        print("\n" + "-"*80)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print("="*80)
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - AGENT BUILDER MODULE FULLY FUNCTIONAL")
            print("\n✅ Template Management: WORKING")
            print("✅ Agent Creation: WORKING")
            print("✅ Agent Cloning: WORKING")
            print("✅ Version Management: WORKING")
            print("✅ Deployment: WORKING")
            print("✅ Execution: WORKING")
            print("✅ Code Generation: WORKING")
            print("✅ Validation: WORKING")
            print("✅ Statistics: WORKING")
        else:
            print("\n⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        print("="*80 + "\n")
    
    async def run_all_tests(self):
        await self.setup()
        await self.test_template_management()
        await self.test_agent_creation()
        await self.test_agent_cloning()
        await self.test_version_management()
        await self.test_deployment()
        await self.test_execution()
        await self.test_code_generation()
        await self.test_validation()
        await self.test_statistics()
        await self.cleanup()
        self.print_summary()

async def main():
    tests = AgentBuilderTests()
    await tests.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
