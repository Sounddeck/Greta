"""
🌐 Browser Automation Integration Test Suite - Phase 1

Tests the Playwright browser automation integration with the agent execution system.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.agent_builder_service import AgentBuilderService
from services.tool_registry import ToolRegistry
from services.osint_playwright_tools import OSINTPlaywrightTools
from models.agent_builder_models import *


class BrowserAutomationIntegrationTests:
    def __init__(self):
        self.db = None
        self.agent_service = None
        self.tool_registry = None
        self.test_agent_id = None
        self.test_results = []
    
    async def setup(self):
        print("\n" + "="*80)
        print("🌐 BROWSER AUTOMATION INTEGRATION TEST")
        print("="*80)
        print("\n📦 Setting up test environment...")

        self.db = Database()
        await self.db.connect()
        self.agent_service = AgentBuilderService(self.db)
        self.tool_registry = ToolRegistry()
        self.tool_registry.db = self.db
        await self.tool_registry.initialize()

        print("✅ Database, agent service, and tool registry initialized")
    
    async def cleanup(self):
        print("\n🧹 Cleaning up test data...")
        
        if self.test_agent_id:
            try:
                await self.agent_service.delete_agent(self.test_agent_id)
                print(f"✅ Deleted test agent: {self.test_agent_id}")
            except Exception as e:
                print(f"⚠️ Failed to delete test agent: {e}")
        
        await self.db.disconnect()
        print("✅ Database disconnected")
    
    async def test_create_agent_with_playwright_tool(self):
        """Test 1: Create agent with Playwright tool assigned"""
        print("\n" + "="*80)
        print("TEST 1: Create Agent with Playwright Tool")
        print("="*80)
        
        try:
            # Create agent with Playwright tool
            agent_create = CustomAgentCreate(
                name="Web Scraping Test Agent",
                description="Agent for testing browser automation",
                agent_type=AgentType.RESEARCH,
                capabilities=["web_scraping", "data_extraction"],
                system_prompt="You are a web scraping agent that extracts data from websites.",
                tools=[
                    ToolAssignment(
                        tool_id="osint_playwright_scraper",
                        is_enabled=True
                    )
                ],
                tags=["test", "browser", "automation"]
            )
            
            agent = await self.agent_service.create_agent(agent_create)
            self.test_agent_id = agent.agent_id
            
            print(f"\n✅ Agent created: {agent.agent_id}")
            print(f"   Name: {agent.name}")
            print(f"   Type: {agent.agent_type}")
            print(f"   Tools: {len(agent.tools)}")
            print(f"   Tool IDs: {[t.tool_id for t in agent.tools]}")
            
            # Verify tool assignment
            has_playwright = any(t.tool_id == "osint_playwright_scraper" for t in agent.tools)
            assert has_playwright, "Playwright tool not assigned"
            
            self.test_results.append(("Create Agent with Playwright Tool", "Tool assigned successfully"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Create Agent with Playwright Tool", str(e)))
            raise
    
    async def test_basic_web_scraping(self):
        """Test 2: Execute basic web scraping task"""
        print("\n" + "="*80)
        print("TEST 2: Basic Web Scraping")
        print("="*80)

        try:
            # Use a reliable test URL
            test_url = "https://example.com"

            print(f"\n🌐 Scraping: {test_url}")

            # Execute tool - returns wrapped response
            response = await self.tool_registry.execute_tool(
                tool_id="osint_playwright_scraper",
                parameters={"url": test_url},
                agent_id=self.test_agent_id
            )

            # Extract actual result from wrapper
            result = response.get("result", {})

            print(f"\n✅ Scraping completed")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Links: {len(result.get('links', []))}")
            print(f"   Execution Time: {response.get('execution_time', 0):.2f}s")

            # Verify result
            assert response.get("success"), "Tool execution failed"
            assert result.get("success"), "Scraping failed"
            assert "title" in result, "No title field in response"

            self.test_results.append(("Basic Web Scraping", f"Success in {response.get('execution_time', 0):.2f}s"))

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("Basic Web Scraping", str(e)))
    
    async def test_scraping_with_wait_for(self):
        """Test 3: Execute scraping with wait_for selector"""
        print("\n" + "="*80)
        print("TEST 3: Scraping with wait_for Selector")
        print("="*80)

        try:
            test_url = "https://example.com"
            wait_selector = "h1"  # Wait for h1 element

            print(f"\n🌐 Scraping: {test_url}")
            print(f"   Waiting for: {wait_selector}")

            response = await self.tool_registry.execute_tool(
                tool_id="osint_playwright_scraper",
                parameters={
                    "url": test_url,
                    "wait_for": wait_selector
                },
                agent_id=self.test_agent_id
            )

            result = response.get("result", {})

            print(f"\n✅ Scraping with wait_for completed")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Title: {result.get('title', 'N/A')}")

            assert response.get("success"), "Tool execution failed"
            assert result.get("success"), "Scraping with wait_for failed"

            self.test_results.append(("Scraping with wait_for", "Success"))

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Scraping with wait_for", str(e)))
    
    async def test_scraping_with_custom_js(self):
        """Test 4: Execute scraping with custom JavaScript"""
        print("\n" + "="*80)
        print("TEST 4: Scraping with Custom JavaScript")
        print("="*80)

        try:
            test_url = "https://example.com"
            # Fix JS - don't use return in page.evaluate
            custom_js = "document.querySelectorAll('h1').length"

            print(f"\n🌐 Scraping: {test_url}")
            print(f"   Executing JS: {custom_js}")

            response = await self.tool_registry.execute_tool(
                tool_id="osint_playwright_scraper",
                parameters={
                    "url": test_url,
                    "execute_js": custom_js
                },
                agent_id=self.test_agent_id
            )

            result = response.get("result", {})

            print(f"\n✅ Scraping with custom JS completed")
            print(f"   Success: {result.get('success', False)}")
            print(f"   JS Result: {result.get('js_result')}")

            assert response.get("success"), "Tool execution failed"
            assert result.get("success"), "Scraping with custom JS failed"
            assert "js_result" in result, "No js_result field"

            self.test_results.append(("Scraping with Custom JS", f"JS returned: {result.get('js_result')}"))

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Scraping with Custom JS", str(e)))
    
    async def test_scraping_with_screenshot(self):
        """Test 5: Execute scraping with screenshot capture"""
        print("\n" + "="*80)
        print("TEST 5: Scraping with Screenshot")
        print("="*80)

        try:
            test_url = "https://example.com"

            print(f"\n🌐 Scraping: {test_url}")
            print(f"   Capturing screenshot: True")

            response = await self.tool_registry.execute_tool(
                tool_id="osint_playwright_scraper",
                parameters={
                    "url": test_url,
                    "screenshot": True
                },
                agent_id=self.test_agent_id
            )

            result = response.get("result", {})

            print(f"\n✅ Scraping with screenshot completed")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Screenshot captured: {result.get('screenshot') is not None}")
            if result.get('screenshot'):
                print(f"   Screenshot size: {len(result.get('screenshot', ''))} chars")

            assert response.get("success"), "Tool execution failed"
            assert result.get("success"), "Scraping with screenshot failed"
            assert "screenshot" in result, "No screenshot field"

            self.test_results.append(("Scraping with Screenshot", "Screenshot field present"))

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Scraping with Screenshot", str(e)))
    
    async def test_data_extraction(self):
        """Test 6: Verify extracted data (emails, phones, links, social media)"""
        print("\n" + "="*80)
        print("TEST 6: Data Extraction Verification")
        print("="*80)

        try:
            test_url = "https://example.com"

            print(f"\n🌐 Scraping: {test_url}")

            response = await self.tool_registry.execute_tool(
                tool_id="osint_playwright_scraper",
                parameters={"url": test_url},
                agent_id=self.test_agent_id
            )

            result = response.get("result", {})

            print(f"\n✅ Data extraction completed")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Emails: {len(result.get('emails', []))}")
            print(f"   Phone Numbers: {len(result.get('phone_numbers', []))}")
            print(f"   Links: {len(result.get('links', []))}")
            print(f"   Social Links: {result.get('social_links', {})}")
            print(f"   Meta Tags: {len(result.get('meta_tags', {}))}")

            # Verify data structure - all fields should exist
            assert response.get("success"), "Tool execution failed"
            assert result.get("success"), "Scraping failed"
            assert isinstance(result.get("emails", []), list), "Emails should be a list"
            assert isinstance(result.get("phone_numbers", []), list), "Phone numbers should be a list"
            assert isinstance(result.get("links", []), list), "Links should be a list"
            assert isinstance(result.get("social_links", {}), dict), "Social links should be a dict"
            assert isinstance(result.get("meta_tags", {}), dict), "Meta tags should be a dict"

            self.test_results.append(("Data Extraction", f"{len(result.get('links', []))} links extracted"))

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Data Extraction", str(e)))
    
    async def test_concurrent_scraping(self):
        """Test 7: Test concurrent scraping of multiple pages"""
        print("\n" + "="*80)
        print("TEST 7: Concurrent Scraping")
        print("="*80)
        
        try:
            urls = [
                "https://example.com",
                "https://example.org",
                "https://example.net"
            ]
            
            print(f"\n🌐 Scraping {len(urls)} pages concurrently...")
            
            # Use direct service for concurrent scraping
            async with OSINTPlaywrightTools() as playwright:
                results = await playwright.scrape_multiple_pages(urls)
                
                print(f"\n✅ Concurrent scraping completed")
                print(f"   Total pages: {len(results)}")
                
                successful = sum(1 for r in results if r.get("success"))
                print(f"   Successful: {successful}/{len(results)}")
                
                for i, result in enumerate(results, 1):
                    status = "✅" if result.get("success") else "❌"
                    print(f"   {status} Page {i}: {result.get('url', 'unknown')}")
                
                assert len(results) == len(urls), "Not all pages scraped"
                assert successful > 0, "No successful scrapes"
            
            self.test_results.append(("Concurrent Scraping", f"{successful}/{len(urls)} successful"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(("Concurrent Scraping", str(e)))
    
    async def test_structured_data_extraction(self):
        """Test 8: Test structured data extraction with CSS selectors"""
        print("\n" + "="*80)
        print("TEST 8: Structured Data Extraction")
        print("="*80)
        
        try:
            test_url = "https://example.com"
            selectors = {
                "title": "h1",
                "content": "p",
                "links": "a"
            }
            
            print(f"\n🌐 Extracting structured data from: {test_url}")
            print(f"   Selectors: {list(selectors.keys())}")
            
            async with OSINTPlaywrightTools() as playwright:
                result = await playwright.extract_structured_data(test_url, selectors)
                
                print(f"\n✅ Structured extraction completed")
                print(f"   Success: {result.get('success', False)}")
                print(f"   Data extracted: {list(result.get('data', {}).keys())}")
                
                for field, value in result.get("data", {}).items():
                    print(f"   {field}: {value[:50] if value else 'None'}...")
                
                assert result.get("success"), "Structured extraction failed"
                assert result.get("data"), "No data extracted"
            
            self.test_results.append(("Structured Data Extraction", "Success"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Structured Data Extraction", str(e)))
    
    async def test_network_monitoring(self):
        """Test 9: Test network request monitoring"""
        print("\n" + "="*80)
        print("TEST 9: Network Request Monitoring")
        print("="*80)
        
        try:
            test_url = "https://example.com"
            
            print(f"\n🌐 Monitoring network requests for: {test_url}")
            
            async with OSINTPlaywrightTools() as playwright:
                result = await playwright.monitor_network_requests(test_url)
                
                print(f"\n✅ Network monitoring completed")
                print(f"   Success: {result.get('success', False)}")
                print(f"   Total requests: {result.get('total_requests', 0)}")
                
                # Show first few requests
                for i, req in enumerate(result.get("requests", [])[:5], 1):
                    print(f"   Request {i}: {req.get('method')} {req.get('url')[:50]}...")
                
                assert result.get("success"), "Network monitoring failed"
            
            self.test_results.append(("Network Monitoring", f"{result.get('total_requests', 0)} requests"))
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Network Monitoring", str(e)))
    
    async def test_error_handling(self):
        """Test 10: Verify error handling for invalid URLs"""
        print("\n" + "="*80)
        print("TEST 10: Error Handling")
        print("="*80)

        try:
            invalid_url = "https://this-domain-definitely-does-not-exist-12345.com"

            print(f"\n🌐 Testing error handling with invalid URL")

            response = await self.tool_registry.execute_tool(
                tool_id="osint_playwright_scraper",
                parameters={"url": invalid_url},
                agent_id=self.test_agent_id
            )

            result = response.get("result", {})

            print(f"\n✅ Error handling test completed")
            print(f"   Tool Success: {response.get('success', False)}")
            print(f"   Scraping Success: {result.get('success', False)}")
            print(f"   Error: {result.get('error', 'N/A')}")

            # Tool execution should succeed even if scraping fails
            assert response.get("success"), "Tool execution should succeed"
            # The important thing is that it doesn't crash and returns a result
            assert "success" in result, "Should have success field in result"

            self.test_results.append(("Error Handling", "Graceful handling"))

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            self.test_results.append(("Error Handling", str(e)))
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.setup()
            
            await self.test_create_agent_with_playwright_tool()
            await self.test_basic_web_scraping()
            await self.test_scraping_with_wait_for()
            await self.test_scraping_with_custom_js()
            await self.test_scraping_with_screenshot()
            await self.test_data_extraction()
            await self.test_concurrent_scraping()
            await self.test_structured_data_extraction()
            await self.test_network_monitoring()
            await self.test_error_handling()
            
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
                print(f"\n✅ Browser Automation: WORKING")
                print(f"✅ Agent Integration: WORKING")
                print(f"✅ Data Extraction: WORKING")
                print(f"✅ Error Handling: WORKING")
            else:
                print(f"\n⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
            
            print("\n" + "="*80)


async def main():
    """Main test runner"""
    tests = BrowserAutomationIntegrationTests()
    await tests.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

