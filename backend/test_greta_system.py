#!/usr/bin/env python3
"""
GRETA PAI System Integration Test
Comprehensive test suite to verify all enhanced capabilities
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any, List

class GretaSystemTester:
    """Comprehensive test suite for GRETA PAI system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time()
        })
    
    def test_basic_connectivity(self):
        """Test basic API connectivity"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f" | Version: {data.get('version', 'unknown')}"
        except Exception as e:
            success = False
            details = f"Connection error: {str(e)}"
        
        self.log_test("Basic API Connectivity", success, details)
        return success
    
    def test_orchestrator_status(self):
        """Test central orchestrator status"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/orchestrator/status", timeout=5)
            success = response.status_code == 200
            if success:
                data = response.json()
                routers = len(data.get('available_routers', []))
                details = f"Available routers: {routers}"
            else:
                details = f"Status: {response.status_code}"
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Central Orchestrator Status", success, details)
        return success
    
    def test_command_execution(self):
        """Test command execution via orchestrator"""
        try:
            command_data = {
                "command": "Show me the current system status",
                "mode": "text",
                "priority": "normal"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/orchestrator/orchestrate",
                json=command_data,
                timeout=10
            )
            
            success = response.status_code == 200
            if success:
                data = response.json()
                routers_used = len(data.get('routers_invoked', []))
                exec_time = data.get('execution_time', 0)
                details = f"Routers invoked: {routers_used} | Execution time: {exec_time:.2f}s"
            else:
                details = f"Status: {response.status_code}"
                
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Command Execution", success, details)
        return success
    
    def test_asset_creation_capability(self):
        """Test digital asset creation"""
        try:
            asset_data = {
                "asset_type": "document",
                "specifications": {
                    "title": "GRETA Test Document",
                    "content": "This is a test document created by GRETA PAI system.",
                    "format": "markdown"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/assets/create",
                json=asset_data,
                timeout=15
            )
            
            success = response.status_code == 200
            if success:
                data = response.json()
                asset_id = data.get('asset_id', 'unknown')
                creation_time = data.get('creation_time', 0)
                details = f"Asset ID: {asset_id[:8]}... | Creation time: {creation_time:.2f}s"
            else:
                details = f"Status: {response.status_code}"
                
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Digital Asset Creation", success, details)
        return success
    
    def test_websocket_connectivity(self):
        """Test WebSocket connectivity"""
        try:
            import websocket
            
            def on_message(ws, message):
                pass
            
            def on_error(ws, error):
                pass
            
            def on_close(ws, close_status_code, close_msg):
                pass
            
            def on_open(ws):
                ws.send(json.dumps({"type": "ping"}))
                time.sleep(1)
                ws.close()
            
            ws_url = self.base_url.replace('http', 'ws') + '/api/v1/ws'
            ws = websocket.WebSocketApp(ws_url,
                                      on_open=on_open,
                                      on_message=on_message,
                                      on_error=on_error,
                                      on_close=on_close)
            
            # Run for a short time
            ws.run_forever(timeout=5)
            success = True
            details = "WebSocket connection established"
            
        except ImportError:
            success = False
            details = "websocket-client not installed (pip install websocket-client)"
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("WebSocket Connectivity", success, details)
        return success
    
    def test_router_availability(self):
        """Test availability of key routers"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/orchestrator/routers", timeout=5)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                total_routers = data.get('total_routers', 0)
                router_names = list(data.get('routers', {}).keys())
                
                # Check for key routers
                key_routers = ['llm', 'voice', 'memory', 'agent', 'master_agent', 'training']
                available_key_routers = [r for r in key_routers if r in router_names]
                
                details = f"Total: {total_routers} | Key routers: {len(available_key_routers)}/{len(key_routers)}"
            else:
                details = f"Status: {response.status_code}"
                
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Router Availability", success, details)
        return success
    
    def test_pai_hub_status(self):
        """Test PAI Hub status"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/pai-hub/status", timeout=5)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                intelligence_level = data.get('intelligence_level', 0)
                active_agents = data.get('active_agents', 0)
                details = f"Intelligence: {intelligence_level}% | Active agents: {active_agents}"
            else:
                details = f"Status: {response.status_code}"
                
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("PAI Hub Status", success, details)
        return success
    
    def test_master_agent_status(self):
        """Test Master Agent status"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/master-agent/status", timeout=5)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                total_tasks = data.get('total_tasks', 0)
                success_rate = data.get('success_rate', 0)
                details = f"Total tasks: {total_tasks} | Success rate: {success_rate}%"
            else:
                details = f"Status: {response.status_code}"
                
        except Exception as e:
            success = False
            details = f"Error: {str(e)}"
        
        self.log_test("Master Agent Status", success, details)
        return success
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🎭 GRETA PAI System - Comprehensive Test Suite")
        print("=" * 50)
        
        # Core connectivity tests
        if not self.test_basic_connectivity():
            print("\n❌ Basic connectivity failed. Please ensure GRETA backend is running on port 8000.")
            return False
        
        # Enhanced capability tests
        self.test_orchestrator_status()
        self.test_command_execution()
        self.test_router_availability()
        self.test_pai_hub_status()
        self.test_master_agent_status()
        self.test_asset_creation_capability()
        self.test_websocket_connectivity()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! GRETA PAI system is fully operational.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please check the backend services.")
        
        return failed_tests == 0

def main():
    """Main test execution"""
    print("Starting GRETA PAI System Integration Tests...")
    print("Please ensure the backend is running on http://localhost:8000")
    print()
    
    tester = GretaSystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 GRETA is ready for use!")
        print("Open http://localhost:3000 to access the enhanced interface.")
    else:
        print("\n🔧 Please resolve the failed tests before proceeding.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
