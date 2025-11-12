"""
Test script to verify Master Agent orchestration package imports
Tests AutoGen, CrewAI, SmolAgents, and LangGraph imports

Run this script after installing orchestration packages to verify everything works.
"""
import sys
import traceback
from datetime import datetime

def test_import(package_name, import_statement, test_code=None):
    """Test a package import and optional functionality"""
    print(f"\n{'='*60}")
    print(f"Testing: {package_name}")
    print(f"{'='*60}")
    
    try:
        # Execute import
        exec(import_statement, globals())
        print(f"✅ Import successful: {import_statement}")
        
        # Execute test code if provided
        if test_code:
            exec(test_code, globals())
            print(f"✅ Functionality test passed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"   Package '{package_name}' is not installed")
        return False
        
    except Exception as e:
        print(f"⚠️  Import succeeded but functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all orchestration package import tests"""
    print("\n" + "="*60)
    print("MASTER AGENT ORCHESTRATION - PACKAGE IMPORT TESTS")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # Test 1: LangChain Core
    results['langchain-core'] = test_import(
        'langchain-core',
        'from langchain_core.messages import HumanMessage, AIMessage, SystemMessage',
        '''
msg = HumanMessage(content="Test message")
assert msg.content == "Test message"
print(f"   Created HumanMessage: {msg.content}")
'''
    )
    
    # Test 2: LangChain
    results['langchain'] = test_import(
        'langchain',
        'from langchain_core.prompts import ChatPromptTemplate',
        '''
template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant")])
print(f"   Created ChatPromptTemplate with {len(template.messages)} message(s)")
'''
    )
    
    # Test 3: LangChain Community
    results['langchain-community'] = test_import(
        'langchain-community',
        'from langchain_community.embeddings import OllamaEmbeddings',
        '''
print(f"   OllamaEmbeddings class available")
'''
    )
    
    # Test 4: LangGraph
    results['langgraph'] = test_import(
        'langgraph',
        'from langgraph.graph import StateGraph, START, END',
        '''
print(f"   StateGraph, START, END imported successfully")
'''
    )
    
    # Test 5: AutoGen (pyautogen)
    results['pyautogen'] = test_import(
        'pyautogen',
        'from autogen import ConversableAgent, GroupChat, GroupChatManager',
        '''
print(f"   ConversableAgent, GroupChat, GroupChatManager imported successfully")
'''
    )
    
    # Test 6: CrewAI
    results['crewai'] = test_import(
        'crewai',
        'from crewai import Crew, Agent, Task, Process',
        '''
print(f"   Crew, Agent, Task, Process imported successfully")
'''
    )
    
    # Test 7: SmolAgents
    results['smolagents'] = test_import(
        'smolagents',
        'from smolagents import CodeAgent, tool',
        '''
print(f"   CodeAgent, tool imported successfully")
'''
    )
    
    # Test 8: FAISS (for vector database)
    results['faiss'] = test_import(
        'faiss',
        'import faiss',
        '''
print(f"   FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
'''
    )
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for package, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10s} {package}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Check critical packages for Master Agent
    critical_packages = ['langchain-core', 'langgraph', 'pyautogen', 'crewai', 'smolagents']
    critical_passed = all(results.get(pkg, False) for pkg in critical_packages)
    
    if critical_passed:
        print("✅ ALL CRITICAL PACKAGES INSTALLED AND WORKING")
        print("   Master Agent can use full orchestration (no fallback mode)")
        return 0
    else:
        print("❌ SOME CRITICAL PACKAGES MISSING OR BROKEN")
        print("   Master Agent will use fallback implementations")
        missing = [pkg for pkg in critical_packages if not results.get(pkg, False)]
        print(f"   Missing/Broken: {', '.join(missing)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

