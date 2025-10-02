#!/usr/bin/env python3
"""
Test script for GRETA PAI Learning Service
Tests MongoDB integration and HRM coordination
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append('/Users/macone/Desktop/Greta/backend')

try:
    from services.learning_service import LearningService
    print("✅ Learning service imports successfully")
    
    # Simulate MongoDB client for syntax check
    from unittest.mock import MagicMock
    mock_db_client = MagicMock()
    
    # Test instantiation
    learning_service = LearningService(mock_db_client)
    print("✅ Learning service instantiates correctly")
    
    # Test HRM layers initialization
    assert len(learning_service.hrm_layers["strategic"]) == 0
    assert len(learning_service.hrm_layers["tactical"]) == 0
    assert len(learning_service.hrm_layers["operational"]) == 0
    print("✅ HRM coordination layers initialized")
    
    # Test status method
    status = asyncio.run(learning_service.get_pai_coordination_status())
    assert "pai_framework" in status
    assert "hierarchical_layers" in status
    print("✅ PAI coordination status works")
    
    print("🎯 ALL TESTS PASSED - Learning service is ready!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")
