"""
Gmail Module Tests

Comprehensive test suite for Gmail API operations including
sending emails, reading messages, searching, and managing labels.

NOTE: These tests use mock data and do not require real Gmail API access.
"""

import asyncio
import unittest
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from models.gmail_models import (
    GmailSendRequest,
    GmailSearchRequest,
    GmailListRequest,
    GmailModifyRequest,
    GmailLabelCreate,
    GmailBatchModifyRequest,
    GmailBatchDeleteRequest,
)
from services.gmail_service import GmailService


class TestGmailModule(unittest.TestCase):
    """Test Gmail module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.db = Database()
        cls.service = GmailService(cls.db)
        cls.test_user_id = "test_gmail_user"
    
    def setUp(self):
        """Set up each test"""
        self.loop = asyncio.get_event_loop()
    
    def test_01_service_initialization(self):
        """Test 1: Gmail service initialization"""
        print("\n" + "="*80)
        print("TEST 1: Gmail Service Initialization")
        print("="*80)
        
        assert self.service is not None
        assert self.service.db is not None
        assert self.service.mock_mode is True
        
        print("✅ Gmail service initialized successfully")
        print(f"   - Mock mode: {self.service.mock_mode}")
        print(f"   - OAuth service: {self.service.oauth_service is not None}")
    
    def test_02_send_email(self):
        """Test 2: Send email"""
        print("\n" + "="*80)
        print("TEST 2: Send Email")
        print("="*80)
        
        async def run_test():
            request = GmailSendRequest(
                to="recipient@example.com",
                subject="Test Email from CPAS4",
                body="This is a test email sent via Gmail API",
                cc="cc@example.com",
                html=False
            )
            
            response = await self.service.send_email(self.test_user_id, request)
            
            assert response is not None
            assert response.success is True
            assert response.message_id is not None
            assert response.thread_id is not None
            assert "SENT" in response.label_ids
            
            print("✅ Email sent successfully")
            print(f"   - Message ID: {response.message_id}")
            print(f"   - Thread ID: {response.thread_id}")
            print(f"   - Label IDs: {response.label_ids}")
            print(f"   - Message: {response.message}")
            
            return response.message_id
        
        self.test_message_id = self.loop.run_until_complete(run_test())
    
    def test_03_send_html_email(self):
        """Test 3: Send HTML email"""
        print("\n" + "="*80)
        print("TEST 3: Send HTML Email")
        print("="*80)
        
        async def run_test():
            request = GmailSendRequest(
                to="recipient@example.com",
                subject="HTML Test Email",
                body="<h1>Hello</h1><p>This is an <strong>HTML</strong> email</p>",
                html=True
            )
            
            response = await self.service.send_email(self.test_user_id, request)
            
            assert response is not None
            assert response.success is True
            
            print("✅ HTML email sent successfully")
            print(f"   - Message ID: {response.message_id}")
        
        self.loop.run_until_complete(run_test())
    
    def test_04_list_emails(self):
        """Test 4: List emails"""
        print("\n" + "="*80)
        print("TEST 4: List Emails")
        print("="*80)
        
        async def run_test():
            request = GmailListRequest(
                max_results=10,
                label_ids=["INBOX"],
                include_spam_trash=False
            )
            
            response = await self.service.list_emails(self.test_user_id, request)
            
            assert response is not None
            assert response.messages is not None
            assert response.result_size_estimate >= 0
            
            print("✅ Emails listed successfully")
            print(f"   - Messages count: {len(response.messages)}")
            print(f"   - Result size estimate: {response.result_size_estimate}")
            print(f"   - Next page token: {response.next_page_token}")
            
            if response.messages:
                msg = response.messages[0]
                print(f"\n   First message:")
                print(f"   - Message ID: {msg.message_id}")
                print(f"   - Subject: {msg.subject}")
                print(f"   - From: {msg.from_email}")
                print(f"   - Snippet: {msg.snippet[:50]}...")
        
        self.loop.run_until_complete(run_test())
    
    def test_05_search_emails(self):
        """Test 5: Search emails"""
        print("\n" + "="*80)
        print("TEST 5: Search Emails")
        print("="*80)
        
        async def run_test():
            request = GmailSearchRequest(
                query="from:example@gmail.com has:attachment",
                max_results=20,
                include_spam_trash=False
            )
            
            response = await self.service.search_emails(self.test_user_id, request)
            
            assert response is not None
            assert response.messages is not None
            assert response.query == request.query
            
            print("✅ Email search successful")
            print(f"   - Query: {response.query}")
            print(f"   - Messages found: {len(response.messages)}")
            print(f"   - Result size estimate: {response.result_size_estimate}")
            
            if response.messages:
                msg = response.messages[0]
                print(f"\n   First result:")
                print(f"   - Message ID: {msg.message_id}")
                print(f"   - Subject: {msg.subject}")
                print(f"   - Snippet: {msg.snippet[:50]}...")
        
        self.loop.run_until_complete(run_test())
    
    def test_06_get_email(self):
        """Test 6: Get email by ID"""
        print("\n" + "="*80)
        print("TEST 6: Get Email by ID")
        print("="*80)
        
        async def run_test():
            message_id = "test_message_123"
            
            message = await self.service.get_email(self.test_user_id, message_id)
            
            assert message is not None
            assert message.message_id == message_id
            
            print("✅ Email retrieved successfully")
            print(f"   - Message ID: {message.message_id}")
            print(f"   - Thread ID: {message.thread_id}")
            print(f"   - Label IDs: {message.label_ids}")
            print(f"   - Snippet: {message.snippet}")
        
        self.loop.run_until_complete(run_test())
    
    def test_07_modify_email_labels(self):
        """Test 7: Modify email labels"""
        print("\n" + "="*80)
        print("TEST 7: Modify Email Labels")
        print("="*80)
        
        async def run_test():
            request = GmailModifyRequest(
                message_id="test_message_123",
                add_label_ids=["STARRED", "IMPORTANT"],
                remove_label_ids=["UNREAD"]
            )
            
            response = await self.service.modify_email(self.test_user_id, request)
            
            assert response is not None
            assert response.success is True
            assert response.message_id == request.message_id
            
            print("✅ Email labels modified successfully")
            print(f"   - Message ID: {response.message_id}")
            print(f"   - Thread ID: {response.thread_id}")
            print(f"   - Label IDs: {response.label_ids}")
            print(f"   - Message: {response.message}")
        
        self.loop.run_until_complete(run_test())
    
    def test_08_delete_email(self):
        """Test 8: Delete email"""
        print("\n" + "="*80)
        print("TEST 8: Delete Email")
        print("="*80)
        
        async def run_test():
            message_id = "test_message_to_delete"
            
            success = await self.service.delete_email(self.test_user_id, message_id)
            
            assert success is True
            
            print("✅ Email deleted successfully")
            print(f"   - Message ID: {message_id}")
            print(f"   - Success: {success}")
        
        self.loop.run_until_complete(run_test())
    
    def test_09_list_labels(self):
        """Test 9: List Gmail labels"""
        print("\n" + "="*80)
        print("TEST 9: List Gmail Labels")
        print("="*80)
        
        async def run_test():
            labels = await self.service.list_labels(self.test_user_id)
            
            assert labels is not None
            assert len(labels) > 0
            
            print("✅ Labels listed successfully")
            print(f"   - Labels count: {len(labels)}")
            
            for label in labels:
                print(f"\n   Label:")
                print(f"   - ID: {label.label_id}")
                print(f"   - Name: {label.name}")
                print(f"   - Type: {label.type}")
                print(f"   - Messages total: {label.messages_total}")
                print(f"   - Messages unread: {label.messages_unread}")
        
        self.loop.run_until_complete(run_test())
    
    def test_10_create_label(self):
        """Test 10: Create Gmail label"""
        print("\n" + "="*80)
        print("TEST 10: Create Gmail Label")
        print("="*80)
        
        async def run_test():
            request = GmailLabelCreate(
                name="Important Projects",
                label_list_visibility="labelShow",
                message_list_visibility="show"
            )
            
            response = await self.service.create_label(self.test_user_id, request)
            
            assert response is not None
            assert response.name == request.name
            assert response.type == "user"
            
            print("✅ Label created successfully")
            print(f"   - Label ID: {response.label_id}")
            print(f"   - Name: {response.name}")
            print(f"   - Type: {response.type}")
        
        self.loop.run_until_complete(run_test())
    
    def test_11_batch_modify_emails(self):
        """Test 11: Batch modify emails"""
        print("\n" + "="*80)
        print("TEST 11: Batch Modify Emails")
        print("="*80)
        
        async def run_test():
            request = GmailBatchModifyRequest(
                message_ids=["msg_1", "msg_2", "msg_3"],
                add_label_ids=["STARRED"],
                remove_label_ids=["UNREAD"]
            )
            
            response = await self.service.batch_modify(self.test_user_id, request)
            
            assert response is not None
            assert response.success is True
            assert response.modified_count == len(request.message_ids)
            
            print("✅ Batch modify successful")
            print(f"   - Success: {response.success}")
            print(f"   - Modified count: {response.modified_count}")
            print(f"   - Message: {response.message}")
        
        self.loop.run_until_complete(run_test())
    
    def test_12_batch_delete_emails(self):
        """Test 12: Batch delete emails"""
        print("\n" + "="*80)
        print("TEST 12: Batch Delete Emails")
        print("="*80)
        
        async def run_test():
            request = GmailBatchDeleteRequest(
                message_ids=["msg_1", "msg_2", "msg_3"]
            )
            
            response = await self.service.batch_delete(self.test_user_id, request)
            
            assert response is not None
            assert response.success is True
            assert response.deleted_count == len(request.message_ids)
            
            print("✅ Batch delete successful")
            print(f"   - Success: {response.success}")
            print(f"   - Deleted count: {response.deleted_count}")
            print(f"   - Message: {response.message}")
        
        self.loop.run_until_complete(run_test())
    
    def test_13_gmail_health_check(self):
        """Test 13: Gmail service health check"""
        print("\n" + "="*80)
        print("TEST 13: Gmail Service Health Check")
        print("="*80)
        
        async def run_test():
            health = await self.service.get_health(self.test_user_id)
            
            assert health is not None
            assert health.status is not None
            assert health.message is not None
            assert health.timestamp is not None
            
            print("✅ Gmail health check successful")
            print(f"   - Status: {health.status}")
            print(f"   - Message: {health.message}")
            print(f"   - Authenticated: {health.authenticated}")
            print(f"   - User email: {health.user_email}")
            print(f"   - Quota remaining: {health.quota_remaining}")
        
        self.loop.run_until_complete(run_test())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print("\n" + "="*80)
        print("GMAIL MODULE TESTS COMPLETE")
        print("="*80)


if __name__ == "__main__":
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGmailModule)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED ({}/{})".format(
            result.testsRun - len(result.failures) - len(result.errors),
            result.testsRun
        ))
    else:
        print("❌ SOME TESTS FAILED ({}/{})".format(
            result.testsRun - len(result.failures) - len(result.errors),
            result.testsRun
        ))
    print("="*80)

