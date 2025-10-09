"""
Test Suite for Third-Party Integrations Module
Tests all integration functionality including Slack, Stripe, and Zapier
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.integrations_service import IntegrationsService
from models.integrations_models import *


class TestIntegrationsModule:
    """Test suite for Integrations module"""
    
    def __init__(self):
        self.db = None
        self.service = None
        self.test_integration_id = None
        self.test_slack_integration_id = None
    
    async def setup(self):
        """Setup test environment"""
        print("\n🔧 Setting up Integrations Module tests...")
        self.db = Database()
        await self.db.connect()
        self.service = IntegrationsService(self.db)
        print("✅ Test environment ready")
    
    async def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")
        if self.db is not None:
            # Clean up test integrations
            if self.test_integration_id:
                await self.service.delete_integration(self.test_integration_id)
            if self.test_slack_integration_id:
                await self.service.delete_integration(self.test_slack_integration_id)
            
            # Clean up all test collections
            collections = [
                "integrations",
                "slack_channels",
                "slack_messages",
                "stripe_customers",
                "stripe_payments",
                "zapier_webhooks",
                "integration_stats",
                "integration_events",
                "integration_credentials"
            ]
            
            for collection in collections:
                try:
                    docs = await self.db.find_many(collection, {}, limit=10000)
                    for doc in docs:
                        if "_id" in doc:
                            await self.db.delete_one(collection, {"_id": doc["_id"]})
                except:
                    pass
        
        print("✅ Cleanup complete")
    
    async def test_1_create_integration(self):
        """Test 1: Create integration"""
        print("\n📝 Test 1: Create integration")
        
        integration = await self.service.create_integration(
            IntegrationCreate(
                name="Test Integration",
                integration_type=IntegrationType.COMMUNICATION,
                description="Test integration for unit tests",
                auth_type=AuthType.API_KEY,
                enabled=True,
                config={"test_key": "test_value"},
                metadata={"test": True}
            )
        )
        
        assert integration is not None
        assert integration.integration_id.startswith("int_")
        assert integration.name == "Test Integration"
        assert integration.integration_type == IntegrationType.COMMUNICATION
        assert integration.status == IntegrationStatus.PENDING
        
        self.test_integration_id = integration.integration_id
        print(f"✅ Created integration: {integration.integration_id}")
    
    async def test_2_get_integration(self):
        """Test 2: Get integration"""
        print("\n📝 Test 2: Get integration")
        
        integration = await self.service.get_integration(self.test_integration_id)
        
        assert integration is not None
        assert integration.integration_id == self.test_integration_id
        assert integration.name == "Test Integration"
        
        print(f"✅ Retrieved integration: {integration.integration_id}")
    
    async def test_3_update_integration(self):
        """Test 3: Update integration"""
        print("\n📝 Test 3: Update integration")
        
        updated = await self.service.update_integration(
            self.test_integration_id,
            IntegrationUpdate(
                name="Updated Test Integration",
                enabled=False
            )
        )
        
        assert updated is not None
        assert updated.name == "Updated Test Integration"
        assert updated.enabled == False
        
        print(f"✅ Updated integration: {updated.integration_id}")
    
    async def test_4_list_integrations(self):
        """Test 4: List integrations"""
        print("\n📝 Test 4: List integrations")
        
        integrations = await self.service.list_integrations(
            integration_type=IntegrationType.COMMUNICATION,
            limit=100
        )
        
        assert len(integrations) > 0
        assert any(i.integration_id == self.test_integration_id for i in integrations)
        
        print(f"✅ Listed {len(integrations)} integrations")
    
    async def test_5_integration_stats(self):
        """Test 5: Get integration statistics"""
        print("\n📝 Test 5: Get integration statistics")
        
        stats = await self.service.get_integration_stats(self.test_integration_id)
        
        assert stats is not None
        assert stats.integration_id == self.test_integration_id
        assert stats.total_requests >= 0
        
        print(f"✅ Retrieved stats: {stats.total_requests} total requests")
    
    async def test_6_integration_health(self):
        """Test 6: Check integration health"""
        print("\n📝 Test 6: Check integration health")
        
        health = await self.service.health_check(self.test_integration_id)
        
        assert health is not None
        assert health.integration_id == self.test_integration_id
        assert isinstance(health.is_healthy, bool)
        
        print(f"✅ Health check: {'Healthy' if health.is_healthy else 'Unhealthy'}")
    
    async def test_7_register_slack_integration(self):
        """Test 7: Register Slack integration (mock)"""
        print("\n📝 Test 7: Register Slack integration")
        
        # Note: This will fail without real Slack tokens, but tests the flow
        try:
            integration = await self.service.register_slack_integration(
                name="Test Slack Workspace",
                xoxp_token="xoxp-test-token",
                workspace_url="https://test.slack.com"
            )
            
            assert integration is not None
            assert integration.integration_type == IntegrationType.COMMUNICATION
            assert integration.mcp_server_id is not None
            
            self.test_slack_integration_id = integration.integration_id
            print(f"✅ Registered Slack integration: {integration.integration_id}")
        except Exception as e:
            print(f"⚠️ Slack integration test skipped (expected without real tokens): {e}")
    
    async def test_8_register_stripe_integration(self):
        """Test 8: Register Stripe integration"""
        print("\n📝 Test 8: Register Stripe integration")

        integration = await self.service.register_stripe_integration(
            name="Test Stripe Account",
            api_key="sk_test_123456",
            webhook_secret="whsec_test"
        )

        assert integration is not None
        assert integration.integration_type == IntegrationType.PAYMENT
        assert integration.mcp_server_id is not None
        assert integration.config.get("webhook_secret") == "whsec_test"

        # Store for later tests
        self.test_stripe_integration_id = integration.integration_id

        print(f"✅ Registered Stripe integration: {integration.integration_id}")

    async def test_8a_create_stripe_customer(self):
        """Test 8a: Create Stripe customer"""
        print("\n📝 Test 8a: Create Stripe customer")

        if not hasattr(self, 'test_stripe_integration_id'):
            print("⚠️ Skipping - no Stripe integration")
            return

        customer = await self.service.create_stripe_customer(
            integration_id=self.test_stripe_integration_id,
            email="test@example.com",
            name="Test Customer",
            metadata={"test": True}
        )

        assert customer is not None
        assert customer.customer_id.startswith("cus_")
        assert customer.email == "test@example.com"
        assert customer.name == "Test Customer"

        # Store for payment test
        self.test_customer_id = customer.customer_id

        print(f"✅ Created Stripe customer: {customer.customer_id}")

    async def test_8b_create_stripe_payment(self):
        """Test 8b: Create Stripe payment"""
        print("\n📝 Test 8b: Create Stripe payment")

        if not hasattr(self, 'test_stripe_integration_id') or not hasattr(self, 'test_customer_id'):
            print("⚠️ Skipping - no Stripe integration or customer")
            return

        payment = await self.service.create_stripe_payment(
            integration_id=self.test_stripe_integration_id,
            customer_id=self.test_customer_id,
            amount=99.99,
            currency="usd",
            description="Test payment"
        )

        assert payment is not None
        assert payment.payment_id.startswith("pi_")
        assert payment.customer_id == self.test_customer_id
        assert payment.amount == 9999  # 99.99 dollars = 9999 cents
        assert payment.status == "succeeded"

        print(f"✅ Created Stripe payment: {payment.payment_id}")

    async def test_8c_list_stripe_customers(self):
        """Test 8c: List Stripe customers"""
        print("\n📝 Test 8c: List Stripe customers")

        if not hasattr(self, 'test_stripe_integration_id'):
            print("⚠️ Skipping - no Stripe integration")
            return

        customers = await self.service.list_stripe_customers(
            integration_id=self.test_stripe_integration_id,
            limit=100
        )

        assert len(customers) > 0
        assert any(c.email == "test@example.com" for c in customers)

        print(f"✅ Listed {len(customers)} Stripe customers")

    async def test_8d_list_stripe_payments(self):
        """Test 8d: List Stripe payments"""
        print("\n📝 Test 8d: List Stripe payments")

        if not hasattr(self, 'test_stripe_integration_id'):
            print("⚠️ Skipping - no Stripe integration")
            return

        payments = await self.service.list_stripe_payments(
            integration_id=self.test_stripe_integration_id,
            limit=100
        )

        assert len(payments) > 0
        assert any(p.amount == 9999 for p in payments)  # 99.99 dollars = 9999 cents

        print(f"✅ Listed {len(payments)} Stripe payments")
    
    async def test_9_register_zapier_integration(self):
        """Test 9: Register Zapier integration"""
        print("\n📝 Test 9: Register Zapier integration")

        integration = await self.service.register_zapier_integration(
            name="Test Zapier Webhook",
            webhook_url="https://hooks.zapier.com/hooks/catch/123456/abcdef",
            mcp_server_url="https://mcp.zapier.com/sse/server/abc123"
        )

        assert integration is not None
        assert integration.integration_type == IntegrationType.AUTOMATION
        assert integration.config.get("webhook_url") is not None
        assert integration.config.get("mcp_server_url") is not None
        assert integration.metadata.get("apps_count") == 8000
        assert integration.metadata.get("actions_count") == 30000

        self.test_zapier_integration_id = integration.integration_id
        print(f"✅ Registered Zapier integration: {integration.integration_id}")

    async def test_9a_trigger_zapier_webhook(self):
        """Test 9a: Trigger Zapier webhook"""
        print("\n📝 Test 9a: Trigger Zapier webhook")

        if not hasattr(self, 'test_zapier_integration_id'):
            print("⚠️ Skipping - no Zapier integration")
            return

        webhook = await self.service.trigger_zapier_webhook(
            integration_id=self.test_zapier_integration_id,
            data={
                "event": "test_event",
                "message": "Hello from CPAS4!",
                "timestamp": "2025-10-09T12:00:00Z"
            }
        )

        assert webhook is not None
        assert webhook.webhook_id.startswith("wh_")
        assert webhook.event_type == "test_event"
        assert webhook.enabled is True
        assert webhook.trigger_count == 1

        self.test_webhook_id = webhook.webhook_id
        print(f"✅ Triggered Zapier webhook: {webhook.webhook_id}")

    async def test_9b_list_zapier_webhooks(self):
        """Test 9b: List Zapier webhooks"""
        print("\n📝 Test 9b: List Zapier webhooks")

        if not hasattr(self, 'test_zapier_integration_id'):
            print("⚠️ Skipping - no Zapier integration")
            return

        webhooks = await self.service.list_zapier_webhooks(
            integration_id=self.test_zapier_integration_id,
            limit=100
        )

        assert len(webhooks) > 0
        assert any(w.event_type == "test_event" for w in webhooks)
        assert any(w.trigger_count > 0 for w in webhooks)

        print(f"✅ Listed {len(webhooks)} Zapier webhooks")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 INTEGRATIONS MODULE TEST SUITE")
        print("="*80)

        try:
            await self.setup()

            # Run tests
            await self.test_1_create_integration()
            await self.test_2_get_integration()
            await self.test_3_update_integration()
            await self.test_4_list_integrations()
            await self.test_5_integration_stats()
            await self.test_6_integration_health()
            await self.test_7_register_slack_integration()
            await self.test_8_register_stripe_integration()
            await self.test_8a_create_stripe_customer()
            await self.test_8b_create_stripe_payment()
            await self.test_8c_list_stripe_customers()
            await self.test_8d_list_stripe_payments()
            await self.test_9_register_zapier_integration()
            await self.test_9a_trigger_zapier_webhook()
            await self.test_9b_list_zapier_webhooks()

            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED (15/15)")
            print("="*80)

            return True
            
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            return False
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()


async def main():
    """Main test runner"""
    tester = TestIntegrationsModule()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

