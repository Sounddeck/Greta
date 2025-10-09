"""
Third-Party Integrations Service for CPAS4
Manages third-party service integrations via MCP Hub
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from database import Database
from models.integrations_models import *
from services.mcp_service import MCPHubService
from models.mcp_models import MCPServerCreate, AgentMCPIntegrationCreate, DiscoverySource

logger = logging.getLogger(__name__)


class IntegrationsService:
    """Service for managing third-party integrations"""

    def __init__(self, db: Database):
        self.db = db
        self.mcp_service = MCPHubService(db)
        
        # Collections
        self.integrations_collection = "integrations"
        self.slack_channels_collection = "slack_channels"
        self.slack_messages_collection = "slack_messages"
        self.stripe_customers_collection = "stripe_customers"
        self.stripe_payments_collection = "stripe_payments"
        self.zapier_webhooks_collection = "zapier_webhooks"
        self.integration_stats_collection = "integration_stats"
        self.integration_events_collection = "integration_events"
        self.integration_credentials_collection = "integration_credentials"
        
        logger.info("✅ Integrations Service initialized")
    
    # Integration Management
    async def create_integration(self, integration: IntegrationCreate) -> Integration:
        """Create new integration"""
        integration_id = f"int_{uuid.uuid4().hex[:12]}"
        
        integration_doc = Integration(
            integration_id=integration_id,
            name=integration.name,
            integration_type=integration.integration_type,
            description=integration.description,
            auth_type=integration.auth_type,
            mcp_server_id=integration.mcp_server_id,
            enabled=integration.enabled,
            config=integration.config,
            metadata=integration.metadata,
            status=IntegrationStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        await self.db.insert_one(self.integrations_collection, integration_doc.model_dump())
        
        # Initialize stats
        stats = IntegrationStats(integration_id=integration_id)
        await self.db.insert_one(self.integration_stats_collection, stats.model_dump())
        
        logger.info(f"Created integration: {integration_id} ({integration.name})")
        return integration_doc
    
    async def get_integration(self, integration_id: str) -> Optional[Integration]:
        """Get integration by ID"""
        doc = await self.db.find_one(self.integrations_collection, {"integration_id": integration_id})
        return Integration(**doc) if doc else None
    
    async def update_integration(self, integration_id: str, integration_update: IntegrationUpdate) -> Optional[Integration]:
        """Update integration"""
        update_data = integration_update.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        success = await self.db.update_one(
            self.integrations_collection,
            {"integration_id": integration_id},
            {"$set": update_data}
        )
        
        if success:
            logger.info(f"Updated integration: {integration_id}")
            return await self.get_integration(integration_id)
        return None
    
    async def delete_integration(self, integration_id: str) -> bool:
        """Delete integration"""
        success = await self.db.delete_one(self.integrations_collection, {"integration_id": integration_id})
        if success:
            # Clean up related data (using find_many + delete_one since delete_many doesn't exist)
            stats_docs = await self.db.find_many(self.integration_stats_collection, {"integration_id": integration_id}, limit=1000)
            for doc in stats_docs:
                if "_id" in doc:
                    await self.db.delete_one(self.integration_stats_collection, {"_id": doc["_id"]})

            events_docs = await self.db.find_many(self.integration_events_collection, {"integration_id": integration_id}, limit=1000)
            for doc in events_docs:
                if "_id" in doc:
                    await self.db.delete_one(self.integration_events_collection, {"_id": doc["_id"]})

            logger.info(f"Deleted integration: {integration_id}")
        return success
    
    async def list_integrations(
        self,
        integration_type: Optional[IntegrationType] = None,
        status: Optional[IntegrationStatus] = None,
        enabled: Optional[bool] = None,
        limit: int = 100
    ) -> List[Integration]:
        """List integrations with filters"""
        query = {}
        if integration_type:
            query["integration_type"] = integration_type
        if status:
            query["status"] = status
        if enabled is not None:
            query["enabled"] = enabled
        
        docs = await self.db.find_many(self.integrations_collection, query, limit=limit)
        return [Integration(**doc) for doc in docs]
    
    # Slack Integration
    async def register_slack_integration(
        self,
        name: str,
        xoxc_token: Optional[str] = None,
        xoxd_token: Optional[str] = None,
        xoxp_token: Optional[str] = None,
        workspace_url: str = "https://slack.com"
    ) -> Integration:
        """Register Slack integration via MCP Hub"""

        # Create MCP server for Slack
        mcp_server = await self.mcp_service.create_server(
            MCPServerCreate(
                name=f"Slack MCP Server - {name}",
                url="https://github.com/korotovsky/slack-mcp-server",
                description="Slack integration via MCP",
                version="1.1.25",
                discovery_source=DiscoverySource.MANUAL,
                tools=[
                    "conversations_history",
                    "conversations_replies",
                    "conversations_add_message",
                    "conversations_search_messages",
                    "channels_list"
                ],
                config={
                    "SLACK_MCP_XOXC_TOKEN": xoxc_token,
                    "SLACK_MCP_XOXD_TOKEN": xoxd_token,
                    "SLACK_MCP_XOXP_TOKEN": xoxp_token,
                    "SLACK_MCP_PORT": "13080",
                    "SLACK_MCP_HOST": "127.0.0.1",
                    "workspace_url": workspace_url
                },
                metadata={"integration_type": "slack"}
            )
        )
        
        # Create integration
        integration = await self.create_integration(
            IntegrationCreate(
                name=name,
                integration_type=IntegrationType.COMMUNICATION,
                description=f"Slack workspace integration: {name}",
                auth_type=AuthType.BEARER_TOKEN,
                mcp_server_id=mcp_server.server_id,
                enabled=True,
                config={"workspace_url": workspace_url},
                metadata={"mcp_server": mcp_server.server_id}
            )
        )
        
        # Update status to active
        await self.update_integration(
            integration.integration_id,
            IntegrationUpdate(status=IntegrationStatus.ACTIVE)
        )
        
        logger.info(f"Registered Slack integration: {integration.integration_id}")
        return integration
    
    async def send_slack_message(
        self,
        integration_id: str,
        channel_id: str,
        text: str,
        thread_ts: Optional[str] = None
    ) -> SlackMessage:
        """Send message to Slack channel"""
        integration = await self.get_integration(integration_id)
        if not integration or not integration.mcp_server_id:
            raise ValueError("Integration not found or not linked to MCP server")

        # TODO: Execute via MCP Hub when execute_tool is implemented
        # For now, create a mock message
        result = {"ts": str(datetime.utcnow().timestamp())}

        # Store message
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        message = SlackMessage(
            message_id=message_id,
            channel_id=channel_id,
            user_id="bot",
            text=text,
            thread_ts=thread_ts,
            ts=result.get("ts", ""),
            integration_id=integration_id,
            created_at=datetime.utcnow()
        )

        await self.db.insert_one(self.slack_messages_collection, message.model_dump())

        # Update stats
        await self._update_integration_stats(integration_id, success=True)

        logger.info(f"Sent Slack message: {message_id}")
        return message
    
    async def get_slack_channels(self, integration_id: str) -> List[SlackChannel]:
        """Get Slack channels"""
        integration = await self.get_integration(integration_id)
        if not integration or not integration.mcp_server_id:
            raise ValueError("Integration not found or not linked to MCP server")

        # TODO: Execute via MCP Hub when execute_tool is implemented
        # For now, return mock channels
        result = {
            "channels": [
                {
                    "id": "C123456",
                    "name": "general",
                    "is_private": False,
                    "topic": {"value": "General discussion"},
                    "purpose": {"value": "Company-wide announcements"},
                    "num_members": 10
                }
            ]
        }

        channels = []
        for channel_data in result.get("channels", []):
            channel = SlackChannel(
                channel_id=channel_data["id"],
                name=channel_data["name"],
                is_private=channel_data.get("is_private", False),
                topic=channel_data.get("topic", {}).get("value"),
                purpose=channel_data.get("purpose", {}).get("value"),
                member_count=channel_data.get("num_members", 0),
                integration_id=integration_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            channels.append(channel)

        return channels
    
    # Stripe Integration via MCP Hub
    async def register_stripe_integration(
        self,
        name: str,
        api_key: str,
        webhook_secret: Optional[str] = None
    ) -> Integration:
        """Register Stripe integration via MCP Hub"""

        # Create MCP server for Stripe
        mcp_server = await self.mcp_service.create_server(
            MCPServerCreate(
                name=f"Stripe MCP Server - {name}",
                url="https://github.com/stripe/agent-toolkit",
                description="Stripe payment integration via MCP - 22 tools for payments, subscriptions, customers",
                version="0.1.0",
                discovery_source=DiscoverySource.MANUAL,
                tools=[
                    "cancel_subscription",
                    "create_coupon",
                    "create_customer",
                    "create_invoice",
                    "create_invoice_item",
                    "create_payment_link",
                    "create_price",
                    "create_product",
                    "create_refund",
                    "finalize_invoice",
                    "list_coupons",
                    "list_customers",
                    "list_disputes",
                    "list_invoices",
                    "list_payment_intents",
                    "list_prices",
                    "list_products",
                    "list_subscriptions",
                    "retrieve_balance",
                    "search_stripe_documentation",
                    "update_dispute",
                    "update_subscription"
                ],
                config={
                    "STRIPE_SECRET_KEY": api_key,
                    "webhook_secret": webhook_secret,
                    "tools": "all"
                },
                metadata={"integration_type": "stripe", "provider": "stripe"}
            )
        )

        # Create integration linked to MCP server
        integration = await self.create_integration(
            IntegrationCreate(
                name=name,
                integration_type=IntegrationType.PAYMENT,
                description=f"Stripe payment integration: {name}",
                auth_type=AuthType.API_KEY,
                enabled=True,
                config={"webhook_secret": webhook_secret},
                metadata={"mcp_server": mcp_server.server_id, "provider": "stripe"}
            )
        )

        # Update integration with MCP server ID
        integration.mcp_server_id = mcp_server.server_id
        await self.db.update_one(
            self.integrations_collection,
            {"integration_id": integration.integration_id},
            {"$set": {"mcp_server_id": mcp_server.server_id}}
        )

        logger.info(f"Registered Stripe integration: {integration.integration_id} with MCP server: {mcp_server.server_id}")
        return integration

    async def create_stripe_customer(
        self,
        integration_id: str,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StripeCustomer:
        """Create Stripe customer"""
        integration = await self.get_integration(integration_id)
        if not integration or not integration.mcp_server_id:
            raise ValueError("Integration not found or not linked to MCP server")

        # TODO: Execute via MCP Hub when execute_tool is implemented
        # For now, create mock customer
        customer_id = f"cus_{uuid.uuid4().hex[:14]}"
        customer = StripeCustomer(
            customer_id=customer_id,
            email=email,
            name=name,
            metadata=metadata or {},
            integration_id=integration_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        await self.db.insert_one(self.stripe_customers_collection, customer.model_dump())
        await self._update_integration_stats(integration_id, success=True)

        logger.info(f"Created Stripe customer: {customer_id}")
        return customer

    async def create_stripe_payment(
        self,
        integration_id: str,
        customer_id: str,
        amount: float,
        currency: str = "usd",
        description: Optional[str] = None
    ) -> StripePayment:
        """Create Stripe payment (amount in dollars, converted to cents)"""
        integration = await self.get_integration(integration_id)
        if not integration or not integration.mcp_server_id:
            raise ValueError("Integration not found or not linked to MCP server")

        # TODO: Execute via MCP Hub when execute_tool is implemented
        # For now, create mock payment
        payment_id = f"pi_{uuid.uuid4().hex[:14]}"
        # Convert dollars to cents (Stripe uses cents)
        amount_cents = int(amount * 100)
        payment = StripePayment(
            payment_id=payment_id,
            customer_id=customer_id,
            amount=amount_cents,
            currency=currency,
            status="succeeded",
            description=description,
            integration_id=integration_id,
            created_at=datetime.utcnow()
        )

        await self.db.insert_one(self.stripe_payments_collection, payment.model_dump())
        await self._update_integration_stats(integration_id, success=True)

        logger.info(f"Created Stripe payment: {payment_id}")
        return payment

    async def list_stripe_customers(
        self,
        integration_id: str,
        limit: int = 100
    ) -> List[StripeCustomer]:
        """List Stripe customers"""
        customers_data = await self.db.find_many(
            self.stripe_customers_collection,
            {"integration_id": integration_id},
            limit=limit
        )

        customers = []
        for data in customers_data:
            customers.append(StripeCustomer(**data))

        return customers

    async def list_stripe_payments(
        self,
        integration_id: str,
        customer_id: Optional[str] = None,
        limit: int = 100
    ) -> List[StripePayment]:
        """List Stripe payments"""
        query = {"integration_id": integration_id}
        if customer_id:
            query["customer_id"] = customer_id

        payments_data = await self.db.find_many(
            self.stripe_payments_collection,
            query,
            limit=limit
        )

        payments = []
        for data in payments_data:
            payments.append(StripePayment(**data))

        return payments

    # Zapier Integration (Webhook-based)
    async def register_zapier_integration(
        self,
        name: str,
        webhook_url: str,
        mcp_server_url: Optional[str] = None
    ) -> Integration:
        """Register Zapier integration (webhook-based or MCP SSE)

        Zapier MCP is cloud-based and uses SSE (Server-Sent Events) endpoints.
        Users create MCP servers via Zapier's web interface and get an SSE URL.

        Args:
            name: Integration name
            webhook_url: Zapier webhook URL for triggering workflows
            mcp_server_url: Optional Zapier MCP SSE server URL
        """
        integration = await self.create_integration(
            IntegrationCreate(
                name=name,
                integration_type=IntegrationType.AUTOMATION,
                description=f"Zapier automation integration: {name} - 8,000+ apps, 30,000+ actions",
                auth_type=AuthType.CUSTOM,
                enabled=True,
                config={
                    "webhook_url": webhook_url,
                    "mcp_server_url": mcp_server_url,
                    "provider": "zapier"
                },
                metadata={
                    "provider": "zapier",
                    "type": "webhook" if not mcp_server_url else "mcp_sse",
                    "apps_count": 8000,
                    "actions_count": 30000
                }
            )
        )

        logger.info(f"Registered Zapier integration: {integration.integration_id}")
        return integration

    async def trigger_zapier_webhook(
        self,
        integration_id: str,
        data: Dict[str, Any]
    ) -> ZapierWebhook:
        """Trigger Zapier webhook with data

        Args:
            integration_id: Integration ID
            data: Data to send to Zapier webhook
        """
        integration = await self.get_integration(integration_id)
        if not integration:
            raise ValueError("Integration not found")

        webhook_url = integration.config.get("webhook_url")
        if not webhook_url:
            raise ValueError("Webhook URL not configured")

        # TODO: Execute webhook via HTTP POST when implemented
        webhook_id = f"wh_{uuid.uuid4().hex[:14]}"
        event_type = data.get("event", "custom_event")

        webhook = ZapierWebhook(
            webhook_id=webhook_id,
            integration_id=integration_id,
            webhook_url=webhook_url,
            event_type=event_type,
            description=f"Triggered with data: {list(data.keys())}",
            enabled=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            last_triggered=datetime.utcnow(),
            trigger_count=1
        )

        await self.db.insert_one(self.zapier_webhooks_collection, webhook.model_dump())
        await self._update_integration_stats(integration_id, success=True)

        return webhook

    async def list_zapier_webhooks(
        self,
        integration_id: str,
        limit: int = 100
    ) -> List[ZapierWebhook]:
        """List Zapier webhooks for integration"""
        docs = await self.db.find_many(
            self.zapier_webhooks_collection,
            {"integration_id": integration_id},
            limit=limit
        )
        return [ZapierWebhook(**doc) for doc in docs]
    
    # Helper Methods
    async def _update_integration_stats(self, integration_id: str, success: bool = True):
        """Update integration statistics"""
        stats = await self.db.find_one(self.integration_stats_collection, {"integration_id": integration_id})
        if stats:
            # Remove _id before updating
            if "_id" in stats:
                del stats["_id"]

            stats["total_requests"] += 1
            if success:
                stats["successful_requests"] += 1
            else:
                stats["failed_requests"] += 1
            stats["last_request"] = datetime.utcnow().isoformat()

            await self.db.update_one(
                self.integration_stats_collection,
                {"integration_id": integration_id},
                {"$set": stats}
            )
    
    async def get_integration_stats(self, integration_id: str) -> Optional[IntegrationStats]:
        """Get integration statistics"""
        doc = await self.db.find_one(self.integration_stats_collection, {"integration_id": integration_id})
        return IntegrationStats(**doc) if doc else None
    
    async def health_check(self, integration_id: str) -> IntegrationHealth:
        """Check integration health"""
        integration = await self.get_integration(integration_id)
        if not integration:
            return IntegrationHealth(
                integration_id=integration_id,
                status=IntegrationStatus.ERROR,
                is_healthy=False,
                last_check=datetime.utcnow(),
                response_time_ms=0,
                error_message="Integration not found"
            )

        # Check MCP server health if linked
        is_healthy = True
        response_time_ms = 0.0
        error_message = None

        if integration.mcp_server_id:
            try:
                # TODO: Implement health check when MCP service has health_check method
                # For now, check if server exists
                server = await self.mcp_service.get_server(integration.mcp_server_id)
                is_healthy = server is not None
                response_time_ms = 10.0  # Mock response time
            except Exception as e:
                is_healthy = False
                error_message = str(e)

        return IntegrationHealth(
            integration_id=integration_id,
            status=integration.status,
            is_healthy=is_healthy,
            last_check=datetime.utcnow(),
            response_time_ms=response_time_ms,
            error_message=error_message
        )

