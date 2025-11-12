"""
GRETA PAI - MCP Server Ecosystem
Core PAI Feature: External service integrations
Browser automation, financial APIs, analytics, communication, etc.
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
import logging
from datetime import datetime
import os
from pathlib import Path
import json
import requests
from urllib.parse import urlparse, urljoin

from utils.error_handling import GretaException, handle_errors
from utils.performance import performance_monitor

logger = logging.getLogger(__name__)


class MCPServerError(GretaException):
    """MCP server-related errors"""


class MCPServiceBase:
    """Base class for MCP services"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.connected = False
        self.health_status = "unknown"
        self.last_used = None
        self.usage_count = 0
        self.error_count = 0
        self.capabilities = []
        self.configuration = {}

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the MCP service"""
        self.configuration.update(config)
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return {
            'service': self.service_name,
            'status': self.health_status,
            'connected': self.connected,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'usage_count': self.usage_count,
            'error_rate': self.error_count / max(1, self.usage_count),
            'capabilities': self.capabilities
        }

    def update_stats(self, success: bool):
        """Update service statistics"""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        if not success:
            self.error_count += 1

    async def execute_task(self, task_name: str, **kwargs) -> Any:
        """Execute a task using this service"""
        raise NotImplementedError(f"{self.service_name} does not implement {task_name}")


class WebBrowserService(MCPServiceBase):
    """Browser automation service using Playwright/Selenium"""

    def __init__(self):
        super().__init__("web_browser")
        self.capabilities = [
            'page_screenshot', 'page_content', 'click_element', 'fill_form',
            'wait_for_selector', 'scroll_page', 'navigate', 'get_cookies',
            'take_screenshot', 'pdf_export', 'run_javascript'
        ]

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize browser automation"""
        try:
            browser_type = config.get('browser', 'chromium')
            headless = config.get('headless', True)

            # Try Playwright first (preferred)
            try:
                from playwright.async_api import async_playwright
                self.playwright_manager = PlaywrightManager(browser_type, headless)
                await self.playwright_manager.initialize()
                self.backend = 'playwright'
                self.connected = True
                logger.info("✅ Web Browser MCP initialized with Playwright")
                return True
            except ImportError:
                pass

            # Fallback to Selenium
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                self.selenium_manager = SeleniumManager(browser_type, headless)
                self.selenium_manager.initialize()
                self.backend = 'selenium'
                self.connected = True
                logger.info("✅ Web Browser MCP initialized with Selenium")
                return True
            except ImportError:
                pass

            logger.warning("No browser automation backend available (Playwright or Selenium)")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize web browser MCP: {e}")
            return False

    async def execute_task(self, task_name: str, **kwargs) -> Any:
        """Execute browser automation task"""
        if not self.connected:
            raise MCPServerError("Web browser service not connected")

        self.update_stats(False)  # Will mark success if task completes

        try:
            if self.backend == 'playwright':
                result = await self.playwright_manager.execute_task(task_name, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.selenium_manager.execute_task, task_name, kwargs
                )

            self.update_stats(True)
            return result

        except Exception as e:
            logger.error(f"Web browser task '{task_name}' failed: {e}")
            raise MCPServerError(f"Browser automation failed: {str(e)}")


class FinancialService(MCPServiceBase):
    """Financial APIs service (Stripe, PayPal, banking)"""

    def __init__(self):
        super().__init__("financial")
        self.capabilities = [
            'stripe_payment', 'stripe_refund', 'stripe_customer_create',
            'paypal_payment', 'square_terminal', 'yfinance_quote',
            'bank_balance', 'transaction_history', 'invoice_create'
        ]
        self.providers = {}

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize financial service providers"""
        providers_initialized = 0

        # Stripe initialization
        if 'stripe_secret_key' in config:
            try:
                import stripe
                stripe.api_key = config['stripe_secret_key']
                self.providers['stripe'] = stripe
                providers_initialized += 1
                logger.info("✅ Stripe integration initialized")
            except ImportError:
                logger.warning("Stripe library not available")

        # Square initialization
        if 'square_access_token' in config:
            try:
                from square.client import Client
                self.providers['square'] = Client(
                    access_token=config['square_access_token'],
                    environment='sandbox' if config.get('sandbox', True) else 'production'
                )
                providers_initialized += 1
                logger.info("✅ Square integration initialized")
            except ImportError:
                logger.warning("Square library not available")

        # Yahoo Finance (free)
        try:
            import yfinance as yf
            self.providers['yfinance'] = yf
            providers_initialized += 1
            logger.info("✅ Yahoo Finance integration initialized")
        except ImportError:
            logger.warning("yfinance library not available")

        self.connected = providers_initialized > 0
        if self.connected:
            logger.info(f"✅ Financial MCP initialized with {providers_initialized} providers")

        return self.connected

    async def execute_task(self, task_name: str, **kwargs) -> Any:
        """Execute financial task"""
        if not self.connected:
            raise MCPServerError("Financial service not connected")

        task_provider_map = {
            'stripe_payment': 'stripe',
            'stripe_refund': 'stripe',
            'stripe_customer_create': 'stripe',
            'paypal_payment': 'paypal',
            'square_terminal': 'square',
            'yfinance_quote': 'yfinance',
            'bank_balance': 'bank_api',
            'transaction_history': 'bank_api',
            'invoice_create': 'stripe'
        }

        provider_name = task_provider_map.get(task_name)
        if not provider_name or provider_name not in self.providers:
            raise MCPServerError(f"Provider for task '{task_name}' not available")

        provider = self.providers[provider_name]
        self.update_stats(False)  # Will mark success if task completes

        try:
            if task_name.startswith('stripe_'):
                result = await self._execute_stripe_task(provider, task_name, **kwargs)
            elif task_name.startswith('square_'):
                result = await self._execute_square_task(provider, task_name, **kwargs)
            elif task_name.startswith('yfinance_'):
                result = await self._execute_yfinance_task(task_name, **kwargs)
            else:
                raise MCPServerError(f"Unknown financial task: {task_name}")

            self.update_stats(True)
            return result

        except Exception as e:
            logger.error(f"Financial task '{task_name}' failed: {e}")
            raise MCPServerError(f"Financial service error: {str(e)}")

    async def _execute_stripe_task(self, stripe, task_name: str, **kwargs) -> Any:
        """Execute Stripe-specific tasks"""
        if task_name == 'stripe_payment':
            payment_intent = stripe.PaymentIntent.create(
                amount=kwargs['amount'],
                currency=kwargs.get('currency', 'usd'),
                payment_method_types=['card']
            )
            return payment_intent

        elif task_name == 'stripe_customer_create':
            customer = stripe.Customer.create(
                email=kwargs.get('email'),
                name=kwargs.get('name')
            )
            return customer

        # Add more Stripe methods as needed

    async def _execute_yfinance_task(self, task_name: str, **kwargs) -> Any:
        """Execute Yahoo Finance tasks"""
        import yfinance as yf

        if task_name == 'yfinance_quote':
            ticker = yf.Ticker(kwargs['symbol'])
            return ticker.info


class AnalyticsService(MCPServiceBase):
    """Analytics and metrics service"""

    def __init__(self):
        super().__init__("analytics")
        self.capabilities = [
            'google_analytics', 'mixpanel_events', 'beehiiv_stats',
            'site_traffic', 'conversion_tracking', 'user_behavior',
            'custom_dashboard', 'report_generation'
        ]

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize analytics services"""
        services_initialized = 0

        # Google Analytics
        if 'google_analytics_key' in config:
            try:
                # GA4 integration would go here
                self.providers['google_analytics'] = {'key': config['google_analytics_key']}
                services_initialized += 1
            except Exception as e:
                logger.warning(f"Google Analytics init failed: {e}")

        # Mixpanel
        if 'mixpanel_token' in config:
            try:
                import mixpanel
                self.providers['mixpanel'] = mixpanel.Mixpanel(config['mixpanel_token'])
                services_initialized += 1
            except ImportError:
                logger.warning("Mixpanel library not available")

        # Beehiiv (newsletter analytics)
        if 'beehiiv_api_key' in config:
            try:
                self.providers['beehiiv'] = {'api_key': config['beehiiv_api_key']}
                services_initialized += 1
            except Exception as e:
                logger.warning(f"Beehiiv init failed: {e}")

        self.connected = services_initialized > 0
        if self.connected:
            logger.info(f"✅ Analytics MCP initialized with {services_initialized} services")

        return self.connected

    async def execute_task(self, task_name: str, **kwargs) -> Any:
        """Execute analytics task"""
        if not self.connected:
            raise MCPServerError("Analytics service not connected")

        self.update_stats(False)  # Will mark success

        try:
            if task_name == 'beehiiv_stats':
                return await self._get_beehiiv_stats(**kwargs)
            elif task_name == 'mixpanel_events':
                return await self._track_mixpanel_event(**kwargs)
            # Add more analytics tasks

            self.update_stats(True)
            return result

        except Exception as e:
            logger.error(f"Analytics task '{task_name}' failed: {e}")
            raise MCPServerError(f"Analytics service error: {str(e)}")

    async def _get_beehiiv_stats(self, **kwargs) -> Dict[str, Any]:
        """Get Beehiiv newsletter statistics"""
        # This would integrate with Beehiiv API
        # Placeholder implementation
        return {
            'total_subscribers': 1500,
            'open_rate': 0.45,
            'click_rate': 0.12,
            'period': 'last_30_days'
        }


class CommunicationService(MCPServiceBase):
    """Communication service (email, SMS, social media)"""

    def __init__(self):
        super().__init__("communication")
        self.capabilities = [
            'send_email', 'send_sms', 'elevenlabs_tts', 'discord_message',
            'twitter_post', 'linkedin_update', 'slack_notification'
        ]

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize communication services"""
        services_initialized = 0

        # SendGrid email
        if 'sendgrid_api_key' in config:
            try:
                import sendgrid
                self.providers['sendgrid'] = sendgrid.SendGridAPIClient(config['sendgrid_api_key'])
                services_initialized += 1
                logger.info("✅ SendGrid email service initialized")
            except ImportError:
                logger.warning("SendGrid library not available")

        # Twilio SMS
        if 'twilio_account_sid' in config and 'twilio_auth_token' in config:
            try:
                from twilio.rest import Client
                self.providers['twilio'] = Client(
                    config['twilio_account_sid'],
                    config['twilio_auth_token']
                )
                services_initialized += 1
                logger.info("✅ Twilio SMS service initialized")
            except ImportError:
                logger.warning("Twilio library not available")

        # ElevenLabs TTS
        if 'elevenlabs_api_key' in config:
            try:
                import requests  # ElevenLabs uses REST API
                self.providers['elevenlabs'] = {'api_key': config['elevenlabs_api_key']}
                services_initialized += 1
                logger.info("✅ ElevenLabs TTS service initialized")
            except Exception as e:
                logger.warning(f"ElevenLabs init failed: {e}")

        # Discord
        if 'discord_webhook_url' in config:
            try:
                self.providers['discord'] = {'webhook_url': config['discord_webhook_url']}
                services_initialized += 1
                logger.info("✅ Discord service initialized")
            except Exception as e:
                logger.warning(f"Discord init failed: {e}")

        self.connected = services_initialized > 0
        if self.connected:
            logger.info(f"✅ Communication MCP initialized with {services_initialized} services")

        return self.connected

    async def execute_task(self, task_name: str, **kwargs) -> Any:
        """Execute communication task"""
        if not self.connected:
            raise MCPServerError("Communication service not connected")

        self.update_stats(False)  # Will mark success

        try:
            result = None

            if task_name == 'send_email':
                result = await self._send_email(**kwargs)
            elif task_name == 'send_sms':
                result = await self._send_sms(**kwargs)
            elif task_name == 'elevenlabs_tts':
                result = await self._generate_tts(**kwargs)
            elif task_name == 'discord_message':
                result = await self._send_discord_message(**kwargs)

            self.update_stats(True)
            return result

        except Exception as e:
            logger.error(f"Communication task '{task_name}' failed: {e}")
            raise MCPServerError(f"Communication service error: {str(e)}")

    async def _send_email(self, to: str, subject: str, body: str, **kwargs) -> Dict[str, Any]:
        """Send email using SendGrid"""
        if 'sendgrid' not in self.providers:
            raise MCPServerError("SendGrid not configured")

        sg = self.providers['sendgrid']
        from_email = kwargs.get('from_email', os.getenv('DEFAULT_FROM_EMAIL'))

        # SendGrid email sending logic
        message = {
            'personalizations': [{'to': [{'email': to}]}],
            'from': {'email': from_email},
            'subject': subject,
            'content': [{'type': 'text/plain', 'value': body}]
        }

        await asyncio.get_event_loop().run_in_executor(
            None, sg.send, message
        )

        return {'status': 'sent', 'to': to, 'provider': 'sendgrid'}

    async def _send_sms(self, to: str, message: str, **kwargs) -> Dict[str, Any]:
        """Send SMS using Twilio"""
        if 'twilio' not in self.providers:
            raise MCPServerError("Twilio not configured")

        twilio = self.providers['twilio']
        from_number = kwargs.get('from_number', os.getenv('TWILIO_FROM_NUMBER'))

        await asyncio.get_event_loop().run_in_executor(
            None, twilio.messages.create,
            to=to, from_=from_number, body=message
        )

        return {'status': 'sent', 'to': to, 'provider': 'twilio'}

    async def _generate_tts(self, text: str, voice: str = "Kore", **kwargs) -> Dict[str, Any]:
        """Generate TTS using ElevenLabs"""
        if 'elevenlabs' not in self.providers:
            raise MCPServerError("ElevenLabs not configured")

        api_key = self.providers['elevenlabs']['api_key']

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
            'xi-api-key': api_key
        }

        data = {
            'text': text,
            'model_id': 'eleven_monolingual_v1',
            'voice_settings': {
                'stability': 0.5,
                'similarity_boost': 0.5
            }
        }

        response = await asyncio.get_event_loop().run_in_executor(
            None, requests.post, url, json=data, headers=headers
        )

        if response.status_code == 200:
            # Save audio file and return path
            filename = f"tts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp3"
            with open(filename, 'wb') as f:
                f.write(response.content)

            return {
                'status': 'generated',
                'audio_file': filename,
                'provider': 'elevenlabs',
                'voice': voice
            }
        else:
            raise MCPServerError(f"ElevenLabs TTS failed: {response.status_code}")

    async def _send_discord_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send Discord message via webhook"""
        if 'discord' not in self.providers:
            raise MCPServerError("Discord not configured")

        webhook_url = self.providers['discord']['webhook_url']

        data = {'content': message}
        response = await asyncio.get_event_loop().run_in_executor(
            None, requests.post, webhook_url, json=data
        )

        if response.status_code == 204:
            return {'status': 'sent', 'provider': 'discord'}
        else:
            raise MCPServerError(f"Discord webhook failed: {response.status_code}")


class MCPServerOrchestrator:
    """
    PAI's MCP Server Orchestrator
    Manages all MCP servers and coordinates between them
    """

    def __init__(self):
        self.servers: Dict[str, MCPServiceBase] = {}
        self._initialize_servers()

    def _initialize_servers(self):
        """Initialize available MCP services"""
        self.servers = {
            'web_browser': WebBrowserService(),
            'financial': FinancialService(),
            'analytics': AnalyticsService(),
            'communication': CommunicationService()
        }

        logger.info(f"🎛️ MCP Orchestrator initialized with {len(self.servers)} servers")

    async def initialize_all_servers(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize all MCP servers with configuration"""
        config = config or {}
        initialization_results = {}

        for server_name, server in self.servers.items():
            server_config = config.get(server_name, {})
            success = await server.initialize(server_config)
            initialization_results[server_name] = {
                'initialized': success,
                'capabilities': server.capabilities,
                'status': await server.health_check()
            }

        connected_count = sum(1 for result in initialization_results.values() if result['initialized'])
        logger.info(f"📡 MCP Orchestrator: {connected_count}/{len(self.servers)} servers initialized")

        return initialization_results

    async def execute_service_task(self, service_name: str, task_name: str, **kwargs) -> Any:
        """
        Execute a task on a specific MCP service
        """
        if service_name not in self.servers:
            raise MCPServerError(f"MCP service '{service_name}' not found")

        server = self.servers[service_name]

        # Check if server is connected and has the capability
        if not server.connected:
            raise MCPServerError(f"MCP service '{service_name}' not connected")

        if task_name not in server.capabilities:
            raise MCPServerError(f"Service '{service_name}' does not support task '{task_name}'")

        logger.info(f"🔌 Executing MCP task: {service_name}.{task_name}")
        result = await server.execute_task(task_name, **kwargs)

        logger.info(f"✅ MCP task completed: {service_name}.{task_name}")
        return result

    async def get_server_health(self) -> Dict[str, Any]:
        """Get health status of all MCP servers"""
        health_status = {}

        for server_name, server in self.servers.items():
            health_status[server_name] = await server.health_check()

        return health_status

    def get_server_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all MCP servers"""
        return {
            server_name: server.capabilities
            for server_name, server in self.servers.items()
        }

    async def discover_services(self) -> Dict[str, Any]:
        """Auto-discover available MCP services in the environment"""
        # This could scan for configuration files, environment variables, etc.
        discovered = {}

        # Check for common service APIs
        service_checks = {
            'web_browser': self._check_browser_services,
            'financial': self._check_financial_services,
            'analytics': self._check_analytics_services,
            'communication': self._check_communication_services
        }

        for service_name, check_func in service_checks.items():
            discovered[service_name] = await check_func()

        return discovered

    async def _check_browser_services(self) -> Dict[str, Any]:
        """Check for available browser automation services"""
        available = []

        try:
            from playwright.async_api import async_playwright
            available.append('playwright')
        except ImportError:
            pass

        try:
            from selenium import webdriver
            available.append('selenium')
        except ImportError:
            pass

        return {'available': available, 'configured': len(available) > 0}

    async def _check_financial_services(self) -> Dict[str, Any]:
        """Check for available financial services"""
        available = []

        if os.getenv('STRIPE_SECRET_KEY'):
            available.append('stripe')

        if os.getenv('SQUARE_ACCESS_TOKEN'):
            available.append('square')

        try:
            import yfinance
            available.append('yfinance')
        except ImportError:
            pass

        return {'available': available, 'configured': len(available) > 0}

    async def _check_analytics_services(self) -> Dict[str, Any]:
        """Check for available analytics services"""
        available = []

        if os.getenv('GOOGLE_ANALYTICS_KEY'):
            available.append('google_analytics')

        if os.getenv('MIXPANEL_TOKEN'):
            available.append('mixpanel')

        if os.getenv('BEEHIIV_API_KEY'):
            available.append('beehiiv')

        return {'available': available, 'configured': len(available) > 0}

    async def _check_communication_services(self) -> Dict[str, Any]:
        """Check for available communication services"""
        available = []

        if os.getenv('SENDGRID_API_KEY'):
            available.append('sendgrid')

        if os.getenv('TWILIO_ACCOUNT_SID'):
            available.append('twilio')

        if os.getenv('ELEVENLABS_API_KEY'):
            available.append('elevenlabs')

        if os.getenv('DISCORD_WEBHOOK_URL'):
            available.append('discord')

        return {'available': available, 'configured': len(available) > 0}


# Global MCP Orchestrator instance
mcp_orchestrator = MCPServerOrchestrator()


# Helper classes for browser automation (placeholders - would implement full functionality)
class PlaywrightManager:
    def __init__(self, browser_type: str, headless: bool):
        self.browser_type = browser_type
        self.headless = headless

    async def initialize(self):
        pass

    async def execute_task(self, task_name: str, **kwargs):
        # Placeholder for Playwright tasks
        return f"Playwright task {task_name} executed"


class SeleniumManager:
    def __init__(self, browser_type: str, headless: bool):
        self.browser_type = browser_type
        self.headless = headless

    def initialize(self):
        pass

    def execute_task(self, task_name: str, **kwargs):
        # Placeholder for Selenium tasks
        return f"Selenium task {task_name} executed"


__all__ = [
    'MCPServerError',
    'MCPServiceBase',
    'WebBrowserService',
    'FinancialService',
    'AnalyticsService',
    'CommunicationService',
    'MCPServerOrchestrator',
    'mcp_orchestrator'
]
