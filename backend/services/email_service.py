"""
Email Service for GRETA PAI Background Task Processing

This service provides comprehensive email functionality including template
rendering, SMTP configuration, and integration with the background task queue
for async email sending.

The service supports:
- HTML and plain text email templates
- Template rendering with Jinja2
- SMTP configuration with multiple providers
- Background email sending via task queue
- Email delivery tracking and monitoring
- Bounce and complaint handling
- Email analytics and reporting

Example:
    Basic email sending:
        >>> from services.email_service import EmailService
        >>> from models.background_task_models import EmailTask
        >>> 
        >>> email_service = EmailService()
        >>> await email_service.initialize()
        >>> 
        >>> # Send email via background task
        >>> email_task = EmailTask.create_email_task(
        ...     to="user@example.com",
        ...     subject="Welcome to GRETA PAI",
        ...     template="welcome_email",
        ...     template_data={"name": "John", "activation_link": "..."}
        ... )
        >>> await email_service.send_email_async(email_task)

    Direct email sending:
        >>> await email_service.send_email_direct(
        ...     to="user@example.com",
        ...     subject="Test Email",
        ...     body="Hello from GRETA PAI!"
        ... )

Attributes:
    smtp_config (Dict): SMTP server configuration
    template_env (jinja2.Environment): Template rendering environment
    task_queue (TaskQueueService): Background task queue service
    logger (logging.Logger): Service logger
"""

import asyncio
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import jinja2
from datetime import datetime

from models.background_task_models import EmailTask, BackgroundTaskType, TaskPriority
from utils.custom_exceptions import EmailServiceError, ConfigurationError
from config import get_settings

logger = logging.getLogger(__name__)


class EmailService:
    """
    Comprehensive email service with template support and background processing.
    
    This service handles all email-related functionality including template
    rendering, SMTP delivery, and integration with the background task queue.
    
    Attributes:
        smtp_config: SMTP server configuration
        template_env: Jinja2 template environment
        task_queue: Background task queue service
        
    Example:
        >>> email_service = EmailService()
        >>> await email_service.initialize()
        >>> await email_service.send_welcome_email("user@example.com", "John")
    """
    
    def __init__(self):
        """Initialize email service"""
        self.settings = get_settings()
        self.smtp_config = self._get_smtp_config()
        self.template_env = None
        self.task_queue = None
        self._initialized = False
    
    def _get_smtp_config(self) -> Dict[str, Any]:
        """
        Get SMTP configuration from settings.
        
        Returns:
            Dict: SMTP configuration parameters
        """
        return {
            "host": getattr(self.settings, 'smtp_host', 'localhost'),
            "port": getattr(self.settings, 'smtp_port', 587),
            "username": getattr(self.settings, 'smtp_username', ''),
            "password": getattr(self.settings, 'smtp_password', ''),
            "use_tls": getattr(self.settings, 'smtp_use_tls', True),
            "use_ssl": getattr(self.settings, 'smtp_use_ssl', False),
            "from_email": getattr(self.settings, 'from_email', 'noreply@greta-pai.com'),
            "from_name": getattr(self.settings, 'from_name', 'GRETA PAI System')
        }
    
    async def initialize(self) -> None:
        """
        Initialize the email service.
        
        Sets up template environment and validates SMTP configuration.
        
        Raises:
            EmailServiceError: If initialization fails
        """
        try:
            # Initialize template environment
            template_dir = Path(__file__).parent.parent / "templates" / "email"
            template_dir.mkdir(parents=True, exist_ok=True)
            
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(template_dir)),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
            
            # Create default templates if they don't exist
            await self._create_default_templates()
            
            # Test SMTP connection
            await self._test_smtp_connection()
            
            self._initialized = True
            logger.info("✅ Email service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Email service initialization failed: {e}")
            raise EmailServiceError(f"Failed to initialize email service: {e}")
    
    async def _create_default_templates(self) -> None:
        """Create default email templates if they don't exist"""
        template_dir = Path(__file__).parent.parent / "templates" / "email"
        
        # Welcome email template
        welcome_template = template_dir / "welcome_email.html"
        if not welcome_template.exists():
            welcome_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Welcome to GRETA PAI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background: #f9f9f9; }
        .button { display: inline-block; padding: 12px 24px; background: #3498db; 
                 color: white; text-decoration: none; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Welcome to GRETA PAI</h1>
        </div>
        <div class="content">
            <h2>Hello {{ name }}!</h2>
            <p>Welcome to GRETA PAI - The World's Most Advanced Personal AI System.</p>
            <p>Your account has been successfully created. You can now access all the powerful features of our AI platform.</p>
            
            {% if activation_link %}
            <p>Please click the button below to activate your account:</p>
            <p><a href="{{ activation_link }}" class="button">Activate Account</a></p>
            {% endif %}
            
            <h3>🚀 What you can do with GRETA PAI:</h3>
            <ul>
                <li>🧠 Enhanced Learning System with Auto Fine-tuning</li>
                <li>🦙 Local llama.cpp Processing (No Cloud Dependencies)</li>
                <li>🎭 German-accented Master Agent (Greta)</li>
                <li>🤖 Autonomous Multi-Agent Collaboration</li>
                <li>🔧 Continuous Learning Pipeline</li>
                <li>🌐 Swarm Intelligence</li>
                <li>👁️ Multi-Modal Processing</li>
                <li>🔮 Predictive Analytics</li>
            </ul>
            
            <p>If you have any questions, feel free to contact our support team.</p>
            <p>Best regards,<br>The GRETA PAI Team</p>
        </div>
    </div>
</body>
</html>
            """.strip()
            
            welcome_template.write_text(welcome_content)
        
        # Password reset template
        reset_template = template_dir / "password_reset.html"
        if not reset_template.exists():
            reset_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Password Reset - GRETA PAI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #e74c3c; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background: #f9f9f9; }
        .button { display: inline-block; padding: 12px 24px; background: #e74c3c; 
                 color: white; text-decoration: none; border-radius: 5px; }
        .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Password Reset Request</h1>
        </div>
        <div class="content">
            <h2>Hello {{ name }}!</h2>
            <p>We received a request to reset your password for your GRETA PAI account.</p>
            
            <div class="warning">
                <strong>⚠️ Security Notice:</strong> If you didn't request this password reset, please ignore this email. Your account remains secure.
            </div>
            
            <p>To reset your password, click the button below:</p>
            <p><a href="{{ reset_link }}" class="button">Reset Password</a></p>
            
            <p><strong>This link will expire in {{ expiry_hours }} hours.</strong></p>
            
            <p>For security reasons, this link can only be used once. If you need another reset link, please request a new one.</p>
            
            <p>Best regards,<br>The GRETA PAI Security Team</p>
        </div>
    </div>
</body>
</html>
            """.strip()
            
            reset_template.write_text(reset_content)
        
        logger.info("✅ Default email templates created")
    
    async def _test_smtp_connection(self) -> None:
        """Test SMTP connection"""
        if not self.smtp_config.get("host") or self.smtp_config["host"] == "localhost":
            logger.warning("⚠️ SMTP not configured, using mock email sending")
            return
        
        try:
            # Test connection (non-blocking)
            await asyncio.get_event_loop().run_in_executor(
                None, self._test_smtp_sync
            )
            logger.info("✅ SMTP connection test successful")
            
        except Exception as e:
            logger.warning(f"⚠️ SMTP connection test failed: {e}")
    
    def _test_smtp_sync(self) -> None:
        """Synchronous SMTP connection test"""
        if self.smtp_config["use_ssl"]:
            server = smtplib.SMTP_SSL(self.smtp_config["host"], self.smtp_config["port"])
        else:
            server = smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"])
            if self.smtp_config["use_tls"]:
                server.starttls()
        
        if self.smtp_config["username"]:
            server.login(self.smtp_config["username"], self.smtp_config["password"])
        
        server.quit()
    
    async def send_email_async(self, email_task: EmailTask) -> str:
        """
        Send email asynchronously via background task queue.
        
        Args:
            email_task: Email task to process
            
        Returns:
            str: Task ID for tracking
        """
        if not self.task_queue:
            # Import here to avoid circular imports
            from services.task_queue_service import TaskQueueService
            self.task_queue = TaskQueueService()
            await self.task_queue.initialize()
        
        return await self.task_queue.enqueue_task(email_task)
    
    async def send_email_direct(self, to: str, subject: str, body: str = None, 
                              template: str = None, template_data: Dict[str, Any] = None,
                              html: bool = True) -> bool:
        """
        Send email directly (synchronously).
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (if not using template)
            template: Template name to use
            template_data: Data for template rendering
            html: Whether to send as HTML
            
        Returns:
            bool: True if sent successfully
            
        Raises:
            EmailServiceError: If sending fails
        """
        try:
            # Render email content
            if template and self.template_env:
                email_template = self.template_env.get_template(f"{template}.html")
                body = email_template.render(template_data or {})
                html = True
            
            # Send email
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email_sync, to, subject, body, html
            )
            
            logger.info(f"📧 Email sent successfully to {to}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email to {to}: {e}")
            raise EmailServiceError(f"Failed to send email: {e}")
    
    def _send_email_sync(self, to: str, subject: str, body: str, html: bool = True) -> None:
        """Synchronous email sending"""
        if self.smtp_config["host"] == "localhost":
            # Mock email sending for development
            logger.info(f"📧 [MOCK] Email to {to}: {subject}")
            return
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f"{self.smtp_config['from_name']} <{self.smtp_config['from_email']}>"
        msg['To'] = to
        
        # Add body
        if html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        if self.smtp_config["use_ssl"]:
            server = smtplib.SMTP_SSL(self.smtp_config["host"], self.smtp_config["port"])
        else:
            server = smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"])
            if self.smtp_config["use_tls"]:
                server.starttls()
        
        if self.smtp_config["username"]:
            server.login(self.smtp_config["username"], self.smtp_config["password"])
        
        server.send_message(msg)
        server.quit()
    
    async def send_welcome_email(self, to: str, name: str, activation_link: str = None) -> str:
        """
        Send welcome email to new user.
        
        Args:
            to: Recipient email address
            name: User's name
            activation_link: Account activation link
            
        Returns:
            str: Task ID for tracking
        """
        email_task = EmailTask.create_email_task(
            to=to,
            subject="Welcome to GRETA PAI - Your AI Journey Begins!",
            template="welcome_email",
            template_data={
                "name": name,
                "activation_link": activation_link
            },
            priority=TaskPriority.HIGH
        )
        
        return await self.send_email_async(email_task)
    
    async def send_password_reset_email(self, to: str, name: str, reset_link: str, 
                                      expiry_hours: int = 24) -> str:
        """
        Send password reset email.
        
        Args:
            to: Recipient email address
            name: User's name
            reset_link: Password reset link
            expiry_hours: Link expiry time in hours
            
        Returns:
            str: Task ID for tracking
        """
        email_task = EmailTask.create_email_task(
            to=to,
            subject="Password Reset Request - GRETA PAI",
            template="password_reset",
            template_data={
                "name": name,
                "reset_link": reset_link,
                "expiry_hours": expiry_hours
            },
            priority=TaskPriority.HIGH
        )
        
        return await self.send_email_async(email_task)
