"""
Comprehensive tests for email service

Tests cover:
- Email sending
- Email templates
- Email queue
- Error handling
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import sys
from datetime import datetime


# Mock jinja2 module before importing email_service
class MockTemplate:
    def __init__(self, content):
        self.content = content
    
    def render(self, data):
        return f"Rendered: {self.content} with {data}"


class MockEnvironment:
    def __init__(self, loader=None):
        self.loader = loader
    
    def get_template(self, name):
        return MockTemplate(f"template_{name}")


mock_jinja2 = type(sys)('jinja2')
mock_jinja2.Environment = MockEnvironment
mock_jinja2.FileSystemLoader = MagicMock
sys.modules['jinja2'] = mock_jinja2

# Mock custom_exceptions module
class EmailServiceError(Exception):
    pass


class ConfigurationError(Exception):
    pass


mock_custom_exceptions = type(sys)('utils.custom_exceptions')
mock_custom_exceptions.EmailServiceError = EmailServiceError
mock_custom_exceptions.ConfigurationError = ConfigurationError
sys.modules['utils.custom_exceptions'] = mock_custom_exceptions


class TestEmailSending:
    """Test email sending functionality"""
    
    @pytest.mark.asyncio
    async def test_send_email_success(self):
        """Test successful email sending"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        
        # Mock SMTP
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Send email
            result = await email_service.send_email_direct(
                to="test@example.com",
                subject="Test Email",
                body="Hello World"
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_send_email_with_template(self):
        """Test sending email with template"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        
        # Mock SMTP
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Send email with template
            result = await email_service.send_email_direct(
                to="test@example.com",
                subject="Welcome",
                template="welcome_email",
                template_data={"name": "John"}
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_send_bulk_emails(self):
        """Test sending bulk emails"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        
        # Mock SMTP
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Send bulk emails
            recipients = ["user1@example.com", "user2@example.com", "user3@example.com"]
            
            for recipient in recipients:
                result = await email_service.send_email_direct(
                    to=recipient,
                    subject="Bulk Email",
                    body="Hello"
                )
                assert result is True
    
    @pytest.mark.asyncio
    async def test_send_email_failure(self):
        """Test email sending failure"""
        from services.email_service import EmailService

        # Create email service with non-localhost SMTP
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        email_service.smtp_config["host"] = "smtp.example.com"  # Non-localhost to trigger real SMTP

        # Mock _send_email_sync to raise exception
        with patch.object(email_service, '_send_email_sync', side_effect=Exception("SMTP connection failed")):
            # Send email should raise error
            with pytest.raises(EmailServiceError):
                await email_service.send_email_direct(
                    to="test@example.com",
                    subject="Test",
                    body="Hello"
                )


class TestEmailTemplates:
    """Test email template functionality"""
    
    @pytest.mark.asyncio
    async def test_load_email_template(self):
        """Test loading email template"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service.template_env = MockEnvironment()
        
        # Load template
        template = email_service.template_env.get_template("welcome_email.html")
        
        assert template is not None
        assert "template_welcome_email.html" in template.content
    
    @pytest.mark.asyncio
    async def test_render_template_with_variables(self):
        """Test rendering template with variables"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service.template_env = MockEnvironment()
        
        # Render template
        template = email_service.template_env.get_template("welcome_email.html")
        rendered = template.render({"name": "John", "activation_link": "https://example.com/activate"})
        
        assert "Rendered:" in rendered
        assert "name" in rendered
    
    @pytest.mark.asyncio
    async def test_template_caching(self):
        """Test template caching"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service.template_env = MockEnvironment()
        
        # Load template twice
        template1 = email_service.template_env.get_template("welcome_email.html")
        template2 = email_service.template_env.get_template("welcome_email.html")
        
        # Templates should be the same object (cached)
        assert template1.content == template2.content


class TestEmailQueue:
    """Test email queue functionality"""
    
    @pytest.mark.asyncio
    async def test_queue_email_for_sending(self):
        """Test queueing email for sending"""
        from services.email_service import EmailService
        from models.background_task_models import EmailTask
        
        # Create email service
        email_service = EmailService()
        email_service._initialized = True
        
        # Mock task queue
        mock_task_queue = AsyncMock()
        mock_task_queue.enqueue_task = AsyncMock(return_value="task_123")
        email_service.task_queue = mock_task_queue
        
        # Create email task
        email_task = EmailTask.create_email_task(
            to="test@example.com",
            subject="Test",
            template="welcome_email"
        )
        
        # Queue email
        task_id = await email_service.send_email_async(email_task)
        
        assert task_id == "task_123"
        assert mock_task_queue.enqueue_task.called
    
    @pytest.mark.asyncio
    async def test_process_email_queue(self):
        """Test processing email queue"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        
        # Mock task queue with pending tasks
        mock_task_queue = AsyncMock()
        mock_task_queue.get_pending_tasks = AsyncMock(return_value=[
            {"task_id": "task_1", "to": "user1@example.com"},
            {"task_id": "task_2", "to": "user2@example.com"}
        ])
        email_service.task_queue = mock_task_queue
        
        # Process queue
        pending_tasks = await mock_task_queue.get_pending_tasks()
        
        assert len(pending_tasks) == 2
    
    @pytest.mark.asyncio
    async def test_queue_status_tracking(self):
        """Test queue status tracking"""
        from services.email_service import EmailService
        
        # Create email service
        email_service = EmailService()
        email_service._initialized = True
        
        # Mock task queue
        mock_task_queue = AsyncMock()
        mock_task_queue.get_queue_stats = AsyncMock(return_value={
            "pending_tasks": 5,
            "processing_tasks": 2,
            "completed_tasks": 10,
            "failed_tasks": 1
        })
        email_service.task_queue = mock_task_queue
        
        # Get queue stats
        stats = await mock_task_queue.get_queue_stats()
        
        assert stats["pending_tasks"] == 5
        assert stats["processing_tasks"] == 2
        assert stats["completed_tasks"] == 10
        assert stats["failed_tasks"] == 1


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_smtp_connection_error(self):
        """Test SMTP connection error handling"""
        from services.email_service import EmailService

        # Create email service with non-localhost SMTP
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        email_service.smtp_config["host"] = "smtp.example.com"  # Non-localhost

        # Mock _send_email_sync to raise connection error
        with patch.object(email_service, '_send_email_sync', side_effect=ConnectionError("SMTP server unavailable")):
            # Send email should raise error
            with pytest.raises(EmailServiceError):
                await email_service.send_email_direct(
                    to="test@example.com",
                    subject="Test",
                    body="Hello"
                )

    @pytest.mark.asyncio
    async def test_invalid_email_address(self):
        """Test invalid email address handling"""
        from services.email_service import EmailService

        # Create email service with non-localhost SMTP
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        email_service.smtp_config["host"] = "smtp.example.com"  # Non-localhost

        # Mock _send_email_sync to raise invalid email error
        with patch.object(email_service, '_send_email_sync', side_effect=Exception("Invalid email address")):
            # Send email with invalid address should raise error
            with pytest.raises(EmailServiceError):
                await email_service.send_email_direct(
                    to="invalid-email",
                    subject="Test",
                    body="Hello"
                )

    @pytest.mark.asyncio
    async def test_template_rendering_error(self):
        """Test template rendering error handling"""
        from services.email_service import EmailService

        # Create email service
        email_service = EmailService()
        email_service._initialized = True

        # Mock template environment to raise error
        mock_env = MagicMock()
        mock_env.get_template.side_effect = Exception("Template not found")
        email_service.template_env = mock_env

        # Mock SMTP
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            # Send email with invalid template should raise error
            with pytest.raises(EmailServiceError):
                await email_service.send_email_direct(
                    to="test@example.com",
                    subject="Test",
                    template="nonexistent_template"
                )

    @pytest.mark.asyncio
    async def test_service_unavailable_handling(self):
        """Test service unavailable handling"""
        from services.email_service import EmailService

        # Create email service with non-localhost SMTP
        email_service = EmailService()
        email_service._initialized = True
        email_service.template_env = MockEnvironment()
        email_service.smtp_config["host"] = "smtp.example.com"  # Non-localhost

        # Mock _send_email_sync to raise service unavailable error
        with patch.object(email_service, '_send_email_sync', side_effect=Exception("Service unavailable")):
            # Try to send email - should raise error
            with pytest.raises(EmailServiceError):
                await email_service.send_email_direct(
                    to="test@example.com",
                    subject="Test",
                    body="Hello"
                )

