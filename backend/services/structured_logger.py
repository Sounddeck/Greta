"""
Structured Logging Service for GRETA PAI
Provides JSON-formatted logging with correlation ID tracking and context enrichment
"""

import sys
import json
import contextvars
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

# Context variables for request tracking
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)
user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'user_id', default=None
)
request_path_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_path', default=None
)


class StructuredLogger:
    """
    Structured logging service with JSON formatting and correlation ID tracking
    
    Features:
    - JSON-formatted logs for machine parsing
    - Correlation ID tracking across requests
    - Context enrichment (user_id, request_path, etc.)
    - File rotation (daily, 30-day retention)
    - Environment-based configuration
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_format: str = "json",
        log_file_path: str = "logs/cpas5.log",
        log_rotation: str = "1 day",
        log_retention: str = "30 days",
        log_compression: str = "zip",
        enable_console: bool = True,
        enable_file: bool = True,
        environment: str = "development"
    ):
        """
        Initialize structured logger
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Format type (json, text)
            log_file_path: Path to log file
            log_rotation: Rotation interval (e.g., "1 day", "500 MB")
            log_retention: Retention period (e.g., "30 days")
            log_compression: Compression format (zip, gz, tar.gz)
            enable_console: Enable console logging
            enable_file: Enable file logging
            environment: Environment name (development, production)
        """
        self.log_level = log_level
        self.log_format = log_format
        self.log_file_path = log_file_path
        self.log_rotation = log_rotation
        self.log_retention = log_retention
        self.log_compression = log_compression
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.environment = environment
        
        # Remove default logger
        logger.remove()
        
        # Configure logger
        self._configure_logger()
    
    def _patcher(self, record):
        """Patch function to add context variables to log records"""
        correlation_id = correlation_id_var.get()
        user_id = user_id_var.get()
        request_path = request_path_var.get()

        if correlation_id:
            record["extra"]["correlation_id"] = correlation_id
        if user_id:
            record["extra"]["user_id"] = user_id
        if request_path:
            record["extra"]["request_path"] = request_path

        record["extra"]["environment"] = self.environment

        # Filter function must return True to allow the log through
        return True

    def _configure_logger(self):
        """Configure loguru with structured logging"""

        # Console logging (development: text, production: JSON)
        if self.enable_console:
            if self.environment == "development" and self.log_format != "json":
                # Human-readable format for development
                logger.add(
                    sys.stdout,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
                    level=self.log_level,
                    colorize=True,
                    enqueue=True,  # Thread-safe
                    filter=self._patcher,  # Apply patcher
                )
            else:
                # JSON format for production or when explicitly requested
                logger.add(
                    sys.stdout,
                    format="{message}",
                    level=self.log_level,
                    colorize=False,
                    enqueue=True,
                    serialize=True,  # Built-in JSON serialization
                    filter=self._patcher,  # Apply patcher
                )

        # File logging (always JSON for machine parsing)
        if self.enable_file:
            # Create log directory if it doesn't exist
            log_path = Path(self.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                self.log_file_path,
                format="{message}",
                level=self.log_level,
                rotation=self.log_rotation,
                retention=self.log_retention,
                compression=self.log_compression,
                encoding="utf-8",
                enqueue=True,
                serialize=True,  # Built-in JSON serialization
                filter=self._patcher,  # Apply patcher
            )
        
        # Log initialization
        logger.info(
            "Structured logging initialized",
            extra={
                "log_level": self.log_level,
                "log_format": self.log_format,
                "log_file_path": self.log_file_path,
                "environment": self.environment,
            }
        )
    
    @staticmethod
    def set_correlation_id(correlation_id: str):
        """Set correlation ID for current context"""
        correlation_id_var.set(correlation_id)

    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get correlation ID from current context"""
        return correlation_id_var.get()

    @staticmethod
    def set_user_id(user_id: str):
        """Set user ID for current context"""
        user_id_var.set(user_id)

    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get user ID from current context"""
        return user_id_var.get()

    @staticmethod
    def set_request_path(request_path: str):
        """Set request path for current context"""
        request_path_var.set(request_path)

    @staticmethod
    def get_request_path() -> Optional[str]:
        """Get request path from current context"""
        return request_path_var.get()

    @staticmethod
    def clear_context():
        """Clear all context variables"""
        correlation_id_var.set(None)
        user_id_var.set(None)
        request_path_var.set(None)


# Global structured logger instance (will be initialized from config)
_structured_logger: Optional[StructuredLogger] = None


def get_structured_logger() -> StructuredLogger:
    """Get the global structured logger instance"""
    global _structured_logger
    if _structured_logger is None:
        raise RuntimeError(
            "Structured logger not initialized. Call setup_structured_logging() first."
        )
    return _structured_logger


def setup_structured_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file_path: str = "logs/cpas5.log",
    log_rotation: str = "1 day",
    log_retention: str = "30 days",
    log_compression: str = "zip",
    enable_console: bool = True,
    enable_file: bool = True,
    environment: str = "development"
) -> StructuredLogger:
    """
    Setup structured logging with configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type (json, text)
        log_file_path: Path to log file
        log_rotation: Rotation interval (e.g., "1 day", "500 MB")
        log_retention: Retention period (e.g., "30 days")
        log_compression: Compression format (zip, gz, tar.gz)
        enable_console: Enable console logging
        enable_file: Enable file logging
        environment: Environment name (development, production)
        
    Returns:
        Configured StructuredLogger instance
    """
    global _structured_logger
    _structured_logger = StructuredLogger(
        log_level=log_level,
        log_format=log_format,
        log_file_path=log_file_path,
        log_rotation=log_rotation,
        log_retention=log_retention,
        log_compression=log_compression,
        enable_console=enable_console,
        enable_file=enable_file,
        environment=environment
    )
    return _structured_logger

