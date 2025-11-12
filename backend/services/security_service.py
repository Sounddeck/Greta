"""
Security Service for GRETA PAI

This service provides comprehensive security features including
CORS configuration, security headers, request signing, IP filtering,
and enhanced password policies.

The service provides:
- CORS (Cross-Origin Resource Sharing) configuration
- Security headers (HSTS, CSP, X-Frame-Options, etc.)
- Request signing for sensitive operations
- IP whitelisting/blacklisting
- Enhanced password policies and validation
- Security event logging and monitoring
- Rate limiting integration
- Security audit logging

Example:
    Basic security configuration:
        >>> from services.security_service import SecurityService
        >>> 
        >>> security_service = SecurityService()
        >>> await security_service.initialize()
        >>> 
        >>> # Validate password
        >>> is_valid, errors = security_service.validate_password("MyP@ssw0rd123")
        >>> 
        >>> # Check IP whitelist
        >>> is_allowed = await security_service.is_ip_allowed("192.168.1.1")

Attributes:
    db (AsyncIOMotorDatabase): MongoDB database instance
    password_policy (PasswordPolicy): Password policy configuration
    ip_whitelist (Set[str]): Set of whitelisted IP addresses
    ip_blacklist (Set[str]): Set of blacklisted IP addresses
    logger (logging.Logger): Service logger
"""

import re
import hashlib
import hmac
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from ipaddress import ip_address, ip_network, IPv4Address, IPv6Address
from motor.motor_asyncio import AsyncIOMotorDatabase

from utils.custom_exceptions import ValidationException, InvalidInputError
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PasswordPolicy:
    """
    Password policy configuration.
    
    Defines password requirements and validation rules.
    
    Attributes:
        min_length: Minimum password length
        max_length: Maximum password length
        require_uppercase: Require at least one uppercase letter
        require_lowercase: Require at least one lowercase letter
        require_digits: Require at least one digit
        require_special: Require at least one special character
        special_characters: Allowed special characters
        min_unique_characters: Minimum number of unique characters
        prevent_common_passwords: Prevent common/weak passwords
        prevent_username_in_password: Prevent username in password
        password_history_count: Number of previous passwords to check
    """
    min_length: int = 12
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special: bool = True
    special_characters: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    min_unique_characters: int = 8
    prevent_common_passwords: bool = True
    prevent_username_in_password: bool = True
    password_history_count: int = 5


@dataclass
class SecurityHeaders:
    """
    Security headers configuration.
    
    Defines HTTP security headers to be added to responses.
    
    Attributes:
        hsts: HTTP Strict Transport Security header
        csp: Content Security Policy header
        x_frame_options: X-Frame-Options header
        x_content_type_options: X-Content-Type-Options header
        x_xss_protection: X-XSS-Protection header
        referrer_policy: Referrer-Policy header
        permissions_policy: Permissions-Policy header
    """
    hsts: str = "max-age=31536000; includeSubDomains"
    csp: str = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"
    permissions_policy: str = "geolocation=(), microphone=(), camera=()"


@dataclass
class CORSConfig:
    """
    CORS configuration.
    
    Defines Cross-Origin Resource Sharing settings.
    
    Attributes:
        allow_origins: List of allowed origins
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers
        allow_credentials: Allow credentials
        max_age: Preflight cache duration in seconds
    """
    allow_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5174"])
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    max_age: int = 600


class SecurityService:
    """
    Comprehensive security service.
    
    This service provides security features including password validation,
    IP filtering, request signing, and security headers.
    
    Attributes:
        db: MongoDB database instance
        password_policy: Password policy configuration
        security_headers: Security headers configuration
        cors_config: CORS configuration
        
    Example:
        >>> security_service = SecurityService()
        >>> await security_service.initialize()
        >>> is_valid, errors = security_service.validate_password("MyP@ssw0rd123")
    """
    
    def __init__(self, database: AsyncIOMotorDatabase = None):
        """
        Initialize security service.
        
        Args:
            database: MongoDB database instance
        """
        self.db = database
        self.settings = get_settings()
        
        # Security configurations
        self.password_policy = PasswordPolicy()
        self.security_headers = SecurityHeaders()
        self.cors_config = CORSConfig()
        
        # IP filtering
        self.ip_whitelist: Set[str] = set()
        self.ip_blacklist: Set[str] = set()
        self.ip_whitelist_networks: List[Any] = []
        self.ip_blacklist_networks: List[Any] = []
        
        # Common weak passwords
        self.common_passwords = {
            "password", "123456", "12345678", "qwerty", "abc123",
            "monkey", "1234567", "letmein", "trustno1", "dragon",
            "baseball", "111111", "iloveyou", "master", "sunshine",
            "ashley", "bailey", "passw0rd", "shadow", "123123",
            "654321", "superman", "qazwsx", "michael", "football"
        }
        
        # Request signing
        self.signing_secret = None
    
    async def initialize(self) -> None:
        """
        Initialize the security service.
        
        Sets up security configurations, loads IP lists, and initializes
        request signing.
        
        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            # Initialize signing secret
            self.signing_secret = getattr(self.settings, 'signing_secret', secrets.token_hex(32))
            
            # Load IP lists from database
            if self.db:
                await self._load_ip_lists()
            
            # Add default IP whitelist (localhost)
            self.ip_whitelist.add("127.0.0.1")
            self.ip_whitelist.add("::1")
            
            logger.info("✅ Security service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Security service initialization failed: {e}")
            raise
    
    async def _load_ip_lists(self) -> None:
        """Load IP whitelist and blacklist from database"""
        try:
            # Load whitelist
            whitelist_doc = await self.db.security_config.find_one({"type": "ip_whitelist"})
            if whitelist_doc:
                self.ip_whitelist = set(whitelist_doc.get("ips", []))
                logger.info(f"✅ Loaded {len(self.ip_whitelist)} whitelisted IPs")
            
            # Load blacklist
            blacklist_doc = await self.db.security_config.find_one({"type": "ip_blacklist"})
            if blacklist_doc:
                self.ip_blacklist = set(blacklist_doc.get("ips", []))
                logger.info(f"✅ Loaded {len(self.ip_blacklist)} blacklisted IPs")
                
        except Exception as e:
            logger.warning(f"Failed to load IP lists: {e}")
    
    def validate_password(self, password: str, username: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate password against security policy.
        
        Args:
            password: Password to validate
            username: Username (optional, for username-in-password check)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
            
        Example:
            >>> is_valid, errors = security_service.validate_password("MyP@ssw0rd123")
            >>> if not is_valid:
            ...     print(f"Password errors: {errors}")
        """
        errors = []
        
        # Length check
        if len(password) < self.password_policy.min_length:
            errors.append(f"Password must be at least {self.password_policy.min_length} characters long")
        
        if len(password) > self.password_policy.max_length:
            errors.append(f"Password must not exceed {self.password_policy.max_length} characters")
        
        # Character requirements
        if self.password_policy.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.password_policy.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.password_policy.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if self.password_policy.require_special:
            special_chars_pattern = f"[{re.escape(self.password_policy.special_characters)}]"
            if not re.search(special_chars_pattern, password):
                errors.append(f"Password must contain at least one special character ({self.password_policy.special_characters})")
        
        # Unique characters check
        unique_chars = len(set(password))
        if unique_chars < self.password_policy.min_unique_characters:
            errors.append(f"Password must contain at least {self.password_policy.min_unique_characters} unique characters")
        
        # Common passwords check
        if self.password_policy.prevent_common_passwords:
            if password.lower() in self.common_passwords:
                errors.append("Password is too common. Please choose a more unique password")
        
        # Username in password check
        if self.password_policy.prevent_username_in_password and username:
            if username.lower() in password.lower():
                errors.append("Password must not contain your username")
        
        return (len(errors) == 0, errors)
    
    async def is_ip_allowed(self, ip: str) -> bool:
        """
        Check if IP address is allowed.
        
        Args:
            ip: IP address to check
            
        Returns:
            bool: True if IP is allowed, False otherwise
        """
        try:
            ip_obj = ip_address(ip)
            
            # Check blacklist first
            if ip in self.ip_blacklist:
                logger.warning(f"⚠️ Blocked blacklisted IP: {ip}")
                return False
            
            # Check blacklist networks
            for network in self.ip_blacklist_networks:
                if ip_obj in network:
                    logger.warning(f"⚠️ Blocked IP in blacklisted network: {ip}")
                    return False
            
            # If whitelist is empty, allow all (except blacklisted)
            if not self.ip_whitelist and not self.ip_whitelist_networks:
                return True
            
            # Check whitelist
            if ip in self.ip_whitelist:
                return True
            
            # Check whitelist networks
            for network in self.ip_whitelist_networks:
                if ip_obj in network:
                    return True
            
            logger.warning(f"⚠️ Blocked non-whitelisted IP: {ip}")
            return False
            
        except Exception as e:
            logger.error(f"IP validation error for {ip}: {e}")
            return False
    
    async def add_to_whitelist(self, ip: str) -> None:
        """
        Add IP address to whitelist.
        
        Args:
            ip: IP address or network to whitelist
        """
        try:
            # Try to parse as network first
            try:
                network = ip_network(ip, strict=False)
                self.ip_whitelist_networks.append(network)
                logger.info(f"✅ Added network to whitelist: {ip}")
            except:
                # Parse as single IP
                ip_address(ip)  # Validate
                self.ip_whitelist.add(ip)
                logger.info(f"✅ Added IP to whitelist: {ip}")
            
            # Persist to database
            if self.db:
                await self.db.security_config.update_one(
                    {"type": "ip_whitelist"},
                    {"$addToSet": {"ips": ip}},
                    upsert=True
                )
                
        except Exception as e:
            logger.error(f"Failed to add IP to whitelist: {e}")
            raise InvalidInputError("ip", ip, str(e))
    
    async def add_to_blacklist(self, ip: str) -> None:
        """
        Add IP address to blacklist.
        
        Args:
            ip: IP address or network to blacklist
        """
        try:
            # Try to parse as network first
            try:
                network = ip_network(ip, strict=False)
                self.ip_blacklist_networks.append(network)
                logger.info(f"✅ Added network to blacklist: {ip}")
            except:
                # Parse as single IP
                ip_address(ip)  # Validate
                self.ip_blacklist.add(ip)
                logger.info(f"✅ Added IP to blacklist: {ip}")
            
            # Persist to database
            if self.db:
                await self.db.security_config.update_one(
                    {"type": "ip_blacklist"},
                    {"$addToSet": {"ips": ip}},
                    upsert=True
                )
                
        except Exception as e:
            logger.error(f"Failed to add IP to blacklist: {e}")
            raise InvalidInputError("ip", ip, str(e))
    
    def sign_request(self, data: Dict[str, Any]) -> str:
        """
        Sign request data for verification.
        
        Args:
            data: Request data to sign
            
        Returns:
            str: Request signature
            
        Example:
            >>> signature = security_service.sign_request({"user_id": "123", "action": "delete"})
            >>> # Include signature in request headers
        """
        import json
        
        # Serialize data
        data_str = json.dumps(data, sort_keys=True, default=str)
        
        # Create HMAC signature
        signature = hmac.new(
            self.signing_secret.encode(),
            data_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_request_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """
        Verify request signature.
        
        Args:
            data: Request data
            signature: Signature to verify
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        expected_signature = self.sign_request(data)
        return hmac.compare_digest(signature, expected_signature)
    
    def get_security_headers(self) -> Dict[str, str]:
        """
        Get security headers to add to responses.
        
        Returns:
            Dict[str, str]: Security headers
        """
        return {
            "Strict-Transport-Security": self.security_headers.hsts,
            "Content-Security-Policy": self.security_headers.csp,
            "X-Frame-Options": self.security_headers.x_frame_options,
            "X-Content-Type-Options": self.security_headers.x_content_type_options,
            "X-XSS-Protection": self.security_headers.x_xss_protection,
            "Referrer-Policy": self.security_headers.referrer_policy,
            "Permissions-Policy": self.security_headers.permissions_policy
        }
    
    async def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log security event for audit trail.
        
        Args:
            event_type: Type of security event
            details: Event details
        """
        if not self.db:
            return
        
        try:
            await self.db.security_events.insert_one({
                "event_type": event_type,
                "details": details,
                "timestamp": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
